#!/usr/bin/env python3
"""Milestone B: quantize the DeltaNet in_proj BF16 weights to Q8 for ALL 48
DeltaNet layers, driven by the safetensors index.

For every layer where layer % 4 != 3 (DeltaNet layers; the every-4th attention
layers have no linear_attn.in_proj_*), this resolves each of the four in_proj_*
tensors to its shard via the index (each looked up independently -- a layer's
tensors need not share a shard), reads only that tensor's byte range (mmap),
quantizes per group of 64, and writes:
    <OUT>/layer-NN/delta_input_q8_{codes.i8,scale.f16,bias.f16}

Q8 convention: see qwen38_q8_delta_quantize.py. Degenerate groups (fp16 scale
below 2^-14) are zeroed to their mean (code 0, scale 1.0) -- fp16 cannot
represent such tiny scales, and the group is below fp16 working precision.

A per-layer round-trip GATE runs inline by default (re-derives weights from the
emitted bytes the way the kernel does, compares to BF16, checks a bias-zeroed
negative control). This is the ONLY oracle in Milestone B (token-equality is
gone). --no-check skips it.

RMS_TOL is 2e-2: the honest ceiling for Q8-of-BF16. The hardest real sub-tensor
(a small in_proj_b with a few legitimately-zeroed tiny groups) sits at ~1.5%;
the negative control fires at ~30%, and structural bugs trip the separate
max_step<1.5 guard. So 2% cleanly separates real Q8 error from any defect.

Usage:
    python3 qwen38_q8_delta_quantize_all.py INDEX.json OUT_DIR [--no-check]
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

GROUP = 64
HIDDEN = 5120
GROUPS_PER_ROW = HIDDEN // GROUP  # 80
LAYER_COUNT = 64
FP16_MIN_NORMAL = np.float32(2.0) ** -14  # ~6.10e-5
RMS_TOL = 2.0e-2
STEP_TOL = 1.5

IN_PROJ = [
    ("linear_attn.in_proj_qkv.weight", 10240),
    ("linear_attn.in_proj_z.weight",    6144),
    ("linear_attn.in_proj_a.weight",      48),
    ("linear_attn.in_proj_b.weight",      48),
]
TOTAL_ROWS = sum(r for _, r in IN_PROJ)  # 16480
PREFIX = "language_model.model.layers.{L}."


class Shard:
    def __init__(self, path):
        self.path = Path(path)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.data_start = 8 + n
        self._mm = None

    def _map(self):
        if self._mm is None:
            import mmap
            self._fh = open(self.path, "rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, prot=mmap.PROT_READ)
        return self._mm

    def load_bf16(self, name, rows):
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
        if info["shape"] != [rows, HIDDEN]:
            raise ValueError(f"{name}: shape {info['shape']} != [{rows},{HIDDEN}]")
        lo, hi = info["data_offsets"]
        mm = self._map()
        raw = mm[self.data_start + lo: self.data_start + hi]
        bits = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
        return bits.view("<f4").reshape(rows, HIDDEN).copy()

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._fh.close()
            self._mm = None


def float32_to_fp16_bits(values):
    return np.asarray(values, dtype="<f4").astype("<f2").view("<u2")


def quantize_group_affine(block):
    """[rows,groups,64] f32 -> (codes int8, scale_f16 u16, bias_f16 u16,
    degenerate_count int).

    Per-group affine map of [min,max] onto [-127,127], with fp16-stored
    scale/bias. Two correctness points:
      * scale/bias are rounded to fp16 FIRST (that is what the kernel reads),
        then codes are computed against those fp16 values, so codes and
        metadata are mutually consistent in the precision that runs.
      * fp16 rounding of the scale can round it slightly DOWN, which would make
        (max-bias)/scale exceed 127 and clamp -> large error. So after
        rounding, the scale is bumped UP by fp16 ULPs until the whole group
        fits in [-127,127]. This preserves the group's signal instead of
        discarding it. Only EXACTLY-constant groups (max==min, true range 0)
        are degenerate: store scale=1.0, code=0, dequant=bias."""
    minimum = block.min(axis=-1)
    maximum = block.max(axis=-1)
    rng = maximum - minimum
    bias_f16 = float32_to_fp16_bits((maximum + minimum) / np.float32(2.0))
    bias_r = bias_f16.view("<f2").astype(np.float32)

    # Exactly-constant groups: true zero range -> degenerate (mean only).
    constant = rng == 0

    scale_f16 = float32_to_fp16_bits(rng / np.float32(254.0))
    scale_r = scale_f16.view("<f2").astype(np.float32)

    # Bump scale up (one fp16 ULP at a time) until the farthest element from
    # bias fits in +/-127 code steps. A few iterations suffice; constants are
    # excluded (their scale is set below).
    far = np.maximum(np.abs(maximum - bias_r), np.abs(minimum - bias_r))
    for _ in range(8):
        active = (~constant) & (scale_r > 0)
        over = active & (far > np.float32(127.0) * scale_r)
        if not over.any():
            break
        bumped = (scale_f16.view("<u2").astype(np.uint32) + 1).astype("<u2")
        scale_f16 = np.where(over, bumped.view("<u2"), scale_f16)
        scale_r = scale_f16.view("<f2").astype(np.float32)

    # Constant groups: store scale=1.0 (fp16-exact), codes will be 0.
    scale_f16 = np.where(constant,
                         float32_to_fp16_bits(np.float32(1.0)), scale_f16)
    scale_r = scale_f16.view("<f2").astype(np.float32)
    scale_use = np.where(scale_r == 0, np.float32(1.0), scale_r)

    codes = np.rint((block - bias_r[..., None]) / scale_use[..., None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    codes = np.where(constant[..., None], np.int8(0), codes)
    return codes, scale_f16, bias_f16, int(constant.sum())

def quantize_layer(tensors):
    codes = np.empty((TOTAL_ROWS, HIDDEN), dtype=np.int8)
    scale = np.empty((TOTAL_ROWS, GROUPS_PER_ROW), dtype="<u2")
    bias = np.empty((TOTAL_ROWS, GROUPS_PER_ROW), dtype="<u2")
    row0 = 0
    degen = 0
    for w, rows in tensors:
        block = w.reshape(rows, GROUPS_PER_ROW, GROUP)
        c, s, b, d = quantize_group_affine(block)
        codes[row0:row0 + rows] = c.reshape(rows, HIDDEN)
        scale[row0:row0 + rows] = s.reshape(rows, GROUPS_PER_ROW)
        bias[row0:row0 + rows] = b.reshape(rows, GROUPS_PER_ROW)
        degen += d
        row0 += rows
    assert row0 == TOTAL_ROWS
    return codes, scale, bias, degen


def dequant_like_kernel(codes, scale_u16, bias_u16, zero_bias=False):
    rows = codes.shape[0]
    s = scale_u16.view("<f2").astype(np.float32).reshape(rows, GROUPS_PER_ROW, 1)
    b = bias_u16.view("<f2").astype(np.float32).reshape(rows, GROUPS_PER_ROW, 1)
    if zero_bias:
        b = np.zeros_like(b)
    c = codes.reshape(rows, GROUPS_PER_ROW, GROUP).astype(np.float32)
    return (s * c + b).reshape(rows, HIDDEN)


def rel_rms(recon, ref):
    err = float(np.sqrt(np.mean((recon - ref) ** 2)))
    mag = float(np.sqrt(np.mean(ref ** 2)))
    return err / mag if mag > 0 else err


def check_layer(tensors, codes, scale, bias):
    ref = np.empty((TOTAL_ROWS, HIDDEN), dtype=np.float32)
    row0 = 0
    for w, rows in tensors:
        ref[row0:row0 + rows] = w
        row0 += rows
    recon = dequant_like_kernel(codes, scale, bias)
    worst_rms = 0.0
    worst_step = 0.0
    ok = True
    row0 = 0
    for _, rows in IN_PROJ:
        sl = slice(row0, row0 + rows)
        rr = rel_rms(recon[sl], ref[sl])
        sc = scale[sl].view("<f2").astype(np.float32).repeat(GROUP, axis=1)
        step = float((np.abs(recon[sl] - ref[sl]) /
                      np.where(sc == 0, 1.0, sc)).max())
        worst_rms = max(worst_rms, rr)
        worst_step = max(worst_step, step)
        if rr >= RMS_TOL or step >= STEP_TOL:
            ok = False
        row0 += rows
    control = dequant_like_kernel(codes, scale, bias, zero_bias=True)
    control_fires = rel_rms(control, ref) > RMS_TOL
    return ok and control_fires, worst_rms, worst_step, control_fires


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2:
        print("usage: python3 qwen38_q8_delta_quantize_all.py "
              "INDEX.json OUT_DIR [--no-check]", file=sys.stderr)
        return 2
    index_path, out_root = Path(args[0]), Path(args[1])
    do_check = "--no-check" not in flags
    shard_dir = index_path.parent
    weight_map = json.loads(index_path.read_text())["weight_map"]

    delta_layers = [L for L in range(LAYER_COUNT) if L % 4 != 3]

    plan = {}
    for L in delta_layers:
        entries = []
        for suffix, rows in IN_PROJ:
            name = PREFIX.format(L=L) + suffix
            if name not in weight_map:
                print(f"FATAL: {name} not in index", file=sys.stderr)
                return 3
            entries.append((name, weight_map[name], rows))
        plan[L] = entries

    open_shards = {}

    def get_shard(name):
        if name not in open_shards:
            open_shards[name] = Shard(shard_dir / name)
        return open_shards[name]

    out_root.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    failures = []
    for L in delta_layers:
        tensors = [(get_shard(shard).load_bf16(name, rows), rows)
                   for name, shard, rows in plan[L]]
        codes, scale, bias, degen = quantize_layer(tensors)

        layer_dir = out_root / f"layer-{L:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "delta_input_q8_codes.i8").write_bytes(codes.tobytes())
        (layer_dir / "delta_input_q8_scale.f16").write_bytes(scale.tobytes())
        (layer_dir / "delta_input_q8_bias.f16").write_bytes(bias.tobytes())

        if do_check:
            ok, wr, ws, ctl = check_layer(tensors, codes, scale, bias)
            status = "ok" if ok else "FAIL"
            print(f"layer {L:02d}  rel_rms {wr:.3e}  max_step {ws:.2f}  "
                  f"degen {degen:4d}  ctrl {'fires' if ctl else 'DEAD'}  "
                  f"{status}")
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append(L)
        else:
            print(f"layer {L:02d}  wrote {layer_dir}  (degen {degen})")

        remaining = set()
        for later in delta_layers:
            if later > L:
                for _, shard, _ in plan[later]:
                    remaining.add(shard)
        for name in list(open_shards.keys()):
            if name not in remaining:
                open_shards[name].close()
                del open_shards[name]

    for s in open_shards.values():
        s.close()

    if do_check:
        print(f"\n{n_ok}/{len(delta_layers)} layers PASS, {n_fail} FAIL")
        if failures:
            print("FAILED layers:", failures)
            return 1
        print("ALL LAYERS PASS")
    else:
        print(f"\nquantized {len(delta_layers)} layers (no check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())