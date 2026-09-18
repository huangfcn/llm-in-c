#!/usr/bin/env python3
"""Single-source Q4: quantize ALL affine-Q4 tensors of the Qwen3.8-27B BF16
checkpoint to the exact plane layout the M3 packers consume, so the build
needs only the BF16 shards (plus tokenizer) instead of a second 4bit download.

Q4 convention -- CONFIRMED against a real mlx-community 4bit tensor and against
the runtime kernels (qwen38_q4.metal / qwen38_layer.metal / qwen38_prefill.metal),
which read:  float2 quant = float2(bits & 0x0f, bits >> 4);  w = scale*quant + bias
i.e. UNSIGNED nibble in [0,15], no sign extension. mlx stores min-anchored affine:
    scale = (max - min) / 15,  bias = min,
    code  = clip(round((w - bias) / scale), 0, 15),   dequant  w = scale*code + bias.
Nibble 0 -> min, nibble 15 -> max. (This differs from the Q8 delta convention,
which is SIGNED [-127,127] midpoint-anchored -- do not cross them.)

Emitted per tensor (raw planes the packer copies/interleaves):
    <base>_codes.u8    packed nibbles, EVEN column in the LOW nibble
                       (packers validate U32 [rows, cols/8]: 8 low-first
                        nibbles per u32; cols/2 bytes == cols/8 u32)
    <base>_scale.f16   fp16 scale, one per group of 64, row-major
    <base>_bias.f16    fp16 bias  (= group min), one per group of 64

Layout:
    <OUT>/layer-NN/<suffix-minus-.weight>_{codes.u8,scale.f16,bias.f16}
    <OUT>/global/{embed_tokens,lm_head}_{codes.u8,scale.f16,bias.f16}

Correctness points (same discipline as the Q8 quantizer):
  * scale/bias rounded to fp16 FIRST (kernel reads fp16), THEN codes computed
    against those fp16 values -> codes and metadata consistent at run precision.
  * fp16 rounding can round the scale DOWN, pushing (max-bias)/scale past 15 and
    clamping -> error. So the scale is bumped UP by fp16 ULPs until the group
    fits in [0,15]. Only EXACTLY-constant groups (max==min) are degenerate:
    scale=1.0, code=0, dequant=bias.
  * the round-trip GATE dequantizes the way the KERNEL reads (unsigned nibbles,
    scale*code + bias) -- NOT the way the quantizer computed -- so a PASS means
    the bytes are right for the consumer, not merely self-consistent.

RMS_TOL 2e-1: affine Q4 group-64 has ~1/18 the levels of Q8, so honest rel_rms
runs ~5-11%; a structural bug (wrong nibble order/stride, swapped scale/bias,
signed-vs-unsigned) lands ~50-100%. 20% cleanly separates the two. Token-level
generation on the rebuilt image is the final quality arbiter.

Usage:
    python3 qwen38_q4_quantize_all.py INDEX.json OUT_DIR [--no-check]
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

GROUP = 64
LAYER_COUNT = 64
RMS_TOL = 2.0e-1
STEP_TOL = 1.5
PREFIX = "language_model.model.layers.{L}."

MLP_TENSORS = [
    ("mlp.gate_proj.weight", 17408, 5120),
    ("mlp.up_proj.weight",   17408, 5120),
    ("mlp.down_proj.weight",  5120, 17408),
]
DELTA_TENSORS = [
    ("linear_attn.in_proj_qkv.weight", 10240, 5120),
    ("linear_attn.in_proj_z.weight",    6144, 5120),
    ("linear_attn.in_proj_a.weight",      48, 5120),
    ("linear_attn.in_proj_b.weight",      48, 5120),
    ("linear_attn.out_proj.weight",     5120, 6144),
]
ATTN_TENSORS = [
    ("self_attn.q_proj.weight", 12288, 5120),
    ("self_attn.k_proj.weight",  1024, 5120),
    ("self_attn.v_proj.weight",  1024, 5120),
    ("self_attn.o_proj.weight",  5120, 6144),
]
GLOBAL_TENSORS = [
    ("language_model.model.embed_tokens.weight", 248320, 5120),
    ("language_model.lm_head.weight",            248320, 5120),
]


def layer_tensors(layer):
    return MLP_TENSORS + (ATTN_TENSORS if layer % 4 == 3 else DELTA_TENSORS)


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
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def load_bf16(self, name, rows, cols):
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
        if info["shape"] != [rows, cols]:
            raise ValueError(f"{name}: shape {info['shape']} != [{rows},{cols}]")
        lo, hi = info["data_offsets"]
        mm = self._map()
        raw = mm[self.data_start + lo: self.data_start + hi]
        bits = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
        return bits.view("<f4").reshape(rows, cols).copy()

    def load_bf16_raw(self, name, expect_values):
        """Load an arbitrary BF16 tensor as float32, flattened row-major.
        Used for pass-through vectors (norms, conv1d, A_log, dt_bias) that are
        copied verbatim (as f32) into the image. expect_values asserts the
        flattened element count; pass None to skip the check (and return the
        actual count)."""
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
        n = 1
        for d in info["shape"]:
            n *= d
        if expect_values is not None and n != expect_values:
            raise ValueError(f"{name}: {n} values != expected {expect_values}")
        lo, hi = info["data_offsets"]
        mm = self._map()
        raw = mm[self.data_start + lo: self.data_start + hi]
        bits = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
        return bits.view("<f4").reshape(-1).copy()

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._fh.close()
            self._mm = None


def float32_to_fp16_bits(values):
    return np.asarray(values, dtype="<f4").astype("<f2").view("<u2")


def quantize_group_affine_q4(block):
    """[rows,groups,64] f32 -> (codes uint8 in [0,15], scale_f16 u16,
    bias_f16 u16, degenerate_count int). UNSIGNED, min-anchored affine."""
    minimum = block.min(axis=-1)
    maximum = block.max(axis=-1)
    rng = maximum - minimum
    bias_f16 = float32_to_fp16_bits(minimum)          # bias = group min
    bias_r = bias_f16.view("<f2").astype(np.float32)

    constant = rng == 0
    scale_f16 = float32_to_fp16_bits(rng / np.float32(15.0))  # 16 levels 0..15
    scale_r = scale_f16.view("<f2").astype(np.float32)

    # Bump scale up (fp16 ULPs) until (max - bias)/scale <= 15 at fp16 precision.
    for _ in range(8):
        active = (~constant) & (scale_r > 0)
        over = active & ((maximum - bias_r) > np.float32(15.0) * scale_r)
        if not over.any():
            break
        bumped = (scale_f16.view("<u2").astype(np.uint32) + 1).astype("<u2")
        scale_f16 = np.where(over, bumped.view("<u2"), scale_f16)
        scale_r = scale_f16.view("<f2").astype(np.float32)

    scale_f16 = np.where(constant,
                         float32_to_fp16_bits(np.float32(1.0)), scale_f16)
    scale_r = scale_f16.view("<f2").astype(np.float32)
    scale_use = np.where(scale_r == 0, np.float32(1.0), scale_r)

    codes = np.rint((block - bias_r[..., None]) / scale_use[..., None])
    codes = np.clip(codes, 0, 15).astype(np.uint8)
    codes = np.where(constant[..., None], np.uint8(0), codes)
    return codes, scale_f16, bias_f16, int(constant.sum())


def pack_nibbles(codes):
    """[rows,cols] uint8 in [0,15] -> [rows,cols/2] uint8, EVEN col low nibble."""
    assert codes.shape[1] % 2 == 0
    c = codes & 0x0F
    return (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)


def dequant_like_kernel(packed, scale_u16, bias_u16, cols):
    """Exactly the Metal kernel read: UNSIGNED nibbles, w = scale*code + bias."""
    rows = packed.shape[0]
    lo = (packed & 0x0F).astype(np.float32)
    hi = ((packed >> 4) & 0x0F).astype(np.float32)
    codes = np.empty((rows, cols), dtype=np.float32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    s = scale_u16.view("<f2").astype(np.float32).repeat(GROUP, axis=1)
    b = bias_u16.view("<f2").astype(np.float32).repeat(GROUP, axis=1)
    return s * codes + b


def rel_rms(recon, ref):
    err = float(np.sqrt(np.mean((recon - ref) ** 2)))
    mag = float(np.sqrt(np.mean(ref ** 2)))
    return err / mag if mag > 0 else err


def quantize_tensor(w):
    rows, cols = w.shape
    block = w.reshape(rows, cols // GROUP, GROUP)
    codes, scale_f16, bias_f16, degen = quantize_group_affine_q4(block)
    return pack_nibbles(codes.reshape(rows, cols)), scale_f16, bias_f16, degen


def write_planes(out_dir, base, packed, scale, bias):
    (out_dir / f"{base}_codes.u8").write_bytes(packed.tobytes())
    (out_dir / f"{base}_scale.f16").write_bytes(scale.tobytes())
    (out_dir / f"{base}_bias.f16").write_bytes(bias.tobytes())


# Pass-through BF16 vectors emitted as fp32 planes (image stores them fp32).
# Flattened element counts asserted so a wrong tensor fails at quantize time.
HIDDEN = 5120
HEAD = 256          # attention head_size (q_norm/k_norm length)
DELTA_HEAD = 128    # linear_attn.norm length
SCALAR = 48         # A_log / dt_bias length
CONV_VALUES = 10240 * 4   # conv1d [10240,4,1] flattened

# (suffix, flattened_values). Emitted per delta layer.
DELTA_PASSTHROUGH = [
    ("input_layernorm.weight", HIDDEN),
    ("post_attention_layernorm.weight", HIDDEN),
    ("linear_attn.norm.weight", DELTA_HEAD),
    ("linear_attn.conv1d.weight", CONV_VALUES),
    ("linear_attn.A_log", SCALAR),
    ("linear_attn.dt_bias", SCALAR),
]
# Emitted per attention layer.
ATTN_PASSTHROUGH = [
    ("input_layernorm.weight", HIDDEN),
    ("post_attention_layernorm.weight", HIDDEN),
    ("self_attn.q_norm.weight", HEAD),
    ("self_attn.k_norm.weight", HEAD),
]


def write_passthrough(out_dir, base, values):
    """base -> <base>.f32 (raw little-endian float32, verbatim into image)."""
    (out_dir / f"{base}.f32").write_bytes(
        np.asarray(values, dtype="<f4").tobytes())


def emit_layer_passthrough(layer, weight_map, get_shard, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = ATTN_PASSTHROUGH if layer % 4 == 3 else DELTA_PASSTHROUGH
    for suffix, nval in entries:
        name = PREFIX.format(L=layer) + suffix
        if name not in weight_map:
            print(f"FATAL: passthrough {name} not in index", file=sys.stderr)
            sys.exit(3)
        vec = get_shard(weight_map[name]).load_bf16_raw(name, nval)
        # plane stem: last dotted component minus nothing (e.g.
        # "input_layernorm.weight" -> "input_layernorm"; "linear_attn.A_log"
        # -> "A_log"; keep it unambiguous by using the suffix with dots->__).
        stem = suffix.replace(".weight", "").replace(".", "__")
        write_passthrough(out_dir, stem, vec)


def process_group(label, entries, weight_map, get_shard, out_dir, do_check):
    tensors = []
    planes = {}
    bases = {}
    degen_total = 0
    for suffix, rows, cols in entries:
        if isinstance(label, int):
            name = PREFIX.format(L=label) + suffix
            base = suffix.rsplit(".", 1)[0]         # 'mlp.gate_proj'
        else:
            name = suffix
            base = name.rsplit(".", 2)[-2]          # '...embed_tokens.weight' -> 'embed_tokens'
        if name not in weight_map:
            print(f"FATAL: {name} not in index", file=sys.stderr)
            sys.exit(3)
        w = get_shard(weight_map[name]).load_bf16(name, rows, cols)
        packed, scale, bias, degen = quantize_tensor(w)
        tensors.append((name, w))
        planes[name] = (packed, scale, bias)
        bases[name] = base
        degen_total += degen

    ok = True
    worst_rms = worst_step = 0.0
    worst_name = tensors[0][0]
    if do_check:
        for name, ref in tensors:
            packed, scale, bias = planes[name]
            cols = ref.shape[1]
            rows = ref.shape[0]
            # Row-chunk the check so peak memory stays bounded even for the
            # 248320-row embed/lm_head (a full-tensor recon+ref would be tens
            # of GB and swap the machine).
            CHUNK = 8192
            sq_err = 0.0
            sq_mag = 0.0
            step = 0.0
            for r0 in range(0, rows, CHUNK):
                r1 = min(r0 + CHUNK, rows)
                rc = dequant_like_kernel(packed[r0:r1], scale[r0:r1],
                                         bias[r0:r1], cols)
                rf = ref[r0:r1]
                d = rc - rf
                sq_err += float(np.sum(d * d))
                sq_mag += float(np.sum(rf * rf))
                sc = scale[r0:r1].view("<f2").astype(np.float32).repeat(
                    GROUP, axis=1)
                step = max(step, float(
                    (np.abs(d) / np.where(sc == 0, 1.0, sc)).max()))
            rr = (sq_err / sq_mag) ** 0.5 if sq_mag > 0 else sq_err ** 0.5
            if rr > worst_rms:
                worst_rms, worst_name = rr, name
            worst_step = max(worst_step, step)
            if rr >= RMS_TOL or step >= STEP_TOL:
                ok = False

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in tensors:
        write_planes(out_dir, bases[name], *planes[name])

    return ok, worst_rms, worst_step, degen_total, worst_name


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2:
        print("usage: python3 qwen38_q4_quantize_all.py "
              "INDEX.json OUT_DIR [--no-check]", file=sys.stderr)
        return 2
    index_path, out_root = Path(args[0]), Path(args[1])
    do_check = "--no-check" not in flags
    shard_dir = index_path.parent
    weight_map = json.loads(index_path.read_text())["weight_map"]

    open_shards = {}

    def get_shard(name):
        if name not in open_shards:
            open_shards[name] = Shard(shard_dir / name)
        return open_shards[name]

    n_ok = n_fail = 0
    failures = []

    ok, wr, ws, degen, worst = process_group(
        "global", GLOBAL_TENSORS, weight_map, get_shard,
        out_root / "global", do_check)
    # global pass-through: model.norm.weight -> global/model__norm.f32
    _gn = "language_model.model.norm.weight"
    if _gn in weight_map:
        _v = get_shard(weight_map[_gn]).load_bf16_raw(_gn, HIDDEN)
        (out_root / "global").mkdir(parents=True, exist_ok=True)
        write_passthrough(out_root / "global", "model__norm", _v)
    else:
        print(f"FATAL: {_gn} not in index", file=sys.stderr); return 3
    if do_check:
        print(f"global               rel_rms {wr:.3e}  max_step {ws:.2f}  "
              f"degen {degen:6d}  worst {worst.split('.')[-2]:14s}  "
              f"{'ok' if ok else 'FAIL'}")
    else:
        print(f"global               wrote (degen {degen})")
    (n_ok, n_fail) = (n_ok + 1, n_fail) if ok else (n_ok, n_fail + 1)
    if not ok:
        failures.append("global")

    for L in range(LAYER_COUNT):
        ok, wr, ws, degen, worst = process_group(
            L, layer_tensors(L), weight_map, get_shard,
            out_root / f"layer-{L:02d}", do_check)
        emit_layer_passthrough(L, weight_map, get_shard,
                               out_root / f"layer-{L:02d}")
        kind = "attn " if L % 4 == 3 else "delta"
        if do_check:
            print(f"layer {L:02d} ({kind})       rel_rms {wr:.3e}  "
                  f"max_step {ws:.2f}  degen {degen:6d}  "
                  f"worst {worst.split('.')[-2]:14s}  {'ok' if ok else 'FAIL'}")
        else:
            print(f"layer {L:02d} ({kind})       wrote (degen {degen})")
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append(L)

        remaining = set()
        for later in range(L + 1, LAYER_COUNT):
            for suffix, _, _ in layer_tensors(later):
                name = PREFIX.format(L=later) + suffix
                if name in weight_map:
                    remaining.add(weight_map[name])
        for name in list(open_shards.keys()):
            if name not in remaining:
                open_shards[name].close()
                del open_shards[name]

    for s in open_shards.values():
        s.close()

    total = LAYER_COUNT + 1
    if do_check:
        print(f"\n{n_ok}/{total} groups PASS, {n_fail} FAIL")
        if failures:
            print("FAILED:", failures)
            return 1
        print("ALL GROUPS PASS")
    else:
        print(f"\nquantized {total} groups (no check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())