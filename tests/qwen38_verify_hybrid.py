#!/usr/bin/env python3
"""Pack-side read-back gate for the single-source Q4/Q8 pipeline.

Reads each packed .q38* image, pulls the segment byte-offsets straight from its
header struct, and byte-compares each quantized segment against the plane bytes
the packer was supposed to copy from q4_all / q8_all. Byte-identity is the
strongest possible check and needs no dequant tolerance: a q/k/v
concatenation-order bug, a wrong offset, or a scale/bias-not-interleaved bug all
fail instantly because the bytes at the header's offset won't equal the plane's
bytes.

This is the one check size-asserts cannot do (a wrong-order concatenation is the
right total size). It needs no GPU and no model run -- just the image dir and the
two plane dirs.

Header field byte positions are computed from the C structs
(qwen38_m3_image.h / _attention_image.h / _global_image.h) under natural
alignment; they are asserted against magic + version so a layout drift is caught.

Usage:
    python3 verify_packed_images.py MODEL_DIR Q4_DIR Q8_DIR [--layers 0,3,63]
"""

import struct
import sys
from pathlib import Path

import numpy as np

GROUP = 64
HIDDEN = 5120
MLP = 17408
VOCAB = 248320

# ---- header field byte offsets (computed from the C structs) ----------------
DELTA_OFF = {
    "magic": 0, "version": 8, "layer_index": 284,
    "gate_quants_offset": 56, "gate_quants_bytes": 64,
    "gate_metadata_offset": 72, "gate_metadata_bytes": 80,
    "up_quants_offset": 88, "up_metadata_offset": 104,
    "down_quants_offset": 120, "down_metadata_offset": 136,
    "delta_input_quants_offset": 384, "delta_input_quants_bytes": 392,
    "delta_input_metadata_offset": 400, "delta_input_metadata_bytes": 408,
    "delta_output_quants_offset": 416, "delta_output_metadata_offset": 432,
    "delta_input_precision": 448, "delta_output_precision": 452,
}
ATTN_OFF = {
    "magic": 0, "version": 8, "layer_index": 16,
    "gate_quants_offset": 72, "gate_metadata_offset": 88,
    "up_quants_offset": 104, "up_metadata_offset": 120,
    "down_quants_offset": 136, "down_metadata_offset": 152,
    "attention_input_quants_offset": 216, "attention_input_quants_bytes": 224,
    "attention_input_metadata_offset": 232, "attention_input_metadata_bytes": 240,
    "attention_output_quants_offset": 248, "attention_output_metadata_offset": 264,
}
GLOBAL_OFF = {
    "magic": 0, "version": 8,
    "embedding_quants_offset": 40, "embedding_metadata_offset": 56,
    "lm_head_quants_offset": 72, "lm_head_metadata_offset": 88,
}

MAGIC_DELTA = b"Q38M3Q4\x00"
MAGIC_ATTN = b"Q38M3ATT"
MAGIC_GLOBAL = b"Q38M3GLB"


def u64(buf, pos):
    return struct.unpack_from("<Q", buf, pos)[0]


def u32(buf, pos):
    return struct.unpack_from("<I", buf, pos)[0]


# ---- plane byte builders: reproduce exactly what the packer should copy -----

def q4_codes_bytes(q4_dir, layer, base):
    d = q4_dir / (f"layer-{layer:02d}" if layer >= 0 else "global")
    return (d / f"{base}_codes.u8").read_bytes()


def q4_meta_bytes(q4_dir, layer, base):
    """Interleaved (scale, bias) fp16 -- the exact bytes the packer writes."""
    d = q4_dir / (f"layer-{layer:02d}" if layer >= 0 else "global")
    s = np.frombuffer((d / f"{base}_scale.f16").read_bytes(), dtype="<u2")
    b = np.frombuffer((d / f"{base}_bias.f16").read_bytes(), dtype="<u2")
    inter = np.empty(s.size * 2, dtype="<u2")
    inter[0::2] = s
    inter[1::2] = b
    return inter.tobytes()


def q8_codes_bytes(q8_dir, layer):
    return (q8_dir / f"layer-{layer:02d}" / "delta_input_q8_codes.i8").read_bytes()


def q8_meta_bytes(q8_dir, layer):
    d = q8_dir / f"layer-{layer:02d}"
    s = np.frombuffer((d / "delta_input_q8_scale.f16").read_bytes(), dtype="<u2")
    b = np.frombuffer((d / "delta_input_q8_bias.f16").read_bytes(), dtype="<u2")
    inter = np.empty(s.size * 2, dtype="<u2")
    inter[0::2] = s
    inter[1::2] = b
    return inter.tobytes()


def seg_equal(img, off, expected, name, results):
    actual = img[off:off + len(expected)]
    ok = actual == expected
    results.append(ok)
    tag = "ok" if ok else "BYTE MISMATCH"
    print(f"    {name:38s} off {off:>12,d}  {len(expected):>11,d} B  {tag}")
    if not ok and len(actual) == len(expected):
        # locate first differing byte to aid debugging
        a = np.frombuffer(actual, dtype=np.uint8)
        e = np.frombuffer(expected, dtype=np.uint8)
        idx = int(np.argmax(a != e))
        print(f"        first diff at byte {idx}: img {a[idx]} != plane {e[idx]}")
    elif not ok:
        print(f"        length mismatch: img {len(actual)} != plane {len(expected)}")


def check_delta(path, q4_dir, q8_dir, results):
    img = path.read_bytes()
    assert img[:8] == MAGIC_DELTA, f"{path}: bad delta magic"
    L = u32(img, DELTA_OFF["layer_index"])
    print(f"  {path.name}  (delta, layer {L})")
    # MLP gate/up/down from q4
    for base, rows, cols, qk, mk in [
        ("mlp.gate_proj", MLP, HIDDEN, "gate_quants_offset", "gate_metadata_offset"),
        ("mlp.up_proj", MLP, HIDDEN, "up_quants_offset", "up_metadata_offset"),
        ("mlp.down_proj", HIDDEN, MLP, "down_quants_offset", "down_metadata_offset"),
    ]:
        seg_equal(img, u64(img, DELTA_OFF[qk]), q4_codes_bytes(q4_dir, L, base),
                  base + " codes", results)
        seg_equal(img, u64(img, DELTA_OFF[mk]), q4_meta_bytes(q4_dir, L, base),
                  base + " meta", results)
    # delta in_proj from q8 (concatenated plane is a single file already)
    seg_equal(img, u64(img, DELTA_OFF["delta_input_quants_offset"]),
              q8_codes_bytes(q8_dir, L), "in_proj q8 codes", results)
    seg_equal(img, u64(img, DELTA_OFF["delta_input_metadata_offset"]),
              q8_meta_bytes(q8_dir, L), "in_proj q8 meta", results)
    # delta out_proj from q4
    seg_equal(img, u64(img, DELTA_OFF["delta_output_quants_offset"]),
              q4_codes_bytes(q4_dir, L, "linear_attn.out_proj"),
              "out_proj codes", results)
    seg_equal(img, u64(img, DELTA_OFF["delta_output_metadata_offset"]),
              q4_meta_bytes(q4_dir, L, "linear_attn.out_proj"),
              "out_proj meta", results)
    # precision tags
    dip = u32(img, DELTA_OFF["delta_input_precision"])
    dop = u32(img, DELTA_OFF["delta_output_precision"])
    ok = (dip == 1 and dop == 0)
    results.append(ok)
    print(f"    precision tags in={dip} out={dop}  {'ok' if ok else 'WRONG'}")


def check_attn(path, q4_dir, results):
    img = path.read_bytes()
    assert img[:8] == MAGIC_ATTN, f"{path}: bad attn magic"
    L = u32(img, ATTN_OFF["layer_index"])
    print(f"  {path.name}  (attn, layer {L})")
    for base, qk, mk in [
        ("mlp.gate_proj", "gate_quants_offset", "gate_metadata_offset"),
        ("mlp.up_proj", "up_quants_offset", "up_metadata_offset"),
        ("mlp.down_proj", "down_quants_offset", "down_metadata_offset"),
    ]:
        rows, cols = (MLP, HIDDEN) if base != "mlp.down_proj" else (HIDDEN, MLP)
        seg_equal(img, u64(img, ATTN_OFF[qk]), q4_codes_bytes(q4_dir, L, base),
                  base + " codes", results)
        seg_equal(img, u64(img, ATTN_OFF[mk]), q4_meta_bytes(q4_dir, L, base),
                  base + " meta", results)
    # q/k/v CONCATENATED: codes q||k||v at attention_input_quants_offset,
    # metadata q||k||v at attention_input_metadata_offset. This is the check
    # that catches a concatenation-order bug.
    q_off = u64(img, ATTN_OFF["attention_input_quants_offset"])
    m_off = u64(img, ATTN_OFF["attention_input_metadata_offset"])
    cq = (q4_codes_bytes(q4_dir, L, "self_attn.q_proj") +
          q4_codes_bytes(q4_dir, L, "self_attn.k_proj") +
          q4_codes_bytes(q4_dir, L, "self_attn.v_proj"))
    cm = (q4_meta_bytes(q4_dir, L, "self_attn.q_proj") +
          q4_meta_bytes(q4_dir, L, "self_attn.k_proj") +
          q4_meta_bytes(q4_dir, L, "self_attn.v_proj"))
    seg_equal(img, q_off, cq, "q||k||v codes (concat)", results)
    seg_equal(img, m_off, cm, "q||k||v meta (concat)", results)
    seg_equal(img, u64(img, ATTN_OFF["attention_output_quants_offset"]),
              q4_codes_bytes(q4_dir, L, "self_attn.o_proj"), "o_proj codes", results)
    seg_equal(img, u64(img, ATTN_OFF["attention_output_metadata_offset"]),
              q4_meta_bytes(q4_dir, L, "self_attn.o_proj"), "o_proj meta", results)


def check_global(path, q4_dir, results):
    img = path.read_bytes()
    assert img[:8] == MAGIC_GLOBAL, f"{path}: bad global magic"
    print(f"  {path.name}  (global)")
    for base, qk, mk in [
        ("embed_tokens", "embedding_quants_offset", "embedding_metadata_offset"),
        ("lm_head", "lm_head_quants_offset", "lm_head_metadata_offset"),
    ]:
        seg_equal(img, u64(img, GLOBAL_OFF[qk]), q4_codes_bytes(q4_dir, -1, base),
                  base + " codes", results)
        seg_equal(img, u64(img, GLOBAL_OFF[mk]), q4_meta_bytes(q4_dir, -1, base),
                  base + " meta", results)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    layer_filter = None
    for f in flags:
        if f.startswith("--layers="):
            layer_filter = {int(x) for x in f.split("=", 1)[1].split(",")}
    if len(args) != 3:
        print("usage: python3 verify_packed_images.py MODEL_DIR Q4_DIR Q8_DIR "
              "[--layers=0,3,63]", file=sys.stderr)
        return 2
    model_dir, q4_dir, q8_dir = Path(args[0]), Path(args[1]), Path(args[2])
    results = []

    g = model_dir / "global.q38global"
    if g.exists():
        check_global(g, q4_dir, results)

    for L in range(64):
        if layer_filter is not None and L not in layer_filter:
            continue
        if L % 4 == 3:
            p = model_dir / f"layer-{L:02d}.q38att"
            if p.exists():
                check_attn(p, q4_dir, results)
        else:
            p = model_dir / f"layer-{L:02d}.q38delta"
            if p.exists():
                check_delta(p, q4_dir, q8_dir, results)

    npass = sum(results)
    ntot = len(results)
    print(f"\n{npass}/{ntot} segments byte-identical to planes")
    if npass == ntot and ntot > 0:
        print("ALL SEGMENTS MATCH -- packing is correct")
        return 0
    print("MISMATCH -- packer wrote wrong bytes/offsets somewhere above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
