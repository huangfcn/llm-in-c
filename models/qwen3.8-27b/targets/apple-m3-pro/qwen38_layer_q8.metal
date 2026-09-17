/* Q8 twins of the DeltaNet projection GEMVs from qwen38_layer.metal.
 *
 * Each is a line-for-line mirror of its Q4 original; the ONLY change is
 * the code-plane read:
 *   Q4: quants[block*32 + lane] -> two unsigned nibbles (bits & 0x0f, bits >> 4)
 *   Q8: quants + block*64, read as SIGNED char2 at [lane] -> two int8 codes
 *
 * The int8 read MUST be signed (device const char2) so codes sign-extend
 * across [-127,127]; reading uchar turns every negative code into +128..+255.
 *
 * Metadata (Q4LayerMeta: half scale, half bias), fp32 accumulation, group
 * counts, activation dtype, residual add, and dispatch shape are all
 * IDENTICAL to the Q4 kernels. The host binds the same buffers in the same
 * order (indices 0..3 for inputs, 0..4 for output_residual); only the
 * pipeline changes. Note the weight plane doubles in size: block stride is
 * 64 bytes (one int8 per weight) vs 32 bytes (two nibbles per byte) for Q4,
 * so the delta-input/-output quant segments in the image are rows*hidden
 * (not rows*hidden/2). The packer must lay code i at byte i in the same
 * weight order the Q4 nibbles used, so char2[lane] = weights {2*lane, 2*lane+1}. */

#include <metal_stdlib>
using namespace metal;

constant uint kHiddenQ8 = 5120;
constant uint kGroupSizeLayerQ8 = 64;
constant uint kDeltaInputRowsQ8 = 16480;
constant uint kDeltaInputGroupsQ8 = 80;
constant uint kDeltaOutputGroupsQ8 = 96;

struct Q8LayerMeta {
    half scale;
    half bias;
};

/* Q8 twin of qwen38_q4_delta_inputs: half activations, 80 groups, no residual. */
kernel void qwen38_q8_delta_inputs(
    device const half *input [[buffer(0)]],
    device const char *quants [[buffer(1)]],
    device const Q8LayerMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kDeltaInputRowsQ8) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kDeltaInputGroupsQ8; ++group) {
        uint block = row * kDeltaInputGroupsQ8 + group;
        char2 codes = reinterpret_cast<device const char2 *>(
            quants + block * 64)[lane];
        Q8LayerMeta meta = metadata[block];
        float2 quant = float2(float(codes.x), float(codes.y));
        device const half2 *activation =
            reinterpret_cast<device const half2 *>(
                input + group * kGroupSizeLayerQ8);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       float2(activation[lane]));
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced;
    }
}

/* Q8 twin of qwen38_q4_delta_output_residual: float activations, 96 groups,
 * + half residual. */
kernel void qwen38_q8_delta_output_residual(
    device const float *input [[buffer(0)]],
    device const char *quants [[buffer(1)]],
    device const Q8LayerMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kHiddenQ8) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kDeltaOutputGroupsQ8; ++group) {
        uint block = row * kDeltaOutputGroupsQ8 + group;
        char2 codes = reinterpret_cast<device const char2 *>(
            quants + block * 64)[lane];
        Q8LayerMeta meta = metadata[block];
        float2 quant = float2(float(codes.x), float(codes.y));
        device const float2 *activation =
            reinterpret_cast<device const float2 *>(
                input + group * kGroupSizeLayerQ8);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       activation[lane]);
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced + float(residual[row]);
    }
}

