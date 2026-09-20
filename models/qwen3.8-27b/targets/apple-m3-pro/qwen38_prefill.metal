#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

/* Batched prompt prefill for the pinned Qwen3.8-27B graph. The batch size is
 * a function constant so each shape bucket (S4..S512) compiles to a fully
 * unrolled pipeline; the kernels index every activation row through kBatch,
 * so any bucket size works as long as the host workspace matches it. Every
 * output element uses the same per-element loop order and reduction shape as
 * the one-token decode kernels, so a prefilled prompt must produce
 * bitwise-identical layer state and downstream tokens. */

constant uint kBatch [[function_constant(0)]];

constant uint kPrefillHidden = 5120;
constant uint kPrefillVocab = 248320;
constant uint kPrefillEmbeddingGroups = 80;
constant uint kPrefillQkvRows = 10240;
constant uint kPrefillDeltaStride = 16480;
constant uint kPrefillDeltaZOffset = 10240;
constant uint kPrefillDeltaAOffset = 16384;
constant uint kPrefillDeltaBOffset = 16432;
constant uint kPrefillDeltaHeads = 48;
constant uint kPrefillDeltaHeadSize = 128;
constant uint kPrefillMixerWidth = 6144;
constant uint kPrefillAttentionStride = 14336;
constant uint kPrefillAttentionKOffset = 12288;
constant uint kPrefillAttentionVOffset = 13312;
constant uint kPrefillQHeads = 24;
constant uint kPrefillKVHeads = 4;
constant uint kPrefillRotarySize = 64;
constant uint kPrefillMlpWidth = 17408;
constant float kPrefillRopeTheta = 10000000.0f;
constant float kPrefillRmsEpsilon = 1.0e-6f;

struct Q4PrefillMeta {
    half scale;
    half bias;
};

struct PrefillGemmParams {
    uint rows;
    uint groups_per_row;
};

struct PrefillAttentionParams {
    uint start_position;
    uint cache_capacity;
    /* Prefill batch size; the flash kernel pads its 8-row query tile with
     * zeros for rows past this edge. */
    uint batch;
};

kernel void qwen38_prefill_embedding(
    device const uchar *quants [[buffer(0)]],
    device const Q4PrefillMeta *metadata [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillHidden || s >= kBatch) return;
    uint token_id = token_ids[s];
    if (token_id >= kPrefillVocab) return;
    uint group = index / 64;
    uint within = index - group * 64;
    uint block = token_id * kPrefillEmbeddingGroups + group;
    uchar bits = quants[block * 32 + (within >> 1)];
    uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
    Q4PrefillMeta meta = metadata[block];
    output[s * kPrefillHidden + index] =
        half(float(meta.scale) * float(quant) + float(meta.bias));
}

kernel void qwen38_prefill_rmsnorm_f16(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    device const half *in = input + s * kPrefillHidden;
    device half *out = output + s * kPrefillHidden;
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = float(in[index]);
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partials[0] / float(kPrefillHidden) +
                          kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        out[index] = half(float(in[index]) * inv_rms * weight[index]);
    }
}

kernel void qwen38_prefill_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    device const float *in = input + s * kPrefillHidden;
    device half *out = output + s * kPrefillHidden;
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = in[index];
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partials[0] / float(kPrefillHidden) +
                          kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        out[index] = half(in[index] * inv_rms * weight[index]);
    }
}

kernel void qwen38_prefill_convert_hidden(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillHidden || s >= kBatch) return;
    output[s * kPrefillHidden + index] =
        half(input[s * kPrefillHidden + index]);
}

/* One simdgroup owns one output row and accumulates a 32-position batch
 * tile at a time, so each Q4 weight group is read once per chunk instead
 * of once per token. The batch is tiled (rather than held in one kBatch-
 * wide per-lane array) so buckets wider than the original 128-row shape,
 * such as the 512-row trunk, stay inside private-storage limits; for
 * kBatch <= 32 the loop runs once and the math is exactly the old form.
 * Every output element keeps the same per-element loop order and reduction
 * shape, so results are bitwise-identical to the one-token decode kernels. */
kernel void qwen38_prefill_q4_gemm_f16(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    for (uint s0 = 0; s0 < kBatch; s0 += 32) {
        float partial[32];
        for (uint t = 0; t < 32; ++t) partial[t] = 0.0f;
        for (uint group = 0; group < p.groups_per_row; ++group) {
            uint block = row * p.groups_per_row + group;
            uchar bits = quants[block * 32 + lane];
            Q4PrefillMeta meta = metadata[block];
            float2 quant = float2(bits & 0x0f, bits >> 4);
            float2 weight = float(meta.scale) * quant + float(meta.bias);
            for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
                device const half2 *x2 =
                    reinterpret_cast<device const half2 *>(
                        x + (s0 + t) * columns + group * 64);
                partial[t] += dot(weight, float2(x2[lane]));
            }
        }
        for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
            float value = simd_sum(partial[t]);
            if (lane == 0) output[(s0 + t) * p.rows + row] = value;
        }
    }
}

/* Scalar Q8 twin of qwen38_prefill_q4_gemm_f16 for the transcoded
 * delta-input plane (Milestone A). Line-for-line mirror; the only change
 * is the code-plane read: one SIGNED int8 per weight (char2 per lane,
 * block stride 64 bytes) instead of two unsigned nibbles per byte (block
 * stride 32). fp32 accumulation, metadata layout, and dispatch shape are
 * identical to the Q4 kernel. */
kernel void qwen38_prefill_q8_gemm_f16_scalar(
    device const half *x [[buffer(0)]],
    device const char *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    for (uint s0 = 0; s0 < kBatch; s0 += 32) {
        float partial[32];
        for (uint t = 0; t < 32; ++t) partial[t] = 0.0f;
        for (uint group = 0; group < p.groups_per_row; ++group) {
            uint block = row * p.groups_per_row + group;
            char2 codes = reinterpret_cast<device const char2 *>(
                quants + block * 64)[lane];
            Q4PrefillMeta meta = metadata[block];
            float2 weight = float(meta.scale) *
                float2(float(codes.x), float(codes.y)) + float(meta.bias);
            for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
                device const half2 *x2 =
                    reinterpret_cast<device const half2 *>(
                        x + (s0 + t) * columns + group * 64);
                partial[t] += dot(weight, float2(x2[lane]));
            }
        }
        for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
            float value = simd_sum(partial[t]);
            if (lane == 0) output[(s0 + t) * p.rows + row] = value;
        }
    }
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    for (uint s0 = 0; s0 < kBatch; s0 += 32) {
        float partial[32];
        for (uint t = 0; t < 32; ++t) partial[t] = 0.0f;
        for (uint group = 0; group < p.groups_per_row; ++group) {
            uint block = row * p.groups_per_row + group;
            uchar bits = quants[block * 32 + lane];
            Q4PrefillMeta meta = metadata[block];
            float2 quant = float2(bits & 0x0f, bits >> 4);
            float2 weight = float(meta.scale) * quant + float(meta.bias);
            for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
                device const float2 *x2 =
                    reinterpret_cast<device const float2 *>(
                        x + (s0 + t) * columns + group * 64);
                partial[t] += dot(weight, x2[lane]);
            }
        }
        for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
            float value = simd_sum(partial[t]);
            if (lane == 0) {
                output[(s0 + t) * p.rows + row] =
                    value + float(residual[(s0 + t) * p.rows + row]);
            }
        }
    }
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    for (uint s0 = 0; s0 < kBatch; s0 += 32) {
        float partial[32];
        for (uint t = 0; t < 32; ++t) partial[t] = 0.0f;
        for (uint group = 0; group < p.groups_per_row; ++group) {
            uint block = row * p.groups_per_row + group;
            uchar bits = quants[block * 32 + lane];
            Q4PrefillMeta meta = metadata[block];
            float2 quant = float2(bits & 0x0f, bits >> 4);
            float2 weight = float(meta.scale) * quant + float(meta.bias);
            for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
                device const float2 *x2 =
                    reinterpret_cast<device const float2 *>(
                        x + (s0 + t) * columns + group * 64);
                partial[t] += dot(weight, x2[lane]);
            }
        }
        for (uint t = 0; s0 + t < kBatch && t < 32; ++t) {
            float value = simd_sum(partial[t]);
            if (lane == 0) {
                output[(s0 + t) * p.rows + row] =
                    value + residual[(s0 + t) * p.rows + row];
            }
        }
    }
}

kernel void qwen38_prefill_silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillMlpWidth || s >= kBatch) return;
    uint flat = s * kPrefillMlpWidth + index;
    float value = gate[flat];
    output[flat] = (value / (1.0f + exp(-value))) * up[flat];
}

/* One thread per channel walks the chunk in order, so the carried 4-tap
 * window and the exiting convolution state match the one-token kernel. */
kernel void qwen38_prefill_delta_conv(
    device const float *projected [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= kPrefillQkvRows) return;
    uint base = channel * 4;
    float w0 = weights[base];
    float w1 = weights[base + 1];
    float w2 = weights[base + 2];
    float w3 = weights[base + 3];
    float h0 = state[base];
    float h1 = state[base + 1];
    float h2 = state[base + 2];
    float h3 = state[base + 3];
    for (uint s = 0; s < kBatch; ++s) {
        h0 = h1;
        h1 = h2;
        h2 = h3;
        h3 = projected[s * kPrefillDeltaStride + channel];
        float value = h0 * w0 + h1 * w1 + h2 * w2 + h3 * w3;
        output[s * kPrefillQkvRows + channel] =
            value / (1.0f + exp(-value));
    }
    state[base] = h0;
    state[base + 1] = h1;
    state[base + 2] = h2;
    state[base + 3] = h3;
}

kernel void qwen38_prefill_delta_prepare(
    device const float *convolved_qkv [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *query [[buffer(4)]],
    device float *key [[buffer(5)]],
    device float *value [[buffer(6)]],
    device float *decay [[buffer(7)]],
    device float *beta [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float q_squared[128];
    threadgroup float k_squared[128];
    uint value_head = group_id.x;
    uint s = group_id.y;
    uint key_head = value_head / 3;
    device const float *conv = convolved_qkv + s * kPrefillQkvRows;
    float raw_q = conv[key_head * 128 + tid];
    float raw_k = conv[2048 + key_head * 128 + tid];
    q_squared[tid] = raw_q * raw_q;
    k_squared[tid] = raw_k * raw_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            q_squared[tid] += q_squared[tid + stride];
            k_squared[tid] += k_squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output_index = s * kPrefillMixerWidth + value_head * 128 + tid;
    query[output_index] = raw_q *
        rsqrt(q_squared[0] + 128.0e-6f) * rsqrt(128.0f);
    key[output_index] = raw_k * rsqrt(k_squared[0] + 128.0e-6f);
    value[output_index] = conv[4096 + value_head * 128 + tid];
    if (tid == 0) {
        device const float *proj = projected + s * kPrefillDeltaStride;
        float a = proj[kPrefillDeltaAOffset + value_head] +
                  dt_bias[value_head];
        float softplus = max(a, 0.0f) + log(1.0f + exp(-abs(a)));
        float g = -exp(a_log[value_head]) * softplus;
        decay[s * kPrefillDeltaHeads + value_head] = exp(g);
        float b = proj[kPrefillDeltaBOffset + value_head];
        beta[s * kPrefillDeltaHeads + value_head] =
            1.0f / (1.0f + exp(-b));
    }
}

/* The delta rule stays sequential in time inside the kernel; parallelism is
 * across the 48 x 128 state columns. Each step is the one-token kernel. */
kernel void qwen38_prefill_delta_recurrent(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *decay [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= kPrefillDeltaHeads * kPrefillDeltaHeadSize) return;
    uint head = index / kPrefillDeltaHeadSize;
    uint value_index = index % kPrefillDeltaHeadSize;
    uint state_base =
        head * kPrefillDeltaHeadSize * kPrefillDeltaHeadSize + value_index;
    for (uint s = 0; s < kBatch; ++s) {
        uint vector_base = s * kPrefillMixerWidth +
                           head * kPrefillDeltaHeadSize;
        float head_decay = decay[s * kPrefillDeltaHeads + head];
        float kv_memory = 0.0f;
        float previous_output = 0.0f;
        float key_query = 0.0f;
        for (uint key_index = 0; key_index < kPrefillDeltaHeadSize;
             ++key_index) {
            float old_value =
                state[state_base + key_index * kPrefillDeltaHeadSize];
            float decayed = old_value * head_decay;
            kv_memory += decayed * key[vector_base + key_index];
            previous_output += decayed * query[vector_base + key_index];
            key_query += key[vector_base + key_index] *
                         query[vector_base + key_index];
        }
        float delta = (value[vector_base + value_index] - kv_memory) *
                      beta[s * kPrefillDeltaHeads + head];
        for (uint key_index = 0; key_index < kPrefillDeltaHeadSize;
             ++key_index) {
            uint state_index =
                state_base + key_index * kPrefillDeltaHeadSize;
            state[state_index] = state[state_index] * head_decay +
                                 key[vector_base + key_index] * delta;
        }
        output[s * kPrefillMixerWidth + index] =
            previous_output + key_query * delta;
    }
}

kernel void qwen38_prefill_delta_gated_norm(
    device const float *core [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float squared[128];
    uint head = group_id.x;
    uint s = group_id.y;
    uint index = s * kPrefillMixerWidth + head * 128 + tid;
    float value = core[index];
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            squared[tid] += squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value * rsqrt(squared[0] / 128.0f +
                                     kPrefillRmsEpsilon) * weight[tid];
    float z = projected[s * kPrefillDeltaStride + kPrefillDeltaZOffset +
                        head * 128 + tid];
    float silu_z = z / (1.0f + exp(-z));
    output[index] = normalized * silu_z;
}


/* Capture variants: identical math to the kernels above plus the
 * side outputs the speculative path needs - per-position GDN factor
 * checkpoints for replay-free partial accepts and half activation
 * copies for the small-batch half-MMA GEMMs. They compile separately
 * so the plain kernels keep their exact code generation, and the
 * host encodes them only for batch 2-8 outside exact mode. */

kernel void qwen38_prefill_delta_conv_cap(
    device const float *projected [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *output [[buffer(3)]],
    device float *window_checkpoint [[buffer(4)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= kPrefillQkvRows) return;
    uint base = channel * 4;
    float w0 = weights[base];
    float w1 = weights[base + 1];
    float w2 = weights[base + 2];
    float w3 = weights[base + 3];
    float h0 = state[base];
    float h1 = state[base + 1];
    float h2 = state[base + 2];
    float h3 = state[base + 3];
    for (uint s = 0; s < kBatch; ++s) {
        h0 = h1;
        h1 = h2;
        h2 = h3;
        h3 = projected[s * kPrefillDeltaStride + channel];
        float value = h0 * w0 + h1 * w1 + h2 * w2 + h3 * w3;
        output[s * kPrefillQkvRows + channel] =
            value / (1.0f + exp(-value));
        if (kBatch <= 8) {
            uint slot = s * kPrefillQkvRows * 4 + base;
            window_checkpoint[slot] = h0;
            window_checkpoint[slot + 1] = h1;
            window_checkpoint[slot + 2] = h2;
            window_checkpoint[slot + 3] = h3;
        }
    }
    state[base] = h0;
    state[base + 1] = h1;
    state[base + 2] = h2;
    state[base + 3] = h3;
}

kernel void qwen38_prefill_delta_prepare_cap(
    device const float *convolved_qkv [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *query [[buffer(4)]],
    device float *key [[buffer(5)]],
    device float *value [[buffer(6)]],
    device float *decay [[buffer(7)]],
    device float *beta [[buffer(8)]],
    device float *key_checkpoint [[buffer(9)]],
    device float *value_checkpoint [[buffer(10)]],
    device float *decay_checkpoint [[buffer(11)]],
    device float *beta_checkpoint [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float q_squared[128];
    threadgroup float k_squared[128];
    uint value_head = group_id.x;
    uint s = group_id.y;
    uint key_head = value_head / 3;
    device const float *conv = convolved_qkv + s * kPrefillQkvRows;
    float raw_q = conv[key_head * 128 + tid];
    float raw_k = conv[2048 + key_head * 128 + tid];
    q_squared[tid] = raw_q * raw_q;
    k_squared[tid] = raw_k * raw_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            q_squared[tid] += q_squared[tid + stride];
            k_squared[tid] += k_squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output_index = s * kPrefillMixerWidth + value_head * 128 + tid;
    query[output_index] = raw_q *
        rsqrt(q_squared[0] + 128.0e-6f) * rsqrt(128.0f);
    key[output_index] = raw_k * rsqrt(k_squared[0] + 128.0e-6f);
    value[output_index] = conv[4096 + value_head * 128 + tid];
    if (kBatch <= 8) {
        key_checkpoint[output_index] = key[output_index];
        value_checkpoint[output_index] = value[output_index];
    }
    if (tid == 0) {
        device const float *proj = projected + s * kPrefillDeltaStride;
        float a = proj[kPrefillDeltaAOffset + value_head] +
                  dt_bias[value_head];
        float softplus = max(a, 0.0f) + log(1.0f + exp(-abs(a)));
        float g = -exp(a_log[value_head]) * softplus;
        decay[s * kPrefillDeltaHeads + value_head] = exp(g);
        float b = proj[kPrefillDeltaBOffset + value_head];
        beta[s * kPrefillDeltaHeads + value_head] =
            1.0f / (1.0f + exp(-b));
        if (kBatch <= 8) {
            decay_checkpoint[s * kPrefillDeltaHeads + value_head] =
                decay[s * kPrefillDeltaHeads + value_head];
            beta_checkpoint[s * kPrefillDeltaHeads + value_head] =
                beta[s * kPrefillDeltaHeads + value_head];
        }
    }
}

kernel void qwen38_prefill_silu_mul_cap(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    device half *x_half [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillMlpWidth || s >= kBatch) return;
    uint flat = s * kPrefillMlpWidth + index;
    float value = gate[flat];
    float activated = (value / (1.0f + exp(-value))) * up[flat];
    output[flat] = activated;
    /* Half copy in row layout feeds the small-batch half-MMA GEMM. */
    x_half[flat] = half(activated);
}

kernel void qwen38_prefill_delta_gated_norm_cap(
    device const float *core [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    device half *x_half [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float squared[128];
    uint head = group_id.x;
    uint s = group_id.y;
    uint index = s * kPrefillMixerWidth + head * 128 + tid;
    float value = core[index];
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            squared[tid] += squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value * rsqrt(squared[0] / 128.0f +
                                     kPrefillRmsEpsilon) * weight[tid];
    float z = projected[s * kPrefillDeltaStride + kPrefillDeltaZOffset +
                        head * 128 + tid];
    float silu_z = z / (1.0f + exp(-z));
    float gated = normalized * silu_z;
    output[index] = gated;
    x_half[index] = half(gated);
}

kernel void qwen38_prefill_attention_softmax_value_cap(
    device const float *scores [[buffer(0)]],
    device const half *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant PrefillAttentionParams &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    device half *x_half [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint s = group_id.y;
    uint context_length = parameters.start_position + s + 1;
    uint score_base = (s * kPrefillQHeads + q_head) *
                      parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < context_length; position += 256) {
        local_max = max(local_max, scores[score_base + position]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] =
            max(reduction[tid], reduction[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduction[0];
    /* The same threadgroup array carries the next reduction; every thread
     * must finish reading the maximum before any thread overwrites slot 0. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < context_length; position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < context_length; ++position) {
        uint cache_index =
            (position * kPrefillKVHeads + kv_head) * 256 + tid;
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * value_cache[cache_index];
    }
    uint output_index = s * kPrefillMixerWidth + q_head * 256 + tid;
    float gated_value = value * query_gate[output_index];
    output[output_index] = gated_value;
    x_half[output_index] = half(gated_value);
}

inline float prefill_rope_component(threadgroup const float *values,
                                    uint dimension, uint position) {
    if (dimension >= kPrefillRotarySize) return values[dimension];
    uint frequency = dimension & 31u;
    float exponent = -2.0f * float(frequency) /
                     float(kPrefillRotarySize);
    float angle = float(position) * pow(kPrefillRopeTheta, exponent);
    float c = cos(angle);
    float ss = sin(angle);
    if (dimension < 32) {
        return values[dimension] * c - values[dimension + 32] * ss;
    }
    return values[dimension] * c + values[dimension - 32] * ss;
}

kernel void qwen38_prefill_attention_query(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *query [[buffer(3)]],
    device float *query_gate [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    uint s = group_id.y;
    device const float *proj = projected + s * kPrefillAttentionStride;
    uint projection_base = head * 512;
    float value = proj[projection_base + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kPrefillRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = s * kPrefillMixerWidth + head * 256 + tid;
    query[index] = prefill_rope_component(
        values, tid, parameters.start_position + s);
    float gate = proj[projection_base + 256 + tid];
    query_gate[index] = 1.0f / (1.0f + exp(-gate));
}

kernel void qwen38_prefill_attention_key_value(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device half *key_cache [[buffer(3)]],
    device half *value_cache [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    uint s = group_id.y;
    device const float *proj = projected + s * kPrefillAttentionStride;
    float value = proj[kPrefillAttentionKOffset + head * 256 + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kPrefillRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint position = parameters.start_position + s;
    uint cache_index = (position * kPrefillKVHeads + head) * 256 + tid;
    key_cache[cache_index] = prefill_rope_component(values, tid, position);
    value_cache[cache_index] =
        proj[kPrefillAttentionVOffset + head * 256 + tid];
}

kernel void qwen38_prefill_attention_scores(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *scores [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint s = group_id.y;
    uint flat = group_id.x * simdgroups_per_group + simdgroup_index;
    uint context_length = parameters.start_position + s + 1;
    uint score_count = kPrefillQHeads * context_length;
    if (flat >= score_count) return;
    uint q_head = flat / context_length;
    uint position = flat - q_head * context_length;
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    uint query_base = s * kPrefillMixerWidth + q_head * 256;
    uint key_base = (position * kPrefillKVHeads + kv_head) * 256;
    float partial = 0.0f;
    for (uint index = lane; index < 256; index += 32) {
        partial += query[query_base + index] * key_cache[key_base + index];
    }
    float score = simd_sum(partial) * (1.0f / 16.0f);
    if (lane == 0) {
        scores[(s * kPrefillQHeads + q_head) * parameters.cache_capacity +
               position] = score;
    }
}

kernel void qwen38_prefill_attention_softmax_value(
    device const float *scores [[buffer(0)]],
    device const half *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant PrefillAttentionParams &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint s = group_id.y;
    uint context_length = parameters.start_position + s + 1;
    uint score_base = (s * kPrefillQHeads + q_head) *
                      parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < context_length; position += 256) {
        local_max = max(local_max, scores[score_base + position]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] =
            max(reduction[tid], reduction[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduction[0];
    /* The same threadgroup array carries the next reduction; every thread
     * must finish reading the maximum before any thread overwrites slot 0. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < context_length; position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < context_length; ++position) {
        uint cache_index =
            (position * kPrefillKVHeads + kv_head) * 256 + tid;
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * value_cache[cache_index];
    }
    uint output_index = s * kPrefillMixerWidth + q_head * 256 + tid;
    output[output_index] = value * query_gate[output_index];
}

/* Q8_0 KV cache (QWEN38_KV_Q8): each 256-dim head vector is stored as
 * 256 int8 values followed by one fp32 scale, so a vector occupies
 * kPrefillKVQ8Stride bytes; the dequantized value is int8 * scale. The
 * scale spans the whole dot-product dimension, so it factors out of the
 * Q.K reduction and multiplies the P.V accumulation per position, which
 * keeps the readers' structure identical to the fp16 path. */
constant uint kPrefillKVQ8Stride = 260;

kernel void qwen38_prefill_attention_key_value_q8(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device char *key_cache [[buffer(3)]],
    device char *value_cache [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    threadgroup float key_magnitudes[256];
    threadgroup float value_magnitudes[256];
    uint head = group_id.x;
    uint s = group_id.y;
    device const float *proj = projected + s * kPrefillAttentionStride;
    float value = proj[kPrefillAttentionKOffset + head * 256 + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kPrefillRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint position = parameters.start_position + s;
    float key = prefill_rope_component(values, tid, position);
    float raw_value = proj[kPrefillAttentionVOffset + head * 256 + tid];
    /* Independent per-vector scales: each of K and V uses the full
     * [-127, 127] code range against its own maximum. */
    key_magnitudes[tid] = fabs(key);
    value_magnitudes[tid] = fabs(raw_value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            key_magnitudes[tid] =
                max(key_magnitudes[tid], key_magnitudes[tid + stride]);
            value_magnitudes[tid] =
                max(value_magnitudes[tid], value_magnitudes[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* scale = max/127; a zero vector keeps scale 0 and all-zero codes. */
    float key_scale = key_magnitudes[0] / 127.0f;
    float value_scale = value_magnitudes[0] / 127.0f;
    uint vector_index = (position * kPrefillKVHeads + head) *
                        kPrefillKVQ8Stride;
    device char *key_vector = key_cache + vector_index;
    device char *value_vector = value_cache + vector_index;
    int code = 0;
    int value_code = 0;
    if (key_scale > 0.0f)
        code = max(-127, min(127, (int)rint(key / key_scale)));
    if (value_scale > 0.0f)
        value_code = max(-127, min(127,
                                   (int)rint(raw_value / value_scale)));
    key_vector[tid] = (char)code;
    value_vector[tid] = (char)value_code;
    if (tid == 0) {
        *(device float *)(key_vector + 256) = key_scale;
        *(device float *)(value_vector + 256) = value_scale;
    }
}

kernel void qwen38_prefill_attention_scores_q8(
    device const float *query [[buffer(0)]],
    device const char *key_cache [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *scores [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint s = group_id.y;
    uint flat = group_id.x * simdgroups_per_group + simdgroup_index;
    uint context_length = parameters.start_position + s + 1;
    uint score_count = kPrefillQHeads * context_length;
    if (flat >= score_count) return;
    uint q_head = flat / context_length;
    uint position = flat - q_head * context_length;
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    uint query_base = s * kPrefillMixerWidth + q_head * 256;
    const device char *key_vector = key_cache +
        ((position * kPrefillKVHeads + kv_head) * kPrefillKVQ8Stride);
    float scale = *(const device float *)(key_vector + 256);
    float partial = 0.0f;
    for (uint index = lane; index < 256; index += 32) {
        partial += query[query_base + index] *
                   (float)(short)(key_vector[index]);
    }
    float score = simd_sum(partial) * scale * (1.0f / 16.0f);
    if (lane == 0) {
        scores[(s * kPrefillQHeads + q_head) * parameters.cache_capacity +
               position] = score;
    }
}

kernel void qwen38_prefill_attention_softmax_value_q8(
    device const float *scores [[buffer(0)]],
    device const char *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant PrefillAttentionParams &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint s = group_id.y;
    uint context_length = parameters.start_position + s + 1;
    uint score_base = (s * kPrefillQHeads + q_head) *
                      parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < context_length; position += 256) {
        local_max = max(local_max, scores[score_base + position]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] =
            max(reduction[tid], reduction[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduction[0];
    /* The same threadgroup array carries the next reduction; every thread
     * must finish reading the maximum before any thread overwrites slot 0. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < context_length; position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < context_length; ++position) {
        const device char *vector = value_cache +
            ((position * kPrefillKVHeads + kv_head) *
             kPrefillKVQ8Stride);
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * *(const device float *)(vector + 256) *
               (float)(short)(vector[tid]);
    }
    uint output_index = s * kPrefillMixerWidth + q_head * 256 + tid;
    output[output_index] = value * query_gate[output_index];
}

kernel void qwen38_prefill_attention_softmax_value_cap_q8(
    device const float *scores [[buffer(0)]],
    device const char *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant PrefillAttentionParams &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    device half *x_half [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint s = group_id.y;
    uint context_length = parameters.start_position + s + 1;
    uint score_base = (s * kPrefillQHeads + q_head) *
                      parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < context_length; position += 256) {
        local_max = max(local_max, scores[score_base + position]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] =
            max(reduction[tid], reduction[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduction[0];
    /* The same threadgroup array carries the next reduction; every thread
     * must finish reading the maximum before any thread overwrites slot 0. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < context_length; position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < context_length; ++position) {
        const device char *vector = value_cache +
            ((position * kPrefillKVHeads + kv_head) *
             kPrefillKVQ8Stride);
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * *(const device float *)(vector + 256) *
               (float)(short)(vector[tid]);
    }
    uint output_index = s * kPrefillMixerWidth + q_head * 256 + tid;
    float gated_value = value * query_gate[output_index];
    output[output_index] = gated_value;
    x_half[output_index] = half(gated_value);
}

/* Flash-attention prefill (QWEN38_FLASH_PREFILL=1), modeled on llama.cpp's
 * Metal flash_attn_ext. One 4-simdgroup threadgroup handles 8 query rows of a
 * single q-head and streams the KV cache in 64-position blocks. QK^T and P.V
 * run as 8x8 simdgroup matrix multiply-accumulates (the Apple-GPU tensor
 * path), and K/V are read straight from device memory through
 * simdgroup_load, so nothing is staged. Threadgroup memory holds the 8x256
 * half query tile (4 KB), the 8x128 float score scratch (4 KB) and the
 * 8x256 float O accumulator (8 KB) - 16 KB total, well under the default
 * budget.
 * Each simdgroup owns two query rows for the online softmax (running max and
 * running sum kept in registers), while all four simdgroups collaborate on
 * every block's QK^T tiles and split the P.V output dimensions. Causality is
 * structural: full blocks need no mask, and the final partial block masks
 * positions past each row's context length during its softmax step. The
 * arithmetic differs from the two-pass scores path, so this kernel is
 * validated by token parity, not bitwise equality. */

constant uint kFlashRows = 8;     /* query rows per threadgroup */
constant uint kFlashBlock = 64;   /* KV positions per block */

kernel void qwen38_prefill_flash_attention(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device const float *query_gate [[buffer(3)]],
    constant PrefillAttentionParams &parameters [[buffer(4)]],
    device float *output [[buffer(5)]],
    device half *x_half [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    /* Shared tiles: all 8 query rows. QK^T and P.V each use one 8x8 MMA per
     * (position tile, dim step), so every simdgroup works on all 8 rows;
     * the work split is over positions (QK^T) and output dims (P.V).
     * Simdgroup g owns output dims {8g + 32k, k = 0..7} for every row. */
    threadgroup half sq[kFlashRows * 256];   /* query rows, pre-scaled 1/16 */
    threadgroup float ss[kFlashRows * 128];  /* scores/probs: [row][position] */
    threadgroup float so[kFlashRows * 256];  /* O accumulator: [row][dim] */

    const uint q_head = group_id.x;
    const uint row0 = group_id.y * kFlashRows;
    const uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);

    /* Load the 8 query rows into the shared tile, pre-scaled by 1/16 so the
     * QK^T products come out as final attention scores. Rows past the batch
     * edge load zeros; their results are never stored. */
    for (uint i = tid; i < kFlashRows * 256; i += 128) {
        const uint r = i / 256;
        const uint d = i % 256;
        sq[r * 256 + d] = (row0 + r < parameters.batch)
            ? half(query[(row0 + r) * kPrefillMixerWidth + q_head * 256 + d]
                   * (1.0f / 16.0f))
            : half(0.0f);
    }

    /* Zero the O accumulator: each thread covers one element of every
     * 32-dim window (k = 0..7), so all 256 dims of every row are zeroed;
     * the four simdgroups write the same zeros redundantly. */
    for (uint r = 0; r < kFlashRows; ++r) {
        for (uint k = 0; k < 8; ++k)
            so[r * 256 + 32 * k + lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Per-row running softmax state. Explicit scalars (not an indexed
     * array) so the two rows' reductions cannot be vectorized into one:
     * m_a/ssum_a track row sgitg, m_b/ssum_b track row sgitg + 4. */
    float m_a, m_b, ssum_a, ssum_b;
    m_a = -INFINITY;
    m_b = -INFINITY;
    ssum_a = 0.0f;
    ssum_b = 0.0f;

    /* Row r attends to positions [0, start_position + row0 + r). The last
     * block is partial only for the smallest rows of the group. */
    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;

        /* QK^T: the block's 64 positions form 8 tiles of 8; simdgroup g
         * computes tiles {g, g+4}, each an 8x8 MMA over all 256 query dims.
         * mq[row i][dim j] = Q[row i][dim], mk[dim j][pos i] via the
         * transposed load with the 1024-half stride that skips the other
         * KV heads, so mqk[row i][pos j] = score. */
        for (uint cc = 0; cc < kFlashBlock / 8 / 4; ++cc) {
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            const uint pos_tile = sgitg + 4 * cc;
            const device half *pk = key_cache +
                ((block_start + 8 * pos_tile) * kPrefillKVHeads + kv_head) * 256;
            for (uint i = 0; i < 32; ++i) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, sq + i * 8, 256);
                simdgroup_load(mk, pk + i * 8, 1024, 0, true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, ss + 8 * pos_tile, 128, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax for this simdgroup's two rows, written with
         * explicit per-row scalars. Lane l owns score columns {2l, 2l+1};
         * the last block masks positions past each row's context length.
         * Row A is sgitg, row B is sgitg + 4. */
        {
            const uint r = sgitg;
            const float old_m = m_a;
            float2 s2 = float2(ss[r * 128 + 2 * lane],
                               ss[r * 128 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_a = simd_max(max(old_m, max(s2[0], s2[1])));
            const float alpha = exp(old_m - m_a);
            const float2 p2 = exp(s2 - m_a);
            ssum_a = ssum_a * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 128 + 2 * lane] = p2[0];
            ss[r * 128 + 2 * lane + 1] = p2[1];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        {
            const uint r = sgitg + 4;
            const float old_m = m_b;
            float2 s2 = float2(ss[r * 128 + 2 * lane],
                               ss[r * 128 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg + 4;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_b = simd_max(max(old_m, max(s2[0], s2[1])));
            const float alpha = exp(old_m - m_b);
            const float2 p2 = exp(s2 - m_b);
            ssum_b = ssum_b * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 128 + 2 * lane] = p2[0];
            ss[r * 128 + 2 * lane + 1] = p2[1];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P.V: O += P.V over the block. All 8 rows' probabilities are valid
         * (each simdgroup softmaxed its own two rows), so one 8x8 MMA per
         * (position, dim) tile covers every row at once. Per 16-position
         * sub-block and per 8-position half k: lo[row i][dim j] += sum
         * over the half's 8 positions of P[row i][pos] * V[pos][dim]. The
         * probability tile must be loaded per half (vs holds only 8
         * columns): vs[row i][pos c] = P[row i][pos 16cc + 8k + c], and V
         * is loaded NOT transposed (the position axis is already V's first
         * memory axis) so mv[pos c][dim j] = V[pos 16cc + 8k + c][db + j];
         * the MMA then contracts the position index exactly. */
        for (uint cc = 0; cc < kFlashBlock / 16; ++cc) {
            /* Dim partition: within each 32-dim window [32m, 32m+32) the
             * four simdgroups own the consecutive 8-dim quarters, so
             * simdgroup g owns chunks [8g + 32k, 8g + 32k + 8) for k = 0..7
             * (64 dims total; the sets tile all 256 without overlap). */
            for (uint ii = 0; ii < 8; ++ii) {
                const uint db = 8 * sgitg + 32 * ii;
                simdgroup_float8x8 lo;
                simdgroup_load(lo, so + db, 256);
                for (uint k = 0; k < 2; ++k) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 16 * cc + 8 * k, 128);
                    const device half *pv = value_cache +
                        ((block_start + 16 * cc + 8 * k) * kPrefillKVHeads +
                         kv_head) * 256;
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + db, 1024, 0, false);
                    simdgroup_multiply_accumulate(lo, vs, mv, lo);
                }
                simdgroup_store(lo, so + db, 256);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Normalize and apply the sigmoid query gate. Every simdgroup has seen
     * every position of its two rows, so ssum is complete; all four
     * simdgroups' O chunks are valid behind the last barrier. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (jj == 0 ? ssum_a : ssum_b) > 0.0f
            ? 1.0f / (jj == 0 ? ssum_a : ssum_b) : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const uint d = lane + 32 * i;
            const uint output_index =
                (row0 + r) * kPrefillMixerWidth + q_head * 256 + d;
            const float gated_value = so[r * 256 + d] * scale *
                                      query_gate[output_index];
            output[output_index] = gated_value;
            if (x_half != nullptr)
                x_half[output_index] = half(gated_value);
        }
    }
}

/* Experimental P.V accumulator-reuse kernel.  Same math and FP16-KV
 * block-64 scan as qwen38_prefill_flash_attention; only the P.V loop order
 * changes so each output accumulator tile stays resident across all four
 * 16-position sub-blocks.  Select with QWEN38_FLASH_PV_REUSE=1. */
kernel void qwen38_prefill_flash_attention_pvreuse(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device const float *query_gate [[buffer(3)]],
    constant PrefillAttentionParams &parameters [[buffer(4)]],
    device float *output [[buffer(5)]],
    device half *x_half [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    /* Shared tiles: all 8 query rows. QK^T and P.V each use one 8x8 MMA per
     * (position tile, dim step), so every simdgroup works on all 8 rows;
     * the work split is over positions (QK^T) and output dims (P.V).
     * Simdgroup g owns output dims {8g + 32k, k = 0..7} for every row. */
    threadgroup half sq[kFlashRows * 256];   /* query rows, pre-scaled 1/16 */
    threadgroup float ss[kFlashRows * 128];  /* scores/probs: [row][position] */
    threadgroup float so[kFlashRows * 256];  /* O accumulator: [row][dim] */

    const uint q_head = group_id.x;
    const uint row0 = group_id.y * kFlashRows;
    const uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);

    /* Load the 8 query rows into the shared tile, pre-scaled by 1/16 so the
     * QK^T products come out as final attention scores. Rows past the batch
     * edge load zeros; their results are never stored. */
    for (uint i = tid; i < kFlashRows * 256; i += 128) {
        const uint r = i / 256;
        const uint d = i % 256;
        sq[r * 256 + d] = (row0 + r < parameters.batch)
            ? half(query[(row0 + r) * kPrefillMixerWidth + q_head * 256 + d]
                   * (1.0f / 16.0f))
            : half(0.0f);
    }

    /* Zero the O accumulator: each thread covers one element of every
     * 32-dim window (k = 0..7), so all 256 dims of every row are zeroed;
     * the four simdgroups write the same zeros redundantly. */
    for (uint r = 0; r < kFlashRows; ++r) {
        for (uint k = 0; k < 8; ++k)
            so[r * 256 + 32 * k + lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Per-row running softmax state. Explicit scalars (not an indexed
     * array) so the two rows' reductions cannot be vectorized into one:
     * m_a/ssum_a track row sgitg, m_b/ssum_b track row sgitg + 4. */
    float m_a, m_b, ssum_a, ssum_b;
    m_a = -INFINITY;
    m_b = -INFINITY;
    ssum_a = 0.0f;
    ssum_b = 0.0f;

    /* Row r attends to positions [0, start_position + row0 + r). The last
     * block is partial only for the smallest rows of the group. */
    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;

        /* QK^T: the block's 64 positions form 8 tiles of 8; simdgroup g
         * computes tiles {g, g+4}, each an 8x8 MMA over all 256 query dims.
         * mq[row i][dim j] = Q[row i][dim], mk[dim j][pos i] via the
         * transposed load with the 1024-half stride that skips the other
         * KV heads, so mqk[row i][pos j] = score. */
        for (uint cc = 0; cc < kFlashBlock / 8 / 4; ++cc) {
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
            const uint pos_tile = sgitg + 4 * cc;
            const device half *pk = key_cache +
                ((block_start + 8 * pos_tile) * kPrefillKVHeads + kv_head) * 256;
            for (uint i = 0; i < 32; ++i) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, sq + i * 8, 256);
                simdgroup_load(mk, pk + i * 8, 1024, 0, true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, ss + 8 * pos_tile, 128, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax for this simdgroup's two rows, written with
         * explicit per-row scalars. Lane l owns score columns {2l, 2l+1};
         * the last block masks positions past each row's context length.
         * Row A is sgitg, row B is sgitg + 4. */
        {
            const uint r = sgitg;
            const float old_m = m_a;
            float2 s2 = float2(ss[r * 128 + 2 * lane],
                               ss[r * 128 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_a = simd_max(max(old_m, max(s2[0], s2[1])));
            const float alpha = exp(old_m - m_a);
            const float2 p2 = exp(s2 - m_a);
            ssum_a = ssum_a * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 128 + 2 * lane] = p2[0];
            ss[r * 128 + 2 * lane + 1] = p2[1];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        {
            const uint r = sgitg + 4;
            const float old_m = m_b;
            float2 s2 = float2(ss[r * 128 + 2 * lane],
                               ss[r * 128 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg + 4;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_b = simd_max(max(old_m, max(s2[0], s2[1])));
            const float alpha = exp(old_m - m_b);
            const float2 p2 = exp(s2 - m_b);
            ssum_b = ssum_b * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 128 + 2 * lane] = p2[0];
            ss[r * 128 + 2 * lane + 1] = p2[1];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P.V: O += P.V over the block. All 8 rows' probabilities are valid
         * (each simdgroup softmaxed its own two rows), so one 8x8 MMA per
         * (position, dim) tile covers every row at once. Per 16-position
         * sub-block and per 8-position half k: lo[row i][dim j] += sum
         * over the half's 8 positions of P[row i][pos] * V[pos][dim]. The
         * probability tile must be loaded per half (vs holds only 8
         * columns): vs[row i][pos c] = P[row i][pos 16cc + 8k + c], and V
         * is loaded NOT transposed (the position axis is already V's first
         * memory axis) so mv[pos c][dim j] = V[pos 16cc + 8k + c][db + j];
         * the MMA then contracts the position index exactly. */
        /* P.V accumulator-reuse variant: keep each 8x8 output tile in the
         * simdgroup matrix across the whole 64-position block.  The original
         * loop order loads/stores `lo` once per 16 positions (4x per block).
         * Swapping ii/cc preserves the eight MMA updates in the same order,
         * but performs only one threadgroup load and one store per output
         * tile per 64 positions.  K/V traffic and arithmetic are unchanged. */
        for (uint ii = 0; ii < 8; ++ii) {
            const uint db = 8 * sgitg + 32 * ii;
            simdgroup_float8x8 lo;
            simdgroup_load(lo, so + db, 256);
            for (uint cc = 0; cc < kFlashBlock / 16; ++cc) {
                for (uint k = 0; k < 2; ++k) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 16 * cc + 8 * k, 128);
                    const device half *pv = value_cache +
                        ((block_start + 16 * cc + 8 * k) * kPrefillKVHeads +
                         kv_head) * 256;
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + db, 1024, 0, false);
                    simdgroup_multiply_accumulate(lo, vs, mv, lo);
                }
            }
            simdgroup_store(lo, so + db, 256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Normalize and apply the sigmoid query gate. Every simdgroup has seen
     * every position of its two rows, so ssum is complete; all four
     * simdgroups' O chunks are valid behind the last barrier. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (jj == 0 ? ssum_a : ssum_b) > 0.0f
            ? 1.0f / (jj == 0 ? ssum_a : ssum_b) : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const uint d = lane + 32 * i;
            const uint output_index =
                (row0 + r) * kPrefillMixerWidth + q_head * 256 + d;
            const float gated_value = so[r * 256 + d] * scale *
                                      query_gate[output_index];
            output[output_index] = gated_value;
            if (x_half != nullptr)
                x_half[output_index] = half(gated_value);
        }
    }
}




/* Experimental QK + P.V reuse kernel.  Builds on the validated P.V
 * accumulator-reuse path and additionally keeps both QK score accumulators
 * for a simdgroup's two 8-position tiles live at once.  This halves
 * threadgroup Q-tile loads while leaving K/V traffic and MMA counts intact.
 * Select with QWEN38_FLASH_QK_REUSE=1 together with
 * QWEN38_FLASH_PV_REUSE=1. */
kernel void qwen38_prefill_flash_attention_qkreuse_pvreuse(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device const float *query_gate [[buffer(3)]],
    constant PrefillAttentionParams &parameters [[buffer(4)]],
    device float *output [[buffer(5)]],
    device half *x_half [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    /* Shared tiles: all 8 query rows. QK^T and P.V each use one 8x8 MMA per
     * (position tile, dim step), so every simdgroup works on all 8 rows;
     * the work split is over positions (QK^T) and output dims (P.V).
     * Simdgroup g owns output dims {8g + 32k, k = 0..7} for every row. */
    threadgroup half sq[kFlashRows * 256];   /* query rows, pre-scaled 1/16 */
    /* This production path is fixed at block64. Keep only 64 score columns
     * instead of the historical 128-column scratch used by the experimental
     * block128 path. This trims 2 KiB of threadgroup storage. */
    threadgroup float ss[kFlashRows * 64];   /* scores/probs: [row][position] */
    threadgroup float so[kFlashRows * 256];  /* O accumulator: [row][dim] */

    const uint q_head = group_id.x;
    const uint row0 = group_id.y * kFlashRows;
    const uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);

    /* Load the 8 query rows into the shared tile, pre-scaled by 1/16 so the
     * QK^T products come out as final attention scores. Rows past the batch
     * edge load zeros; their results are never stored. */
    for (uint i = tid; i < kFlashRows * 256; i += 128) {
        const uint r = i / 256;
        const uint d = i % 256;
        sq[r * 256 + d] = (row0 + r < parameters.batch)
            ? half(query[(row0 + r) * kPrefillMixerWidth + q_head * 256 + d]
                   * (1.0f / 16.0f))
            : half(0.0f);
    }

    /* Zero O once: simdgroup g owns rows {g, g+4} for softmax state, so
     * let it initialize exactly those two rows instead of having all four
     * simdgroups redundantly zero all eight rows. */
    const uint zero_r0 = sgitg;
    const uint zero_r1 = sgitg + 4;
    for (uint k = 0; k < 8; ++k) {
        so[zero_r0 * 256 + 32 * k + lane] = 0.0f;
        so[zero_r1 * 256 + 32 * k + lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Per-row running softmax state. Explicit scalars (not an indexed
     * array) so the two rows' reductions cannot be vectorized into one:
     * m_a/ssum_a track row sgitg, m_b/ssum_b track row sgitg + 4. */
    float m_a, m_b, ssum_a, ssum_b;
    m_a = -INFINITY;
    m_b = -INFINITY;
    ssum_a = 0.0f;
    ssum_b = 0.0f;

    /* Row r attends to positions [0, start_position + row0 + r). The last
     * block is partial only for the smallest rows of the group. */
    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;

        /* QK^T: the block's 64 positions form 8 tiles of 8; simdgroup g
         * computes tiles {g, g+4}, each an 8x8 MMA over all 256 query dims.
         * mq[row i][dim j] = Q[row i][dim], mk[dim j][pos i] via the
         * transposed load with the 1024-half stride that skips the other
         * KV heads, so mqk[row i][pos j] = score. */
        /* QK reuse variant: each simdgroup owns two position tiles
         * (g and g+4).  Keep both score accumulators live so each 8x8 Q
         * dimension tile is loaded from threadgroup memory once, then
         * multiplied by both K tiles.  K traffic and MMA count are unchanged;
         * Q threadgroup loads are halved.  Accumulation order within each
         * score tile remains i=0..31. */
        simdgroup_float8x8 mqk0 =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_float8x8 mqk1 =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        const uint pos_tile0 = sgitg;
        const uint pos_tile1 = sgitg + 4;
        const device half *pk0 = key_cache +
            ((block_start + 8 * pos_tile0) * kPrefillKVHeads + kv_head) * 256;
        const device half *pk1 = key_cache +
            ((block_start + 8 * pos_tile1) * kPrefillKVHeads + kv_head) * 256;
        for (uint i = 0; i < 32; ++i) {
            simdgroup_half8x8 mq, mk0, mk1;
            simdgroup_load(mq, sq + i * 8, 256);
            simdgroup_load(mk0, pk0 + i * 8, 1024, 0, true);
            simdgroup_multiply_accumulate(mqk0, mq, mk0, mqk0);
            simdgroup_load(mk1, pk1 + i * 8, 1024, 0, true);
            simdgroup_multiply_accumulate(mqk1, mq, mk1, mqk1);
        }
        simdgroup_store(mqk0, ss + 8 * pos_tile0, 64, 0, false);
        simdgroup_store(mqk1, ss + 8 * pos_tile1, 64, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax for this simdgroup's two rows, written with
         * explicit per-row scalars. Lane l owns score columns {2l, 2l+1};
         * the last block masks positions past each row's context length.
         * Row A is sgitg, row B is sgitg + 4. */
        {
            const uint r = sgitg;
            const float old_m = m_a;
            float2 s2 = float2(ss[r * 64 + 2 * lane],
                               ss[r * 64 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_a = simd_max(max(old_m, max(s2[0], s2[1])));
            /* If this block does not raise the running maximum, alpha is
             * exactly 1.  Skip both exp(0) and the 256-dim O rescale. */
            const bool rescale = m_a != old_m;
            const float alpha = rescale ? exp(old_m - m_a) : 1.0f;
            const float2 p2 = exp(s2 - m_a);
            ssum_a = ssum_a * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 64 + 2 * lane] = p2[0];
            ss[r * 64 + 2 * lane + 1] = p2[1];
            if (rescale) {
                for (uint k = 0; k < 8; ++k)
                    so[r * 256 + 32 * k + lane] *= alpha;
            }
        }
        {
            const uint r = sgitg + 4;
            const float old_m = m_b;
            float2 s2 = float2(ss[r * 64 + 2 * lane],
                               ss[r * 64 + 2 * lane + 1]);
            {
                const uint ctx = ctx_lo + sgitg + 4;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m_b = simd_max(max(old_m, max(s2[0], s2[1])));
            const bool rescale = m_b != old_m;
            const float alpha = rescale ? exp(old_m - m_b) : 1.0f;
            const float2 p2 = exp(s2 - m_b);
            ssum_b = ssum_b * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 64 + 2 * lane] = p2[0];
            ss[r * 64 + 2 * lane + 1] = p2[1];
            if (rescale) {
                for (uint k = 0; k < 8; ++k)
                    so[r * 256 + 32 * k + lane] *= alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P.V: O += P.V over the block. All 8 rows' probabilities are valid
         * (each simdgroup softmaxed its own two rows), so one 8x8 MMA per
         * (position, dim) tile covers every row at once. Per 16-position
         * sub-block and per 8-position half k: lo[row i][dim j] += sum
         * over the half's 8 positions of P[row i][pos] * V[pos][dim]. The
         * probability tile must be loaded per half (vs holds only 8
         * columns): vs[row i][pos c] = P[row i][pos 16cc + 8k + c], and V
         * is loaded NOT transposed (the position axis is already V's first
         * memory axis) so mv[pos c][dim j] = V[pos 16cc + 8k + c][db + j];
         * the MMA then contracts the position index exactly. */
        /* P.V accumulator-reuse variant: keep each 8x8 output tile in the
         * simdgroup matrix across the whole 64-position block.  The original
         * loop order loads/stores `lo` once per 16 positions (4x per block).
         * Swapping ii/cc preserves the eight MMA updates in the same order,
         * but performs only one threadgroup load and one store per output
         * tile per 64 positions.  K/V traffic and arithmetic are unchanged. */
        for (uint ii = 0; ii < 8; ++ii) {
            const uint db = 8 * sgitg + 32 * ii;
            simdgroup_float8x8 lo;
            simdgroup_load(lo, so + db, 256);
            for (uint cc = 0; cc < kFlashBlock / 16; ++cc) {
                for (uint k = 0; k < 2; ++k) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 16 * cc + 8 * k, 64);
                    const device half *pv = value_cache +
                        ((block_start + 16 * cc + 8 * k) * kPrefillKVHeads +
                         kv_head) * 256;
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + db, 1024, 0, false);
                    simdgroup_multiply_accumulate(lo, vs, mv, lo);
                }
            }
            simdgroup_store(lo, so + db, 256);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Normalize and apply the sigmoid query gate. Every simdgroup has seen
     * every position of its two rows, so ssum is complete; all four
     * simdgroups' O chunks are valid behind the last barrier. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (jj == 0 ? ssum_a : ssum_b) > 0.0f
            ? 1.0f / (jj == 0 ? ssum_a : ssum_b) : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const uint d = lane + 32 * i;
            const uint output_index =
                (row0 + r) * kPrefillMixerWidth + q_head * 256 + d;
            const float gated_value = so[r * 256 + d] * scale *
                                      query_gate[output_index];
            output[output_index] = gated_value;
            if (x_half != nullptr)
                x_half[output_index] = half(gated_value);
        }
    }
}




/* 128-position FP16 flash-attention prefill variant.
 *
 * This keeps the same 8-query-row / 4-simdgroup layout as the 64-position
 * kernel, but consumes 128 KV positions per outer iteration.  The score
 * scratch was already 128 columns wide; the important difference is that
 * the online-softmax reduction must cover all 128 scores at once.  Each
 * SIMD lane therefore owns four columns {2l, 2l+1, 64+2l, 64+2l+1}.
 *
 * The goal is to reduce outer-loop/barrier/rescale cadence at long context
 * without changing KV precision or adding dequantization work.  Select with
 * QWEN38_FLASH_BLOCK=128; the original 64-position kernel remains the
 * default/control path. */
constant uint kFlashBlock128 = 128;

kernel void qwen38_prefill_flash_attention_128(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device const float *query_gate [[buffer(3)]],
    constant PrefillAttentionParams &parameters [[buffer(4)]],
    device float *output [[buffer(5)]],
    device half *x_half [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup half sq[kFlashRows * 256];
    threadgroup float ss[kFlashRows * kFlashBlock128];
    threadgroup float so[kFlashRows * 256];

    const uint q_head = group_id.x;
    const uint row0 = group_id.y * kFlashRows;
    const uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);

    for (uint i = tid; i < kFlashRows * 256; i += 128) {
        const uint r = i / 256;
        const uint d = i % 256;
        sq[r * 256 + d] = (row0 + r < parameters.batch)
            ? half(query[(row0 + r) * kPrefillMixerWidth + q_head * 256 + d]
                   * (1.0f / 16.0f))
            : half(0.0f);
    }

    for (uint r = 0; r < kFlashRows; ++r) {
        for (uint k = 0; k < 8; ++k)
            so[r * 256 + 32 * k + lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_a = -INFINITY;
    float m_b = -INFINITY;
    float ssum_a = 0.0f;
    float ssum_b = 0.0f;

    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock128 - 1) / kFlashBlock128;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock128;

        /* 128 positions = 16 x 8-position tiles. Four simdgroups compute
         * four tiles each: g, g+4, g+8, g+12. */
        for (uint cc = 0; cc < kFlashBlock128 / 8 / 4; ++cc) {
            simdgroup_float8x8 mqk =
                make_filled_simdgroup_matrix<float, 8>(0.0f);
            const uint pos_tile = sgitg + 4 * cc;
            const device half *pk = key_cache +
                ((block_start + 8 * pos_tile) * kPrefillKVHeads + kv_head) * 256;
            for (uint i = 0; i < 32; ++i) {
                simdgroup_half8x8 mq, mk;
                simdgroup_load(mq, sq + i * 8, 256);
                simdgroup_load(mk, pk + i * 8, 1024, 0, true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            simdgroup_store(mqk, ss + 8 * pos_tile, kFlashBlock128,
                            0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* One online-softmax update over all 128 positions.  Each lane owns
         * two columns in each 64-position half, so the SIMD reduction sees
         * all scores before m/ssum are updated. */
        {
            const uint r = sgitg;
            const float old_m = m_a;
            float4 s4 = float4(
                ss[r * kFlashBlock128 + 2 * lane],
                ss[r * kFlashBlock128 + 2 * lane + 1],
                ss[r * kFlashBlock128 + 64 + 2 * lane],
                ss[r * kFlashBlock128 + 64 + 2 * lane + 1]);
            const uint ctx = ctx_lo + sgitg;
            if (block_start + 2 * lane >= ctx) s4[0] = -INFINITY;
            if (block_start + 2 * lane + 1 >= ctx) s4[1] = -INFINITY;
            if (block_start + 64 + 2 * lane >= ctx) s4[2] = -INFINITY;
            if (block_start + 64 + 2 * lane + 1 >= ctx) s4[3] = -INFINITY;
            const float lane_max = max(max(s4[0], s4[1]),
                                       max(s4[2], s4[3]));
            m_a = simd_max(max(old_m, lane_max));
            const float alpha = exp(old_m - m_a);
            const float4 p4 = exp(s4 - m_a);
            ssum_a = ssum_a * alpha +
                simd_sum(p4[0] + p4[1] + p4[2] + p4[3]);
            ss[r * kFlashBlock128 + 2 * lane] = p4[0];
            ss[r * kFlashBlock128 + 2 * lane + 1] = p4[1];
            ss[r * kFlashBlock128 + 64 + 2 * lane] = p4[2];
            ss[r * kFlashBlock128 + 64 + 2 * lane + 1] = p4[3];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        {
            const uint r = sgitg + 4;
            const float old_m = m_b;
            float4 s4 = float4(
                ss[r * kFlashBlock128 + 2 * lane],
                ss[r * kFlashBlock128 + 2 * lane + 1],
                ss[r * kFlashBlock128 + 64 + 2 * lane],
                ss[r * kFlashBlock128 + 64 + 2 * lane + 1]);
            const uint ctx = ctx_lo + sgitg + 4;
            if (block_start + 2 * lane >= ctx) s4[0] = -INFINITY;
            if (block_start + 2 * lane + 1 >= ctx) s4[1] = -INFINITY;
            if (block_start + 64 + 2 * lane >= ctx) s4[2] = -INFINITY;
            if (block_start + 64 + 2 * lane + 1 >= ctx) s4[3] = -INFINITY;
            const float lane_max = max(max(s4[0], s4[1]),
                                       max(s4[2], s4[3]));
            m_b = simd_max(max(old_m, lane_max));
            const float alpha = exp(old_m - m_b);
            const float4 p4 = exp(s4 - m_b);
            ssum_b = ssum_b * alpha +
                simd_sum(p4[0] + p4[1] + p4[2] + p4[3]);
            ss[r * kFlashBlock128 + 2 * lane] = p4[0];
            ss[r * kFlashBlock128 + 2 * lane + 1] = p4[1];
            ss[r * kFlashBlock128 + 64 + 2 * lane] = p4[2];
            ss[r * kFlashBlock128 + 64 + 2 * lane + 1] = p4[3];
            for (uint k = 0; k < 8; ++k)
                so[r * 256 + 32 * k + lane] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P.V over eight 16-position sub-blocks. */
        for (uint cc = 0; cc < kFlashBlock128 / 16; ++cc) {
            for (uint ii = 0; ii < 8; ++ii) {
                const uint db = 8 * sgitg + 32 * ii;
                simdgroup_float8x8 lo;
                simdgroup_load(lo, so + db, 256);
                for (uint k = 0; k < 2; ++k) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 16 * cc + 8 * k,
                                   kFlashBlock128);
                    const device half *pv = value_cache +
                        ((block_start + 16 * cc + 8 * k) *
                             kPrefillKVHeads + kv_head) * 256;
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + db, 1024, 0, false);
                    simdgroup_multiply_accumulate(lo, vs, mv, lo);
                }
                simdgroup_store(lo, so + db, 256);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float sum = jj == 0 ? ssum_a : ssum_b;
        const float scale = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const uint d = lane + 32 * i;
            const uint output_index =
                (row0 + r) * kPrefillMixerWidth + q_head * 256 + d;
            const float gated_value = so[r * 256 + d] * scale *
                                      query_gate[output_index];
            output[output_index] = gated_value;
            if (x_half != nullptr)
                x_half[output_index] = half(gated_value);
        }
    }
}



/* Q8_0 KV variant of the flash kernel. The int8 codes cannot feed the 8x8
 * half matrix units directly, so each 16-position sub-block's K and V are
 * dequantized into shared tiles (8 KB each) first; QK^T, the online softmax
 * and P.V then run with exactly the same 8x8 simdgroup multiply-accumulate
 * structure as the fp16 kernel, with the per-vector scales folded into the
 * staged codes. Static threadgroup memory holds the query tile (4 KB), the
 * score scratch (4 KB) and the O accumulator (8 KB); the K/V tiles (8 KB
 * each) live in a dynamic [[threadgroup(0)]] buffer sized by the host.
 * Total per-threadgroup storage is 32 KB. */

kernel void qwen38_prefill_flash_attention_q8(
    device const float *query [[buffer(0)]],
    device const char *key_cache [[buffer(1)]],
    device const char *value_cache [[buffer(2)]],
    device const float *query_gate [[buffer(3)]],
    constant PrefillAttentionParams &parameters [[buffer(4)]],
    device float *output [[buffer(5)]],
    device half *x_half [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]],
    threadgroup half *kv_tiles [[threadgroup(0)]]) {
    /* Shared tiles: all 8 query rows (same structure as the fp16 kernel;
     * every simdgroup works on all 8 rows, split over positions for QK^T
     * and output dims for P.V). The int8 codes cannot feed the matrix units
     * directly, so each 16-position sub-block's K and V are dequantized into
     * shared tiles (8 KB each) first. */
    threadgroup half sq[kFlashRows * 256];   /* query rows, pre-scaled 1/16 */
    threadgroup float ss[kFlashRows * 128];  /* scores/probs: [row][position] */
    threadgroup float so[kFlashRows * 256];  /* O accumulator: [row][dim] */
    threadgroup half *sk = kv_tiles;
    threadgroup half *sv = kv_tiles + 16 * 256;

    const uint q_head = group_id.x;
    const uint row0 = group_id.y * kFlashRows;
    const uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);

    for (uint i = tid; i < kFlashRows * 256; i += 128) {
        const uint r = i / 256;
        const uint d = i % 256;
        sq[r * 256 + d] = (row0 + r < parameters.batch)
            ? half(query[(row0 + r) * kPrefillMixerWidth + q_head * 256 + d]
                   * (1.0f / 16.0f))
            : half(0.0f);
    }

    /* Zero the O accumulator (same redundant all-dims form as the fp16
     * kernel: each thread covers one element of every 32-dim window). */
    for (uint r = 0; r < kFlashRows; ++r) {
        for (uint k = 0; k < 8; ++k)
            so[r * 256 + 32 * k + lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Per-row running softmax state, explicit scalars (see the fp16
     * kernel for why an indexed array is unsafe here). m_a/ssum_a track
     * row sgitg, m_b/ssum_b track row sgitg + 4. */
    float m_a, m_b, ssum_a, ssum_b;
    m_a = -INFINITY;
    m_b = -INFINITY;
    ssum_a = 0.0f;
    ssum_b = 0.0f;

    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;

        for (uint t = 0; t < kFlashBlock / 16; ++t) {
            const uint sub_start = block_start + 16 * t;

            /* Dequantize this sub-block's K and V: 128 threads cover the
             * 16 positions x 256 dims; each thread takes one position (the
             * 8 threads of a warp quad share it) and a contiguous 32-dim
             * chunk. Positions past the group's largest row context are
             * zeroed so the masked scores stay finite. */
            for (uint p = tid / 8; p < 16; p += 16) {
                const uint position = sub_start + p;
                const bool live = position < ctx_lo + kFlashRows - 1;
                const uint c0 = 32 * (tid % 8);
                if (live) {
                    const device char *kv = key_cache +
                        ((position * kPrefillKVHeads + kv_head) *
                         kPrefillKVQ8Stride);
                    const float ks = *(const device float *)(kv + 256);
                    for (uint d = c0; d < c0 + 32; ++d)
                        sk[p * 256 + d] = half((float)(char)kv[d] * ks);
                    const device char *vv = value_cache +
                        ((position * kPrefillKVHeads + kv_head) *
                         kPrefillKVQ8Stride);
                    const float vs = *(const device float *)(vv + 256);
                    for (uint d = c0; d < c0 + 32; ++d)
                        sv[p * 256 + d] = half((float)(char)vv[d] * vs);
                } else {
                    for (uint d = c0; d < c0 + 32; ++d) {
                        sk[p * 256 + d] = half(0.0f);
                        sv[p * 256 + d] = half(0.0f);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* QK^T for the 16 positions: every simdgroup computes both
             * 8-position tiles (cc = 0,1) of the sub-block so that all rows'
             * scores for the whole sub-block are ready before P.V. */
            for (uint cc = 0; cc < 2; ++cc) {
                /* Tile cc covers positions 8cc..8cc+7 of the sub-block:
                 * the K base must advance with cc as well as with the dim
                 * loop i. mk[r][c] = K_tile[pos 8cc + c][dim 8i + r], so
                 * the MMA inner product pairs Q[row][dim 8i+j] with
                 * K[pos 8cc + c][dim 8i+j]. (Without the 8*cc term both
                 * tiles score positions 0..7 and the second half of every
                 * 16-position sub-block is garbage.) */
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (uint i = 0; i < 32; ++i) {
                    simdgroup_half8x8 mq, mk;
                    simdgroup_load(mq, sq + i * 8, 256);
                    simdgroup_load(mk, sk + 8 * cc * 256 + i * 8, 256, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }
                simdgroup_store(mqk, ss + 16 * t + 8 * cc, 128, 0, false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* Online softmax for this simdgroup's two rows over the 16
             * positions. Masking is applied to every sub-block (not just
             * the last): in a partial final block the earlier sub-blocks
             * still contain positions past the shorter rows' contexts. */
            {
                const uint r = sgitg;
                const float old_m = m_a;
                float2 s2 = float2(ss[r * 128 + 16 * t + 2 * lane],
                                   ss[r * 128 + 16 * t + 2 * lane + 1]);
                {
                    const uint ctx = ctx_lo + sgitg;
                    if (2 * lane >= 16 || sub_start + 2 * lane >= ctx)
                        s2[0] = -INFINITY;
                    if (2 * lane + 1 >= 16 || sub_start + 2 * lane + 1 >= ctx)
                        s2[1] = -INFINITY;
                }
                m_a = simd_max(max(old_m, max(s2[0], s2[1])));
                const float alpha = exp(old_m - m_a);
                const float2 p2 = exp(s2 - m_a);
                ssum_a = ssum_a * alpha + simd_sum(p2[0] + p2[1]);
                ss[r * 128 + 16 * t + 2 * lane] = p2[0];
                ss[r * 128 + 16 * t + 2 * lane + 1] = p2[1];
                for (uint k = 0; k < 8; ++k)
                    so[r * 256 + 32 * k + lane] *= alpha;
            }
            {
                const uint r = sgitg + 4;
                const float old_m = m_b;
                float2 s2 = float2(ss[r * 128 + 16 * t + 2 * lane],
                                   ss[r * 128 + 16 * t + 2 * lane + 1]);
                {
                    const uint ctx = ctx_lo + sgitg + 4;
                    if (2 * lane >= 16 || sub_start + 2 * lane >= ctx)
                        s2[0] = -INFINITY;
                    if (2 * lane + 1 >= 16 || sub_start + 2 * lane + 1 >= ctx)
                        s2[1] = -INFINITY;
                }
                m_b = simd_max(max(old_m, max(s2[0], s2[1])));
                const float alpha = exp(old_m - m_b);
                const float2 p2 = exp(s2 - m_b);
                ssum_b = ssum_b * alpha + simd_sum(p2[0] + p2[1]);
                ss[r * 128 + 16 * t + 2 * lane] = p2[0];
                ss[r * 128 + 16 * t + 2 * lane + 1] = p2[1];
                for (uint k = 0; k < 8; ++k)
                    so[r * 256 + 32 * k + lane] *= alpha;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* P.V over the 16 positions: all 8 rows' probabilities are
             * valid, so one 8x8 MMA per (position, dim) tile covers every
             * row. Same structure as the fp16 kernel: the probability tile
             * is loaded per 8-position half (it holds only 8 columns), and
             * V comes from the dequantized shared tile (256-half position
             * stride) NOT transposed, so mv[pos c][dim j] =
             * V[pos 16t + 8k + c][dim db + j] and the MMA contracts the
             * position index exactly. */
            {
                for (uint ii = 0; ii < 8; ++ii) {
                    const uint db = 8 * sgitg + 32 * ii;
                    simdgroup_float8x8 lo;
                    simdgroup_load(lo, so + db, 256);
                    for (uint k = 0; k < 2; ++k) {
                        simdgroup_float8x8 vs;
                        simdgroup_load(vs, ss + 16 * t + 8 * k, 128);
                        const threadgroup half *pv = sv +
                            (8 * k) * 256 + db;
                        simdgroup_half8x8 mv;
                        simdgroup_load(mv, pv, 256, 0, false);
                        simdgroup_multiply_accumulate(lo, vs, mv, lo);
                    }
                    simdgroup_store(lo, so + db, 256);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    /* Normalize and apply the sigmoid query gate. Every simdgroup has seen
     * every position of its two rows, so ssum is complete; all four
     * simdgroups' O chunks are valid behind the last barrier. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (jj == 0 ? ssum_a : ssum_b) > 0.0f
            ? 1.0f / (jj == 0 ? ssum_a : ssum_b) : 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const uint d = lane + 32 * i;
            const uint output_index =
                (row0 + r) * kPrefillMixerWidth + q_head * 256 + d;
            const float gated_value = so[r * 256 + d] * scale *
                                      query_gate[output_index];
            output[output_index] = gated_value;
            if (x_half != nullptr)
                x_half[output_index] = half(gated_value);
        }
    }
}




/* Tiled simdgroup-matrix GEMM path. The first-generation batched GEMM above
 * keeps decode-identical arithmetic per element but re-reads every batch
 * activation row from device memory once per weight group per simdgroup,
 * which multiplies activation traffic by the batch size. This path stages a
 * [batch x 64] activation tile and a dequantized [64 x 32] weight tile in
 * threadgroup memory once per threadgroup and consumes them with 8x8
 * simdgroup matrix multiply-accumulates, the standard bandwidth shape for
 * Apple-GPU prefill. Accumulation order differs from the one-token kernel,
 * so this path is gated by the argmax/token-parity standard, not bitwise. */

constant uint kGemmTileRows = 32;
constant uint kGemmTileK = 64;
constant uint kGemmTileBatch = 32;

#define QWEN38_PREFILL_GEMM_MMA_BODY(X_LOAD, STORE)                       \
    threadgroup float x_tile[kGemmTileBatch * kGemmTileK];                \
    threadgroup float w_tile[kGemmTileK * kGemmTileRows];                 \
    threadgroup float c_tile[kGemmTileBatch * kGemmTileRows];             \
    uint row0 = group_id.x * kGemmTileRows;                               \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_float8x8 accumulator[4];                                    \
    for (uint n = 0; n < 4; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    uint b0 = simdgroup_index * 8;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        for (uint i = 0; i < 16; ++i) {                                   \
            uint linear = tid * 16 + i;                                   \
            uint b = linear >> 6;                                         \
            uint k = linear & 63u;                                        \
            x_tile[linear] = b < kBatch ?                                 \
                X_LOAD(b * columns + group * 64 + k) : 0.0f;              \
        }                                                                 \
        uint r = tid & 31u;                                               \
        uint k_base = (tid >> 5) * 16;                                    \
        uint block = (row0 + r) * p.groups_per_row + group;               \
        Q4PrefillMeta meta = metadata[block];                             \
        float scale = float(meta.scale);                                  \
        float bias = float(meta.bias);                                    \
        for (uint i = 0; i < 16; i += 2) {                                \
            uchar bits = quants[block * 32 + ((k_base + i) >> 1)];        \
            w_tile[(k_base + i) * kGemmTileRows + r] =                    \
                scale * float(bits & 0x0f) + bias;                        \
            w_tile[(k_base + i + 1) * kGemmTileRows + r] =                \
                scale * float(bits >> 4) + bias;                          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_float8x8 a;                                         \
            simdgroup_load(a, x_tile + b0 * kGemmTileK + kk,              \
                           kGemmTileK);                                   \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_float8x8 b_fragment;                            \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmTileRows + n * 8,       \
                               kGemmTileRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
    }                                                                     \
    for (uint n = 0; n < 4; ++n)                                          \
        simdgroup_store(accumulator[n],                                   \
                        c_tile + b0 * kGemmTileRows + n * 8,              \
                        kGemmTileRows);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    for (uint i = 0; i < 8; ++i) {                                        \
        uint linear = tid * 8 + i;                                        \
        uint b = linear >> 5;                                             \
        uint r = linear & 31u;                                            \
        if (b < kBatch) {                                                 \
            uint out_index = b * p.rows + row0 + r;                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q4_gemm_f16_mma(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_HALF(index) float(x[index])
#define STORE_PLAIN output[out_index] = c_tile[linear]
    QWEN38_PREFILL_GEMM_MMA_BODY(X_LOAD_HALF, STORE_PLAIN)
#undef X_LOAD_HALF
#undef STORE_PLAIN
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16_mma(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_FLOAT(index) x[index]
#define STORE_RESIDUAL_HALF \
    output[out_index] = c_tile[linear] + float(residual[out_index])
    QWEN38_PREFILL_GEMM_MMA_BODY(X_LOAD_FLOAT, STORE_RESIDUAL_HALF)
#undef X_LOAD_FLOAT
#undef STORE_RESIDUAL_HALF
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32_mma(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_FLOAT(index) x[index]
#define STORE_RESIDUAL_FLOAT \
    output[out_index] = c_tile[linear] + residual[out_index]
    QWEN38_PREFILL_GEMM_MMA_BODY(X_LOAD_FLOAT, STORE_RESIDUAL_FLOAT)
#undef X_LOAD_FLOAT
#undef STORE_RESIDUAL_FLOAT
}

/* Half-precision MMA path. The float MMA above is FP32-ALU-bound on M3
 * (measured ~13-14 ms per layer for a 32-token chunk against a ~1.8 ms
 * weight-streaming floor), so the 2x-rate half pipes are the remaining
 * lever. Tiles are staged in half and consumed with half 8x8 MMAs; the
 * half accumulators spill into per-thread float accumulators every four
 * K-groups (256 columns), which bounds the half-precision accumulation
 * window. Gated by the argmax/token-parity standard like the float MMA
 * path; QWEN38_PREFILL_MMA=1 restores the float MMA, =0 the exact path. */

constant uint kGemmSpillGroups = 1;

/* All three variants read half activations directly from device memory
 * with strided simdgroup loads (no activation staging), so float
 * activations are converted once into a half scratch first. Activation
 * rows at batch indices >= kBatch hold stale data; their products stay
 * inside accumulator rows that the guarded store discards. */
kernel void qwen38_prefill_convert_x(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= columns || s >= kBatch) return;
    output[s * columns + index] = half(input[s * columns + index]);
}

#define QWEN38_PREFILL_GEMM_MMA2_BODY(STORE)                              \
    threadgroup half w_tile[kGemmTileK * kGemmTileRows];                  \
    threadgroup half spill[kGemmTileBatch * kGemmTileRows];               \
    uint row0 = group_id.x * kGemmTileRows;                               \
    uint batch0 = group_id.y * kGemmTileBatch;                            \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[4];                                     \
    for (uint n = 0; n < 4; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[8];                                                       \
    for (uint i = 0; i < 8; ++i) c_acc[i] = 0.0f;                         \
    uint b0 = batch0 + simdgroup_index * 8;                               \
    uint spill0 = simdgroup_index * 8;                                    \
    uint r = tid & 31u;                                                   \
    uint k_base = (tid >> 5) * 16;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        uint block = (row0 + r) * p.groups_per_row + group;               \
        Q4PrefillMeta meta = metadata[block];                             \
        half scale = meta.scale;                                          \
        half bias = meta.bias;                                            \
        device const uint *words = (device const uint *)                  \
            (quants + block * 32 + (k_base >> 1));                        \
        for (uint word = 0; word < 2; ++word) {                           \
            uint bits = words[word];                                      \
            half4 lo = half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) *       \
                       scale + bias;                                      \
            half4 hi = half4(as_type<uchar4>((bits >> 4) &                \
                                             0x0f0f0f0fu)) *              \
                       scale + bias;                                      \
            uint base = (k_base + word * 8) * kGemmTileRows + r;          \
            w_tile[base] = lo.x;                                          \
            w_tile[base + kGemmTileRows] = hi.x;                          \
            w_tile[base + 2 * kGemmTileRows] = lo.y;                      \
            w_tile[base + 3 * kGemmTileRows] = hi.y;                      \
            w_tile[base + 4 * kGemmTileRows] = lo.z;                      \
            w_tile[base + 5 * kGemmTileRows] = hi.z;                      \
            w_tile[base + 6 * kGemmTileRows] = lo.w;                      \
            w_tile[base + 7 * kGemmTileRows] = hi.w;                      \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + b0 * columns + group * 64 + kk,         \
                           columns);                                      \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmTileRows + n * 8,       \
                               kGemmTileRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        if ((group & (kGemmSpillGroups - 1)) == kGemmSpillGroups - 1 ||   \
            group == p.groups_per_row - 1) {                              \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill + spill0 * kGemmTileRows + n * 8,   \
                                kGemmTileRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            simdgroup_barrier(mem_flags::mem_threadgroup);                \
            for (uint i = 0; i < 8; ++i)                                  \
                c_acc[i] += float(spill[tid * 8 + i]);                    \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 8; ++i) {                                        \
        uint linear = tid * 8 + i;                                        \
        uint b = batch0 + (linear >> 5);                                  \
        uint out_row = linear & 31u;                                      \
        if (b < kBatch) {                                                 \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q4_gemm_f16_mma2(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN2 output[out_index] = value
    QWEN38_PREFILL_GEMM_MMA2_BODY(STORE_PLAIN2)
#undef STORE_PLAIN2
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16_mma2(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_HALF2 \
    output[out_index] = value + float(residual[out_index])
    QWEN38_PREFILL_GEMM_MMA2_BODY(STORE_RESIDUAL_HALF2)
#undef STORE_RESIDUAL_HALF2
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32_mma2(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_FLOAT2 \
    output[out_index] = value + residual[out_index]
    QWEN38_PREFILL_GEMM_MMA2_BODY(STORE_RESIDUAL_FLOAT2)
#undef STORE_RESIDUAL_FLOAT2
}

/* Q8 half MMA: the exact shape of the Q4 mma2 above (32 output rows,
 * 32 batch rows, four 8x8 accumulators per simdgroup), for the
 * QWEN38_PREFILL_MMA=2 level. The code plane holds one signed int8 per
 * weight — 64 bytes per block — so the dequant loop reads each lane's 16
 * consecutive codes as four uint words and sign-extends them through
 * char4 before scaling. */

#define QWEN38_PREFILL_GEMM_Q8_MMA2_BODY(STORE)                           \
    threadgroup half w_tile[kGemmTileK * kGemmTileRows];                  \
    threadgroup half spill[kGemmTileBatch * kGemmTileRows];               \
    uint row0 = group_id.x * kGemmTileRows;                               \
    uint batch0 = group_id.y * kGemmTileBatch;                            \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[4];                                     \
    for (uint n = 0; n < 4; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[8];                                                       \
    for (uint i = 0; i < 8; ++i) c_acc[i] = 0.0f;                         \
    uint b0 = batch0 + simdgroup_index * 8;                               \
    uint spill0 = simdgroup_index * 8;                                    \
    uint r = tid & 31u;                                                   \
    uint k_base = (tid >> 5) * 16;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        uint block = (row0 + r) * p.groups_per_row + group;               \
        Q4PrefillMeta meta = metadata[block];                             \
        half scale = meta.scale;                                          \
        half bias = meta.bias;                                            \
        device const uint *words = (device const uint *)                  \
            (quants + block * 64 + k_base);                               \
        for (uint word = 0; word < 4; ++word) {                           \
            half4 w = half4(as_type<char4>(words[word])) *                \
                       scale + bias;                                      \
            uint base = (k_base + word * 4) * kGemmTileRows + r;          \
            w_tile[base] = w.x;                                           \
            w_tile[base + kGemmTileRows] = w.y;                           \
            w_tile[base + 2 * kGemmTileRows] = w.z;                       \
            w_tile[base + 3 * kGemmTileRows] = w.w;                       \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + b0 * columns + group * 64 + kk,         \
                           columns);                                      \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmTileRows + n * 8,       \
                               kGemmTileRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        if ((group & (kGemmSpillGroups - 1)) == kGemmSpillGroups - 1 ||   \
            group == p.groups_per_row - 1) {                              \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill + spill0 * kGemmTileRows + n * 8,   \
                                kGemmTileRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            simdgroup_barrier(mem_flags::mem_threadgroup);                \
            for (uint i = 0; i < 8; ++i)                                  \
                c_acc[i] += float(spill[tid * 8 + i]);                    \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 8; ++i) {                                        \
        uint linear = tid * 8 + i;                                        \
        uint b = batch0 + (linear >> 5);                                  \
        uint out_row = linear & 31u;                                      \
        if (b < kBatch) {                                                 \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q8_gemm_f16_mma2(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN_Q8_2 output[out_index] = value
    QWEN38_PREFILL_GEMM_Q8_MMA2_BODY(STORE_PLAIN_Q8_2)
#undef STORE_PLAIN_Q8_2
}

/* Small-batch half MMA: the speculative verify runs at batch 2-8, where
 * the 32-wide batch tile above pays for four times the useful math. This
 * variant keeps the same cooperative weight staging and per-group float
 * spill, but its batch tile is eight rows and the four simdgroups split
 * the 64-column K-tile instead of the batch, so their partial tiles are
 * summed through the spill buffer. Activation rows at batch indices >=
 * kBatch hold stale data; their products stay inside accumulator rows
 * that the guarded store discards. Gated by the argmax/token-parity
 * standard like the other MMA paths. */

#define QWEN38_PREFILL_GEMM_MMA8_BODY(STORE)                              \
    threadgroup half w_tile[kGemmTileK * kGemmTileRows];                  \
    threadgroup half spill[4 * 8 * kGemmTileRows];                        \
    uint row0 = group_id.x * kGemmTileRows;                               \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[4];                                     \
    for (uint n = 0; n < 4; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[2];                                                       \
    c_acc[0] = 0.0f;                                                      \
    c_acc[1] = 0.0f;                                                      \
    uint kk0 = simdgroup_index * 16;                                      \
    uint r = tid & 31u;                                                   \
    uint k_base = (tid >> 5) * 16;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        uint block = (row0 + r) * p.groups_per_row + group;               \
        Q4PrefillMeta meta = metadata[block];                             \
        half scale = meta.scale;                                          \
        half bias = meta.bias;                                            \
        device const uint *words = (device const uint *)                  \
            (quants + block * 32 + (k_base >> 1));                        \
        for (uint word = 0; word < 2; ++word) {                           \
            uint bits = words[word];                                      \
            half4 lo = half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) *       \
                       scale + bias;                                      \
            half4 hi = half4(as_type<uchar4>((bits >> 4) &                \
                                             0x0f0f0f0fu)) *              \
                       scale + bias;                                      \
            uint base = (k_base + word * 8) * kGemmTileRows + r;          \
            w_tile[base] = lo.x;                                          \
            w_tile[base + kGemmTileRows] = hi.x;                          \
            w_tile[base + 2 * kGemmTileRows] = lo.y;                      \
            w_tile[base + 3 * kGemmTileRows] = hi.y;                      \
            w_tile[base + 4 * kGemmTileRows] = lo.z;                      \
            w_tile[base + 5 * kGemmTileRows] = hi.z;                      \
            w_tile[base + 6 * kGemmTileRows] = lo.w;                      \
            w_tile[base + 7 * kGemmTileRows] = hi.w;                      \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = kk0; kk < kk0 + 16; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + group * 64 + kk, columns);              \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmTileRows + n * 8,       \
                               kGemmTileRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        if ((group & 3u) == 3u || group == p.groups_per_row - 1) {        \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill +                                   \
                                    simdgroup_index * 8 * kGemmTileRows + \
                                    n * 8,                                \
                                kGemmTileRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
            for (uint i = 0; i < 2; ++i) {                                \
                uint linear = tid * 2 + i;                                \
                float sum = 0.0f;                                         \
                for (uint sg = 0; sg < 4; ++sg)                           \
                    sum += float(spill[sg * 8 * kGemmTileRows + linear]); \
                c_acc[i] += sum;                                          \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 2; ++i) {                                        \
        uint linear = tid * 2 + i;                                        \
        uint b = linear >> 5;                                             \
        uint out_row = linear & 31u;                                      \
        if (b < kBatch) {                                                 \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q4_gemm_f16_mma8(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN8 output[out_index] = value
    QWEN38_PREFILL_GEMM_MMA8_BODY(STORE_PLAIN8)
#undef STORE_PLAIN8
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16_mma8(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_HALF8 \
    output[out_index] = value + float(residual[out_index])
    QWEN38_PREFILL_GEMM_MMA8_BODY(STORE_RESIDUAL_HALF8)
#undef STORE_RESIDUAL_HALF8
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32_mma8(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_FLOAT8 \
    output[out_index] = value + residual[out_index]
    QWEN38_PREFILL_GEMM_MMA8_BODY(STORE_RESIDUAL_FLOAT8)
#undef STORE_RESIDUAL_FLOAT8
}

constant uint kGemmWideRows = 64;

/* Wide-tile variant of the small-batch MMA: the same eight-row batch
 * tile and K-split simdgroups, but each threadgroup covers 64 output
 * rows, which halves the activation-fragment loads and barrier rounds
 * per unit of math. Partial tiles are summed through the spill buffer
 * every fourth weight group (a 64-column half-accumulation window per
 * simdgroup, matching the other MMA paths). */

#define QWEN38_PREFILL_GEMM_MMA8W_BODY(STORE)                             \
    threadgroup half w_tile[kGemmTileK * kGemmWideRows];                  \
    threadgroup half spill[4 * 8 * kGemmWideRows];                        \
    uint row0 = group_id.x * kGemmWideRows;                               \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[8];                                     \
    for (uint n = 0; n < 8; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[4];                                                       \
    for (uint i = 0; i < 4; ++i) c_acc[i] = 0.0f;                         \
    uint kk0 = simdgroup_index * 16;                                      \
    uint r = tid & 63u;                                                   \
    uint k_base = (tid >> 6) * 32;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        if (row0 + r < p.rows) {                                          \
            uint block = (row0 + r) * p.groups_per_row + group;           \
            Q4PrefillMeta meta = metadata[block];                         \
            half scale = meta.scale;                                      \
            half bias = meta.bias;                                        \
            device const uint *words = (device const uint *)              \
                (quants + block * 32 + (k_base >> 1));                    \
            for (uint word = 0; word < 4; ++word) {                       \
                uint bits = words[word];                                  \
                half4 lo = half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) *   \
                           scale + bias;                                  \
                half4 hi = half4(as_type<uchar4>((bits >> 4) &            \
                                                 0x0f0f0f0fu)) *          \
                           scale + bias;                                  \
                uint base = (k_base + word * 8) * kGemmWideRows + r;      \
                w_tile[base] = lo.x;                                      \
                w_tile[base + kGemmWideRows] = hi.x;                      \
                w_tile[base + 2 * kGemmWideRows] = lo.y;                  \
                w_tile[base + 3 * kGemmWideRows] = hi.y;                  \
                w_tile[base + 4 * kGemmWideRows] = lo.z;                  \
                w_tile[base + 5 * kGemmWideRows] = hi.z;                  \
                w_tile[base + 6 * kGemmWideRows] = lo.w;                  \
                w_tile[base + 7 * kGemmWideRows] = hi.w;                  \
            }                                                             \
        } else {                                                          \
            for (uint i = 0; i < 32; ++i)                                 \
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = kk0; kk < kk0 + 16; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + group * 64 + kk, columns);              \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmWideRows + n * 8,       \
                               kGemmWideRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        if ((group & 3u) == 3u || group == p.groups_per_row - 1) {        \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill +                                   \
                                    simdgroup_index * 8 * kGemmWideRows + \
                                    n * 8,                                \
                                kGemmWideRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
            for (uint i = 0; i < 4; ++i) {                                \
                uint linear = tid * 4 + i;                                \
                float sum = 0.0f;                                         \
                for (uint sg = 0; sg < 4; ++sg)                           \
                    sum += float(spill[sg * 8 * kGemmWideRows + linear]); \
                c_acc[i] += sum;                                          \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 4; ++i) {                                        \
        uint linear = tid * 4 + i;                                        \
        uint b = linear >> 6;                                             \
        uint out_row = linear & 63u;                                      \
        if (b < kBatch && row0 + out_row < p.rows) {                      \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q4_gemm_f16_mma8w(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN8W output[out_index] = value
    QWEN38_PREFILL_GEMM_MMA8W_BODY(STORE_PLAIN8W)
#undef STORE_PLAIN8W
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16_mma8w(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_HALF8W \
    output[out_index] = value + float(residual[out_index])
    QWEN38_PREFILL_GEMM_MMA8W_BODY(STORE_RESIDUAL_HALF8W)
#undef STORE_RESIDUAL_HALF8W
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32_mma8w(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_FLOAT8W \
    output[out_index] = value + residual[out_index]
    QWEN38_PREFILL_GEMM_MMA8W_BODY(STORE_RESIDUAL_FLOAT8W)
#undef STORE_RESIDUAL_FLOAT8W
}

/* Q8 wide small-batch half MMA: the exact shape of the Q4 mma8w above
 * (64 output rows, eight-row batch tile, K-split simdgroups), for the
 * 3-8 token verify range. The code plane holds one signed int8 per
 * weight — 64 bytes per block — so the dequant loop reads each lane's 32
 * consecutive codes as four uint words and sign-extends them through
 * char4 before scaling. */

#define QWEN38_PREFILL_GEMM_Q8_MMA8W_BODY(STORE)                          \
    threadgroup half w_tile[kGemmTileK * kGemmWideRows];                  \
    threadgroup half spill[4 * 8 * kGemmWideRows];                        \
    uint row0 = group_id.x * kGemmWideRows;                               \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[8];                                     \
    for (uint n = 0; n < 8; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[4];                                                       \
    for (uint i = 0; i < 4; ++i) c_acc[i] = 0.0f;                         \
    uint kk0 = simdgroup_index * 16;                                      \
    uint r = tid & 63u;                                                   \
    uint k_base = (tid >> 6) * 32;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        if (row0 + r < p.rows) {                                          \
            uint block = (row0 + r) * p.groups_per_row + group;           \
            Q4PrefillMeta meta = metadata[block];                         \
            half scale = meta.scale;                                      \
            half bias = meta.bias;                                        \
            device const uint *words = (device const uint *)              \
                (quants + block * 64 + k_base);                           \
            for (uint word = 0; word < 8; ++word) {                       \
                half4 w = half4(as_type<char4>(words[word])) *            \
                           scale + bias;                                  \
                uint base = (k_base + word * 4) * kGemmWideRows + r;      \
                w_tile[base] = w.x;                                       \
                w_tile[base + kGemmWideRows] = w.y;                       \
                w_tile[base + 2 * kGemmWideRows] = w.z;                   \
                w_tile[base + 3 * kGemmWideRows] = w.w;                   \
            }                                                             \
        } else {                                                          \
            for (uint i = 0; i < 32; ++i)                                 \
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = kk0; kk < kk0 + 16; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + group * 64 + kk, columns);              \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmWideRows + n * 8,       \
                               kGemmWideRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        if ((group & 3u) == 3u || group == p.groups_per_row - 1) {        \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill +                                   \
                                    simdgroup_index * 8 * kGemmWideRows + \
                                    n * 8,                                \
                                kGemmWideRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
            for (uint i = 0; i < 4; ++i) {                                \
                uint linear = tid * 4 + i;                                \
                float sum = 0.0f;                                         \
                for (uint sg = 0; sg < 4; ++sg)                           \
                    sum += float(spill[sg * 8 * kGemmWideRows + linear]); \
                c_acc[i] += sum;                                          \
            }                                                             \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 4; ++i) {                                        \
        uint linear = tid * 4 + i;                                        \
        uint b = linear >> 6;                                             \
        uint out_row = linear & 63u;                                      \
        if (b < kBatch && row0 + out_row < p.rows) {                      \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q8_gemm_f16_mma8w(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN_Q8_8W output[out_index] = value
    QWEN38_PREFILL_GEMM_Q8_MMA8W_BODY(STORE_PLAIN_Q8_8W)
#undef STORE_PLAIN_Q8_8W
}

/* Wide-tile half MMA: the same half staging, device-direct activation
 * fragments and per-group float spill as the path above, but each
 * threadgroup covers 64 output rows and each simdgroup holds eight 8x8
 * accumulators. Per unit of math this halves the activation-fragment
 * loads and the barrier rounds — the output-tile shape mature Metal
 * GEMM implementations use. Row counts need not divide 64; the last
 * row block stages zeros and guards its stores. */

#define QWEN38_PREFILL_GEMM_MMA3_BODY(STORE)                              \
    threadgroup half w_tile[kGemmTileK * kGemmWideRows];                  \
    threadgroup half spill[kGemmTileBatch * kGemmWideRows];               \
    uint row0 = group_id.x * kGemmWideRows;                               \
    uint batch0 = group_id.y * kGemmTileBatch;                            \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[8];                                     \
    for (uint n = 0; n < 8; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[16];                                                      \
    for (uint i = 0; i < 16; ++i) c_acc[i] = 0.0f;                        \
    uint b0 = batch0 + simdgroup_index * 8;                               \
    uint spill0 = simdgroup_index * 8;                                    \
    uint r = tid & 63u;                                                   \
    uint k_base = (tid >> 6) * 32;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        if (row0 + r < p.rows) {                                          \
            uint block = (row0 + r) * p.groups_per_row + group;           \
            Q4PrefillMeta meta = metadata[block];                         \
            half scale = meta.scale;                                      \
            half bias = meta.bias;                                        \
            device const uint *words = (device const uint *)              \
                (quants + block * 32 + (k_base >> 1));                    \
            for (uint word = 0; word < 4; ++word) {                       \
                uint bits = words[word];                                  \
                half4 lo = half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) *   \
                           scale + bias;                                  \
                half4 hi = half4(as_type<uchar4>((bits >> 4) &            \
                                                 0x0f0f0f0fu)) *          \
                           scale + bias;                                  \
                uint base = (k_base + word * 8) * kGemmWideRows + r;      \
                w_tile[base] = lo.x;                                      \
                w_tile[base + kGemmWideRows] = hi.x;                      \
                w_tile[base + 2 * kGemmWideRows] = lo.y;                  \
                w_tile[base + 3 * kGemmWideRows] = hi.y;                  \
                w_tile[base + 4 * kGemmWideRows] = lo.z;                  \
                w_tile[base + 5 * kGemmWideRows] = hi.z;                  \
                w_tile[base + 6 * kGemmWideRows] = lo.w;                  \
                w_tile[base + 7 * kGemmWideRows] = hi.w;                  \
            }                                                             \
        } else {                                                          \
            for (uint i = 0; i < 32; ++i)                                 \
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + b0 * columns + group * 64 + kk,         \
                           columns);                                      \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmWideRows + n * 8,       \
                               kGemmWideRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        {                                                                 \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill + spill0 * kGemmWideRows + n * 8,   \
                                kGemmWideRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            simdgroup_barrier(mem_flags::mem_threadgroup);                \
            for (uint i = 0; i < 16; ++i)                                 \
                c_acc[i] += float(spill[tid * 16 + i]);                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 16; ++i) {                                       \
        uint linear = tid * 16 + i;                                       \
        uint b = batch0 + (linear >> 6);                                  \
        uint out_row = linear & 63u;                                      \
        if (b < kBatch && row0 + out_row < p.rows) {                      \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q4_gemm_f16_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN3 output[out_index] = value
    QWEN38_PREFILL_GEMM_MMA3_BODY(STORE_PLAIN3)
#undef STORE_PLAIN3
}


/* Partial MLP fusion for large-batch prefill.
 *
 * This keeps the proven mma3 GEMM kernel shape/occupancy and moves the
 * elementwise work into the GEMM stores:
 *   1) gate GEMM writes SiLU(gate), not raw gate
 *   2) up GEMM multiplies its result by the stored SiLU(gate)
 *
 * Compared with the ordinary path this removes the raw-up temporary plane
 * and the standalone SiLU kernel, while avoiding the extra threadgroup
 * memory/register pressure of the full gate+up fused kernel. */
kernel void qwen38_prefill_q4_gate_silu_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_GATE_SILU3 output[out_index] = value / (1.0f + exp(-value))
    QWEN38_PREFILL_GEMM_MMA3_BODY(STORE_GATE_SILU3)
#undef STORE_GATE_SILU3
}

kernel void qwen38_prefill_q4_up_mul_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *gate_silu [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_UP_MUL3 output[out_index] = value * gate_silu[out_index]
    QWEN38_PREFILL_GEMM_MMA3_BODY(STORE_UP_MUL3)
#undef STORE_UP_MUL3
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f16_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_HALF3 \
    output[out_index] = value + float(residual[out_index])
    QWEN38_PREFILL_GEMM_MMA3_BODY(STORE_RESIDUAL_HALF3)
#undef STORE_RESIDUAL_HALF3
}

kernel void qwen38_prefill_q4_gemm_f32_residual_f32_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_RESIDUAL_FLOAT3 \
    output[out_index] = value + residual[out_index]
    QWEN38_PREFILL_GEMM_MMA3_BODY(STORE_RESIDUAL_FLOAT3)
#undef STORE_RESIDUAL_FLOAT3
}


/* Fused Q4 gate + up + SiLU for large-batch prefill.
 *
 * The ordinary MLP path launches two independent mma3 GEMMs, materializes
 * both [batch x 17408] float planes, then launches a third kernel to read
 * both planes and write SiLU(gate) * up. This kernel keeps the exact mma3
 * arithmetic order for each projection but stores the completed gate tile
 * in threadgroup memory, reuses the same GEMM scratch for the up projection,
 * and writes only the activated plane to device memory.
 *
 * It does not reduce Q4 weight traffic, so the expected gain is bounded.
 * What it removes is two large device writes, two large device reads and one
 * separate elementwise dispatch per MLP. The host selects it only with
 * QWEN38_PREFILL_FUSED_MLP=1 and only on the mma3 large-batch path. */
kernel void qwen38_prefill_q4_gate_up_silu_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *gate_quants [[buffer(1)]],
    device const Q4PrefillMeta *gate_metadata [[buffer(2)]],
    device const uchar *up_quants [[buffer(3)]],
    device const Q4PrefillMeta *up_metadata [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant PrefillGemmParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {

    threadgroup half w_tile[kGemmTileK * kGemmWideRows];
    threadgroup half spill[kGemmTileBatch * kGemmWideRows];
    threadgroup float gate_tile[kGemmTileBatch * kGemmWideRows];

    const uint row0 = group_id.x * kGemmWideRows;
    const uint batch0 = group_id.y * kGemmTileBatch;
    const uint columns = p.groups_per_row * 64;
    const uint b0 = batch0 + simdgroup_index * 8;
    const uint spill0 = simdgroup_index * 8;
    const uint r = tid & 63u;
    const uint k_base = (tid >> 6) * 32;

    float c_acc[16];
    for (uint i = 0; i < 16; ++i) c_acc[i] = 0.0f;

    /* Gate projection. */
    for (uint group = 0; group < p.groups_per_row; ++group) {
        if (row0 + r < p.rows) {
            const uint block = (row0 + r) * p.groups_per_row + group;
            const Q4PrefillMeta meta = gate_metadata[block];
            const half scale = meta.scale;
            const half bias = meta.bias;
            device const uint *words = (device const uint *)
                (gate_quants + block * 32 + (k_base >> 1));
            for (uint word = 0; word < 4; ++word) {
                const uint bits = words[word];
                const half4 lo =
                    half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) * scale + bias;
                const half4 hi =
                    half4(as_type<uchar4>((bits >> 4) & 0x0f0f0f0fu)) *
                    scale + bias;
                const uint base =
                    (k_base + word * 8) * kGemmWideRows + r;
                w_tile[base] = lo.x;
                w_tile[base + kGemmWideRows] = hi.x;
                w_tile[base + 2 * kGemmWideRows] = lo.y;
                w_tile[base + 3 * kGemmWideRows] = hi.y;
                w_tile[base + 4 * kGemmWideRows] = lo.z;
                w_tile[base + 5 * kGemmWideRows] = hi.z;
                w_tile[base + 6 * kGemmWideRows] = lo.w;
                w_tile[base + 7 * kGemmWideRows] = hi.w;
            }
        } else {
            for (uint i = 0; i < 32; ++i)
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 accumulator[8];
        for (uint n = 0; n < 8; ++n)
            accumulator[n] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint kk = 0; kk < kGemmTileK; kk += 8) {
            simdgroup_half8x8 a;
            simdgroup_load(a,
                           x + b0 * columns + group * 64 + kk,
                           columns);
            for (uint n = 0; n < 8; ++n) {
                simdgroup_half8x8 b_fragment;
                simdgroup_load(b_fragment,
                               w_tile + kk * kGemmWideRows + n * 8,
                               kGemmWideRows);
                simdgroup_multiply_accumulate(accumulator[n], a, b_fragment,
                                              accumulator[n]);
            }
        }

        for (uint n = 0; n < 8; ++n)
            simdgroup_store(accumulator[n],
                            spill + spill0 * kGemmWideRows + n * 8,
                            kGemmWideRows);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; ++i)
            c_acc[i] += float(spill[tid * 16 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 16; ++i) {
        const uint linear = tid * 16 + i;
        const uint local_b = linear >> 6;
        const uint out_row = linear & 63u;
        if (batch0 + local_b < kBatch && row0 + out_row < p.rows)
            gate_tile[local_b * kGemmWideRows + out_row] = c_acc[i];
        c_acc[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Up projection, using the same weight/spill scratch. */
    for (uint group = 0; group < p.groups_per_row; ++group) {
        if (row0 + r < p.rows) {
            const uint block = (row0 + r) * p.groups_per_row + group;
            const Q4PrefillMeta meta = up_metadata[block];
            const half scale = meta.scale;
            const half bias = meta.bias;
            device const uint *words = (device const uint *)
                (up_quants + block * 32 + (k_base >> 1));
            for (uint word = 0; word < 4; ++word) {
                const uint bits = words[word];
                const half4 lo =
                    half4(as_type<uchar4>(bits & 0x0f0f0f0fu)) * scale + bias;
                const half4 hi =
                    half4(as_type<uchar4>((bits >> 4) & 0x0f0f0f0fu)) *
                    scale + bias;
                const uint base =
                    (k_base + word * 8) * kGemmWideRows + r;
                w_tile[base] = lo.x;
                w_tile[base + kGemmWideRows] = hi.x;
                w_tile[base + 2 * kGemmWideRows] = lo.y;
                w_tile[base + 3 * kGemmWideRows] = hi.y;
                w_tile[base + 4 * kGemmWideRows] = lo.z;
                w_tile[base + 5 * kGemmWideRows] = hi.z;
                w_tile[base + 6 * kGemmWideRows] = lo.w;
                w_tile[base + 7 * kGemmWideRows] = hi.w;
            }
        } else {
            for (uint i = 0; i < 32; ++i)
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 accumulator[8];
        for (uint n = 0; n < 8; ++n)
            accumulator[n] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        for (uint kk = 0; kk < kGemmTileK; kk += 8) {
            simdgroup_half8x8 a;
            simdgroup_load(a,
                           x + b0 * columns + group * 64 + kk,
                           columns);
            for (uint n = 0; n < 8; ++n) {
                simdgroup_half8x8 b_fragment;
                simdgroup_load(b_fragment,
                               w_tile + kk * kGemmWideRows + n * 8,
                               kGemmWideRows);
                simdgroup_multiply_accumulate(accumulator[n], a, b_fragment,
                                              accumulator[n]);
            }
        }

        for (uint n = 0; n < 8; ++n)
            simdgroup_store(accumulator[n],
                            spill + spill0 * kGemmWideRows + n * 8,
                            kGemmWideRows);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; ++i)
            c_acc[i] += float(spill[tid * 16 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 16; ++i) {
        const uint linear = tid * 16 + i;
        const uint local_b = linear >> 6;
        const uint out_row = linear & 63u;
        const uint b = batch0 + local_b;
        if (b < kBatch && row0 + out_row < p.rows) {
            const uint out_index = b * p.rows + row0 + out_row;
            const float gate =
                gate_tile[local_b * kGemmWideRows + out_row];
            output[out_index] =
                (gate / (1.0f + exp(-gate))) * c_acc[i];
        }
    }
}

/* Q8 wide-tile half MMA: the exact shape of the Q4 mma3 above (64 output
 * rows, 32 batch rows, eight 8x8 accumulators per simdgroup), but the code
 * plane holds one signed int8 per weight — 64 bytes per block instead of 32
 * nibble-pair bytes. The only difference is the dequant loop: each lane
 * reads its 32 consecutive codes as eight uint words (double the Q4
 * nibble-pair traffic) and sign-extends them through char4 before scaling,
 * so negative codes stay negative. */

#define QWEN38_PREFILL_GEMM_Q8_MMA3_BODY(STORE)                           \
    threadgroup half w_tile[kGemmTileK * kGemmWideRows];                  \
    threadgroup half spill[kGemmTileBatch * kGemmWideRows];               \
    uint row0 = group_id.x * kGemmWideRows;                               \
    uint batch0 = group_id.y * kGemmTileBatch;                            \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_half8x8 accumulator[8];                                     \
    for (uint n = 0; n < 8; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);  \
    float c_acc[16];                                                      \
    for (uint i = 0; i < 16; ++i) c_acc[i] = 0.0f;                        \
    uint b0 = batch0 + simdgroup_index * 8;                               \
    uint spill0 = simdgroup_index * 8;                                    \
    uint r = tid & 63u;                                                   \
    uint k_base = (tid >> 6) * 32;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        if (row0 + r < p.rows) {                                          \
            uint block = (row0 + r) * p.groups_per_row + group;           \
            Q4PrefillMeta meta = metadata[block];                         \
            half scale = meta.scale;                                      \
            half bias = meta.bias;                                        \
            device const uint *words = (device const uint *)              \
                (quants + block * 64 + k_base);                           \
            for (uint word = 0; word < 8; ++word) {                       \
                half4 w = half4(as_type<char4>(words[word])) *            \
                           scale + bias;                                  \
                uint base = (k_base + word * 4) * kGemmWideRows + r;      \
                w_tile[base] = w.x;                                       \
                w_tile[base + kGemmWideRows] = w.y;                       \
                w_tile[base + 2 * kGemmWideRows] = w.z;                   \
                w_tile[base + 3 * kGemmWideRows] = w.w;                   \
            }                                                             \
        } else {                                                          \
            for (uint i = 0; i < 32; ++i)                                 \
                w_tile[(k_base + i) * kGemmWideRows + r] = 0.0h;          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_half8x8 a;                                          \
            simdgroup_load(a, x + b0 * columns + group * 64 + kk,         \
                           columns);                                      \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_half8x8 b_fragment;                             \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmWideRows + n * 8,       \
                               kGemmWideRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        {                                                                 \
            for (uint n = 0; n < 8; ++n) {                                \
                simdgroup_store(accumulator[n],                           \
                                spill + spill0 * kGemmWideRows + n * 8,   \
                                kGemmWideRows);                           \
                accumulator[n] =                                          \
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);       \
            }                                                             \
            simdgroup_barrier(mem_flags::mem_threadgroup);                \
            for (uint i = 0; i < 16; ++i)                                 \
                c_acc[i] += float(spill[tid * 16 + i]);                   \
            threadgroup_barrier(mem_flags::mem_threadgroup);              \
        }                                                                 \
    }                                                                     \
    for (uint i = 0; i < 16; ++i) {                                       \
        uint linear = tid * 16 + i;                                       \
        uint b = batch0 + (linear >> 6);                                  \
        uint out_row = linear & 63u;                                      \
        if (b < kBatch && row0 + out_row < p.rows) {                      \
            uint out_index = b * p.rows + row0 + out_row;                 \
            float value = c_acc[i];                                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen38_prefill_q8_gemm_f16_mma3(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define STORE_PLAIN_Q8_3 output[out_index] = value
    QWEN38_PREFILL_GEMM_Q8_MMA3_BODY(STORE_PLAIN_Q8_3)
#undef STORE_PLAIN_Q8_3
}

/* MTP input fusion: normalized token embedding concatenated with the
 * normalized main-model hidden state, producing the [batch x 10240] input
 * of the MTP fc projection. One threadgroup per batch position. The same
 * threadgroup array carries two reductions, so a barrier separates the
 * read of one result from the next reduction's writes. */
kernel void qwen38_prefill_mtp_fuse(
    device const uchar *embedding_quants [[buffer(0)]],
    device const Q4PrefillMeta *embedding_metadata [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device const half *hidden [[buffer(3)]],
    device const float *embedding_norm [[buffer(4)]],
    device const float *hidden_norm [[buffer(5)]],
    device half *output [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    uint token_id = token_ids[s];
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        uint group = index / 64;
        uint within = index - group * 64;
        uint block = token_id * kPrefillEmbeddingGroups + group;
        uchar bits = embedding_quants[block * 32 + (within >> 1)];
        uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
        Q4PrefillMeta meta = embedding_metadata[block];
        float value = float(meta.scale) * float(quant) + float(meta.bias);
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_embedding = rsqrt(partials[0] / float(kPrefillHidden) +
                                    kPrefillRmsEpsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        uint group = index / 64;
        uint within = index - group * 64;
        uint block = token_id * kPrefillEmbeddingGroups + group;
        uchar bits = embedding_quants[block * 32 + (within >> 1)];
        uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
        Q4PrefillMeta meta = embedding_metadata[block];
        float value = float(meta.scale) * float(quant) + float(meta.bias);
        output[s * 2 * kPrefillHidden + index] =
            half(value * inv_rms_embedding * embedding_norm[index]);
    }
    float hidden_sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = float(hidden[s * kPrefillHidden + index]);
        hidden_sum += value * value;
    }
    partials[tid] = hidden_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_hidden = rsqrt(partials[0] / float(kPrefillHidden) +
                                 kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        output[s * 2 * kPrefillHidden + kPrefillHidden + index] =
            half(float(hidden[s * kPrefillHidden + index]) *
                 inv_rms_hidden * hidden_norm[index]);
    }
}

/* -------------------------------------------------------------------------
 * DFlash2 draft model for Qwen3.8-27B.
 *
 * The published drafter is 5120-wide with 32 Q heads, 8 KV heads, 128-dim
 * heads, a 2048 sliding window, non-causal proposal-block attention, two-tap
 * grouped dynamic convolutions, and a 256-rank candidate selector.  These
 * kernels intentionally use the existing kBatch function constant so the
 * host can reuse the S1..S8 prefill pipeline buckets.
 * ------------------------------------------------------------------------- */
constant uint kDFlashHidden = 5120;
constant uint kDFlashHeads = 32;
constant uint kDFlashKVHeads = 8;
constant uint kDFlashHeadDim = 128;
constant uint kDFlashQWidth = 4096;
constant uint kDFlashKVWidth = 1024;
constant uint kDFlashWindow = 2047; /* RotatingKVCache max_size = sliding_window - 1. */
constant uint kDFlashGroups = 320;  /* 5120 / 16 */
constant uint kDFlashConvProj = 1280; /* 2 sides x 2 taps x 320 groups */
constant float kDFlashRopeTheta = 10000000.0f;
constant float kDFlashRmsEps = 1.0e-6f;

struct DFlashCaptureParams {
    uint start_position;
    uint tap_slot;
};
struct DFlashGatherParams {
    uint start_position;
    uint rows;
};
struct DFlashContextParams {
    uint start_position;
    uint rows;
};
struct DFlashAttentionParams {
    uint proposal_position;
    uint context_count;
    uint proposal_rows;
    uint reserved;
};

kernel void qwen38_dflash_capture_target(
    device const float *input [[buffer(0)]],
    device half *ring [[buffer(1)]],
    device uint *tags [[buffer(2)]],
    constant DFlashCaptureParams &p [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint d = pos.x, s = pos.y;
    if (d >= kDFlashHidden || s >= kBatch) return;
    uint phys = (p.start_position + s) & 2047u;
    ring[((phys * 5u + p.tap_slot) * kDFlashHidden) + d] =
        half(input[s * kDFlashHidden + d]);
    if (d == 0u && p.tap_slot == 4u) tags[phys] = p.start_position + s + 1u;
}

kernel void qwen38_dflash_gather_target(
    device const half *ring [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant DFlashGatherParams &p [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint d = pos.x, s = pos.y;
    if (d >= 5u * kDFlashHidden || s >= p.rows || s >= kBatch) return;
    uint phys = (p.start_position + s) & 2047u;
    uint tap = d / kDFlashHidden;
    uint within = d - tap * kDFlashHidden;
    output[s * (5u * kDFlashHidden) + d] =
        ring[((phys * 5u + tap) * kDFlashHidden) + within];
}

kernel void qwen38_dflash_rms_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float part[256];
    uint s = gid.x;
    float sum = 0.0f;
    for (uint d = tid; d < kDFlashHidden; d += 256) {
        float x = float(input[s * kDFlashHidden + d]); sum += x*x;
    }
    part[tid] = sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128; stride; stride>>=1) {
        if (tid < stride) part[tid] += part[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(part[0] / float(kDFlashHidden) + kDFlashRmsEps);
    for (uint d = tid; d < kDFlashHidden; d += 256)
        output[s*kDFlashHidden+d] = half(float(input[s*kDFlashHidden+d]) * inv * float(weight[d]));
}

kernel void qwen38_dflash_rms_f32(
    device const float *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float part[256];
    uint s=gid.x; float sum=0.0f;
    for (uint d=tid; d<kDFlashHidden; d+=256) { float x=input[s*kDFlashHidden+d]; sum+=x*x; }
    part[tid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128; stride; stride>>=1) { if(tid<stride) part[tid]+=part[tid+stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=rsqrt(part[0]/float(kDFlashHidden)+kDFlashRmsEps);
    for(uint d=tid; d<kDFlashHidden; d+=256) output[s*kDFlashHidden+d]=half(input[s*kDFlashHidden+d]*inv*float(weight[d]));
}

/* Small FP16 row-major matrix multiply used by the dynamic-conv projection
 * and selector projection.  Large backbone matrices stay on Q4G64. */
kernel void qwen38_dflash_f16_gemm(
    device const half *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant PrefillGemmParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
    uint row=gid.x*sgs+sg; if(row>=p.rows) return;
    uint cols=p.groups_per_row; /* here this field means literal columns */
    for(uint s=0;s<kBatch;++s) {
        float partial=0.0f;
        for(uint c=lane;c<cols;c+=32) partial += float(x[s*cols+c])*float(w[row*cols+c]);
        float sum=simd_sum(partial);
        if(lane==0) out[s*p.rows+row]=sum;
    }
}

kernel void qwen38_dflash_dynamic_prepare(
    device const half *hidden [[buffer(0)]],
    device const float *dynamic [[buffer(1)]],
    device const half *base [[buffer(2)]],
    device half *output [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint c=pos.x,s=pos.y; if(c>=kDFlashHidden||s>=kBatch) return;
    uint g=c>>4; float y=0.0f;
    for(uint tap=0;tap<2;++tap) {
        if(s<tap) continue;
        float coeff=float(base[tap*kDFlashHidden+c]) + dynamic[s*kDFlashConvProj + tap*kDFlashGroups + g];
        y += coeff * float(hidden[(s-tap)*kDFlashHidden+c]);
    }
    output[s*kDFlashHidden+c]=half(y);
}

kernel void qwen38_dflash_dynamic_finish(
    device const float *hidden [[buffer(0)]],
    device const float *dynamic [[buffer(1)]],
    device const half *base [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint c=pos.x,s=pos.y; if(c>=kDFlashHidden||s>=kBatch) return;
    uint g=c>>4; float y=0.0f;
    for(uint tap=0;tap<2;++tap) {
        if(s<tap) continue;
        float coeff=float(base[(2u+tap)*kDFlashHidden+c]) + dynamic[s*kDFlashConvProj + (2u+tap)*kDFlashGroups + g];
        y += coeff * hidden[(s-tap)*kDFlashHidden+c];
    }
    output[s*kDFlashHidden+c]=y;
}

kernel void qwen38_dflash_add_residual(
    device const float *delta [[buffer(0)]],
    device const float *residual [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint d=pos.x,s=pos.y; if(d>=kDFlashHidden||s>=kBatch)return;
    uint i=s*kDFlashHidden+d; output[i]=residual[i]+delta[i];
}

kernel void qwen38_dflash_copy_float(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < kDFlashHidden && gid.y < kBatch)
        output[gid.y * kDFlashHidden + gid.x] =
            input[gid.y * kDFlashHidden + gid.x];
}

kernel void qwen38_dflash_half_to_float(
    device const half *input [[buffer(0)]], device float *output [[buffer(1)]],
    uint2 pos [[thread_position_in_grid]]) {
    uint d=pos.x,s=pos.y; if(d>=kDFlashHidden||s>=kBatch)return;
    output[s*kDFlashHidden+d]=float(input[s*kDFlashHidden+d]);
}

kernel void qwen38_dflash_float_to_half_width(
    device const float *input [[buffer(0)]], device half *output [[buffer(1)]],
    constant uint &width [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    uint d=pos.x,s=pos.y; if(d>=width||s>=kBatch)return;
    output[s*width+d]=half(input[s*width+d]);
}

inline float dflash_rope(threadgroup const float *v,uint d,uint pos) {
    uint f=d&63u; float exponent=-2.0f*float(f)/128.0f;
    float a=float(pos)*pow(kDFlashRopeTheta,exponent), c=cos(a), ss=sin(a);
    return d<64 ? v[d]*c-v[d+64]*ss : v[d]*c+v[d-64]*ss;
}

kernel void qwen38_dflash_prepare_q(
    device const float *projected [[buffer(0)]], device const half *norm [[buffer(1)]],
    constant DFlashAttentionParams &p [[buffer(2)]], device float *q [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float vals[128]; threadgroup float sq[128];
    uint head=gid.x,s=gid.y; if(head>=kDFlashHeads||s>=kBatch)return;
    float x=projected[s*kDFlashQWidth+head*kDFlashHeadDim+tid]; vals[tid]=x; sq[tid]=x*x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=64;st;st>>=1){if(tid<st)sq[tid]+=sq[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}
    vals[tid]=x*rsqrt(sq[0]/128.0f+kDFlashRmsEps)*float(norm[tid]); threadgroup_barrier(mem_flags::mem_threadgroup);
    q[s*kDFlashQWidth+head*kDFlashHeadDim+tid]=dflash_rope(vals,tid,p.proposal_position+s);
}

kernel void qwen38_dflash_prepare_prop_k(
    device const float *projected [[buffer(0)]], device const half *norm [[buffer(1)]],
    constant DFlashAttentionParams &p [[buffer(2)]], device half *key [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float vals[128]; threadgroup float sq[128];
    uint head=gid.x,s=gid.y; if(head>=kDFlashKVHeads||s>=kBatch)return;
    float x=projected[s*kDFlashKVWidth+head*kDFlashHeadDim+tid];vals[tid]=x;sq[tid]=x*x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=64;st;st>>=1){if(tid<st)sq[tid]+=sq[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}
    vals[tid]=x*rsqrt(sq[0]/128.0f+kDFlashRmsEps)*float(norm[tid]);threadgroup_barrier(mem_flags::mem_threadgroup);
    key[s*kDFlashKVWidth+head*kDFlashHeadDim+tid]=half(dflash_rope(vals,tid,p.proposal_position+s));
}

kernel void qwen38_dflash_prepare_context_k(
    device const float *projected [[buffer(0)]], device const half *norm [[buffer(1)]],
    constant DFlashContextParams &p [[buffer(2)]], device half *cache [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float vals[128]; threadgroup float sq[128];
    uint head=gid.x,s=gid.y; if(head>=kDFlashKVHeads||s>=p.rows||s>=kBatch)return;
    float x=projected[s*kDFlashKVWidth+head*kDFlashHeadDim+tid];vals[tid]=x;sq[tid]=x*x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=64;st;st>>=1){if(tid<st)sq[tid]+=sq[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}
    vals[tid]=x*rsqrt(sq[0]/128.0f+kDFlashRmsEps)*float(norm[tid]);threadgroup_barrier(mem_flags::mem_threadgroup);
    uint abs=p.start_position+s, phys=abs%kDFlashWindow;
    cache[(phys*kDFlashKVHeads+head)*kDFlashHeadDim+tid]=half(dflash_rope(vals,tid,abs));
}

kernel void qwen38_dflash_store_context_v(
    device const float *projected [[buffer(0)]], constant DFlashContextParams &p [[buffer(1)]],
    device half *cache [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    uint d=pos.x,s=pos.y; if(d>=kDFlashKVWidth||s>=p.rows||s>=kBatch)return;
    uint phys=(p.start_position+s)%kDFlashWindow; cache[phys*kDFlashKVWidth+d]=half(projected[s*kDFlashKVWidth+d]);
}

kernel void qwen38_dflash_scores(
    device const float *q [[buffer(0)]], device const half *ctx_k [[buffer(1)]],
    device const half *prop_k [[buffer(2)]], constant DFlashAttentionParams &p [[buffer(3)]],
    device float *scores [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
    uint flat=gid.y; uint s=flat/kDFlashHeads, qh=flat-s*kDFlashHeads;
    uint total=p.context_count+p.proposal_rows, kvh=qh/4u;
    for(uint j=tid;j<total;j+=256) {
        float dotv=0.0f; device const float *qq=q+(s*kDFlashHeads+qh)*kDFlashHeadDim;
        if(j<p.context_count) {
            /* Match upstream sliding-window semantics exactly. Proposal row s
             * is at logical offset context_count+s, so later proposal rows
             * drop the corresponding number of oldest context positions. */
            if (p.context_count + s - j >= 2048u) {
                scores[(flat*(kDFlashWindow+8u))+j] = -INFINITY;
                continue;
            }
            uint start=p.proposal_position-p.context_count, abs=start+j, phys=abs%kDFlashWindow;
            device const half *kk=ctx_k+(phys*kDFlashKVHeads+kvh)*kDFlashHeadDim;
            for(uint d=0;d<kDFlashHeadDim;++d) dotv += qq[d]*float(kk[d]);
        } else {
            uint ps=j-p.context_count; device const half *kk=prop_k+(ps*kDFlashKVHeads+kvh)*kDFlashHeadDim;
            for(uint d=0;d<kDFlashHeadDim;++d) dotv += qq[d]*float(kk[d]);
        }
        scores[(flat*(kDFlashWindow+8u))+j]=dotv*(1.0f/sqrt(128.0f));
    }
}

kernel void qwen38_dflash_value(
    device const float *scores [[buffer(0)]], device const half *ctx_v [[buffer(1)]],
    device const half *prop_v [[buffer(2)]], constant DFlashAttentionParams &p [[buffer(3)]],
    device float *output [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
    threadgroup float red[128];
    uint flat=gid.x, s=flat/kDFlashHeads, qh=flat-s*kDFlashHeads, kvh=qh/4u;
    uint total=p.context_count+p.proposal_rows; device const float *sc=scores+flat*(kDFlashWindow+8u);
    float m=-INFINITY; for(uint j=tid;j<total;j+=128)m=max(m,sc[j]); red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=64;st;st>>=1){if(tid<st)red[tid]=max(red[tid],red[tid+st]);threadgroup_barrier(mem_flags::mem_threadgroup);} m=red[0];
    float z=0.0f; for(uint j=tid;j<total;j+=128)z+=exp(sc[j]-m);red[tid]=z;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=64;st;st>>=1){if(tid<st)red[tid]+=red[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);} z=red[0];
    float acc=0.0f;
    for(uint j=0;j<total;++j){float pr=exp(sc[j]-m)/z; if(j<p.context_count){uint start=p.proposal_position-p.context_count,abs=start+j,phys=abs%kDFlashWindow;acc+=pr*float(ctx_v[(phys*kDFlashKVHeads+kvh)*kDFlashHeadDim+tid]);}else{uint ps=j-p.context_count;acc+=pr*float(prop_v[(ps*kDFlashKVHeads+kvh)*kDFlashHeadDim+tid]);}}
    output[(s*kDFlashHeads+qh)*kDFlashHeadDim+tid]=acc;
}
