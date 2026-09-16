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
 * half query tile (8 KB), the 8x128 float score scratch (8 KB) and the
 * 8x256 float O accumulator (16 KB) - exactly the 32 KB default budget.
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
    threadgroup half sq[kFlashRows * 256];
    threadgroup float ss[kFlashRows * 128];
    threadgroup float so[kFlashRows * 256];

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

    /* Zero the O accumulator; simdgroup g owns output dims {8*(g+4k) ...}
     * for k = 0..7, i.e. 64 dims of every row. */
    for (uint i = 0; i < 16; ++i) {
        const uint r = i / 2;
        const uint d = 8 * sgitg + 32 * (i % 2) + lane;
        so[r * 256 + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[2];
    float ssum[2];
    m[0] = -INFINITY;
    m[1] = -INFINITY;
    ssum[0] = 0.0f;
    ssum[1] = 0.0f;

    /* Row r attends to positions [0, start_position + row0 + r). The last
     * block is partial only for the smallest rows of the group. */
    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;
        const bool is_last = (b == total_blocks - 1);

        /* QK^T: the block's 64 positions form 8 tiles of 8; simdgroup g
         * computes tiles {g, g+4}, each an 8x8 MMA over all 256 query dims.
         * K comes straight from device memory (transposed load with the
         * 1024-half stride that skips the other KV heads), Q from the
         * shared tile. */
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

        /* Online softmax for this simdgroup's two rows. Lane l owns score
         * columns {2l, 2l+1} as a float2; the last block masks positions
         * past each row's context length. */
        for (uint jj = 0; jj < 2; ++jj) {
            const uint r = sgitg + 4 * jj;
            const float old_m = m[jj];
            float2 s2 = float2(ss[r * 128 + 2 * lane],
                               ss[r * 128 + 2 * lane + 1]);
            if (is_last) {
                const uint ctx = ctx_lo + sgitg + 4 * jj;
                if (block_start + 2 * lane >= ctx)
                    s2[0] = -INFINITY;
                if (block_start + 2 * lane + 1 >= ctx)
                    s2[1] = -INFINITY;
            }
            m[jj] = simd_max(max(old_m, max(s2[0], s2[1])));
            const float alpha = exp(old_m - m[jj]);
            const float2 p2 = exp(s2 - m[jj]);
            ssum[jj] = ssum[jj] * alpha + simd_sum(p2[0] + p2[1]);
            ss[r * 128 + 2 * lane] = p2[0];
            ss[r * 128 + 2 * lane + 1] = p2[1];
            for (uint i = 0; i < 8; ++i)
                so[r * 256 + lane + 32 * i] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* P.V: O += P.V over the block. lo[k] covers all 8 query rows and
         * output dims {8*(g+4k) ...}; simdgroup g thus owns 64 of the 256
         * output dims per row. V is read straight from device memory; the
         * 1024-half row stride skips the other three KV heads between
         * positions. */
        for (uint cc = 0; cc < kFlashBlock / 16; ++cc) {
            simdgroup_float8x8 vs[2];
            simdgroup_load(vs[0], ss + 16 * cc, 128);
            simdgroup_load(vs[1], ss + 16 * cc + 8, 128);
            const device half *pv = value_cache +
                ((block_start + 16 * cc) * kPrefillKVHeads + kv_head) * 256;
            for (uint ii = 0; ii < 4; ++ii) {
                simdgroup_float8x8 lo[2];
                simdgroup_load(lo[0], so + 8 * sgitg + 64 * ii, 256);
                simdgroup_load(lo[1], so + 8 * (sgitg + 4) + 64 * ii, 256);
                simdgroup_half8x8 mv[4];
                simdgroup_load(mv[0], pv + 8 * sgitg + 64 * ii, 1024);
                simdgroup_load(mv[1], pv + 8 * (sgitg + 4) + 64 * ii, 1024);
                simdgroup_load(mv[2], pv + 8 * 1024 + 8 * sgitg + 64 * ii, 1024);
                simdgroup_load(mv[3], pv + 8 * 1024 + 8 * (sgitg + 4) + 64 * ii, 1024);
                simdgroup_multiply_accumulate(lo[0], vs[0], mv[0], lo[0]);
                simdgroup_multiply_accumulate(lo[1], vs[0], mv[1], lo[1]);
                simdgroup_multiply_accumulate(lo[0], vs[1], mv[2], lo[0]);
                simdgroup_multiply_accumulate(lo[1], vs[1], mv[3], lo[1]);
                simdgroup_store(lo[0], so + 8 * sgitg + 64 * ii, 256);
                simdgroup_store(lo[1], so + 8 * (sgitg + 4) + 64 * ii, 256);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Normalize and apply the sigmoid query gate. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (ssum[jj] > 0.0f) ? 1.0f / ssum[jj] : 0.0f;
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
 * score scratch (4 KB) and the O accumulator (16 KB); the K/V tiles (8 KB
 * each) live in a dynamic [[threadgroup(0)]] buffer because the 40 KB total
 * exceeds the 32 KB static budget. */

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
    threadgroup half sq[kFlashRows * 256];
    threadgroup float ss[kFlashRows * 128];
    threadgroup float so[kFlashRows * 256];
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

    for (uint i = 0; i < 16; ++i) {
        const uint r = i / 2;
        const uint d = 8 * sgitg + 32 * (i % 2) + lane;
        so[r * 256 + d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[2];
    float ssum[2];
    m[0] = -INFINITY;
    m[1] = -INFINITY;
    ssum[0] = 0.0f;
    ssum[1] = 0.0f;

    const uint ctx_lo = parameters.start_position + row0 + 1;
    const uint total_blocks =
        (ctx_lo + kFlashRows - 1 + kFlashBlock - 1) / kFlashBlock;

    for (uint b = 0; b < total_blocks; ++b) {
        const uint block_start = b * kFlashBlock;
        const bool is_last = (b == total_blocks - 1);

        for (uint t = 0; t < kFlashBlock / 16; ++t) {
            const uint sub_start = block_start + 16 * t;

            /* Dequantize this sub-block's K and V: each thread dequantizes
             * one 256-dim vector (256 int8 codes times its fp32 scale).
             * Positions past the group's largest row context are zeroed so
             * the masked scores stay finite. */
            for (uint p = tid / 8; p < 16; p += 16) {
                const uint position = sub_start + p;
                const bool live = position < ctx_lo + kFlashRows - 1;
                const uint c0 = 8 * (tid % 8);
                if (live) {
                    const device char *kv = key_cache +
                        ((position * kPrefillKVHeads + kv_head) *
                         kPrefillKVQ8Stride);
                    const float ks = *(const device float *)(kv + 256);
                    for (uint d = c0; d < c0 + 8; ++d)
                        sk[p * 256 + d] = half((float)(char)kv[d] * ks);
                    const device char *vv = value_cache +
                        ((position * kPrefillKVHeads + kv_head) *
                         kPrefillKVQ8Stride);
                    const float vs = *(const device float *)(vv + 256);
                    for (uint d = c0; d < c0 + 8; ++d)
                        sv[p * 256 + d] = half((float)(char)vv[d] * vs);
                } else {
                    for (uint d = c0; d < c0 + 8; ++d) {
                        sk[p * 256 + d] = half(0.0f);
                        sv[p * 256 + d] = half(0.0f);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* QK^T for the 16 positions: simdgroup g computes position
             * tiles {g, g+4} of the sub-block. */
            for (uint cc = 0; cc < 2; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                const uint pos_tile = sgitg + 4 * cc;
                for (uint i = 0; i < 32; ++i) {
                    simdgroup_half8x8 mq, mk;
                    simdgroup_load(mq, sq + i * 8, 256);
                    simdgroup_load(mk, sk + 8 * pos_tile, 256, 0, true);
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }
                simdgroup_store(mqk, ss + 16 * t + 8 * pos_tile, 128, 0, false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* Online softmax for this simdgroup's two rows over the 16
             * positions; the last block masks past each row's context. */
            for (uint jj = 0; jj < 2; ++jj) {
                const uint r = sgitg + 4 * jj;
                const float old_m = m[jj];
                float2 s2 = float2(ss[r * 128 + 16 * t + 2 * lane],
                                   ss[r * 128 + 16 * t + 2 * lane + 1]);
                if (is_last) {
                    const uint ctx = ctx_lo + sgitg + 4 * jj;
                    if (sub_start + 2 * lane >= ctx)
                        s2[0] = -INFINITY;
                    if (sub_start + 2 * lane + 1 >= ctx)
                        s2[1] = -INFINITY;
                }
                m[jj] = simd_max(max(old_m, max(s2[0], s2[1])));
                const float alpha = exp(old_m - m[jj]);
                const float2 p2 = exp(s2 - m[jj]);
                ssum[jj] = ssum[jj] * alpha + simd_sum(p2[0] + p2[1]);
                ss[r * 128 + 16 * t + 2 * lane] = p2[0];
                ss[r * 128 + 16 * t + 2 * lane + 1] = p2[1];
                for (uint i = 0; i < 8; ++i)
                    so[r * 256 + lane + 32 * i] *= alpha;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* P.V over the 16 positions, same dimension split as fp16. */
            {
                simdgroup_float8x8 vs[2];
                simdgroup_load(vs[0], ss + 16 * t, 128);
                simdgroup_load(vs[1], ss + 16 * t + 8, 128);
                for (uint ii = 0; ii < 4; ++ii) {
                    simdgroup_float8x8 lo[2];
                    simdgroup_load(lo[0], so + 8 * sgitg + 64 * ii, 256);
                    simdgroup_load(lo[1], so + 8 * (sgitg + 4) + 64 * ii, 256);
                    simdgroup_half8x8 mv[4];
                    simdgroup_load(mv[0], sv + 8 * sgitg + 64 * ii, 256);
                    simdgroup_load(mv[1], sv + 8 * (sgitg + 4) + 64 * ii, 256);
                    simdgroup_load(mv[2], sv + 8 * 256 + 8 * sgitg + 64 * ii, 256);
                    simdgroup_load(mv[3], sv + 8 * 256 + 8 * (sgitg + 4) + 64 * ii, 256);
                    simdgroup_multiply_accumulate(lo[0], vs[0], mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs[0], mv[1], lo[1]);
                    simdgroup_multiply_accumulate(lo[0], vs[1], mv[2], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs[1], mv[3], lo[1]);
                    simdgroup_store(lo[0], so + 8 * sgitg + 64 * ii, 256);
                    simdgroup_store(lo[1], so + 8 * (sgitg + 4) + 64 * ii, 256);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    /* Normalize and apply the sigmoid query gate. */
    for (uint jj = 0; jj < 2; ++jj) {
        const uint r = sgitg + 4 * jj;
        if (row0 + r >= parameters.batch)
            break;
        const float scale = (ssum[jj] > 0.0f) ? 1.0f / ssum[jj] : 0.0f;
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
