#ifndef QWEN38_M3_DFLASH2_IMAGE_H
#define QWEN38_M3_DFLASH2_IMAGE_H

#include <stdint.h>

#define QWEN38_M3_DFLASH2_IMAGE_MAGIC "Q38DF2\0\0"
#define QWEN38_M3_DFLASH2_IMAGE_VERSION 1u
#define QWEN38_M3_DFLASH2_LAYERS 5u
#define QWEN38_M3_DFLASH2_HIDDEN 5120u
#define QWEN38_M3_DFLASH2_INTERMEDIATE 17408u
#define QWEN38_M3_DFLASH2_VOCAB 248320u
#define QWEN38_M3_DFLASH2_HEAD_DIM 128u
#define QWEN38_M3_DFLASH2_Q_HEADS 32u
#define QWEN38_M3_DFLASH2_KV_HEADS 8u
#define QWEN38_M3_DFLASH2_WINDOW 2048u
#define QWEN38_M3_DFLASH2_MASK_ID 248070u
#define QWEN38_M3_DFLASH2_SELECTOR_RANK 256u
#define QWEN38_M3_DFLASH2_SELECTOR_TOPK 16u
#define QWEN38_M3_DFLASH2_GROUP_SIZE 64u

typedef struct {
    uint64_t quants_offset;
    uint64_t quants_bytes;
    uint64_t metadata_offset;
    uint64_t metadata_bytes;
    uint32_t rows;
    uint32_t columns;
} qwen38_m3_dflash2_q4_matrix;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t columns;
} qwen38_m3_dflash2_f16_matrix;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
    uint32_t count;
    uint32_t reserved;
} qwen38_m3_dflash2_f16_vector;

typedef struct {
    qwen38_m3_dflash2_f16_vector input_norm;
    qwen38_m3_dflash2_f16_vector attention_conv_base; /* [2,2,5120] */
    qwen38_m3_dflash2_f16_matrix attention_conv_projection; /* [1280,5120] */
    qwen38_m3_dflash2_q4_matrix q_proj;
    qwen38_m3_dflash2_q4_matrix k_proj;
    qwen38_m3_dflash2_q4_matrix v_proj;
    qwen38_m3_dflash2_f16_vector q_norm; /* 128 */
    qwen38_m3_dflash2_f16_vector k_norm; /* 128 */
    qwen38_m3_dflash2_q4_matrix o_proj;
    qwen38_m3_dflash2_f16_vector post_attention_norm;
    qwen38_m3_dflash2_f16_vector mlp_conv_base; /* [2,2,5120] */
    qwen38_m3_dflash2_f16_matrix mlp_conv_projection; /* [1280,5120] */
    qwen38_m3_dflash2_q4_matrix gate_proj;
    qwen38_m3_dflash2_q4_matrix up_proj;
    qwen38_m3_dflash2_q4_matrix down_proj;
} qwen38_m3_dflash2_layer_desc;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t file_bytes;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t vocab_size;
    uint32_t num_layers;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t sliding_window;
    uint32_t mask_token_id;
    uint32_t selector_rank;
    uint32_t selector_top_k;
    uint32_t group_size;
    uint32_t target_layer_ids[5];
    float rope_theta;
    float rms_epsilon;
    uint32_t reserved0[6];
    qwen38_m3_dflash2_q4_matrix feature_projection; /* 5120 x 25600 */
    qwen38_m3_dflash2_f16_vector context_norm;
    qwen38_m3_dflash2_layer_desc layers[5];
    qwen38_m3_dflash2_f16_vector final_norm;
    qwen38_m3_dflash2_f16_matrix selector_hidden_projection; /* 256x5120 */
    qwen38_m3_dflash2_f16_matrix selector_predecessor; /* vocab x 256 */
    qwen38_m3_dflash2_f16_matrix selector_successor; /* vocab x 256 */
    char source_sha256[64];
} qwen38_m3_dflash2_image_header;

#endif
