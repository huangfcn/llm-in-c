/* Folder-only DeltaNet layer packer.
 *
 *   qwen38-m3-pack Q4_DIR Q8_DIR OUTPUT.q38delta LAYER
 *
 * Every byte comes from the plane dirs produced by qwen38_q4_quantize_all.py
 * (Q4_DIR) and qwen38_q8_delta_quantize_all.py (Q8_DIR):
 *   Q4_DIR/layer-NN/  mlp.{gate,up,down}_proj, linear_attn.out_proj  (codes.u8,
 *                     scale.f16, bias.f16)  +  pass-through fp32 vectors
 *                     input_layernorm.f32, post_attention_layernorm.f32,
 *                     linear_attn__norm.f32, linear_attn__conv1d.f32,
 *                     linear_attn__A_log.f32, linear_attn__dt_bias.f32
 *   Q8_DIR/layer-NN/  delta_input_q8_{codes.i8,scale.f16,bias.f16}
 *
 * No safetensors, no SHA. Integrity = exact plane size checks + the delta
 * cursor self-checks. delta_input is Q8 (precision tag 1); delta_output Q4.
 */
#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_image.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DELTA_QKV_ROWS = 10240, DELTA_Z_ROWS = 6144, DELTA_SCALAR_ROWS = 48,
    DELTA_INPUT_ROWS = DELTA_QKV_ROWS + DELTA_Z_ROWS + 2 * DELTA_SCALAR_ROWS,
    DELTA_OUTPUT_INPUTS = 6144, DELTA_CONV_VALUES = DELTA_QKV_ROWS * 4
};

static uint64_t align_page(uint64_t v) {
    return (v + QWEN38_M3_IMAGE_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN38_M3_IMAGE_HEADER_BYTES - 1);
}
static int pwrite_exact(int f, const void *in, size_t n, uint64_t off) {
    const unsigned char *b = in; size_t d = 0;
    while (d < n) {
        ssize_t a = pwrite(f, b + d, n - d, (off_t)(off + d));
        if (a < 0 && errno == EINTR) continue;
        if (a <= 0) return -1;
        d += (size_t)a;
    }
    return 0;
}
static int pread_exact(int f, void *out, size_t n, uint64_t off) {
    unsigned char *b = out; size_t d = 0;
    while (d < n) {
        ssize_t a = pread(f, b + d, n - d, (off_t)(off + d));
        if (a < 0 && errno == EINTR) continue;
        if (a <= 0) return -1;
        d += (size_t)a;
    }
    return 0;
}

/* Open a plane and require its exact size. */
static int open_sized(const char *dir, int layer, const char *base,
                      const char *suffix, uint64_t expect) {
    char path[512];
    if (layer < 0) snprintf(path, sizeof(path), "%s/global/%s%s", dir, base, suffix);
    else snprintf(path, sizeof(path), "%s/layer-%02d/%s%s", dir, layer, base, suffix);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size != expect) {
        fprintf(stderr, "%s: size %lld != expected %" PRIu64 "\n",
                path, (long long)st.st_size, expect);
        close(fd); return -1;
    }
    return fd;
}
static int copy_fd(int fd, int out, uint64_t bytes, uint64_t out_off) {
    enum { CH = 8 << 20 }; unsigned char *buf = malloc(CH);
    if (!buf) return -1;
    uint64_t done = 0;
    while (done < bytes) {
        size_t a = (size_t)(bytes - done); if (a > CH) a = CH;
        if (pread_exact(fd, buf, a, done) != 0 ||
            pwrite_exact(out, buf, a, out_off + done) != 0) { free(buf); return -1; }
        done += a;
    }
    free(buf); return 0;
}

/* Q4 tensor: codes.u8 verbatim into quant plane, scale/bias interleaved (no
 * convert -- already fp16) into meta plane. Advances cursors. */
static int put_q4(const char *dir, int layer, const char *base, int out,
                  uint64_t *qc, uint64_t *mc, uint64_t rows, uint64_t cols) {
    uint64_t code_bytes = rows * cols / 2;
    uint64_t groups = rows * (cols / 64);
    int cf = open_sized(dir, layer, base, "_codes.u8", code_bytes);
    if (cf < 0) return -1;
    int st = copy_fd(cf, out, code_bytes, *qc); close(cf);
    if (st != 0) return -1;
    *qc += code_bytes;
    uint64_t plane = groups * 2;
    int sf = open_sized(dir, layer, base, "_scale.f16", plane);
    if (sf < 0) return -1;
    int bf = open_sized(dir, layer, base, "_bias.f16", plane);
    if (bf < 0) { close(sf); return -1; }
    enum { VPC = 1 << 20 };
    uint16_t *s = malloc(VPC * 2), *b = malloc(VPC * 2), *c = malloc(VPC * 4);
    st = (s && b && c) ? 0 : -1;
    uint64_t done = 0;
    while (done < groups && st == 0) {
        size_t cnt = (size_t)(groups - done); if (cnt > VPC) cnt = VPC;
        if (pread_exact(sf, s, cnt * 2, done * 2) != 0 ||
            pread_exact(bf, b, cnt * 2, done * 2) != 0) { st = -1; break; }
        for (size_t i = 0; i < cnt; ++i) { c[i * 2] = s[i]; c[i * 2 + 1] = b[i]; }
        if (pwrite_exact(out, c, cnt * 4, *mc + done * 4) != 0) { st = -1; break; }
        done += cnt;
    }
    free(s); free(b); free(c); close(sf); close(bf);
    if (st == 0) *mc += groups * 4;
    return st;
}

/* Q8 delta_input: codes.i8 verbatim + fp16 scale/bias interleaved. */
static int put_q8_input(const char *q8, int layer, int out,
                        uint64_t *qc, uint64_t *mc) {
    uint64_t code_bytes = (uint64_t)DELTA_INPUT_ROWS * QWEN38_HIDDEN_SIZE;
    uint64_t groups = (uint64_t)DELTA_INPUT_ROWS *
                      (QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE);
    int cf = open_sized(q8, layer, "delta_input_q8", "_codes.i8", code_bytes);
    if (cf < 0) return -1;
    int st = copy_fd(cf, out, code_bytes, *qc); close(cf);
    if (st != 0) return -1;
    *qc += code_bytes;
    uint64_t plane = groups * 2;
    int sf = open_sized(q8, layer, "delta_input_q8", "_scale.f16", plane);
    if (sf < 0) return -1;
    int bf = open_sized(q8, layer, "delta_input_q8", "_bias.f16", plane);
    if (bf < 0) { close(sf); return -1; }
    enum { VPC = 1 << 20 };
    uint16_t *s = malloc(VPC * 2), *b = malloc(VPC * 2), *c = malloc(VPC * 4);
    st = (s && b && c) ? 0 : -1;
    uint64_t done = 0;
    while (done < groups && st == 0) {
        size_t cnt = (size_t)(groups - done); if (cnt > VPC) cnt = VPC;
        if (pread_exact(sf, s, cnt * 2, done * 2) != 0 ||
            pread_exact(bf, b, cnt * 2, done * 2) != 0) { st = -1; break; }
        for (size_t i = 0; i < cnt; ++i) { c[i * 2] = s[i]; c[i * 2 + 1] = b[i]; }
        if (pwrite_exact(out, c, cnt * 4, *mc + done * 4) != 0) { st = -1; break; }
        done += cnt;
    }
    free(s); free(b); free(c); close(sf); close(bf);
    if (st == 0) *mc += groups * 4;
    return st;
}

/* Pass-through fp32 vector plane copied verbatim to a constants slot. */
static int put_vec(const char *q4, int layer, const char *base,
                   int out, uint64_t out_off, uint64_t values) {
    int fd = open_sized(q4, layer, base, ".f32", values * 4);
    if (fd < 0) return -1;
    int st = copy_fd(fd, out, values * 4, out_off);
    close(fd);
    return st;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s Q4_DIR Q8_DIR OUTPUT.q38delta LAYER\n", argv[0]);
        return 2;
    }
    const char *q4 = argv[1], *q8 = argv[2], *out_path = argv[3];
    char *end = NULL; errno = 0;
    unsigned long L = strtoul(argv[4], &end, 10);
    if (errno || end == argv[4] || *end || L >= 64 || L % 4 == 3) {
        fprintf(stderr, "layer must be a DeltaNet layer (0..63, layer%%4!=3)\n");
        return 2;
    }
    int layer = (int)L;

    uint64_t gpr = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    uint64_t weight_bytes = (uint64_t)QWEN38_MLP_SIZE * QWEN38_HIDDEN_SIZE / 2;
    uint64_t meta_plane = (uint64_t)QWEN38_MLP_SIZE * gpr * 2;
    uint64_t di_groups = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    uint64_t dqkv = (uint64_t)DELTA_QKV_ROWS * QWEN38_HIDDEN_SIZE / 2;
    uint64_t dz = (uint64_t)DELTA_Z_ROWS * QWEN38_HIDDEN_SIZE / 2;
    uint64_t dsc = (uint64_t)DELTA_SCALAR_ROWS * QWEN38_HIDDEN_SIZE / 2;
    uint64_t do_groups = DELTA_OUTPUT_INPUTS / QWEN38_Q4_GROUP_SIZE;
    uint64_t do_weight = (uint64_t)QWEN38_HIDDEN_SIZE * DELTA_OUTPUT_INPUTS / 2;

    qwen38_m3_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN38_M3_IMAGE_MAGIC, sizeof(h.magic));
    h.version = QWEN38_M3_IMAGE_VERSION;
    h.header_bytes = QWEN38_M3_IMAGE_HEADER_BYTES;
    h.hidden_size = QWEN38_HIDDEN_SIZE;
    h.rows = QWEN38_MLP_SIZE;
    h.group_size = QWEN38_Q4_GROUP_SIZE;
    h.down_rows = QWEN38_HIDDEN_SIZE;
    h.down_groups_per_row = QWEN38_MLP_SIZE / QWEN38_Q4_GROUP_SIZE;
    h.gate_quants_offset = QWEN38_M3_IMAGE_HEADER_BYTES;
    h.gate_quants_bytes = weight_bytes;
    h.gate_metadata_offset = h.gate_quants_offset + weight_bytes;
    h.gate_metadata_bytes = meta_plane * 2;
    h.up_quants_offset = h.gate_metadata_offset + h.gate_metadata_bytes;
    h.up_quants_bytes = weight_bytes;
    h.up_metadata_offset = h.up_quants_offset + weight_bytes;
    h.up_metadata_bytes = meta_plane * 2;
    h.down_quants_offset = h.up_metadata_offset + h.up_metadata_bytes;
    h.down_quants_bytes = weight_bytes;
    h.down_metadata_offset = h.down_quants_offset + weight_bytes;
    h.down_metadata_bytes = meta_plane * 2;
    h.layer_index = (uint32_t)layer;
    h.delta_input_rows = DELTA_INPUT_ROWS;
    h.delta_input_groups_per_row = di_groups;
    h.delta_output_rows = QWEN38_HIDDEN_SIZE;
    h.delta_output_groups_per_row = do_groups;
    h.input_norm_constants_index = 0;
    h.post_norm_constants_index = QWEN38_HIDDEN_SIZE;
    h.conv_constants_index = 2 * QWEN38_HIDDEN_SIZE;
    h.a_log_constants_index = h.conv_constants_index + DELTA_CONV_VALUES;
    h.dt_bias_constants_index = h.a_log_constants_index + DELTA_SCALAR_ROWS;
    h.recurrent_norm_constants_index = h.dt_bias_constants_index + DELTA_SCALAR_ROWS;
    h.constants_f32_count =
        (uint32_t)(h.recurrent_norm_constants_index + QWEN38_DELTA_HEAD_SIZE);
    h.constants_offset = align_page(h.down_metadata_offset + h.down_metadata_bytes);
    h.constants_bytes = align_page((uint64_t)h.constants_f32_count * sizeof(float));
    h.delta_input_quants_offset = h.constants_offset + h.constants_bytes;
    h.delta_input_precision = 1;
    h.delta_output_precision = 0;
    h.delta_input_quants_bytes = 2 * (dqkv + dz + 2 * dsc);
    h.delta_input_metadata_offset = h.delta_input_quants_offset + h.delta_input_quants_bytes;
    uint64_t di_meta_payload = (uint64_t)DELTA_INPUT_ROWS * di_groups * 2 * 2;
    h.delta_input_metadata_bytes = align_page(di_meta_payload);
    h.delta_output_quants_offset = h.delta_input_metadata_offset + h.delta_input_metadata_bytes;
    h.delta_output_quants_bytes = do_weight;
    h.delta_output_metadata_offset = h.delta_output_quants_offset + h.delta_output_quants_bytes;
    h.delta_output_metadata_bytes = (uint64_t)QWEN38_HIDDEN_SIZE * do_groups * 2 * 2;
    /* source_sha256 / mlp_source_sha256 left zeroed (folder-only build). */
    /* reference blocks left zeroed (no mlx MLP source to compute them). */

    int out = open(out_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out < 0) { fprintf(stderr, "create %s: %s\n", out_path, strerror(errno)); return 5; }

    uint64_t qc = h.delta_input_quants_offset;
    uint64_t mc = h.delta_input_metadata_offset;
    uint64_t goff = h.gate_quants_offset, gm = h.gate_metadata_offset;
    uint64_t uoff = h.up_quants_offset, um = h.up_metadata_offset;
    uint64_t doff = h.down_quants_offset, dm = h.down_metadata_offset;
    uint64_t ooff = h.delta_output_quants_offset, om = h.delta_output_metadata_offset;

    int failed =
        pwrite_exact(out, &h, sizeof(h), 0) != 0 ||
        put_q4(q4, layer, "mlp.gate_proj", out, &goff, &gm, QWEN38_MLP_SIZE, QWEN38_HIDDEN_SIZE) != 0 ||
        put_q4(q4, layer, "mlp.up_proj", out, &uoff, &um, QWEN38_MLP_SIZE, QWEN38_HIDDEN_SIZE) != 0 ||
        put_q4(q4, layer, "mlp.down_proj", out, &doff, &dm, QWEN38_HIDDEN_SIZE, QWEN38_MLP_SIZE) != 0 ||
        /* pass-through constants (fp32) */
        put_vec(q4, layer, "input_layernorm", out,
                h.constants_offset + h.input_norm_constants_index * 4, QWEN38_HIDDEN_SIZE) != 0 ||
        put_vec(q4, layer, "post_attention_layernorm", out,
                h.constants_offset + h.post_norm_constants_index * 4, QWEN38_HIDDEN_SIZE) != 0 ||
        put_vec(q4, layer, "linear_attn__conv1d", out,
                h.constants_offset + h.conv_constants_index * 4, DELTA_CONV_VALUES) != 0 ||
        put_vec(q4, layer, "linear_attn__A_log", out,
                h.constants_offset + h.a_log_constants_index * 4, DELTA_SCALAR_ROWS) != 0 ||
        put_vec(q4, layer, "linear_attn__dt_bias", out,
                h.constants_offset + h.dt_bias_constants_index * 4, DELTA_SCALAR_ROWS) != 0 ||
        put_vec(q4, layer, "linear_attn__norm", out,
                h.constants_offset + h.recurrent_norm_constants_index * 4, QWEN38_DELTA_HEAD_SIZE) != 0 ||
        /* delta in_proj from Q8, out_proj from Q4 */
        put_q8_input(q8, layer, out, &qc, &mc) != 0 ||
        put_q4(q4, layer, "linear_attn.out_proj", out, &ooff, &om, QWEN38_HIDDEN_SIZE, DELTA_OUTPUT_INPUTS) != 0 ||
        /* cursor self-checks: each segment exactly filled */
        qc != h.delta_input_quants_offset + h.delta_input_quants_bytes ||
        mc != h.delta_input_metadata_offset + di_meta_payload ||
        goff != h.gate_quants_offset + h.gate_quants_bytes ||
        ooff != h.delta_output_quants_offset + h.delta_output_quants_bytes ||
        om != h.delta_output_metadata_offset + h.delta_output_metadata_bytes ||
        ftruncate(out, (off_t)(h.delta_output_metadata_offset + h.delta_output_metadata_bytes)) != 0 ||
        fsync(out) != 0;

    close(out);
    if (failed) { fprintf(stderr, "packing failed: %s\n", strerror(errno)); unlink(out_path); return 8; }
    printf("{\"output\":\"%s\",\"layer\":%d,\"bytes\":%" PRIu64 "}\n",
           out_path, layer, h.delta_output_metadata_offset + h.delta_output_metadata_bytes);
    return 0;
}