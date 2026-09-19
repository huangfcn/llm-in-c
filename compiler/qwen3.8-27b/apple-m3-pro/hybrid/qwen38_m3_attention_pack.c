/* Folder-only attention layer packer.
 *
 *   qwen38-m3-attention-pack Q4_DIR OUTPUT.q38att LAYER
 *
 * Every byte from Q4_DIR/layer-NN/ (produced by qwen38_q4_quantize_all.py):
 *   mlp.{gate,up,down}_proj, self_attn.{q,k,v,o}_proj  (codes.u8/scale.f16/bias.f16)
 *   pass-through fp32: input_layernorm, post_attention_layernorm,
 *                      self_attn__q_norm, self_attn__k_norm
 * q/k/v are CONCATENATED (q||k||v order) into the fused attention_input plane.
 * No safetensors, no SHA. Integrity = exact plane size checks + cursor checks.
 */
#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_attention_image.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t align_page(uint64_t v) {
    return (v + QWEN38_M3_ATTENTION_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN38_M3_ATTENTION_HEADER_BYTES - 1);
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
static int open_sized(const char *dir, int layer, const char *base,
                      const char *suffix, uint64_t expect) {
    char path[512];
    snprintf(path, sizeof(path), "%s/layer-%02d/%s%s", dir, layer, base, suffix);
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
/* Append a Q4 tensor's codes at *qc and interleaved fp16 meta at *mc,
 * advancing both. */
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
static int put_vec(const char *q4, int layer, const char *base,
                   int out, uint64_t out_off, uint64_t values) {
    int fd = open_sized(q4, layer, base, ".f32", values * 4);
    if (fd < 0) return -1;
    int st = copy_fd(fd, out, values * 4, out_off);
    close(fd);
    return st;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s Q4_DIR OUTPUT.q38att LAYER\n", argv[0]);
        return 2;
    }
    const char *q4 = argv[1], *out_path = argv[2];
    char *end = NULL; errno = 0;
    unsigned long L = strtoul(argv[3], &end, 10);
    if (errno || end == argv[3] || *end || L >= 64 || L % 4 != 3) {
        fprintf(stderr, "layer must be an attention layer (layer%%4==3)\n");
        return 2;
    }
    int layer = (int)L;

    const uint64_t hidden_groups = 80, mlp_groups = 272;
    const uint64_t mlp_weight = (uint64_t)QWEN38_MLP_SIZE * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t mlp_meta = (uint64_t)QWEN38_MLP_SIZE * hidden_groups * 2;
    const uint64_t q_weight = (uint64_t)QWEN38_ATTENTION_Q_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t kv_weight = (uint64_t)QWEN38_ATTENTION_K_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t o_weight = (uint64_t)QWEN38_HIDDEN_SIZE *
        (QWEN38_ATTENTION_HEADS * QWEN38_ATTENTION_HEAD_SIZE) / 2;

    qwen38_m3_attention_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN38_M3_ATTENTION_IMAGE_MAGIC, 8);
    h.version = QWEN38_M3_ATTENTION_IMAGE_VERSION;
    h.header_bytes = QWEN38_M3_ATTENTION_HEADER_BYTES;
    h.layer_index = (uint32_t)layer;
    h.hidden_size = QWEN38_HIDDEN_SIZE;
    h.intermediate_size = QWEN38_MLP_SIZE;
    h.group_size = QWEN38_Q4_GROUP_SIZE;
    h.q_heads = QWEN38_ATTENTION_HEADS;
    h.kv_heads = QWEN38_ATTENTION_KV_HEADS;
    h.head_size = QWEN38_ATTENTION_HEAD_SIZE;
    h.rotary_size = QWEN38_ATTENTION_ROTARY_SIZE;
    h.input_rows = QWEN38_ATTENTION_INPUT_ROWS;
    h.input_groups_per_row = hidden_groups;
    h.output_rows = QWEN38_HIDDEN_SIZE;
    h.output_groups_per_row = 96;
    uint64_t offset = QWEN38_M3_ATTENTION_HEADER_BYTES;
#define SEG(field, bytes) do { h.field##_offset = offset; h.field##_bytes = (bytes); offset += (bytes); } while (0)
    SEG(gate_quants, mlp_weight);
    SEG(gate_metadata, mlp_meta * 2);
    SEG(up_quants, mlp_weight);
    SEG(up_metadata, mlp_meta * 2);
    SEG(down_quants, mlp_weight);
    SEG(down_metadata, mlp_meta * 2);
    offset = align_page(offset);
    h.input_norm_constants_index = 0;
    h.post_norm_constants_index = QWEN38_HIDDEN_SIZE;
    h.q_norm_constants_index = 2 * QWEN38_HIDDEN_SIZE;
    h.k_norm_constants_index = h.q_norm_constants_index + QWEN38_ATTENTION_HEAD_SIZE;
    h.constants_f32_count =
        (uint32_t)(h.k_norm_constants_index + QWEN38_ATTENTION_HEAD_SIZE);
    h.constants_offset = offset;
    h.constants_bytes = align_page((uint64_t)h.constants_f32_count * sizeof(float));
    offset += h.constants_bytes;
    SEG(attention_input_quants, q_weight + 2 * kv_weight);
    SEG(attention_input_metadata, (uint64_t)QWEN38_ATTENTION_INPUT_ROWS * hidden_groups * 4);
    SEG(attention_output_quants, o_weight);
    SEG(attention_output_metadata, (uint64_t)QWEN38_HIDDEN_SIZE * 96 * 4);
#undef SEG

    int out = open(out_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out < 0) { fprintf(stderr, "create %s: %s\n", out_path, strerror(errno)); return 5; }

    uint64_t goff = h.gate_quants_offset, gm = h.gate_metadata_offset;
    uint64_t uoff = h.up_quants_offset, um = h.up_metadata_offset;
    uint64_t doff = h.down_quants_offset, dm = h.down_metadata_offset;
    uint64_t iq = h.attention_input_quants_offset, im = h.attention_input_metadata_offset;
    uint64_t ooff = h.attention_output_quants_offset, om = h.attention_output_metadata_offset;

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
        put_vec(q4, layer, "self_attn__q_norm", out,
                h.constants_offset + h.q_norm_constants_index * 4, QWEN38_ATTENTION_HEAD_SIZE) != 0 ||
        put_vec(q4, layer, "self_attn__k_norm", out,
                h.constants_offset + h.k_norm_constants_index * 4, QWEN38_ATTENTION_HEAD_SIZE) != 0 ||
        /* q||k||v CONCATENATED: codes then metadata, q,k,v order */
        put_q4(q4, layer, "self_attn.q_proj", out, &iq, &im, QWEN38_ATTENTION_Q_ROWS, QWEN38_HIDDEN_SIZE) != 0 ||
        put_q4(q4, layer, "self_attn.k_proj", out, &iq, &im, QWEN38_ATTENTION_K_ROWS, QWEN38_HIDDEN_SIZE) != 0 ||
        put_q4(q4, layer, "self_attn.v_proj", out, &iq, &im, QWEN38_ATTENTION_V_ROWS, QWEN38_HIDDEN_SIZE) != 0 ||
        put_q4(q4, layer, "self_attn.o_proj", out, &ooff, &om, QWEN38_HIDDEN_SIZE, 6144) != 0 ||
        /* cursor self-checks: fused input plane exactly filled, in q,k,v order */
        iq != h.attention_input_quants_offset + h.attention_input_quants_bytes ||
        im != h.attention_input_metadata_offset + h.attention_input_metadata_bytes ||
        goff != h.gate_quants_offset + h.gate_quants_bytes ||
        ooff != h.attention_output_quants_offset + h.attention_output_quants_bytes ||
        ftruncate(out, (off_t)offset) != 0 || fsync(out) != 0;

    close(out);
    if (failed) { fprintf(stderr, "packing failed: %s\n", strerror(errno)); unlink(out_path); return 6; }
    printf("{\"output\":\"%s\",\"layer\":%d,\"bytes\":%" PRIu64 "}\n", out_path, layer, offset);
    return 0;
}