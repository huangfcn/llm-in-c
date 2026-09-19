/* Folder-only global packer.
 *
 *   qwen38-m3-global-pack Q4_DIR OUTPUT.q38global
 *
 * embed_tokens + lm_head Q4 planes from Q4_DIR/global/, model.norm fp32 plane
 * from Q4_DIR/global/model__norm.f32. No safetensors, no SHA.
 */
#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_global_image.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
static int open_sized(const char *dir, const char *base, const char *suffix,
                      uint64_t expect) {
    char path[512];
    snprintf(path, sizeof(path), "%s/global/%s%s", dir, base, suffix);
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
static int put_codes(const char *q4, const char *base, int out,
                     uint64_t out_off, uint64_t code_bytes) {
    int fd = open_sized(q4, base, "_codes.u8", code_bytes);
    if (fd < 0) return -1;
    int st = copy_fd(fd, out, code_bytes, out_off);
    close(fd);
    return st;
}
static int put_meta(const char *q4, const char *base, int out,
                    uint64_t out_off, uint64_t groups) {
    uint64_t plane = groups * 2;
    int sf = open_sized(q4, base, "_scale.f16", plane);
    if (sf < 0) return -1;
    int bf = open_sized(q4, base, "_bias.f16", plane);
    if (bf < 0) { close(sf); return -1; }
    enum { VPC = 1 << 20 };
    uint16_t *s = malloc(VPC * 2), *b = malloc(VPC * 2), *c = malloc(VPC * 4);
    int st = (s && b && c) ? 0 : -1;
    uint64_t done = 0;
    while (done < groups && st == 0) {
        size_t cnt = (size_t)(groups - done); if (cnt > VPC) cnt = VPC;
        if (pread_exact(sf, s, cnt * 2, done * 2) != 0 ||
            pread_exact(bf, b, cnt * 2, done * 2) != 0) { st = -1; break; }
        for (size_t i = 0; i < cnt; ++i) { c[i * 2] = s[i]; c[i * 2 + 1] = b[i]; }
        if (pwrite_exact(out, c, cnt * 4, out_off + done * 4) != 0) { st = -1; break; }
        done += cnt;
    }
    free(s); free(b); free(c); close(sf); close(bf);
    return st;
}

static int put_vec(const char *q4, const char *base, int out,
                   uint64_t out_off, uint64_t values) {
    int fd = open_sized(q4, base, ".f32", values * 4);
    if (fd < 0) return -1;
    int st = copy_fd(fd, out, values * 4, out_off);
    close(fd);
    return st;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s Q4_DIR OUTPUT.q38global\n", argv[0]);
        return 2;
    }
    const char *q4 = argv[1], *out_path = argv[2];

    const uint64_t quant_bytes = (uint64_t)QWEN38_VOCAB_SIZE * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t meta_plane = (uint64_t)QWEN38_VOCAB_SIZE * 80 * 2;
    const uint64_t vocab_groups = (uint64_t)QWEN38_VOCAB_SIZE * 80;

    qwen38_m3_global_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN38_M3_GLOBAL_IMAGE_MAGIC, 8);
    h.version = QWEN38_M3_GLOBAL_IMAGE_VERSION;
    h.header_bytes = QWEN38_M3_GLOBAL_HEADER_BYTES;
    h.vocab_size = QWEN38_VOCAB_SIZE;
    h.hidden_size = QWEN38_HIDDEN_SIZE;
    h.group_size = QWEN38_Q4_GROUP_SIZE;
    h.constants_f32_count = QWEN38_HIDDEN_SIZE;
    uint64_t offset = QWEN38_M3_GLOBAL_HEADER_BYTES;
#define SEG(field, bytes) do { h.field##_offset = offset; h.field##_bytes = (bytes); offset += (bytes); } while (0)
    SEG(embedding_quants, quant_bytes);
    SEG(embedding_metadata, meta_plane * 2);
    SEG(lm_head_quants, quant_bytes);
    SEG(lm_head_metadata, meta_plane * 2);
    SEG(constants, QWEN38_HIDDEN_SIZE * sizeof(float));
#undef SEG

    int out = open(out_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out < 0) { fprintf(stderr, "create %s: %s\n", out_path, strerror(errno)); return 5; }

    int failed =
        pwrite_exact(out, &h, sizeof(h), 0) != 0 ||
        put_codes(q4, "embed_tokens", out, h.embedding_quants_offset, quant_bytes) != 0 ||
        put_meta(q4, "embed_tokens", out, h.embedding_metadata_offset, vocab_groups) != 0 ||
        put_codes(q4, "lm_head", out, h.lm_head_quants_offset, quant_bytes) != 0 ||
        put_meta(q4, "lm_head", out, h.lm_head_metadata_offset, vocab_groups) != 0 ||
        /* model.norm fp32 pass-through */
        put_vec(q4, "model__norm", out, h.constants_offset, QWEN38_HIDDEN_SIZE) != 0 ||
        ftruncate(out, (off_t)offset) != 0 || fsync(out) != 0;

    close(out);
    if (failed) { fprintf(stderr, "global packing failed: %s\n", strerror(errno)); unlink(out_path); return 6; }
    printf("{\"output\":\"%s\",\"bytes\":%" PRIu64 "}\n", out_path, offset);
    return 0;
}