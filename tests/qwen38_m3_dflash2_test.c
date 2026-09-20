#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3_decode.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t argmax_token(const float *logits, size_t count) {
    uint32_t best = 0;
    float value = -FLT_MAX;
    for (size_t i = 0; i < count; ++i) {
        if (logits[i] > value) {
            value = logits[i];
            best = (uint32_t)i;
        }
    }
    return best;
}

static void make_prompt(uint32_t tokens[64]) {
    /* Any in-vocabulary IDs are sufficient for a runtime/module smoke test.
     * Keep them deterministic and away from the special-token tail. */
    for (uint32_t i = 0; i < 64; ++i)
        tokens[i] = 1000u + ((i * 7919u + 17u) % 200000u);
}

static int prime_model(qwen38_m3_model *model, uint32_t *pending,
                       char *error, size_t error_cap) {
    uint32_t prompt[64];
    make_prompt(prompt);
    qwen38_m3_prefill_result prefill = {0};
    if (qwen38_m3_model_prefill(model, prompt, 63, 0, &prefill,
                                error, error_cap) != 0)
        return 1;

    qwen38_m3_decode_result result = {0};
    const float *logits = NULL;
    size_t logit_count = 0;
    if (qwen38_m3_model_forward(model, prompt[63], 63, &result,
                                &logits, &logit_count,
                                error, error_cap) != 0)
        return 2;
    if (logits == NULL || logit_count == 0) {
        snprintf(error, error_cap, "final forward returned no logits");
        return 3;
    }
    *pending = argmax_token(logits, logit_count);
    return 0;
}

static void fail(const char *stage, int status, const char *error) {
    fprintf(stderr, "FAIL %-12s status=%d: %s\n",
            stage, status, error != NULL && *error ? error : "unknown error");
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s MODEL_DIR METALLIB [CONTEXT]\n"
                "example: %s ../models/qwen38-runtime-q8 "
                "build/qwen38-m3-q4.metallib 4096\n",
                argv[0], argv[0]);
        return 2;
    }
    const char *model_dir = argv[1];
    const char *metallib = argv[2];
    uint32_t context = argc == 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096;
    if (context < 128) context = 128;

    /* Must be set before model_open(), because prefill4/prefill8 are built
     * during normal runtime initialization and need the DFlash pipelines. */
    setenv("QWEN38_DFLASH2", "1", 1);
    setenv("QWEN38_MTP", "0", 1);

    char error[2048] = {0};
    qwen38_m3_model *model = qwen38_m3_model_open(
        model_dir, metallib, context, error, sizeof(error));
    if (model == NULL) {
        fail("model-open", 1, error);
        return 3;
    }

    char dflash_path[2048];
    snprintf(dflash_path, sizeof(dflash_path), "%s/dflash2.q38df2", model_dir);
    int status = qwen38_m3_model_dflash2_open(
        model, dflash_path, error, sizeof(error));
    if (status != 0) {
        fail("dflash-open", status, error);
        qwen38_m3_model_close(model);
        return 4;
    }

    error[0] = '\0';
    status = qwen38_m3_model_dflash2_test_validate(model, error, sizeof(error));
    if (status != 0) {
        fail("validate", status, error);
        qwen38_m3_model_close(model);
        return 5;
    }
    fprintf(stderr, "PASS validate     DFlash + target verifier resources ready\n");

    uint32_t pending = 0;
    error[0] = '\0';
    status = prime_model(model, &pending, error, sizeof(error));
    if (status != 0) {
        fail("prime", status, error);
        qwen38_m3_model_close(model);
        return 6;
    }
    fprintf(stderr, "PASS prime        64-token target context, pending=%u\n", pending);

    uint32_t drafts[7] = {0};
    uint32_t draft_count = 0;
    error[0] = '\0';
    status = qwen38_m3_model_dflash2_test_propose(
        model, pending, 64, 8, drafts, &draft_count, error, sizeof(error));
    if (status != 0) {
        fail("proposal", status, error);
        qwen38_m3_model_close(model);
        return 7;
    }
    fprintf(stderr, "PASS proposal     %u drafts:", draft_count);
    for (uint32_t i = 0; i < draft_count; ++i) fprintf(stderr, " %u", drafts[i]);
    fputc('\n', stderr);

    uint32_t verify_tokens[8] = {0};
    verify_tokens[0] = pending;
    for (uint32_t i = 0; i < draft_count; ++i)
        verify_tokens[i + 1] = drafts[i];
    error[0] = '\0';
    status = qwen38_m3_model_dflash2_test_verify(
        model, verify_tokens, draft_count + 1, 64, error, sizeof(error));
    if (status != 0) {
        fail("target-verify", status, error);
        qwen38_m3_model_close(model);
        return 8;
    }
    fprintf(stderr, "PASS target-verify batch=%u\n", draft_count + 1);

    /* Re-prime from a clean target/DFlash state before testing the complete
     * speculative path, because the verify-only diagnostic intentionally ran
     * speculative target rows. */
    qwen38_m3_model_reset(model);
    error[0] = '\0';
    status = prime_model(model, &pending, error, sizeof(error));
    if (status != 0) {
        fail("re-prime", status, error);
        qwen38_m3_model_close(model);
        return 9;
    }

    uint32_t position = 64;
    uint32_t emitted[8] = {0};
    uint32_t emitted_count = 0;
    int accepted = 0;
    uint64_t rng = 0x243f6a8885a308d3ULL;
    error[0] = '\0';
    status = qwen38_m3_model_dflash2_sample_step(
        model, &pending, &position, emitted, &emitted_count, &accepted,
        1.0f, 20, 0.95f, &rng, error, sizeof(error));
    if (status != 0) {
        fail("full-step", status, error);
        qwen38_m3_model_close(model);
        return 10;
    }
    fprintf(stderr,
            "PASS full-step    emitted=%u accepted_drafts=%d next=%u position=%u\n",
            emitted_count, accepted, pending, position);

    qwen38_m3_model_close(model);
    fprintf(stderr, "PASS all DFlash2 module tests\n");
    return 0;
}
