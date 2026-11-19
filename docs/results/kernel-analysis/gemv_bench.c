// The ggml kernel copies below come from llama.cpp 8ca5a80: Copyright (c) 2023-2026 The ggml authors,
// MIT License (see LICENSE-llama.cpp.txt).
// Single-token (GEMV) kernel comparison on Apple M1 Max.
//   kai      : pinned KleidiAI kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod (unchanged asm)
//   acc1/4/8 : same packed data and arithmetic, with 1, 4 or 8 independent int32 accumulators
//   ggml     : copy of ggml_gemv_q8_0_4x4_q8_0 (llama.cpp 8ca5a80, MIT) on equivalent Q8_0x4 data
// Integer accumulation is associative, so acc1/4/8 must match kai bit-for-bit.
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"

static void *xalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n + 64)) { perror("alloc"); exit(1); } return p; }

static double now_ns(void) { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// ---- KleidiAI-format variants -------------------------------------------------
static size_t rhs_stride(size_t k) { return 4 * k + 48; }  // 4 cols: int8 values + rsum + scale + bias
static size_t lhs_stride(size_t k) { return k + 8; }       // int8 values + zero point + scale

#define EPILOGUE(acc, r)                                                             \
    do {                                                                             \
        int32x4_t rsum = vld1q_s32((const int32_t *) (r));                           \
        float32x4_t rscale = vld1q_f32((const float *) ((r) + 16));                  \
        float32x4_t bias = vld1q_f32((const float *) ((r) + 32));                    \
        acc = vmlaq_n_s32(acc, rsum, lzp);                                           \
        float32x4_t sc = vmulq_f32(rscale, vdupq_n_f32(lscale));                     \
        float32x4_t o = vaddq_f32(vmulq_f32(vcvtq_f32_s32(acc), sc), bias);          \
        o = vminq_f32(vmaxq_f32(o, vdupq_n_f32(-3.4e38f)), vdupq_n_f32(3.4e38f));    \
        vst1q_f32(dst + 4 * g, o);                                                   \
    } while (0)

static void gemv_acc1(size_t n, size_t k, const int8_t *lhs, const uint8_t *rhs, float *dst) {
    const size_t nb = k / 32;
    const int32_t lzp = *(const int32_t *) (lhs + k);
    const float lscale = *(const float *) (lhs + k + 4);
    for (size_t g = 0; g < n / 4; g++) {
        const uint8_t *r = rhs + g * rhs_stride(k);
        int32x4_t a = vdupq_n_s32(0);
        for (size_t b = 0; b < nb; b++, r += 128) {
            int8x16_t la = vld1q_s8(lhs + 32 * b), lb = vld1q_s8(lhs + 32 * b + 16);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 0), la, 0);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 16), la, 1);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 32), la, 2);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 48), la, 3);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 64), lb, 0);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 80), lb, 1);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 96), lb, 2);
            a = vdotq_laneq_s32(a, vld1q_s8((const int8_t *) r + 112), lb, 3);
        }
        EPILOGUE(a, r);
    }
}

static void gemv_acc4(size_t n, size_t k, const int8_t *lhs, const uint8_t *rhs, float *dst) {
    const size_t nb = k / 32;
    const int32_t lzp = *(const int32_t *) (lhs + k);
    const float lscale = *(const float *) (lhs + k + 4);
    for (size_t g = 0; g < n / 4; g++) {
        const uint8_t *r = rhs + g * rhs_stride(k);
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
        for (size_t b = 0; b < nb; b++, r += 128) {
            int8x16_t la = vld1q_s8(lhs + 32 * b), lb = vld1q_s8(lhs + 32 * b + 16);
            a0 = vdotq_laneq_s32(a0, vld1q_s8((const int8_t *) r + 0), la, 0);
            a1 = vdotq_laneq_s32(a1, vld1q_s8((const int8_t *) r + 16), la, 1);
            a2 = vdotq_laneq_s32(a2, vld1q_s8((const int8_t *) r + 32), la, 2);
            a3 = vdotq_laneq_s32(a3, vld1q_s8((const int8_t *) r + 48), la, 3);
            a0 = vdotq_laneq_s32(a0, vld1q_s8((const int8_t *) r + 64), lb, 0);
            a1 = vdotq_laneq_s32(a1, vld1q_s8((const int8_t *) r + 80), lb, 1);
            a2 = vdotq_laneq_s32(a2, vld1q_s8((const int8_t *) r + 96), lb, 2);
            a3 = vdotq_laneq_s32(a3, vld1q_s8((const int8_t *) r + 112), lb, 3);
        }
        int32x4_t a = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
        EPILOGUE(a, r);
    }
}

static void gemv_acc8(size_t n, size_t k, const int8_t *lhs, const uint8_t *rhs, float *dst) {
    const size_t nb = k / 32;
    const int32_t lzp = *(const int32_t *) (lhs + k);
    const float lscale = *(const float *) (lhs + k + 4);
    for (size_t g = 0; g < n / 4; g++) {
        const uint8_t *r = rhs + g * rhs_stride(k);
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (size_t b = 0; b < nb; b++, r += 128) {
            int8x16_t la = vld1q_s8(lhs + 32 * b), lb = vld1q_s8(lhs + 32 * b + 16);
            a0 = vdotq_laneq_s32(a0, vld1q_s8((const int8_t *) r + 0), la, 0);
            a1 = vdotq_laneq_s32(a1, vld1q_s8((const int8_t *) r + 16), la, 1);
            a2 = vdotq_laneq_s32(a2, vld1q_s8((const int8_t *) r + 32), la, 2);
            a3 = vdotq_laneq_s32(a3, vld1q_s8((const int8_t *) r + 48), la, 3);
            a4 = vdotq_laneq_s32(a4, vld1q_s8((const int8_t *) r + 64), lb, 0);
            a5 = vdotq_laneq_s32(a5, vld1q_s8((const int8_t *) r + 80), lb, 1);
            a6 = vdotq_laneq_s32(a6, vld1q_s8((const int8_t *) r + 96), lb, 2);
            a7 = vdotq_laneq_s32(a7, vld1q_s8((const int8_t *) r + 112), lb, 3);
        }
        int32x4_t a = vaddq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)),
                                vaddq_s32(vaddq_s32(a4, a5), vaddq_s32(a6, a7)));
        EPILOGUE(a, r);
    }
}

static void gemv_kai(size_t n, size_t k, const int8_t *lhs, const uint8_t *rhs, float *dst) {
    kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod(1, n, k, lhs, rhs, dst, n * sizeof(float), sizeof(float),
                                                                     -3.4e38f, 3.4e38f);
}

// ---- ggml Q8_0x4 reference ------------------------------------------------------
typedef struct { __fp16 d[4]; int8_t qs[128]; } block_q8_0x4;  // 136 bytes per 4 cols x 32 k
typedef struct { __fp16 d; int8_t qs[32]; } block_q8_0;         // 34 bytes per 32 activations

static void gemv_ggml(size_t n, size_t k, const int8_t *lhs, const uint8_t *rhs, float *dst) {
    const int nb = (int) (k / 32);
    const block_q8_0x4 *b_ptr = (const block_q8_0x4 *) rhs;
    float *s = dst;
    for (size_t c = 0; c < n; c += 4) {
        const block_q8_0 *a_ptr = (const block_q8_0 *) lhs;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16x4_t b_low = vld1q_s8_x4((const int8_t *) b_ptr->qs);
            int8x16x4_t b_high = vld1q_s8_x4((const int8_t *) b_ptr->qs + 64);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);
            int8x16x2_t a = vld1q_s8_x2(a_ptr->qs);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);
            int32x4_t ret = vdupq_n_s32(0);
            ret = vdotq_laneq_s32(ret, b_low.val[0], a.val[0], 0);
            ret = vdotq_laneq_s32(ret, b_low.val[1], a.val[0], 1);
            ret = vdotq_laneq_s32(ret, b_low.val[2], a.val[0], 2);
            ret = vdotq_laneq_s32(ret, b_low.val[3], a.val[0], 3);
            ret = vdotq_laneq_s32(ret, b_high.val[0], a.val[1], 0);
            ret = vdotq_laneq_s32(ret, b_high.val[1], a.val[1], 1);
            ret = vdotq_laneq_s32(ret, b_high.val[2], a.val[1], 2);
            ret = vdotq_laneq_s32(ret, b_high.val[3], a.val[1], 3);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += 4;
    }
}

// ---- threading harness ----------------------------------------------------------
typedef void (*gemv_fn)(size_t, size_t, const int8_t *, const uint8_t *, float *);
typedef struct {
    gemv_fn fn; size_t n, k, group_bytes; const int8_t *lhs; const uint8_t *rhs; float *dst; int reps;
} job_t;

static void *worker(void *p) {
    job_t *j = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int r = 0; r < j->reps; r++) j->fn(j->n, j->k, j->lhs, j->rhs, j->dst);
    return NULL;
}

static double run(gemv_fn fn, int threads, size_t n, size_t k, size_t group_bytes, const int8_t *lhs, const uint8_t *rhs,
                  float *dst, int reps) {
    pthread_t t[16];
    job_t jobs[16];
    size_t groups = n / 4, per = groups / threads;
    for (int i = 0; i < threads; i++) {
        size_t g0 = i * per, g1 = (i == threads - 1) ? groups : g0 + per;
        jobs[i] = (job_t){fn, (g1 - g0) * 4, k, group_bytes, lhs, rhs + g0 * group_bytes, dst + g0 * 4, reps};
    }
    double t0 = now_ns();
    for (int i = 0; i < threads; i++) pthread_create(&t[i], NULL, worker, &jobs[i]);
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    return now_ns() - t0;
}

int main(int argc, char **argv) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const size_t k = 4096;
    srand(42);

    size_t n_big = 98304;  // ~400 MB of weights: larger than L2 and SLC, so streamed from memory
    size_t kai_bytes = (n_big / 4) * rhs_stride(k), ggml_bytes = (n_big / 4) * (k / 32) * sizeof(block_q8_0x4);
    uint8_t *rhs = xalloc(kai_bytes), *rhs_g = xalloc(ggml_bytes);
    int8_t *lhs = xalloc(lhs_stride(k)), *lhs_g = xalloc((k / 32) * sizeof(block_q8_0) + 64);
    float *d0 = xalloc(n_big * 4), *d1 = xalloc(n_big * 4);
    for (size_t i = 0; i < kai_bytes; i++) rhs[i] = (uint8_t) rand();
    for (size_t g = 0; g < n_big / 4; g++) {  // valid float scales and bias in each group's epilogue
        uint8_t *e = rhs + g * rhs_stride(k) + 4 * k;
        for (int c = 0; c < 4; c++) {
            ((int32_t *) e)[c] = (rand() % 20001) - 10000;
            ((float *) (e + 16))[c] = 1e-3f * (1 + rand() % 1000);
            ((float *) (e + 32))[c] = 0.01f * (rand() % 100);
        }
    }
    for (size_t i = 0; i < ggml_bytes; i++) rhs_g[i] = (uint8_t) rand();
    for (size_t b = 0; b < (n_big / 4) * (k / 32); b++)
        for (int c = 0; c < 4; c++) ((block_q8_0x4 *) rhs_g)[b].d[c] = (__fp16) (1e-3f * (1 + rand() % 1000));
    for (size_t i = 0; i < k; i++) lhs[i] = (int8_t) rand();
    *(int32_t *) (lhs + k) = 7;
    *(float *) (lhs + k + 4) = 0.0123f;
    for (size_t b = 0; b < k / 32; b++) {
        block_q8_0 *a = (block_q8_0 *) lhs_g + b;
        a->d = (__fp16) 0.01f;
        for (int i = 0; i < 32; i++) a->qs[i] = (int8_t) rand();
    }

    // Bit-identity check against the unchanged KleidiAI kernel.
    gemv_fn variants[3] = {gemv_acc1, gemv_acc4, gemv_acc8};
    const char *vnames[3] = {"acc1", "acc4", "acc8"};
    gemv_kai(1024, k, lhs, rhs, d0);
    for (int v = 0; v < 3; v++) {
        variants[v](1024, k, lhs, rhs, d1);
        printf("bit-identical to kai: %s = %s\n", vnames[v], memcmp(d0, d1, 1024 * sizeof(float)) == 0 ? "yes" : "NO");
    }

    struct { const char *name; gemv_fn fn; const uint8_t *rhs; const int8_t *lhs; size_t group_bytes; double bytes_per_weight; } k_list[] = {
        {"kai (KleidiAI asm)", gemv_kai, rhs, lhs, rhs_stride(k), 1.0},
        {"acc1 (same chain)", gemv_acc1, rhs, lhs, rhs_stride(k), 1.0},
        {"acc4", gemv_acc4, rhs, lhs, rhs_stride(k), 1.0},
        {"acc8", gemv_acc8, rhs, lhs, rhs_stride(k), 1.0},
        {"ggml q8_0 4x4", gemv_ggml, rhs_g, lhs_g, (k / 32) * sizeof(block_q8_0x4), 34.0 / 32.0},
    };
    struct { const char *label; size_t n; int reps; } shapes[] = {{"cache  (1 MB)", 256, 400}, {"memory (400 MB)", n_big, 3}};
    int thread_counts[] = {1, 4, 8};

    for (int s = 0; s < 2; s++) {
        for (int ti = 0; ti < 3; ti++) {
            int th = thread_counts[ti];
            if (s == 0 && th != 1) continue;  // cache-resident case is a single-core kernel test
            printf("\n%s, %d thread(s): weight GB/s (higher is faster)\n", shapes[s].label, th);
            for (int v = 0; v < 5; v++) {
                double best = 1e30;
                for (int trial = 0; trial < 3; trial++) {
                    double t = run(k_list[v].fn, th, shapes[s].n, k, k_list[v].group_bytes, k_list[v].lhs, k_list[v].rhs, d1,
                                   shapes[s].reps);
                    if (t < best) best = t;
                }
                double bytes = (double) shapes[s].n * k * k_list[v].bytes_per_weight * shapes[s].reps;
                printf("  %-20s %7.1f GB/s  (%.1f weight bytes per ns per thread)\n", k_list[v].name, bytes / best,
                       bytes / best / th);
            }
        }
    }
    return 0;
}
