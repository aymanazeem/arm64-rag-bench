// The ggml kernel copies below come from llama.cpp 8ca5a80: Copyright (c) 2023-2026 The ggml authors,
// MIT License (see LICENSE-llama.cpp.txt).
// Prompt (GEMM) kernel comparison on Apple M1 Max, using the model's real layer shapes, 512 tokens,
// and the same column chunking both llama.cpp drivers use (4 chunks per thread, first chunk = thread id).
//   kai      : pinned KleidiAI kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod (unchanged)
//   ggml     : copy of ggml_gemm_q8_0_4x4_q8_0 (llama.cpp 8ca5a80, MIT) on Q8_0x4 data
//   ggml_int : ggml's 4x4 loop with the per-block float scaling replaced by one integer add per block
//              (scale applied once per row at the end), to isolate the cost of per-block scaling
// Timing only: values are random; scale fields are set to finite values.
#include <arm_neon.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"

static void *xalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 128, n + 128)) { perror("alloc"); exit(1); } return p; }
static double now_ns(void) { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

typedef struct { uint16_t d[4]; int8_t qs[128]; } block_q8_0x4;

static void ggml_gemm(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc) {
    const int nb = n / 32;
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / 4; x++) {
            const block_q8_0x4 *b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) sumf[m] = vdupq_n_f32(0);
            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *) a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *) b_ptr[l].d));
                int32x4_t sumi_0 = vdupq_n_s32(0), sumi_1 = vdupq_n_s32(0), sumi_2 = vdupq_n_s32(0), sumi_3 = vdupq_n_s32(0);
                for (int k_group = 0; k_group < 8; k_group += 4) {
                    int8x16x4_t a = vld1q_s8_x4(a_ptr[l].qs + 16 * k_group);
                    int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * k_group);
                    for (int k = 0; k < 4; k++) {
                        sumi_0 = vdotq_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }
                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }
            for (int m = 0; m < 4; m++) vst1q_f32(s + (y * 4 + m) * bs + x * 4, sumf[m]);
        }
    }
}

static void ggml_gemm_int(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc) {
    const int nb = n / 32;
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / 4; x++) {
            const block_q8_0x4 *b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            int32x4_t isum[4];
            for (int m = 0; m < 4; m++) isum[m] = vdupq_n_s32(0);
            for (int l = 0; l < nb; l++) {
                int32x4_t sumi_0 = vdupq_n_s32(0), sumi_1 = vdupq_n_s32(0), sumi_2 = vdupq_n_s32(0), sumi_3 = vdupq_n_s32(0);
                for (int k_group = 0; k_group < 8; k_group += 4) {
                    int8x16x4_t a = vld1q_s8_x4(a_ptr[l].qs + 16 * k_group);
                    int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * k_group);
                    for (int k = 0; k < 4; k++) {
                        sumi_0 = vdotq_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }
                isum[0] = vaddq_s32(isum[0], sumi_0);
                isum[1] = vaddq_s32(isum[1], sumi_1);
                isum[2] = vaddq_s32(isum[2], sumi_2);
                isum[3] = vaddq_s32(isum[3], sumi_3);
            }
            float32x4_t sc = vcvt_f32_f16(vld1_f16((const float16_t *) b_ptr[0].d));
            for (int m = 0; m < 4; m++) vst1q_f32(s + (y * 4 + m) * bs + x * 4, vmulq_f32(vcvtq_f32_s32(isum[m]), sc));
        }
    }
}

enum { KAI, GGML, GGML_INT };
static const char *names[] = {"kai 16x4", "ggml 4x4", "ggml 4x4, no per-block scaling"};

typedef struct {
    int kind; size_t m, n, k, chunk_cols; int nchunk;
    const uint8_t *lhs, *rhs; float *dst; atomic_int *next; int ith;
} job_t;

static void run_chunk(const job_t *j, int c) {
    size_t c0 = (size_t) c * j->chunk_cols, cols = j->chunk_cols;
    if (c0 + cols > j->n) cols = j->n - c0;
    if (j->kind == KAI) {
        size_t off = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(c0, j->k);
        kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(j->m, cols, j->k, j->lhs, j->rhs + off, j->dst + c0,
                                                                          j->n * 4, 4, -3.4e38f, 3.4e38f);
    } else {
        const block_q8_0x4 *w = (const block_q8_0x4 *) j->rhs + (c0 / 4) * (j->k / 32);
        (j->kind == GGML ? ggml_gemm : ggml_gemm_int)((int) j->k, j->dst + c0, j->n, w, j->lhs, (int) j->m, (int) cols);
    }
}

static void *worker(void *p) {
    job_t *j = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int c = j->ith; c < j->nchunk; c = atomic_fetch_add(j->next, 1)) run_chunk(j, c);
    return NULL;
}

typedef struct { uint8_t *lhs, *rhs; float *dst; } bufs_t;

static bufs_t make(int kind, size_t m, size_t n, size_t k) {
    bufs_t b;
    size_t lhs_bytes, rhs_bytes;
    if (kind == KAI) {
        lhs_bytes = kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(m, k);
        rhs_bytes = kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod(n, k);
    } else {
        lhs_bytes = (m / 4) * (k / 32) * sizeof(block_q8_0x4);
        rhs_bytes = (n / 4) * (k / 32) * sizeof(block_q8_0x4);
    }
    b.lhs = xalloc(lhs_bytes); b.rhs = xalloc(rhs_bytes); b.dst = xalloc(m * n * 4);
    for (size_t i = 0; i < lhs_bytes; i++) b.lhs[i] = (uint8_t) (i * 40503u >> 8);
    for (size_t i = 0; i < rhs_bytes; i++) b.rhs[i] = (uint8_t) (i * 2654435761u >> 24);
    if (kind == KAI) {  // zero the per-row zero points, scales, sums and biases
        size_t ls = 4 * k + 32, rs = 4 * k + 48;
        for (size_t r = 0; r < m / 4; r++) memset(b.lhs + r * ls + 4 * k, 0, 32);
        for (size_t c = 0; c < n / 4; c++) memset(b.rhs + c * rs + 4 * k, 0, 48);
    } else {
        for (size_t i = 0; i < (m / 4) * (k / 32); i++) for (int q = 0; q < 4; q++) ((block_q8_0x4 *) b.lhs)[i].d[q] = 0x3C00;
        for (size_t i = 0; i < (n / 4) * (k / 32); i++) for (int q = 0; q < 4; q++) ((block_q8_0x4 *) b.rhs)[i].d[q] = 0x3C00;
    }
    return b;
}

// Best of `trials` wall times (ns) for one full matmul with `th` threads.
static double time_matmul(int kind, bufs_t *b, size_t m, size_t n, size_t k, int th, int trials) {
    double best = 1e30;
    int nchunk = th == 1 ? 1 : th * 4;
    size_t chunk_cols = ((n + nchunk - 1) / nchunk + 3) / 4 * 4;
    for (int t = 0; t < trials; t++) {
        atomic_int next = th;
        pthread_t tid[16]; job_t j[16];
        for (int i = 0; i < th; i++) j[i] = (job_t){kind, m, n, k, chunk_cols, nchunk, b->lhs, b->rhs, b->dst, &next, i};
        double t0 = now_ns();
        for (int i = 0; i < th; i++) pthread_create(&tid[i], NULL, worker, &j[i]);
        for (int i = 0; i < th; i++) pthread_join(tid[i], NULL);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const int th = argc > 1 ? atoi(argv[1]) : 4;
    const size_t m = 512;
    // Q8_0 weight shapes of Qwen3.5-9B per 512-token pass (token embedding and output head excluded).
    struct { size_t k, n; int count; const char *what; } shapes[] = {
        {4096, 12288, 64, "ffn_gate, ffn_up"},
        {12288, 4096, 32, "ffn_down"},
        {4096, 8192, 32, "attn_qkv, attn_q"},
        {4096, 4096, 56, "attn_gate, ssm_out, attn_output"},
        {4096, 1024, 16, "attn_k, attn_v"},
    };
    const int nshape = sizeof(shapes) / sizeof(shapes[0]);
    printf("threads=%d m=%zu\n", th, m);
    for (int kind = 0; kind < 3; kind++) {
        double total_ns = 0, total_mac = 0;
        printf("%s\n", names[kind]);
        for (int s = 0; s < nshape; s++) {
            bufs_t b = make(kind, m, shapes[s].n, shapes[s].k);
            time_matmul(kind, &b, m, shapes[s].n, shapes[s].k, th, 1);  // warm
            double ns = time_matmul(kind, &b, m, shapes[s].n, shapes[s].k, th, 5);
            double mac = (double) m * shapes[s].n * shapes[s].k;
            printf("  k=%5zu n=%5zu x%2d  %7.2f ms each  %6.1f GMAC/s\n", shapes[s].k, shapes[s].n, shapes[s].count, ns / 1e6, mac / ns);
            total_ns += ns * shapes[s].count; total_mac += mac * shapes[s].count;
            free(b.lhs); free(b.rhs); free(b.dst);
        }
        printf("  weight matmuls per 512-token pass: %.3f s, %.1f GMAC/s\n", total_ns / 1e9, total_mac / total_ns);
    }
    // In-cache single-core kernel speed: small problem repeated.
    printf("in-cache, 1 thread (m=64, n=256, k=1024, 200 reps)\n");
    for (int kind = 0; kind < 3; kind++) {
        bufs_t b = make(kind, 64, 256, 1024);
        double best = 1e30;
        for (int t = 0; t < 5; t++) {
            double t0 = now_ns();
            for (int r = 0; r < 200; r++) {
                job_t j = {kind, 64, 256, 1024, 256, 1, b.lhs, b.rhs, b.dst, NULL, 0};
                run_chunk(&j, 0);
            }
            double dt = now_ns() - t0;
            if (dt < best) best = dt;
        }
        printf("  %-32s %6.1f GMAC/s\n", names[kind], 64.0 * 256 * 1024 * 200 / best);
        free(b.lhs); free(b.rhs); free(b.dst);
    }
    return 0;
}
