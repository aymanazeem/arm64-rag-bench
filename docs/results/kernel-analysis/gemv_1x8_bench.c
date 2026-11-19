// Time the KleidiAI GEMV kernels llama.cpp selects on dotprod-only cores (1x4, used on M1)
// and on i8mm cores (1x8 dotprod GEMV, two accumulators). Timing only; data is random.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"

typedef void (*run_fn)(size_t, size_t, size_t, const void *, const void *, float *, size_t, size_t, float, float);
typedef struct { run_fn fn; size_t n, k; const uint8_t *lhs, *rhs; float *dst; int reps; } job_t;

static double now_ns(void) { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static void *xalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n + 64)) exit(1); return p; }

static void *worker(void *p) {
    job_t *j = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int r = 0; r < j->reps; r++) j->fn(1, j->n, j->k, j->lhs, j->rhs, j->dst, j->n * 4, 4, -3.4e38f, 3.4e38f);
    return NULL;
}

static double gbps(run_fn fn, size_t stride, int th, size_t n, size_t k, const uint8_t *lhs, const uint8_t *rhs, float *dst, int reps) {
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
        pthread_t t[16]; job_t j[16];
        size_t groups = n / 4, per = groups / th;
        for (int i = 0; i < th; i++) {
            size_t g0 = i * per, g1 = i == th - 1 ? groups : g0 + per;
            j[i] = (job_t){fn, (g1 - g0) * 4, k, lhs, rhs + g0 * stride, dst + g0 * 4, reps};
        }
        double t0 = now_ns();
        for (int i = 0; i < th; i++) pthread_create(&t[i], NULL, worker, &j[i]);
        for (int i = 0; i < th; i++) pthread_join(t[i], NULL);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    return (double) n * k * reps / best;
}

int main(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const size_t k = 4096, n_big = 98304;
    struct { const char *name; run_fn fn; size_t stride, lhs_size; } ks[] = {
        {"1x4 (selected on M1, 1 accumulator)", kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
         kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod(4, k),
         kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod(1, k)},
        {"1x8 (selected with i8mm, 2 accumulators)", kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
         kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod(4, k),
         kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod(1, k)},
    };
    for (int v = 0; v < 2; v++) {
        size_t bytes = (n_big / 4) * ks[v].stride;
        uint8_t *rhs = xalloc(bytes), *lhs = xalloc(ks[v].lhs_size);
        float *dst = xalloc(n_big * 4);
        for (size_t i = 0; i < bytes; i++) rhs[i] = (uint8_t) (i * 2654435761u >> 24);
        for (size_t i = 0; i < ks[v].lhs_size; i++) lhs[i] = (uint8_t) (i * 40503u >> 8);
        printf("%-42s cache 1T %6.1f | memory 1T %6.1f | 4T %6.1f | 8T %6.1f GB/s\n", ks[v].name,
               gbps(ks[v].fn, ks[v].stride, 1, 256, k, lhs, rhs, dst, 400),
               gbps(ks[v].fn, ks[v].stride, 1, n_big, k, lhs, rhs, dst, 3),
               gbps(ks[v].fn, ks[v].stride, 4, n_big, k, lhs, rhs, dst, 3),
               gbps(ks[v].fn, ks[v].stride, 8, n_big, k, lhs, rhs, dst, 3));
        free(rhs); free(lhs); free(dst);
    }
    return 0;
}
