// Measure SDOT (by element) latency and throughput on this core, calibrated
// against a dependent 1-cycle integer add chain.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ITERS 20000000ULL

static double now_ns(void) { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static double add_chain(void) {
    uint64_t x = 0;
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
            "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
            "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
            "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
            : "+r"(x));
    }
    return (now_ns() - t0) / (ITERS * 16.0);
}

static double sdot_dependent(void) {
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "sdot v0.4s, v16.16b, v17.4b[0]\n sdot v0.4s, v16.16b, v17.4b[1]\n"
            "sdot v0.4s, v16.16b, v17.4b[2]\n sdot v0.4s, v16.16b, v17.4b[3]\n"
            "sdot v0.4s, v16.16b, v17.4b[0]\n sdot v0.4s, v16.16b, v17.4b[1]\n"
            "sdot v0.4s, v16.16b, v17.4b[2]\n sdot v0.4s, v16.16b, v17.4b[3]\n"
            ::: "v0");
    }
    return (now_ns() - t0) / (ITERS * 8.0);
}

static double sdot_independent(void) {
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "sdot v0.4s, v16.16b, v17.4b[0]\n sdot v1.4s, v16.16b, v17.4b[1]\n"
            "sdot v2.4s, v16.16b, v17.4b[2]\n sdot v3.4s, v16.16b, v17.4b[3]\n"
            "sdot v4.4s, v16.16b, v17.4b[0]\n sdot v5.4s, v16.16b, v17.4b[1]\n"
            "sdot v6.4s, v16.16b, v17.4b[2]\n sdot v7.4s, v16.16b, v17.4b[3]\n"
            ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    }
    return (now_ns() - t0) / (ITERS * 8.0);
}

int main(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int rep = 0; rep < 3; rep++) {
        double a = add_chain();
        double d = sdot_dependent();
        double n = sdot_independent();
        printf("rep %d: add %.4f ns (%.2f GHz) | dependent sdot %.4f ns = %.2f cycles | independent sdot %.4f ns = %.2f per cycle\n",
               rep, a, 1.0 / a, d, d / a, n, a / n);
    }
    return 0;
}
