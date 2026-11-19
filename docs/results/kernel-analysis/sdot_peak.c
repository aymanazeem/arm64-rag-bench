// Peak SDOT throughput (16 independent accumulators) and clock, with 1..N threads running at once.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERS 20000000ULL
static double now_ns(void) { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

typedef struct { double add_ns, sdot_ns; } res_t;

static void *worker(void *p) {
    res_t *r = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t x = 0;
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++)
        __asm__ volatile("add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n" : "+r"(x));
    r->add_ns = (now_ns() - t0) / (ITERS * 16.0);
    t0 = now_ns();
    for (uint64_t i = 0; i < ITERS; i++)
        __asm__ volatile(
            "sdot v16.4s, v0.16b, v1.4b[0]\n sdot v17.4s, v0.16b, v1.4b[1]\n sdot v18.4s, v0.16b, v1.4b[2]\n sdot v19.4s, v0.16b, v1.4b[3]\n"
            "sdot v20.4s, v0.16b, v2.4b[0]\n sdot v21.4s, v0.16b, v2.4b[1]\n sdot v22.4s, v0.16b, v2.4b[2]\n sdot v23.4s, v0.16b, v2.4b[3]\n"
            "sdot v24.4s, v0.16b, v3.4b[0]\n sdot v25.4s, v0.16b, v3.4b[1]\n sdot v26.4s, v0.16b, v3.4b[2]\n sdot v27.4s, v0.16b, v3.4b[3]\n"
            "sdot v28.4s, v0.16b, v4.4b[0]\n sdot v29.4s, v0.16b, v4.4b[1]\n sdot v30.4s, v0.16b, v4.4b[2]\n sdot v31.4s, v0.16b, v4.4b[3]\n"
            ::: "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    r->sdot_ns = (now_ns() - t0) / (ITERS * 16.0);
    (void) x;
    return NULL;
}

int main(void) {
    for (int th = 1; th <= 4; th *= 2) {
        for (int rep = 0; rep < 2; rep++) {
            pthread_t t[8]; res_t r[8];
            for (int i = 0; i < th; i++) pthread_create(&t[i], NULL, worker, &r[i]);
            for (int i = 0; i < th; i++) pthread_join(t[i], NULL);
            double ghz = 0, per = 0;
            for (int i = 0; i < th; i++) { ghz += 1.0 / r[i].add_ns / th; per += r[i].add_ns / r[i].sdot_ns / th; }
            printf("threads %d rep %d: clock %.2f GHz, SDOT %.2f per cycle per core (%.1f GMAC/s per core)\n",
                   th, rep, ghz, per, per * ghz * 16);
        }
    }
    return 0;
}
