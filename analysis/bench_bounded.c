/* Bounded noise generators on the same xorshift32 RNG: throughput and worst-case work.
 * All outputs lie in a bounded interval. Build: gcc -O2 -std=gnu99 bench_bounded.c -lm */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../src/sharma_abs.h"

static uint32_t S = 12345u;
static long iters, max_iters;            /* RNG-loop iterations for rejection methods */

static inline uint32_t rnd(void) { return sharma_abs_xorshift32_r(&S); }
static inline float u01(void) { return ((rnd() >> 8) + 0.5f) * (1.0f / 16777216.0f); } /* (0,1) */

/* ABS, integer path: n draws on [a,b], one integer division, Q16 result */
static inline int32_t abs_q16(int a, int b, int n) { return compute_eta_q16_r(&S, a, b, n); }

/* Irwin-Hall: sum of n integer uniforms, centred (integer only, no division) */
static inline int32_t ih_int(int n) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)(rnd() >> 16) - 32768;
    return s;
}

static inline float gauss(void) {       /* Box-Muller, one output */
    return sqrtf(-2.0f * logf(u01())) * cosf(6.2831853f * u01());
}

static inline float clipped_gauss(float sd) {
    float z = sd * gauss();
    return z > 1 ? 1 : z < -1 ? -1 : z;
}

static inline float trunc_gauss(float sd) {   /* rejection: redraw until inside [-1,1] */
    long k = 0; float z;
    do { z = sd * gauss(); k++; } while (z > 1 || z < -1);
    iters += k; if (k > max_iters) max_iters = k;
    return z;
}

static float gamma_mt(float a) {              /* Marsaglia-Tsang, a >= 1 */
    float d = a - 1.0f / 3, c = 1.0f / sqrtf(9 * d);
    long k = 0;
    for (;;) {
        float x, v;
        do { x = gauss(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v; float u = u01(); k++;
        if (u < 1 - 0.0331f * x * x * x * x || logf(u) < 0.5f * x * x + d * (1 - v + logf(v))) {
            iters += k; if (k > max_iters) max_iters = k;
            return d * v;
        }
    }
}

static inline float beta(float a, float b) { float x = gamma_mt(a); return x / (x + gamma_mt(b)); }

static inline float triangular(float c) {     /* mode c in (0,1), inverse CDF */
    float u = u01();
    return u < c ? sqrtf(u * c) : 1 - sqrtf((1 - u) * (1 - c));
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

#define RUN(label, expr) do { \
    iters = max_iters = 0; double best = 1e9; volatile double sink = 0; \
    for (int r = 0; r < 3; r++) { double t0 = now(); double acc = 0; \
        for (int i = 0; i < T; i++) { acc += (expr); } sink += acc; \
        double dt = (now() - t0) / T * 1e9; if (dt < best) best = dt; } \
    printf("%-28s %6.1f ns   worst-case loop count %ld\n", label, best, max_iters); \
    (void)sink; } while (0)

int main(void) {
    const int T = 5000000;
    RUN("ABS q16 n=4 symmetric", abs_q16(900, 1000, 4));
    RUN("ABS q16 n=4 skewed", abs_q16(-10, 10, 4));
    RUN("ABS q16 n=1", abs_q16(-10, 10, 1));
    RUN("Irwin-Hall int n=4", ih_int(4));
    RUN("Irwin-Hall int n=1 (uniform)", ih_int(1));
    RUN("Gaussian (unbounded)", gauss());
    RUN("Clipped Gaussian sd=0.4", clipped_gauss(0.4f));
    RUN("Truncated Gaussian sd=0.4", trunc_gauss(0.4f));
    RUN("Truncated Gaussian sd=1.0", trunc_gauss(1.0f));
    RUN("Beta(2,2) symmetric", beta(2, 2));
    RUN("Beta(2,5) skewed", beta(2, 5));
    RUN("Triangular c=0.3 skewed", triangular(0.3f));
    return 0;
}
