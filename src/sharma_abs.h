#ifndef SHARMA_ABS_H
#define SHARMA_ABS_H

/*
 * Sharma Absolute Boundary Summation (ABS) Model -- single-header C library
 *
 *   eta    = N / (N + S),   N = n(|a| + |b|),   S = r_1 + ... + r_n,
 *            r_i iid uniform integers on [a, b]
 *   v_real = v_ideal * (1 + alpha * (eta - 1))
 *
 * Properties (proved in the paper):
 *   - If b != 0 the denominator N + S is always > 0, so eta is always
 *     defined and strictly positive. b == 0 is rejected (returns 1).
 *   - For 0 <= alpha <= 1 the multiplier is > 0: the sign of v_ideal is kept.
 *   - With symmetric bounds [-c, c], eta lies in [2/3, 2] and its spread is
 *     set by n (std ~ 1/sqrt(12 n)), almost independent of c.
 *   - E[eta] > 1 (bias). Divide by the exact mean (computed on the host with
 *     analysis/abs_theory.py or SharmaABS.exact_mean()) if a mean of 1 matters.
 *
 * RNG: xorshift32 (Marsaglia 2003) with Lemire's unbiased bounded sampling.
 * Not cryptographic. Call sharma_abs_seed() once; the state is global and
 * not thread/ISR safe -- use the *_r variants with your own state if needed.
 *
 * Limits: |a|, |b| <= 32767 and 1 <= n <= 32767, so all sums fit in int32.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- RNG ---------------- */

static uint32_t sharma_abs_state_ = 2463534242u;

static inline void sharma_abs_seed(uint32_t seed) {
    sharma_abs_state_ = seed ? seed : 2463534242u; /* xorshift state must be non-zero */
}

static inline uint32_t sharma_abs_xorshift32_r(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

/* Unbiased uniform integer on [lo, hi] (Lemire 2019, multiply-shift with rejection). */
static inline int32_t sharma_abs_uniform_r(uint32_t *s, int32_t lo, int32_t hi) {
    uint32_t range = (uint32_t)(hi - lo) + 1u;
    uint64_t m = (uint64_t)sharma_abs_xorshift32_r(s) * range;
    uint32_t l = (uint32_t)m;
    if (l < range) {
        uint32_t t = (0u - range) % range;
        while (l < t) {
            m = (uint64_t)sharma_abs_xorshift32_r(s) * range;
            l = (uint32_t)m;
        }
    }
    return lo + (int32_t)(m >> 32);
}

/* ---------------- core: integer sum ---------------- */

static inline int32_t sharma_abs_abs_(int32_t v) { return v < 0 ? -v : v; }

/* Returns 0 on bad input, else 1 and writes N and D = N + S (D > 0 guaranteed). */
static inline int sharma_abs_draw_r(uint32_t *s, int32_t a, int32_t b, int32_t n,
                                    int32_t *N_out, int32_t *D_out) {
    if (n <= 0 || b == 0) return 0;
    if (a > b) { int32_t t = a; a = b; b = t; }
    int32_t N = n * (sharma_abs_abs_(a) + sharma_abs_abs_(b));
    int32_t S = 0;
    for (int32_t i = 0; i < n; i++) S += sharma_abs_uniform_r(s, a, b);
    *N_out = N;
    *D_out = N + S;
    return 1;
}

/* ---------------- floating-point API ---------------- */

static inline float compute_eta_r(uint32_t *s, int min_val, int max_val, int n) {
    int32_t N, D;
    if (!sharma_abs_draw_r(s, min_val, max_val, n, &N, &D)) return 1.0f;
    return (float)N / (float)D;
}

static inline float compute_eta(int min_val, int max_val, int n) {
    return compute_eta_r(&sharma_abs_state_, min_val, max_val, n);
}

/* v_real = v_ideal * (1 - alpha + alpha * eta) */
static inline float apply_sharma_noise(float ideal_val, float eta, float alpha) {
    return ideal_val * (1.0f - alpha + alpha * eta);
}

/* ---------------- fixed-point API (no FPU needed) ---------------- */
/* Q16.16: 1.0 == 65536. eta_q16 is in (0, 2^17] for symmetric bounds. */

static inline int32_t compute_eta_q16_r(uint32_t *s, int min_val, int max_val, int n) {
    int32_t N, D;
    if (!sharma_abs_draw_r(s, min_val, max_val, n, &N, &D)) return 65536;
    if (N < 65536) /* N << 16 fits in 32 bits: one 32-bit hardware divide */
        return (int32_t)(((uint32_t)N << 16) / (uint32_t)D);
    return (int32_t)(((uint64_t)(uint32_t)N << 16) / (uint32_t)D);
}

static inline int32_t compute_eta_q16(int min_val, int max_val, int n) {
    return compute_eta_q16_r(&sharma_abs_state_, min_val, max_val, n);
}

/* ideal and result share the caller's fixed-point scale; alpha_q16 in [0, 65536]. */
static inline int32_t apply_sharma_noise_q16(int32_t ideal, int32_t eta_q16, int32_t alpha_q16) {
    int32_t mult = 65536 + (int32_t)(((int64_t)alpha_q16 * (eta_q16 - 65536)) >> 16);
    return (int32_t)(((int64_t)ideal * mult) >> 16);
}

#ifdef __cplusplus
}
#endif

#endif /* SHARMA_ABS_H */
