#ifndef SHARMA_ABS_H
#define SHARMA_ABS_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/*
 * Sharma Absolute Boundary Summation (ABS) Model
 * 
 * Generates a noise factor eta used to perturb a deterministic signal.
 */
static inline float compute_eta(int min_val, int max_val, int n) {
    if (n <= 0) return 1.0f;
    
    float minimest = (float)(n * min_val);
    float miximet = (float)(n * max_val);
    
    int sum_random = 0;
    for (int i = 0; i < n; i++) {
        // Uniform random integer between min_val and max_val inclusive
        int rand_val = min_val + (rand() % (max_val - min_val + 1));
        sum_random += rand_val;
    }
    
    float numerator = fabsf(minimest) + fabsf(miximet);
    float denominator = numerator + (float)sum_random;
    
    // Stability Guard: Always check if the denominator equals zero; if it does, default to 1.0
    if (denominator == 0.0f) {
        return 1.0f;
    }
    
    return numerator / denominator;
}

/*
 * Appy Sharma noise to an ideal value with an influence coefficient (alpha).
 * alpha = 0.0 -> Perfectly Deterministic
 * alpha = 1.0 -> Maximum Turbulence
 *
 * Final Signal S_sharma = S_ideal * (1 - alpha + alpha * eta)
 */
static inline float apply_sharma_noise(float ideal_val, float eta, float alpha) {
    // Ensures alpha is physically meaningful (bound to 0.0 - 1.0 if strictly needed, but math holds)
    return ideal_val * (1.0f - alpha + alpha * eta);
}

#endif // SHARMA_ABS_H
