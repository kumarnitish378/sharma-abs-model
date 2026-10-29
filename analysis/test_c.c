/* Checks the C header against the exact theory: mean/std, positivity, q16 vs float. */
#include <stdio.h>
#include <math.h>
#include "../src/sharma_abs.h"
int main(void) {
    sharma_abs_seed(2026);
    const int T = 2000000; double s1 = 0, s2 = 0, mn = 9, mx = 0, maxerr = 0;
    for (int i = 0; i < T; i++) {
        float e = compute_eta(-10, 10, 12);
        s1 += e; s2 += (double)e * e; if (e < mn) mn = e; if (e > mx) mx = e;
    }
    double m = s1 / T;
    printf("float: mean=%.4f std=%.4f min=%.4f max=%.4f (theory 1.0078 0.0901 0.6667 2.0)\n",
           m, sqrt(s2 / T - m * m), mn, mx);
    uint32_t a = 7, b = 7;
    for (int i = 0; i < T; i++) {
        float e = compute_eta_r(&a, -10, 10, 12);
        int32_t q = compute_eta_q16_r(&b, -10, 10, 12);
        double err = fabs(q / 65536.0 - e); if (err > maxerr) maxerr = err;
    }
    printf("q16 vs float max |diff| = %.2e (Q16 step %.2e)\n", maxerr, 1 / 65536.0);
    printf("b==0 guard -> %.1f, q16 apply(1000, eta=1.5, alpha=0.5) = %d (expect 1250)\n",
           compute_eta(-5, 0, 4), apply_sharma_noise_q16(1000, 98304, 32768));
    return 0;
}
