"""Exact and approximate properties of the Sharma ABS noise factor.

eta = N / (N + S),  N = n(|a| + |b|),  S = sum of n iid uniform integers on {a..b}.
Everything here is exact (finite sums over the PMF of S) except the *_approx functions.
"""
import numpy as np


def pmf_S(a, b, n):
    """Exact PMF of S on support {n*a, ..., n*b} by repeated convolution."""
    m = b - a + 1
    p1 = np.full(m, 1.0 / m)
    p = np.array([1.0])
    for _ in range(n):
        p = np.convolve(p, p1)
    s = np.arange(n * a, n * b + 1)
    return s, p


def eta_support(a, b, n):
    s, p = pmf_S(a, b, n)
    N = n * (abs(a) + abs(b))
    if np.any(N + s == 0):
        raise ValueError("denominator can be zero: requires b != 0")
    return N / (N + s), p


def exact_moments(a, b, n):
    e, p = eta_support(a, b, n)
    mean = float(np.sum(p * e))
    var = float(np.sum(p * (e - mean) ** 2))
    skew = float(np.sum(p * (e - mean) ** 3) / var ** 1.5)
    return dict(mean=mean, std=var ** 0.5, skew=skew,
                lo=float(e.min()), hi=float(e.max()), states=len(e))


def approx_moments(a, b, n):
    """Second-order delta-method approximation."""
    W = abs(a) + abs(b)
    mu_r = (a + b) / 2
    var_r = ((b - a + 1) ** 2 - 1) / 12
    mx = mu_r / W
    vx = var_r / (n * W * W)
    mean = 1 - mx + mx * mx + vx
    std = vx ** 0.5 / (1 + mx) ** 2
    return dict(mean=mean, std=std)


def expected_distinct(a, b, n, T):
    """Expected number of distinct eta values seen in T independent draws."""
    _, p = pmf_S(a, b, n)
    return float(np.sum(1 - (1 - p) ** T))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    print("a b n | exact mean std skew lo hi states | approx mean std | MC mean std")
    for a, b, n in [(-10, 10, 12), (-12, 11, 12), (-50, 50, 50), (-1000, 1000, 12),
                    (-1, 1, 12), (-10, 10, 4), (-10, 10, 48)]:
        ex = exact_moments(a, b, n)
        ap = approx_moments(a, b, n)
        S = rng.integers(a, b + 1, size=(200000, n)).sum(1)
        N = n * (abs(a) + abs(b))
        e = N / (N + S)
        print(a, b, n, "|", {k: round(v, 4) for k, v in ex.items()},
              "|", round(ap["mean"], 4), round(ap["std"], 4),
              "|", round(e.mean(), 4), round(e.std(), 4))
    for a, b, n, T in [(-12, 11, 12, 100), (-50, 50, 50, 1000)]:
        S = rng.integers(a, b + 1, size=(T, n)).sum(1)
        print("distinct", a, b, n, T, "expected", round(expected_distinct(a, b, n, T), 1),
              "one run", len(np.unique(S)))
