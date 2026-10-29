"""Fit real sensor error data with Sharma ABS and other noise models.

Input: a CSV with two columns, true value and measured value (header optional), e.g.
       a lidar pointed at a target at known distances, or a DS18B20 next to a reference
       thermometer. Use several thousand readings if you can.

Method (the same for every model, so the comparison is fair):
  1. error = measured - true; standardise it (subtract mean, divide by sd)
  2. for each family, pick the parameters whose skewness and kurtosis are closest
     to the data's (method of moments on the 3rd and 4th moments)
  3. score the fit with the two-sample Kolmogorov-Smirnov distance (smaller = better;
     with ~5000 readings, differences below ~0.02 are within sampling noise, and
     heavily quantised data inflates KS for every model)
  4. for bounded families, report the fraction of real errors that fall OUTSIDE the
     fitted bound -- a bounded model that the data exceeds cannot be right

Usage:
  python3 analysis/fit_sensor_data.py readings.csv [--true-col 0 --meas-col 1]
  python3 analysis/fit_sensor_data.py --demo          # synthetic self-test
"""
import argparse
import csv

import numpy as np
from scipy import stats

RNG = np.random.default_rng(2026)
NS = 200_000          # model samples per fit


# ---------------------------------------------------------------- family grids
# Each entry: (params, skew, excess kurtosis, sampler(size) -> raw samples)

def abs_grid():
    out = []
    b = 100
    for n in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        for a in list(range(-100, 100, 5)) + [95, 98, 99]:
            if a == 0:
                continue
            m = b - a + 1
            p = np.ones(1)
            for _ in range(n):
                p = np.convolve(p, np.full(m, 1.0 / m))
            s = np.arange(n * a, n * b + 1)
            N = n * (abs(a) + abs(b))
            e = N / (N + s)
            mu = p @ e
            var = p @ (e - mu) ** 2
            sk = (p @ (e - mu) ** 3) / var ** 1.5
            ku = (p @ (e - mu) ** 4) / var ** 2 - 3

            def sampler(size, n=n, a=a, b=b, N=N):
                S = RNG.integers(a, b + 1, size=(size, n)).sum(1)
                return N / (N + S)
            out.append(((n, a, b), sk, ku, sampler))
            # mirrored version gives the negative skews
            out.append(((n, a, b, "mirrored"), -sk, ku, lambda size, f=sampler: -f(size)))
    return out


def irwin_hall_grid():
    return [((n,), 0.0, -6 / (5 * n), lambda size, n=n: RNG.random((size, n)).sum(1))
            for n in range(1, 65)]


def trunc_gauss_grid():
    out = []
    for lo in np.linspace(-6, 0.5, 30):
        for hi in np.linspace(0.6, 6, 30):
            if hi <= lo + 0.1:
                continue
            d = stats.truncnorm(lo, hi)
            _, _, s, k = d.stats(moments="mvsk")
            out.append(((round(float(lo), 2), round(float(hi), 2)), float(s), float(k),
                        lambda size, d=d: d.rvs(size, random_state=RNG)))
    return out


def beta_grid():
    out = []
    for a in np.geomspace(0.3, 40, 35):
        for b in np.geomspace(0.3, 40, 35):
            d = stats.beta(a, b)
            _, _, s, k = d.stats(moments="mvsk")
            out.append(((round(float(a), 2), round(float(b), 2)), float(s), float(k),
                        lambda size, d=d: d.rvs(size, random_state=RNG)))
    return out


def triangular_grid():
    out = []
    for c in np.linspace(0, 1, 51):
        d = stats.triang(c)
        _, _, s, k = d.stats(moments="mvsk")
        out.append(((round(float(c), 2),), float(s), float(k), lambda size, d=d: d.rvs(size, random_state=RNG)))
    return out


FAMILIES = {
    "Sharma ABS": abs_grid,
    "Irwin-Hall": irwin_hall_grid,
    "Truncated Gaussian": trunc_gauss_grid,
    "Beta": beta_grid,
    "Triangular": triangular_grid,
}


# ---------------------------------------------------------------- fitting

def standardise(x):
    return (x - x.mean()) / x.std()


def fit(err):
    z = standardise(np.asarray(err, float))
    sk, ku = stats.skew(z), stats.kurtosis(z)
    results = []
    g = RNG.standard_normal(NS)
    results.append(("Gaussian (unbounded)", "-", stats.ks_2samp(z, g).statistic, 0.0, float("inf")))
    for name, make in FAMILIES.items():
        grid = make()
        params, gs, gk, sampler = min(grid, key=lambda r: (r[1] - sk) ** 2 + 0.25 * (r[2] - ku) ** 2)
        y = standardise(sampler(NS))
        bound = float(np.max(np.abs(y)))
        outside = float(np.mean(np.abs(z) > bound * 1.0001))
        results.append((name, params, stats.ks_2samp(z, y).statistic, outside, bound))
    return sk, ku, sorted(results, key=lambda r: r[2])


def report(err, title):
    sk, ku, res = fit(err)
    print(f"\n{title}\n  {len(err)} readings   skew {sk:+.3f}   excess kurtosis {ku:+.3f}")
    print(f"  {'model':24s} {'KS distance':>11s} {'data outside bound':>19s} {'bound (sd)':>11s}  params")
    for name, params, ks, outside, bound in res:
        b = "none" if bound == float("inf") else f"{bound:.2f}"
        print(f"  {name:24s} {ks:11.4f} {outside * 100:18.2f}% {b:>11s}  {params}")


def load_csv(path, tcol, mcol):
    t, m = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                t.append(float(row[tcol])); m.append(float(row[mcol]))
            except (ValueError, IndexError):
                continue  # header or blank line
    return np.array(m) - np.array(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--true-col", type=int, default=0)
    ap.add_argument("--meas-col", type=int, default=1)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data to check the fitter")
    a = ap.parse_args()
    if a.demo:
        report(RNG.normal(0, 1, 5000), "demo 1: Gaussian errors")
        report(stats.beta(2, 6).rvs(5000, random_state=RNG), "demo 2: right-skewed bounded errors (Beta 2,6)")
        S = RNG.integers(-10, 11, size=(5000, 3)).sum(1)
        report(60 / (60 + S), "demo 3: Sharma ABS errors (n=3, bounds -10..10)")
    elif a.csv:
        report(load_csv(a.csv, a.true_col, a.meas_col), f"data: {a.csv}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
