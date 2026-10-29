"""Which (spread, skew) shapes can each bounded noise family produce?

Every distribution is normalised the same way: shifted to mean 0, then scaled so the
largest possible |value| is 1 (so the datasheet limit maps to +/-1). We then record
sd (spread relative to the limit) and skewness over each family's parameter range.

Run from the repo root:  python3 analysis/bounded_coverage.py
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def norm_moments(lo, hi, mean, var, skew):
    D = max(hi - mean, mean - lo)
    return np.sqrt(var) / D, skew


def abs_family():
    pts = []
    b = 100
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
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
            pts.append(norm_moments(e.min(), e.max(), mu, var, sk) + (n,))
    return np.array(pts)


def irwin_hall():
    return np.array([(1 / np.sqrt(3 * n), 0.0) for n in range(1, 65)])


def trunc_gauss():
    pts = []
    for lo in np.linspace(-6, 0.5, 40):
        for hi in np.linspace(0.6, 6, 40):
            if hi <= lo + 0.1:
                continue
            d = stats.truncnorm(lo, hi)
            m, v, s = d.stats(moments="mvs")
            pts.append(norm_moments(lo, hi, float(m), float(v), float(s)))
    return np.array(pts)


def beta_family():
    pts = []
    for a in np.geomspace(0.3, 40, 45):
        for b in np.geomspace(0.3, 40, 45):
            d = stats.beta(a, b)
            m, v, s = d.stats(moments="mvs")
            pts.append(norm_moments(0, 1, float(m), float(v), float(s)))
    return np.array(pts)


def triangular():
    pts = []
    for c in np.linspace(0, 1, 101):
        d = stats.triang(c)
        m, v, s = d.stats(moments="mvs")
        pts.append(norm_moments(0, 1, float(m), float(v), float(s)))
    return np.array(pts)


def main(out="docs/media/bounded_coverage.png"):
    fams = {
        "Beta (2 params)": (beta_family(), "#94a3b8", 6, "o"),
        "Truncated Gaussian (2 params)": (trunc_gauss(), "#fbbf24", 6, "o"),
        "Triangular (1 param)": (triangular(), "#10b981", 10, "s"),
        "Irwin-Hall (n)": (irwin_hall(), "#f43f5e", 18, "D"),
    }
    absp = abs_family()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, (p, col, sz, mk) in fams.items():
        ax.scatter(p[:, 1], p[:, 0], s=sz, color=col, alpha=0.5, marker=mk, lw=0, label=name)
    sc = ax.scatter(absp[:, 1], absp[:, 0], s=16, c=absp[:, 2], cmap="viridis", lw=0,
                    label="Sharma ABS (n, bounds)")
    plt.colorbar(sc, ax=ax, label="ABS complexity n")
    ax.set_xlabel("skewness")
    ax.set_ylabel("sd / max error")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Shapes each bounded family can produce (mean 0, max |error| = 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    print("family: sd range | skew range")
    for name, (p, *_ ) in list(fams.items()) + [("Sharma ABS", (absp,))]:
        print(f"  {name:30s} sd {p[:, 0].min():.3f}-{p[:, 0].max():.3f} | "
              f"skew {p[:, 1].min():+.2f} to {p[:, 1].max():+.2f}")


if __name__ == "__main__":
    main()
