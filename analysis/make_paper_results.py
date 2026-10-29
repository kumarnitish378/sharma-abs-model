"""Regenerates every figure and table in paper/ from a fixed seed.
Run from the repo root:  python3 analysis/make_paper_results.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from abs_theory import pmf_S, eta_support, exact_moments, approx_moments, expected_distinct

SEED = 2026
rng = np.random.default_rng(SEED)
FIG = "paper/figs"; TAB = "paper/tables"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

def draw(a, b, n, T):
    S = rng.integers(a, b + 1, size=(T, n)).sum(1)
    N = n * (abs(a) + abs(b))
    return N / (N + S)

# ---- Table: exact vs approximation vs Monte Carlo
rows = []
for a, b, n in [(-1, 1, 12), (-10, 10, 4), (-10, 10, 12), (-10, 10, 48),
                (-1000, 1000, 12), (-12, 11, 12), (-50, 50, 50)]:
    ex, ap, e = exact_moments(a, b, n), approx_moments(a, b, n), draw(a, b, n, 10**6)
    rows.append(f"$[{a},{b}]$ & {n} & {ex['states']} & {ex['mean']:.4f} & {ap['mean']:.4f} & {e.mean():.4f}"
                f" & {ex['std']:.4f} & {ap['std']:.4f} & {e.std():.4f} & {ex['skew']:.2f} \\\\")
open(f"{TAB}/moments.tex", "w").write("\n".join(rows) + "\n")

# ---- Table: distinct values (replaces the old 'Magic 11' table)
rows = []
for a, b, n, T in [(-12, 11, 12, 100), (-12, 11, 12, 10000), (-50, 50, 50, 1000)]:
    obs = [len(np.unique(rng.integers(a, b + 1, size=(T, n)).sum(1))) for _ in range(20)]
    rows.append(f"$[{a},{b}]$ & {n} & {T} & {n*(b-a)+1} & {expected_distinct(a, b, n, T):.1f}"
                f" & {np.mean(obs):.1f} ({min(obs)}--{max(obs)}) \\\\")
open(f"{TAB}/distinct.tex", "w").write("\n".join(rows) + "\n")

# ---- Table: alpha (exact, a=-10, b=10, n=12)
e, p = eta_support(-10, 10, 12)
mad = float(np.sum(p * np.abs(e - 1)))
q = np.cumsum(p)
p999 = float(np.max(np.abs(e[[np.searchsorted(q, 0.0005), np.searchsorted(q, 0.9995)]] - 1)))
rows = [f"{al:.2f} & {1 + al*(exact_moments(-10,10,12)['mean']-1):.4f} & {100*al*mad:.2f} & {100*al*p999:.2f} \\\\"
        for al in (0, 0.25, 0.5, 1.0)]
open(f"{TAB}/alpha.tex", "w").write("\n".join(rows) + "\n")

# ---- Fig 1: exact distribution
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = draw(-10, 10, 12, 10**6)
vals, cnt = np.unique(np.round(x, 12), return_counts=True)
ax.vlines(vals, 0, cnt / len(x), color="#a5b4fc", lw=1.2, label="Monte Carlo ($10^6$)")
ax.plot(e, p, ".", ms=2.5, color="#3730a3", label="exact PMF")
m = exact_moments(-10, 10, 12)["mean"]
ax.axvline(1, color="0.5", lw=0.8, ls=":"); ax.axvline(m, color="#e11d48", lw=0.8, label=f"$E[\\eta]$={m:.4f}")
ax.set_xlim(0.7, 1.45); ax.set_xlabel(r"$\eta$"); ax.set_ylabel("probability"); ax.legend(fontsize=7)
fig.savefig(f"{FIG}/pmf.pdf"); plt.close(fig)

# ---- Fig 2: spread depends on n, not on the bounds
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ns = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
for c, mk in [(1, "o"), (10, "s"), (1000, "^")]:
    ax.plot(ns, [exact_moments(-c, c, n)["std"] for n in ns] if c < 1000 else
            [draw(-c, c, n, 200000).std() for n in ns], mk, ms=3.5, mfc="none", label=f"$\\psi=[-{c},{c}]$")
nn = np.linspace(1, 64, 200)
ax.plot(nn, 1 / np.sqrt(12 * nn), "k-", lw=0.8, label=r"$1/\sqrt{12n}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("complexity $n$"); ax.set_ylabel(r"std$(\eta)$")
ax.legend(fontsize=7); fig.savefig(f"{FIG}/std_vs_n.pdf"); plt.close(fig)

# ---- Fig 3: applications (ADC reading on 0-5 V, damped oscillator), alpha=0.5, normalised eta
def noisy(v, a, b, n, alpha):
    et = draw(a, b, n, len(v)) / exact_moments(a, b, n)["mean"]
    return v * (1 + alpha * (et - 1))
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2))
t = np.linspace(0, 10, 400)
v = 2.5 + 2.4 * np.sin(2 * np.pi * 0.3 * t)
axs[0].plot(t, noisy(v, -10, 10, 48, 0.5), color="#4f46e5", lw=0.6, label="ABS, $n$=48, $\\alpha$=0.5")
axs[0].plot(t, v, "--", color="#10b981", lw=1, label="ideal")
axs[0].axhline(0, color="0.6", lw=0.6); axs[0].axhline(5, color="0.6", lw=0.6)
axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("ADC input (V)"); axs[0].legend(fontsize=7, loc="lower left")
d = 10 * np.exp(-0.3 * t) * np.cos(2 * np.pi * 0.8 * t)
axs[1].plot(t, noisy(d, -10, 10, 12, 0.5), color="#4f46e5", lw=0.6, label="ABS, $n$=12, $\\alpha$=0.5")
axs[1].plot(t, d, "--", color="#10b981", lw=1, label="ideal")
axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("displacement"); axs[1].legend(fontsize=7)
fig.savefig(f"{FIG}/applications.pdf"); plt.close(fig)
print("mad", mad, "p999", p999)

# ---- check used in Section 8.5: how often the ADC example exceeds the 5 V rail
hits, neg = 0, 0
for _ in range(200):
    y = noisy(v, -10, 10, 48, 0.5)
    hits += (y > 5).any(); neg += (y < 0).any()
print(f"ADC example: runs exceeding 5 V = {hits}/200, runs below 0 V = {neg}/200")

# ---- Table: skew is set by the bounds (centred eta, exact)
rows = []
for a, b in [(-10, 10), (-10, 100), (10, 100), (50, 100), (900, 1000)]:
    cells = []
    for n in (1, 4, 16):
        e, p = eta_support(a, b, n)
        mu = float(p @ e)
        var = float(p @ (e - mu) ** 2)
        cells.append(f"{float(p @ (e - mu) ** 3) / var ** 1.5:+.3f}")
    rows.append(f"$[{a},{b}]$ & " + " & ".join(cells) + " \\\\")
open(f"{TAB}/skew.tex", "w").write("\n".join(rows) + "\n")
print("skew table written")
