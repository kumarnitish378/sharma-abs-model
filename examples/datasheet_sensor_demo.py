"""Datasheet-calibrated sensor noise: Sharma ABS vs Gaussian.

Simulates 30 Benewake TFmini-S lidar units (spec: +/-6 cm up to 6 m, +/-1% from 6 to 12 m,
1 cm resolution) reading targets from 0.2 to 11.8 m. Both models have the same per-device
and per-sample spread; only the shape differs.

  ABS model      : every reading stays inside the datasheet limit
  Gaussian model : a fraction of readings breaks the limit when the noise is set near the
                   limit (stress test); at gentle settings both stay inside

Run:  python examples/datasheet_sensor_demo.py      (writes datasheet_sensor_demo.png)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sensor_model import tfmini_s  # noqa: E402


class GaussianDraws:
    """Drop-in replacement for UnitABS with the same sd but Gaussian shape."""

    def __init__(self, sd, rng):
        self.sd, self.rng = sd, rng

    def draw(self, size=None):
        v = self.rng.normal(0.0, self.sd, size)
        return v if size is not None else float(v)


def simulate(gaussian, n=1, devices=30, reads=400, seed=7):
    rng = np.random.default_rng(seed)
    xs, errs, lims = [], [], []
    for d in range(devices):
        s = tfmini_s(seed=seed * 1000 + d, n=n)
        if gaussian:
            s.u = GaussianDraws(s.u.sd, s.rng)
            s.new_device()
        x = rng.uniform(0.2, 11.8, reads)
        y = s.read(x)
        xs.append(x); errs.append(y - x); lims.append(s.spec_limit(x) + s.resolution / 2)
    return np.concatenate(xs), np.concatenate(errs), np.concatenate(lims)


def main(out="datasheet_sensor_demo.png"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    d = np.linspace(0.1, 12, 500)
    env = np.where(d <= 6, 0.06, 0.01 * d)
    for ax, gaussian, title, col in ((axs[0], False, "Sharma ABS (symmetric)", "#4f46e5"),
                                     (axs[1], True, "Gaussian, same spread", "#d97706")):
        x, e, lim = simulate(gaussian)
        bad = np.abs(e) > lim + 1e-12
        ax.fill_between(d, -100 * env, 100 * env, color="#10b981", alpha=0.12, lw=0,
                        label="datasheet limit")
        ax.plot(d, 100 * env, color="#10b981", lw=1); ax.plot(d, -100 * env, color="#10b981", lw=1)
        ax.scatter(x[~bad], 100 * e[~bad], s=2, color=col, alpha=0.35, lw=0)
        ax.scatter(x[bad], 100 * e[bad], s=10, color="#e11d48", label=f"out of spec: {bad.mean()*100:.2f}%")
        ax.set_title(f"{title}\nworst error = {np.max(np.abs(e) / lim) * 100:.0f}% of limit", fontsize=10)
        ax.set_xlabel("true distance (m)")
        ax.legend(loc="upper left", fontsize=8, markerscale=1.5)
        print(f"{title:28s} out of spec {bad.mean()*100:.3f}%   worst {np.max(np.abs(e)/lim)*100:.1f}% of limit")
    axs[0].set_ylabel("reading error (cm)")
    fig.suptitle("TFmini-S lidar, stress test (n=1), 30 simulated units x 400 readings  "
                 "(spec: ±6 cm to 6 m, ±1% to 12 m)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    print("\nout-of-spec readings vs noise level (more n = gentler noise):")
    for n in (1, 2, 4, 8):
        rows = []
        for g in (False, True):
            _, e, lim = simulate(g, n=n, devices=100, reads=2000, seed=11)
            rows.append(np.mean(np.abs(e) > lim + 1e-12) * 100)
        print(f"  n={n}:  ABS {rows[0]:.3f}%   Gaussian {rows[1]:.3f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
