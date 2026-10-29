"""Datasheet-calibrated sensor model built on the Sharma ABS factor.

A datasheet usually states accuracy as a hard limit, e.g.  "+/-(1% of reading + 6 cm)".
This model produces readings whose error NEVER exceeds that limit (plus half an LSB of
rounding), because the ABS factor is bounded. Gaussian noise with the same spread will
always break the limit on a fraction of readings.

    reading = quantize( clip( x + error ) )
    error   = |x| * rel * [f*g + (1-f)*u1]  +  abs_ * [f*o + (1-f)*u2]

  rel, abs_ : datasheet accuracy (fraction of reading, absolute units); may depend on x
  g, o      : per-device gain and offset errors, drawn ONCE per simulated unit
  u1, u2    : per-sample noise, drawn every reading
  f         : share of the error budget that is fixed per device (calibration error)

g, o, u1, u2 are all "unit ABS" variables: mean 0, always inside [-1, 1]. So the error is
always within rel*|x| + abs_, whatever the draw. The ABS bounds (a, b) choose the shape:
narrow positive bounds give a symmetric error (for "+/-" specs), bounds around zero give a
right-skewed one.

The bounded property is what keeps readings in spec. Other bounded distributions (e.g. a
scaled sum of uniforms) would do the same; ABS adds a slight right skew and runs with
integer arithmetic on an MCU.
"""
import numpy as np


SHAPES = {
    # (a, b) bounds for each shape. The bounds set the skew of eta:
    "symmetric": (900, 1000),   # narrow, positive bounds: eta ~ linear in S -> ~no skew
    "skewed": (-10, 10),        # symmetric bounds around 0: right-skewed (skew ~ +0.5)
}


class UnitABS:
    """ABS factor rescaled to mean 0 and range within [-1, 1].

    u = (eta - E[eta]) / D,  D = max(eta_max - E[eta], E[eta] - eta_min)
    n sets how concentrated u is (its sd); the bounds (a, b) set its skew.
    """

    def __init__(self, n=4, bounds=(900, 1000), rng=None):
        a, b = bounds
        self.n, self.a, self.b = n, a, b
        self.rng = rng if rng is not None else np.random.default_rng()
        m = b - a + 1
        p = np.ones(1)
        for _ in range(n):
            p = np.convolve(p, np.full(m, 1.0 / m))
        s = np.arange(n * a, n * b + 1)
        self.N = n * (abs(a) + abs(b))
        eta = self.N / (self.N + s)
        self.mean = float(p @ eta)
        self.D = float(max(eta.max() - self.mean, self.mean - eta.min()))
        u = (eta - self.mean) / self.D
        self.sd = float(np.sqrt(p @ u**2))
        self.skew = float((p @ u**3) / self.sd**3)

    def draw(self, size=None):
        k = int(np.prod(size)) if size is not None else 1
        S = self.rng.integers(self.a, self.b + 1, size=(k, self.n)).sum(1)
        u = (self.N / (self.N + S) - self.mean) / self.D
        return u.reshape(size) if size is not None else float(u[0])


def n_for_sd(target_sd, shape="symmetric", n_max=64):
    """Smallest n whose unit-ABS sd is <= target_sd (sd as a fraction of the max error).

    Use it when a datasheet gives both a typical and a max error:
    n_for_sd(typical / max).
    """
    n = 1
    while n < n_max and UnitABS(n, SHAPES[shape]).sd > target_sd:
        n += 1
    return n


class SensorModel:
    def __init__(self, accuracy, resolution=None, limits=(None, None),
                 fixed_share=0.5, n=4, shape="symmetric", seed=None, rng=None):
        """
        accuracy    : (rel, abs_) tuple, or a function x -> (rel, abs_) for range-dependent specs
        resolution  : output step (LSB), e.g. 0.01 m; None = no quantization
        limits      : (lo, hi) measurement range; readings are clipped to it
        fixed_share : 0..1, part of the error budget that is a fixed per-device error
        n           : ABS complexity; sets the typical/max error ratio (see n_for_sd)
        shape       : "symmetric", "skewed", or explicit ABS bounds (a, b)
        rng         : random source with integers()/normal(), e.g. CameraNumpyRNG for
                      camera noise; default numpy.random.default_rng(seed)
        """
        self.accuracy = accuracy if callable(accuracy) else (lambda x, a=accuracy: a)
        self.resolution, self.limits, self.f = resolution, limits, fixed_share
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        bounds = SHAPES[shape] if isinstance(shape, str) else shape
        self.u = UnitABS(n, bounds, self.rng)
        self.new_device()

    def new_device(self):
        """Draw a new physical unit: new fixed gain and offset errors."""
        self.g, self.o = self.u.draw(), self.u.draw()

    def spec_limit(self, x):
        x = np.asarray(x, dtype=float)
        rel, abs_ = np.vectorize(self.accuracy, otypes=[float, float])(x)
        return np.abs(x) * rel + abs_

    def read(self, x):
        x = np.asarray(x, dtype=float)
        rel, abs_ = np.vectorize(self.accuracy, otypes=[float, float])(x)
        u1, u2 = self.u.draw(x.shape or None), self.u.draw(x.shape or None)
        f = self.f
        err = np.abs(x) * rel * (f * self.g + (1 - f) * u1) + abs_ * (f * self.o + (1 - f) * u2)
        y = x + err
        lo, hi = self.limits
        if lo is not None or hi is not None:
            y = np.clip(y, lo, hi)
        if self.resolution:
            y = np.round(y / self.resolution) * self.resolution
        return y


# ------------------------------------------------------------------ presets
# Numbers taken from the manufacturers' datasheets / product pages (see README).

def tfmini_s(seed=None, **kw):
    """Benewake TFmini-S lidar: +/-6 cm (0.1-6 m), +/-1% (6-12 m), 1 cm resolution, 0.1-12 m."""
    acc = lambda d: (0.0, 0.06) if d <= 6.0 else (0.01, 0.0)  # noqa: E731
    return SensorModel(acc, resolution=0.01, limits=(0.1, 12.0), seed=seed, **kw)


def ds18b20(seed=None, **kw):
    """Maxim/ADI DS18B20: +/-0.5 C from -10 to +85 C, 12-bit = 0.0625 C steps."""
    return SensorModel((0.0, 0.5), resolution=0.0625, limits=(-55.0, 125.0), seed=seed, **kw)


def ina219b_current(seed=None, shunt_ohm=0.1, **kw):
    """TI INA219B current via shunt: 0.5% max gain error over temperature, plus offset.

    Offset uses 100 uV, the larger of the two offset limits TI lists (conservative),
    divided by the shunt resistance to give amps.
    """
    return SensorModel((0.005, 100e-6 / shunt_ohm), seed=seed, **kw)
