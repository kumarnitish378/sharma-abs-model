"""
Live datasheet sensor demo -- Sharma ABS vs Gaussian
====================================================

Pick a real sensor. It measures a physical value that changes over time, and two
simulated copies of the sensor report readings:

  ABS model      : errors built from the bounded ABS factor (src/sensor_model.py)
  Gaussian model : same per-device and per-sample spread, Gaussian shape

The datasheet accuracy limit is drawn live. Readings outside it turn red and are counted.
A real unit that meets its datasheet never produces them.

Sensors (numbers from the manufacturers' datasheets)
  1  Benewake TFmini-S lidar    : +/-6 cm to 6 m, +/-1% 6-12 m, 1 cm steps
  2  DS18B20 temperature        : +/-0.5 C (-10 to +85 C), 0.0625 C steps
  3  INA219B current (0.1 ohm)  : 0.5% gain error + 100 uV offset

Controls
  1 / 2 / 3     choose sensor
  LEFT / RIGHT  n -/+  (lower n = noisier, closer to the limit: a stress test)
  S             error shape: symmetric / skewed
  D             new device (a different physical unit, new fixed errors)
  F             speed x1 / x4     SPACE pause     R reset counters     ESC quit

Run:  python examples/live_sensor_demo.py            (add --camera for webcam randomness)
"""
import argparse
import math
import os
import sys
from collections import deque

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sensor_model import ds18b20, ina219b_current, tfmini_s  # noqa: E402
from camera_entropy import CameraNumpyRNG, add_rng_args, make_entropy, source_label  # noqa: E402

W, H = 1280, 790
BG, PANEL, GRID = (15, 17, 26), (24, 27, 40), (45, 50, 70)
TXT, DIM = (225, 228, 240), (130, 136, 160)
GREEN, INDIGO, AMBER, ROSE = (16, 185, 129), (129, 140, 248), (251, 191, 36), (244, 63, 94)
RATE = 20            # readings per second
HISTORY = 20 * RATE  # seconds shown in the plots


# ---------------------------------------------------------------- sensors

def lidar_truth(t):
    """Target moving between 0.3 m and 11.7 m."""
    return 6.0 + 5.7 * math.sin(2 * math.pi * t / 24 - math.pi / 2)


def temp_truth(t):
    """Heat-up / cool-down cycle between -5 and 80 C."""
    ph = (t % 40) / 40
    return -5 + 85 * (ph / 0.45 if ph < 0.45 else 1.0 if ph < 0.55 else (1 - ph) / 0.45)


def current_truth(t):
    """Motor load profile: idle, ramp, steady, step, back to idle (amps)."""
    ph = t % 30
    if ph < 4:
        return 0.15
    if ph < 10:
        return 0.15 + (ph - 4) / 6 * 1.6
    if ph < 16:
        return 1.75 + 0.05 * math.sin(2 * math.pi * ph)
    if ph < 22:
        return 2.9
    return 0.6


SENSORS = [
    dict(name="Benewake TFmini-S lidar", make=tfmini_s, truth=lidar_truth, unit="m", fmt="{:6.2f}",
         spec="±6 cm (0.1-6 m), ±1% (6-12 m), 1 cm steps", lo=0.0, hi=12.0),
    dict(name="DS18B20 temperature", make=ds18b20, truth=temp_truth, unit="°C", fmt="{:7.3f}",
         spec="±0.5 °C (-10 to +85 °C), 0.0625 °C steps", lo=-10.0, hi=90.0),
    dict(name="INA219B current, 0.1 Ω shunt", make=ina219b_current, truth=current_truth, unit="A",
         fmt="{:6.4f}", spec="0.5% gain error + 100 µV offset (1 mA)", lo=0.0, hi=3.2),
]


class GaussianUnit:
    """Same sd as the ABS unit variable, Gaussian shape (unbounded)."""

    def __init__(self, sd, rng):
        self.sd, self.rng = sd, rng

    def draw(self, size=None):
        v = self.rng.normal(0.0, self.sd, size)
        return v if size is not None else float(v)


class Channel:
    def __init__(self, model):
        self.model = model
        self.read_hist = deque(maxlen=HISTORY)
        self.err_hist = deque(maxlen=HISTORY)       # error / datasheet limit
        self.count = self.out = 0
        self.worst = 0.0
        self.last = float("nan")

    def sample(self, x):
        y = float(self.model.read(np.array([x]))[0])
        lim = float(self.model.spec_limit(np.array([x]))[0]) + (self.model.resolution or 0) / 2
        r = (y - x) / lim
        self.last = y
        self.read_hist.append(y)
        self.err_hist.append(r)
        self.count += 1
        self.out += abs(r) > 1 + 1e-9
        self.worst = max(self.worst, abs(r))


class Demo:
    def __init__(self, rng):
        self.rng = rng
        self.si, self.n, self.shape, self.device = 0, 1, "symmetric", 1
        self.build()

    def build(self):
        s = SENSORS[self.si]
        seed = 1000 + self.device
        a = s["make"](seed=seed, n=self.n, shape=self.shape, rng=self.rng)
        g = s["make"](seed=seed + 50000, n=self.n, shape=self.shape, rng=self.rng)
        g.u = GaussianUnit(g.u.sd, g.rng)
        g.new_device()
        self.abs, self.gauss = Channel(a), Channel(g)
        self.truth_hist = deque(maxlen=HISTORY)
        self.t = 0.0

    def step(self):
        x = SENSORS[self.si]["truth"](self.t)
        self.truth_hist.append(x)
        self.abs.sample(x)
        self.gauss.sample(x)
        self.t += 1 / RATE


# ---------------------------------------------------------------- drawing

def panel(screen, rect, title, f):
    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.rect(screen, GRID, rect, 1)
    screen.blit(f.render(title, True, TXT), (rect[0] + 10, rect[1] + 8))


def draw_scene(screen, demo, rect, fonts):
    small, mid, big = fonts
    x0, y0, w, h = rect
    s = SENSORS[demo.si]
    x = demo.truth_hist[-1] if demo.truth_hist else 0
    panel(screen, rect, f"scene: {s['name']}", small)
    if demo.si == 0:                       # lidar: sensor on the left, target moving
        gy = y0 + h * 0.62
        pygame.draw.line(screen, GRID, (x0 + 20, gy + 28), (x0 + w - 20, gy + 28), 2)
        sx = x0 + 40
        scale = (w - 120) / 12.0
        for m in range(0, 13, 2):
            px = sx + m * scale
            pygame.draw.line(screen, GRID, (px, gy + 28), (px, gy + 36), 1)
            screen.blit(small.render(f"{m}m", True, DIM), (px - 8, gy + 40))
        tx = sx + x * scale
        pygame.draw.rect(screen, (60, 70, 100), (sx - 26, gy - 18, 30, 36), border_radius=4)
        pygame.draw.circle(screen, ROSE, (sx + 4, gy), 5)
        pygame.draw.line(screen, ROSE, (sx + 6, gy), (tx, gy), 1)
        pygame.draw.rect(screen, (150, 155, 175), (tx, gy - 60, 14, 88))
        ra = demo.abs.last
        if ra == ra:
            pygame.draw.line(screen, INDIGO, (sx + ra * scale, gy - 70), (sx + ra * scale, gy - 62), 3)
        screen.blit(small.render("target", True, DIM), (tx - 10, gy - 82))
    elif demo.si == 1:                     # thermometer column
        cx, top, bot = x0 + w // 2, y0 + 50, y0 + h - 60
        pygame.draw.rect(screen, (60, 70, 100), (cx - 14, top, 28, bot - top), border_radius=14)
        frac = (x - s["lo"]) / (s["hi"] - s["lo"])
        lvl = bot - frac * (bot - top)
        pygame.draw.rect(screen, ROSE, (cx - 8, lvl, 16, bot - lvl), border_radius=8)
        pygame.draw.circle(screen, ROSE, (cx, bot + 14), 22)
        for tval in range(-10, 91, 20):
            yy = bot - (tval - s["lo"]) / (s["hi"] - s["lo"]) * (bot - top)
            pygame.draw.line(screen, GRID, (cx + 20, yy), (cx + 34, yy), 1)
            screen.blit(small.render(f"{tval}°C", True, DIM), (cx + 40, yy - 8))
        screen.blit(small.render("heater cycle", True, DIM), (x0 + 20, y0 + h - 30))
    else:                                  # current: motor + gauge
        cx, cy, R = x0 + w // 2, y0 + h // 2 + 20, 100
        pygame.draw.arc(screen, GRID, (cx - R, cy - R, 2 * R, 2 * R), math.pi * 0.15, math.pi * 0.85, 8)
        frac = (x - s["lo"]) / (s["hi"] - s["lo"])
        a = math.pi * (0.85 - 0.7 * frac)
        pygame.draw.line(screen, AMBER, (cx, cy), (cx + R * math.cos(a), cy - R * math.sin(a)), 4)
        pygame.draw.circle(screen, (60, 70, 100), (cx, cy), 12)
        for v in (0, 1, 2, 3):
            aa = math.pi * (0.85 - 0.7 * (v - s["lo"]) / (s["hi"] - s["lo"]))
            screen.blit(small.render(f"{v}A", True, DIM),
                        (cx + (R + 18) * math.cos(aa) - 10, cy - (R + 18) * math.sin(aa) - 8))
        rot = demo.t * x * 6
        for k in range(3):
            aa = rot + k * 2 * math.pi / 3
            pygame.draw.line(screen, DIM, (x0 + 90, cy + 40),
                             (x0 + 90 + 28 * math.cos(aa), cy + 40 + 28 * math.sin(aa)), 3)
        pygame.draw.circle(screen, GRID, (x0 + 90, cy + 40), 32, 2)
        screen.blit(small.render("motor load", True, DIM), (x0 + 58, cy + 80))
    screen.blit(mid.render(f"true value: {s['fmt'].format(x)} {s['unit']}", True, GREEN),
                (x0 + 10, y0 + 32))


def draw_readouts(screen, demo, rect, fonts):
    small, mid, big = fonts
    x0, y0, w, h = rect
    s = SENSORS[demo.si]
    panel(screen, rect, f"readings  -  device #{demo.device}   spec: {s['spec']}", small)
    x = demo.truth_hist[-1] if demo.truth_hist else 0
    rows = [("TRUE VALUE", GREEN, x, None), ("ABS MODEL", INDIGO, demo.abs.last, demo.abs),
            ("GAUSSIAN MODEL", AMBER, demo.gauss.last, demo.gauss)]
    for i, (label, col, val, ch) in enumerate(rows):
        yy = y0 + 44 + i * 92
        screen.blit(small.render(label, True, col), (x0 + 20, yy))
        bad = ch is not None and ch.err_hist and abs(ch.err_hist[-1]) > 1 + 1e-9
        txt = (s["fmt"].format(val) if val == val else "  --") + " " + s["unit"]
        screen.blit(big.render(txt, True, ROSE if bad else col), (x0 + 20, yy + 18))
        if ch is not None:
            pct = 100 * ch.out / max(1, ch.count)
            screen.blit(small.render(f"out of spec: {ch.out} / {ch.count}  ({pct:.2f}%)", True,
                                     ROSE if ch.out else GREEN), (x0 + w - 330, yy + 10))
            screen.blit(small.render(f"worst error: {ch.worst * 100:5.1f}% of limit", True,
                                     ROSE if ch.worst > 1 else DIM), (x0 + w - 330, yy + 32))
            screen.blit(small.render(f"error now:   {ch.err_hist[-1] * 100:+6.1f}% of limit" if ch.err_hist
                                     else "", True, DIM), (x0 + w - 330, yy + 54))


def draw_signal(screen, demo, rect, fonts):
    small = fonts[0]
    x0, y0, w, h = rect
    s = SENSORS[demo.si]
    panel(screen, rect, "readings over time", small)
    lo, hi = s["lo"], s["hi"]
    pw = w - 20

    def P(i, v, n):
        return (x0 + 10 + (i + HISTORY - n) * pw / HISTORY,
                y0 + h - 10 - (min(hi, max(lo, v)) - lo) / (hi - lo) * (h - 50))

    for data, col, lw in ((demo.gauss.read_hist, AMBER, 1), (demo.abs.read_hist, INDIGO, 1),
                          (demo.truth_hist, GREEN, 2)):
        d = list(data)
        if len(d) > 1:
            pygame.draw.lines(screen, col, False, [P(i, v, len(d)) for i, v in enumerate(d)], lw)
    screen.blit(small.render(f"{hi:g} {s['unit']}", True, DIM), (x0 + w - 70, y0 + 30))
    screen.blit(small.render(f"{lo:g} {s['unit']}", True, DIM), (x0 + w - 70, y0 + h - 26))


def draw_error(screen, demo, rect, fonts):
    small = fonts[0]
    x0, y0, w, h = rect
    panel(screen, rect, "reading error / datasheet limit   (green lines = the limit)", small)
    lo, hi = -2.0, 2.0
    pw = w - 20

    def Y(v):
        return y0 + h - 10 - (min(hi, max(lo, v)) - lo) / (hi - lo) * (h - 50)

    for v, col in ((1, GREEN), (-1, GREEN), (0, GRID)):
        pygame.draw.line(screen, col, (x0 + 10, Y(v)), (x0 + w - 10, Y(v)), 1)
    for v in (2, 1, 0, -1, -2):
        screen.blit(small.render(f"{v:+d}", True, DIM), (x0 + w - 34, Y(v) - 8))
    for ch, col in ((demo.gauss, AMBER), (demo.abs, INDIGO)):
        d = list(ch.err_hist)
        n = len(d)
        for i, r in enumerate(d):
            px = x0 + 10 + (i + HISTORY - n) * pw / HISTORY
            bad = abs(r) > 1 + 1e-9
            pygame.draw.circle(screen, ROSE if bad else col, (int(px), int(Y(r))), 4 if bad else 2)
    screen.blit(small.render("ABS", True, INDIGO), (x0 + 12, y0 + 30))
    screen.blit(small.render("Gaussian", True, AMBER), (x0 + 50, y0 + 30))
    screen.blit(small.render("out of spec", True, ROSE), (x0 + 130, y0 + 30))


def main():
    ap = add_rng_args(argparse.ArgumentParser(description="live datasheet sensor demo"))
    ap.add_argument("--frames", type=int, default=0, help="headless test: stop after N frames")
    ap.add_argument("--shot", default="live_sensor_demo.png")
    ap.add_argument("--sensor", type=int, default=1, choices=(1, 2, 3))
    args = ap.parse_args()
    entropy = make_entropy(args)
    rng = CameraNumpyRNG(entropy) if entropy else None

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Datasheet sensor: Sharma ABS vs Gaussian")
    small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
    mid = pygame.font.SysFont("dejavusansmono,consolas,monospace", 18, bold=True)
    big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 34, bold=True)
    fonts = (small, mid, big)
    clock = pygame.time.Clock()
    demo = Demo(rng)
    demo.si = args.sensor - 1
    demo.build()
    paused, speed, frame, acc = False, 1, 0, 0.0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                    demo.si = k - pygame.K_1; demo.build()
                elif k == pygame.K_RIGHT:
                    demo.n = min(64, demo.n * 2); demo.build()
                elif k == pygame.K_LEFT:
                    demo.n = max(1, demo.n // 2); demo.build()
                elif k == pygame.K_s:
                    demo.shape = "skewed" if demo.shape == "symmetric" else "symmetric"; demo.build()
                elif k == pygame.K_d:
                    demo.device += 1; demo.build()
                elif k == pygame.K_r:
                    demo.build()
                elif k == pygame.K_f:
                    speed = 4 if speed == 1 else 1
                elif k == pygame.K_SPACE:
                    paused = not paused

        dt = clock.tick(30) / 1000
        if not paused:
            # headless test runs use a fixed 1/30 s step; live runs follow the real clock
            acc += (1 / 30 if args.frames else dt) * speed * RATE
            while acc >= 1:
                demo.step()
                acc -= 1

        screen.fill(BG)
        head = (f"t {demo.t:6.1f}s  x{speed}{'  PAUSED' if paused else ''}   |   n {demo.n}"
                f"{' (stress test)' if demo.n == 1 else ''}   shape {demo.shape}   |   "
                f"same spread for both models: only the shape differs")
        screen.blit(small.render(head, True, TXT), (20, 8))
        draw_scene(screen, demo, (20, 34, 520, 330), fonts)
        draw_readouts(screen, demo, (550, 34, 710, 330), fonts)
        draw_signal(screen, demo, (20, 374, 520, 380), fonts)
        draw_error(screen, demo, (550, 374, 710, 380), fonts)
        help_txt = "1/2/3 sensor  LEFT/RIGHT n  S shape  D new device  F speed  SPACE pause  R reset"
        screen.blit(small.render(help_txt, True, DIM), (20, H - 22))
        src = "random source: " + source_label(entropy)
        screen.blit(small.render(src, True, GREEN if entropy else DIM), (W - 20 - small.size(src)[0], H - 22))
        pygame.display.flip()

        frame += 1
        if args.frames and frame >= args.frames:
            pygame.image.save(screen, args.shot)
            running = False
    if entropy:
        entropy.close()
    pygame.quit()


if __name__ == "__main__":
    main()
