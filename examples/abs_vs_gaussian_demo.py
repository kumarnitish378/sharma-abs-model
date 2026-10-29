"""
Sharma ABS vs Gaussian -- three-way pygame demo
================================================

Three identical differential-drive rovers get the SAME wheel-speed commands and
drive a figure-8 by dead reckoning:

  LEFT   : ideal wheels (no noise)
  MIDDLE : real speed = command x M_abs,    M_abs   = 1 + alpha (eta/E[eta] - 1)
  RIGHT  : real speed = command x M_gauss,  M_gauss = 1 + sigma Z,  Z ~ N(0, 1)

sigma is matched to the ABS noise, so both noisy rovers have the SAME mean (1) and
the SAME standard deviation. The only difference is the SHAPE of the noise:

  * ABS is bounded and always positive: M is in [1 - alpha/3, 1 + alpha] roughly,
    and never below 0, so a wheel can never spin backwards.
  * Gaussian is unbounded: with enough noise M < 0 happens, i.e. the motor
    "reverses" -- something a real motor/encoder at constant command never does.
  * ABS is slightly right-skewed; Gaussian is symmetric.

At low noise the two look almost the same. Press X for the high-noise preset to
see where they differ.

Controls
  UP / DOWN     alpha  +/- 0.05
  RIGHT / LEFT  n      x2 / /2     (more n = less noise, more Gaussian-like)
  B             cycle bounds [-1,1] -> [-10,10] -> [-1000,1000]
  X             high-noise preset (alpha 1, n 1)  /  back to default
  F             fast-forward x1 / x4 / x16
  SPACE pause   R reset   C clear trails   ESC quit

Run:  pip install pygame    then    python examples/abs_vs_gaussian_demo.py
      add --camera to draw every random number from your webcam's sensor noise
      (needs opencv-python; --cam 1 for another camera, --fake-camera to test without one)
"""
import argparse
import math
import os
import random
import sys
from collections import deque

import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from camera_entropy import CameraRandom, add_rng_args, make_entropy, source_label  # noqa: E402


# ---------------------------------------------------------------- noise models
def eta_stats(c, n, rng):
    """Mean and sd of eta for bounds [-c, c]: exact when cheap, else estimated."""
    m = 2 * c + 1
    N = 2 * c * n
    if (n * (m - 1) + 1) * n * m < 2_000_000:          # exact PMF by convolution
        p = [1.0]
        for _ in range(n):
            q = [0.0] * (len(p) + m - 1)
            for i, pi in enumerate(p):
                w = pi / m
                for j in range(m):
                    q[i + j] += w
            p = q
        lo = -n * c
        e = [N / (N + lo + k) for k in range(len(p))]
        mean = sum(pk * ek for pk, ek in zip(p, e))
        var = sum(pk * (ek - mean) ** 2 for pk, ek in zip(p, e))
        return mean, math.sqrt(var)
    if n <= 32:                                         # Monte Carlo
        xs = []
        for _ in range(20000):
            s = sum(rng.randint(-c, c) for _ in range(n))
            xs.append(N / (N + s))
        mean = sum(xs) / len(xs)
        return mean, math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    v = (1 + 1 / c) / (12 * n)                          # paper's approximation (n large)
    return 1 + v, math.sqrt(v)


class Noise:
    def __init__(self, alpha, c, n, rng):
        self.alpha, self.c, self.n, self.rng = alpha, c, n, rng
        self.N = 2 * c * n
        self.eta_mean, self.eta_sd = eta_stats(c, n, random.Random(0))   # stats only
        self.sigma = alpha * self.eta_sd / self.eta_mean   # sd of the normalised ABS multiplier

    def abs_mult(self):
        r = self.rng.randint
        s = 0
        for _ in range(self.n):
            s += r(-self.c, self.c)
        eta = self.N / (self.N + s) / self.eta_mean
        return 1 + self.alpha * (eta - 1)

    def gauss_mult(self):
        return 1 + self.sigma * self.rng.gauss(0.0, 1.0)


# ---------------------------------------------------------------- settings
W, H = 1300, 830
PANEL_W, PANEL_H = 400, 400
GAP = 20
TOP = 66
BOTTOM_Y = TOP + PANEL_H + 16
DT = 1 / 60
V = 60.0
R = 78.0
TRACK = 26.0
TRAIL_MAX = 6000

BG = (15, 17, 26)
PANEL = (24, 27, 40)
GRID = (38, 42, 60)
TXT = (225, 228, 240)
DIM = (130, 136, 160)
GHOST = (60, 110, 95)
GREEN = (16, 185, 129)
INDIGO = (129, 140, 248)
AMBER = (251, 191, 36)
ROSE = (244, 63, 94)

BOUNDS = [1, 10, 1000]
SPEEDS = [1, 4, 16]
DEFAULT = (0.8, 4)        # alpha, n
EXTREME = (1.0, 1)


class Rover:
    def __init__(self):
        self.x, self.y, self.th = 0.0, 0.0, math.pi / 2
        self.trail = deque(maxlen=TRAIL_MAX)

    def step(self, vl, vr):
        v = (vl + vr) / 2
        self.x += v * math.cos(self.th) * DT
        self.y += v * math.sin(self.th) * DT
        self.th += (vr - vl) / TRACK * DT
        self.trail.append((self.x, self.y))


def command(t):
    lap = 2 * math.pi * R / V
    w = V / R if int(t // lap) % 2 == 0 else -V / R
    return V - w * TRACK / 2, V + w * TRACK / 2


class Track:
    """One noisy channel: rover + signal + multiplier history + event counters."""

    def __init__(self):
        self.rover = Rover()
        self.sig = deque(maxlen=900)
        self.mults = deque(maxlen=6000)
        self.neg = 0          # wheel reversed (M < 0)
        self.over = 0         # M > 2

    def push(self, vl, vr, ml, mr):
        self.rover.step(vl * ml, vr * mr)
        self.sig.append(vl * ml)
        for m in (ml, mr):
            self.mults.append(m)
            self.neg += m < 0
            self.over += m > 2


class Sim:
    def __init__(self, rng=None):
        self.alpha, self.n = DEFAULT
        self.bi = 1
        self.rng = rng if rng is not None else random.Random(2026)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.ideal = Rover()
        self.sig_ideal = deque(maxlen=900)
        self.abs, self.gauss = Track(), Track()
        self.rebuild()

    def rebuild(self):
        self.noise = Noise(self.alpha, BOUNDS[self.bi], self.n, self.rng)
        for tr in (self.abs, self.gauss):
            tr.mults.clear()
            tr.neg = tr.over = 0

    def step(self):
        vl, vr = command(self.t)
        self.ideal.step(vl, vr)
        self.sig_ideal.append(vl)
        nz = self.noise
        self.abs.push(vl, vr, nz.abs_mult(), nz.abs_mult())
        self.gauss.push(vl, vr, nz.gauss_mult(), nz.gauss_mult())
        self.t += DT


# ---------------------------------------------------------------- drawing
def to_screen(x, y):
    return PANEL_W / 2 + x, PANEL_H / 2 - y


def draw_rover(surf, rv, color):
    cx, cy = to_screen(rv.x, rv.y)
    th = -rv.th
    poly = [(cx + px * math.cos(th) - py * math.sin(th), cy + px * math.sin(th) + py * math.cos(th))
            for px, py in ((12, 0), (-9, 8), (-9, -8))]
    pygame.draw.polygon(surf, color, poly)
    pygame.draw.polygon(surf, TXT, poly, 1)


def draw_trail(surf, rv, color, width=2):
    if len(rv.trail) > 1:
        pygame.draw.lines(surf, color, False, [to_screen(*p) for p in rv.trail], width)


def stats(ms):
    if not ms:
        return 0.0, 0.0
    mean = sum(ms) / len(ms)
    return mean, math.sqrt(sum((m - mean) ** 2 for m in ms) / len(ms))


def draw_panel(screen, x0, title, color, sim, track, fonts):
    font, small = fonts
    surf = pygame.Surface((PANEL_W, PANEL_H))
    surf.fill(PANEL)
    for g in range(0, PANEL_W, 40):
        pygame.draw.line(surf, GRID, (g, 0), (g, PANEL_H))
        pygame.draw.line(surf, GRID, (0, g), (PANEL_W, g))
    if track is None:
        draw_trail(surf, sim.ideal, color)
        draw_rover(surf, sim.ideal, color)
    else:
        draw_trail(surf, sim.ideal, GHOST, 1)
        draw_trail(surf, track.rover, color)
        draw_rover(surf, track.rover, color)
    screen.blit(surf, (x0, TOP))
    pygame.draw.rect(screen, color, (x0, TOP, PANEL_W, PANEL_H), 1)
    screen.blit(font.render(title, True, color), (x0 + 2, TOP - 27))
    y = TOP + PANEL_H - 44
    if track is None:
        screen.blit(small.render("no noise: perfect figure-8 forever", True, DIM), (x0 + 8, y + 22))
        return
    rv, iv = track.rover, sim.ideal
    err = math.hypot(rv.x - iv.x, rv.y - iv.y)
    dth = math.degrees((rv.th - iv.th + math.pi) % (2 * math.pi) - math.pi)
    mean, sd = stats(track.mults)
    screen.blit(small.render(f"pos err {err:6.1f}px  head err {dth:+6.1f}deg", True, TXT), (x0 + 8, y))
    neg_col = ROSE if track.neg else GREEN
    screen.blit(small.render(f"M<0 (wheel reversed): {track.neg:<6d}", True, neg_col), (x0 + 8, y + 22))
    screen.blit(small.render(f"sd {sd:.3f}", True, DIM), (x0 + PANEL_W - 90, y + 22))


def draw_signal(screen, x0, y0, w, h, sig, color, label, sim, small):
    pygame.draw.rect(screen, PANEL, (x0, y0, w, h))
    pygame.draw.rect(screen, GRID, (x0, y0, w, h), 1)
    lo, hi = -V * 0.6, V * 2.6

    def py(v):
        v = max(lo, min(hi, v))
        return y0 + h - 4 - (v - lo) / (hi - lo) * (h - 8)

    pygame.draw.line(screen, ROSE, (x0, py(0)), (x0 + w, py(0)), 1)
    for buf, col, lw in ((sig, color, 1), (sim.sig_ideal, GREEN, 2)):
        data = list(buf)[-(w - 4):]
        if len(data) > 1:
            pygame.draw.lines(screen, col, False, [(x0 + 2 + i, py(v)) for i, v in enumerate(data)], lw)
    screen.blit(small.render(label, True, color), (x0 + 8, y0 + 4))
    screen.blit(small.render("0", True, ROSE), (x0 + w - 14, py(0) - 16))


def draw_hist(screen, sim, small):
    x0 = 20 + 2 * (PANEL_W + GAP) - 120
    y0 = BOTTOM_Y
    w, h = W - x0 - 20, H - BOTTOM_Y - 26
    pygame.draw.rect(screen, PANEL, (x0, y0, w, h))
    pygame.draw.rect(screen, GRID, (x0, y0, w, h), 1)
    lo, hi, bins = -0.5, 2.5, 120
    pw = w - 20

    def hist(ms):
        cnt = [0] * bins
        for m in ms:
            k = int((m - lo) / (hi - lo) * bins)
            cnt[min(bins - 1, max(0, k))] += 1
        return cnt

    ha, hg = hist(sim.abs.mults), hist(sim.gauss.mults)
    top = max(max(ha), max(hg)) or 1
    bw = pw / bins
    base = y0 + h - 22
    for k, c in enumerate(ha):
        bh = c / top * (h - 56)
        pygame.draw.rect(screen, INDIGO, (x0 + 10 + k * bw, base - bh, max(1, bw), bh))
    pts = [(x0 + 10 + (k + 0.5) * bw, base - c / top * (h - 56)) for k, c in enumerate(hg)]
    pygame.draw.lines(screen, AMBER, False, pts, 2)
    for v, col in ((0.0, ROSE), (1.0, GREEN), (2.0, DIM)):
        gx = x0 + 10 + (v - lo) / (hi - lo) * pw
        pygame.draw.line(screen, col, (gx, y0 + 40), (gx, base), 1)
        screen.blit(small.render(f"{v:.0f}", True, col), (gx - 4, base + 3))
    screen.blit(small.render("multiplier M:", True, TXT), (x0 + 8, y0 + 4))
    screen.blit(small.render("ABS (bars)", True, INDIGO), (x0 + 130, y0 + 4))
    screen.blit(small.render("Gaussian (line)", True, AMBER), (x0 + 240, y0 + 4))
    screen.blit(small.render("same mean & sd -- different shape", True, DIM), (x0 + 8, y0 + 22))


def main():
    ap = add_rng_args(argparse.ArgumentParser(description="Sharma ABS vs Gaussian demo"))
    args = ap.parse_args()
    entropy = make_entropy(args)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Sharma ABS vs Gaussian noise")
    font = pygame.font.SysFont("dejavusans,arial", 18, bold=True)
    small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
    clock = pygame.time.Clock()
    sim = Sim(CameraRandom(entropy) if entropy else None)
    paused, speed_i = False, 0
    frames = int(os.environ.get("ABS_DEMO_FRAMES", "0"))     # >0: headless test run
    if os.environ.get("ABS_DEMO_EXTREME"):
        sim.alpha, sim.n = EXTREME
        sim.rebuild()

    running, frame = True, 0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_UP:
                    sim.alpha = min(1.0, round(sim.alpha + 0.05, 2)); sim.rebuild()
                elif k == pygame.K_DOWN:
                    sim.alpha = max(0.0, round(sim.alpha - 0.05, 2)); sim.rebuild()
                elif k == pygame.K_RIGHT:
                    sim.n = min(256, sim.n * 2); sim.rebuild()
                elif k == pygame.K_LEFT:
                    sim.n = max(1, sim.n // 2); sim.rebuild()
                elif k == pygame.K_b:
                    sim.bi = (sim.bi + 1) % len(BOUNDS); sim.rebuild()
                elif k == pygame.K_x:
                    sim.alpha, sim.n = EXTREME if (sim.alpha, sim.n) != EXTREME else DEFAULT
                    sim.rebuild()
                elif k == pygame.K_f:
                    speed_i = (speed_i + 1) % len(SPEEDS)
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    sim.reset()
                elif k == pygame.K_c:
                    for rv in (sim.ideal, sim.abs.rover, sim.gauss.rover):
                        rv.trail.clear()

        if not paused:
            for _ in range(SPEEDS[speed_i]):
                sim.step()

        screen.fill(BG)
        fonts = (font, small)
        draw_panel(screen, 20, "NO NOISE (ideal)", GREEN, sim, None, fonts)
        draw_panel(screen, 20 + PANEL_W + GAP, "SHARMA ABS NOISE", INDIGO, sim, sim.abs, fonts)
        draw_panel(screen, 20 + 2 * (PANEL_W + GAP), "GAUSSIAN NOISE", AMBER, sim, sim.gauss, fonts)

        sw = 2 * (PANEL_W + GAP) - 120 - 20 - 20
        sh = (H - BOTTOM_Y - 26 - 8) // 2
        draw_signal(screen, 20, BOTTOM_Y, sw, sh, sim.abs.sig, INDIGO,
                    "left wheel speed: ideal vs ABS", sim, small)
        draw_signal(screen, 20, BOTTOM_Y + sh + 8, sw, sh, sim.gauss.sig, AMBER,
                    "left wheel speed: ideal vs Gaussian", sim, small)
        draw_hist(screen, sim, small)

        nz = sim.noise
        status = (f"t {sim.t:6.1f}s  x{SPEEDS[speed_i]}{'  PAUSED' if paused else ''}   |   "
                  f"alpha {sim.alpha:.2f}   n {sim.n}   bounds [-{BOUNDS[sim.bi]},{BOUNDS[sim.bi]}]   |   "
                  f"both noises: mean 1, sd {nz.sigma:.4f}")
        screen.blit(small.render(status, True, TXT), (20, 6))
        help_txt = ("UP/DOWN alpha  LEFT/RIGHT n  B bounds  X high-noise  "
                    "F speed  SPACE pause  R reset  C clear")
        screen.blit(small.render(help_txt, True, DIM), (20, H - 20))
        src_txt = "random source: " + source_label(entropy)
        screen.blit(small.render(src_txt, True, GREEN if entropy else DIM),
                    (W - 20 - small.size(src_txt)[0], H - 20))

        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if frames and frame >= frames:
            pygame.image.save(screen, os.environ.get("ABS_DEMO_SHOT", "abs_vs_gaussian.png"))
            running = False
    if entropy:
        entropy.close()
    pygame.quit()


if __name__ == "__main__":
    main()
