"""
Sharma ABS Model -- side-by-side pygame demo
=============================================

Two identical differential-drive rovers get the SAME wheel-speed commands and
drive a figure-8 by dead reckoning.

  LEFT  : ideal wheels (no noise)
  RIGHT : each wheel's real speed = command x [1 + alpha (eta - 1)]   (Sharma ABS)

The ideal path is drawn as a faint ghost in the right panel, so you can watch the
noisy rover drift away from it. The bottom strip shows the left-wheel speed
signal (ideal vs noisy) and a histogram of the multiplier M = 1 + alpha(eta - 1).

Controls
  UP / DOWN     alpha  +/- 0.05   (0 = no noise, 1 = full noise)
  RIGHT / LEFT  n      x2 / /2    (complexity: more n = less noise)
  B             cycle bounds [-1,1] -> [-10,10] -> [-1000,1000]
                (watch: the noise barely changes -- a result from the paper)
  M             toggle mean-correction (eta / E[eta])
  F             fast-forward x1 / x4 / x16
  SPACE pause   R reset   C clear trails   ESC quit

Run:  pip install pygame    then    python examples/abs_pygame_demo.py
"""
import math
import os
import random
from collections import deque

import pygame

# ---------------------------------------------------------------- ABS model
class ABS:
    """Minimal Sharma ABS factor: eta = N / (N + S)."""

    def __init__(self, c, n, normalize, rng):
        self.a, self.b, self.n = -c, c, n
        self.N = n * 2 * c
        self.rng = rng
        # E[eta] ~= 1 + (1 + 1/c) / (12 n) for symmetric bounds (paper, Eq. 8)
        self.mean = 1 + (1 + 1 / c) / (12 * n) if normalize else 1.0

    def eta(self):
        r = self.rng.randint
        s = 0
        for _ in range(self.n):
            s += r(self.a, self.b)
        return self.N / (self.N + s) / self.mean


# ---------------------------------------------------------------- settings
W, H = 1240, 790
PANEL_W, PANEL_H = 590, 440
TOP = 66
BOTTOM_Y = TOP + PANEL_H + 20
DT = 1 / 60                  # simulation step (s)
V = 70.0                     # forward speed (px/s)
R = 105.0                    # figure-8 lobe radius (px)
TRACK = 30.0                 # wheel separation (px)
TRAIL_MAX = 6000

BG = (15, 17, 26)
PANEL = (24, 27, 40)
GRID = (38, 42, 60)
TXT = (225, 228, 240)
DIM = (130, 136, 160)
GREEN = (16, 185, 129)
INDIGO = (129, 140, 248)
ROSE = (244, 63, 94)
AMBER = (251, 191, 36)

BOUNDS = [1, 10, 1000]
SPEEDS = [1, 4, 16]


class Rover:
    def __init__(self):
        self.x, self.y, self.th = 0.0, 0.0, math.pi / 2   # start at origin, heading "north"
        self.trail = deque(maxlen=TRAIL_MAX)

    def step(self, vl, vr):
        v = (vl + vr) / 2
        w = (vr - vl) / TRACK
        self.x += v * math.cos(self.th) * DT
        self.y += v * math.sin(self.th) * DT
        self.th += w * DT
        self.trail.append((self.x, self.y))


def command(t):
    """Figure-8: one full left circle, then one full right circle, forever."""
    lap = 2 * math.pi * R / V
    w = V / R if int(t // lap) % 2 == 0 else -V / R
    return V - w * TRACK / 2, V + w * TRACK / 2


class Sim:
    def __init__(self):
        self.alpha, self.n, self.bi, self.normalize = 0.6, 4, 1, True
        self.rng = random.Random(2026)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.ideal, self.noisy, self.ghost = Rover(), Rover(), Rover()
        self.sig_ideal = deque(maxlen=PANEL_W * 2 - 40)
        self.sig_noisy = deque(maxlen=PANEL_W * 2 - 40)
        self.mults = deque(maxlen=4000)
        self.rebuild()

    def rebuild(self):
        self.abs = ABS(BOUNDS[self.bi], self.n, self.normalize, self.rng)

    def mult(self):
        return 1 + self.alpha * (self.abs.eta() - 1)

    def step(self):
        vl, vr = command(self.t)
        ml, mr = self.mult(), self.mult()
        self.ideal.step(vl, vr)
        self.ghost.step(vl, vr)
        self.noisy.step(vl * ml, vr * mr)
        self.sig_ideal.append(vl)
        self.sig_noisy.append(vl * ml)
        self.mults.extend((ml, mr))
        self.t += DT


# ---------------------------------------------------------------- drawing
def to_screen(x, y):
    return PANEL_W / 2 + x, PANEL_H / 2 - y


def draw_rover(surf, rv, color):
    cx, cy = to_screen(rv.x, rv.y)
    th = -rv.th
    pts = [(14, 0), (-10, 9), (-10, -9)]
    poly = [(cx + px * math.cos(th) - py * math.sin(th),
             cy + px * math.sin(th) + py * math.cos(th)) for px, py in pts]
    pygame.draw.polygon(surf, color, poly)
    pygame.draw.polygon(surf, TXT, poly, 1)


def draw_trail(surf, rv, color, width=2):
    if len(rv.trail) > 1:
        pygame.draw.lines(surf, color, False, [to_screen(*p) for p in rv.trail], width)


def draw_panel(screen, x0, title, color, sim, noisy, font, small):
    surf = pygame.Surface((PANEL_W, PANEL_H))
    surf.fill(PANEL)
    for gx in range(0, PANEL_W, 40):
        pygame.draw.line(surf, GRID, (gx, 0), (gx, PANEL_H))
    for gy in range(0, PANEL_H, 40):
        pygame.draw.line(surf, GRID, (0, gy), (PANEL_W, gy))
    if noisy:
        draw_trail(surf, sim.ghost, (60, 110, 95), 1)
        draw_trail(surf, sim.noisy, color)
        draw_rover(surf, sim.noisy, color)
    else:
        draw_trail(surf, sim.ideal, color)
        draw_rover(surf, sim.ideal, color)
    screen.blit(surf, (x0, TOP))
    pygame.draw.rect(screen, color, (x0, TOP, PANEL_W, PANEL_H), 1)
    screen.blit(font.render(title, True, color), (x0 + 2, TOP - 27))
    if noisy:
        err = math.hypot(sim.noisy.x - sim.ideal.x, sim.noisy.y - sim.ideal.y)
        dth = math.degrees((sim.noisy.th - sim.ideal.th + math.pi) % (2 * math.pi) - math.pi)
        screen.blit(small.render(f"position error {err:6.1f} px   heading error {dth:+6.1f} deg",
                                 True, AMBER), (x0 + 10, TOP + PANEL_H - 22))
    else:
        screen.blit(small.render("drives a perfect figure-8 forever", True, DIM),
                    (x0 + 10, TOP + PANEL_H - 22))


def draw_signal(screen, sim, small):
    x0, y0, w, h = 20, BOTTOM_Y, PANEL_W + 140, H - BOTTOM_Y - 20
    pygame.draw.rect(screen, PANEL, (x0, y0, w, h))
    pygame.draw.rect(screen, GRID, (x0, y0, w, h), 1)
    lo, hi = 0.0, V * 2.2

    def py(v):
        return y0 + h - 6 - (v - lo) / (hi - lo) * (h - 12)

    pygame.draw.line(screen, ROSE, (x0, py(0)), (x0 + w, py(0)), 1)
    for buf, col, lw in ((sim.sig_noisy, INDIGO, 1), (sim.sig_ideal, GREEN, 2)):
        data = list(buf)[-(w - 4):]
        if len(data) > 1:
            pygame.draw.lines(screen, col, False,
                              [(x0 + 2 + i, py(v)) for i, v in enumerate(data)], lw)
    screen.blit(small.render("left wheel speed (px/s):  ideal", True, GREEN), (x0 + 8, y0 + 6))
    lab = small.render("left wheel speed (px/s):  ideal", True, GREEN)
    screen.blit(small.render("  vs  with ABS", True, INDIGO), (x0 + 8 + lab.get_width(), y0 + 6))
    screen.blit(small.render("red line = 0  (ABS never crosses it for alpha <= 1)", True, ROSE),
                (x0 + 8, y0 + h - 20))


def draw_hist(screen, sim, small):
    x0, y0 = 20 + PANEL_W + 160, BOTTOM_Y
    w, h = W - x0 - 20, H - BOTTOM_Y - 20
    pygame.draw.rect(screen, PANEL, (x0, y0, w, h))
    pygame.draw.rect(screen, GRID, (x0, y0, w, h), 1)
    lo, hi, bins = 0.0, 2.0, 80
    counts = [0] * bins
    for m in sim.mults:
        k = int((m - lo) / (hi - lo) * bins)
        if 0 <= k < bins:
            counts[k] += 1
    top = max(counts) or 1
    bw = (w - 20) / bins
    for k, c in enumerate(counts):
        bh = c / top * (h - 50)
        pygame.draw.rect(screen, INDIGO, (x0 + 10 + k * bw, y0 + h - 22 - bh, max(1, bw - 1), bh))
    for v, col in ((0.0, ROSE), (1.0, GREEN), (2.0, DIM)):
        gx = x0 + 10 + (v - lo) / (hi - lo) * (w - 20)
        pygame.draw.line(screen, col, (gx, y0 + 26), (gx, y0 + h - 22), 1)
        screen.blit(small.render(f"{v:.0f}", True, col), (gx - 4, y0 + h - 20))
    screen.blit(small.render("histogram of multiplier M = 1 + alpha(eta - 1)", True, TXT),
                (x0 + 8, y0 + 6))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Sharma ABS Model - with vs without noise")
    font = pygame.font.SysFont("dejavusans,arial", 20, bold=True)
    small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
    clock = pygame.time.Clock()
    sim = Sim()
    paused, speed_i = False, 0
    frames = int(os.environ.get("ABS_DEMO_FRAMES", "0"))  # >0: headless test run

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
                    sim.alpha = min(1.0, round(sim.alpha + 0.05, 2))
                elif k == pygame.K_DOWN:
                    sim.alpha = max(0.0, round(sim.alpha - 0.05, 2))
                elif k == pygame.K_RIGHT:
                    sim.n = min(256, sim.n * 2); sim.rebuild(); sim.mults.clear()
                elif k == pygame.K_LEFT:
                    sim.n = max(1, sim.n // 2); sim.rebuild(); sim.mults.clear()
                elif k == pygame.K_b:
                    sim.bi = (sim.bi + 1) % len(BOUNDS); sim.rebuild(); sim.mults.clear()
                elif k == pygame.K_m:
                    sim.normalize = not sim.normalize; sim.rebuild(); sim.mults.clear()
                elif k == pygame.K_f:
                    speed_i = (speed_i + 1) % len(SPEEDS)
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    sim.reset()
                elif k == pygame.K_c:
                    for rv in (sim.ideal, sim.noisy, sim.ghost):
                        rv.trail.clear()

        if not paused:
            for _ in range(SPEEDS[speed_i]):
                sim.step()

        screen.fill(BG)
        draw_panel(screen, 20, "WITHOUT ABS  (ideal)", GREEN, sim, False, font, small)
        draw_panel(screen, W - 20 - PANEL_W, "WITH SHARMA ABS NOISE", INDIGO, sim, True, font, small)
        draw_signal(screen, sim, small)
        draw_hist(screen, sim, small)

        ms = sim.mults
        c = BOUNDS[sim.bi]
        theory = sim.alpha * math.sqrt((1 + 1 / c) / (12 * sim.n))
        if ms:
            mean = sum(ms) / len(ms)
            sd = math.sqrt(sum((m - mean) ** 2 for m in ms) / len(ms))
            meas = f"measured sd {sd:.4f}  mean {mean:.4f}  min {min(ms):.3f}  max {max(ms):.3f}"
        else:
            meas = ""
        status = (f"t {sim.t:6.1f}s  x{SPEEDS[speed_i]}{'  PAUSED' if paused else ''}   |   "
                  f"alpha {sim.alpha:.2f}   n {sim.n}   bounds [-{c},{c}]   "
                  f"mean-corr {'ON' if sim.normalize else 'OFF'}   theory sd ~{theory:.4f}")
        screen.blit(small.render(status, True, TXT), (20, 6))
        screen.blit(small.render(meas, True, AMBER), (W - 20 - small.size(meas)[0], 6))
        help_txt = "UP/DOWN alpha  LEFT/RIGHT n  B bounds  M mean-corr  F speed  SPACE pause  R reset  C clear"
        screen.blit(small.render(help_txt, True, DIM), (20, H - 18))

        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if frames and frame >= frames:
            pygame.image.save(screen, os.environ.get("ABS_DEMO_SHOT", "abs_demo.png"))
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
