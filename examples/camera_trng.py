"""
Camera random number generator -- live quality monitor
======================================================

Turns a webcam into a physical random number source and shows, live, how random
the numbers are.

How it works
  1. RAW: take the least significant bit (LSB) of every pixel, skipping pixels stuck
     at 0 or 255. The LSB is dominated by sensor noise (photon shot noise, thermal and
     read noise), which is physically random -- but the raw bits are still biased and
     correlated (flat areas, compression, auto-exposure), so they are NOT used directly.
  2. CONDITIONED: every 128 raw bytes are hashed with SHA-256 into 32 output bytes
     (4:1), chained with the previous hash. This is the standard way to turn a noisy,
     biased physical source into uniform bits.

Screen
  left    live camera, and the raw LSB bit-plane (structure here = non-random raw bits)
  middle  polar histogram of byte values 0..255 -- a perfect circle means uniform
  right   polar scatter: each number pair = (angle, radius) -- even fill means random
  bottom  entropy per byte (max 8.0) and chi-square z-score over time, raw vs conditioned

Controls
  W  show RAW or CONDITIONED in the polar plots
  S  save the last 1 MB of conditioned bytes to camera_random.bin (for ent / dieharder)
  R  reset statistics     SPACE pause     ESC quit

Run
  pip install pygame opencv-python numpy
  python examples/camera_trng.py            # default camera
  python examples/camera_trng.py --cam 1    # another camera
  python examples/camera_trng.py --fake     # no camera: simulated noisy frames

Tips: a normally lit scene or a covered lens both work (dark noise is still noise).
A saturated (white) or very flat, heavily compressed image gives less raw entropy.
This is a learning tool, not a certified TRNG: real TRNGs add health tests and
validated entropy estimates (NIST SP 800-90B).
"""
import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from camera_entropy import CameraSource, Extractor, FakeSource  # noqa: E402


def entropy_per_byte(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def chi2_z(counts):
    """Chi-square vs uniform over 256 bins, as a z-score (|z| < 3 is normal)."""
    n = counts.sum()
    if n == 0:
        return 0.0
    e = n / 256
    chi2 = float(((counts - e) ** 2 / e).sum())
    return (chi2 - 255) / math.sqrt(510)


# ---------------------------------------------------------------- drawing

W, H = 1280, 760
BG, PANEL, GRID = (15, 17, 26), (24, 27, 40), (45, 50, 70)
TXT, DIM = (225, 228, 240), (130, 136, 160)
INDIGO, ROSE, GREEN, AMBER = (129, 140, 248), (244, 63, 94), (16, 185, 129), (251, 191, 36)


def panel(screen, rect, title, small):
    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.rect(screen, GRID, rect, 1)
    screen.blit(small.render(title, True, TXT), (rect[0] + 8, rect[1] + 6))


def blit_frame(screen, arr, rect):
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))
    screen.blit(pygame.transform.smoothscale(surf, rect[2:]), rect[:2])


def polar_hist(screen, center, R, counts, color, fill):
    n = counts.sum()
    cx, cy = center
    if n == 0:
        return
    rel = np.clip(counts / (n / 256), 0, 1.5) / 1.5 * R
    ang = np.arange(256) / 256 * 2 * math.pi - math.pi / 2
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(rel, ang)]
    if fill:
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(s, color + (70,), pts)
        screen.blit(s, (0, 0))
    pygame.draw.lines(screen, color, True, pts, 2)
    for f in (0.5, 1.0, 1.5):
        pygame.draw.circle(screen, GRID if f != 1.0 else GREEN, center, int(R * f / 1.5), 1)


def polar_scatter(screen, center, R, pts, color):
    cx, cy = center
    pygame.draw.circle(screen, GRID, center, R, 1)
    for k in (0.33, 0.66):
        pygame.draw.circle(screen, GRID, center, int(R * k), 1)
    for i in range(8):
        a = i * math.pi / 4
        pygame.draw.line(screen, GRID, center, (cx + R * math.cos(a), cy + R * math.sin(a)), 1)
    n = len(pts)
    for i, (a, r) in enumerate(pts):
        shade = 0.25 + 0.75 * i / max(1, n)
        c = tuple(int(v * shade) for v in color)
        pygame.draw.circle(screen, c, (int(cx + R * r * math.cos(a)), int(cy + R * r * math.sin(a))), 1)


def line_plot(screen, rect, series, lo, hi, small, labels):
    x0, y0, w, h = rect
    for (data, col), lab in zip(series, labels):
        d = list(data)[-(w - 10):]
        if len(d) > 1:
            pts = [(x0 + 5 + i, y0 + h - 5 - (min(hi, max(lo, v)) - lo) / (hi - lo) * (h - 50))
                   for i, v in enumerate(d)]
            pygame.draw.lines(screen, col, False, pts, 2)
    for v in (lo, hi):
        yy = y0 + h - 5 - (v - lo) / (hi - lo) * (h - 50)
        screen.blit(small.render(f"{v:g}", True, DIM), (x0 + w - 34, yy - 8))


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="headless test: stop after N frames")
    ap.add_argument("--shot", default="camera_trng.png")
    args = ap.parse_args()

    src = FakeSource() if args.fake else CameraSource(args.cam)
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Camera random number generator")
    small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
    mid = pygame.font.SysFont("dejavusansmono,consolas,monospace", 18, bold=True)
    big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 30, bold=True)
    clock = pygame.time.Clock()

    ex = Extractor()

    def reset():
        return dict(raw=np.zeros(256, np.int64), out=np.zeros(256, np.int64),
                    pts_raw=deque(maxlen=4000), pts_out=deque(maxlen=4000),
                    h_raw=deque(maxlen=800), h_out=deque(maxlen=800), z_out=deque(maxlen=800),
                    total=0, t0=time.time())
    st = reset()
    pool = deque(maxlen=1 << 20)      # last 1 MB of conditioned bytes
    show_raw, paused, frame_i = False, False, 0
    last = dict(frame=None, raw=None, out=np.zeros(4, np.uint8), valid=0.0)
    msg, msg_t = "", 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_w:
                    show_raw = not show_raw
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    st = reset()
                elif e.key == pygame.K_s:
                    with open("camera_random.bin", "wb") as f:
                        f.write(bytes(pool))
                    msg, msg_t = f"saved {len(pool):,} bytes to camera_random.bin", time.time()

        if not paused:
            frame = src.read()
            if frame is not None:
                raw, out, valid = ex.process(frame)
                last.update(frame=frame, raw=raw, out=out if len(out) else last["out"], valid=valid)
                st["raw"] += np.bincount(raw, minlength=256)
                st["out"] += np.bincount(out, minlength=256)
                st["total"] += len(out)
                pool.extend(out.tobytes())
                # same sample size for both, so the comparison is fair
                st["h_raw"].append(entropy_per_byte(np.bincount(raw[:len(out)], minlength=256)))
                st["h_out"].append(entropy_per_byte(np.bincount(out, minlength=256)))
                st["z_out"].append(chi2_z(st["out"]))
                for arr, key in ((raw, "pts_raw"), (out, "pts_out")):
                    pairs = arr[: (min(len(arr), 400) // 2) * 2].reshape(-1, 2)
                    for b0, b1 in pairs:
                        st[key].append((b0 / 256 * 2 * math.pi, math.sqrt(b1 / 255)))

        # ---------------- draw
        screen.fill(BG)
        if last["frame"] is not None:
            panel(screen, (20, 36, 400, 330), f"live feed: {src.name}", small)
            blit_frame(screen, last["frame"], (25, 60, 390, 300))
            panel(screen, (20, 376, 400, 330), "raw LSB bit-plane (structure = bias)", small)
            lsb = ((last["frame"][:, :, 1] & 1) * 255).astype(np.uint8)
            blit_frame(screen, np.dstack([lsb] * 3), (25, 400, 390, 300))

        which = "RAW LSBs" if show_raw else "CONDITIONED (SHA-256)"
        col = ROSE if show_raw else INDIGO
        counts = st["raw"] if show_raw else st["out"]
        panel(screen, (430, 36, 410, 410), f"byte histogram, polar: {which}", small)
        polar_hist(screen, (635, 250), 175, counts, col, True)
        screen.blit(small.render("green circle = perfectly uniform", True, GREEN), (440, 424))
        panel(screen, (850, 36, 410, 410), f"random pairs (angle, radius): {which}", small)
        polar_scatter(screen, (1055, 250), 175, st["pts_raw"] if show_raw else st["pts_out"], col)

        panel(screen, (430, 456, 830, 250), "over time", small)
        line_plot(screen, (430, 476, 410, 228), [(st["h_raw"], ROSE), (st["h_out"], INDIGO)],
                  7.95, 8.0, small, ["raw", "out"])
        screen.blit(small.render("entropy per byte, per frame (8 = ideal)", True, TXT), (440, 480))
        screen.blit(small.render("raw", True, ROSE), (440, 498))
        screen.blit(small.render("conditioned", True, INDIGO), (480, 498))
        line_plot(screen, (850, 476, 410, 228), [(st["z_out"], AMBER)], -5, 5, small, ["z"])
        screen.blit(small.render("chi-square z, conditioned (|z|<3 = OK)", True, AMBER), (860, 480))
        zl = 476 + 228 - 5 - (0 + 5) / 10 * (228 - 50)
        pygame.draw.line(screen, GREEN, (855, zl), (1220, zl), 1)

        # header + numbers
        el = max(1e-6, time.time() - st["t0"])
        z = chi2_z(st["out"])
        ones_raw = (np.unpackbits(last["raw"]).mean() if last["raw"] is not None and len(last["raw"]) else 0)
        head = (f"{st['total']:,} random bytes   {st['total'] / el / 1024:7.1f} KB/s   "
                f"usable pixels {last['valid'] * 100:5.1f}%   raw 1-bits {ones_raw * 100:5.2f}%   "
                f"entropy raw {entropy_per_byte(st['raw']):.4f}  out {entropy_per_byte(st['out']):.4f}   "
                f"z {z:+.2f}{'  PAUSED' if paused else ''}")
        screen.blit(small.render(head, True, TXT), (20, 10))
        o = last["out"]
        n32 = int.from_bytes(bytes(o[:4]), "big") if len(o) >= 4 else 0
        screen.blit(big.render(f"0x{n32:08X}", True, INDIGO), (20, 712))
        screen.blit(mid.render(f"= {n32:>10d}   dice {n32 % 6 + 1}   0-99: {n32 % 100:02d}", True, TXT),
                    (230, 720))
        hint = msg if time.time() - msg_t < 3 else "W raw/conditioned   S save 1MB   R reset   SPACE pause   ESC quit"
        screen.blit(small.render(hint, True, DIM), (720, 728))

        pygame.display.flip()
        clock.tick(30)
        frame_i += 1
        if args.frames and frame_i >= args.frames:
            pygame.image.save(screen, args.shot)
            running = False
    src.close()
    pygame.quit()


if __name__ == "__main__":
    main()
