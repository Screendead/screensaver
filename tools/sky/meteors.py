#!/usr/bin/env python3
# The shooting stars meteor() in Shaders/Saver.metal draws for a seed, timed against the live sky clock (skyclock.py, as
# SkyClock in Sources/StrataCore/Sky/SkyClock.swift, stepped at 60 Hz from the saver's start). The hashes and constants must match the shader.
#   tools/sky/meteors.py <seed> <from s> <to s>   every meteor in a range: start, duration, screen start, brightness
#   tools/sky/meteors.py <seed> --nights N        meteors per night and the gaps between them over the first N nights
import sys

import numpy as np

from skyclock import advance, elevation

M = 0xffffffff
SLOT = 30.0
DT = 1 / 60


def pcg(v):
    s = (v * 747796405 + 2891336453) & M
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M
    return (w >> 22) ^ w


def hf(h): return float(np.float32(h & 0xffffff) / np.float32(16777215.0))


def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def nights(seconds):
    """L.night at each 1/60 s from the saver's start."""
    phase, out = -0.035, np.zeros(int(seconds / DT) + 1)
    for i in range(len(out)):
        out[i] = smooth(-0.04, -0.20, elevation(phase))
        phase = advance(phase, DT)
        if phase > 1:
            phase -= 1
    return out


def meteors(seed, t0, t1, night):
    """(start, dur, h, brightness) for each meteor drawn starting in [t0, t1]; brightness is the night ramp mid-flight."""
    out = []
    for k in range(int(t0 // SLOT), int(t1 // SLOT) + 1):
        h = pcg(((k * 2654435761) & M) ^ ((seed + 4242) & M))
        dur = 0.4 + 0.4 * hf(pcg(h + 1))
        start = k * SLOT + 5.0 + (20.0 - dur) * hf(pcg(h + 2))
        n = night[min(int((start + dur / 2) / DT), len(night) - 1)]
        if t0 <= start <= t1 and n >= 0.6:
            out.append((start, dur, h, smooth(0.6, 1.0, n)))
    return out


seed = int(sys.argv[1])
if sys.argv[2] == "--nights":
    count = int(sys.argv[3])
    night = nights(720 * (count + 1))
    edges = np.flatnonzero(np.diff((night >= 0.6).astype(int)))
    per, gaps = [], []
    for a, b in list(zip(edges[::2] * DT, edges[1::2] * DT))[:count]:
        bright = [m for m in meteors(seed, a, b, night) if m[3] >= 0.5]
        per.append(len(bright))
        gaps += [q[0] - p[0] for p, q in zip(bright, bright[1:])]
        print(f"night {len(per)}: dark {a:7.1f}..{b:7.1f} s, {len(bright)} at half brightness or more: "
              + " ".join(f"{m[0]:.1f}" for m in bright))
    print(f"per night: mean {np.mean(per):.2f}, min {min(per)}, max {max(per)}; "
          f"gap: min {min(gaps):.1f} s, mean {np.mean(gaps):.1f} s, max {max(gaps):.1f} s")
else:
    t0, t1 = float(sys.argv[2]), float(sys.argv[3])
    for start, dur, h, b in meteors(seed, t0, t1, nights(t1 + 1)):
        sx = (-0.65 + 1.3 * hf(pcg(h + 3))) * 432
        sy = 0.5 + 0.42 * hf(pcg(h + 4))
        print(f"t={start:8.2f}..{start + dur:8.2f}  dur={dur:.2f}  start x={sx:+5.0f} cells, "
              f"y={sy:.2f} of the half-height  brightness {b:.2f}")
