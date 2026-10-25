#!/usr/bin/env python3
# VR_PATH files for the sea-of-cloud proofs over seed 1's lagoon basin (t=200 pose).
# Usage: tools/cloudsea_paths.py strip|flight > path.txt
#   strip:  one held pose at eight sun elevations, each repeated 8 frames so the temporal history settles (keep the 8th)
#   flight: 10 s at 30 fps straight over the basin at sunrise, the sky clock advancing at its live rate
import math, sys

X0, Z0, Y0, YAW, PITCH = 7236.0, 1477.0, 116.0, 26.9, -15.9

def elevation(p):
    w = math.sin(2 * math.pi * p)
    return w * (0.42 if w > 0 else 0.56) + 0.07

def exposure(p):  # Renderer.swift's exposure target
    e = elevation(p)
    if e > 0: return 0.62 + 1.0 * max(0, 0.3 - e) / 0.3 + 0.08 * min(1, max(0, (e - 0.3) / 0.2))
    return 2.02 + 0.4 * (-e / 0.1) if e > -0.1 else 1.6 + 0.4 * max(0, 0.3 + e) / 0.2

def rate(p):  # SkyClock.advance, phase per second
    y = elevation(p)
    s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
    warp = 5.0 if y < -0.25 else 0.5 + 1.5 * s * s * (3 - 2 * s)
    return warp * 1.05 / 720

mode = sys.argv[1]
t = 200.0
if mode == 'strip':
    for ph in (0.92, 0.95, 0.98, 0.0, 0.03, 0.05, 0.07, 0.09):
        for _ in range(8):
            print(f'{t:.4f} {X0} {Z0} {Y0} {YAW} {PITCH} 0 {exposure(ph):.4f} {ph} 0')
            t += 1 / 30
else:
    # Starts past the phase wrap at 1.0, where the sun's azimuth jumps half a turn.
    ph, x, z, y = 0.0, X0, Z0, Y0
    for i in range(300):
        print(f'{t:.4f} {x:.3f} {z:.3f} {y:.3f} {YAW} {PITCH + 4 * i / 300:.3f} 0 {exposure(ph):.4f} {ph % 1:.6f} 0')
        t += 1 / 30
        x += math.sin(math.radians(YAW)) * 26 / 30
        z += math.cos(math.radians(YAW)) * 26 / 30
        y -= 40 / 300
        ph += rate(ph) / 30
