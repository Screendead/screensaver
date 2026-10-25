#!/usr/bin/env python3
# VR_PATH files for the sea-of-cloud proofs over seed 1's lagoon basin (t=200 pose).
# Usage: tools/cloudsea_paths.py strip|basin|flight [fps] > path.txt
#   strip:  one held pose at eight sun elevations, each repeated 8 frames so the temporal history settles (keep the 8th)
#   basin:  the basin from 60 higher and looking further down, at sunrise, 8 frames (keep the 8th)
#   flight [fps]: 10 s straight over the basin at sunrise, down to the autopilot's floor, the sky clock at its live rate
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
elif mode == 'basin':
    for _ in range(8):
        print(f'{t:.4f} {X0} {Z0} {Y0 + 60} {YAW} {PITCH - 10} 0 {exposure(0.0):.4f} 0.0 0')
        t += 1 / 30
else:
    # Starts past the phase wrap at 1.0, where the sun's azimuth jumps half a turn. Descends to 42 over the water, the
    # autopilot's floor at dawn (22 over the sea).
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    n = int(10 * fps)
    ph, x, z = 0.0, X0, Z0
    for i in range(n):
        u = i / (n - 1)
        s = u * u * (3 - 2 * u)
        y = Y0 - (Y0 - 42.0) * s
        print(f'{t:.4f} {x:.3f} {z:.3f} {y:.3f} {YAW} {PITCH + 12 * s:.3f} 0 {exposure(ph):.4f} {ph % 1:.6f} 0')
        t += 1 / fps
        x += math.sin(math.radians(YAW)) * 26 / fps
        z += math.cos(math.radians(YAW)) * 26 / fps
        ph += rate(ph) / fps
