#!/usr/bin/env python3
# A VR_PATH file for a held camera with the sky clock running at its live rate (SkyClock.advance in Sources/Renderer.swift).
# Usage: tools/sky/skypath.py PHASE FRAMES FPS "x,z,y,yawDeg,pitchDeg" EXPOSURE [--warm N] [--day D] > path.txt
# --warm adds N frames at the start phase first, so the temporal history settles before the clock moves.
# Frame i of the clock-running part is printed after the warm frames; output names are <screenshot>_pNNNN.png.
import math, sys

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
warm = int(opt('--warm', 8)); day = int(opt('--day', 0))
phase, frames, fps = float(args[0]), int(args[1]), float(args[2])
x, z, y, yaw, pitch = map(float, args[3].split(','))
exposure = float(args[4])

def advance(p, dt):
    w = math.sin(2 * math.pi * p)
    e = w * (0.42 if w > 0 else 0.56) + 0.07
    s = min(max((abs(e) - 0.22) / 0.35, 0), 1)
    warp = 5.0 if e < -0.25 else 0.5 + 1.5 * s * s * (3 - 2 * s)
    return p + dt / 720 * warp * 1.05

t, dt = 10.0, 1 / fps
for i in range(warm + frames):
    print('%.6f %.3f %.3f %.3f %.4f %.4f 0 %.4f %.8f %d' % (t, x, z, y, yaw, pitch, exposure, phase, day))
    t += dt
    if i >= warm - 1: phase = advance(phase, dt)
