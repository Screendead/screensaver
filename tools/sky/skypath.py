#!/usr/bin/env python3
# A VR_PATH file for a held camera with the sky clock running at its live rate (skyclock.py, as SkyClock.advance in
# Sources/StrataCore/Sky/SkyClock.swift).
# Usage: tools/sky/skypath.py PHASE FRAMES FPS "x,z,y,yawDeg,pitchDeg" EXPOSURE [--warm N] [--day D] > path.txt
# --warm adds N frames at the start phase first, so the temporal history settles before the clock moves.
# Frame i of the clock-running part is printed after the warm frames; output names are <screenshot>_pNNNN.png.
import sys

from skyclock import advance

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
warm = int(opt('--warm', 8)); day = int(opt('--day', 0))
phase, frames, fps = float(args[0]), int(args[1]), float(args[2])
x, z, y, yaw, pitch = map(float, args[3].split(','))
exposure = float(args[4])

t, dt = 10.0, 1 / fps
for i in range(warm + frames):
    print('%.6f %.3f %.3f %.3f %.4f %.4f 0 %.4f %.8f %d' % (t, x, z, y, yaw, pitch, exposure, phase, day))
    t += dt
    if i >= warm - 1: phase = advance(phase, dt)
