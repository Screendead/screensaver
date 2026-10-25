#!/usr/bin/env python3
# Flicker inside a fixed crop around the moon, for a sequence of PNGs (VR_FRAMES or VR_PATH renders).
# Usage: tools/moonflicker.py <glob> [--box WxH] [--step N] [--big N] [--at X,Y] [--skip n]
# The crop is centred on --at, or else on the moon: the brightest blob of the first frame's top half. Compare builds
# with the same --at, since a thin moon may not be the brightest thing in the sky. "blink" is the share of pixel-frames
# that jump at least --step levels away from both neighbours in the same direction; "big" the share of frame pairs that
# change by at least --big levels.
import glob, sys
import numpy as np
from PIL import Image

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
bw, bh = map(int, opt('--box', '40x34').split('x'))
step = float(opt('--step', 12)); big = float(opt('--big', 30)); skip = int(opt('--skip', 0))
at = opt('--at', None)
files = sorted(glob.glob(args[0]), key=lambda f: int(''.join(c for c in f.rsplit('_', 1)[1] if c.isdigit())))[skip:]
k = np.array([0.2126, 0.7152, 0.0722])
Y = np.stack([np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) @ k for f in files])
top = Y[0][: Y.shape[1] // 2]
if at:
    cx, cy = map(int, at.split(','))
else:
    ys, xs = np.where(top > 0.8 * top.max())
    cy, cx = int(round(np.median(ys))), int(round(np.median(xs)))
y0 = max(0, cy - bh // 2); x0 = max(0, min(Y.shape[2] - bw, cx - bw // 2))
C = Y[:, y0:y0 + bh, x0:x0 + bw]
d = np.diff(C, axis=0)
blink = ((d[:-1] >= step) & (d[1:] <= -step)) | ((d[:-1] <= -step) & (d[1:] >= step))
print(f'{len(files)} frames, crop {bw}x{bh} at ({x0},{y0}): blink>={step:g} {blink.mean() * 100:.2f}%  change>={big:g} {(np.abs(d) >= big).mean() * 100:.2f}%  mean |dY| {np.abs(d).mean():.2f}')
