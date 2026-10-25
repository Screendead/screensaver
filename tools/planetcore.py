#!/usr/bin/env python3
# The evening star's core per frame of a PNG sequence rendered on the cell grid (one pixel per cell): how many cells are
# white in every channel, their bounding box and centroid, inside a window round the brightest such cell of frame 0.
# Usage: tools/planetcore.py <dir or glob> [--win N] [--skip n]
# White in every channel is the core: the dusk sky dithers red or green to 255 but never all three.
import glob, os, sys
import numpy as np
from PIL import Image

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
win = int(opt('--win', 12)); skip = int(opt('--skip', 0))
src = args[0]
files = sorted(glob.glob(os.path.join(src, '*.png')) if os.path.isdir(src) else glob.glob(src))[skip:]
centre = None
counts = []
for f in files:
    white = np.asarray(Image.open(f).convert('RGB')).min(2) == 255
    if centre is None:
        ys, xs = np.nonzero(white)
        centre = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (white.shape[0] // 2, white.shape[1] // 2)
    y0, x0 = max(centre[0] - win, 0), max(centre[1] - win, 0)
    w = white[y0:centre[0] + win + 1, x0:centre[1] + win + 1]
    ys, xs = np.nonzero(w)
    n = len(ys)
    counts.append(n)
    box = f'{xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1}' if n else '-'
    cen = f'({x0 + xs.mean():.1f}, {y0 + ys.mean():.1f})' if n else '-'
    print(f'{os.path.basename(f)} white={n} box={box} centroid={cen}')
c = np.array(counts)
print(f'{len(c)} frames: white cells min {c.min()} max {c.max()}; series ' + ' '.join(map(str, c)))
