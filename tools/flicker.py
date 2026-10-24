#!/usr/bin/env python3
# Frame-to-frame flicker in a sequence of PNGs (a clip's decoded frames or VR_PATH / VR_FRAMES renders).
# Usage: tools/flicker.py <dir or glob> [--dark Y] [--step N] [--crop WxH] [--skip n] (n leading frames, e.g. bugclip's warm-up)
# Per frame pair, on BT.709 luma in 8-bit levels: "flip" is the share of pixels that change by at least N levels;
# "blink" the share that jump at least N away from both neighbours in the same direction (a one-frame pop, which
# smooth camera motion rarely makes). Dark is pixels whose mean luma over the sequence is under Y, in the lower
# two thirds of the frame; bright is the rest of the lower two thirds.
import glob, os, sys
import numpy as np
from PIL import Image

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
dark_y = float(opt('--dark', 60)); step = float(opt('--step', 12)); crop = opt('--crop', None)
skip = int(opt('--skip', 0))
src = args[0]
files = sorted(glob.glob(os.path.join(src, '*.png')) if os.path.isdir(src) else glob.glob(src))[skip:]
k = np.array([0.2126, 0.7152, 0.0722])
fr = []
for f in files:
    a = np.asarray(Image.open(f).convert('RGB'), dtype=np.float32)
    if crop:
        w, h = map(int, crop.split('x')); a = a[:h, :w]
    fr.append(a @ k)
Y = np.stack(fr)
H = Y.shape[1]
low = np.zeros(Y.shape[1:], bool); low[H // 3:] = True
mean = Y.mean(0)
masks = {'dark': low & (mean < dark_y), 'bright': low & (mean >= dark_y)}
d = np.diff(Y, axis=0)
flip = np.abs(d) >= step
up = (d[:-1] >= step) & (d[1:] <= -step)
dn = (d[:-1] <= -step) & (d[1:] >= step)
blink = up | dn
print(f'{len(files)} frames {Y.shape[2]}x{H}, step {step:g}, dark < {dark_y:g}')
for name, m in masks.items():
    if not m.any(): continue
    print(f'  {name:6s} {m.mean() * 100:5.1f}% of frame: flip {flip[:, m].mean() * 100:6.2f}%  blink {blink[:, m].mean() * 100:6.2f}%'
          f'  mean |dY| {np.abs(d[:, m]).mean():5.2f}  mean Y {mean[m].mean():5.1f}')
