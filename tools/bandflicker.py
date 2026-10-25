#!/usr/bin/env python3
# Flicker inside the region a change touches: two VR_PATH frame dirs (A before, B after) of the same held path; the
# mask is pixels whose sequence mean differs by more than 4 levels. Usage: tools/bandflicker.py <dirA> <dirB> [--skip n]
import glob, sys, numpy as np
from PIL import Image
skip = int(sys.argv[sys.argv.index('--skip') + 1]) if '--skip' in sys.argv else 30
k = np.array([0.2126, 0.7152, 0.0722])
def load(d): return np.stack([np.asarray(Image.open(f).convert('RGB'), np.float32) @ k for f in sorted(glob.glob(d + '/f_p*.png'))[skip:]])
A, B = load(sys.argv[1]), load(sys.argv[2])
mask = np.abs(B.mean(0) - A.mean(0)) > 4
print(f'shore band mask: {mask.mean()*100:.2f}% of frame ({mask.sum()} px), {len(A)} frames')
for name, Y in (('master', A), ('wet', B)):
    d = np.diff(Y, axis=0)
    for step in (6, 12):
        flip = (np.abs(d) >= step)[:, mask].mean() * 100
        blink = (((d[:-1] >= step) & (d[1:] <= -step)) | ((d[:-1] <= -step) & (d[1:] >= step)))[:, mask].mean() * 100
        print(f'{name:6s} step {step:2d}: flip {flip:.3f}%  blink {blink:.4f}%  mean|dY| {np.abs(d)[:, mask].mean():.3f}')
