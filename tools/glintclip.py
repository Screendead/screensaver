#!/usr/bin/env python3
"""Clipping in a highlight region of renders: for each PNG, within the box x0,y0,x1,y1 given as fractions of
the frame, the share of bright pixels (luma >= 128) that are clipped (every channel >= 250), plus the mean luma.
Use --nodither renders for the clean measure; dithered frames count what the screen shows.

usage: tools/glintclip.py x0,y0,x1,y1 a.png [b.png ...]"""
import sys
from PIL import Image

box = [float(v) for v in sys.argv[1].split(",")]
for path in sys.argv[2:]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).getdata()
    n = bright = clipped = 0
    total = 0.0
    for r, g, b in px:
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        n += 1
        total += y
        if y >= 128:
            bright += 1
            if min(r, g, b) >= 250:
                clipped += 1
    print(f"{path}: bright {bright / n:6.2%} of region, clipped {clipped / max(bright, 1):6.2%} of bright"
          f" ({clipped / n:6.2%} of region), mean luma {total / n:6.1f}")
