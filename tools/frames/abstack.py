#!/usr/bin/env python3
# Stacks two dumped frame sequences (VR_FRAMES' <name>_fN.png) top over bottom with a label on each, into a
# directory of PNGs for ffmpeg, and prints each sequence's mean frame-to-frame change inside a crop.
#   tools/frames/abstack.py <top dir> "<top label>" <bottom dir> "<bottom label>" <out dir> [x0,y0,x1,y1]
import glob
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def frames(d):
    fs = glob.glob(f"{d}/*_f*.png")
    return sorted(fs, key=lambda f: int(re.search(r"_f(\d+)\.png$", f).group(1)))


def change(fs, box):
    prev, total = None, []
    for f in fs:
        a = np.asarray(Image.open(f).convert("L").crop(box), dtype=np.float32)
        if prev is not None: total.append(np.abs(a - prev).mean())
        prev = a
    return float(np.mean(total))


def label(img, text, font):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=font)
    d.rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 5], fill=(0, 0, 0))
    d.text((12, 10), text, fill=(255, 255, 255), font=font)


if __name__ == "__main__":
    top, tl, bot, bl, out = sys.argv[1:6]
    box = tuple(int(v) for v in sys.argv[6].split(",")) if len(sys.argv) > 6 else None
    a, b = frames(top), frames(bot)
    n = min(len(a), len(b))
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    if box:
        print(f"mean change per frame in {box}: top {change(a[:n], box):.2f}, bottom {change(b[:n], box):.2f} (0-255 grey)")
    for i in range(n):
        ia, ib = Image.open(a[i]).convert("RGB"), Image.open(b[i]).convert("RGB")
        label(ia, tl, font)
        label(ib, bl, font)
        canvas = Image.new("RGB", (ia.width, ia.height + ib.height))
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (0, ia.height))
        canvas.save(f"{out}/s{i:04d}.png")
