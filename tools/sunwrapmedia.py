#!/usr/bin/env python3
"""Labelled sheets and a clip from the sun-wrap renders.

Usage: tools/sunwrapmedia.py RENDER_DIR OUT_DIR
RENDER_DIR holds wrap_<S>_p0004..6.png (VR_PATH across the wrap) and st_<S>_<phase>.png for S in M A B C.
"""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
NAMES = {'M': 'master', 'A': 'A  full turn a day', 'B': 'B  catch up at night', 'C': 'C  swing out and back'}
font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 26)
small = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 20)

def label(im, text, f=font):
    d = ImageDraw.Draw(im)
    x, y = 14, 10
    b = d.textbbox((x, y), text, font=f)
    d.rectangle((b[0] - 8, b[1] - 6, b[2] + 8, b[3] + 6), fill=(0, 0, 0))
    d.text((x, y), text, font=f, fill=(255, 255, 255))
    return im

def tile(path, w, text):
    im = Image.open(path).convert('RGB')
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    return label(im, text, small)

def sheet(rows, w, title, name):
    th = 50
    h = rows[0][0].height
    o = Image.new('RGB', (w * len(rows[0]), th + h * len(rows)), (0, 0, 0))
    label(o, title)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            o.paste(im, (c * w, th + r * h))
    o.save(os.path.join(out, name))

W = 864
phases = [('p0004', 'phase 0.9997 (day 0)'), ('p0005', 'phase 0.0000 (day 1)'), ('p0006', 'phase 0.0004 (day 1)')]
rows = [[tile(os.path.join(src, 'wrap_%s_%s.png' % (s, p)), W, '%s | %s' % (NAMES[s], t)) for p, t in phases] for s in 'MABC']
sheet(rows, W, 'Across the day wrap, same camera: master jumps, A B C continuous', 'wrap_strip_master_A_B_C.png')

stills = [('0.03', 'dawn 0.03'), ('0.25', 'noon 0.25'), ('0.50', 'sunset 0.50'), ('0.75', 'midnight 0.75')]
rows = [[tile(os.path.join(src, 'st_%s_%s.png' % (s, p)), 576, '%s | %s' % (NAMES[s].split()[0], t)) for p, t in stills] for s in 'MABC']
sheet(rows, 576, 'Seed 1, 10 s flight, phase pinned: rows master A B C', 'stills_dawn_noon_sunset_midnight_master_A_B_C.png')

# The clip: the three wrap frames in a 2x2 grid, each held 1 s, looped three times.
tmp = os.path.join(out, '_clip')
os.makedirs(tmp, exist_ok=True)
n = 0
for loop in range(3):
    for p, t in phases:
        o = Image.new('RGB', (W * 2, 2 * round(1117 * W / 1728) + 50))
        label(o, 'Across the day wrap: ' + t)
        for i, s in enumerate('MABC'):
            o.paste(tile(os.path.join(src, 'wrap_%s_%s.png' % (s, p)), W, NAMES[s]), ((i % 2) * W, 50 + (i // 2) * round(1117 * W / 1728)))
        o.save(os.path.join(tmp, 'f%03d.png' % n)); n += 1
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '1', '-i', os.path.join(tmp, 'f%03d.png'), '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '16', '-r', '30', os.path.join(out, 'wrap_across_dawn_master_vs_A_B_C.mp4')], check=True)
for f in os.listdir(tmp):
    os.replace(os.path.join(tmp, f), os.path.join(src, 'clip_' + f))
os.rmdir(tmp)
