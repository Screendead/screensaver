#!/usr/bin/env python3
"""Labelled 2x2 summary sheet from four 1728x1117 renders.
usage: tools/picksmedia.py <out.png> <png> <label> <png> <label> <png> <label> <png> <label>"""
import sys
from PIL import Image, ImageDraw, ImageFont

out, rest = sys.argv[1], sys.argv[2:]
F = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 26)
tiles = []
for path, text in zip(rest[::2], rest[1::2]):
    im = Image.open(path).convert('RGB').resize((864, 558), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    b = d.textbbox((14, 12), text, font=F)
    d.rectangle((b[0] - 8, b[1] - 6, b[2] + 8, b[3] + 6), fill=(16, 18, 30))
    d.text((14, 12), text, font=F, fill=(238, 238, 242))
    tiles.append(im)
sheet = Image.new('RGB', (864 * 2 + 6, 558 * 2 + 6), (8, 8, 12))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % 2) * 870, (i // 2) * 564))
sheet.save(out)
