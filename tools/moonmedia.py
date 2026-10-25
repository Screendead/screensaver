#!/usr/bin/env python3
"""Pitch media for the moon-glint variants, from tools/moonpose.sh renders.
usage: tools/moonmedia.py <build dir> <out dir>
expects <build>/shots/full_{A,B,C}.png (1728x1117), e24_{A,B,C}.png (864x558, VR_EXPOSURE=2.4) and
<build>/motion/seq_{A,B,C}/NNNN.png (864x558 VR_MOTION frames)."""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

S, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
F = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 22)
L = {'A': 'A  current', 'B': 'B  soft roll-off', 'C': 'C  soft + finer glints'}


def label(im, text, xy=(10, 10)):
    d = ImageDraw.Draw(im)
    b = d.textbbox(xy, text, font=F)
    d.rectangle((b[0] - 8, b[1] - 6, b[2] + 8, b[3] + 6), fill=(16, 18, 30))
    d.text(xy, text, font=F, fill=(235, 235, 240))
    return im


def row(ims, gap=4):
    o = Image.new('RGB', (sum(i.width for i in ims) + gap * (len(ims) - 1), ims[0].height))
    x = 0
    for i in ims:
        o.paste(i, (x, 0))
        x += i.width + gap
    return o


def shot(name):
    return Image.open(f'{S}/shots/{name}.png').convert('RGB')


row([label(shot(f'full_{v}').crop((760, 640, 1320, 1117)), L[v]) for v in 'ABC']).save(
    f'{O}/moon_glints_triptych_A_current_B_soft_C_finer_1728.png')
row([label(shot(f'e24_{v}').crop((380, 320, 660, 558)).resize((560, 476), Image.NEAREST), L[v] + ' (exp 2.4)')
     for v in 'ABC']).save(f'{O}/moon_glints_triptych_brighter_night_exposure2.4.png')
for v, n in zip('ABC', ['A_current', 'B_soft_rolloff', 'C_soft_plus_finer']):
    label(shot(f'full_{v}'), L[v]).save(f'{O}/moon_glints_fullframe_{n}_1728.png')

frames = sorted(os.listdir(f'{S}/motion/seq_A'))[12:]
for d in ('vid1', 'vid2'):
    os.makedirs(f'{S}/motion/{d}', exist_ok=True)
for j, f in enumerate(frames):
    seq = {v: Image.open(f'{S}/motion/seq_{v}/{f}').convert('RGB') for v in 'ABC'}
    o = Image.new('RGB', (864, 1116))
    o.paste(label(seq['A'].copy(), L['A']), (0, 0))
    o.paste(label(seq['B'].copy(), L['B']), (0, 558))
    o.save(f'{S}/motion/vid1/{j:04d}.png')
    row([label(seq[v].crop((380, 300, 660, 558)).resize((560, 516), Image.NEAREST), L[v]) for v in 'ABC']).save(
        f'{S}/motion/vid2/{j:04d}.png')
for d, n in [('vid1', 'moon_glints_motion_A_current_over_B_soft_864.mp4'),
             ('vid2', 'moon_glints_motion_river_closeup_A_B_C_2x.mp4')]:
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '60', '-i', f'{S}/motion/{d}/%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', f'{O}/{n}'], check=True)
