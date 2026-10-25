#!/usr/bin/env python3
# Labelled moon-phase media from rendered frames: a 2x2 still of four nights, a clip of four nights (master left,
# branch right, each with a zoomed moon inset) and a terminator close-up clip (master, before, after; flight and live clock).
# Usage: tools/moonmedia.py <frames dir> <out dir>
# The frames dir holds (names as written by the renders that feed it):
#   big/d{0,1,2,4}_p0002.png          1728x1117 stills, one per night (VR_PATH, day field 0, 1, 2, 4)
#   live/master/l_p*.png              864x558 held-camera frames with the sky clock running (master: full every night)
#   live/fix4pct/l_p*.png, live/fix_d1/l_p*.png, live/fix_d2/l_p*.png   the branch on nights 1-3 (night 4 is master's)
#   live/orig/l_p*.png                the branch before the terminator ramp, night 1
#   flight/{master,orig,fix}_f*.png   864x558 VR_MOTION flight frames, seed 6 phase 0.75
# Blink numbers on the labels come from tools/moonflicker.py --skip 4.
import glob, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 16)
except OSError:
    font = small = ImageFont.load_default()

def label(im, xy, text, f=font):
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    d.rectangle((x0 - 6, y0 - 5, x1 + 6, y1 + 5), fill=(0, 0, 0))
    d.text(xy, text, font=f, fill=(255, 255, 255))

def frames(pattern, skip=4):
    return sorted(glob.glob(os.path.join(src, pattern)))[skip:]

def encode(imgs, path, fps):
    with tempfile.TemporaryDirectory() as t:
        for i, im in enumerate(imgs):
            im.save(os.path.join(t, f'{i:04d}.png'))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(t, '%04d.png'),
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', path], check=True)

nights = [('night 1: 13% lit', 'd0'), ('night 2: 39% lit', 'd1'), ('night 3: 68% lit', 'd2'), ('night 4: full, identical to master', 'd4')]

# Four nights, one still.
grid = Image.new('RGB', (1728, 1117))
for i, (name, d) in enumerate(nights):
    im = Image.open(os.path.join(src, 'big', f'{d}_p0002.png')).convert('RGB').resize((864, 558), Image.LANCZOS)
    label(im, (14, 12), name)
    grid.paste(im, ((i % 2) * 864, (i // 2) * 558))
grid.save(os.path.join(out, 'still_after_four-nights-13-39-68-100pct_seed6_1728.png'))

# Four nights in motion: master left, branch right, a x6 moon inset in each.
MOON = (583, 32, 631, 80)
def panel(path, text):
    im = Image.open(path).convert('RGB')
    inset = im.crop(MOON).resize((288, 288), Image.NEAREST)
    im.paste(inset, (864 - 288 - 10, 558 - 288 - 10))
    label(im, (14, 12), text)
    return im
seqs = [('live/master/l_p*.png', 'live/fix4pct/l_p*.png'), ('live/master/l_p*.png', 'live/fix_d1/l_p*.png'),
        ('live/master/l_p*.png', 'live/fix_d2/l_p*.png'), ('live/master/l_p*.png', 'live/master/l_p*.png')]
clip = []
for (name, _), (a, b) in zip(nights, seqs):
    for fa, fb in zip(frames(a), frames(b)):
        fr = Image.new('RGB', (1728, 558))
        fr.paste(panel(fa, 'master: full moon every night'), (0, 0))
        fr.paste(panel(fb, 'branch ' + name), (864, 0))
        label(fr, (14, 524), 'sky clock at its live rate, camera held; frames 1/60 s apart shown at 1/6 speed', small)
        clip.append(fr)
encode(clip, os.path.join(out, 'clip_master-left-branch-right_four-nights_live-clock_moon-inset-x6_864.mp4'), 10)

# Terminator close-up: master, before the ramp, after; flight above, live clock below.
cols = [('master', 'flight/master_f*.png', 'live/master/l_p*.png', '0.02%', '0.00%'),
        ('before: hard block edges', 'flight/orig_f*.png', 'live/orig/l_p*.png', '0.15%', '0.11%'),
        ('after: edges ramp over 4% of radius', 'flight/fix_f*.png', 'live/fix4pct/l_p*.png', '0.11%', '0.06%')]
def key(f):
    return int(''.join(c for c in f.rsplit('_', 1)[1] if c.isdigit()))
W, H = 400, 420
seq = []
fl = [sorted(glob.glob(os.path.join(src, c[1])), key=key)[4:] for c in cols]
lv = [frames(c[2]) for c in cols]
n = min(min(map(len, fl)), min(map(len, lv)))
for i in range(n):
    fr = Image.new('RGB', (3 * W, 2 * H + 40), (20, 20, 24))
    for j, c in enumerate(cols):
        a = Image.open(fl[j][i]).convert('RGB').crop((591, 0, 631, 34)).resize((320, 272), Image.NEAREST)
        b = Image.open(lv[j][i]).convert('RGB').crop(MOON).resize((384, 384), Image.NEAREST)
        fr.paste(a, (j * W + (W - 320) // 2, 40 + (H - 40 - 272) // 2 + 20))
        fr.paste(b, (j * W + (W - 384) // 2, H + 40 + 4))
        label(fr, (j * W + 10, 8), c[0], small)
        label(fr, (j * W + 10, 40 + 8), f'flight: blinks {c[3]}', small)
        label(fr, (j * W + 10, H + 40 + 12), f'live clock: blinks {c[4]}', small)
    seq.append(fr)
encode(seq, os.path.join(out, 'clip_terminator-closeup-x8_master-before-after_flight-top-live-clock-bottom.mp4'), 6)
seq[len(seq) // 2].save(os.path.join(out, 'still_terminator-closeup-x8_master-before-after_flight-top-live-clock-bottom.png'))
