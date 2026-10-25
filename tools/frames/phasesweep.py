#!/usr/bin/env python3
"""Sweep --phase across a held pose and report each step's largest 16x16-block mean change (0-255), the measure the
light handovers were fixed against (PR #19: sunrise; the moon handover after it).

  tools/frames/phasesweep.py render APP OUTDIR START STOP STEP -- [render args, e.g. --seed 202 --times 10]
      Renders OUTDIR/p<phase>.png through tools/vrr at 864x558 (VR_* from the environment) and prints the steps.
  tools/frames/phasesweep.py steps OUTDIR
      Prints the steps of an existing sweep.
  tools/frames/phasesweep.py compare BEFORE AFTER OUT_PREFIX [--crop x0,y0,x1,y1]
      Writes OUT_PREFIX_chart.png (per-step change, both sweeps) and OUT_PREFIX_strip.png (the frames either side of
      each sweep's largest step, before over after, cropped if asked).
"""
import glob, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
VRR = os.path.join(HERE, '..', 'vrr')


def frames(d):
    fs = sorted(glob.glob(os.path.join(d, 'p*.png')), key=lambda f: float(os.path.basename(f)[1:-4]))
    return [float(os.path.basename(f)[1:-4]) for f in fs], fs


def block_max(a, b, n=16):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    h, w = (d.shape[0] // n) * n, (d.shape[1] // n) * n
    return float(d[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3)).max())


def steps(d):
    ph, fs = frames(d)
    ims = [np.asarray(Image.open(f).convert('RGB')) for f in fs]
    return ph, [block_max(ims[i], ims[i + 1]) for i in range(len(ims) - 1)]


def report(d):
    ph, st = steps(d)
    for i, s in enumerate(st):
        print(f'{ph[i]:.5f} -> {ph[i + 1]:.5f}  {s:6.2f}')
    i = int(np.argmax(st))
    print(f'max {st[i]:.2f} at {ph[i]:.5f} -> {ph[i + 1]:.5f}; median {np.median(st):.2f}')


def render(app, out, start, stop, step, extra):
    os.makedirs(out, exist_ok=True)
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        p = start + i * step
        f = os.path.join(out, f'p{p:.5f}.png')
        if not os.path.exists(f):
            subprocess.run([VRR, app, '--screenshot', f, '--size', '864x558', '--scale', '1', '--phase', f'{p:.5f}', *extra],
                           check=True, stdout=subprocess.DEVNULL)
    report(out)


def compare(before, after, prefix, crop):
    (pb, sb), (pa, sa) = steps(before), steps(after)
    W, H, L, B = 900, 360, 50, 40
    img = Image.new('RGB', (W, H), 'white')
    g = ImageDraw.Draw(img)
    top = max(max(sb), max(sa), 4.0) * 1.1
    x0, x1 = min(pb[0], pa[0]), max(pb[-1], pa[-1])
    X = lambda p: L + (p - x0) / (x1 - x0) * (W - L - 20)
    Y = lambda v: H - B - v / top * (H - B - 30)
    for v in range(0, int(top) + 1, max(1, int(top) // 8)):
        g.line([(L, Y(v)), (W - 20, Y(v))], fill=(225, 225, 225))
        g.text((8, Y(v) - 6), f'{v}', fill='black')
    g.line([(L, Y(3)), (W - 20, Y(3))], fill=(200, 120, 120))
    g.text((W - 150, Y(3) - 14), '3/255', fill=(200, 120, 120))
    for ph, st, col, name, dy in ((pb, sb, (200, 60, 40), 'before', 0), (pa, sa, (40, 90, 200), 'after', 14)):
        pts = [(X((ph[i] + ph[i + 1]) / 2), Y(s)) for i, s in enumerate(st)]
        g.line(pts, fill=col, width=2)
        for p in pts:
            g.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col)
        g.text((L + 10, 8 + dy), f'{name}: max {max(st):.1f}/255', fill=col)
    g.text((L, H - B + 8), f'phase {x0:.4f}', fill='black')
    g.text((W - 120, H - B + 8), f'{x1:.4f}', fill='black')
    g.text((W // 2 - 160, H - 16), 'largest 16x16-block mean change per phase step', fill='black')
    img.save(prefix + '_chart.png')

    rows = []
    for d, (ph, st) in ((before, (pb, sb)), (after, (pa, sa))):
        _, fs = frames(d)
        i = int(np.argmax(sb))  # the same step in both sweeps: before's spike
        i = min(i, len(fs) - 2)
        pair = [Image.open(fs[j]).convert('RGB') for j in (max(i - 1, 0), i, i + 1, min(i + 2, len(fs) - 1))]
        if crop:
            pair = [p.crop(crop) for p in pair]
        rows.append(pair)
    w, h = rows[0][0].size
    strip = Image.new('RGB', (w * 4, h * 2), 'black')
    for r, pair in enumerate(rows):
        for c, p in enumerate(pair):
            strip.paste(p, (c * w, r * h))
    strip.save(prefix + '_strip.png')
    print(f'before max {max(sb):.2f}, after max {max(sa):.2f}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'render':
        cut = a.index('--') if '--' in a else len(a)
        render(a[1], a[2], float(a[3]), float(a[4]), float(a[5]), a[cut + 1:])
    elif a[0] == 'steps':
        report(a[1])
    elif a[0] == 'compare':
        crop = tuple(int(v) for v in a[a.index('--crop') + 1].split(',')) if '--crop' in a else None
        compare(a[1], a[2], a[3], crop)
    else:
        sys.exit(__doc__)
