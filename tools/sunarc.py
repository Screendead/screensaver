#!/usr/bin/env python3
"""Sun azimuth schemes across one live day (SkyClock.advance's warp): arcs, wrap jumps and the moon's speed.

Usage: tools/sunarc.py
"""
import math

def elevation(p):
    w = math.sin(2 * math.pi * p)
    return w * (0.42 if w > 0 else 0.56) + 0.07

def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)

def frac(p):
    return p - math.floor(p)

SCHEMES = {
    'master': lambda p: 0.9 + math.pi * p,
    'A': lambda p: 0.9 + 2 * math.pi * p,
    'B': lambda p: 0.9 + math.pi * frac(p) + math.pi * smooth(0.58, 0.92, frac(p)),
    'C': lambda p: 0.9 + 0.834 * (1 - math.cos(2 * math.pi * p)),
}

def wrapdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def run():
    dt, t, p = 1 / 60, 0.0, -0.035
    rows = []
    while p < 1.0 - 0.035:
        y = elevation(p)
        s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
        warp = 5.0 if y < -0.25 else 0.5 + 1.5 * s * s * (3 - 2 * s)
        rows.append((t, p, y))
        p += dt / 720 * warp * 1.05
        t += dt
    return rows

rows = run()
deg = 180 / math.pi
print('day length %.0f s' % rows[-1][0])
night = [r for r in rows if r[2] < -0.02]
print('sun below -0.02: %.0f s; deep night (key fully moon, e < -0.16): %.0f s' %
      (len(night) / 60, len([r for r in rows if r[2] < -0.16]) / 60))
for name, az in SCHEMES.items():
    jump = abs(wrapdiff(az(1 - 1e-9), az(0.0))) * deg
    rise = az(-0.0265); sset = az(0.5265)
    day_arc = (sset - rise) * deg if name != 'B' else (az(0.5265) - (az(0.9735) - 2 * math.pi)) * deg
    # The moon is the anti-sun: its azimuth moves as the sun's does.
    rates = []
    for (t0, p0, y0), (t1, p1, y1) in zip(rows, rows[1:]):
        if y0 < -0.16:
            rates.append(abs(wrapdiff(az(p1), az(p0))) * deg / (t1 - t0))
    moved = sum(r / 60 for r in rates)
    noon = wrapdiff(az(0.25), SCHEMES['master'](0.25)) * deg
    sunset = wrapdiff(az(0.5265), SCHEMES['master'](0.5265)) * deg
    print('%-6s wrap jump %5.1f deg | day arc %5.1f deg | vs master: noon %+5.1f, sunset %+5.1f deg | '
          'moon in deep night: moves %5.1f deg, mean %.2f deg/s, peak %.2f deg/s' %
          (name, jump, day_arc, noon, sunset, moved, moved / max(len(rates) / 60, 1e-9), max(rates)))
