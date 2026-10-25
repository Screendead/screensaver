#!/usr/bin/env python3
# Steps SkyClock (Sources/Renderer.swift) through one day as advance() does, for the current clock and the slow-dawn
# one, and prints day length, golden-hour seconds and the sun's climb at 3 degrees; with a path, also draws sun
# elevation against time for both as a PNG. Keep warp_current and warp_slow in step with SkyClock.advance.
#   tools/dayclock.py [chart.png]
import math
import sys


def elevation(p):
    w = math.sin(2 * math.pi * p)
    return w * (0.42 if w > 0 else 0.56) + 0.07


def ss(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def warp_current(y):
    s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
    return 5.0 if y < -0.25 else 0.5 + 1.5 * ss(s)


# Slow dawn: half speed within |y| < 0.08 (about 4.6 degrees), easing back by 0.15; everything else runs `hurry` times
# as fast. The saver ships hurry 1, so the day grows to about 15 minutes; solve_hurry finds the hurry that keeps 12.
LOW, BAND = 0.08, 0.15


def warp_slow(y, hurry=1.0):
    return warp_current(y) * (0.5 + (hurry - 0.5) * ss((abs(y) - LOW) / (BAND - LOW)))


def day(warp, dt=0.01, length=720.0):
    p, t, out = -0.035, 0.0, []
    while p < 0.965:
        out.append((t, elevation(p)))
        p += dt / length * warp(elevation(p)) * 1.05
        t += dt
    return t, out


def seconds_where(samples, lo, hi, dt=0.01):
    return sum(dt for _, y in samples if lo <= y <= hi)


def solve_hurry(total):
    lo, hi = 1.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if day(lambda y: warp_slow(y, mid))[0] > total: lo = mid
        else: hi = mid
    return (lo + hi) / 2


GOLD = (math.sin(math.radians(-4)), math.sin(math.radians(6)))


def climb_at(warp, deg=3.0):
    y = math.sin(math.radians(deg))
    p = math.asin((y - 0.07) / 0.42) / (2 * math.pi)
    dydp = 2 * math.pi * math.cos(2 * math.pi * p) * 0.42
    return math.degrees(dydp * warp(y) * 1.05 / 720 / math.cos(math.radians(deg)))


def chart(path, series, total):
    from PIL import Image, ImageDraw, ImageFont
    S = 2
    W, H = 864 * S, 558 * S
    L, R, T, B = 78 * S, 30 * S, 92 * S, 62 * S
    surf, ink, ink2, grid = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 222)
    img = Image.new("RGB", (W, H), surf)
    d = ImageDraw.Draw(img)

    def font(sz):
        for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
            try: return ImageFont.truetype(f, sz * S)
            except OSError: pass
        return ImageFont.load_default()
    f13, f15, f20 = font(13), font(15), font(20)
    ymin, ymax = -32.0, 32.0
    X = lambda t: L + (W - L - R) * t / total
    Y = lambda e: T + (H - T - B) * (ymax - e) / (ymax - ymin)
    d.rectangle([L, Y(6), W - R, Y(-4)], fill=(253, 240, 214))
    d.text((W - R - 6 * S, Y(6) - 2 * S), "golden hour, -4 to 6 deg", fill=ink2, font=f13, anchor="rb")
    for e in range(-30, 31, 10):
        d.line([L, Y(e), W - R, Y(e)], fill=(150, 149, 144) if e == 0 else grid, width=S)
        d.text((L - 8 * S, Y(e)), f"{e}°", fill=ink2, font=f13, anchor="rm")
    for m in range(0, int(total // 60) + 1, 2):
        d.text((X(m * 60), H - B + 10 * S), f"{m} min", fill=ink2, font=f13, anchor="mt")
    for name, colour, pts, label in series:
        xy = [(X(t), Y(math.degrees(math.asin(y)))) for t, y in pts[::20]]
        d.line(xy, fill=colour, width=2 * S, joint="curve")
    d.text((L, 22 * S), "Sun elevation over one day", fill=ink, font=f20)
    lx = L
    for name, colour, pts, label in series:
        d.rounded_rectangle([lx, 60 * S, lx + 18 * S, 64 * S], radius=2 * S, fill=colour)
        d.text((lx + 24 * S, 62 * S), label, fill=ink, font=f15, anchor="lm")
        lx += 24 * S + d.textlength(label, font=f15) + 28 * S
    img.save(path)


if __name__ == "__main__":
    total, cur = day(warp_current)
    hurry = 1.0
    tot2, slow = day(lambda y: warp_slow(y, hurry))
    print(f"current day {total:.1f} s; slow dawn (hurry 1) {tot2:.1f} s; hurry {solve_hurry(total):.3f} would keep the day")
    rows = []
    for name, s, w in (("current", cur, warp_current), ("slow dawn", slow, lambda y: warp_slow(y, hurry))):
        gold = seconds_where(s, *GOLD)
        up = seconds_where(s, 0, 1)
        night = seconds_where(s, -1, math.sin(math.radians(-8)))
        print(f"{name}: golden hour {gold:.0f} s a day ({gold / 2:.0f} s each sunrise and sunset), sun up {up:.0f} s,"
              f" night below -8 deg {night:.0f} s, sun climbs {climb_at(w):.3f} deg/s at 3 deg")
        rows.append(gold / 2)
    if len(sys.argv) > 1:
        chart(sys.argv[1], [("current", (42, 120, 214), cur, f"Current: {rows[0]:.0f} s golden hour each sunrise and sunset"),
                            ("slow", (235, 104, 52), slow, f"Slow dawn: {rows[1]:.0f} s")], max(total, tot2))
