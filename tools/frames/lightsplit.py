#!/usr/bin/env python3
"""Mean scene-linear light per surface class, split into the shading's terms.

Reads the renders tools/frames/lightsplit.sh writes for one pose: <stem>.png (as seen), <stem>_v1.png (class view) and
<stem>_v2.png ... _v8.png (VR_PROFILE=bits(16384 * view), see Shaders/Shade/Shade.metal), plus <stem>.txt (the render's
log line, for exposure and sunY). Prints one row per class: pixel count, as-seen luminance, the terms, and each class's
ratio to the ground's as-seen luminance. 'other' is as seen less the three terms: glow, fog and island mist (the term views have neither) and the small extras.

usage: lightsplit.py <stem> [<stem> ...]
"""
import re
import sys

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722])
CLASSES = {  # class view colour -> name
    (0, 1, 0): "ground", (1, 0, 0): "cliff key", (0, 0, 1): "cliff away",
    (0, 1, 1): "isle top", (1, 1, 0): "isle key", (1, 0, 1): "isle away",
}


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def scene_linear(path, scale):
    """Undo the post pass (sRGB encode, ACES fit, exposure); --nodither renders only."""
    e = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    y = np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)
    y = np.minimum(y, 0.9999)
    a, b, c = 2.43 * y - 2.51, 0.59 * y - 0.03, 0.14 * y
    disc = np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))
    x = np.where(np.abs(a) > 1e-9, (-b - disc) / (2 * a), 0.0)
    return np.maximum(x, 0.0) / scale


def main():
    for stem in sys.argv[1:]:
        log = open(stem + ".txt").read()
        exp = float(re.search(r"exposure=([\d.]+)", log).group(1))
        sun = float(re.search(r"sunY=(-?[\d.]+)", log).group(1))
        gold = smoothstep(0.30, 0.12, sun) * smoothstep(-0.16, -0.04, sun)
        scale = exp * (1.0 - 0.25 * gold * smoothstep(-0.04, 0.04, sun))
        cls = np.asarray(Image.open(stem + "_v1.png").convert("RGB"), dtype=np.float64) / 255.0
        seen = scene_linear(stem + ".png", scale) @ LUMA
        v = {i: scene_linear(f"{stem}_v{i}.png", scale) for i in range(2, 9)}
        print(f"{stem.split('/')[-1]}  exposure {exp}  sunY {sun}")
        print(f"  {'class':11s} {'px':>6s} {'seen':>8s} {'/ground':>8s} {'key':>8s} {'fill':>8s} {'bounce':>8s} "
              f"{'other':>8s} {'sv':>5s} {'roof':>5s} {'shadow':>6s} {'ao':>5s} {'isOcc':>5s} {'ndl':>5s} {'n.y':>5s} {'alb':>5s}")
        ground = None
        rows = []
        for rgb, name in CLASSES.items():
            on = np.all((cls > 0.25) == np.array(rgb, bool), axis=2) & np.all(
                (cls < 0.02) | (cls > 0.25), axis=2)
            n = int(on.sum())
            if n < 30:
                continue
            m = lambda img: float(img[on].mean())
            key, fill, bounce = m(v[4] @ LUMA), m(v[5] @ LUMA), m(v[6] @ LUMA)
            s = m(seen)
            if name == "ground":
                ground = s
            rows.append((name, n, s, key, fill, bounce, s - key - fill - bounce,
                         m(v[2][..., 0]), m(v[2][..., 1]), m(v[3][..., 0]), m(v[3][..., 1]), m(v[3][..., 2]),
                         m(v[7][..., 0]), m(v[7][..., 1]) * 2 - 1, m(v[8] @ LUMA)))
        for r in rows:
            ratio = f"1:{ground / r[2]:.1f}" if ground and r[2] > 0 else "-"
            print(f"  {r[0]:11s} {r[1]:6d} {r[2]:8.4f} {ratio:>8s} {r[3]:8.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.4f} "
                  + " ".join(f"{q:5.2f}" for q in r[7:]))


if __name__ == "__main__":
    main()
