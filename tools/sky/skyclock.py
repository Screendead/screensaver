"""The live sky clock (SkyClock.advance in the Swift sources); the constants must match it."""
import math


def elevation(phase):
    """The sun's height for a day phase (0 = sunrise)."""
    w = math.sin(2 * math.pi * phase)
    return w * (0.42 if w > 0 else 0.56) + 0.07


def advance(phase, dt):
    """The phase dt seconds later: the clock runs slow round the horizon and fast through deep night."""
    e = elevation(phase)
    s = min(max((abs(e) - 0.22) / 0.35, 0), 1)
    warp = 5.0 if e < -0.25 else 0.5 + 1.5 * s * s * (3 - 2 * s)
    return phase + dt / 720 * warp * 1.05
