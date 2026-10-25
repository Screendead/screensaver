#!/usr/bin/env python3
# The shooting stars meteor() in Shaders/Saver.metal draws for a seed: start time, duration and screen start (cells
# from the centre) for every slot in a time range. Usage: tools/meteors.py <seed> <from s> <to s>
import sys
import numpy as np
M = 0xffffffff
def pcg(v):
    s = (v * 747796405 + 2891336453) & M
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M
    return (w >> 22) ^ w
def hf(h): return float(np.float32(h & 0xffffff) / np.float32(16777215.0))
seed, t0, t1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for k in range(int(t0 // 7), int(t1 // 7) + 1):
    h = pcg(((k * 2654435761) & M) ^ ((seed + 4242) & M))
    if hf(h) > 0.2: continue
    dur = 0.4 + 0.4 * hf(pcg(h + 1))
    start = k * 7 + hf(pcg(h + 2)) * (7 - dur)
    sx = (-0.65 + 1.3 * hf(pcg(h + 3))) * 432
    sy = (0.5 + 0.42 * hf(pcg(h + 4)))
    print(f"t={start:8.2f}..{start + dur:8.2f}  dur={dur:.2f}  start x={sx:+5.0f} cells, y={sy:.2f} of the half-height")
