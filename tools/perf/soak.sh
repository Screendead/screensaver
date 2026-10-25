#!/bin/zsh
# Soak test off screen: flies the autopilot frame by frame on mains and on battery, then approaches two caves, logging
# each frame's GPU time. Nothing opens on screen. Runs under the exclusive GPU lock, so the times are comparable.
# Usage: tools/perf/soak.sh [out dir, default build/soak] [app, default build/VoxelReverie.app] [seed, default 1]
set -euo pipefail
tools=${0:A:h:h}
out=${1:-$tools/../build/soak} app=${2:-$tools/../build/VoxelReverie.app} seed=${3:-1}
bin=$app/Contents/MacOS/voxelreverie
mkdir -p "$out"
cat > "$out/run.sh" <<EOF
#!/bin/zsh
set -e
cd "$out"
export VR_TIMING=1
# 30 s of flight at each of four points over half an hour, at the live size; VR_CHUNKED updates the rings as live.
VR_MOTION=1 VR_MOTION_FRAMES=1800 VR_CHUNKED=1 VR_CAVES=2000 "$bin" --screenshot mains.png --size 3456x2234 --scale 1 \
    --seed $seed --times 60,600,1200,1800 > mains.log
VR_MOTION=1 VR_MOTION_FRAMES=1800 VR_CHUNKED=1 VR_CAVES=2000 "$bin" --screenshot battery.png --size 3456x2234 --battery \
    --seed $seed --times 60,600,1200,1800 > battery.log
# From 300 voxels out, 1 voxel a frame toward the nearest big cave seen clear, then the nearest lived-in one.
VR_CAVE=300,10,0,0,2 VR_DOLLY=1 VR_BENCH=280 VR_CHUNKED=1 "$bin" --screenshot cave_clear.png --size 3456x2234 --scale 1 \
    --seed $seed --times 600 > cave_clear.log
VR_CAVE=300,10,0,0,1 VR_DOLLY=1 VR_BENCH=280 VR_CHUNKED=1 "$bin" --screenshot cave_home.png --size 3456x2234 --scale 1 \
    --seed $seed --times 600 > cave_home.log
EOF
chmod +x "$out/run.sh"
"$tools/vrgpu" --time -- "$out/run.sh"
python3 -I - "$out" <<'EOF'
import re, statistics, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + '/*.log')):
    ms = [float(m.group(1)) for m in re.finditer(r'^frame \d+ gpu ([\d.]+) ms', open(f).read(), re.M)]
    if not ms:
        print(os.path.basename(f), 'no frames'); continue
    s = sorted(ms)
    print('%-15s frames %5d  median %6.2f  p95 %6.2f  p99 %6.2f  worst %6.2f ms  over 16.7 %4d  over 33.3 %4d' % (os.path.basename(f), len(s),
          statistics.median(s), s[int(.95 * len(s))], s[int(.99 * len(s))], s[-1], sum(x > 16.7 for x in s), sum(x > 33.3 for x in s)))
EOF
