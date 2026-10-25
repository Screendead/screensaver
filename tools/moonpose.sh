#!/bin/bash
# Render the 10-12 12.54 bug report's moonlit-river pose (seed 827296504) with each given app.
# usage: tools/moonpose.sh <out prefix> <size> <app>... [-- extra screenshot args]
# writes <out prefix>_<app basename without .app>.png
set -e
out=$1 size=$2; shift 2
apps=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do apps+=("$1"); shift; done
[ "$1" = "--" ] && shift
export VR_CAM=704790.7479682205,309151.55578665674,47.52305369742964,-1.546140580587581,-0.30783337250314785,1,-0.060186864301448144
export VR_EXPOSURE=${VR_EXPOSURE:-1.59948534474975}
here=$(cd "$(dirname "$0")" && pwd)
for a in "${apps[@]}"; do
    n=$(basename "$a" .app)
    "$here/vrr" "$a" --screenshot "${out}_$n.png" --size "$size" --seed 827296504 --times 26.3060558750527 --scale 1.0 --phase 0.75 "$@" | tail -1
done
