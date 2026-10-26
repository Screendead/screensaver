#!/bin/zsh
# Renders one pose for tools/frames/lightsplit.py: as seen, then each VR_PROFILE term view (fog off), all undithered.
# usage: lightsplit.sh <app> <out stem> <seed> <phase> <VR_CAM> [size]
set -e
app=$1 stem=$2 seed=$3 phase=$4 cam=$5 size=${6:-864x558}
here=${0:A:h}
mkdir -p ${stem:h}
common=(--size $size --scale 1 --seed $seed --times 10 --phase $phase --nodither)
VR_CAM=$cam $here/../vrr $app --screenshot $stem.png $common | grep exposure= > $stem.txt
for v in 1 2 3 4 5 6 7 8; do
  VR_CAM=$cam VR_PROFILE=bits$((16384 * v)) $here/../vrr $app --screenshot ${stem}_v$v.png $common --debug 2 > /dev/null
done
