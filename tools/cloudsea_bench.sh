#!/bin/zsh
# Sea-of-cloud A/B at sunrise (phase 0.0) on the mains and battery paths: VR_BENCH=12 stills at 3456x2234 scale 1,
# min gpu ms per scene and build, build order alternating each round. Run inside the exclusive lock:
#   tools/vrgpu --time -- tools/cloudsea_bench.sh ROUNDS A.app B.app
R=$1 A=$2 B=$3
D=${0:A:h}/../build/bench
mkdir -p $D
SCENES=("basin1|1|200" "plain303|303|30")
for r in $(seq $R); do
  ord=($A $B); (( r % 2 == 0 )) && ord=($B $A)
  for sc in $SCENES; do
    p=("${(@s:|:)sc}")
    for power in mains battery; do
      for app in $ord; do
        args=(--screenshot $D/cloudsea_out.png --size 3456x2234 --scale 1 --seed $p[2] --times $p[3] --phase 0.0)
        [[ $power == battery ]] && args+=(--battery)
        v=$(VR_BENCH=12 $app/Contents/MacOS/voxelreverie $args 2>/dev/null | grep -o "gpu=[0-9.]*" | head -1)
        echo "$p[1] $power ${app:t} ${v#gpu=}"
      done
    done
  done
done
