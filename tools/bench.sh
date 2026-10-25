#!/bin/zsh
# A/B GPU timing over twelve fixed scenes: VR_BENCH=12 stills at 3456x2234 scale 1 per scene and build, the build order
# rotating each round, reporting each scene's minimum gpu ms per build. Run it inside the exclusive lock:
#   tools/vrgpu --time -- tools/bench.sh ROUNDS "label=app label=app" [scene filter regex]
# A label of the form name@prof=app runs that build with VR_PROFILE=prof; PROFILE=... in the environment applies to all.
# Copy each build's app aside first (e.g. into build/bench/), since make overwrites build/VoxelReverie.app.
# BATTERY=1 runs every scene on the battery path (--battery). BENCH_SCENES replaces the scene list with its own
# newline-separated entries in the same name|seed|time|phase|cam form.
R=$1
pairs=(${=2})
F=${3:-.}
labels=() apps=()
for p in $pairs; do labels+=(${p%%=*}); apps+=(${p#*=}); done
D=${BENCH_OUT:-${0:A:h}/../build/bench}
mkdir -p $D
RAW=$D/bench_raw_$$.txt
: > $RAW
SCENES=(
  "pin202n|202|1215|-|3706,1705,34,21.7,-0.5,1"
  "pin606n|606|1215|-|13016,5093,80,-7,-0.5,1"
  "s606n|606|1215|-|-"
  "tepui|404|420|0.25|9500,4000,60,-75,3"
  "karst|808|420|0.25|-3974,2255,60,75,2"
  "escarp|303|60|0.005|-104,2255,40,-75,3"
  "shelters03|303|800|0.47|300,2740,60,-90,4"
  "jungle|101|420|0.25|3400,300,40,20,-6"
  "oldgrowth|101|420|0.25|354,-3300,100,180,-10"
  "giantNear2|101|420|0.25|3658,-60,60,0,8"
  "twrHero|101|420|0.25|4071.2,-3942.3,40,220,14,0"
  "stilt_close_dusk|1|20|0.53|SHOW:4,30,10,270,3"
)
[[ -n $BENCH_SCENES ]] && SCENES=("${(@f)BENCH_SCENES}")
for r in $(seq $R); do
  n=${#labels}
  k=$(( (r - 1) % n ))
  ord=(${(@)$(seq $n)[k+1,-1]} ${(@)$(seq $n)[1,k]})
  (( r % 2 == 0 )) && ord=(${(Oa)ord})
  for sc in $SCENES; do
    p=("${(@s:|:)sc}")
    [[ $p[1] =~ $F ]] || continue
    for i in $ord; do
      l=$labels[i]
      name=${l%%@*}; prof=""; [[ $l == *@* ]] && prof=${l#*@}
      [[ -n $PROFILE ]] && prof=$PROFILE
      B=$apps[i]/Contents/MacOS/voxelreverie
      args=(--screenshot $D/bench_out.png --size 3456x2234 --scale 1 --seed $p[2] --times $p[3])
      [[ $p[4] != - ]] && args+=(--phase $p[4])
      [[ -n $BATTERY ]] && args+=(--battery)
      envs=(VR_BENCH=12)
      [[ -n $prof ]] && envs+=(VR_PROFILE=$prof)
      if [[ $p[5] == SHOW:* ]]; then envs+=(VR_SHOW=${p[5]#SHOW:})
      elif [[ $p[5] != - ]]; then envs+=(VR_CAM=$p[5]); fi
      out=$(env $envs $B $args 2>/dev/null)
      v=$(echo $out | grep -o "gpu=[0-9.]*" | head -1); v=${v#gpu=}
      echo "$p[1] $l $v" >> $RAW
    done
  done
done
python3 -I - $RAW "${(j: :)labels}" <<'EOF'
import sys
r = {}
labels = sys.argv[2].split()
for l in open(sys.argv[1]):
    f = l.split()
    if len(f) >= 3: r.setdefault(f[0], {}).setdefault(f[1], []).append(float(f[2]))
print('%-17s ' % 'scene' + ' '.join('%10s' % l[:10] for l in labels) + '   d(last-first)')
for k, v in r.items():
    print('%-17s ' % k + ' '.join('%10.2f' % min(v[l]) for l in labels) + '   %+.2f' % (min(v[labels[-1]]) - min(v[labels[0]])))
EOF
