#!/bin/zsh
# Output regression check: renders a fixed set of stills, a short motion run and a recorded-path replay through
# tools/vrr at 864x558, and compares the sha256 of every PNG and of each run's normalised stdout against
# tests/golden/manifest.sha256. Exits nonzero and lists what differs on any mismatch.
#   tools/golden.sh [app, default build/VoxelReverie.app]            check
#   tools/golden.sh --update [app]                                   rewrite the manifest and the stdout texts
# Renders land in build/golden (GOLDEN_OUT=dir, relative to the repo root, to run two checks at once).
# The stdout texts are committed beside the manifest (tests/golden/stdout/) so a failing one can be diffed.
here=${0:A:h}
root=${here:h}
update=0
[[ $1 == --update ]] && { update=1; shift }
app=${1:-$root/build/VoxelReverie.app}
app=${app:A}
golden=$root/tests/golden
manifest=$golden/manifest.sha256
cd $root || exit 1
out=${GOLDEN_OUT:-build/golden}
mkdir -p $out/stdout
find $out -type f -delete

# name|seed|times|phase|env|flags; "-" for none. Poses mostly from tools/bench.sh, at short times: simulating minutes of
# flight costs minutes. The sea of cloud (function constant 8) is compiled in only while the sky clock's cloudSea is above
# 0: escarp (phase 0.005) and the path (phase -0.01) have it, the others don't. Seed 606 has an aurora on night 0 (day 0:
# no scene runs the clock to its first wrap), looked at along its azimuth; 202 does not, and its night looks at the moon (opposite the sun, at most ~20 degrees up). The path fixture is a synthetic
# drift off the escarp pose (lines "simTime x z y yawDeg pitchDeg rollDeg exposure phase day"); a VR_PATH run prints
# nothing, so its stdout text is empty.
SCENES=(
  "escarp_dawn|303|60|0.005|VR_CAM=-104,2255,40,-75,3|-"
  "autopilot_dawn|101|60|0.005|-|-"
  "tepui_noon|404|30|0.25|VR_CAM=9500,4000,60,-75,3|-"
  "towers_noon|101|30|0.25|VR_CAM=4071.2,-3942.3,40,220,14,0|-"
  "shelters_sunset|303|30|0.47|VR_CAM=300,2740,60,-90,4|-"
  "stilt_dusk|1|20|0.53|VR_SHOW=4,30,10,270,0|-"
  "aurora_night|606|30|0.75|VR_CAM=13016,5093,80,225.5,12,1|-"
  "moon_night|202|30|0.75|VR_CAM=3706,1705,320,141.6,16,1|-"
  "isle|303|10|0.12|VR_CAM=9822,-2224,480,180,4,1|-"
  "beach_battery|303|20|0.3|VR_CAM=6324,-945,29,10,-27.2,1|--battery"
  "cave_noon|404|30|0.25|VR_CAVE=40,6,0,0,1|-"
  "jungle_debug1|101|30|0.25|VR_CAM=3400,300,40,20,-6|--debug 1"
  "giant_debug3|101|30|0.25|VR_CAM=3658,-60,60,0,8|--debug 3"
  "motion|202|30|-|VR_MOTION=1 VR_MOTION_FRAMES=12 VR_FRAMES=12|-"
  "path|303|60|-|VR_PATH=tests/fixtures/golden_path.txt VR_CAM=-104,2255,40,-75,3|-"
)

failed=()
for sc in $SCENES; do
  p=("${(@s:|:)sc}")
  args=(--screenshot $out/$p[1].png --size 864x558 --scale 1 --seed $p[2] --times $p[3])
  [[ $p[4] != - ]] && args+=(--phase $p[4])
  [[ $p[6] != - ]] && args+=(${=p[6]})
  envs=()
  [[ $p[5] != - ]] && envs=(${=p[5]})
  if ! env $envs $here/vrr $app $args > $out/stdout/$p[1].raw 2>/dev/null; then failed+=($p[1]); continue; fi
  # gpu= and mem= vary from run to run and machine to machine, and the output directory with GOLDEN_OUT.
  sed -E 's/ gpu=[0-9.]+ms//; s/ mem=[0-9.]+MB//; s# -> .*/# -> #' $out/stdout/$p[1].raw > $out/stdout/$p[1].txt
done
(( ${#failed} )) && { print -u2 "golden: render failed: $failed"; exit 1 }

(cd $out && shasum -a 256 *.png stdout/*.txt) > $out/manifest.sha256
if (( update )); then
  mkdir -p $golden/stdout
  cp $out/manifest.sha256 $manifest
  cp $out/stdout/*.txt $golden/stdout/
  print "golden: manifest updated ($(wc -l < $manifest | tr -d ' ') entries)"
  exit 0
fi
[[ -f $manifest ]] || { print -u2 "golden: no manifest at $manifest"; exit 1 }
bad=$(diff <(sort -k2 $manifest) <(sort -k2 $out/manifest.sha256) | grep '^[<>]' | awk '{print $3}' | sort -u)
if [[ -n $bad ]]; then
  print -u2 "golden: FAIL, these differ from the manifest (renders in $out):"
  print -u2 -l -- "  "${(f)^bad}
  for f in ${(f)bad}; do
    [[ $f == stdout/* && -f $golden/$f && -f $out/$f ]] && diff $golden/$f $out/$f | head -6 >&2
  done
  exit 1
fi
print "golden: pass ($(wc -l < $manifest | tr -d ' ') hashes)"
