#!/bin/zsh
# The standard 32-frame set: dawn, noon, sunset and night on eight seeds at 1728x1117, rendered through tools/vrr, plus
# one 4x2 contact sheet per time of day (where Pillow is to hand). For milestones only: the baseline and the deliverable.
# Night pins the phase master reached at 1215 s (0.535), since the clock's pace is not fixed across builds.
# Usage: tools/render.sh <out dir> [app, default build/VoxelReverie.app] [seeds..., default 101 202 303 404 606 707 808 909]
set -e
here=${0:A:h}
out=${1:?usage: tools/render.sh <out dir> [app] [seeds...]}
app=${2:-$here/../build/VoxelReverie.app}
(( $# > 2 )) && shift 2 && seeds=($@) || seeds=(101 202 303 404 606 707 808 909)
mkdir -p "$out"
for s in $seeds; do
  $here/vrr "$app" --screenshot $out/s${s}_dawn.png   --size 1728x1117 --times 60   --phase 0.005 --scale 1 --seed $s >/dev/null
  $here/vrr "$app" --screenshot $out/s${s}_noon.png   --size 1728x1117 --times 420  --phase 0.25  --scale 1 --seed $s >/dev/null
  $here/vrr "$app" --screenshot $out/s${s}_sunset.png --size 1728x1117 --times 800  --phase 0.49  --scale 1 --seed $s >/dev/null
  $here/vrr "$app" --screenshot $out/s${s}_night.png  --size 1728x1117 --times 1215 --phase 0.535 --scale 1 --seed $s >/dev/null
done
python3 -I - "$out" <<'EOF' 2>/dev/null || true
import glob, sys
from PIL import Image
out = sys.argv[1]
for t in ('dawn', 'noon', 'sunset', 'night'):
    ims = [Image.open(f) for f in sorted(glob.glob(f'{out}/s*_{t}.png'))]
    tw, th = ims[0].width // 4, ims[0].height // 4
    sheet = Image.new('RGB', (tw * 4, th * ((len(ims) + 3) // 4)))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tw, th)), ((i % 4) * tw, (i // 4) * th))
    sheet.save(f'{out}/sheet_{t}.png')
EOF
echo "rendered $out"
