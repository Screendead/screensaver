#!/bin/zsh
# Proves a fix against a bug report: replays the report's recorded camera path (state.json) through a build and stacks
# the report's own clip (top) over the replay (bottom) in one video.
# Usage: ./bugclip.sh "<bug report folder>" [app, default build/VoxelReverie.app] [out, default <folder>/before_after.mp4]
set -e
rep=${1:?usage: ./bugclip.sh "<bug report folder>" [app] [out.mp4]}
app=${2:-${0:A:h}/build/VoxelReverie.app}
out=${3:-$rep/before_after.mp4}
tmp=$(mktemp -d)
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$rep/clip.mp4")

# The recorded frames over the clip, and a second before it to settle the temporal history and the streamed maps.
args=("${(@f)$(python3 -I - "$rep/state.json" "$dur" "$tmp" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); dur = float(sys.argv[2]); tmp = sys.argv[3]
f = s['frames']; T = f[-1]['time']; ts = s.get('timeScale', 1)
sel = [x for x in f if x['time'] >= T - dur - 1.0]
clip = [x for x in sel if x['time'] >= T - dur]
with open(tmp + '/path.txt', 'w') as o:
    for x in sel:
        p = x['position']
        o.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d\n' % (x['simTime'] / ts, p[0], p[2], p[1], x['yawDeg'], x['pitchDeg'], x['rollDeg'],
                                                                    x['exposure'], x['phase'], x['day']))
with open(tmp + '/concat.txt', 'w') as o:
    w = len(sel) - len(clip)
    for i, x in enumerate(clip):
        d = clip[i + 1]['time'] - x['time'] if i + 1 < len(clip) else 1 / 60
        # A still image's stream runs at 25 fps by default, which rounds 60 fps durations away: most frames vanished.
        o.write(f"file 'f_p{w + i:04d}.png'\noption framerate 1000\nduration {d:.6f}\n")
    o.write(f"file 'f_p{w + len(clip) - 1:04d}.png'\noption framerate 1000\n")
print(sel[0]['VR_CAM']); print(sel[0]['exposure']); print(sel[0]['simTime'] / ts); print(s['seed'])
print('%dx%d' % tuple(s['drawable'])); print('%dx%d' % (s['clip']['width'], s['clip']['height']))
print(s['renderScale']); print('--battery' if s['battery'] else ''); print('--nodither' if s.get('dither') is False else ''); print(ts)
EOF
)}")
cam=$args[1] exposure=$args[2] simTime=$args[3] seed=$args[4] size=$args[5] clipSize=$args[6] scale=$args[7] battery=$args[8] nodither=$args[9]
speed=$args[10]
# The report's replay commands carry --phase when the menu pinned Time of Day: the warm-up before the path then runs at
# the recorded hour too, so the wildlife in the air at the first frame is the recorded hour's.
pinned=(${=$(grep -o -m1 -E -- '--phase [-0-9.]+' "$rep/replay.txt" 2>/dev/null || true)})

VR_PATH=$tmp/path.txt VR_CAM=$cam VR_EXPOSURE=$exposure "$app/Contents/MacOS/voxelreverie" --screenshot "$tmp/f.png" \
    --size $size --seed $seed --times $simTime --scale $scale --speed $speed $pinned ${battery:+$battery} ${nodither:+$nodither} > /dev/null

# Labels, where Pillow is to hand (this ffmpeg has no drawtext).
labels=()
python3 -I - "$tmp" <<'EOF' 2>/dev/null && labels=(-i "$tmp/top.png" -i "$tmp/bot.png")
import sys
from PIL import Image, ImageDraw, ImageFont
f = ImageFont.load_default(size=22)
for name, text in (('top', 'Bug report clip'), ('bot', 'Same camera path, this build')):
    im = Image.new('RGBA', (int(f.getlength(text)) + 24, 40), (0, 0, 0, 150))
    ImageDraw.Draw(im).text((12, 7), text, font=f, fill=(255, 255, 255, 255))
    im.save(f'{sys.argv[1]}/{name}.png')
EOF
w=${clipSize%x*} h=${clipSize#*x}
stack="[0:v]fps=60,setpts=PTS-STARTPTS[a0];[1:v]fps=60,setpts=PTS-STARTPTS,crop=$w:$h:0:0[b0]"
if (( $#labels )); then stack="$stack;[a0][2:v]overlay=12:12[a];[b0][3:v]overlay=12:12[b]"; else stack="$stack;[a0]null[a];[b0]null[b]"; fi
ffmpeg -loglevel error -y -i "$rep/clip.mp4" -f concat -safe 0 -i "$tmp/concat.txt" $labels \
    -filter_complex "$stack;[a][b]vstack=shortest=1,format=yuv420p" -c:v libx264 -crf 16 -preset slow -movflags +faststart "$out"
trash "$tmp"
echo "$out"
