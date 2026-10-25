# STRATA

A generative screensaver for macOS. An endless, slow flight over Perlin-noise terrain built from exact unit voxels. Each one-voxel step of the smooth noise field is a contour line, and the piece lights those contour lines through a full day. It runs through golden and blue hour, midday, and a moonlit night.

Everything is procedural, including terrain, mesas and arches, rock strata, water, stepped voxel clouds, sky, stars and the blocky moon. Nothing is loaded from disk except the compiled shader.

## How it runs

`install.sh` builds the app, copies it to `~/Applications/VoxelReverie.app` and registers a login agent (`com.jacklusher.voxelreverie`). The agent watches HID idle time:

- After 30 s with no input, it fades in fullscreen on every display.
- It doesn't start while something holds a display-sleep assertion. Video players, calls and presentations all do that.
- It doesn't start while the screen is locked or the displays are asleep.
- Any mouse or keyboard input dismisses it within a frame or two.
- B instead files a bug report: the last 6 to 8 s of the first screen as video, the camera, clock and GPU time for the last 10 s, and replay commands, in `~/Library/Application Support/STRATA/Bug Reports/<date>/`. A window then asks what you saw (Return saves `note.txt`, Esc skips). The menu bar item's "Bug Reports…" opens the folder.
- It never keeps the display awake. Your normal display-sleep and lock settings still apply, and it gets out of the way when the display sleeps or the system screensaver starts.
- On battery or in Low Power Mode it renders at a lower scale and 30 fps.
- On EDR-capable displays, highlights extend into HDR headroom: lit step lips, sun glitter on water, the sun, the moon and bright stars. Headroom is capped at 4x.

It replaces the system screensaver. If you like, set System Settings › Lock Screen › "Start Screen Saver when inactive" to Never, so the two don't compete.

```sh
./install.sh      # build + install + start the agent
./uninstall.sh    # stop the agent and move the app to the Bin
```

## Flags

| Flag | Meaning |
| --- | --- |
| `--preview` | Show immediately; exits on input |
| `--duration N` | Preview: exit after N seconds |
| `--idle N` | Idle threshold in seconds; overrides the menu's setting (default 30) |
| `--menu` | Run the menu bar item (the installed second agent) |
| `--nodither` | Flat quantised colour, no ordered-dither pattern |
| `--seed N` | Fixed world seed (default: random per activation) |
| `--speed X` | Time multiplier (soak testing). Snapshots honour it too: `--times 60 --speed 20` simulates 1200 s |
| `--battery` | Force the battery-quality path |
| `--screenshot out.png --times 10,60 --size WxH --scale S` | Render offscreen PNGs after simulating to each time |
| `--phase P`, `--pitch R`, `--debug M` | Snapshot overrides: day phase (0 = sunrise), camera pitch, debug view |

Environment: `VR_TIMING=1` logs GPU frame times and world regeneration, and `VR_DEBUG=1` logs idle and activation decisions.
Snapshot test switches: `VR_MOTION=1` (with `VR_MOTION_FRAMES=n`, `VR_FRAMES=n`) advances the camera between frames and dumps them,
`VR_CHUNKED=1` updates the LOD rings a piece per frame as live, `VR_COARSE_FULL=1` rebuilds them whole every frame,
`VR_LODSHIFT=v` moves the rings' centre v voxels ahead of the camera, and `VR_POWER_FLIP=i` switches power source at frame i.
`VR_EXPOSURE=e` holds a bug report's recorded exposure, and `VR_BUGTEST=s` flies s seconds recording as the saver does, then
writes a report into `<screenshot>_reports/`. `VR_PATH=file` renders a recorded camera path frame by frame, and
`./bugclip.sh "<bug report folder>" [app] [out.mp4]` uses it to prove a fix: the report's clip over the same camera path
through the build, in one video.
Caves: `VR_CAVES=r` lists the caves within r of each new map's centre (and counts the lowland ones), `VR_CAVE=distance,height,bearing[,rank[,which]]` holds the
camera on the rank-th nearest cave (which: 1 lived-in, 2 big and seen clear, 3 grottoes), `VR_DOLLY=v` moves a held camera v voxels
a frame along its heading (approaches across the rings), and `VR_NOCAVES=1` leaves the cave table empty (renders as before caves).
Live frame times are measured at `--size 3456x2234 --scale 1`: the scene renders on the 864-cell dither grid, so scale 1 is the AC live size.

## Layout

- `Sources/main.swift`: idle agent, power and lock probes, fullscreen windows, dismissal.
- `Sources/BugReport.swift`: the B-key bug report: hardware-encoded rolling capture, per-frame state, report files, note window.
- `Sources/Renderer.swift`: world (2048² column map with max mips, async double-buffered regeneration), day clock, frame setup, TAA, snapshots.
- `Sources/Camera.swift`: valley- and ridge-following autopilot with a pitch cycle, so the horizon sweeps the whole screen.
- `Shaders/Saver.metal`: noise, terrain generation, the quadtree voxel DDA, shading, sky, clouds, fog, water, and the temporal and EDR post passes.
- `tools/golden.sh`: output regression check; renders a fixed set and compares sha256s with `tests/golden/manifest.sha256` (`tests/fixtures/golden_path.txt` is its recorded-path replay).
