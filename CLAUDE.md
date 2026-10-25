# STRATA

A generative macOS screensaver (Swift + one Metal shader). README.md describes behaviour, flags and layout; HANDOFF.md says where things stand and what comes next.

## Build

```sh
make            # builds build/VoxelReverie.app
./install.sh    # build, install to ~/Applications and (re)start the login agents
```

## Snapshot check

Render an offscreen PNG at the live size (the scene renders on the 864-cell dither grid, so scale 1 is the AC live size):

```sh
build/VoxelReverie.app/Contents/MacOS/voxelreverie --screenshot out.png --size 3456x2234 --scale 1 --seed 1 --times 10,60
```

Run every launch of the binary through the GPU lock, `tools/vrgpu -- <binary> ...`, and any run whose ms you will report through `tools/vrgpu --time -- <cmd>`; `tools/vrgpu --status` shows who holds it.

`--phase P`, `--pitch R` and `--debug M` override the day phase, camera pitch and debug view. Look at every PNG you render.

## Tools

| Tool | Use |
| --- | --- |
| `tools/vrgpu -- <cmd>` / `--time -- <cmd>` / `--status` | GPU lock: renders share it, timing runs hold it alone |
| `tools/vrr <app> --screenshot ...` | The same render through a warm per-build server (byte-identical; skips the shader compile) |
| `tools/render.sh <out dir> [app]` | The standard 32 frames (8 seeds x dawn, noon, sunset, night) and four contact sheets; milestones only |
| `tools/vrgpu --time -- tools/bench.sh ROUNDS "a=<app> b=<app>" [regex]` | A/B GPU ms over twelve fixed scenes |
| `tools/flicker.py <dir or glob> [--skip n] [--crop WxH]` | Frame-to-frame flicker of a PNG sequence (a report's decoded clip, VR_PATH or VR_FRAMES renders): flip and one-frame blink rates in dark and bright regions |
| `tools/soak.sh` | Off-screen soak: frame-by-frame flight on mains and battery plus two cave approaches, with frame-time percentiles |
| `tools/powersoak [app]` | On the laptop, unplugged: baseline then the live saver until 3% battery, logging watts, battery, heat and memory |

## VR_* switches (from README)

| Switch | Effect |
| --- | --- |
| `VR_TIMING=1` | Log GPU frame times and world regeneration |
| `VR_DEBUG=1` | Log idle and activation decisions |
| `VR_MOTION=1` (`VR_MOTION_FRAMES=n`, `VR_FRAMES=n`) | Advance the camera between snapshot frames and dump them |
| `VR_CHUNKED=1` | Update the LOD rings a piece per frame, as live |
| `VR_COARSE_FULL=1` | Rebuild the LOD rings whole every frame |
| `VR_LODSHIFT=v` | Move the rings' centre v voxels ahead of the camera |
| `VR_POWER_FLIP=i` | Switch power source at frame i |
| `VR_EXPOSURE=e` | Hold a bug report's recorded exposure |
| `VR_BUGTEST=s` | Fly s seconds recording as the saver does, then write a report into `<screenshot>_reports/` |
| `VR_PATH=file` | Render a recorded camera path frame by frame |
| `VR_CAVES=r` | List the caves within r of each new map's centre (and count the lowland ones) |
| `VR_CAVE=distance,height,bearing[,rank[,which]]` | Hold the camera on the rank-th nearest cave (which: 1 lived-in, 2 big and seen clear, 3 grottoes) |
| `VR_DOLLY=v` | Move a held camera v voxels a frame along its heading |
| `VR_NOCAVES=1` | Leave the cave table empty (renders as before caves) |

## Proving a bug fix

Every fix is proved against the bug report that found it. Reports live in `~/Library/Application Support/STRATA/Bug Reports/<date>/` (press B while the saver shows). After `make`:

```sh
./bugclip.sh "<bug report folder>" [app] [out.mp4]
```

It replays the report's recorded camera path through the build and stacks the report's clip (top) over the replay (bottom) in `<folder>/before_after.mp4`. Watch it, and cite it in the PR.

## Branches

Work on a topic branch (`strata/<topic>` or `fix/<topic>`) and merge it to master through a GitHub PR, as PR #1 (`fix/giant-skirt-morph`) did. Don't commit straight to master.

## End of session

Update HANDOFF.md before the session ends: the status table, the open questions, and one next step.

## Never

Put personal details or paths containing user names in tracked files.
