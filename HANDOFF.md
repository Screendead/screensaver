# STRATA handoff

## What STRATA is

STRATA is a generative macOS screensaver: an endless, slow flight over procedural terrain built from exact unit voxels, lit through a full day from golden hour to a moonlit night. It runs as a login agent that fades in on every display after an idle delay and gets out of the way on any input, a display-sleep assertion, a lock or a sleeping display. Everything in it (terrain, biomes, water, clouds, sky, wildlife, caves) is generated in a Metal shader and a few Swift sources, and nothing is loaded from disk except the compiled shader.

## Where things stand

Statuses come from the commit history as of 2026-10-24 (master at 440f08b). "Merged" means the work is on master; it does not mean it has been soak-tested on real hardware unless the row says so.

| Area | Status | Commits that prove it |
| --- | --- | --- |
| Terrain and day cycle | On master since 2026-10-10; mega terrain merged 2026-10-11; lighting fixes as late as 2026-10-12 | 91ee219, a7aa2c9, b87858c, 5eab04a, 25f8acf, fd986b7, c48780b |
| Biomes | Merged 2026-10-10 to 2026-10-12: autumn and painted hills, lagoons, salt flats, hamlets and towers; a menu can pin one biome and pinned biomes keep only their own landforms | 3ba1881, 52f7851, 80c0bc5, 5c09a40, 81e7019, 87247f1, f19c791 |
| Wildlife | Boid flocks (songbirds, raptors, gulls, geese, starlings, bats) on master 2026-10-10; birds never pop in or out of view (2026-10-10, kept under mega terrain 2026-10-11); coral reefs and fish wakes merged 2026-10-12 | 7404e6c, b5c0805, fd986b7, 3a8bb97 |
| Night glow | Sea glow, forest glow and glowing bugs merged 2026-10-12; distant glow brighter with distance and less lost to fog merged 2026-10-12 | 1842545, 3a8252c, 988778a, b7f16a4 |
| Caves | Merged 2026-10-12. Off-screen soak 2026-10-24 (below): two cave approaches at live size, median 6.3 and 4.8 ms, no slow frames after the first | 3e19413, ed9a4ee, 8e68647 |
| Performance | Three rounds merged 2026-10-11 to 2026-10-12: cell-grid drawable (-176 MB), off-main far readback, CPU per frame -43%, adaptive idle poll, shared pipelines across screens, shading skips | 319d1e7, c41e403, 685c113 |
| Menu | Menu bar item merged 2026-10-12: start now, pause, idle delay, dithering; later Time of Day presets, biome pinning, S for a screenshot | 871f5e1, ac00d66, 1a2dfd3, 87247f1, 7c27069 |
| Bug reports | B while the saver shows files a report (clip, per-frame state, replay commands, note); merged 2026-10-12 | 55bfef4, 4478756 |
| Render server | `--serve` answers `--screenshot` requests without a fresh process; prewarm race fixed; merged 2026-10-12 | 4abe916, aae1492, e394297 |
| bugclip | `bugclip.sh` added 2026-10-23 and used to prove PR #1 (giants' skirts across the morph band), merged 2026-10-23 | d661c58, 74d61e1, 2b71c8e |

Branches with commits that master lacks (from `git branch --no-merged master`): `strata/islands2` (31 commits, floating islands), `strata/islands-fix`, `strata/islands-b`, `strata/islands`, `strata/giant-C3` and `strata/giant-C2-roots` (morph-band work from 2026-10-13, before PR #1), `strata/vines3`, `strata/waves`, `strata/tundra`, `strata/merge4`, and single-commit prototypes. None of them is described in a merge message, so whether any is still wanted is unknown.

## Goal (Jack, 2026-10-24)

STRATA has no finish line: it evolves for as long as it is used. The goal is a screensaver Jack enjoys and finds beautiful, that stays fast and efficient on battery as well as on mains. The test is that Jack actually uses it on his MacBook; if he doesn't, something is missing, and finding what is the work. It stays a personal install (`install.sh`) for now; no signing or release.

Auto-start was off until 2026-10-24, so the saver never came on by itself. It is now on with a 15 s idle delay; the saver still holds off while a video or other app keeps the display awake. Watch whether that changes how often Jack sees it.

The flight runs at twice its old speed (11-30 voxels/s by height, Jack's call 2026-10-25). 15-minute flights on seeds 1, 303 and 808 kept their clearance and never dropped low; frame times held (no frame over 16.7 ms on mains or battery).

## Open questions for Jack

1. **What is missing when you don't use it?** If there are days the saver isn't running, or you turn it off, say why (too hot, too slow on battery, a fault, a scene you tire of). Each answer becomes the next step.
2. **Shooting stars (prototype, `strata/idea-shooting-stars`, 2026-10-25).** On a dark night, one meteor in each 30 s of the clock (wall time), 0.4-0.8 s, a streak a cell or two wide and up to ~20 cells long on the dither grid with a fading warm tail. Dark sky lasts about 2 minutes of each 12-minute day (deep night runs at 5x), so that is 3 to 5 a night (10 nights x 5 seeds: mean 4.1, never fewer than 3, gaps 12-49 s); the first draft's random 1-in-5 slots gave 0 to 8 (same 50 nights: 17 had 2 or fewer, one had none). On renders of two nights 7 of 9 showed; clouds hid the other two, and clouds and terrain hide them like the stars, never by day. Unbroken on the battery path too. Frames with no meteor in the last few seconds are byte-identical to master (day and night, mains and battery), and GPU time on two night scenes is unchanged (-0.04 and +0.12 ms, noise). Each activation starts a fresh clock at blue hour, so the first dark sky comes 9 minutes in (the second at 21): a shorter session sees none, unless the menu pins Time of Day to night, which gives one every 30 s throughout. Keep, tune (rarer, longer) or drop? `tools/meteors.py <seed> --nights N` lists them against the live clock.
3. **Unmerged branches.** Should `strata/islands2` (floating islands) and the other unmerged branches above be finished or archived?

## Bug reports (triaged 2026-10-24)

Every report under `~/Library/Application Support/STRATA/Bug Reports/` replayed through master 67ef343 with `bugclip.sh`, judged from frames and region measurements, and each verdict checked by a second agent trying to refute it (none disputed). Videos and evidence frames are in each report folder (`triage_before_after.mp4`, `triage/`, `verify/`, `fixcheck*`).

| Report | What Jack saw | Status | Evidence |
| --- | --- | --- | --- |
| 10-12 09.42 | Hometree vines falling and receding | Fixed (54532f4, then PR #1) | Vines run unbroken at constant length |
| 10-12 11.38 | Flickery water highlights | Fixed (e3a7050) | Single-frame blinks about 4x lower, at the noise floor of edges |
| 10-12 12.39 | Specular water highlights wrong | Fixed | Mean frame difference on the water 9.3 to 1.7 |
| 10-12 12.54 | Moon reflection too large | Fixed (a1a3357) | Path about 0.7x as wide |
| 10-12 13.13 | Near-camera glow sporadic, tacky | Fixed (stable) | No confetti or brightness jumps; whether it reads as Avatar is Jack's call |
| 10-12 16.12 | Vines still messed up | Fixed | Needed the phase fix in bugclip: at night the vines reach the ground in every frame, no floating pieces |
| 10-12 17.46 | Foliage disappears near the bottom | Fixed | Bare ground in the lower third stays 5-7% (was climbing to 31%) |
| 10-12 17.57 | Black bars on mountainsides | Fixed | Bands 2-3x brighter, level with the rock |
| 10-12 21.56 | Karst in a grassland-only world | Fixed (f19c791), judged from code | The fix raised the terrain, so the replay camera sits inside it |
| 10-12 22.00 | Light lines in the forest at night | Fixed (c48780b) | Crest lines gone |
| 10-23 16.05 | Tree vines weird again | Fixed (PR #1) | Dark streaks per frame 78 to 2 |
| 10-24 22.21 | Bars travelling down a tower with the camera | Physics, not a bug (kept) | Dawn cloud shadows at a 3-degree sun swing fast as the 12-minute day lifts the sun; camera held still, they still slide. A clamp (PR #7) was reverted at Jack's call. If it grates, the lever is a slower clock around sunrise, not the shadows |
| 10-25 00.21 | Tree vines still scuffed (far giant) | Fixed (fix/giant-band-scuff) | Not the vines: the same pieces show with the strand pass off. Columns under the crown's rim slid as slabs through the air across level 1's morph band (1376-1720 voxels). In a dolly through the band, pixels jumping around the tree 11.1 to 0.3 a frame; level-2 frames byte-identical. Needs a live look |
| 10-25 00.25 | Flicker on dark foliage before dawn | Fixed by Jack's option B (strata/shadow-dither) | The screen-fixed 8-level Bayer dither crawled under motion; near black each step was a whole level. Four times finer steps below 2/7: dark-area flips per frame on the replay 3.31% to 0.52% (dither off was 0.77%), blinks 0.28% to 0.01%; mid-tones and highlights keep the 8 levels. Needs a live look |

Taste questions for Jack, not faults: should moon glints roll off softly rather than clip to flat white, and is the dense blue glow near the camera the Avatar look he wants?

`bugclip.sh` itself was the open fault: it replayed every report at the free-running clock, so night reports filed with Time of Day pinned replayed at dusk; it ignored dither off; and ffmpeg kept only about 25 of each 60 replayed frames. Fixed on `fix/bugclip-phase`.

## Soak results (2026-10-24)

`tools/soak.sh`, seed 1, 3456x2234, exclusive GPU lock, on mains power (`--battery` selects the battery path's 0.7 scale; the machine was plugged in). 30 s of frame-by-frame flight at t = 60, 600, 1200 and 1800 s, then two cave approaches from 300 voxels.

| Run | Frames | Median | p95 | p99 | Worst |
| --- | --- | --- | --- | --- | --- |
| Flight, mains path | 7200 | 6.39 ms | 8.43 | 9.29 | 33.16 |
| Flight, battery path | 7200 | 3.74 ms | 5.98 | 6.50 | 14.81 |
| Approach, big cave in clear view | 280 | 6.31 ms | 7.94 | 10.11 | 10.32 |
| Approach, lived-in cave | 280 | 4.83 ms | 6.05 | 10.32 | 22.19 |

- Every frame over 16.7 ms is the first after a simulated time jump (cold temporal history and freshly built maps); steady flight peaks at 12.3 ms. The live saver never jumps, so these do not show live.
- GPU memory in snapshot runs climbed with simulated time (433 MB at 60 s, 2.9 GB at 7200 s). It was an undrained autorelease pool in the snapshot path, fixed in 43ac04a (411 MB throughout). The live saver drains its pool every frame, so it should not have had this; the power soak measures the live process's memory to confirm.
- Not measured: watts, battery drain and heat. Frame times suggest the cost but do not prove it.

## The plan

**Next step: the battery soak on Jack's MacBook, overnight.** Run `tools/powersoak` (after `make`) and leave the laptop unplugged and untouched: it waits for battery power and 5 idle minutes, measures 20 minutes of baseline with the display held on, then runs the live saver until the battery reaches 3%, logging watts, battery %, battery temperature and the saver's memory every 30 s into `build/power/`. Its `summary.txt` gives the saver's watts over baseline, battery %/hour and whether memory stays flat; those decide the next step.
