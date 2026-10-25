# STRATA handoff

## What STRATA is

STRATA is a generative macOS screensaver: an endless, slow flight over procedural terrain built from exact unit voxels, lit through a full day from golden hour to a moonlit night. It runs as a login agent that fades in on every display after an idle delay and gets out of the way on any input, a display-sleep assertion, a lock or a sleeping display. Everything in it (terrain, biomes, water, clouds, sky, wildlife, caves) is generated in a Metal shader and a few Swift sources, and nothing is loaded from disk except the compiled shader.

## Where things stand

Statuses come from the commit history as of 2026-10-25 (master after PR #24; earlier rows as of 2026-10-24, master at 440f08b). "Merged" means the work is on master; it does not mean it has been soak-tested on real hardware unless the row says so.

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
| Overnight picks | PR #17, merged 2026-10-25: Jack's picks in one bundle: floating islands, aurora (1.8x brighter), moon phases (darker crescent nights), shooting stars, evening star, wet sand, the dawn sea of cloud (compiled into the scene and vine passes only while it is out: the cloud-sea gate), the sun's full turn a day, moon glints C, sunrise clock kept at 12 minutes | 46ca68e, 35b6244, 9d99059 |
| Cloud shadows at the horizon | PR #18, merged 2026-10-25: cloud shadows fade in with the sun's height (none at the horizon, full by L.y 0.5) instead of cutting off at 0.02, so the land no longer flashes as the sun crosses the horizon | 934e770, 950ac83 |
| Sunrise face jump | PR #19, merged 2026-10-25: the key light holds the afterglow's height (0.05) until the sun clears it, and terrain shadows fade in from the handover (sun -0.02 to 0.03); sunlit faces no longer darken by up to 24/255 in one frame | c091e27, a5ce882 |
| Output oracle | PR #20, merged 2026-10-25: `tools/golden.sh` renders 15 scenes and checks 51 sha256s against `tests/golden/manifest.sha256`; every refactor below passed it unchanged. Also the picks-perf tidy-up (no measurable speed difference) | 3bf3c3e |
| Shader layout | PR #21, merged 2026-10-25: `Shaders/Saver.metal` is an include list over 84 topic files under `Shaders/<topic>/`; duplicates and dead code removed, output byte-identical | 79c796d |
| Tools layout | PR #22, merged 2026-10-25: `tools/perf/` (bench, soak, powersoak), `tools/frames/`, `tools/sky/`; one-off scripts on local `archive/one-off-tools`; `bugclip.sh` under the GPU lock | f9fece0 |
| Swift layout and unit tests | PR #23, merged 2026-10-25: three SwiftPM targets (`StrataCore` without Metal or AppKit, `Strata`, `voxelreverie`), one job per file, every VR_* switch in `Knobs`; `make test` runs 42 XCTest tests (`StrataTests` checks the CPU's ports, constants and record layouts against the shader's own; `StrataCoreTests` flies the camera and birds over a synthetic world) then `tools/golden.sh`; `tools/mutants.py` shows each test bites | 80f26c8 |
| Cross-references | PR #24, merged 2026-10-25: comments name the new Swift and shader files, `Sight.sea` replaces the Swift copies of 19.85, `make lint` (`tools/lint.sh`) finds unused shader code (only `F_FLAT`, `M_ROOF`, kept) | 5bd0598 |

Branches with commits that master lacks (from `git branch --no-merged master`): `strata/islands2` (31 commits, floating islands), `strata/islands-fix`, `strata/islands-b`, `strata/islands`, `strata/giant-C3` and `strata/giant-C2-roots` (morph-band work from 2026-10-13, before PR #1), `strata/vines3`, `strata/waves`, `strata/tundra`, `strata/merge4`, and single-commit prototypes. None of them is described in a merge message, so whether any is still wanted is unknown.

## Goal (Jack, 2026-10-24)

STRATA has no finish line: it evolves for as long as it is used. The goal is a screensaver Jack enjoys and finds beautiful, that stays fast and efficient on battery as well as on mains. The test is that Jack actually uses it on his MacBook; if he doesn't, something is missing, and finding what is the work. It stays a personal install (`install.sh`) for now; no signing or release.

Auto-start was off until 2026-10-24, so the saver never came on by itself. It is now on with a 15 s idle delay; the saver still holds off while a video or other app keeps the display awake. Watch whether that changes how often Jack sees it.

The flight runs at twice its old speed (11-30 voxels/s by height, Jack's call 2026-10-25). 15-minute flights on seeds 1, 303 and 808 kept their clearance and never dropped low; frame times held (no frame over 16.7 ms on mains or battery).

## Open questions for Jack

1. **What is missing when you don't use it?** If there are days the saver isn't running, or you turn it off, say why (too hot, too slow on battery, a fault, a scene you tire of). Each answer becomes the next step.
2. **Moon handover: shadows switch on in one frame.** Below the horizon, as the key light blends from the afterglow to the moon, terrain shadows switch on at once (`L.shadows = m > 0.5` in `makeLight`, `Shaders/Sky/Light.metal`): the same kind of step PR #19 removed at sunrise. A fade like #19's was offered; whether the step shows has not been measured.
3. **Battery soak.** `tools/perf/powersoak` needs the laptop unplugged overnight; it is the only measure of watts, drain and heat on battery.
4. **Vine pass terrain bound (possible bug, changes output).** The vine pass bounds its terrain trace with the coarse levels' top mips without `abs()` (`Shaders/Passes/Vines.metal`), where the scene pass takes `abs()` (`Shaders/Passes/Scene.metal`): a negative max flags an open band, so the vine pass may trace too low a ceiling over a crown or an island. Fixing it changes output; your call.
5. **Overnight pitches (2026-10-25), merged as PR #17.** Each has its own branch, pushed, with media in `build/overnight/` and a report page for Jack. Recommended picks in brackets:
   - **All picks in one merge: `strata/overnight-picks`** holds every branch below at its bracketed pick (cloud sea included), conflicts resolved, one PR away from master.
   - **Jack's picks: `strata/overnight-picks2`** is that bundle (with the cloud-sea gate) at Jack's answers where they differ from the brackets, marked "Jack:" below.
   - Floating islands, `strata/islands3` (master merged in, conflicts reviewed): merge now [yes]; follow-ups: birds no longer circle islands at the doubled flight speed, island tops dark at night.
   - Moon glints, `archive/strata-pitch-moon-glints`: A keep / B soft roll-off [B] / C finer. Jack: C.
   - Near-camera night glow, `archive/strata-pitch-near-glow`: A keep / B 40% lower near [B] / C plus slower pulse.
   - Sunrise clock, `archive/strata-pitch-slow-dawn`: A keep / B slow dawn in 12 min / C slow dawn, ~15-min day [C]. Jack: A, the 12-minute clock stays (slow dawn removed).
   - Archive waves, tundra, drama, mesa, noonart, texel, far2 (rename to `archive/`) [yes].
   - Sun jumps half a turn at every dawn wrap (a bug on master since the clock was written), `strata/fix-sun-wrap`: A full turn a day / B catch up at night / C out by day, back by night [C]. Jack: A, a full turn a day (sun sets opposite where it rose).
   - Prototypes, each reviewed and passed: `archive/strata-idea-aurora-nights`, `archive/strata-idea-moon-phases`, `archive/strata-idea-shooting-stars`, `archive/strata-idea-evening-star`, `archive/strata-idea-wet-sand` [merge each]. Jack: merge each, the aurora brighter and moon phases with darker crescent nights. `archive/strata-idea-crepuscular-rays` was dropped (beams never form). `archive/strata-idea-cloud-sea` (dawn sea of cloud in basins) is still being tuned: it must not change with the camera's height.
   - Fixed overnight without asking (defects): far giants shedding slabs in the morph band (PR #10), dark-foliage dither flicker (PR #13, Jack's pick), and birds starved by the doubled flight speed (PR #15: raptors and bats never spawned).

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

**Next step: the battery soak** (`tools/perf/powersoak`, on the laptop unplugged overnight; needs Jack to unplug it). The overnight picks, the horizon fixes and the refactor are merged; what the saver costs on battery is the one goal still unmeasured.
