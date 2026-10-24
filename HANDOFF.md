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

## Open questions for Jack

1. **What is missing when you don't use it?** If there are days the saver isn't running, or you turn it off, say why (too hot, too slow on battery, a fault, a scene you tire of). Each answer becomes the next step.
2. **Unmerged branches.** Should `strata/islands2` (floating islands) and the other unmerged branches above be finished or archived?

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
