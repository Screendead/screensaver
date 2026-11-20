export const meta = {
  name: 'constitution-audit',
  description: 'Audit STRATA code against CONSTITUTION.md: each owner brings one area into line and commits; nothing is only reported',
  whenToUse: 'After a wave has merged and before the next starts, when no other worker edits the code: e.g. every few waves, or when the suite or the code has grown. Args: {base: "<branch>", tag: "<short id, e.g. a1>"}.',
  phases: [{ title: 'Audit', detail: 'eight owners, one area each, fixing in their own worktree' }],
}
const base = (args && args.base) || 'master'
const tag = (args && args.tag) || 'a1'
const REPO = '~/development/screensaver'
const ROOT = '~/development/screensaver-worktrees'
const common = (name) => `
You are a senior owner at the studio making STRATA, a generative macOS screensaver in Swift and one Metal shader (Shaders/Saver.metal includes the topic files under Shaders/). Read CONSTITUTION.md and CLAUDE.md first, then the global code rule: comments state constraints the code cannot show, and nothing else; an abstraction exists only to remove a repeated idea; notes and analysis stay out of tracked files unless they are a document someone asked for.

Your piece: bring one area of the code into line with the constitution and that rule. You fix what you find and commit it; nothing is only reported. You are not reviewing anyone: you own this area's fitness now.

Set up, with the Bash sandbox disabled (git worktree add fails inside it): cd ${REPO} && git worktree add -b strata/${name} ${ROOT}/${name} ${base}. Work only in ${ROOT}/${name}. Commit as you go (plain messages, no attribution, no model names); do not push. Do not edit CONSTITUTION.md, HANDOFF.md or .claude/. Other owners audit other areas at the same time: stay in yours.

Behaviour must not change. Prove it the cheapest way that could fail: make (the release binary and the metallib build), the unit tests that touch your area (tools/vrgpu -- swift test --filter <names>), and for anything that draws, tools/golden.sh, which must pass byte-identical. Every launch of the binary goes through tools/vrgpu or tools/vrr (CLAUDE.local.md); never take the exclusive --time lock. Never run install.sh or touch the installed saver; trash, never rm.

Hand-in (at most 150 words): what you removed or changed and why, in numbers where they exist (tests, lines, seconds), anything you judged should stay against first appearance and why, commit ids.
`
const unitTests = `YOUR AREA: the unit tests: tests/StrataCoreTests (with Support/), tests/StrataTests (BugReport/, Shader/ with ShaderProbes.metal), and tools/mutants.py. The constitution: "The eye is the test." Time them first (tools/vrgpu -- swift test, note seconds per test from XCTest's output). For each test decide: does it guard something the eye cannot see (a CPU port, constant or record layout drifting from the shader's own, a settings or table decode, the state.json bugclip.sh reads, a determinism the replays rely on) that would be costly to break unnoticed? Keep those, made fast. Delete tests that only prove a piece was done, pin today's numbers, duplicate another test, or are stale; merge near-duplicates; cut the cost of slow ones (a smaller synthetic world, fewer steps) without losing what they guard. Helpers and probes used only by deleted tests go too, and each deleted test's mutants in tools/mutants.py; run tools/mutants.py on what you changed to show the kept tests still bite. Report the before and after count and time.`
const goldenTests = `YOUR AREA: the golden renders: the SCENES in tools/golden.sh, tests/golden/ and tests/fixtures/golden_path.txt, plus CLAUDE.md's golden row. The constitution: golden proves a refactor changed nothing, never that a picture is right, and a scene earns its place only by covering a code path no other scene does. Time the run first (each scene's seconds through tools/vrr). For each scene decide what it uniquely exercises (the sea-of-cloud function constant, battery mode, a debug view, caves, towers, motion, a recorded-path replay, the aurora, the moon); merge or drop scenes whose coverage another already gives, and cut the cost of slow ones (shorter --times, fewer motion frames) where the coverage stays. Look at every PNG you keep or change (Read). Rewrite the manifest only with tools/golden.sh --update after a scene list change, never to make a render pass, and say in the commit which scenes went and why. Make the scene and hash counts in CLAUDE.md's golden row and the comment block in golden.sh match. Report the before and after scene count, hash count and time.`
const codeOwner = (area) => `YOUR AREA: ${area}. Read every file. Remove comments that narrate, restate the code, record history ("was", "before", "now", "used to", PR or report numbers), or hold analysis; keep and sharpen those stating a constraint the code cannot show (why a number is what it is, what must match the shader or the CPU, an ordering or a hardware quirk). Remove dead code (unused functions, parameters, flags, VR_* knobs nothing reads, branches no caller reaches: prove each with a search of the whole repo, including Shaders/, tools/ and the scripts), and abstractions that wrap one use. Leave formatting alone; do not rename for taste.`
const shaderOwner = (dirs) => codeOwner(`the shader topic files under ${dirs}`) + ` Also run make lint: remove any unused function or constant it reports in your files, except F_FLAT and M_ROOF, which are known and kept. A comment that names the Swift file mirroring a shader function, or a constant the CPU must match, is a constraint: keep it. tools/golden.sh must pass byte-identical after every commit that touches code.`
const parts = {
  [`${tag}-tests-unit`]: unitTests,
  [`${tag}-tests-golden`]: goldenTests,
  [`${tag}-core`]: codeOwner('Sources/StrataCore (the clock, autopilot, birds, islands, settings and tables; no Metal or AppKit)'),
  [`${tag}-gpu`]: codeOwner('Sources/Strata/GPU, Sources/Strata/Render and Sources/Strata/World (encoding the frame, the clipmap, world generation and readback)'),
  [`${tag}-app`]: codeOwner('Sources/Strata/App, Sources/Strata/BugReport, Sources/Strata/Server, Sources/Strata/Snapshot and Sources/voxelreverie (windows, menu, idle and power handling, bug reports, the render server, snapshots)') + ' The saver\'s behaviour (idle delay, dismiss on any input, no display-sleep assertions, battery mode) is Jack\'s decision: never change it.',
  [`${tag}-shaders-sky`]: shaderOwner('Shaders/Atmos, Shaders/Common, Shaders/Light, Shaders/Noise, Shaders/Passes, Shaders/Shade, Shaders/Sky, Shaders/Trace, Shaders/Water and Shaders/Saver.metal'),
  [`${tag}-shaders-land`]: shaderOwner('Shaders/Life, Shaders/Maps, Shaders/Sites, Shaders/Surface and Shaders/World'),
  [`${tag}-tools`]: `YOUR AREA: tools/ (not golden.sh or mutants.py, which the test owners hold), bugclip.sh, install.sh, uninstall.sh, Makefile, README.md, and CLAUDE.md apart from its golden row. Delete scripts nothing uses any more (search the repo, the docs and the Makefile for callers; a past brief or PR is history, not a caller), and stray files git tracks that nothing needs (compiled caches, leftovers). Make README.md and CLAUDE.md's tables match the code: every flag, VR_* switch and tool they name exists and does what they say, and every one the code has that a user needs is named (fix or cut what is wrong). Prove a script you edit still runs (its --help, or one cheap render through tools/vrr). Do not rewrite Jack's decisions or quotes, and do not run install.sh or uninstall.sh.`,
}
const picks = {
  // Opus: deciding which tests guard real risk is judgement a plausible wrong deletion would hide; medium: one bounded list each.
  [`${tag}-tests-unit`]: ['opus', 'medium'],
  [`${tag}-tests-golden`]: ['opus', 'medium'],
  // Sonnet: comment and dead-code removal in plain Swift is checked by the build, the tests and a search; medium: many files to read.
  [`${tag}-core`]: ['sonnet', 'medium'],
  [`${tag}-app`]: ['sonnet', 'medium'],
  // Opus: frame encoding and shader math, where telling a constraint from narration takes judgement golden cannot check; medium: proved byte-identical by golden.
  [`${tag}-gpu`]: ['opus', 'medium'],
  [`${tag}-shaders-sky`]: ['opus', 'medium'],
  [`${tag}-shaders-land`]: ['opus', 'medium'],
  // Sonnet: finding unused scripts and stale docs is checked by searching; medium: broad but shallow.
  [`${tag}-tools`]: ['sonnet', 'medium'],
}
phase('Audit')
const results = await parallel(Object.keys(parts).map(name => () =>
  agent(common(name) + parts[name], { label: name, phase: 'Audit', model: picks[name][0], effort: picks[name][1] })
    .then(r => ({ name, branch: `strata/${name}`, report: r }))))
return results.filter(Boolean)
