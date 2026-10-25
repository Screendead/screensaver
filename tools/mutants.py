#!/usr/bin/env python3
"""Shows that each unit test bites: makes one deliberate change to the code a test guards, runs that test, and restores
the file with git checkout. A mutant the test lets through is a test to strengthen.

  tools/mutants.py [regex]    run the mutants whose test name matches (all by default)

Run from a clean tree (it refuses a file with uncommitted changes). The GPU tests run under tools/vrgpu.
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S, C = 'Sources/StrataCore/', 'Sources/Strata/'

# (test, file, old, new): old must occur exactly once in file.
MUTANTS = [
    # Shader contracts: the CPU side, and the shader side once.
    ('ShaderFunctionTests/testHashesMatchBitForBit', S + 'World/Provinces.swift', '&* 277803737', '&* 277803739'),
    ('ShaderFunctionTests/testHashesMatchBitForBit', 'Shaders/Saver.metal', '* 1597334677u ^ pcg(uint(p.y) * 3812015801u ^ s)); }',
     '* 1597334677u ^ pcg(uint(p.y) * 3812015803u ^ s)); }'),
    ('ShaderFunctionTests/testProvincesMatchTheShader', S + 'World/Provinces.swift', 'smoothstep(0.24, 0.32,', 'smoothstep(0.25, 0.32,'),
    ('ShaderFunctionTests/testKeyLightMatchesMakeLight', S + 'Sky/KeyLight.swift', 'max(-sun.y * 0.7, 0.08)', 'max(-sun.y * 0.7, 0.10)'),
    ('ShaderFunctionTests/testCloudSeaMatchesTheShader', S + 'Sky/SkyClock.swift', 'smoothstep(-0.24, -0.08, e)', 'smoothstep(-0.24, -0.10, e)'),
    ('ShaderFunctionTests/testLitSitesMatchLampLevel', S + 'Camera/FlightCamera+Shots.swift', '% MapLayout.maxLamps) &* 7717)',
     '% MapLayout.maxLamps) &* 7719)'),
    ('ShaderLayoutTests/testConstantsMatchTheShader', S + 'World/MapLayout.swift', 'caveCell = 256.0', 'caveCell = 128.0'),
    ('ShaderLayoutTests/testLayoutsMatchTheShader', C + 'GPU/Frame.swift', '    var lights = SIMD4<Float>()\n    var refs = SIMD4<Int32>()\n',
     '    var refs = SIMD4<Int32>()\n    var lights = SIMD4<Float>()\n'),
    # Bug reports.
    ('BugReportStateTests/testHoldsEveryKeyBugclipReads', C + 'BugReport/BugReport.swift', '"yawDeg": num(r.yaw * d)', '"yaw": num(r.yaw * d)'),
    ('BugReportStateTests/testHoldsEveryKeyBugclipReads', C + 'BugReport/BugReport.swift', 'state["renderScale"] = num(renderScale)\n', ''),
    ('BugReportStateTests/testFramesReplayAsRecorded', C + 'BugReport/BugReport.swift', '"pitchDeg": num(r.pitch * d)', '"pitchDeg": num(r.pitch)'),
    ('BugReportStateTests/testCamHoldsTheRecordedPose', C + 'BugReport/BugRecorder.swift', r'"\(pos.x),\(pos.z),\(pos.y),', r'"\(pos.x),\(pos.y),\(pos.z),'),
    # The sky clock and key light.
    ('SkyClockTests/testOpensBeforeSunrise', S + 'Sky/SkyClock.swift', 'phase: Double = -0.035', 'phase: Double = 0.035'),
    ('SkyClockTests/testADayLastsAbout713SecondsWithTheSunUpFor460', S + 'Sky/SkyClock.swift', '? 5.0 : 0.5', '? 4.0 : 0.5'),
    ('SkyClockTests/testADayLastsAbout713SecondsWithTheSunUpFor460', S + 'Sky/SkyClock.swift', ') + 0.07', ') + 0.09'),
    ('SkyClockTests/testSunPeaksAt049AndSinksAsFar', S + 'Sky/SkyClock.swift', '0.42 : 0.56', '0.45 : 0.56'),
    ('SkyClockTests/testSunTurnsOnceADayWithoutAJumpAtTheWrap', S + 'Sky/SkyClock.swift', '0.9 + 2 * .pi * (phase - floor(phase))',
     '0.9 + 1.9 * .pi * (phase - floor(phase))'),
    ('SkyClockTests/testNightIndexStepsAtMidMorningNotAtTheWrap', S + 'Sky/SkyClock.swift', '< 0.25 ? n - 1 : n', '< 0.25 ? n : n'),
    ('SkyClockTests/testNightProgressAndCloudSeaAreContinuous', S + 'Sky/SkyClock.swift', 'return ph < 0.25 ?', 'return ph < 0.15 ?'),
    ('KeyLightTests/testKeyLightHasNoStepAtItsHandovers', S + 'Sky/KeyLight.swift', 'if e > -0.02 { return (simd_normalize(SIMD3(sun.x, 0.05,',
     'if e > -0.02 { return (simd_normalize(SIMD3(sun.x, 0.06,'),
    ('KeyLightTests/testOnlyTheSunOrTheMoonCastsShadows', S + 'Sky/KeyLight.swift', 'm > 0.5)', 'm > 1.5)'),
    # Settings.
    ('SettingsTests/testParsesASnapshotCommand', S + 'Config/Settings.swift', 'CGSize(width: p[0], height: p[1])', 'CGSize(width: p[1], height: p[0])'),
    ('SettingsTests/testDefaultsLeaveTheSaversChoices', S + 'Config/Settings.swift', 'package var phase: Double?', 'package var phase: Double? = 0'),
    ('SettingsTests/testToleratesBadInput', S + 'Config/Settings.swift', 'if p.count == 2 {', 'if p.count >= 1, p.count <= 2 {'),
    ('SettingsTests/testBiomeByNameOrNumber', S + 'Config/Biome.swift', '$0.caseInsensitiveCompare(name) == .orderedSame', '$0 == name'),
    ('SettingsTests/testBiomeRoundTripsThroughTheSeed', S + 'Config/Biome.swift', 'UInt32(biome) << 20', 'UInt32(biome) << 19'),
    # Table decoders.
    ('MapTablesTests/testSitesDecodeWithTheirLamps', S + 'World/MapTables.swift', 'code / 32 - 16', 'code / 32 - 15'),
    ('MapTablesTests/testGiantsAndCavesDecodeFromTheirSharedBuffer', S + 'World/MapTables.swift', 'p[$0].flags & 1 != 0', 'p[$0].flags & 2 != 0'),
    ('MapTablesTests/testTowersDecodeTheirPackedFields', S + 'World/TowerInfo.swift', 'Double((d.w >> 8) & 255)', 'Double((d.w >> 16) & 255)'),
    ('IslandsTests/testParsesStandingSlots', S + 'Islands/Islands.swift', 'massif: i >= win * win', 'massif: i > win * win + 2'),
    ('IslandsTests/testShadingTableHoldsTheNearestFirst', S + 'Islands/Islands+ShadingTable.swift', '.sorted { simd_length($0.pos - p) < simd_length($1.pos - p) }',
     '.sorted { simd_length($0.pos - p) > simd_length($1.pos - p) }'),
    ('IslandsTests/testOcclusionReachIsWhereTheSkyComesBack', S + 'Islands/Islands+Bounds.swift', 'if worst(m) < 0.002', 'if worst(m) < 0.003'),
    ('IslandsTests/testBoundingSpheresSeeAnIslandOnlyWhereTheViewCanReach', S + 'Islands/Islands+Bounds.swift',
     '< half + margin + asin(min(1, r / l))', '< half + margin + 3 + asin(min(1, r / l))'),
    # The autopilot.
    ('FlightCameraTests/testASeedFliesTheSamePathEveryTime', S + 'Camera/FlightCamera+Autopilot.swift',
     'sin(t * 0.0131 * k + 1.3) * 0.5', 'sin(t * 0.0131 * k + 1.3) * 0.5 + Double.random(in: 0..<1e-6)'),
    ('FlightCameraTests/testLongFlightsAreSmoothAndKeepTheirClearance', S + 'Camera/FlightCamera+Autopilot.swift',
     'climbRate += max(-accel * dt, min(accel * dt, want - climbRate))', 'climbRate = want'),
    ('FlightCameraTests/testLongFlightsAreSmoothAndKeepTheirClearance', S + 'Camera/FlightCamera+Terrain.swift',
     'm = max(m, h(x + dx, z + dz))', 'm = max(m, h(x + dx, z + dz) - 60)'),
    # The birds.
    ('WildlifeTests/testASeedRaisesTheSameBirds', S + 'Wildlife/Wildlife.swift', 'return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)',
     'return Double.random(in: 0..<1)'),
    # Birds steer by both lookups, either alone keeps them clear: lowered together, they fly into the hills.
    ('WildlifeTests/testFlocksStayInTheAirAndInTheBuffer', S + 'Wildlife/Wildlife.swift',
     '{ heightAt(x, z) ?? 200 }\n    // The ground as a bird at height y meets it: well under an island, the land beneath it; else the island\'s top.\n'
     '    package func ground(_ x: Double, _ z: Double, at y: Double) -> Double {\n        guard let c = columnAt(x, z) else { return 200 }\n'
     '        if let u = c.under, y < u - 12 { return c.ground }\n        return c.top\n',
     '{ (heightAt(x, z) ?? 200) - 100 }\n    package func ground(_ x: Double, _ z: Double, at y: Double) -> Double {\n        guard let c = columnAt(x, z) else { return 200 }\n'
     '        if let u = c.under, y < u - 12 { return c.ground }\n        return c.top - 100\n'),
    ('WildlifeTests/testUploadWritesEveryBirdWhereTheShaderReadsIt', S + 'Wildlife/Wildlife+Upload.swift',
     'let birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2', 'let birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2 + 8'),
    ('WildlifeTests/testOnlyBatsFlyAtNight', S + 'Wildlife/Wildlife+Step.swift', 'if sunY < -0.16 { return [.bat, .bat, .bat] }',
     'if sunY < -0.6 { return [.bat, .bat, .bat] }'),
    # Scene tiles and the terrain lookup (tests from the Swift restructure).
    ('SceneTilesTests/testRectsMergeRunsAndRows', S + 'Render/SceneTiles.swift', 'rects[i].x1 == x1, rects[i].v == v', 'rects[i].v == v'),
    ('SceneTilesTests/testRectsCoverEveryTileOnce', S + 'Render/SceneTiles.swift', 'tx = x1 + 1', 'tx = x1 + 2'),
    ('SceneTilesTests/testVariantPicksTheCheapestThatCovers', S + 'Render/SceneTiles.swift', 'return need & 4 != 0 ? 6 : 5', 'return need & 4 != 0 ? 5 : 6'),
    # TerrainTests/testNothingBeforeTheFirstMap has none: before the first map both the origin sentinel and the empty
    # arrays put every point off the map, so removing either alone changes nothing.
    ('TerrainTests/testLooksUpTheCellUnderAPoint', S + 'World/Terrain.swift', 'return cz * s + cx', 'return cx * s + cz'),
    ('TerrainTests/testColumnReportsAnIslandOnlyWhereOneHangs', S + 'World/Terrain.swift', 'u < 4000 ? u : nil', 'u < 5000 ? u : nil'),
]

GPU = ('ShaderFunctionTests', 'ShaderLayoutTests')


def main():
    pick = re.compile(sys.argv[1] if len(sys.argv) > 1 else '')
    caught = missed = 0
    for test, path, old, new in MUTANTS:
        if not pick.search(test):
            continue
        full = os.path.join(ROOT, path)
        if subprocess.run(['git', 'diff', '--quiet', '--', path], cwd=ROOT).returncode:
            sys.exit(f'{path} has uncommitted changes')
        text = open(full).read()
        if text.count(old) != 1:
            print(f'SKIP   {test}: {path}: the original occurs {text.count(old)} times')
            continue
        open(full, 'w').write(text.replace(old, new))
        try:
            cmd = ['swift', 'test', '--filter', test]
            if test.startswith(GPU):
                cmd = ['tools/vrgpu', '--'] + cmd
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        finally:
            subprocess.run(['git', 'checkout', '--', path], cwd=ROOT, check=True)
        out = r.stdout + r.stderr
        # A compiler error carries line and column; a failed assertion only the line.
        built = not re.search(r'\.swift:\d+:\d+: error:', out)
        failed = r.returncode != 0 and built
        caught += failed
        missed += not failed
        why = next((l.split('error: ', 1)[1][:160] for l in out.splitlines() if 'XCTAssert' in l or 'Fatal error' in l), '')
        print(f'{"CAUGHT" if failed else "MISSED" if built else "NOBUILD"} {test}: {path}: {old.strip()[:50]!r} -> {new.strip()[:50]!r}  {why}',
              flush=True)
    print(f'{caught} caught, {missed} not caught')
    sys.exit(1 if missed else 0)


if __name__ == '__main__':
    main()
