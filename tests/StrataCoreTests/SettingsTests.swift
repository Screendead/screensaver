@testable import StrataCore
import XCTest

final class SettingsTests: XCTestCase {
    private func parse(_ args: String) -> Settings { Settings.parse(["voxelreverie"] + args.split(separator: " ").map(String.init)) }

    // The snapshot check's command line, as CLAUDE.md gives it.
    func testParsesASnapshotCommand() {
        let s = parse("--screenshot out.png --size 3456x2234 --scale 1 --seed 1 --times 10,60 --phase -0.03 --pitch 0.2 --debug 3 --battery")
        XCTAssertEqual(s.screenshotPath, "out.png")
        XCTAssertEqual(s.screenshotSize, CGSize(width: 3456, height: 2234))
        XCTAssertEqual(s.renderScale, 1)
        XCTAssertEqual(s.seed, 1)
        XCTAssertEqual(s.screenshotTimes, [10, 60])
        XCTAssertEqual(s.phase, -0.03)
        XCTAssertEqual(s.pitch, 0.2)
        XCTAssertEqual(s.debugMode, 3)
        XCTAssertTrue(s.forceBattery)
    }

    // Unset, the overrides stay nil so the saver's own choices hold.
    func testDefaultsLeaveTheSaversChoices() {
        let s = parse("")
        XCTAssertNil(s.screenshotPath)
        XCTAssertNil(s.phase)
        XCTAssertNil(s.pitch)
        XCTAssertNil(s.renderScale)
        XCTAssertFalse(s.forceBattery)
        XCTAssertEqual(s.screenshotTimes, [8])
        XCTAssertEqual(s.timeScale, 1)
    }

    // A malformed size keeps the default, an unknown flag is skipped, and a flag missing its value at the end parses.
    func testToleratesBadInput() {
        let s = parse("--size 864 --frobnicate --nodither --seed")
        XCTAssertEqual(s.screenshotSize, CGSize(width: 1728, height: 1117))
        XCTAssertTrue(s.noDither)
        XCTAssertEqual(s.seed, 0)
    }

    // The menu's biome by name (any case) or number; anything else is the whole world.
    func testBiomeByNameOrNumber() {
        XCTAssertEqual(parse("--biome jungle").biome, 8)
        XCTAssertEqual(parse("--biome 3").biome, 3)
        XCTAssertEqual(parse("--biome Atlantis").biome, 0)
    }

    // A pinned biome rides in the seed's top byte, which random seeds never reach, and reads back out.
    func testBiomeRoundTripsThroughTheSeed() {
        for b in 1...Biome.names.count {
            let seed = Biome.seed(0x1234_5678, biome: b)
            XCTAssertGreaterThan(seed, Biome.maxSeed)
            XCTAssertEqual(Biome.pinned(seed), b)
        }
        XCTAssertEqual(Biome.seed(0x1234_5678, biome: 0), 0x1234_5678)
        XCTAssertEqual(Biome.pinned(Biome.maxSeed), 0)
    }
}
