@testable import StrataCore
import XCTest
import simd

final class SkyClockTests: XCTestCase {
    // Nothing the clock drives steps between frames over a whole day, the wrap into the next included: not the sun, the
    // burn-down schedule nor the sea of cloud. A night's beacons and rites hold from dusk past the wrap: the night index
    // steps once, at mid-morning.
    func testNothingStepsThroughADay() {
        var clock = SkyClock()
        clock.phase = 0.4
        var prev = clock, steps: [Double] = []
        for _ in 0..<(60 * 800) {
            clock.advance(1.0 / 60)
            XCTAssertLessThan(simd_length(clock.sunDirection - prev.sunDirection), 1e-3, "phase \(clock.phase)")
            XCTAssertLessThan(abs(clock.nightProgress - prev.nightProgress), 1e-3, "phase \(clock.phase)")
            XCTAssertLessThan(abs(clock.cloudSea - prev.cloudSea), 1e-3, "phase \(clock.phase)")
            if clock.nightIndex != prev.nightIndex { steps.append(clock.phase) }
            prev = clock
        }
        XCTAssertEqual(clock.day, 1)
        XCTAssertGreaterThan(clock.phase, 0.4)
        XCTAssertEqual(steps.count, 1)
        XCTAssertEqual(steps.first ?? 0, 0.25, accuracy: 1e-3)
    }
}
