@testable import StrataCore
import XCTest
import simd

final class SkyClockTests: XCTestCase {
    // The day opens on blue hour: the sun just under the horizon, rising.
    func testOpensBeforeSunrise() {
        var clock = SkyClock()
        XCTAssertLessThan(clock.sunDirection.y, 0)
        let before = clock.sunDirection.y
        clock.advance(1)
        XCTAssertGreaterThan(clock.sunDirection.y, before)
    }
}

final class KeyLightTests: XCTestCase {
    // No step where makeLight switches branches: onto the sun at the afterglow's height (0.05) and onto the afterglow at
    // -0.02 (a step there flashes every sunlit face).
    func testKeyLightHasNoStepAtItsHandovers() {
        func sun(_ y: Double) -> SIMD3<Double> { let c = (1 - y * y).squareRoot(); return SIMD3(c * 0.6, y, c * 0.8) }
        for y in [0.05, -0.02] {
            let above = SkyClock.keyLight(sun(y + 1e-9)).0, below = SkyClock.keyLight(sun(y - 1e-9)).0
            XCTAssertLessThan(simd_length(above - below), 1e-6, "at \(y)")
        }
    }

    func testOnlyTheSunOrTheMoonCastsShadows() {
        XCTAssertTrue(SkyClock.keyLight(SkyClock.direction(0.25)).1)
        XCTAssertTrue(SkyClock.keyLight(SkyClock.direction(0.75)).1)
    }
}
