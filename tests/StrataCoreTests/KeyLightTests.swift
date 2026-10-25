@testable import StrataCore
import XCTest
import simd

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

    // In the twilight between, the afterglow leads the blend toward the moon and casts none.
    func testOnlyTheSunOrTheMoonCastsShadows() {
        XCTAssertTrue(SkyClock.keyLight(SkyClock.direction(0.25)).1)
        XCTAssertTrue(SkyClock.keyLight(SkyClock.direction(0.75)).1)
        for y in [-0.03, -0.08, -0.12] {
            let c = (1 - y * y).squareRoot()
            XCTAssertFalse(SkyClock.keyLight(SIMD3(c * 0.6, y, c * 0.8)).1, "sun.y \(y)")
        }
    }
}
