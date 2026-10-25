@testable import StrataCore
import XCTest
import simd

final class SkyClockTests: XCTestCase {
    // Seconds of simulation from phase `from` until the day count steps, in steps of dt; and how many of them had the
    // sun up.
    private func day(from phase: Double, dt: Double) -> (seconds: Double, sunUp: Double) {
        var clock = SkyClock()
        clock.phase = phase
        var t = 0.0, up = 0.0
        while clock.day == 0 {
            if clock.sunDirection.y > 0 { up += dt }
            clock.advance(dt)
            t += dt
        }
        return (t, up)
    }

    // The day opens on blue hour: the sun just under the horizon, rising.
    func testOpensBeforeSunrise() {
        var clock = SkyClock()
        XCTAssertLessThan(clock.sunDirection.y, 0)
        let before = clock.sunDirection.y
        clock.advance(1)
        XCTAssertGreaterThan(clock.sunDirection.y, before)
    }

    // The time-warp makes a 12-minute nominal day about 713 s, about 460 s of it with the sun up, whatever the frame rate.
    func testADayLastsAbout713SecondsWithTheSunUpFor460() {
        let at60 = day(from: 0, dt: 1.0 / 60), at30 = day(from: 0, dt: 1.0 / 30)
        XCTAssertEqual(at60.seconds, 712.6, accuracy: 1)
        XCTAssertEqual(at60.sunUp, 460.5, accuracy: 1)
        XCTAssertEqual(at30.seconds, at60.seconds, accuracy: 1)
    }

    // Peak day is held to a long-afternoon sun; night keeps the same depth below.
    func testSunPeaksAt049AndSinksAsFar() {
        let e = (0..<10000).map { SkyClock.elevation(Double($0) / 10000) }
        XCTAssertEqual(e.max()!, 0.49, accuracy: 1e-4)
        XCTAssertEqual(e.min()!, -0.49, accuracy: 1e-4)
    }

    // A full turn of azimuth a day, with no jump where the phase wraps from 1 to 0.
    func testSunTurnsOnceADayWithoutAJumpAtTheWrap() {
        var turned = 0.0
        var last = SkyClock.direction(0)
        for i in 1...1000 {
            let d = SkyClock.direction(Double(i) / 1000)
            turned += atan2(simd_cross(SIMD3(last.x, 0, last.z), SIMD3(d.x, 0, d.z)).y, last.x * d.x + last.z * d.z)
            last = d
        }
        XCTAssertEqual(abs(turned), 2 * .pi, accuracy: 1e-9)
        XCTAssertLessThan(simd_length(SkyClock.direction(1 - 1e-9) - SkyClock.direction(1e-9)), 1e-6)

        var clock = SkyClock()
        clock.phase = 0.99999
        let before = clock.sunDirection
        clock.advance(1.0 / 60)
        XCTAssertEqual(clock.day, 1)
        XCTAssertLessThan(simd_length(clock.sunDirection - before), 1e-3)
    }

    // A night's beacons and rites hold from dusk past the dawn wrap: the index steps once a day, at mid-morning.
    func testNightIndexStepsAtMidMorningNotAtTheWrap() {
        var clock = SkyClock()
        clock.phase = 0.4
        var steps: [(Double, Int)] = []
        var last = clock.nightIndex
        for _ in 0..<(60 * 800) {
            clock.advance(1.0 / 60)
            if clock.nightIndex != last { steps.append((clock.phase, clock.day)); last = clock.nightIndex }
        }
        XCTAssertEqual(steps.count, 1)
        XCTAssertEqual(steps.first?.0 ?? 0, 0.25, accuracy: 1e-3)
        XCTAssertEqual(steps.first?.1, 1)
    }

    // The burn-down schedule and the sea of cloud never step, wrap included; night progress runs 0 at dusk to 1 by dawn.
    func testNightProgressAndCloudSeaAreContinuous() {
        func at(_ p: Double) -> SkyClock { var c = SkyClock(); c.phase = p; return c }
        var prev = at(0)
        for i in 1...100_000 {
            let c = at(Double(i) / 100_000)
            XCTAssertLessThan(abs(c.nightProgress - prev.nightProgress), 1e-3, "phase \(c.phase)")
            XCTAssertLessThan(abs(c.cloudSea - prev.cloudSea), 1e-3, "phase \(c.phase)")
            prev = c
        }
        XCTAssertEqual(at(0.52).nightProgress, 0)
        XCTAssertEqual(at(0.99).nightProgress, 1)
        XCTAssertEqual(at(0.21).nightProgress, 0)
        XCTAssertEqual(at(0.25).cloudSea, 0)
        XCTAssertGreaterThan((0..<1000).map { at(Double($0) / 1000).cloudSea }.max()!, 0.5)
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
