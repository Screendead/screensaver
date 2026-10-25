@testable import StrataCore
import XCTest
import simd

// The autopilot over rolling synthetic land: the same seed flies the same path, and every flight stays smooth, clear
// of the ground and under the clouds.
final class FlightCameraTests: XCTestCase {
    func testASeedFliesTheSamePathEveryTime() {
        var a = Flight(seed: 303, phase: 0.1), b = Flight(seed: 303, phase: 0.1), c = Flight(seed: 304, phase: 0.1)
        a.fly(seconds: 40)
        b.fly(seconds: 40)
        c.fly(seconds: 40)
        XCTAssertEqual(a.poses, b.poses)
        XCTAssertNotEqual(a.poses.last, c.poses.last)
    }

    // Two and a half minutes each at dawn, midday and night. Bounds, per 1/30 s step: under 33 voxels a second, a
    // velocity change under 45 voxels a second squared, a turn under 0.18 rad a second, pitch moving under 0.12 rad a
    // second; and always 15 over the ground, under the ceiling, never rolling past 0.15 rad.
    func testLongFlightsAreSmoothAndKeepTheirClearance() {
        for (seed, phase) in [(UInt32(303), 0.02), (1, 0.25), (606, 0.75)] {
            var f = Flight(seed: seed, phase: phase)
            var lowest = Double.infinity
            f.fly(seconds: 150) { c, _ in lowest = min(lowest, c.position.y - c.h(c.position.x, c.position.z)) }
            let p = f.poses
            XCTAssertTrue(p.allSatisfy { $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite && $0.yaw.isFinite && $0.pitch.isFinite })
            XCTAssertGreaterThan(lowest, 15, "seed \(seed)")
            XCTAssertLessThanOrEqual(p.map(\.position.y).max()!, FlightCamera.ceiling, "seed \(seed)")
            XCTAssertLessThan(p.map { abs($0.roll) }.max()!, 0.15, "seed \(seed)")
            var step = 0.0, accel = 0.0, turn = 0.0, tilt = 0.0
            for i in 2..<p.count {
                step = max(step, simd_length(p[i].position - p[i - 1].position))
                accel = max(accel, simd_length(p[i].position - 2 * p[i - 1].position + p[i - 2].position))
                turn = max(turn, abs(FlightCamera.angleDiff(p[i].yaw, p[i - 1].yaw)))
                tilt = max(tilt, abs(p[i].pitch - p[i - 1].pitch))
            }
            XCTAssertLessThan(step, 33.0 / 30, "seed \(seed)")
            XCTAssertLessThan(accel, 45.0 / 900, "seed \(seed)")
            XCTAssertLessThan(turn, 0.18 / 30, "seed \(seed)")
            XCTAssertLessThan(tilt, 0.12 / 30, "seed \(seed)")
            // It goes somewhere.
            XCTAssertGreaterThan(simd_length(p.last!.position - p.first!.position), 1000, "seed \(seed)")
        }
    }
}
