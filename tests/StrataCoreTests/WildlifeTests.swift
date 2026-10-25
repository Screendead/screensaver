@testable import StrataCore
import XCTest
import simd

// The birds round an autopilot flight over synthetic land.
final class WildlifeTests: XCTestCase {
    private func birds(_ w: Wildlife) -> [SIMD3<Double>] { w.flocks.flatMap { $0.birds.map(\.pos) } }

    // All randomness comes from the seed in step order, so a snapshot reproduces.
    func testASeedRaisesTheSameBirds() {
        let a = Aviary(seed: 303, phase: 0.05), b = Aviary(seed: 303, phase: 0.05)
        a.fly(seconds: 45)
        b.fly(seconds: 45)
        XCTAssertFalse(birds(a.wildlife).isEmpty)
        XCTAssertEqual(birds(a.wildlife), birds(b.wildlife))
        XCTAssertEqual(a.wildlife.flocks.map(\.kind), b.wildlife.flocks.map(\.kind))
        let c = Aviary(seed: 304, phase: 0.05)
        c.fly(seconds: 45)
        XCTAssertNotEqual(birds(a.wildlife), birds(c.wildlife))
    }

    // A minute after dawn: flocks within the shader's caps, every bird above the ground under it, and the upload
    // writing within the bird buffer.
    func testFlocksStayInTheAirAndInTheBuffer() {
        let a = Aviary(seed: 303, phase: 0.05)
        var most = 0
        a.fly(seconds: 60) { w in
            XCTAssertLessThanOrEqual(w.flocks.count, Wildlife.maxFlocks)
            for f in w.flocks {
                for b in f.birds {
                    XCTAssertTrue(b.pos.x.isFinite && b.pos.y.isFinite && b.pos.z.isFinite)
                    XCTAssertGreaterThan(b.pos.y, w.ground(b.pos.x, b.pos.z, at: b.pos.y), "\(f.kind)")
                }
            }
            most = max(most, w.flocks.map(\.birds.count).reduce(0, +))
        }
        XCTAssertGreaterThan(most, 40)

        let guardBytes = 256
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: Wildlife.bufferLength + guardBytes, alignment: 16)
        defer { buffer.deallocate() }
        buffer.initializeMemory(as: UInt8.self, repeating: 0xA5, count: Wildlife.bufferLength + guardBytes)
        let flocks = a.wildlife.upload(into: buffer, origin: SIMD3(1000, 0, -1000))
        XCTAssertEqual(flocks, min(a.wildlife.flocks.filter { !$0.birds.isEmpty }.count, Wildlife.maxFlocks))
        let tail = UnsafeBufferPointer(start: (buffer + Wildlife.bufferLength).assumingMemoryBound(to: UInt8.self), count: guardBytes)
        XCTAssertTrue(tail.allSatisfy { $0 == 0xA5 })
    }

    // Deep in the night only bats fly.
    func testOnlyBatsFlyAtNight() {
        let a = Aviary(seed: 606, phase: 0.75)
        a.fly(seconds: 60) { _ in a.flight.clock.phase = 0.75 }
        XCTAssertFalse(a.wildlife.flocks.isEmpty)
        XCTAssertEqual(Set(a.wildlife.flocks.map(\.kind)), [.bat])
    }
}
