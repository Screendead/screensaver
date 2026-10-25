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

    // A minute after dawn over high hills: flocks within the shader's caps, every bird above the ground under it, and the upload
    // writing within the bird buffer.
    func testFlocksStayInTheAirAndInTheBuffer() {
        // Hills to 400 or so: the flocks' cruising height alone would not clear them.
        let a = Aviary(seed: 303, phase: 0.05, relief: 3), terrain = a.flight.world.terrain
        var most = 0
        a.fly(seconds: 60) { w in
            XCTAssertLessThanOrEqual(w.flocks.count, Wildlife.maxFlocks)
            for f in w.flocks {
                for b in f.birds {
                    XCTAssertTrue(b.pos.x.isFinite && b.pos.y.isFinite && b.pos.z.isFinite)
                    XCTAssertGreaterThan(b.pos.y, terrain.height(x: b.pos.x, z: b.pos.z)!, "\(f.kind)")
                }
            }
            most = max(most, w.flocks.map(\.birds.count).reduce(0, +))
        }
        XCTAssertGreaterThan(most, 40)
    }

    // The bird buffer as the shader walks it: per flock a sphere and its clusters, per cluster a sphere and its birds, each
    // bird map-local; every drawn bird inside its cluster's and flock's spheres, nothing written past the buffer. A
    // murmuration of 320 to 420 starlings fills it most.
    func testUploadWritesEveryBirdWhereTheShaderReadsIt() {
        let a = Aviary(seed: 303, phase: 0.05)
        a.fly(seconds: 20)
        a.wildlife.testMurmuration(camera: a.flight.camera, distance: 300)
        a.fly(seconds: 1)
        let origin = SIMD3<Double>(1024, 0, -2048), guardBytes = 256
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: Wildlife.bufferLength + guardBytes, alignment: 16)
        defer { buffer.deallocate() }
        buffer.initializeMemory(as: UInt8.self, repeating: 0xA5, count: Wildlife.bufferLength + guardBytes)
        let n = a.wildlife.upload(into: buffer, origin: origin)
        let tail = UnsafeBufferPointer(start: (buffer + Wildlife.bufferLength).assumingMemoryBound(to: UInt8.self), count: guardBytes)
        XCTAssertTrue(tail.allSatisfy { $0 == 0xA5 })

        let v = buffer.bindMemory(to: SIMD4<Float>.self, capacity: Wildlife.bufferLength / 16)
        let clusters = Wildlife.maxFlocks * 2, birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2
        let drawn = a.wildlife.drawList()
        XCTAssertEqual(n, drawn.count)
        XCTAssertGreaterThan(drawn.map { $0.flock.birds.count }.max() ?? 0, 300)
        func inside(_ p: SIMD4<Float>, _ s: SIMD4<Float>) -> Bool { simd_length(SIMD3(p.x, p.y, p.z) - SIMD3(s.x, s.y, s.z)) <= s.w + 1e-2 }
        for (i, d) in drawn.enumerated() {
            let fs = v[2 * i], fr = v[2 * i + 1]
            XCTAssertEqual(Int(fr.y), d.clusters.count)
            for (j, range) in d.clusters.enumerated() {
                let c = Int(fr.x) + j, cs = v[clusters + 2 * c], cr = v[clusters + 2 * c + 1]
                XCTAssertEqual(Int(cr.y), range.count)
                for (k, b) in range.enumerated() {
                    let bird = v[birds + 4 * (Int(cr.x) + k)], want = SIMD3<Float>(d.flock.birds[b].pos - origin)
                    XCTAssertEqual(SIMD3(bird.x, bird.y, bird.z), want)
                    XCTAssertTrue(inside(bird, cs) && inside(bird, fs))
                }
            }
        }
    }

    // Deep in the night only bats fly.
    func testOnlyBatsFlyAtNight() {
        let a = Aviary(seed: 606, phase: 0.75)
        a.fly(seconds: 60) { _ in a.flight.clock.phase = 0.75 }
        XCTAssertFalse(a.wildlife.flocks.isEmpty)
        XCTAssertEqual(Set(a.wildlife.flocks.map(\.kind)), [.bat])
    }
}
