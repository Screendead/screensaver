@testable import StrataCore
import XCTest
import simd

// The autopilot and the birds round it, over synthetic land.
final class WildlifeTests: XCTestCase {
    private func birds(_ w: Wildlife) -> [SIMD3<Double>] { w.flocks.flatMap { $0.birds.map(\.pos) } }

    // All randomness comes from the seed in step order, so a snapshot reproduces: the same seed flies the same path and
    // raises the same birds, through two of its shot choices (12 s apart) and a bird spawn each second or so.
    func testASeedFliesTheSamePathAndRaisesTheSameBirds() {
        let a = Flight(seed: 303, phase: 0.05), b = Flight(seed: 303, phase: 0.05)
        a.fly(seconds: 13)
        b.fly(seconds: 13)
        XCTAssertFalse(birds(a.wildlife).isEmpty)
        XCTAssertEqual(a.poses, b.poses)
        XCTAssertEqual(birds(a.wildlife), birds(b.wildlife))
        XCTAssertEqual(a.wildlife.flocks.map(\.kind), b.wildlife.flocks.map(\.kind))
    }

    // The bird buffer as the shader walks it: per flock a sphere and its clusters, per cluster a sphere and its birds, each
    // bird map-local; every drawn bird inside its cluster's and flock's spheres, nothing written past the buffer. A
    // murmuration of 320 to 420 starlings fills it most.
    func testUploadWritesEveryBirdWhereTheShaderReadsIt() {
        let a = Flight(seed: 303, phase: 0.05)
        a.fly(seconds: 3)
        a.wildlife.testMurmuration(camera: a.camera, distance: 300)
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
}
