@testable import StrataCore
import XCTest
import simd

final class IslandsTests: XCTestCase {
    private func isle(_ x: Double, _ z: Double, radius: Double = 80, hash: UInt32 = 1) -> IslandInfo {
        IslandInfo(pos: SIMD2(x, z), radius: radius, reach: 1.45 * radius + 8, top: 400, keel: 300, base: 60, hash: hash, massif: false)
    }

    // A cleared table holds no island; a standing slot becomes one, a massif past the plain window, with its fall only
    // where it has one.
    func testParsesStandingSlots() {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Islands.slots * Islands.recordSize, alignment: 16)
        defer { raw.deallocate() }
        Islands.clear(raw)
        XCTAssertTrue(Islands.parse(raw).isEmpty)
        let slots = raw.bindMemory(to: IsleRec.self, capacity: Islands.slots)
        slots[7] = IsleRec(cell: SIMD2(1, 1), c: SIMD2(100, 200), r: 50, a: 380, d: 0, b: 40, k: 290, h: 0)
        slots[Islands.win * Islands.win + 2] = IsleRec(cell: SIMD2(0, 0), c: SIMD2(-500, 0), r: 200, a: 600, b: 30, k: 420, h: 5,
                                                         f: SIMD4(-450, 10, 560, 25))
        let list = Islands.parse(raw)
        XCTAssertEqual(list.count, 2)
        XCTAssertEqual(list[0].pos, SIMD2(100, 200))
        XCTAssertFalse(list[0].massif)
        XCTAssertNil(list[0].fall)
        XCTAssertEqual(list[0].keel, 290)
        XCTAssertEqual(list[0].base, 40)
        XCTAssertTrue(list[1].massif)
        XCTAssertEqual(list[1].fall, SIMD4(-450, 10, 560, 25))
        // Nothing of an island lies past its reach, nor above its top.
        for s in list { XCTAssertGreaterThanOrEqual(s.reach, 1.45 * s.radius + 8); XCTAssertGreaterThan(s.top, s.keel) }
    }

    // The shading table holds the 16 islands nearest the point, nearest first, each ending with its occlusion reach squared.
    func testShadingTableHoldsTheNearestFirst() {
        let list = (0..<40).map { isle(Double(($0 * 37) % 40) * 300, 0, hash: UInt32($0)) }
        let t = Islands.table(list, near: SIMD2(0, 0), within: 20000)
        XCTAssertEqual(t.bytes.count, Islands.tableBytes)
        t.bytes.withUnsafeBytes { b in
            XCTAssertEqual(b.load(as: Int32.self), Int32(Islands.tableSize))
            var last = -1.0
            for i in 0..<Islands.tableSize {
                let o = 16 + i * Islands.entryBytes
                let a = b.load(fromByteOffset: o, as: SIMD4<Float>.self)
                XCTAssertGreaterThan(Double(a.x), last)
                last = Double(a.x)
                let occ = b.load(fromByteOffset: o + 64 + 16 * Islands.mistPuffs, as: SIMD4<Float>.self).x
                XCTAssertEqual(Double(occ), pow(Islands.occlusionReach(list.first { $0.pos.x == Double(a.x) }!), 2), accuracy: 1)
            }
            XCTAssertEqual(last, 15 * 300)
        }
    }

    // Past its occlusion reach an island hides under a five-hundredth of the sky from anywhere under it, and just inside
    // more; unless that lies past three times its reach, where the reach stops.
    func testOcclusionReachIsWhereTheSkyComesBack() {
        var bisected = 0, capped = 0
        for radius in [30.0, 80.0, 160.0, 260.0] {
            for top in [60.0, 150.0, 400.0] {
                var s = isle(0, 0, radius: radius)
                (s.top, s.keel) = (top, top - 40)
                let r = Islands.occlusionReach(s), hMax = max(0.5 * (s.keel + s.top) - Sight.sea, 4)
                func worst(_ d: Double) -> Double { (0...32).map { Islands.skyOcc(s.radius, 4 + (hMax - 4) * Double($0) / 32, d) }.max()! }
                if worst(3 * s.reach) >= 0.002 {
                    XCTAssertEqual(r, 3 * s.reach)
                    capped += 1
                } else {
                    XCTAssertLessThan(worst(r), 0.002, "radius \(radius) top \(top)")
                    XCTAssertGreaterThan(worst(r * 0.98), 0.002, "radius \(radius) top \(top)")
                    bisected += 1
                }
            }
        }
        XCTAssertGreaterThan(bisected, 2)
        XCTAssertGreaterThan(capped, 0)
    }

    // The view test passes an island ahead or round the eye and drops one behind it.
    func testBoundingSpheresSeeAnIslandOnlyWhereTheViewCanReach() {
        let s = isle(0, 2000)
        let key = simd_normalize(SIMD3(0.3, 0.6, 0.2))
        let spheres = Islands.spheres([s], eye: SIMD3(0, 100, 0), within: 5000, key: key)
        XCTAssertTrue(spheres.contains { simd_length($0.c - SIMD3(0, s.top, 2000)) <= $0.r })
        XCTAssertTrue(spheres.contains { simd_length($0.c - SIMD3(0, s.base, 2000)) <= $0.r })
        XCTAssertTrue(Islands.inView(spheres, eye: SIMD3(0, 100, 0), fwd: SIMD3(0, 0, 1), half: 0.6, margin: 0))
        XCTAssertFalse(Islands.inView(spheres, eye: SIMD3(0, 100, 0), fwd: SIMD3(0, 0, -1), half: 0.6, margin: 0))
        XCTAssertTrue(Islands.spheres([s], eye: SIMD3(0, 100, -9000), within: 5000, key: key).isEmpty)
    }
}
