@testable import StrataCore
import XCTest
import simd

final class IslandsTests: XCTestCase {
    private func isle(_ x: Double, _ z: Double, hash: UInt32) -> IslandInfo {
        IslandInfo(pos: SIMD2(x, z), radius: 80, reach: 1.45 * 80 + 8, top: 400, keel: 300, base: 60, hash: hash, massif: false)
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
}
