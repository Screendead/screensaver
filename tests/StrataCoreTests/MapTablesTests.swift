@testable import StrataCore
import XCTest
import simd

// The tables placed with each map, decoded from the bytes the shader writes.
final class MapTablesTests: XCTestCase {
    // A site table holding one hamlet in slot 5 at world (1000, -2000) with two lamps: a window lamp at the door whose
    // chimney smokes 3 columns east and 2 south, rising 6; and a beacon with no smoke.
    func testSitesDecodeWithTheirLamps() {
        var bytes = [UInt8](repeating: 0, count: MapLayout.siteGrid * MapLayout.siteGrid * MapLayout.siteStride)
        bytes.withUnsafeMutableBytes { b in
            let o = 5 * MapLayout.siteStride
            b.storeBytes(of: SIMD4<Int32>(1000, -2000, Int32(SiteInfo.Kind.hamlet.rawValue), -7), toByteOffset: o, as: SIMD4<Int32>.self)
            b.storeBytes(of: SIMD4<Float>(40, 0, 3, 2), toByteOffset: o + 16, as: SIMD4<Float>.self)
            let code = (3 + 16) + 32 * (2 + 16)
            b.storeBytes(of: SIMD4<Float>(4, 30, -5, Float(3 + 16 * 6 + 1024 * code)), toByteOffset: o + 32, as: SIMD4<Float>.self)
            b.storeBytes(of: SIMD4<Float>(-8, 35, 1, 5), toByteOffset: o + 48, as: SIMD4<Float>.self)
        }
        let sites = bytes.withUnsafeBytes { MapTables.sites(in: $0.baseAddress!) }
        XCTAssertEqual(sites.count, 1)
        let s = sites[0]
        XCTAssertEqual(s.index, 5)
        XCTAssertEqual(s.id, UInt32(bitPattern: -7))
        XCTAssertEqual(s.kind, .hamlet)
        XCTAssertEqual(s.pos, SIMD2(1000.5, -1999.5))
        XCTAssertEqual(s.radius, 40)
        XCTAssertEqual(s.lamps.count, 2)
        XCTAssertEqual(s.lamps[0].pos, SIMD3(1004, 30, -2005))
        XCTAssertEqual(s.lamps[0].kind, 3)
        XCTAssertEqual(s.lamps[0].rise, 6)
        XCTAssertEqual(s.lamps[0].smoke, SIMD3(1007, 36, -2003))
        XCTAssertEqual(s.lamps[1].kind, 5)
        XCTAssertEqual(s.lamps[1].index, 5 * MapLayout.maxLamps + 1)
        XCTAssertEqual(s.lamps[1].smoke, s.lamps[1].pos)
    }

    // A cleared table holds no giant or cave; a slot with ground under it is a giant, a cave slot flagged on is a cave.
    func testGiantsAndCavesDecodeFromTheirSharedBuffer() {
        let raw = UnsafeMutableRawPointer.allocate(byteCount: MapTables.giantTableBytes, alignment: 16)
        defer { raw.deallocate() }
        MapTables.clearGiantsAndCaves(raw)
        XCTAssertTrue(MapTables.giants(in: raw).isEmpty)
        XCTAssertTrue(MapTables.caves(in: raw).isEmpty)

        let giants = raw.bindMemory(to: GiantRec.self, capacity: MapLayout.giantWin * MapLayout.giantWin)
        giants[3] = GiantRec(cell: SIMD2(1, 2), c: SIMD2(500, 700), g: 40, yT: 120, h: 9, kind: 2)
        let caves = (raw + MapLayout.caveOffset).bindMemory(to: CaveRec.self, capacity: MapLayout.caveWin * MapLayout.caveWin)
        caves[MapLayout.caveWin * MapLayout.caveWin - 1] = CaveRec(cell: SIMD2(4, 5), m: SIMD2(10, 20), flags: 1 | 2)
        caves[0] = CaveRec(cell: SIMD2(4, 6), m: SIMD2(30, 40), flags: 2)

        let g = MapTables.giants(in: raw)
        XCTAssertEqual(g.count, 1)
        XCTAssertEqual(g.first?.pos, SIMD2(500, 700))
        XCTAssertEqual(g.first?.ground, 40)
        XCTAssertEqual(g.first?.kind, 2)
        // Above the crown's highest clumps (yT + 84).
        XCTAssertGreaterThanOrEqual(g.first!.top, 40 + 120 + 84)
        let c = MapTables.caves(in: raw)
        XCTAssertEqual(c.map(\.m), [SIMD2(10, 20)])
    }

    // Slots without a tower have height 0; the packed word holds member, moisture and temperature.
    func testTowersDecodeTheirPackedFields() {
        var bytes = [UInt8](repeating: 0, count: 3 * TowerInfo.recordSize)
        bytes.withUnsafeMutableBytes { b in
            let o = 2 * TowerInfo.recordSize
            b.storeBytes(of: SIMD4<Float>(9000, -18000, 12, 50), toByteOffset: o, as: SIMD4<Float>.self)
            b.storeBytes(of: SIMD4<Float>(180, 0, 0, 0), toByteOffset: o + 16, as: SIMD4<Float>.self)
            b.storeBytes(of: SIMD4<UInt32>(0, 2, 1, 3 | 51 << 8 | 204 << 16), toByteOffset: o + 48, as: SIMD4<UInt32>.self)
        }
        let towers = bytes.withUnsafeBytes { TowerInfo.decode($0.baseAddress!, count: 3) }
        XCTAssertEqual(towers.count, 1)
        let t = towers[0]
        XCTAssertEqual(t.pos, SIMD2(9000, -18000))
        XCTAssertEqual(t.top, 230)
        XCTAssertEqual(t.kind, 2)
        XCTAssertTrue(t.lit)
        XCTAssertEqual(t.member, 3)
        XCTAssertEqual(t.moist, 0.2, accuracy: 1e-9)
        XCTAssertEqual(t.temp, 0.8, accuracy: 1e-9)
    }
}
