@testable import StrataCore
import XCTest

final class TerrainTests: XCTestCase {
    // The map's CPU copy as the GPU reads it back: cells 4 voxels wide, rows along z, from the map's origin; an underside
    // of 4000 or more means no island. Here a map at (1000, -2000) whose cell i holds height i, floor i / 2, ground i / 4,
    // and an island underside only on cell 1.
    func testLooksUpTheColumnUnderAPoint() {
        let t = Terrain(), n = MapLayout.size >> MapLayout.cpuLevel, s = Double(n)
        t.origin = SIMD2(1000, -2000)
        t.heights = (0..<(n * n)).map { Float($0) }
        t.floors = t.heights.map { $0 / 2 }
        t.ground = t.heights.map { $0 / 4 }
        t.under = (0..<(n * n)).map { $0 == 1 ? 300 : 4096 }
        XCTAssertEqual(t.height(x: 1000, z: -2000), 0)
        XCTAssertEqual(t.height(x: 1003.9, z: -2000), 0)
        XCTAssertEqual(t.height(x: 1004, z: -2000), 1)
        XCTAssertEqual(t.height(x: 1000, z: -1996), s)
        XCTAssertEqual(t.solidFloor(x: 1004, z: -1996), (s + 1) / 2)
        XCTAssertNil(t.height(x: 999.9, z: -2000))
        XCTAssertNil(t.height(x: 1000 + Double(MapLayout.size), z: -2000))
        XCTAssertEqual(t.column(x: 1004, z: -2000)?.under, 300)
        XCTAssertEqual(t.column(x: 1004, z: -2000)?.ground, 0.25)
        XCTAssertNil(t.column(x: 1008, z: -2000)?.under)
    }
}
