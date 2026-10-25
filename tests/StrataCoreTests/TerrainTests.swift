@testable import StrataCore
import XCTest

final class TerrainTests: XCTestCase {
    // A map at (1000, -2000) whose cell i holds height i, floor i / 2, ground i / 4, and an island underside only on cell 1.
    private func terrain() -> Terrain {
        let t = Terrain(), s = MapLayout.size >> MapLayout.cpuLevel
        t.origin = SIMD2(1000, -2000)
        t.heights = (0..<(s * s)).map { Float($0) }
        t.floors = t.heights.map { $0 / 2 }
        t.ground = t.heights.map { $0 / 4 }
        t.under = (0..<(s * s)).map { $0 == 1 ? 300 : 4096 }
        return t
    }

    func testNothingBeforeTheFirstMap() {
        XCTAssertNil(Terrain().height(x: 0, z: 0))
        XCTAssertNil(Terrain().column(x: 0, z: 0))
    }

    // Cells are 4 voxels wide, rows along z, from the map's origin.
    func testLooksUpTheCellUnderAPoint() {
        let t = terrain(), s = Double(MapLayout.size >> MapLayout.cpuLevel)
        XCTAssertEqual(t.height(x: 1000, z: -2000), 0)
        XCTAssertEqual(t.height(x: 1003.9, z: -2000), 0)
        XCTAssertEqual(t.height(x: 1004, z: -2000), 1)
        XCTAssertEqual(t.height(x: 1000, z: -1996), s)
        XCTAssertEqual(t.solidFloor(x: 1004, z: -1996), (s + 1) / 2)
        XCTAssertNil(t.height(x: 999.9, z: -2000))
        XCTAssertNil(t.height(x: 1000 + Double(MapLayout.size), z: -2000))
    }

    // An underside of 4000 or more means no island.
    func testColumnReportsAnIslandOnlyWhereOneHangs() {
        let t = terrain()
        XCTAssertEqual(t.column(x: 1004, z: -2000)?.under, 300)
        XCTAssertEqual(t.column(x: 1004, z: -2000)?.ground, 0.25)
        XCTAssertNil(t.column(x: 1008, z: -2000)?.under)
    }
}
