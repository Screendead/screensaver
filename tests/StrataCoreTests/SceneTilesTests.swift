@testable import StrataCore
import XCTest

final class SceneTilesTests: XCTestCase {
    // Runs of one variant on a row merge, and a run matching the one above it extends that rectangle down.
    func testRectsMergeRunsAndRows() {
        let v = [0, 0, 1,
                 0, 0, 1,
                 2, 0, 1]
        let rects = SceneTiles.rects(v, nx: 3, ny: 3).map { [$0.v, $0.x0, $0.x1, $0.y0, $0.y1] }
        XCTAssertEqual(rects, [[0, 0, 1, 0, 1], [1, 2, 2, 0, 2], [2, 0, 0, 2, 2], [0, 1, 1, 2, 2]])
    }

    // Every tile lies in exactly one rectangle, of its own variant.
    func testRectsCoverEveryTileOnce() {
        let nx = 7, ny = 5
        let v = (0..<(nx * ny)).map { ($0 * 7 + $0 / 3) % 4 }
        var hits = [Int](repeating: 0, count: nx * ny)
        for r in SceneTiles.rects(v, nx: nx, ny: ny) {
            for y in r.y0...r.y1 { for x in r.x0...r.x1 { XCTAssertEqual(v[y * nx + x], r.v); hits[y * nx + x] += 1 } }
        }
        XCTAssertEqual(hits, [Int](repeating: 1, count: nx * ny))
    }

    func testVariantPicksTheCheapestThatCovers() {
        XCTAssertEqual(SceneTiles.variant(need: 0, isles: false, shapes: false, bands: false), 0)
        XCTAssertEqual(SceneTiles.variant(need: 0, isles: false, shapes: false, bands: true), 1)
        XCTAssertEqual(SceneTiles.variant(need: 1, isles: false, shapes: true, bands: true), 3)
        XCTAssertEqual(SceneTiles.variant(need: 1, isles: false, shapes: false, bands: false), 0)
        XCTAssertEqual(SceneTiles.variant(need: 2, isles: true, shapes: false, bands: true), 4)
        XCTAssertEqual(SceneTiles.variant(need: 3, isles: true, shapes: true, bands: false), 5)
        XCTAssertEqual(SceneTiles.variant(need: 8, isles: true, shapes: false, bands: false), 5)
        XCTAssertEqual(SceneTiles.variant(need: 12, isles: true, shapes: false, bands: false), 6)
        XCTAssertEqual(SceneTiles.variant(need: 12, isles: false, shapes: false, bands: false), 0)
    }
}
