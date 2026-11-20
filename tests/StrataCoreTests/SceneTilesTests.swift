@testable import StrataCore
import XCTest

final class SceneTilesTests: XCTestCase {
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
}
