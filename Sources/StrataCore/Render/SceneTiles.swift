// The scene pass is drawn in tiles, each with the cheapest sceneFragment variant its pixels need, indexed as
// GPU.sceneVariants: 0 none, 1 the band test, 2 island surfaces, 3 both, 4 the islands' shade, 5 close by an island,
// 6 with its mist.
package enum SceneTiles {
    // need, per tile: 1 an island's surfaces may show, 2 its shade or sky occlusion, 8 close by one within
    // isleShadeReach, 4 its mist. isles: the islands' shade variants are on; shapes: their surfaces; bands: the band test.
    package static func variant(need: UInt8, isles: Bool, shapes: Bool, bands: Bool) -> Int {
        let shapeT = shapes && need & 1 != 0
        if isles && (need & 8 != 0 || (need & 2 != 0 && shapeT)) { return need & 4 != 0 ? 6 : 5 }
        if isles && need & 2 != 0 { return 4 }
        return (bands ? 1 : 0) + (shapeT ? 2 : 0)
    }

    package typealias Rect = (v: Int, x0: Int, x1: Int, y0: Int, y1: Int)

    // Rectangles of tiles of one variant (nx x ny tiles, rows first), each drawn as the whole frame's triangle scissored
    // to it: a pair of triangles per tile would shade the 2x2 quads along every diagonal twice.
    package static func rects(_ variant: [Int], nx: Int, ny: Int) -> [Rect] {
        var rects: [Rect] = []
        var open: [Int: Int] = [:]
        for ty in 0..<ny {
            var next: [Int: Int] = [:]
            var tx = 0
            while tx < nx {
                let v = variant[ty * nx + tx]
                var x1 = tx
                while x1 + 1 < nx && variant[ty * nx + x1 + 1] == v { x1 += 1 }
                if let i = open[tx], rects[i].x1 == x1, rects[i].v == v {
                    rects[i].y1 = ty
                    next[tx] = i
                } else {
                    rects.append((v, tx, x1, ty, ty))
                    next[tx] = rects.count - 1
                }
                tx = x1 + 1
            }
            open = next
        }
        return rects
    }
}
