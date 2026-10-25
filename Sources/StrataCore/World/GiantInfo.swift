import simd

// A giant tree as the camera and the birds see it, in world coordinates: a cylinder of `reach` round its axis, its crown
// top at `top`.
package struct GiantInfo {
    package static let reach = 210.0
    package var pos: SIMD2<Double>
    package var ground: Double
    package var top: Double
    // 0 banyan, 1 kapok, 2 weeping (placeGiants).
    package var kind = 0
    // The record's trunk height and hash (Vines bakes its strands from them).
    package var yT = 0.0
    package var h: UInt32 = 0
}
