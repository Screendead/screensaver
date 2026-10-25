import simd

// A giant tree as the camera and the birds see it, in world coordinates: a cylinder of `reach` round its axis, its crown
// top at `top`.
struct GiantInfo {
    static let reach = 210.0
    var pos: SIMD2<Double>
    var ground: Double
    var top: Double
    // 0 banyan, 1 kapok, 2 weeping (placeGiants).
    var kind = 0
    // The record's trunk height and hash (Vines bakes its strands from them).
    var yT = 0.0
    var h: UInt32 = 0
}
