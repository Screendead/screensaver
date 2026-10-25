import simd

// One table slot (IsleRec in Shaders/World/Islands/Shape.metal): r > 0 an island standing.
struct IsleRec {
    package var cell = SIMD2<Int32>(Int32.min, Int32.min)
    var c = SIMD2<Float>()
    var r: Float = 0
    var a: Float = 0
    var d: Float = 0
    var b: Float = 0
    var k: Float = 0
    package var h: UInt32 = 0
    var fa = SIMD2<Float>()
    var f = SIMD4<Float>()
}

// A floating island as the camera, the birds and the shading see it, in world coordinates: nothing of it lies further
// than `reach` from `pos`; its top stands about `top` at the centre, its underside nowhere under `keel`, over ground no
// higher than `base`.
package struct IslandInfo {
    package var pos: SIMD2<Double>
    package var radius: Double
    package var reach: Double
    package var top: Double
    package var keel: Double
    package var base: Double
    var hash: UInt32
    package var massif: Bool
    // Its waterfall: the lip's centre (x, z), the water's top there and the fall's foot.
    package var fall: SIMD4<Double>?

    // Room under the keel for the flight to pass beneath.
    var passable: Bool { keel - base >= 90 }
}
