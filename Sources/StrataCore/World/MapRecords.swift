import simd

// One giant-table entry (GiantRec in Shaders/Life/Giants/Place.metal): g <= 0 marks a cell without a giant.
struct GiantRec {
    package var cell = SIMD2<Int32>()
    var c = SIMD2<Float>()
    var g: Float = 0
    package var yT: Float = 0
    package var h: UInt32 = 0
    package var kind: UInt32 = 0
}

// One cave-table entry (CaveRec in Shaders/World/Caves/Shape.metal), kept after the giant table in its buffer.
package struct CaveRec {
    package var cell = SIMD2<Int32>()
    package var m = SIMD2<Float>()
    package var u = SIMD2<Float>()
    package var yF: Float = 0
    package var w: Float = 0
    package var l: Float = 0
    package var hc: Float = 0
    package var h: UInt32 = 0
    package var flags: UInt32 = 0
}
