import simd

// One giant-table entry (Shaders/Saver.metal GiantRec): g <= 0 marks a cell without a giant.
struct GiantRec {
    var cell = SIMD2<Int32>()
    var c = SIMD2<Float>()
    var g: Float = 0
    var yT: Float = 0
    var h: UInt32 = 0
    var kind: UInt32 = 0
}

// One cave-table entry (Shaders/Saver.metal CaveRec), kept after the giant table in its buffer.
struct CaveRec {
    var cell = SIMD2<Int32>()
    var m = SIMD2<Float>()
    var u = SIMD2<Float>()
    var yF: Float = 0
    var w: Float = 0
    var l: Float = 0
    var hc: Float = 0
    var h: UInt32 = 0
    var flags: UInt32 = 0
}
