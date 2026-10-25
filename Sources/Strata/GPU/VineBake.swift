import simd

// Must match VineBake in Shaders/Life/Vines/Bake.metal.
struct VineBake {
    var a: SIMD4<Float>
    var b: SIMD4<Int32>
    var c: SIMD4<UInt32>
}
