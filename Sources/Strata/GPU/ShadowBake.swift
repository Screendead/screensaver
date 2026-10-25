import simd

// Must match ShadowBake in Shaders/Light/Shadows.metal.
struct ShadowBake {
    var key: SIMD4<Float>
    var cam: SIMD4<Float>
    var rows: SIMD4<Int32>
}
