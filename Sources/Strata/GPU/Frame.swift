import simd

// Must match `Frame` in Shaders/Common/Frame.metal.
struct Frame {
    var camPos = SIMD4<Float>()
    var camFwd = SIMD4<Float>()
    var camRight = SIMD4<Float>()
    var camUp = SIMD4<Float>()
    var sunDir = SIMD4<Float>()
    var resolution = SIMD4<Float>()
    var mapInfo = SIMD4<Int32>()
    var misc = SIMD4<Float>()
    var misc2 = SIMD4<Float>()
    var prevPos = SIMD4<Float>()
    var prevFwd = SIMD4<Float>()
    var prevRight = SIMD4<Float>()
    var prevUp = SIMD4<Float>()
    var jitter = SIMD4<Float>()
    var cloud = SIMD4<Float>()
    var lights = SIMD4<Float>()
    var refs = SIMD4<Int32>()
    var birds = SIMD4<Float>()
    // lod0...lod4 are `lod[5]` in the shader.
    var lod0 = SIMD4<Float>()
    var lod1 = SIMD4<Float>()
    var lod2 = SIMD4<Float>()
    var lod3 = SIMD4<Float>()
    var lod4 = SIMD4<Float>()
    var fade = SIMD4<Float>()
    var shadow = SIMD4<Float>()
    var slowCloud = SIMD4<Float>()
    var planet = SIMD4<Float>()
    var planetPx = SIMD4<Float>()
}
