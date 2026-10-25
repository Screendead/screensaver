import simd

extension SkyClock {
    // The key light as the shader's makeLight picks it, and whether it casts shadows.
    package static func keyLight(_ sun: SIMD3<Double>) -> (SIMD3<Double>, Bool) {
        let e = sun.y
        if e >= 0.05 { return (sun, true) }
        if e > -0.02 { return (simd_normalize(SIMD3(sun.x, 0.05, sun.z)), true) }
        let glow = simd_normalize(SIMD3(sun.x, 0.05, sun.z))
        let moon = simd_normalize(SIMD3(-sun.x, max(-sun.y * 0.7, 0.08), -sun.z))
        let t = min(max((e + 0.09) / (-0.16 + 0.09), 0), 1)
        let m = t * t * (3 - 2 * t)
        return (simd_normalize(glow + (moon - glow) * m), m > 0.5)
    }
}
