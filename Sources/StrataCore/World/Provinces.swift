import Foundation

// CPU port of the forest-province fields (provinces, fbm2, perlin2 and their hashes in Shaders/Saver.metal), in Double:
// close to the shader's floats, which is all the renderer's choices from it need.
package enum Provinces {
    static func pcg(_ v: UInt32) -> UInt32 {
        let state = v &* 747796405 &+ 2891336453
        let word = ((state >> ((state >> 28) &+ 4)) ^ state) &* 277803737
        return (word >> 22) ^ word
    }
    static func hash2i(_ x: Int32, _ y: Int32, _ s: UInt32) -> UInt32 {
        pcg(UInt32(bitPattern: x) &* 1597334677 ^ pcg(UInt32(bitPattern: y) &* 3812015801 ^ s))
    }
    static func hashf(_ h: UInt32) -> Double { Double(h & 0xffffff) / 16777215.0 }

    static func perlin2(_ x: Double, _ y: Double, _ s: UInt32) -> Double {
        let ix = floor(x), iy = floor(y), fx = x - ix, fy = y - iy
        let cx = Int32(ix), cy = Int32(iy)
        func g(_ ox: Int32, _ oy: Int32) -> Double {
            let a = hashf(hash2i(cx &+ ox, cy &+ oy, s)) * 6.2831853
            return cos(a) * (fx - Double(ox)) + sin(a) * (fy - Double(oy))
        }
        let ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10), uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        let a = g(0, 0), b = g(1, 0), c = g(0, 1), d = g(1, 1)
        return ((a + (b - a) * ux) * (1 - uy) + (c + (d - c) * ux) * uy) * 1.414
    }

    static func fbm2(_ x: Double, _ y: Double, _ oct: Int, _ s: UInt32) -> Double {
        var sum = 0.0, amp = 0.5, norm = 0.0, px = x, py = y
        for i in 0..<oct {
            sum += amp * perlin2(px, py, s &+ UInt32(i) &* 1013)
            norm += amp
            (px, py) = ((0.8 * px - 0.6 * py) * 2 + 17.3, (0.6 * px + 0.8 * py) * 2 + 17.3)
            amp *= 0.5
        }
        return sum / norm
    }

    static func seedOffset(_ seed: UInt32) -> (Double, Double) {
        (hashf(pcg(seed)) * 50000, hashf(pcg(seed ^ 0x9e3779b9)) * 50000)
    }

    // (jungle, old growth) at world xz.
    package static func at(_ x: Double, _ z: Double, seed: UInt32) -> (Double, Double) {
        let b = Biome.pinned(seed)
        if b != 0 { return (b == 8 ? 1 : 0, b == 9 ? 1 : 0) }
        let o = seedOffset(seed)
        let px = x + o.0, pz = z + o.1
        let j = smoothstep(0.24, 0.32, fbm2(px / 6000 + 300.7, pz / 6000 + 300.7, 2, seed &+ 300))
        let og = smoothstep(0.30, 0.38, fbm2(px / 6000 + 411.3, pz / 6000 + 411.3, 2, seed &+ 301)) * (1 - j)
        return (j, og)
    }
}
