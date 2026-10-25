import Foundation
import simd

// Birds (and bats) in a few small flocks around the camera. Each flock steers an anchor through the terrain on a
// slow noise heading; its birds follow the anchor as boids. All randomness is drawn from the seed in step order,
// so a snapshot reproduces.
package final class Wildlife {

    static let maxFlocks = 8
    static let maxBirds = 600
    // The shader culls by flock, then by cluster: runs of up to clusterSize birds under one bounding sphere, so a
    // pixel near a big murmuration tests a few clusters, not every bird. Must match MAX_FLOCKS and MAX_CLUSTERS in
    // Shaders/Saver.metal.
    static let maxClusters = 48, clusterSize = 16
    package static let bufferLength = ((maxFlocks + maxClusters) * 2 + maxBirds * 4) * MemoryLayout<SIMD4<Float>>.stride

    package var flocks: [Flock] = []
    private var state: UInt64
    var time = 0.0
    var spawnTimer = 0.0
    var nextId = 1
    var spawnTurn = 0
    var lastCam: SIMD3<Double>?
    var camVel = SIMD3<Double>.zero
    var lastLook: (yaw: Double, pitch: Double)?
    var camYawRate = 0.0, camPitchRate = 0.0
    // What the frame can show this step; set by step().
    var sight: Sight?
    var hiddenFrames: [(Sight, Double)] = []
    // Spawn attempts that found no place out of sight, and those that did, by kind; for VR_POPLOG.
    var spawnStats: [String: (tries: Int, made: Int, transit: Double)] = [:]
    var failWhy: [String: Int] = [:]
    var spawnMs = 0.0
    let noiseSeed: UInt32
    package var heightAt: (Double, Double) -> Double? = { _, _ in nil }
    // The CPU cell under world (x, z) with its islands (Terrain.column): top, ground islands aside, lowest underside.
    package var columnAt: (Double, Double) -> (top: Double, ground: Double, under: Double?)? = { _, _ in nil }
    // Giant trees on the loaded map (world coordinates).
    package var giantsAt: () -> [GiantInfo] = { [] }
    // Floating islands round the camera.
    package var islesAt: () -> [IslandInfo] = { [] }

    package init(seed: UInt32) {
        state = UInt64(seed) &* 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03
        noiseSeed = seed &* 2_654_435_761
    }

    func rand() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }

    func rand(_ r: ClosedRange<Double>) -> Double { r.lowerBound + (r.upperBound - r.lowerBound) * rand() }

    // 2D gradient noise in [-1, 1]: the flow field flocks drift on.
    func noise(_ x: Double, _ y: Double, _ s: UInt32) -> Double {
        let ix = Int32(floor(x)), iy = Int32(floor(y))
        let fx = x - floor(x), fy = y - floor(y)
        func grad(_ i: Int32, _ j: Int32, _ dx: Double, _ dy: Double) -> Double {
            var h = UInt32(bitPattern: i) &* 1_597_334_677 ^ UInt32(bitPattern: j) &* 3_812_015_801 ^ s
            h = (h ^ (h >> 16)) &* 0x7FEB_352D
            h = (h ^ (h >> 15)) &* 0x846C_A68B
            let a = Double(h & 0xFFFF) / 65536 * 2 * .pi
            return cos(a) * dx + sin(a) * dy
        }
        let u = fx * fx * fx * (fx * (fx * 6 - 15) + 10), v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        let a = grad(ix, iy, fx, fy), b = grad(ix + 1, iy, fx - 1, fy)
        let c = grad(ix, iy + 1, fx, fy - 1), d = grad(ix + 1, iy + 1, fx - 1, fy - 1)
        return ((a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v) * 1.414
    }

    package func ground(_ x: Double, _ z: Double) -> Double { heightAt(x, z) ?? 200 }
    // The ground as a bird at height y meets it: well under an island, the land beneath it; else the island's top.
    package func ground(_ x: Double, _ z: Double, at y: Double) -> Double {
        guard let c = columnAt(x, z) else { return 200 }
        if let u = c.under, y < u - 12 { return c.ground }
        return c.top
    }
    // The land with every island lifted away.
    func groundBare(_ x: Double, _ z: Double) -> Double { columnAt(x, z)?.ground ?? 200 }
    // Highest a flock flies over ground g: under the cloud base, unless the ground itself rises past it.
    func topAlt(over g: Double, clearance c: Double) -> Double {
        max(FlightCamera.ceiling - 20, g + 2 * c)
    }
}
