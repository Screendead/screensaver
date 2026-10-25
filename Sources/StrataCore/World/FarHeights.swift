import Foundation
import simd

// Far heights for the camera: level 3's max mip 2 (32-voxel cells, toroidal, about +-7.7k around the camera),
// read back whenever level 3 changes; conservative, being a max.
package final class FarHeights {
    package static let n = 512, cell = 32.0
    package var heights: [Float16] = []
    package var centre = SIMD2<Int>(Int.min, Int.min)

    package init() {}

    // Conservative terrain height at world (x, z) out to about 7.7k, or nil.
    package func height(x: Double, z: Double) -> Double? {
        guard !heights.isEmpty else { return nil }
        let cx = Int(floor(x / FarHeights.cell)), cz = Int(floor(z / FarHeights.cell))
        guard abs(cx - centre.x) < 236, abs(cz - centre.y) < 236 else { return nil }
        let n = FarHeights.n
        return Double(heights[(cz & (n - 1)) * n + (cx & (n - 1))])
    }
}
