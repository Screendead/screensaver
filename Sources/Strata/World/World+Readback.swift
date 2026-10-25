import Foundation
import Metal
import simd

extension World {
    static var logSites: Bool { Knobs.current.sites }

    // The CPU copy's heights are 4x4 maxima of the column tops, so their maximum is the map's.
    static func readBack(_ cpu: MTLTexture) -> (heights: [Float], floors: [Float], ground: [Float], under: [Float], top: Float) {
        let s = MapLayout.size >> MapLayout.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 4)
        cpu.getBytes(&raw, bytesPerRow: s * 8, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: 0)
        let heights = (0..<(s * s)).map { Float(raw[$0 * 4]) }
        return (heights, (0..<(s * s)).map { Float(raw[$0 * 4 + 1]) }, (0..<(s * s)).map { Float(raw[$0 * 4 + 2]) },
                (0..<(s * s)).map { Float(raw[$0 * 4 + 3]) }, heights.max() ?? 0)
    }
}
