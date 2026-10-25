import Foundation
import simd

// The map shown as the CPU sees it: per 4x4 cell of its columns, heights (maxima, so conservative), and what stands on
// it. World refills it as it swaps in each new map; the camera and the birds read it thousands of times a frame.
package final class Terrain {
    // The map's origin in world voxels; Int32.min before the first map.
    package var origin = SIMD2<Int32>(Int32.min, Int32.min)
    package var heights: [Float] = []
    package var floors: [Float] = []
    // Per cell, the highest ground islands aside and the lowest island underside (4096: none).
    package var ground: [Float] = []
    package var under: [Float] = []
    package var sites: [SiteInfo] = []
    // sites' lamps by Lamp.index.
    package var lamps: [Int: SiteInfo.Lamp] = [:]
    package var giants: [GiantInfo] = []
    package var isles: [IslandInfo] = []
    // The caves of the cave table (placeCaves), world xz of each mouth.
    package var caves: [CaveRec] = []
    // Whether the jungle's roof (jungleCanopy in Shaders/Life/Jungle.metal) may stand on this map.
    package var jungle = false
    // The jungle's extent on this map: whether the province reaches each point of a 64-voxel grid over it.
    package var jungleGrid: [Bool] = []

    package init() {}

    // Conservative (4x4 max) height in world coords.
    package func height(x: Double, z: Double) -> Double? {
        cell(x, z).map { Double(heights[$0]) }
    }

    // The 4x4 cell under world (x, z) as islands shape it: its highest top, its highest ground islands aside, and the
    // lowest island underside over it (nil: no island).
    package func column(x: Double, z: Double) -> (top: Double, ground: Double, under: Double?)? {
        guard let i = cell(x, z) else { return nil }
        let u = Double(under[i])
        return (Double(heights[i]), Double(ground[i]), u < 4000 ? u : nil)
    }

    // Lower bound in world coords: every column of the 4x4 cell is solid from the ground up to at least this.
    func solidFloor(x: Double, z: Double) -> Double? {
        cell(x, z).map { Double(floors[$0]) }
    }

    // The CPU cell under world (x, z); nil off the map, or before the first (the four arrays are filled together).
    private func cell(_ x: Double, _ z: Double) -> Int? {
        let s = MapLayout.size >> MapLayout.cpuLevel
        let cx = Int(floor((x - Double(origin.x)) / Double(1 << MapLayout.cpuLevel)))
        let cz = Int(floor((z - Double(origin.y)) / Double(1 << MapLayout.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !heights.isEmpty else { return nil }
        return cz * s + cx
    }

    // Whether the jungle province reaches each point of a 64-voxel grid over the map at `origin`.
    package static func jungleReach(origin o: SIMD2<Int32>, seed: UInt32) -> [Bool] {
        stride(from: 0, through: MapLayout.size, by: 64).flatMap { z in
            stride(from: 0, through: MapLayout.size, by: 64).map { x in
                Provinces.at(Double(Int(o.x) + x), Double(Int(o.y) + z), seed: seed).0 > 0.1
            }
        }
    }

    // Whether a flagged cell of the map's band mips (under a giant's crown or the jungle's roof) may lie within r of
    // world xz p.
    package func bandsNear(_ p: SIMD2<Double>, radius r: Double) -> Bool {
        if giants.contains(where: { simd_length($0.pos - p) < r + GiantInfo.reach }) { return true }
        guard jungle else { return false }
        let n = MapLayout.size / 64 + 1
        for z in 0..<n {
            for x in 0..<n where jungleGrid[z * n + x] {
                let q = SIMD2(Double(Int(origin.x) + x * 64), Double(Int(origin.y) + z * 64))
                if simd_length(q - p) < r + 64 { return true }
            }
        }
        return false
    }
}
