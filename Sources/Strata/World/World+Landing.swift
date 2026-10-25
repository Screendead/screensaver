import Foundation
import QuartzCore
import StrataCore
import simd

extension World {
    // Swaps in the map just built into buffers[index], once the GPU has finished it: its tables and CPU copy become the
    // terrain's. On the main thread.
    func land(_ r: Readback, into buf: Buffer, index: Int, origin o: SIMD2<Int32>, started t0: CFTimeInterval, sync: Bool) {
        buf.origin = o
        buf.globalMax = r.top + 1
        terrain.sites = MapTables.sites(in: buf.sites.contents())
        terrain.lamps = Dictionary(terrain.sites.flatMap { $0.lamps.map { ($0.index, $0) } }, uniquingKeysWith: { $1 })
        terrain.giants = MapTables.giants(in: buf.giants.contents())
        terrain.isles = Islands.parse(buf.isles.contents())
        terrain.caves = MapTables.caves(in: buf.giants.contents())
        if let r = Knobs.current.caveList { logCaves(near: o, within: r) }
        terrain.jungleGrid = Terrain.jungleReach(origin: o, seed: seed)
        terrain.jungle = terrain.jungleGrid.contains(true)
        if Knobs.current.sites { logSites(in: buf, origin: o) }
        // Frames in flight hold the map they read until they complete.
        if index != active { buffers[active] = nil }
        active = index
        scratch = nil
        terrain.origin = o
        terrain.heights = r.heights
        terrain.floors = r.floors
        terrain.ground = r.ground
        terrain.under = r.under
        pending = false
        if Knobs.current.timing { print(String(format: "world regen %.1f ms (%@)", (CACurrentMediaTime() - t0) * 1000, sync ? "sync" : "async")) }
    }
}
