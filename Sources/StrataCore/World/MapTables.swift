import simd

// The tables placed with each map (placeSites, placeGiants, placeCaves), read from their buffers' bytes. The cave table
// follows the giant table in its buffer.
package enum MapTables {
    // No slot holds a cell yet: placeGiants and placeCaves fill a slot only when its cell changes.
    package static func clearGiantsAndCaves(_ table: UnsafeMutableRawPointer) {
        let none = GiantRec(cell: SIMD2(Int32.min, Int32.min))
        let slots = table.bindMemory(to: GiantRec.self, capacity: MapLayout.giantWin * MapLayout.giantWin)
        for i in 0..<(MapLayout.giantWin * MapLayout.giantWin) { slots[i] = none }
        let caveSlots = (table + MapLayout.caveOffset).bindMemory(to: CaveRec.self, capacity: MapLayout.caveWin * MapLayout.caveWin)
        for i in 0..<(MapLayout.caveWin * MapLayout.caveWin) { caveSlots[i] = CaveRec(cell: SIMD2(Int32.min, Int32.min)) }
    }

    // The giant and cave tables' bytes, the cave table after the giants'.
    package static var giantTableBytes: Int { MapLayout.caveOffset + MapLayout.caveWin * MapLayout.caveWin * MemoryLayout<CaveRec>.stride }

    package static func sites(in raw: UnsafeRawPointer) -> [SiteInfo] {
        var list: [SiteInfo] = []
        for i in 0..<(MapLayout.siteGrid * MapLayout.siteGrid) {
            let base = raw + i * MapLayout.siteStride
            let head = base.load(as: SIMD4<Int32>.self)
            guard let kind = SiteInfo.Kind(rawValue: Int(head.z)), kind != .none else { continue }
            let info = (base + 16).load(as: SIMD4<Float>.self)
            let c = SIMD2(Double(head.x), Double(head.y))
            var lamps: [SiteInfo.Lamp] = []
            for l in 0..<Int(info.w) {
                let v = (base + 32 + 16 * l).load(as: SIMD4<Float>.self)
                let w = Int(v.w), k = w % 16, rise = Double((w / 16) % 64), code = w / 1024
                let pos = SIMD3(c.x + Double(v.x), Double(v.y), c.y + Double(v.z))
                let off = code > 0 ? SIMD3(Double(code % 32 - 16), rise, Double(code / 32 - 16)) : SIMD3(0, rise, 0)
                lamps.append(.init(pos: pos, kind: k, rise: rise, index: i * MapLayout.maxLamps + l, smoke: pos + off))
            }
            list.append(SiteInfo(index: i, id: UInt32(bitPattern: head.w), kind: kind, pos: c + 0.5, radius: Double(info.x), lamps: lamps))
        }
        return list
    }

    package static func caves(in giantTable: UnsafeRawPointer) -> [CaveRec] {
        let p = (giantTable + MapLayout.caveOffset).bindMemory(to: CaveRec.self, capacity: MapLayout.caveWin * MapLayout.caveWin)
        return (0..<(MapLayout.caveWin * MapLayout.caveWin)).compactMap { p[$0].flags & 1 != 0 ? p[$0] : nil }
    }

    package static func giants(in table: UnsafeRawPointer) -> [GiantInfo] {
        let p = table.bindMemory(to: GiantRec.self, capacity: MapLayout.giantWin * MapLayout.giantWin)
        return (0..<(MapLayout.giantWin * MapLayout.giantWin)).compactMap { i in
            let r = p[i]
            // The crown's top: giantPad's dome in Shaders/Saver.metal rises about yT + 36 + 0.55 * 56 + 9.5, the high limbs'
            // clumps to about yT + 84.
            return r.g > 0 ? GiantInfo(pos: SIMD2(Double(r.c.x), Double(r.c.y)), ground: Double(r.g), top: Double(r.g + r.yT) + 85, kind: Int(r.kind), yT: Double(r.yT), h: r.h) : nil
        }
    }
}
