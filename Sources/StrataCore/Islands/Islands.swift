import simd

enum Islands {
    // Must match ISLE_WIN, ISLE_MWIN and ISLE_SLOTS (and IsleRec's 64 bytes) in Shaders/Saver.metal.
    static let win = 60, mwin = 13
    static let slots = win * win + mwin * mwin
    static let recordSize = 64
    // Must match ISLE_TABLE and IsleShade in Shaders/Saver.metal: the islands the shading reads, nearest the camera first.
    static let tableSize = 16
    static let mistPuffs = 6
    static let entryBytes = 80 + mistPuffs * 16
    static let tableBytes = 16 + tableSize * entryBytes
    // VR_ISLE_FIELD=1: the test field round (-3500, -2500) in place of island country; 2: no islands (tests).
    static var field: Int32 { Int32(Knobs.current.env["VR_ISLE_FIELD"] ?? "") ?? 0 }

    static func stretch(_ r: Double, _ h: UInt32) -> Double {
        (r > 40 && (r <= 210 || r >= 420) && Provinces.hashf(Provinces.pcg(h &+ 10)) < 0.33) ? 1.5 : 1.0
    }

    // How far out from its lip (lipD from the centre) a fall's foot lands. Must match isleFallOut in Shaders/Saver.metal.
    static func fallOut(_ L: Double, _ reach: Double, _ lipD: Double) -> Double { max(2, min(24, min(0.09 * L, reach - 6 - lipD))) }

    // Its form (Shaders/Saver.metal isleForm) and the crag a spire carries over its top (isleCrag).
    static func form(_ r: Double, _ h: UInt32) -> Int {
        let c = Provinces.hashf(Provinces.pcg(h &+ 30))
        return r >= 30 && r <= 130 && c < 0.2 ? 1 : c < 0.45 ? 2 : c < 0.7 ? 3 : 0
    }
    static func crag(_ r: Double, _ h: UInt32) -> Double { form(r, h) == 1 ? 0.3 * r : 0 }

    // No slot holds a cell yet: placeIslands fills a slot only when its cell changes.
    static func clear(_ table: UnsafeMutableRawPointer) {
        let slots = table.bindMemory(to: IsleRec.self, capacity: Islands.slots)
        for i in 0..<Islands.slots { slots[i] = IsleRec() }
    }

    static func parse(_ table: UnsafeRawPointer) -> [IslandInfo] {
        let p = table.bindMemory(to: IsleRec.self, capacity: slots)
        var out: [IslandInfo] = []
        for i in 0..<slots {
            let r = p[i]
            guard r.r > 0, r.cell.x != Int32.min else { continue }
            let R = Double(r.r)
            out.append(IslandInfo(pos: SIMD2(Double(r.c.x), Double(r.c.y)), radius: R, reach: stretch(R, r.h) * (1.45 * R + 8),
                                  top: Double(r.a) + 0.06 * R + 10 + crag(R, r.h), keel: Double(r.k), base: Double(r.b), hash: r.h, massif: i >= win * win,
                                  fall: r.f.w > 0 ? SIMD4(Double(r.f.x), Double(r.f.y), Double(r.f.z), Double(r.f.w)) : nil))
        }
        return out
    }
}
