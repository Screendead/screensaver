import Metal
import simd

// One table slot (Shaders/Saver.metal IsleRec): r > 0 an island standing.
struct IsleRec {
    var cell = SIMD2<Int32>(Int32.min, Int32.min)
    var c = SIMD2<Float>()
    var r: Float = 0
    var a: Float = 0
    var d: Float = 0
    var b: Float = 0
    var k: Float = 0
    var h: UInt32 = 0
    var fa = SIMD2<Float>()
    var f = SIMD4<Float>()
}

// A floating island as the camera, the birds and the shading see it, in world coordinates: nothing of it lies further
// than `reach` from `pos`; its top stands about `top` at the centre, its underside nowhere under `keel`, over ground no
// higher than `base`.
struct IslandInfo {
    var pos: SIMD2<Double>
    var radius: Double
    var reach: Double
    var top: Double
    var keel: Double
    var base: Double
    var hash: UInt32
    var massif: Bool
    // Its waterfall: the lip's centre (x, z), the water's top there and the fall's foot.
    var fall: SIMD4<Double>?

    // Room under the keel for the flight to pass beneath.
    var passable: Bool { keel - base >= 90 }
}

enum Islands {
    // Must match ISLE_WIN, ISLE_MWIN and ISLE_SLOTS (and IsleRec's 64 bytes) in Shaders/Saver.metal.
    static let win = 60, mwin = 13
    static let slots = win * win + mwin * mwin
    static let recordSize = 64
    // Must match ISLE_TABLE and IsleShade in Shaders/Saver.metal: the islands the shading reads, nearest the camera first.
    static let tableSize = 16
    static let mistPuffs = 6
    static let entryBytes = 64 + mistPuffs * 16
    static let tableBytes = 16 + tableSize * entryBytes
    // VR_ISLE_FIELD=1: the test field round (-3500, -2500) in place of island country; 2: no islands (tests).
    static let field = Int32(ProcessInfo.processInfo.environment["VR_ISLE_FIELD"] ?? "") ?? 0

    static func stretch(_ r: Double, _ h: UInt32) -> Double {
        (r > 40 && (r <= 210 || r >= 420) && Provinces.hashf(Provinces.pcg(h &+ 10)) < 0.33) ? 1.5 : 1.0
    }

    // How far out from its lip (lipD from the centre) a fall's foot lands. Must match isleFallOut in Shaders/Saver.metal.
    static func fallOut(_ L: Double, _ reach: Double, _ lipD: Double) -> Double { max(2, min(24, min(0.09 * L, reach - 6 - lipD))) }

    // Its form (Shaders/Saver.metal isleForm) and the crag a spire carries over its top (isleCrag).
    static func form(_ r: Double, _ h: UInt32) -> Int {
        let c = Provinces.hashf(Provinces.pcg(h &+ 30))
        return r >= 30 && r <= 220 && c < 0.4 ? 1 : c < 0.58 ? 2 : c < 0.76 ? 3 : 0
    }
    static func crag(_ r: Double, _ h: UInt32) -> Double { form(r, h) == 1 ? 0.3 * r : 0 }

    static func parse(_ buf: MTLBuffer) -> [IslandInfo] {
        let p = buf.contents().bindMemory(to: IsleRec.self, capacity: slots)
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

    // The shading table: the islands nearest p within reach of the view (count, then per island centre xz, reach and
    // underside's mean height; radius, top, keel and its mist's fade with distance from p; its fall's lip xz, top and
    // foot, or zeros; its mist bank's bounding sphere, radius 0 for none, and puffs (centre, radius), all in world
    // coordinates), as bytes for setFragmentBytes. view: the eye, its forward axis and the half angle of a cone round the
    // frame; mist that lies wholly outside it is left out (the scene pass draws mist on primary rays only).
    static func table(_ list: [IslandInfo], near p: SIMD2<Double>, within d: Double,
                      view: (eye: SIMD3<Double>, fwd: SIMD3<Double>, half: Double)? = nil) -> [UInt8] {
        func seen(_ c: SIMD3<Double>, _ r: Double) -> Bool {
            guard let v = view else { return true }
            let w = c - v.eye, l = simd_length(w)
            if l <= r { return true }
            return acos(max(-1, min(1, simd_dot(w / l, v.fwd)))) < v.half + asin(min(1, r / l))
        }
        let near = list.filter { simd_length($0.pos - p) < d + $0.reach }.sorted { simd_length($0.pos - p) < simd_length($1.pos - p) }
        var bytes = [UInt8](repeating: 0, count: tableBytes)
        let n = min(near.count, tableSize)
        bytes.withUnsafeMutableBytes { b in
            b.storeBytes(of: Int32(n), as: Int32.self)
            for i in 0..<n {
                let s = near[i]
                let o = 16 + i * entryBytes
                let d = simd_length(s.pos - p)
                let fade = 1 - Provinces.smoothstep(2400, 3000, d)
                let R = s.radius, th = s.top - s.keel
                // A ring of flattened puffs round the keel, on most bigger islands, at its flanks' lower half; the plume at a
                // fall's foot (bounds as isleMist takes them).
                let ring = SIMD3(s.pos.x, s.top - 0.48 * th, s.pos.y), ringR = 1.4 * R + 30
                let hasRing = R >= 60 && Provinces.hashf(Provinces.pcg(s.hash &+ 740)) < 0.85 && seen(ring, ringR)
                let plume = s.fall.map { f -> Bool in
                    let L = f.z - f.w, rw = 5 + 0.012 * R, lip = SIMD2(f.x, f.y)
                    let u = simd_normalize(lip - s.pos), q = lip + u * (fallOut(L, s.reach, simd_length(lip - s.pos)) + 0.6 * rw)
                    return seen(SIMD3(q.x, f.w + 0.06 * L, q.y), 0.36 * L + 3 * rw + 6)
                } ?? false
                b.storeBytes(of: SIMD4<Float>(Float(s.pos.x), Float(s.pos.y), Float(s.reach), Float(0.5 * (s.keel + s.top))), toByteOffset: o, as: SIMD4<Float>.self)
                b.storeBytes(of: SIMD4<Float>(Float(R), Float(s.top), Float(s.keel), Float(R >= 60 && (hasRing || plume) ? fade : 0)), toByteOffset: o + 16, as: SIMD4<Float>.self)
                let f = s.fall.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), Float($0.w)) } ?? SIMD4<Float>()
                b.storeBytes(of: f, toByteOffset: o + 32, as: SIMD4<Float>.self)
                let an0 = Provinces.hashf(Provinces.pcg(s.hash &+ 740))
                guard hasRing, fade > 0 else { continue }
                b.storeBytes(of: SIMD4<Float>(Float(ring.x), Float(ring.y), Float(ring.z), Float(ringR)), toByteOffset: o + 48, as: SIMD4<Float>.self)
                for j in 0..<mistPuffs {
                    let hj = Provinces.pcg(s.hash &+ 741 &+ UInt32(j) &* 7)
                    let an = an0 * 8.9759 + Double(j) * 2 * .pi / Double(mistPuffs) + 0.5 * (Provinces.hashf(hj) - 0.5)
                    let rr = R * (0.6 + 0.35 * Provinces.hashf(Provinces.pcg(hj)))
                    let y = s.top - (0.4 + 0.16 * Provinces.hashf(Provinces.pcg(hj &+ 1))) * th
                    let rh = (0.3 + 0.14 * Provinces.hashf(Provinces.pcg(hj &+ 2))) * R + 15
                    b.storeBytes(of: SIMD4<Float>(Float(s.pos.x + cos(an) * rr), Float(y), Float(s.pos.y + sin(an) * rr), Float(rh)),
                                 toByteOffset: o + 64 + 16 * j, as: SIMD4<Float>.self)
                }
            }
        }
        return bytes
    }
}
