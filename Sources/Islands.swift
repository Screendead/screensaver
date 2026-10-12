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

    // Spheres bounding where an island within d of the eye shows: its column of air down to the ground under it and its
    // reflection in the sea; with `shade`, also the ground its sky occlusion darkens and its shade cast along the key light.
    // occlusion: the ground its sky occlusion darkens as a sphere round its base (else see occlusionBox).
    static func spheres(_ list: [IslandInfo], eye: SIMD3<Double>, within d: Double, key: SIMD3<Double>, shade: Bool,
                        occlusion: Bool = true) -> [(c: SIMD3<Double>, r: Double)] {
        var out: [(c: SIMD3<Double>, r: Double)] = []
        let e2 = SIMD2(eye.x, eye.z)
        for s in list where simd_length(s.pos - e2) < d + s.reach {
            let r = 1.2 * s.reach + 30
            let n = max(1, Int(((s.top - s.base) / s.reach).rounded(.up)))
            for i in 0...n {
                let y = s.base + (s.top - s.base) * Double(i) / Double(n)
                out.append((SIMD3(s.pos.x, y, s.pos.y), r))
                out.append((SIMD3(s.pos.x, 2 * Sight.sea - y, s.pos.y), r))
            }
            guard shade else { continue }
            if occlusion { out.append((SIMD3(s.pos.x, s.base, s.pos.y), 3 * s.reach)) }
            guard key.y > 0.02 else { continue }
            let mid = 0.5 * (s.top + s.keel)
            let run = min((mid - s.base) / key.y, 4000)
            let m = max(1, Int((run / s.reach).rounded(.up)))
            for i in 1...m { out.append((SIMD3(s.pos.x, mid, s.pos.y) - key * (run * Double(i) / Double(m)), r)) }
        }
        return out
    }

    // The share of the sky a disc of radius R a height h overhead hides from a level patch r off its axis (isleSkyOcc).
    static func skyOcc(_ R: Double, _ h: Double, _ r: Double) -> Double {
        let q = h * h + r * r + R * R
        return 0.5 * (1 - (h * h + r * r - R * R) / max(q * q - 4 * r * r * R * R, 1e-3).squareRoot())
    }

    // How far from its axis an island's sky occlusion (isleSkyOcc) reaches: where it hides a five-hundredth of the sky
    // at most, from any height under its keel down to the sea.
    static func occlusionReach(_ s: IslandInfo) -> Double {
        let R = s.radius, hMax = max(0.5 * (s.keel + s.top) - Sight.sea, 4)
        func worst(_ r: Double) -> Double { (0...32).map { skyOcc(R, 4 + (hMax - 4) * Double($0) / 32, r) }.max()! }
        var lo = R, hi = 3 * s.reach
        if worst(hi) >= 0.002 { return hi }
        for _ in 0..<24 { let m = 0.5 * (lo + hi); if worst(m) < 0.002 { hi = m } else { lo = m } }
        return hi
    }

    // The box its sky occlusion reaches: from the sea's floor up to its underside's mean height.
    static func occlusionBox(_ s: IslandInfo) -> (lo: SIMD3<Double>, hi: SIMD3<Double>) {
        let r = occlusionReach(s)
        return (SIMD3(s.pos.x - r, 0, s.pos.y - r), SIMD3(s.pos.x + r, 0.5 * (s.keel + s.top), s.pos.y + r))
    }

    // Whether any of the spheres could show in the view cone (eye, forward axis, half angle, widened by `margin`).
    static func inView(_ spheres: [(c: SIMD3<Double>, r: Double)], eye: SIMD3<Double>, fwd: SIMD3<Double>, half: Double, margin: Double) -> Bool {
        spheres.contains { c, r in
            let w = c - eye, l = simd_length(w)
            if l <= r { return true }
            return acos(max(-1, min(1, simd_dot(w / l, fwd)))) < half + margin + asin(min(1, r / l))
        }
    }

    // The shading table: the islands nearest p within reach of the view (count, then per island centre xz, reach and
    // underside's mean height; radius, top, keel and its mist's fade with distance from p; its fall's lip xz, top and
    // foot, or zeros; its mist bank's bounding sphere, radius 0 for none, and puffs (centre, radius), all in world
    // coordinates), as bytes for setFragmentBytes. view: the eye, its forward axis and the half angle of a cone round the
    // frame; mist that lies wholly outside it is left out (the scene pass draws mist on primary rays only).
    // mist: spheres bounding the mist banks and plumes it holds.
    // Each entry ends with the square of its sky occlusion's reach (occlusionReach).
    static func table(_ list: [IslandInfo], near p: SIMD2<Double>, within d: Double,
                      view: (eye: SIMD3<Double>, fwd: SIMD3<Double>, half: Double)? = nil) -> (bytes: [UInt8], mist: [(c: SIMD3<Double>, r: Double)]) {
        func seen(_ c: SIMD3<Double>, _ r: Double) -> Bool {
            guard let v = view else { return true }
            let w = c - v.eye, l = simd_length(w)
            if l <= r { return true }
            return acos(max(-1, min(1, simd_dot(w / l, v.fwd)))) < v.half + asin(min(1, r / l))
        }
        let near = list.filter { simd_length($0.pos - p) < d + $0.reach }.sorted { simd_length($0.pos - p) < simd_length($1.pos - p) }
        var bytes = [UInt8](repeating: 0, count: tableBytes)
        var mist: [(c: SIMD3<Double>, r: Double)] = []
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
                let plume = s.fall.map { f -> (c: SIMD3<Double>, r: Double)? in
                    let L = f.z - f.w, rw = 5 + 0.012 * R, lip = SIMD2(f.x, f.y)
                    let u = simd_normalize(lip - s.pos), q = lip + u * (fallOut(L, s.reach, simd_length(lip - s.pos)) + 0.6 * rw)
                    let c = SIMD3(q.x, f.w + 0.06 * L, q.y), r = 0.36 * L + 3 * rw + 6
                    return seen(c, r) ? (c, r) : nil
                } ?? nil
                b.storeBytes(of: SIMD4<Float>(Float(s.pos.x), Float(s.pos.y), Float(s.reach), Float(0.5 * (s.keel + s.top))), toByteOffset: o, as: SIMD4<Float>.self)
                let misty = R >= 60 && (hasRing || plume != nil) && fade > 0
                b.storeBytes(of: SIMD4<Float>(Float(R), Float(s.top), Float(s.keel), Float(misty ? fade : 0)), toByteOffset: o + 16, as: SIMD4<Float>.self)
                if misty {
                    if hasRing { mist.append((ring, ringR)) }
                    if let pl = plume { mist.append(pl) }
                }
                let f = s.fall.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), Float($0.w)) } ?? SIMD4<Float>()
                b.storeBytes(of: f, toByteOffset: o + 32, as: SIMD4<Float>.self)
                let occ = occlusionReach(s)
                b.storeBytes(of: SIMD4<Float>(Float(occ * occ), 0, 0, 0), toByteOffset: o + 64 + 16 * mistPuffs, as: SIMD4<Float>.self)
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
        return (bytes, mist)
    }
}
