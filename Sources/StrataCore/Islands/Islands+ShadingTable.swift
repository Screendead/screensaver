import Foundation
import simd

extension Islands {
    // The shading table: the islands nearest p within reach of the view (count, then per island centre xz, reach and
    // underside's mean height; radius, top, keel and its mist's fade with distance from p; its fall's lip xz, top and
    // foot, or zeros; its mist bank's bounding sphere, radius 0 for none, and puffs (centre, radius), all in world
    // coordinates), as bytes for setFragmentBytes. view: the eye, its forward axis and the half angle of a cone round the
    // frame; mist that lies wholly outside it is left out (the scene pass draws mist on primary rays only). Each entry
    // ends with the square of its sky occlusion's reach (occlusionReach). mist: boxes bounding the mist banks and plumes
    // it holds (isleMist's ellipsoids).
    package static func table(_ list: [IslandInfo], near p: SIMD2<Double>, within d: Double,
                      view: (eye: SIMD3<Double>, fwd: SIMD3<Double>, half: Double)? = nil) -> (bytes: [UInt8], mist: [Box]) {
        func seen(_ c: SIMD3<Double>, _ r: Double) -> Bool {
            guard let v = view else { return true }
            let w = c - v.eye, l = simd_length(w)
            if l <= r { return true }
            return acos(max(-1, min(1, simd_dot(w / l, v.fwd)))) < v.half + asin(min(1, r / l))
        }
        let near = list.filter { simd_length($0.pos - p) < d + $0.reach }.sorted { simd_length($0.pos - p) < simd_length($1.pos - p) }
        var bytes = [UInt8](repeating: 0, count: tableBytes)
        var mist: [Box] = []
        let n = min(near.count, tableSize)
        bytes.withUnsafeMutableBytes { b in
            b.storeBytes(of: Int32(n), as: Int32.self)
            for i in 0..<n {
                let s = near[i]
                let o = 16 + i * entryBytes
                let d = simd_length(s.pos - p)
                let fade = 1 - smoothstep(2400, 3000, d)
                let R = s.radius, th = s.top - s.keel
                // A ring of flattened puffs round the keel, on most bigger islands, at its flanks' lower half; the plume at a
                // fall's foot (bounds as isleMist takes them).
                let ring = SIMD3(s.pos.x, s.top - 0.48 * th, s.pos.y), ringR = 1.4 * R + 30
                let hasRing = R >= 60 && Provinces.hashf(Provinces.pcg(s.hash &+ 740)) < 0.85 && seen(ring, ringR)
                let plume = s.fall.map { f -> Box? in
                    let L = f.z - f.w, rw = 5 + 0.012 * R, lip = SIMD2(f.x, f.y)
                    let u = simd_normalize(lip - s.pos), q = lip + u * (fallOut(L, s.reach, simd_length(lip - s.pos)) + 0.6 * rw)
                    let c = SIMD3(q.x, f.w + 0.06 * L, q.y)
                    guard seen(c, 0.36 * L + 3 * rw + 6) else { return nil }
                    let e = SIMD3(2.6 * rw + 2, 0.15 * L + rw + 2, 2.6 * rw + 2)
                    return (c - e, c + e)
                } ?? nil
                b.storeBytes(of: SIMD4<Float>(Float(s.pos.x), Float(s.pos.y), Float(s.reach), Float(0.5 * (s.keel + s.top))), toByteOffset: o, as: SIMD4<Float>.self)
                let misty = R >= 60 && (hasRing || plume != nil) && fade > 0
                b.storeBytes(of: SIMD4<Float>(Float(R), Float(s.top), Float(s.keel), Float(misty ? fade : 0)), toByteOffset: o + 16, as: SIMD4<Float>.self)
                if misty {
                    if hasRing {
                        let e = SIMD3(ringR, 0.08 * th + 0.28 * (0.44 * R + 15) + 2, ringR)
                        mist.append((ring - e, ring + e))
                    }
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
