import Foundation
import simd

extension Islands {
    // Spheres bounding where an island within d of the eye shows: its column of air down to the ground under it and its
    // reflection in the sea (columns); the ground its sky occlusion darkens, as a sphere round its base (occlusion; else see
    // occlusionBox); its shade cast along the key light (shade).
    static func spheres(_ list: [IslandInfo], eye: SIMD3<Double>, within d: Double, key: SIMD3<Double>, columns: Bool = true,
                        occlusion: Bool = true, shade: Bool = true) -> [(c: SIMD3<Double>, r: Double)] {
        var out: [(c: SIMD3<Double>, r: Double)] = []
        let e2 = SIMD2(eye.x, eye.z)
        for s in list where simd_length(s.pos - e2) < d + s.reach {
            let r = 1.2 * s.reach + 30
            let n = max(1, Int(((s.top - s.base) / s.reach).rounded(.up)))
            for i in 0...n where columns {
                let y = s.base + (s.top - s.base) * Double(i) / Double(n)
                out.append((SIMD3(s.pos.x, y, s.pos.y), r))
                out.append((SIMD3(s.pos.x, 2 * Sight.sea - y, s.pos.y), r))
            }
            if occlusion { out.append((SIMD3(s.pos.x, s.base, s.pos.y), 3 * s.reach)) }
            guard shade else { continue }
            guard key.y > 0.02 else { continue }
            let mid = 0.5 * (s.top + s.keel)
            let run = min((mid - s.base) / key.y, 4000)
            let m = max(1, Int((run / s.reach).rounded(.up)))
            for i in 1...m { out.append((SIMD3(s.pos.x, mid, s.pos.y) - key * (run * Double(i) / Double(m)), r)) }
        }
        return out
    }

    typealias Box = (lo: SIMD3<Double>, hi: SIMD3<Double>)

    // Boxes bounding, for an island within d of the eye, its column (all of it lies within its reach of its centre) from
    // the sea's floor to its top, and that column's reflection in the sea (columns); its shade cast along the key light
    // (shade, Islands.spheres' spheres).
    static func boxes(_ list: [IslandInfo], eye: SIMD3<Double>, within d: Double, key: SIMD3<Double>, columns: Bool, shade: Bool) -> [Box] {
        var out: [Box] = []
        let e2 = SIMD2(eye.x, eye.z)
        for s in list where simd_length(s.pos - e2) < d + s.reach {
            if columns {
                // The slope and stair-shadow filters read a few columns round a point (16 voxels wide on the coarsest ring).
                let r = s.reach + 96, lo = 0.0, hi = s.top + 8
                out.append((SIMD3(s.pos.x - r, lo, s.pos.y - r), SIMD3(s.pos.x + r, hi, s.pos.y + r)))
                out.append((SIMD3(s.pos.x - r, 2 * Sight.sea - hi, s.pos.y - r), SIMD3(s.pos.x + r, 2 * Sight.sea - lo, s.pos.y + r)))
            }
            if shade {
                for sp in spheres([s], eye: eye, within: .infinity, key: key, columns: false, occlusion: false) {
                    out.append((sp.c - sp.r, sp.c + sp.r))
                }
            }
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
    static func occlusionBox(_ s: IslandInfo) -> Box {
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
}
