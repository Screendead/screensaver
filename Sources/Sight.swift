import Foundation
import simd

// What one rendered frame can show of a bird: the view frustum, terrain in front of it, fog, and its reflection in
// water. Every test errs toward "seen": the terrain test uses a lower bound of the solid ground, the fog test leaves
// out the valley mist, which only thickens it.
struct Sight {
    var pos: SIMD3<Double>
    var fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>
    var tanV: Double
    var aspect: Double
    // Angle one scene cell subtends; birds are never drawn thinner than half of it.
    var pixelAngle: Double
    var sunY: Double
    var mapOrigin: SIMD2<Double>
    var mapSize: Double
    var floorAt: (Double, Double) -> Double?
    // > 0 for decisions that must hold however the bird moves in the next instant (spawning, retiring): occlusion
    // needs the sightline this far inside the ground, water anywhere within a few voxels counts, fog must be thicker.
    var slack = 0.0

    // Must match SEA in Shaders/Saver.metal.
    static let sea = 19.85
    // A bird fogged below this transmittance differs from the sky behind it by under half of one of the 8 output
    // levels: applyFog's in-scatter darkening, the cool aerial tint and a bird anywhere from black to twice the sky's
    // radiance, through the post tonemap and sRGB (bisected offline; half a level is reached at T = 0.14).
    static let fogHidden = 0.12

    // Half-extent a bird can cover at distance t: wings and body, plus the shader's minimum box footprint.
    func reach(size: Double, at t: Double) -> Double { 0.62 * size + 0.9 * t * pixelAngle }

    // Inside the frustum widened by `margin` radians on every side and by the bird's own extent; subpixel jitter and
    // the cell grid add one cell.
    func inFrame(_ p: SIMD3<Double>, size: Double, margin: Double = 0) -> Bool {
        let v = p - pos
        let t = simd_length(v)
        let r = reach(size: size, at: t) + t * pixelAngle
        if t <= r { return true }
        let z = simd_dot(v, fwd)
        if z <= -r { return false }
        let x = abs(simd_dot(v, right)), y = abs(simd_dot(v, up))
        // Distance from the side planes turned out by `margin`, positive outside.
        let cm = margin == 0 ? 1 : cos(margin), sm = margin == 0 ? 0 : sin(margin)
        func out(_ a: Double, _ tanEdge: Double) -> Double {
            let c0 = 1 / (1 + tanEdge * tanEdge).squareRoot(), s0 = tanEdge * c0
            let c = c0 * cm - s0 * sm, s = s0 * cm + c0 * sm
            return c <= 0 ? -.infinity : a * c - z * s
        }
        return out(x, tanV * aspect) < r && out(y, tanV) < r
    }

    // Fog over the bird's footprint is thinnest at its top edge or its near face: an upward ray leaves the height fog, and at long
    // range a pixel's change in slope moves the transmittance by several percent.
    func clearest(_ p: SIMD3<Double>, size: Double) -> Double {
        let v = p - pos, t = simd_length(v)
        let r = reach(size: size, at: t) + t * pixelAngle
        // The near face too: the shader fogs a bird at the box entry, and near the map edge the edge fog changes by
        // several percent per voxel.
        return [p, p + up * r, p - up * r, p - v / max(t, 1e-9) * min(r, 0.9 * t)].map { q in
            let w = q - pos, l = simd_length(w)
            return transmittance(from: pos, dir: w / l, t: l)
        }.max()!
    }

    // Port of fogTransmittance in Shaders/Saver.metal without the mist term (mist only lowers it).
    func transmittance(from o: SIMD3<Double>, dir d: SIMD3<Double>, t: Double) -> Double {
        let hf = 30.0, a = 0.0009, c2 = 1.3e-6
        let k = exp(-(o.y - Sight.sea) / hf)
        let dy = abs(d.y) < 1e-4 ? (d.y < 0 ? -1e-4 : 1e-4) : d.y
        let heightTerm = a * hf * k * (1 - exp(-dy * t / hf)) / dy
        let night = Sight.smoothstep(-0.04, -0.20, sunY)
        var clear = 1 + (0.15 - 1) * Sight.smoothstep(0.2, 0.45, sunY)
        clear *= 1 + (0.35 - 1) * night
        let ep = o + d * t
        let lx = ep.x - mapOrigin.x, lz = ep.z - mapOrigin.y
        let edgeDist = min(min(lx, mapSize - lx), min(lz, mapSize - lz))
        let edge = 5 * Sight.smoothstep(110 + 70 * night, 10, edgeDist)
        return exp(-((heightTerm + c2 * t * t) * clear + edge))
    }

    // True only when the segment provably passes through solid ground.
    func blocked(_ a: SIMD3<Double>, _ b: SIMD3<Double>, step: Double = 1.5) -> Bool {
        let v = b - a
        let len = simd_length(v)
        guard len > 2 * step else { return false }
        let n = Int(len / step)
        for i in 1..<n {
            let q = a + v * (Double(i) / Double(n))
            if let f = floorAt(q.x, q.z), q.y < f - 0.5 - slack { return true }
        }
        return false
    }

    private func hiddenBehindGround(from o: SIMD3<Double>, _ p: SIMD3<Double>, extent r: Double) -> Bool {
        let d = simd_normalize(p - o)
        let side = simd_normalize(simd_cross(d, up) + SIMD3(1e-9, 0, 0)), lift = simd_cross(side, d)
        for q in [p, p + lift * r, p - lift * r, p + side * r, p - side * r] where !blocked(o, q) { return false }
        return true
    }

    // Could this frame show the bird, directly or mirrored in water?
    func sees(_ p: SIMD3<Double>, size: Double, margin: Double = 0) -> Bool {
        if direct(p, size: size, margin: margin) { return true }
        return reflected(p, size: size, margin: margin)
    }

    func direct(_ p: SIMD3<Double>, size: Double, margin: Double = 0) -> Bool {
        guard inFrame(p, size: size, margin: margin) else { return false }
        let v = p - pos, t = simd_length(v)
        if t > 1, clearest(p, size: size) < Sight.fogHidden * (slack > 0 ? 0.8 : 1) { return false }
        return !hiddenBehindGround(from: pos, p, extent: reach(size: size, at: t))
    }

    // In the frame with its centre in the clear: for judging where a flock will show, not for hiding one.
    func shows(_ p: SIMD3<Double>, size: Double) -> Bool {
        inFrame(p, size: size) && !blocked(pos, p, step: 4)
    }

    // The water mirrors the bird about the sea plane; ripples tilt the mirror by a few hundredths of a radian, which
    // the extra margin covers.
    func reflected(_ p: SIMD3<Double>, size: Double, margin: Double = 0) -> Bool {
        guard p.y > Sight.sea, pos.y > Sight.sea else { return false }
        let m = SIMD3(p.x, 2 * Sight.sea - p.y, p.z)
        guard inFrame(m, size: size, margin: margin + 0.04) else { return false }
        let v = m - pos, t = simd_length(v)
        // Where the camera ray meets the water.
        let s = (pos.y - Sight.sea) / (pos.y - m.y)
        let w = pos + v * s
        let near = slack > 0 ? [SIMD2(0.0, 0), SIMD2(4, 0), SIMD2(-4, 0), SIMD2(0, 4), SIMD2(0, -4)] : [SIMD2(0.0, 0)]
        if near.allSatisfy({ floorAt(w.x + $0.x, w.z + $0.y).map { $0 > Sight.sea + 0.5 } ?? false }) { return false }
        let d = v / t
        let t1 = t * s
        if t > 1, transmittance(from: pos, dir: d, t: t1) * transmittance(from: w, dir: simd_normalize(p - w), t: t - t1)
            < Sight.fogHidden * (slack > 0 ? 0.8 : 1) {
            return false
        }
        let r = reach(size: size, at: t)
        if hiddenBehindGround(from: pos, w, extent: 0) { return false }
        return !hiddenBehindGround(from: w + SIMD3(0, 0.05, 0), p, extent: r)
    }

    static func smoothstep(_ a: Double, _ b: Double, _ x: Double) -> Double {
        let t = min(max((x - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
