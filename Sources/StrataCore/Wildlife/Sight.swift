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
    // The world's edge: a circle this far around the camera, hazed from 0.7 of it (the largest it can grow to, so
    // the edge haze is never thicker here than drawn).
    var worldEdge: Double
    // The 1-voxel map, which the floor heights describe, is drawn unmorphed this far around the camera; past it the
    // coarse levels' mean heights can dip below the floor.
    var floorReach: Double
    // The Renderer owns the world and outlives every Sight it makes.
    unowned(unsafe) let world: World
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
        let hf = 30.0, a = 0.0009, c2 = 1.3e-6, K = 1.6, lambda = 6e-5
        let k = exp(-(o.y - Sight.sea) / hf)
        // As the shader: a downward ray flattens to +1e-4.
        let dy = abs(d.y) < 1e-4 ? 1e-4 : d.y
        let heightTerm = a * hf * k * (1 - exp(-dy * t / hf)) / dy
        let ep = o + d * t
        let r = simd_length(SIMD2(ep.x - pos.x, ep.z - pos.z))
        if r >= worldEdge { return 0 }
        let night = Sight.smoothstep(-0.04, -0.20, sunY)
        var clear = 1 + (0.15 - 1) * Sight.smoothstep(0.2, 0.45, sunY)
        clear *= 1 + (0.35 - 1) * night
        // As the shader's airAloft: the haze thins above 120.
        let air = (K * (1 - exp(-c2 * t * t / K)) + lambda * t * t / (t + 1500)) * exp(-max(o.y + min(d.y, 0.2) * 0.75 * min(t, 2000) - 120, 0) / 400)
        return exp(-((heightTerm + air) * clear + 6 * Sight.smoothstep(0.7 * worldEdge, worldEdge, r)))
    }

    // True only when the segment provably passes through solid ground.
    func blocked(_ a: SIMD3<Double>, _ b: SIMD3<Double>, step: Double = 1.5) -> Bool {
        let v = b - a
        let len = simd_length(v)
        guard len > 2 * step else { return false }
        let n = Int(len / step)
        for i in 1..<n {
            let q = a + v * (Double(i) / Double(n))
            if simd_length(SIMD2(q.x - pos.x, q.z - pos.z)) < floorReach, let f = world.solidFloor(x: q.x, z: q.z), q.y < f - 0.5 - slack { return true }
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
    // the extra margin covers, and shift where the mirrored ray meets the water and which way it leaves it.
    func reflected(_ p: SIMD3<Double>, size: Double, margin: Double = 0) -> Bool {
        guard pos.y > Sight.sea else { return false }
        let m = SIMD3(p.x, 2 * Sight.sea - p.y, p.z)
        guard inFrame(m, size: size, margin: margin + 0.04) else { return false }
        // At or under the surface the mirror geometry below breaks down: count it seen.
        if p.y < Sight.sea + 0.5 { return true }
        let v = m - pos, t = simd_length(v)
        // Where the camera ray meets the water.
        let s = (pos.y - Sight.sea) / (pos.y - m.y)
        let w = pos + v * s
        let near = [SIMD2(0.0, 0), SIMD2(4, 0), SIMD2(-4, 0), SIMD2(0, 4), SIMD2(0, -4)]
        if near.allSatisfy({ world.solidFloor(x: w.x + $0.x, z: w.z + $0.y).map { $0 > Sight.sea + 0.5 } ?? false }) { return false }
        let d = v / t
        let t1 = t * s
        let up = SIMD3(0.0, 1, 0) * reach(size: size, at: t)
        let clear = [p, p + up, p - up].map { q in
            transmittance(from: pos, dir: d, t: t1) * transmittance(from: w, dir: simd_normalize(q - w), t: simd_length(q - w))
        }.max()!
        if t > 1, clear < Sight.fogHidden * (slack > 0 ? 0.8 : 1) { return false }
        let r = reach(size: size, at: t)
        if hiddenBehindGround(from: pos, w, extent: 0.04 * t1) { return false }
        return !hiddenBehindGround(from: w + SIMD3(0, 0.05, 0), p, extent: r + 0.04 * (t - t1))
    }

    static func smoothstep(_ a: Double, _ b: Double, _ x: Double) -> Double {
        let t = min(max((x - a) / (b - a), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
