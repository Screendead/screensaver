import Foundation
import Metal
import simd

// Birds (and bats) in a few small flocks around the camera. Each flock steers an anchor through the terrain on a
// slow noise heading; its birds follow the anchor as boids. All randomness is drawn from the seed in step order,
// so a snapshot reproduces.
final class Wildlife {
    enum Kind: Int { case songbird, raptor, gull, starling, bat, goose }

    struct Bird {
        var pos: SIMD3<Double>
        var vel: SIMD3<Double>
        var flap = 0.0
        var beat = 0.0 // > 0: seconds of flapping left; < 0: of gliding
        var bank = 0.0
        var size: Double
        var tint: Double
        var phase: Double
        var sheet = SIMD2<Double>.zero // starlings: place on the murmuration's sheet
    }

    struct Flight {
        let speed: Double, minSpeed: Double, maxSpeed: Double
        let span: Double
        let flapHz: Double, amp: Double, bias: Double
        let flapFor: ClosedRange<Double>, glideFor: ClosedRange<Double>
        let glideAngle: Double, glideExt: Double
        let handDroop: Double, sweep: Double // gliding hand angle relative to the arm; hand swept back (rad)
        let reach: Double, separation: Double, cohesion: Double, alignment: Double, pull: Double, wander: Double
        let clearance: Double
    }

    static func flight(_ k: Kind) -> Flight {
        switch k {
        case .songbird: // bounding: bursts of wingbeats between folded-wing swoops
            return Flight(speed: 9, minSpeed: 6, maxSpeed: 12, span: 0.85, flapHz: 11, amp: 0.85, bias: 0.1,
                          flapFor: 0.3...0.7, glideFor: 0.2...0.45, glideAngle: -0.1, glideExt: 0.35, handDroop: 0.0, sweep: 0.45,
                          reach: 9, separation: 1.8, cohesion: 0.25, alignment: 0.9, pull: 0.05, wander: 3.0, clearance: 6)
        case .raptor:
            return Flight(speed: 7, minSpeed: 5, maxSpeed: 9, span: 2.3, flapHz: 2.2, amp: 0.45, bias: 0.05,
                          flapFor: 0.8...1.5, glideFor: 5...11, glideAngle: 0.17, glideExt: 1, handDroop: 0.12, sweep: 0.1,
                          reach: 0, separation: 14, cohesion: 0, alignment: 0, pull: 0, wander: 0.6, clearance: 18)
        case .gull:
            return Flight(speed: 8, minSpeed: 6, maxSpeed: 11, span: 1.7, flapHz: 3, amp: 0.65, bias: 0.05,
                          flapFor: 1...2.6, glideFor: 1.5...4, glideAngle: 0.12, glideExt: 1, handDroop: -0.42, sweep: 0.45,
                          reach: 16, separation: 4.5, cohesion: 0.08, alignment: 0.35, pull: 0.03, wander: 2.0, clearance: 4)
        case .starling:
            return Flight(speed: 12, minSpeed: 4, maxSpeed: 18, span: 0.7, flapHz: 8, amp: 0.8, bias: 0.05,
                          flapFor: 0.6...1.8, glideFor: 0.2...0.5, glideAngle: 0, glideExt: 0.9, handDroop: -0.05, sweep: 0.6,
                          reach: 4, separation: 0.9, cohesion: 0, alignment: 0.3, pull: 0, wander: 1.0, clearance: 8)
        case .bat: // never glides; jinks constantly
            return Flight(speed: 6, minSpeed: 3.5, maxSpeed: 9, span: 0.65, flapHz: 10, amp: 1.0, bias: 0,
                          flapFor: 2...4, glideFor: 0.05...0.1, glideAngle: 0, glideExt: 1, handDroop: -0.2, sweep: 0.25,
                          reach: 6, separation: 2.5, cohesion: 0.05, alignment: 0.1, pull: 0.12, wander: 14, clearance: 4)
        case .goose: // formation flyers: slots in a V behind the leader
            return Flight(speed: 11, minSpeed: 8, maxSpeed: 14, span: 2.0, flapHz: 3.2, amp: 0.55, bias: 0.05,
                          flapFor: 4...9, glideFor: 0.5...1.2, glideAngle: 0.05, glideExt: 1, handDroop: -0.1, sweep: 0.15,
                          reach: 0, separation: 1.6, cohesion: 0, alignment: 0, pull: 0, wander: 0.4, clearance: 15)
        }
    }

    final class Flock {
        let kind: Kind
        let slot: Int
        var birds: [Bird] = []
        var anchor: SIMD3<Double>
        var heading: Double
        var speed: Double
        var altitude: Double
        var presence = 0.0
        var age = 0.0
        let life: Double
        var leaving = false
        var leftFor = 0.0 // seconds spent leaving while still in view
        var fading = false
        var unseen = 0.0 // seconds since any bird was in frame
        let seedPhase: Double

        init(kind: Kind, slot: Int, anchor: SIMD3<Double>, heading: Double, speed: Double, altitude: Double, life: Double, seedPhase: Double) {
            self.kind = kind
            self.slot = slot
            self.anchor = anchor
            self.heading = heading
            self.speed = speed
            self.altitude = altitude
            self.life = life
            self.seedPhase = seedPhase
        }

        var dir: SIMD3<Double> { SIMD3(sin(heading), 0, cos(heading)) }
        // Flocks fade by headcount, not size: a far bird is held at a cell's footprint, so shrinking it hides nothing.
        var fadeTime: Double { kind == .starling ? 7 : 2.5 }
    }

    static let maxFlocks = 8
    static let maxBirds = 600
    // The shader culls by flock, then by cluster: runs of up to clusterSize birds under one bounding sphere, so a
    // pixel near a big murmuration tests a few clusters, not every bird. Must match MAX_FLOCKS and MAX_CLUSTERS in
    // Shaders/Saver.metal.
    static let maxClusters = 48, clusterSize = 16
    static let bufferLength = ((maxFlocks + maxClusters) * 2 + maxBirds * 4) * MemoryLayout<SIMD4<Float>>.stride

    private(set) var flocks: [Flock] = []
    private var state: UInt64
    private var time = 0.0
    private var spawnTimer = 0.0
    private var lastCam: SIMD3<Double>?
    private var camVel = SIMD3<Double>.zero
    private let noiseSeed: UInt32
    var heightAt: (Double, Double) -> Double? = { _, _ in nil }

    init(seed: UInt32) {
        state = UInt64(seed) &* 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03
        noiseSeed = seed &* 2_654_435_761
    }

    private func rand() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
    }

    private func rand(_ r: ClosedRange<Double>) -> Double { r.lowerBound + (r.upperBound - r.lowerBound) * rand() }

    // 2D gradient noise in [-1, 1]: the flow field flocks drift on.
    private func noise(_ x: Double, _ y: Double, _ s: UInt32) -> Double {
        let ix = Int32(floor(x)), iy = Int32(floor(y))
        let fx = x - floor(x), fy = y - floor(y)
        func grad(_ i: Int32, _ j: Int32, _ dx: Double, _ dy: Double) -> Double {
            var h = UInt32(bitPattern: i) &* 1_597_334_677 ^ UInt32(bitPattern: j) &* 3_812_015_801 ^ s
            h = (h ^ (h >> 16)) &* 0x7FEB_352D
            h = (h ^ (h >> 15)) &* 0x846C_A68B
            let a = Double(h & 0xFFFF) / 65536 * 2 * .pi
            return cos(a) * dx + sin(a) * dy
        }
        let u = fx * fx * fx * (fx * (fx * 6 - 15) + 10), v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        let a = grad(ix, iy, fx, fy), b = grad(ix + 1, iy, fx - 1, fy)
        let c = grad(ix, iy + 1, fx, fy - 1), d = grad(ix + 1, iy + 1, fx - 1, fy - 1)
        return ((a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v) * 1.414
    }

    private func ground(_ x: Double, _ z: Double) -> Double { heightAt(x, z) ?? 200 }

    // Which flocks the hour calls for, one per slot. Gull slots fall back to songbirds away from water.
    private static func roster(sunY: Double, phase: Double) -> [Kind] {
        if sunY < -0.16 { return [.bat] }
        let dusk = phase > 0.25 && phase < 0.75
        if sunY < 0.14 { return dusk ? [.starling, .songbird, .gull, .goose] : [.goose, .songbird, .gull, .songbird] }
        return [.songbird, .raptor, .gull, .songbird]
    }

    private func nearWater(_ p: SIMD2<Double>) -> SIMD2<Double>? {
        var best: SIMD2<Double>?, bestD = Double.infinity
        for r in stride(from: 0.0, through: 240, by: 40) {
            for k in 0..<(r == 0 ? 1 : 8) {
                let a = Double(k) * .pi / 4
                let q = p + SIMD2(sin(a), cos(a)) * r
                if ground(q.x, q.y) <= 20, r < bestD { best = q; bestD = r }
            }
            if best != nil { return best }
        }
        return best
    }

    func step(dt: Double, camera: FlightCamera, sunY: Double, phase: Double) {
        time += dt
        let roster = Wildlife.roster(sunY: sunY, phase: phase)
        let cam = camera.position
        let fwd = camera.forward
        let camYaw = atan2(fwd.x, fwd.z)
        if let l = lastCam, dt > 0 { camVel += ((cam - l) / dt - camVel) * min(1, dt) }
        lastCam = cam

        // Retire flocks the hour no longer calls for, and those far away or long gone behind. Visibility is judged
        // by the birds, not the anchor: a spread flock or a trailing skein reaches well past it. A flock is only
        // dropped out of view; one that lingers in view while leaving fades out by headcount first.
        flocks.removeAll { f in
            let want = f.slot < roster.count ? roster[f.slot] : nil
            if want == nil || !(want == f.kind || (want == .gull && f.kind == .songbird)) { f.leaving = true }
            if f.age > f.life { f.leaving = true }
            let seen = visible(f, camera: camera).count > 0
            f.unseen = seen ? 0 : f.unseen + dt
            let v = f.anchor - cam
            let dist = simd_length(SIMD2(v.x, v.z))
            let ahead = simd_dot(simd_normalize(SIMD2(v.x, v.z) + 1e-6), simd_normalize(SIMD2(fwd.x, fwd.z) + 1e-6))
            if dist > 750 || f.birds.isEmpty || (f.fading && f.presence <= 0) { return true }
            // Out by the map's edge a flock still in view fades before the hard limit above.
            if dist > 600 && seen { f.fading = true }
            // Crossed the frame and gone, or fallen behind: make room for the next.
            if (f.unseen > 5 && f.age > 12) || (ahead < -0.2 && dist > 120) { f.leaving = true }
            guard f.leaving else { return false }
            if !seen { return true }
            f.leftFor += dt
            // Staggered per flock, so flocks the hour retires together do not all dissolve at once.
            if f.leftFor > 15 + 20 * f.seedPhase.truncatingRemainder(dividingBy: 1) { f.fading = true }
            return false
        }
        spawnTimer -= dt
        if spawnTimer <= 0 {
            let used = Set(flocks.filter { !$0.leaving }.map(\.slot))
            if let slot = roster.indices.first(where: { !used.contains($0) }), flocks.count < Wildlife.maxFlocks {
                spawn(kind: roster[slot], slot: slot, cam: cam, camYaw: camYaw, pitch: asin(fwd.y))
                spawnTimer = 1.5
            }
        }
        for f in flocks { advance(f, dt: dt, cam: cam) }
    }

    // Birds inside a generous view frustum (a 16:10 frame and then some; terrain occlusion ignored) and the nearest.
    private func visible(_ f: Flock, camera: FlightCamera, aspect: Double = 1.8, tanHalf: Double = 0.52) -> (count: Int, nearest: Double) {
        let b = camera.basis
        var n = 0, near = Double.infinity
        for bird in f.birds {
            let v = bird.pos - camera.position
            let z = simd_dot(v, b.fwd)
            guard z > 1, z < 900 else { continue }
            if abs(simd_dot(v, b.right)) / z < tanHalf * aspect, abs(simd_dot(v, b.up)) / z < tanHalf {
                n += 1
                near = min(near, simd_length(v))
            }
        }
        return (n, near)
    }

    private func spawn(kind wanted: Kind, slot: Int, cam: SIMD3<Double>, camYaw: Double, pitch: Double) {
        var kind = wanted
        let roll = rand()
        // Distance band and bearing: crossing flocks are placed so that, moving relative to the camera, they enter
        // past a side edge of the frame (half-field ~0.62 rad) and cross its middle at distance r; distant ones may
        // start in frame and fade in.
        var r = 0.0, bearing = 0.0, crossing = false
        var p = SIMD2<Double>.zero, alt = 0.0
        var fl = Wildlife.flight(kind)
        let pace = rand(0.85...1.05)
        var heading = 0.0
        let c2 = SIMD2(cam.x, cam.z), cv = SIMD2(camVel.x, camVel.z)
        // A few tries for a spot the camera can see: hills between would hide the flock.
        for attempt in 0..<4 {
            kind = wanted
            switch kind {
            case .starling: r = rand(240...420); bearing = rand(-0.35...0.35); crossing = false
            case .raptor: r = rand(110...320); bearing = rand(-0.5...0.5); crossing = false
            case .bat: r = rand(25...60); bearing = rand(-0.4...0.4); crossing = false
            default:
                if roll < 0.3 { r = rand(45...90) } else if roll < 0.75 { r = rand(80...220) } else { r = rand(220...420) }
                crossing = r < 300
                bearing = crossing ? (rand() < 0.5 ? -1 : 1) * rand(0.0...0.25) : rand(-0.5...0.5)
            }
            let a = camYaw + bearing
            p = c2 + SIMD2(sin(a), cos(a)) * r
            if crossing {
                // Back along the flock's path relative to the camera from the crossing point, until past the edge.
                // A camera much faster than the flock may leave no side entry; it then starts ahead and fades in.
                let side: Double = bearing < 0 ? -1 : 1
                heading = camYaw - side * .pi / 2 + rand(-0.6...0.6)
                let w = SIMD2(sin(heading), cos(heading)) * Wildlife.flight(kind).speed * pace - cv
                let mid = p
                for k in 1...20 {
                    p = mid - w * (0.5 * Double(k))
                    let q = p - c2
                    let off = atan2(sin(atan2(q.x, q.y) - camYaw), cos(atan2(q.x, q.y) - camYaw))
                    if abs(off) > 0.72 { break }
                }
            }
            if kind == .gull {
                if let w = nearWater(p) { p = w; crossing = false } else { kind = .songbird }
            }
            fl = Wildlife.flight(kind)
            let g = ground(p.x, p.y)
            // Elevation inside the frame's vertical field (half-field 0.436 rad), mostly above the horizon line so
            // birds read against sky.
            let elev: Double
            switch kind {
            case .starling: elev = max(pitch, -0.05) + rand(0.08...0.22)
            case .bat: elev = max(pitch, -0.1) + rand(0.12...0.3)
            case .raptor: elev = pitch + rand(0.0...0.35)
            default: elev = pitch + rand(-0.15...0.32)
            }
            let minAlt = g + (kind == .gull ? 4 : 1) * fl.clearance + (kind == .raptor ? 25 : 0) + (kind == .starling ? 25 : 0)
            alt = max(cam.y + r * tan(elev), minAlt)
            // Raise it over the sightline: every point on the line from the camera clears the ground.
            for k in 1...10 {
                let sfrac = 0.1 + 0.09 * Double(k)
                let q = c2 + (p - c2) * sfrac
                alt = max(alt, cam.y + (ground(q.x, q.y) + 4 - cam.y) / sfrac)
            }
            if alt <= max(190, g + 120) || attempt == 3 { break }
        }
        // Under the cloud base, unless the ground itself rises past it.
        alt = max(min(alt, FlightCamera.ceiling - 20), min(alt, ground(p.x, p.y) + 2 * fl.clearance))
        if !crossing { heading = rand(0...(2 * .pi)) }
        let speed: Double
        switch kind {
        case .starling: speed = 1.5
        case .raptor: speed = rand(0.8...2.0)
        case .bat: speed = 2.5
        default: speed = fl.speed * pace
        }
        let n: Int
        switch kind {
        case .songbird: n = Int(rand(5...22))
        case .raptor: n = Int(rand(1...3.99))
        case .gull: n = Int(rand(3...11))
        case .starling: n = Int(rand(320...420))
        case .bat: n = Int(rand(3...7))
        case .goose: n = Int(rand(7...15)) | 1
        }
        let f = Flock(kind: kind, slot: slot, anchor: SIMD3(p.x, alt, p.y), heading: heading, speed: speed, altitude: alt,
                      life: rand(70...150), seedPhase: rand(0...1000))
        let spread: Double = kind == .starling ? 18 : (kind == .raptor ? 25 : 6 + Double(n) * 0.4)
        let budget = Wildlife.maxBirds - flocks.reduce(0) { $0 + $1.birds.count }
        for i in 0..<min(n, budget) {
            var off = SIMD3(rand(-1...1), rand(-0.4...0.4), rand(-1...1)) * spread
            if kind == .goose { off = slotOffset(i, heading: heading, span: fl.span) }
            let v = SIMD3(sin(heading), 0, cos(heading)) * fl.speed + SIMD3(rand(-1...1), rand(-0.3...0.3), rand(-1...1))
            var b = Bird(pos: f.anchor + off, vel: v, size: fl.span * rand(0.88...1.12), tint: rand(0...1), phase: rand(0...1))
            if kind == .starling {
                // Denser toward the sheet's middle.
                let ang = rand(0...(2 * .pi)), rad = pow(rand(), 0.7)
                b.sheet = SIMD2(cos(ang), sin(ang)) * rad
                b.pos = sheetPoint(f, b.sheet, b.phase)
            }
            b.flap = rand(0...(2 * .pi))
            b.beat = rand() < 0.5 ? rand(fl.flapFor) : -rand(fl.glideFor)
            f.birds.append(b)
        }
        if kind == .starling {
            // Neighbours on the sheet stay neighbours in the air: order by strips of the sheet so each cluster of
            // consecutive birds is a compact patch.
            let strips = Double(max(1, f.birds.count / (Wildlife.clusterSize * 4)))
            f.birds.sort { a, b in
                let sa = floor((a.sheet.x + 1) / 2 * strips), sb = floor((b.sheet.x + 1) / 2 * strips)
                return sa != sb ? sa < sb : a.sheet.y < b.sheet.y
            }
        }
        flocks.append(f)
    }

    // Murmuration: birds hold places on a thin sheet that turns, tilts, stretches and ripples. Edge-on it reads as a
    // dense dark band, face-on as a thin veil, so the cloud pulses in density as it wheels.
    private func sheetPoint(_ f: Flock, _ uv: SIMD2<Double>, _ ph: Double, dt: Double = 0) -> SIMD3<Double> {
        let t = time + dt + f.seedPhase
        let lx = 30 + 12 * sin(t * 0.11), lz = 15 + 7 * sin(t * 0.17 + 1)
        let wave = 7 * sin(uv.x * 2.6 - t * 0.9) + 4 * sin(uv.y * 3.1 + t * 0.7) + 2.5 * sin((uv.x + uv.y) * 5 + t * 1.3)
        let fold = 10 * sin(t * 0.07) * uv.x * uv.x
        let qy: Double = wave + fold + (ph - 0.5) * 2.5
        var q = SIMD3(uv.x * lx, qy, uv.y * lz)
        let tilt = 0.7 * sin(t * 0.13) + 0.3 * sin(t * 0.31)
        let yaw = t * 0.16 + 0.8 * sin(t * 0.05)
        q = SIMD3(q.x, q.y * cos(tilt) - q.z * sin(tilt), q.y * sin(tilt) + q.z * cos(tilt))
        q = SIMD3(q.x * cos(yaw) + q.z * sin(yaw), q.y, -q.x * sin(yaw) + q.z * cos(yaw))
        return f.anchor + q + f.dir * f.speed * dt
    }

    // Goose skein: the leader at the point, the rest alternately down either arm of the V.
    private func slotOffset(_ i: Int, heading: Double, span: Double) -> SIMD3<Double> {
        if i == 0 { return .zero }
        let rank = Double((i + 1) / 2), side: Double = i % 2 == 1 ? 1 : -1
        let f = SIMD3(sin(heading), 0, cos(heading)), r = SIMD3(cos(heading), 0, -sin(heading))
        return -f * rank * 1.5 * span + r * side * rank * 1.25 * span + SIMD3(0, -0.15 * rank, 0)
    }

    private func advance(_ f: Flock, dt: Double, cam: SIMD3<Double>) {
        let fl = Wildlife.flight(f.kind)
        f.age += dt
        f.presence = max(0, min(1, f.presence + (f.fading ? -dt : dt) / f.fadeTime))

        // Anchor: a heading that drifts on the noise flow field, bends down valleys (toward the lower side ahead),
        // and for gulls toward water. Leaving flocks just hold course.
        let ap = SIMD2(f.anchor.x, f.anchor.z)
        if !f.leaving && f.kind != .goose {
            var turn = 0.35 * noise(ap.x / 400 + f.seedPhase, time * 0.02, noiseSeed)
            let fd = SIMD2(sin(f.heading), cos(f.heading)), side = SIMD2(fd.y, -fd.x)
            let look = ap + fd * 60
            let hl = ground(look.x - side.x * 30, look.y - side.y * 30), hr = ground(look.x + side.x * 30, look.y + side.y * 30)
            turn += max(-0.25, min(0.25, (hl - hr) * 0.006))
            if f.kind == .gull {
                let wl = hl <= 20 ? 1.0 : 0.0, wr = hr <= 20 ? 1.0 : 0.0, wa = ground(look.x, look.y) <= 20 ? 1.0 : 0.0
                // Shorelines: over land, turn toward the water side; over open water, toward the land side.
                turn += wa > 0 ? 0.12 * (wl - wr) : 0.3 * (wr - wl)
            }
            f.heading += turn * dt
        }
        if f.kind == .starling || f.kind == .bat {
            // These hold to one spot: the anchor circles slowly about where it was spawned.
            f.heading += 0.12 * dt
        }
        if f.kind == .starling {
            // The murmuration drifts clear of the camera's path: flown through, it is hundreds of boxes filling the
            // frame (and the GPU's worst case).
            let d = SIMD2(f.anchor.x - cam.x, f.anchor.z - cam.z), l = simd_length(d)
            if l < 110 {
                let cv = SIMD2(camVel.x, camVel.z)
                let cd = simd_length(cv) > 0.1 ? simd_normalize(cv) : SIMD2<Double>.zero
                var away = d - cd * simd_dot(d, cd)
                if simd_length(away) < 1 { away = SIMD2(cd.y, -cd.x) }
                let push = simd_normalize(away) * 18 * (1 - l / 110) * dt
                f.anchor.x += push.x
                f.anchor.z += push.y
            }
        }
        let vA = f.dir * f.speed
        f.anchor += vA * dt
        var gAhead = ground(f.anchor.x, f.anchor.z)
        for d in [20.0, 45.0, 80.0] {
            gAhead = max(gAhead, ground(f.anchor.x + vA.x / max(f.speed, 0.1) * d, f.anchor.z + vA.z / max(f.speed, 0.1) * d) - d * 0.15)
        }
        // Ground-relative: flocks rise with the land (over a wall, before it) and otherwise keep under the clouds.
        let lift = gAhead + fl.clearance * (f.kind == .raptor ? 2 : 1.5)
        let yT = max(min(f.altitude, FlightCamera.ceiling - 20), lift)
        f.anchor.y += (yT - f.anchor.y) * min(1, dt * (yT > f.anchor.y && lift > f.anchor.y - 5 ? 1.5 : 0.4))

        // Murmuration: a falcon-like disturbance loops through the flock, so it folds, splits and pulses.
        let hawk = f.anchor + SIMD3(sin(time * 0.31 + f.seedPhase) * 28, sin(time * 0.53) * 9, sin(time * 0.23 + 1.7) * 28)

        let count = f.birds.count
        for i in 0..<count {
            var b = f.birds[i]
            var acc = SIMD3<Double>.zero
            let speedNow = simd_length(b.vel)
            let vdir = b.vel / max(speedNow, 1e-3)
            if fl.reach > 0 || fl.separation > 0 {
                var sumP = SIMD3<Double>.zero, sumV = SIMD3<Double>.zero, nb = 0.0
                let r2 = fl.reach * fl.reach, s2 = fl.separation * fl.separation
                // Starlings heed a fixed handful of neighbours (stride sampling keeps a 200-bird flock cheap).
                let stride = f.kind == .starling ? 7 : 1
                var j = f.kind == .starling ? (i + Int(time * 60)) % stride : 0
                while j < count {
                    if j != i {
                        let dv = f.birds[j].pos - b.pos
                        let d2 = simd_length_squared(dv)
                        if d2 < r2 { sumP += f.birds[j].pos; sumV += f.birds[j].vel; nb += 1 }
                        if d2 < s2 { acc -= dv / max(d2, 0.05) * fl.separation * 2.0 }
                    }
                    j += stride
                }
                if nb > 0 {
                    acc += (sumP / nb - b.pos) * fl.cohesion + (sumV / nb - b.vel) * fl.alignment
                }
            }
            let toA = f.anchor - b.pos
            switch f.kind {
            case .raptor:
                // Thermal soaring: each bird circles its own ring about the anchor, rising and sinking slowly.
                let th = time * (0.22 + 0.05 * b.phase) + b.phase * 2 * .pi
                let ring = 16 + 18 * b.phase
                let target = f.anchor + SIMD3(cos(th) * ring, 6 * sin(time * 0.05 + b.phase * 6), sin(th) * ring)
                let tangent = SIMD3(-sin(th), 0, cos(th)) * fl.speed
                acc += (tangent + (target - b.pos) * 0.3 - b.vel) * 0.9
            case .goose:
                let target = i == 0 ? f.anchor : f.birds[0].pos + slotOffset(i, heading: f.heading, span: fl.span)
                acc += (target - b.pos) * 0.8 + (vA - b.vel) * 1.2
            case .starling:
                let target = sheetPoint(f, b.sheet, b.phase)
                let ahead = sheetPoint(f, b.sheet, b.phase, dt: 0.1)
                let sheetVel: SIMD3<Double> = (ahead - target) / 0.1
                acc += (target - b.pos) * 1.6
                acc += (sheetVel - b.vel) * 1.2
                let dh = b.pos - hawk, dl = simd_length(dh)
                if dl < 12 { acc += dh / max(dl, 0.5) * 40 * (1 - dl / 12) }
            default:
                acc += toA * fl.pull + (vA - b.vel) * 0.15
            }
            if f.leaving { acc += f.dir * 1.5 }
            // Birds shy from the camera: within a few units they read as boxes, not birds.
            // The push is across the camera's path: birds are slower than it and cannot outrun it ahead.
            let dc = b.pos - cam, lc = simd_length(dc)
            let shy = f.kind == .bat ? 10.0 : 40.0
            if lc < shy {
                let cd = simd_length(camVel) > 0.1 ? simd_normalize(camVel) : SIMD3<Double>.zero
                var away = dc - cd * simd_dot(dc, cd)
                if simd_length(away) < 1 { away = SIMD3(0, 1, 0) }
                acc += simd_normalize(away) * 70 * (1 - lc / shy)
            }
            let wt = time * (f.kind == .bat ? 1.7 : 0.45)
            let ph = b.phase * 97
            let wx: Double = sin(wt * 1.13 + ph) + 0.5 * sin(wt * 2.71 + ph * 1.7)
            let wy: Double = 0.35 * sin(wt * 0.87 + ph * 2.3)
            let wz: Double = sin(wt * 0.97 + ph * 3.1) + 0.5 * sin(wt * 2.29 + ph * 0.7)
            acc += SIMD3(wx, wy, wz) * fl.wander
            // Terrain: climb away from ground ahead, and keep well under the cloud base.
            let look = b.pos + b.vel * 1.2
            let g = max(ground(b.pos.x, b.pos.z), ground(look.x, look.z))
            let clear = fl.clearance
            if b.pos.y < g + clear { acc.y += (g + clear - b.pos.y) * 4 }
            if b.pos.y > 215 { acc.y -= (b.pos.y - 215) * 2 }

            b.vel += acc * dt
            var s = simd_length(b.vel)
            if s > fl.maxSpeed { b.vel *= fl.maxSpeed / s; s = fl.maxSpeed }
            if s < fl.minSpeed { b.vel = (s > 1e-3 ? b.vel / s : SIMD3(sin(f.heading), 0, cos(f.heading))) * fl.minSpeed; s = fl.minSpeed }
            // Birds climb and dive at modest angles.
            let maxY = 0.45 * s
            b.vel.y = max(-maxY, min(maxY, b.vel.y))
            b.pos += b.vel * dt
            b.pos.y = max(b.pos.y, ground(b.pos.x, b.pos.z) + 1.5)

            // Wingbeats in bursts between glides; a climbing bird keeps flapping.
            if b.beat > 0 {
                b.flap += 2 * .pi * fl.flapHz * (0.9 + 0.2 * b.phase) * dt
                b.beat -= dt
                if b.beat <= 0 && b.vel.y < 0.2 * s { b.beat = -rand(fl.glideFor) }
                if b.beat <= 0 && b.vel.y >= 0.2 * s { b.beat = 0.3 }
            } else {
                b.beat += dt
                if b.beat >= 0 { b.beat = rand(fl.flapFor) }
            }
            // Bank into the turn by the sideways acceleration.
            let side = simd_normalize(simd_cross(SIMD3<Double>(0, 1, 0), vdir) + SIMD3(1e-6, 0, 0))
            let lat = simd_dot(acc, side)
            let bankT = max(-0.9, min(0.9, -atan(lat / 9.8) * 1.4))
            b.bank += (bankT - b.bank) * min(1, dt * 3)
            f.birds[i] = b
        }
    }

    // Arm and hand angles (up from level), extension and sweep for a bird this instant. On the beat the hand lags
    // the arm, so the wing whips rather than hinging as one plank.
    private func wing(_ b: Bird, _ fl: Flight) -> (arm: Double, hand: Double, ext: Double, sweep: Double) {
        if b.beat > 0 {
            let arm = fl.bias + fl.amp * sin(b.flap)
            return (arm, arm + 0.55 * fl.amp * sin(b.flap - 1.1), 1, fl.sweep * (0.75 + 0.25 * cos(b.flap)))
        }
        let folded = 1 - fl.glideExt
        return (fl.glideAngle, fl.glideAngle + fl.handDroop, fl.glideExt, fl.sweep + 0.6 * folded)
    }

    // Writes cluster spheres then birds into `buffer`, map-local (relative to `origin`). Returns the cluster count.
    func upload(into buffer: MTLBuffer, origin o: SIMD3<Double>) -> Int {
        let out = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: Wildlife.bufferLength / 16)
        var nc = 0, nb = 0
        let clusters = Wildlife.maxFlocks * 2
        let birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2
        var nf = 0
        for f in flocks where nf < Wildlife.maxFlocks {
            let fl = Wildlife.flight(f.kind)
            let grow = f.presence * f.presence * (3 - 2 * f.presence)
            let firstCluster = nc
            var flo = SIMD3<Double>(repeating: .infinity), fhi = SIMD3<Double>(repeating: -.infinity)
            for start in stride(from: 0, to: f.birds.count, by: Wildlife.clusterSize) where nc < Wildlife.maxClusters {
                let first = nb
                var lo = SIMD3<Double>(repeating: .infinity), hi = SIMD3<Double>(repeating: -.infinity)
                for i in start..<min(start + Wildlife.clusterSize, f.birds.count) where nb < Wildlife.maxBirds {
                    // Golden-ratio order spreads each step of the fade evenly across the flock.
                    if (Double(i) * 0.618034).truncatingRemainder(dividingBy: 1) >= grow { continue }
                    let b = f.birds[i]
                    let p = b.pos - o
                    lo = simd_min(lo, p); hi = simd_max(hi, p)
                    let v = simd_normalize(b.vel)
                    let w = wing(b, fl)
                    let base = birds + nb * 4
                    out[base] = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(b.size))
                    out[base + 1] = SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w.arm))
                    out[base + 2] = SIMD4(Float(b.bank), Float(f.kind.rawValue), Float(b.tint), Float(w.ext))
                    out[base + 3] = SIMD4(Float(w.hand), Float(w.sweep), 0, 0)
                    nb += 1
                }
                if nb == first { continue }
                flo = simd_min(flo, lo); fhi = simd_max(fhi, hi)
                let c = (lo + hi) / 2
                let r = simd_length(hi - lo) / 2 + fl.span
                out[clusters + nc * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
                out[clusters + nc * 2 + 1] = SIMD4(Float(first), Float(nb - first), 0, 0)
                nc += 1
            }
            if nc == firstCluster { continue }
            let c = (flo + fhi) / 2
            let r = simd_length(fhi - flo) / 2 + fl.span
            out[nf * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
            out[nf * 2 + 1] = SIMD4(Float(firstCluster), Float(nc - firstCluster), 0, 0)
            nf += 1
        }
        return nf
    }

    // Debug: replaces the flocks with one murmuration `distance` ahead of the camera, for timing the worst case.
    func testMurmuration(camera: FlightCamera, distance: Double) {
        flocks = []
        let fw = camera.forward
        let saved = spawnTimer
        spawn(kind: .starling, slot: 0, cam: camera.position, camYaw: atan2(fw.x, fw.z), pitch: asin(fw.y))
        spawnTimer = saved
        guard let f = flocks.first else { return }
        let target = camera.position + fw * distance
        let shift = target - f.anchor
        f.anchor += shift
        f.presence = 1
        for i in f.birds.indices { f.birds[i].pos += shift }
    }

    // Debug: replaces the flocks with a grid of birds of every kind close in front of the camera, at varied headings
    // and wingbeat phases, for judging their shapes.
    func testPattern(camera: FlightCamera, distance: Double) {
        let b = camera.basis
        let kinds: [Kind] = [.songbird, .raptor, .gull, .starling, .bat, .goose]
        flocks = kinds.map { Flock(kind: $0, slot: 99, anchor: camera.position, heading: 0, speed: 0, altitude: 0, life: 1e9, seedPhase: 0) }
        for f in flocks { f.presence = 1 }
        for row in 0..<3 {
            for col in 0..<6 {
                let k = kinds[col]
                let f = flocks[col]
                let fl = Wildlife.flight(k)
                let p = camera.position + b.fwd * distance + b.right * (Double(col) - 2.5) * distance * 0.22
                    + b.up * (Double(row) - 1) * distance * 0.22
                let yaw = Double(row) * 2.1 + Double(col) * 0.4
                var bird = Bird(pos: p, vel: SIMD3(sin(yaw), 0.1 * Double(row - 1), cos(yaw)) * fl.speed, size: fl.span, tint: 0.5, phase: 0.5)
                bird.beat = row == 1 ? -1 : 1
                bird.flap = Double(row) * 2.0 + Double(col)
                f.birds.append(bird)
            }
        }
    }

    // Birds inside the camera's view (ignoring terrain occlusion), and the nearest one's distance.
    func inView(camera: FlightCamera, aspect: Double, tanHalf: Double) -> (count: Int, nearest: Double, flocks: Int) {
        var n = 0, near = Double.infinity, fs = 0
        for f in flocks {
            let v = visible(f, camera: camera, aspect: aspect, tanHalf: tanHalf)
            n += v.count
            near = min(near, v.nearest)
            if v.count > 0 { fs += 1 }
        }
        return (n, near, fs)
    }
}
