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
        var id = 0
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
            return Flight(speed: 12, minSpeed: 4, maxSpeed: 22, span: 0.7, flapHz: 8, amp: 0.8, bias: 0.05,
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
        var age = 0.0
        let life: Double
        var leaving = false
        var unseen = 0.0 // seconds since any bird was in frame
        // Seconds until the flock, flying in from where nothing could see it, reaches the part of the frame it was
        // aimed at; until then it holds its course and speed.
        var transit = 0.0
        var cruise = 0.0
        var climb = 0.0 // while flying in
        // Speed once arrived, for kinds that then hold to one spot.
        var hold: Double?
        // Murmuration: a slow drift along with the camera, so it stays in the frame while it gathers and wheels.
        var drift = SIMD3<Double>.zero
        let seedPhase: Double
        var serial = 0
        var misses = 0 // murmuration: tries in a row to gather again ahead that found no place
        // Circling an island: its centre (x, z), the ring's radius and height, and which way round.
        var orbit: SIMD4<Double>?
        var orbitSense = 1.0

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
    private var nextId = 1
    private var spawnTurn = 0
    private var lastCam: SIMD3<Double>?
    private var camVel = SIMD3<Double>.zero
    private var lastLook: (yaw: Double, pitch: Double)?
    private var camYawRate = 0.0, camPitchRate = 0.0
    // What the frame can show this step; set by step().
    private var sight: Sight?
    private var hiddenFrames: [(Sight, Double)] = []
    // Spawn attempts that found no place out of sight, and those that did, by kind; for VR_POPLOG.
    private(set) var spawnStats: [String: (tries: Int, made: Int, transit: Double)] = [:]
    private(set) var failWhy: [String: Int] = [:]
    private(set) var spawnMs = 0.0
    private let noiseSeed: UInt32
    var heightAt: (Double, Double) -> Double? = { _, _ in nil }
    // The CPU cell under world (x, z) with its islands (World.column): top, ground islands aside, lowest underside.
    var columnAt: (Double, Double) -> (top: Double, ground: Double, under: Double?)? = { _, _ in nil }
    // Giant trees on the loaded map (world coordinates).
    var giantsAt: () -> [GiantInfo] = { [] }
    // Floating islands round the camera.
    var islesAt: () -> [IslandInfo] = { [] }

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
    // The ground as a bird at height y meets it: well under an island, the land beneath it; else the island's top.
    private func ground(_ x: Double, _ z: Double, at y: Double) -> Double {
        guard let c = columnAt(x, z) else { return 200 }
        if let u = c.under, y < u - 12 { return c.ground }
        return c.top
    }
    // The land with every island lifted away.
    private func groundBare(_ x: Double, _ z: Double) -> Double { columnAt(x, z)?.ground ?? 200 }
    // Highest a flock flies over ground g: under the cloud base, unless the ground itself rises past it.
    private func topAlt(over g: Double, clearance c: Double) -> Double {
        max(FlightCamera.ceiling - 20, g + 2 * c)
    }

    // Which flocks the hour calls for, one per slot. Gull slots fall back to songbirds away from water.
    // Among floating islands raptors take a songbird's slot: they circle the islands.
    private static func roster(sunY: Double, phase: Double, isles: Bool = false) -> [Kind] {
        if sunY < -0.16 { return [.bat, .bat, .bat] }
        let dusk = phase > 0.25 && phase < 0.75
        // Flocks spend a while flying in out of sight, so the hour keeps more of them on the wing.
        if sunY < 0.14 { return dusk ? [.starling, .songbird, .gull, .goose, .songbird, .songbird] : [.goose, .songbird, .gull, .songbird, .songbird, .goose] }
        return [.songbird, .raptor, .gull, isles ? .raptor : .songbird, .songbird, .gull, .songbird]
    }

    private func nearWater(_ p: SIMD2<Double>) -> SIMD2<Double>? {
        var best: SIMD2<Double>?, bestD = Double.infinity
        for r in stride(from: 0.0, through: 240, by: 40) {
            for k in 0..<(r == 0 ? 1 : 8) {
                let a = Double(k) * .pi / 4
                let q = p + SIMD2(sin(a), cos(a)) * r
                if groundBare(q.x, q.y) <= 20, r < bestD { best = q; bestD = r }
            }
            if best != nil { return best }
        }
        return best
    }

    func step(dt: Double, camera: FlightCamera, sight: Sight, sunY: Double, phase: Double) {
        time += dt
        // The last frame drew the birds where they are now, from where the camera was then.
        var sight = sight
        sight.slack = 2
        let shown = self.sight ?? sight
        self.sight = sight
        hiddenFrames = []
        let cam = camera.position
        let cam2 = SIMD2(cam.x, cam.z)
        let roster = Wildlife.roster(sunY: sunY, phase: phase, isles: islesAt().contains { $0.radius >= 40 && simd_length($0.pos - cam2) < $0.reach + 1500 })
        let fwd = camera.forward
        let camYaw = atan2(fwd.x, fwd.z), camPitch = asin(max(-1, min(1, fwd.y)))
        if let l = lastCam, dt > 0 { camVel += ((cam - l) / dt - camVel) * min(1, dt) }
        lastCam = cam
        if let l = lastLook, dt > 0 {
            camYawRate += (atan2(sin(camYaw - l.yaw), cos(camYaw - l.yaw)) / dt - camYawRate) * min(1, dt * 2)
            camPitchRate += ((camPitch - l.pitch) / dt - camPitchRate) * min(1, dt * 2)
        }
        lastLook = (camYaw, camPitch)

        // Retire flocks the hour no longer calls for, and those long gone from the frame, far off or fallen behind.
        // A leaving flock turns out of the frame and is dropped only once no frame could show any of its birds.
        flocks.removeAll { f in
            if f.birds.isEmpty { return true }
            let want = f.slot < roster.count ? roster[f.slot] : nil
            if want == nil || !(want == f.kind || (want == .gull && f.kind == .songbird)) { f.leaving = true }
            if f.age > f.life { f.leaving = true }
            let framed = f.birds.contains { sight.inFrame($0.pos, size: $0.size) }
            f.unseen = framed ? 0 : f.unseen + dt
            let v = f.anchor - cam
            let dist = simd_length(SIMD2(v.x, v.z))
            let ahead = simd_dot(simd_normalize(SIMD2(v.x, v.z) + 1e-6), simd_normalize(SIMD2(fwd.x, fwd.z) + 1e-6))
            let lost = dist > 750 || (f.transit <= 0 && ((f.unseen > 5 && f.age > 12) || (ahead < -0.2 && dist > 120)))
            // A murmuration the camera has turned or flown away from gathers again ahead, its birds streaming back
            // in from wherever they are out of sight; only if no place ahead will do does it leave.
            if f.kind == .starling, !f.leaving, f.unseen > 4, lost || !sight.inFrame(f.anchor, size: 60), floor(f.unseen) != floor(f.unseen - dt) {
                // Still on show, mirrored in water: not lost yet.
                if f.birds.contains(where: { sight.sees($0.pos, size: $0.size) }) {
                    f.misses = 0
                } else if regather(f, cam: cam, camYaw: camYaw, pitch: camPitch) {
                    f.misses = 0
                    return false
                } else {
                    f.misses += 1
                }
            }
            if lost && (f.kind != .starling || f.misses >= 8 || f.unseen > 60) { f.leaving = true }
            guard f.leaving else { return false }
            return !f.birds.contains { shown.sees($0.pos, size: $0.size, margin: 0.03) || sight.sees($0.pos, size: $0.size, margin: 0.03) }
        }
        spawnTimer -= dt
        if spawnTimer <= 0 {
            // No place out of sight that would bring the flock into the frame: look again in a second.
            spawnTimer = 1.0
            // Empty slots take turns, so one kind with no way in does not hold up the others.
            let used = Set(flocks.filter { !$0.leaving }.map(\.slot))
            let empty = roster.indices.filter { !used.contains($0) }
            if !empty.isEmpty, flocks.count >= Wildlife.maxFlocks { failWhy["full", default: 0] += 1 }
            if !empty.isEmpty, flocks.count < Wildlife.maxFlocks {
                let slot = empty[spawnTurn % empty.count]
                spawnTurn += 1
                spawnStats["\(roster[slot])", default: (0, 0, 0)].tries += 1
                let t0 = PopLog.enabled ? ProcessInfo.processInfo.systemUptime : 0
                let made = spawn(kind: roster[slot], slot: slot, cam: cam, camYaw: camYaw, pitch: camPitch)
                if PopLog.enabled { spawnMs = max(spawnMs, (ProcessInfo.processInfo.systemUptime - t0) * 1000) }
                if made {
                    spawnTimer = 1.5
                    spawnStats["\(roster[slot])", default: (0, 0, 0)].made += 1
                    spawnStats["\(roster[slot])", default: (0, 0, 0)].transit += max(0, flocks.last!.transit)
                }
            }
        }
        for f in flocks { advance(f, dt: dt, cam: cam) }
        if Wildlife.trace, floor(time) != floor(time - dt) {
            let desc = flocks.map { f -> String in
                let framed = f.birds.filter { sight.inFrame($0.pos, size: $0.size) }.count
                let d = simd_length(f.anchor - cam)
                return String(format: "%@%@ n=%d framed=%d d=%.0f tr=%.0f", "\(f.kind)", f.leaving ? "(L)" : "", f.birds.count, framed, d, max(0, f.transit))
            }
            print(String(format: "TRACE t=%.0f sunY=%.2f ", time, sunY) + desc.joined(separator: " | "))
        }
    }
    static var trace: Bool { Knobs.current.birdTrace }

    // Birds inside a generous view frustum (terrain occlusion ignored) and the nearest; for the snapshot report.
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

    // The frame `tau` seconds on, if the camera holds its velocity, its roll and its turn and pitch rates (those for at
    // most `turnFor` seconds: the shot director's turns ease out within a few).
    private func predicted(_ s: Sight, tau: Double, turnFor: Double = 3) -> Sight {
        var p = s
        let t = min(tau, turnFor)
        let yawQ = simd_quatd(angle: camYawRate * t, axis: SIMD3(0, 1, 0))
        let right = yawQ.act(s.right)
        let q = simd_quatd(angle: -camPitchRate * t, axis: right) * yawQ
        p.fwd = q.act(s.fwd)
        p.right = q.act(s.right)
        p.up = q.act(s.up)
        p.pos = s.pos + camVel * tau
        return p
    }

    // No frame from now to two seconds on could show any of these points. The camera's turn (at most 0.09 rad/s),
    // pitch and climb are carried forward; the margin grows for their rates changing (yaw by at most 0.011 rad/s^2,
    // pitch as its target moves) and roll easing (at most 0.11 rad/s).
    private func hidden(_ pts: [SIMD3<Double>], size: Double) -> Bool {
        guard let s = sight else { return true }
        if hiddenFrames.isEmpty { hiddenFrames = [0.0, 1.0, 2.0].map { (predicted(s, tau: $0), 0.04 + 0.03 * $0) } }
        let frames = hiddenFrames
        for (f, m) in frames {
            for p in pts where f.sees(p, size: size, margin: m) { return false }
        }
        return true
    }

    // Height for a flock at `p` seen from `c`: `elev` up from the camera, clear of the ground, and above the sightline
    // so the hills between do not hide it.
    private func flightLevel(_ kind: Kind, at p: SIMD2<Double>, from c: SIMD3<Double>, elev: Double) -> Double {
        let fl = Wildlife.flight(kind)
        let c2 = SIMD2(c.x, c.z)
        let r = simd_length(p - c2)
        let minAlt = groundBare(p.x, p.y) + (kind == .gull ? 4 : 1) * fl.clearance + (kind == .raptor ? 25 : 0) + (kind == .starling ? 25 : 0)
        var alt = max(c.y + r * tan(elev), minAlt)
        for k in 1...10 {
            let sfrac = 0.1 + 0.09 * Double(k)
            let q = c2 + (p - c2) * sfrac
            alt = max(alt, c.y + (groundBare(q.x, q.y) + 4 - c.y) / sfrac)
        }
        // An island there: under it with room to spare, else over it.
        if let col = columnAt(p.x, p.y), let u = col.under, alt > u - 20 - fl.clearance {
            alt = u - 20 - fl.clearance >= minAlt ? u - 20 - fl.clearance : max(alt, col.top + fl.clearance)
        }
        return alt
    }

    private func elevation(_ kind: Kind, pitch: Double) -> Double {
        // Inside the frame's vertical field (half-field 0.436 rad), mostly above the horizon so birds read against sky.
        switch kind {
        case .starling: return max(pitch, -0.05) + rand(0.08...0.22)
        case .bat: return max(pitch, -0.1) + rand(0.12...0.3)
        case .raptor: return pitch + rand(0.0...0.35)
        default: return pitch + rand(-0.15...0.32)
        }
    }

    // Where advance() would put a bird at p: never closer to the ground than it allows.
    private func aboveGround(_ p: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3(p.x, max(p.y, ground(p.x, p.z, at: p.y) + 2), p.z)
    }

    // Room left in the bird buffer, so the renderer's caps never drop a bird that is drawn.
    private var room: Int {
        let birds = flocks.reduce(0) { $0 + $1.birds.count }
        let clusters = flocks.reduce(0) { $0 + ($1.birds.count + Wildlife.clusterSize - 1) / Wildlife.clusterSize }
        return max(0, min(Wildlife.maxBirds - birds, (Wildlife.maxClusters - clusters) * Wildlife.clusterSize))
    }

    // A flock comes into being only where no frame could show it: just beyond an edge of the frame, or in the frame
    // but behind a hill. Of the paths from there it takes one that brings it into the frame soon, judged against where
    // the camera will be, and keeps it there a while: against a camera faster than the birds that is mostly a flock
    // flying along with it, drifting in across a side, gliding down past the top edge or rising over a ridge. Where
    // no such path exists, the flock waits.
    private func spawn(kind wanted: Kind, slot: Int, cam: SIMD3<Double>, camYaw: Double, pitch: Double) -> Bool {
        if wanted == .starling { return spawnMurmuration(slot: slot, cam: cam, camYaw: camYaw, pitch: pitch) }
        guard let s = sight else { return false }
        let roll = rand()
        let pace = rand(0.85...1.05)
        let life = rand(70...150), seedPhase = rand(0...1000)
        let count: Int
        switch wanted {
        case .songbird: count = Int(rand(5...22))
        case .raptor: count = Int(rand(1...3.99))
        case .gull: count = Int(rand(3...11))
        case .starling: count = 0
        case .bat: count = Int(rand(3...7))
        case .goose: count = Int(rand(7...15)) | 1
        }
        let unit = (0..<count).map { _ in SIMD3(rand(-1...1), rand(-0.4...0.4), rand(-1...1)) }
        let hold = rand(0.8...2.0)
        let n = min(unit.count, room)
        guard n > 0 else { return false }
        let tanX = s.tanV * s.aspect
        // The frames to come, every half second for 30 s.
        let ahead = (1...60).map { predicted(s, tau: Double($0) * 0.5, turnFor: 6) }
        for _ in 0..<12 {
            var kind = wanted
            let r: Double
            switch kind {
            case .raptor: r = rand(110...320)
            case .bat: r = rand(40...140)
            default: r = roll < 0.3 ? rand(45...90) : (roll < 0.75 ? rand(80...220) : rand(220...420))
            }
            let edge = rand(), along = rand(-0.9...0.9), low = rand(-0.9...0.1), base = rand(0...(2 * .pi)), pick = rand()
            var fl = Wildlife.flight(kind)
            // Flying in they go near their top speed, so some can keep pace with the camera; then raptors circle, bats
            // jink about where they arrived, and the rest settle to their own pace and the camera overtakes them.
            let cruise = fl.maxSpeed * (0.92 + 0.08 * pace)
            let spread: Double = kind == .raptor ? 25 : 6 + Double(n) * 0.4
            let size = fl.span * 1.12
            // The start: out past a side or the top of the frame, or (one in five) low inside it, hoping for a hill
            // in between. Angles are as tangents in the frame: x/z and y/z.
            let out = 0.11 + (spread + 0.62 * size) / r + 1.5 * s.pixelAngle
            let behindHill = edge >= 0.8
            var ax = atan(along * tanX), ay = atan(along * 0.8 * s.tanV)
            if edge < 0.3 { ax = -atan(tanX) - out } else if edge < 0.6 { ax = atan(tanX) + out } else if edge < 0.8 {
                ay = atan(s.tanV) + out
            } else { ay = atan(low * s.tanV) }
            let d = simd_normalize(s.fwd + s.right * tan(ax) + s.up * tan(ay))
            var start = s.pos + d * r
            let floorY = ground(start.x, start.z) + fl.clearance * 1.5
            if behindHill { start.y = floorY } else if start.y < floorY { failWhy["ground\(edge < 0.6 ? "S" : "T")", default: 0] += 1; continue }
            if start.y > topAlt(over: ground(start.x, start.z), clearance: fl.clearance) { failWhy["high", default: 0] += 1; continue }
            // Paths: twelve headings, level, gliding down or climbing; scored by how soon the anchor is in the frame
            // and how long it stays.
            var best: [(score: Double, v: SIMD3<Double>, enter: Double, leave: Double)] = []
            for h in 0..<12 {
                let heading = base + Double(h) * .pi / 6
                for climb in kind == .goose ? [0.0] : [0.0, -0.35, 0.2] {
                    let vy = climb * cruise, vh = sqrt(cruise * cruise - vy * vy)
                    let v = SIMD3(sin(heading) * vh, vy, cos(heading) * vh)
                    var enter = -1.0, leave = 30.0
                    // Behind a hill the test marches the sightline, so it looks once a second.
                    for i in stride(from: behindHill ? 1 : 0, to: 60, by: behindHill ? 2 : 1) {
                        let k = Double(i + 1) * 0.5
                        let p = start + v * k
                        if p.y < ground(p.x, p.z) + 2 || p.y > topAlt(over: ground(p.x, p.z), clearance: fl.clearance) + 15 { if enter < 0 { enter = 99 }; leave = min(leave, k); break }
                        let f = ahead[i]
                        let seen = behindHill ? f.shows(p, size: size) : f.inFrame(p, size: size)
                        if enter < 0 { if seen { enter = k } } else if !seen { leave = k; break }
                    }
                    guard enter >= 0, enter <= 20, leave - enter >= 4 else { continue }
                    best.append((min(leave - enter, 20) - 0.7 * enter, v, enter, leave))
                }
            }
            guard let top = best.map(\.score).max() else { failWhy[behindHill ? "nopathH" : "nopath", default: 0] += 1; continue }
            // Among the good ones, any: so flocks do not all come the one best way.
            let good = best.filter { $0.score >= top - 3 }
            let o = good[min(good.count - 1, Int(pick * Double(good.count)))]
            let heading = atan2(o.v.x, o.v.z)
            if kind == .gull {
                let m = start + o.v * (o.enter + o.leave) / 2
                if nearWater(SIMD2(m.x, m.z)) == nil { kind = .songbird; fl = Wildlife.flight(kind) }
            }
            let offs = (0..<n).map { kind == .goose ? slotOffset($0, heading: heading, span: fl.span) : unit[$0] * spread }
            let pts = offs.map { aboveGround(start + $0) }
            guard hidden(pts, size: size) else { failWhy[behindHill ? "seenH" : "seen", default: 0] += 1; continue }
            let f = Flock(kind: kind, slot: slot, anchor: start, heading: heading, speed: cruise, altitude: start.y, life: life, seedPhase: seedPhase)
            f.serial = nextId
            f.cruise = cruise
            f.hold = kind == .raptor ? hold : (kind == .bat ? 2.5 : fl.speed * pace)
            f.transit = (o.enter + o.leave) / 2
            f.climb = o.v.y
            if Wildlife.trace {
                print(String(format: "SPAWN t=%.1f %@ %@ r=%.0f in frame %.1f-%.1f s", time, "\(kind)", behindHill ? "behind a hill" : "off frame", r, o.enter, o.leave))
            }
            for i in 0..<n {
                let v = o.v + SIMD3(rand(-1...1), rand(-0.3...0.3), rand(-1...1))
                var b = Bird(pos: pts[i], vel: v, size: fl.span * rand(0.88...1.12), tint: rand(0...1), phase: rand(0...1))
                b.flap = rand(0...(2 * .pi))
                b.beat = rand() < 0.5 ? rand(fl.flapFor) : -rand(fl.glideFor)
                b.id = nextId
                nextId += 1
                f.birds.append(b)
            }
            flocks.append(f)
            return true
        }
        return false
    }

    // A murmuration gathers: its place is in the frame ahead, but its birds start out of sight in a few long streams
    // beside and behind the frame and fly in to their places on the sheet.
    private func spawnMurmuration(slot: Int, cam: SIMD3<Double>, camYaw: Double, pitch: Double) -> Bool {
        guard let s = sight else { return false }
        let n = min(Int(rand(320...420)), room)
        let life = rand(70...150), seedPhase = rand(0...1000)
        guard n >= 200 else { return false }
        let c2 = SIMD2(cam.x, cam.z)
        for _ in 0..<6 {
            let r = rand(240...400), bearing = rand(-0.3...0.3)
            let a = camYaw + bearing
            let p = c2 + SIMD2(sin(a), cos(a)) * r
            let alt = flightLevel(.starling, at: p, from: cam, elev: elevation(.starling, pitch: pitch))
            let anchor = SIMD3(p.x, alt, p.y)
            let streams = Int(rand(3...5.99))
            let tilts = (0..<streams).map { _ in (rand(-0.8...0.1), rand(-0.05...0.25), rand(40...100)) }
            let firstSide: Double = rand() < 0.5 ? -1 : 1
            guard alt <= topAlt(over: ground(p.x, p.y), clearance: Wildlife.flight(.starling).clearance), s.direct(anchor, size: 40),
                  predicted(s, tau: 10, turnFor: 6).direct(anchor, size: 40) else { continue }
            let f = makeMurmuration(slot: slot, anchor: anchor, n: n, life: life, seedPhase: seedPhase)
            let right = SIMD3(cos(camYaw), 0, -sin(camYaw)), ahead = SIMD3(sin(camYaw), 0, cos(camYaw))
            var staged = true
            for j in 0..<streams where staged {
                let (tilt, rise, length) = tilts[j]
                let side = j % 2 == 0 ? firstSide : -firstSide
                let u = simd_normalize(right * side * cos(tilt) + ahead * sin(tilt) + SIMD3(0, rise, 0))
                // The stream's head starts just out of sight, its tail further out along the same line.
                guard let d0 = stride(from: 20.0, through: 1200, by: 10).first(where: { hidden([anchor + u * $0], size: 40) }) else {
                    staged = false
                    break
                }
                let lo = j * n / streams, hi = (j + 1) * n / streams
                for i in lo..<hi {
                    let q = Double(i - lo) / Double(max(hi - lo, 1))
                    var d = d0 + 15 + q * length
                    let jitter = SIMD3(rand(-10...10), rand(-5...5), rand(-10...10))
                    var tries = 0
                    while !hidden([aboveGround(anchor + u * d + jitter)], size: f.birds[i].size) && tries < 20 { d += 10; tries += 1 }
                    if tries == 20 { staged = false; break }
                    f.birds[i].pos = aboveGround(anchor + u * d + jitter)
                    f.birds[i].vel = -u * 14 + SIMD3(rand(-1...1), rand(-0.5...0.5), rand(-1...1))
                }
            }
            guard staged else { continue }
            f.transit = gatherTime(f)
            if Wildlife.trace { print(String(format: "MURMUR t=%.1f r=%.0f streams=%d n=%d gathered in %.0f s", time, r, streams, n, f.transit)) }
            flocks.append(f)
            return true
        }
        return false
    }

    // Moves an unseen murmuration's place into the frame ahead; its birds fly there from where they are.
    private func regather(_ f: Flock, cam: SIMD3<Double>, camYaw: Double, pitch: Double) -> Bool {
        guard let s = sight else { return false }
        let c2 = SIMD2(cam.x, cam.z)
        for _ in 0..<8 {
            let r = rand(220...400), a = camYaw + rand(-0.3...0.3)
            let p = c2 + SIMD2(sin(a), cos(a)) * r
            let alt = flightLevel(.starling, at: p, from: cam, elev: elevation(.starling, pitch: pitch))
            let anchor = SIMD3(p.x, alt, p.y)
            guard alt <= topAlt(over: ground(p.x, p.y), clearance: Wildlife.flight(.starling).clearance), s.direct(anchor, size: 40), predicted(s, tau: 10, turnFor: 6).direct(anchor, size: 40) else { continue }
            f.anchor = anchor
            f.altitude = alt
            f.drift = SIMD3(camVel.x, 0, camVel.z) * 0.8
            f.unseen = 0
            f.transit = gatherTime(f)
            if Wildlife.trace { print(String(format: "REGATHER t=%.1f r=%.0f in %.0f s", time, r, f.transit)) }
            return true
        }
        return false
    }

    // Roughly how long the furthest bird takes to reach its place on the sheet.
    private func gatherTime(_ f: Flock) -> Double {
        let far = f.birds.reduce(0.0) { max($0, simd_length(sheetPoint(f, $1.sheet, $1.phase) - $1.pos)) }
        return far / 15 + 3
    }

    // A murmuration flock with its birds in their places on the sheet.
    private func makeMurmuration(slot: Int, anchor: SIMD3<Double>, n: Int, life: Double, seedPhase: Double) -> Flock {
        let fl = Wildlife.flight(.starling)
        let f = Flock(kind: .starling, slot: slot, anchor: anchor, heading: rand(0...(2 * .pi)), speed: 1.5, altitude: anchor.y,
                      life: life, seedPhase: seedPhase)
        f.serial = nextId
        f.cruise = 8
        f.hold = 1.5
        for _ in 0..<n {
            var b = Bird(pos: anchor, vel: f.dir * fl.speed, size: fl.span * rand(0.88...1.12), tint: rand(0...1), phase: rand(0...1))
            // Denser toward the sheet's middle.
            let ang = rand(0...(2 * .pi)), rad = pow(rand(), 0.7)
            b.sheet = SIMD2(cos(ang), sin(ang)) * rad
            b.pos = sheetPoint(f, b.sheet, b.phase)
            b.flap = rand(0...(2 * .pi))
            b.beat = rand() < 0.5 ? rand(fl.flapFor) : -rand(fl.glideFor)
            b.id = nextId
            nextId += 1
            f.birds.append(b)
        }
        // Neighbours on the sheet stay neighbours in the air: order by strips of the sheet so each cluster of
        // consecutive birds is a compact patch.
        let strips = Double(max(1, f.birds.count / (Wildlife.clusterSize * 4)))
        f.birds.sort { a, b in
            let sa = floor((a.sheet.x + 1) / 2 * strips), sb = floor((b.sheet.x + 1) / 2 * strips)
            return sa != sb ? sa < sb : a.sheet.y < b.sheet.y
        }
        return f
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
        return f.anchor + q + (f.dir * f.speed + f.drift) * dt
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
        f.transit -= dt
        let arrived = f.transit <= 0
        f.speed += (((arrived || f.kind == .starling) && !f.leaving ? f.hold ?? f.cruise : f.cruise) - f.speed) * min(1, dt * 0.3)

        // Anchor: a heading that drifts on the noise flow field, bends down valleys (toward the lower side ahead),
        // and for gulls toward water. Flocks flying in hold course; leaving flocks in frame turn for its nearer side.
        let ap = SIMD2(f.anchor.x, f.anchor.z)
        if f.leaving, f.unseen == 0, let s = sight {
            let side: Double = simd_dot(f.anchor - s.pos, s.right) >= 0 ? 1 : -1
            let out = atan2(s.fwd.x, s.fwd.z) + side * .pi / 2
            f.heading += max(-0.3 * dt, min(0.3 * dt, atan2(sin(out - f.heading), cos(out - f.heading))))
        }
        if !f.leaving && arrived && f.kind != .goose {
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
        // Giant trees: flown round, not over: a flock below a crown's top turns away from its axis well before its reach,
        // and one that holds to a spot is pushed off it.
        for g in giantsAt() where f.anchor.y < g.top + 10 {
            let d = SIMD2(f.anchor.x - g.pos.x, f.anchor.z - g.pos.y), l = simd_length(d)
            let near = GiantInfo.reach + 160
            guard l < near, l > 1 else { continue }
            let away = d / l, fd = SIMD2(sin(f.heading), cos(f.heading))
            let w = 1 - l / near
            if simd_dot(fd, away) < 0.3 {
                let side: Double = fd.x * away.y - fd.y * away.x > 0 ? -1 : 1
                f.heading += side * 0.6 * w * dt
            }
            if f.kind == .starling || (f.kind == .bat && arrived) || l < GiantInfo.reach + 20 {
                f.anchor.x += away.x * 12 * w * dt
                f.anchor.z += away.y * 12 * w * dt
            }
        }
        // Raptors and gulls near an island circle it, on a ring clear of its reach at the height of its flanks, rising to it
        // slowly; a ring for a smaller island keeps them in sight of a camera passing it.
        if !f.leaving && arrived && (f.kind == .raptor || f.kind == .gull) && f.orbit == nil {
            var best: IslandInfo?, bd = Double.infinity
            for s in islesAt() where s.radius >= 40 && s.radius <= 320 {
                let l = simd_length(ap - s.pos)
                if l < s.reach + 450 && l < bd { bd = l; best = s }
            }
            if let s = best {
                let d = ap - s.pos, fd = SIMD2(sin(f.heading), cos(f.heading))
                f.orbitSense = simd_dot(fd, SIMD2(-d.y, d.x)) >= 0 ? 1 : -1
                f.orbit = SIMD4(s.pos.x, s.pos.y, s.reach + (f.kind == .gull ? 70 : 45), s.keel + 0.55 * (s.top - s.keel))
                // Round in a few minutes, about 0.04 rad/s.
                f.hold = min(max(f.hold ?? 0, 0.04 * (s.reach + 45)), fl.maxSpeed * 0.8)
            }
        }
        if let o = f.orbit, !f.leaving {
            let d = ap - SIMD2(o.x, o.y), l = max(simd_length(d), 1)
            let want = SIMD2(-d.y, d.x) / l * f.orbitSense + d / l * max(-1, min(1, (o.z - l) / 60))
            let off = atan2(want.x, want.y) - f.heading
            f.heading += max(-0.5 * dt, min(0.5 * dt, atan2(sin(off), cos(off))))
            f.altitude += max(-3 * dt, min(3 * dt, o.w - f.altitude))
        }
        // Islands: flown round where the flock is level with one, flown under or over otherwise (the one it circles aside).
        for s in islesAt() where f.anchor.y > s.keel - 40 && f.anchor.y < s.top + 20 && !(f.orbit.map { simd_length(SIMD2($0.x, $0.y) - s.pos) < 1 } ?? false) {
            let d = SIMD2(f.anchor.x - s.pos.x, f.anchor.z - s.pos.y), l = simd_length(d)
            let near = s.reach + 140
            guard l < near, l > 1 else { continue }
            let away = d / l, fd = SIMD2(sin(f.heading), cos(f.heading))
            let w = 1 - l / near
            if simd_dot(fd, away) < 0.3 {
                let side: Double = fd.x * away.y - fd.y * away.x > 0 ? -1 : 1
                f.heading += side * 0.6 * w * dt
            }
            if l < s.reach + 30 {
                f.anchor.x += away.x * 12 * w * dt
                f.anchor.z += away.y * 12 * w * dt
            }
        }
        if (f.kind == .starling || (f.kind == .bat && arrived)) && !f.leaving {
            // These hold to one spot: the anchor circles slowly about where it arrived.
            f.heading += 0.12 * dt
        }
        if f.kind == .starling {
            let pace = f.leaving ? SIMD3<Double>.zero : SIMD3(camVel.x, 0, camVel.z) * 0.8
            f.drift += (pace - f.drift) * min(1, dt * 0.2)
            f.anchor += f.drift * dt
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
        if !arrived && f.kind != .starling { f.altitude += f.climb * dt }
        let vA = f.dir * f.speed + SIMD3(0, arrived || f.kind == .starling ? 0 : f.climb, 0)
        f.anchor += vA * dt
        let ay = f.anchor.y
        var gAhead = ground(f.anchor.x, f.anchor.z, at: ay)
        for d in [20.0, 45.0, 80.0] {
            gAhead = max(gAhead, ground(f.anchor.x + vA.x / max(f.speed, 0.1) * d, f.anchor.z + vA.z / max(f.speed, 0.1) * d, at: ay) - d * 0.15)
        }
        // Birds spread to either side of the anchor (raptors circle ~35 out): a tall crown beside its track lifts the
        // flock too, so its birds pass over the canopy rather than climbing out of it one by one.
        let sideV = simd_normalize(SIMD2(-vA.z, vA.x) + 1e-9) * 35
        gAhead = max(gAhead, ground(f.anchor.x + sideV.x, f.anchor.z + sideV.y, at: ay), ground(f.anchor.x - sideV.x, f.anchor.z - sideV.y, at: ay))
        // Ground-relative: flocks rise with the land (over a wall, before it) and otherwise keep under the clouds.
        let lift = gAhead + fl.clearance * (f.kind == .raptor ? 2 : 1.5)
        var yT = max(f.orbit != nil && !f.leaving ? f.altitude : min(f.altitude, FlightCamera.ceiling - 20), lift)
        // Under an island the flock keeps below its underside.
        if let c = columnAt(f.anchor.x, f.anchor.z), let u = c.under, ay < u - 12 { yT = min(yT, max(u - 20 - fl.clearance, c.ground + fl.clearance)) }
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
            case .raptor where arrived && !f.leaving:
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
                let gap = target - b.pos, gl = simd_length(gap)
                if gl > 20 {
                    // Still streaming in: a steady cruise toward its place, on top of the sheet's own drift, within
                    // the top speed so the drift does not eat the approach.
                    let closing = min(gl * 0.5, (max(16, fl.maxSpeed * fl.maxSpeed * 0.92 - simd_length_squared(sheetVel))).squareRoot())
                    acc += (sheetVel + gap / gl * closing - b.vel) * 1.5
                } else {
                    acc += gap * 1.6
                    acc += (sheetVel - b.vel) * 1.2
                }
                let dh = b.pos - hawk, dl = simd_length(dh)
                if dl < 12 { acc += dh / max(dl, 0.5) * 40 * (1 - dl / 12) }
            default:
                // Bats flying in hold their line; once arrived they jink about the spot.
                let commute = f.kind == .bat && !arrived
                acc += toA * (commute ? 0.4 : fl.pull) + (vA - b.vel) * (commute ? 1.2 : 0.15)
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
            acc += SIMD3(wx, wy, wz) * fl.wander * (f.kind == .bat && !arrived ? 0.3 : 1)
            // Terrain: climb away from ground ahead, and keep well under the cloud base, and under an island's underside.
            let look = b.pos + b.vel * 1.2, look2 = b.pos + b.vel * 2.5
            let by = b.pos.y
            let g = max(ground(b.pos.x, b.pos.z, at: by), ground(look.x, look.z, at: by), ground(look2.x, look2.z, at: by) - 3)
            let clear = fl.clearance
            if b.pos.y < g + clear { acc.y += (g + clear - b.pos.y) * 4 }
            let roof = max(215, g + clear + 10, f.orbit != nil ? f.anchor.y + 40 : 0)
            if b.pos.y > roof { acc.y -= (b.pos.y - roof) * 2 }
            if let c = columnAt(look.x, look.z), let u = c.under, by < u - 12, by > u - 12 - clear { acc.y -= (by - (u - 12 - clear)) * 4 }

            b.vel += acc * dt
            var s = simd_length(b.vel)
            if s > fl.maxSpeed { b.vel *= fl.maxSpeed / s; s = fl.maxSpeed }
            if s < fl.minSpeed { b.vel = (s > 1e-3 ? b.vel / s : SIMD3(sin(f.heading), 0, cos(f.heading))) * fl.minSpeed; s = fl.minSpeed }
            // Birds climb and dive at modest angles.
            let maxY = 0.45 * s
            b.vel.y = max(-maxY, min(maxY, b.vel.y))
            b.pos += b.vel * dt
            // Never into the ground. Its 4x4 maximum steps by whole voxels from cell to cell, so a bird under it
            // climbs out at its top speed rather than jumping onto it; one caught in an island's rock leaves by the nearer
            // of its top and underside.
            if let c = columnAt(b.pos.x, b.pos.z) {
                if let u = c.under, b.pos.y >= u - 12, b.pos.y < c.top + 1.5, b.pos.y - u < c.top - b.pos.y {
                    b.pos.y = max(u - 12, b.pos.y - fl.maxSpeed * dt)
                    b.vel.y = min(b.vel.y, -maxY)
                } else {
                    let h = c.under.map { b.pos.y < $0 - 12 ? c.ground : c.top } ?? c.top
                    if b.pos.y < h + 1.5 {
                        b.pos.y = min(h + 1.5, b.pos.y + fl.maxSpeed * dt)
                        b.vel.y = max(b.vel.y, maxY)
                    }
                }
            }

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

    // The birds the renderer is given, flock by flock and cluster by cluster, within the bird buffer's caps.
    func drawList() -> [(flock: Flock, clusters: [Range<Int>])] {
        var list: [(flock: Flock, clusters: [Range<Int>])] = []
        var nc = 0, nb = 0
        for f in flocks where list.count < Wildlife.maxFlocks {
            var cs: [Range<Int>] = []
            for start in stride(from: 0, to: f.birds.count, by: Wildlife.clusterSize) where nc < Wildlife.maxClusters {
                let n = min(min(start + Wildlife.clusterSize, f.birds.count) - start, Wildlife.maxBirds - nb)
                if n <= 0 { continue }
                cs.append(start..<(start + n))
                nb += n
                nc += 1
            }
            if !cs.isEmpty { list.append((f, cs)) }
        }
        return list
    }

    // Writes cluster spheres then birds into `buffer`, map-local (relative to `origin`). Returns the flock count.
    func upload(into buffer: MTLBuffer, origin o: SIMD3<Double>) -> Int {
        let out = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: Wildlife.bufferLength / 16)
        var nc = 0, nb = 0
        let clusters = Wildlife.maxFlocks * 2
        let birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2
        var nf = 0
        for (f, cs) in drawList() {
            let fl = Wildlife.flight(f.kind)
            let firstCluster = nc
            var flo = SIMD3<Double>(repeating: .infinity), fhi = SIMD3<Double>(repeating: -.infinity)
            for c in cs {
                let first = nb
                var lo = SIMD3<Double>(repeating: .infinity), hi = SIMD3<Double>(repeating: -.infinity)
                for i in c {
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
                flo = simd_min(flo, lo); fhi = simd_max(fhi, hi)
                let c = (lo + hi) / 2
                let r = simd_length(hi - lo) / 2 + fl.span
                out[clusters + nc * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
                out[clusters + nc * 2 + 1] = SIMD4(Float(first), Float(nb - first), 0, 0)
                nc += 1
            }
            let c = (flo + fhi) / 2
            let r = simd_length(fhi - flo) / 2 + fl.span
            out[nf * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
            out[nf * 2 + 1] = SIMD4(Float(firstCluster), Float(nc - firstCluster), 0, 0)
            nf += 1
        }
        return nf
    }

    // Debug: replaces the flocks with one murmuration `distance` ahead of the camera, gathered, for timing the worst case.
    func testMurmuration(camera: FlightCamera, distance: Double) {
        flocks = []
        let n = Int(rand(320...420))
        flocks = [makeMurmuration(slot: 0, anchor: camera.position + camera.forward * distance, n: n, life: rand(70...150), seedPhase: rand(0...1000))]
    }

    // Debug: replaces the flocks with a grid of birds of every kind close in front of the camera, at varied headings
    // and wingbeat phases, for judging their shapes.
    func testPattern(camera: FlightCamera, distance: Double) {
        let b = camera.basis
        let kinds: [Kind] = [.songbird, .raptor, .gull, .starling, .bat, .goose]
        flocks = kinds.map { Flock(kind: $0, slot: 99, anchor: camera.position, heading: 0, speed: 0, altitude: 0, life: 1e9, seedPhase: 0) }
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
