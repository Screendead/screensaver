import Foundation
import simd

extension Wildlife {
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
    func spawn(kind wanted: Kind, slot: Int, cam: SIMD3<Double>, camYaw: Double, pitch: Double) -> Bool {
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
            // Against a camera much faster than their top speed no path would overtake the frame's edges as it closes on
            // them, so none would spawn: they glide in at three quarters of the camera's pace, over their top speed.
            let cruise = max(fl.maxSpeed * (0.92 + 0.08 * pace), 0.75 * simd_length(SIMD2(camVel.x, camVel.z)))
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
    func regather(_ f: Flock, cam: SIMD3<Double>, camYaw: Double, pitch: Double) -> Bool {
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
    func makeMurmuration(slot: Int, anchor: SIMD3<Double>, n: Int, life: Double, seedPhase: Double) -> Flock {
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
    func sheetPoint(_ f: Flock, _ uv: SIMD2<Double>, _ ph: Double, dt: Double = 0) -> SIMD3<Double> {
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
    func slotOffset(_ i: Int, heading: Double, span: Double) -> SIMD3<Double> {
        if i == 0 { return .zero }
        let rank = Double((i + 1) / 2), side: Double = i % 2 == 1 ? 1 : -1
        let f = SIMD3(sin(heading), 0, cos(heading)), r = SIMD3(cos(heading), 0, -sin(heading))
        return -f * rank * 1.5 * span + r * side * rank * 1.25 * span + SIMD3(0, -0.15 * rank, 0)
    }
}
