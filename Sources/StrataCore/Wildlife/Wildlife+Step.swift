import Foundation
import simd

extension Wildlife {
    // Which flocks the hour calls for, one per slot. Gull slots fall back to songbirds away from water.
    // Among floating islands raptors take a songbird's slot: they circle the islands.
    private static func roster(sunY: Double, phase: Double, isles: Bool = false) -> [Kind] {
        if sunY < -0.16 { return [.bat, .bat, .bat] }
        let dusk = phase > 0.25 && phase < 0.75
        // Flocks spend a while flying in out of sight, so the hour keeps more of them on the wing.
        if sunY < 0.14 { return dusk ? [.starling, .songbird, .gull, .goose, .songbird, .songbird] : [.goose, .songbird, .gull, .songbird, .songbird, .goose] }
        return [.songbird, .raptor, .gull, isles ? .raptor : .songbird, .songbird, .gull, .songbird]
    }

    func nearWater(_ p: SIMD2<Double>) -> SIMD2<Double>? {
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

    package func step(dt: Double, camera: FlightCamera, sight: Sight, sunY: Double, phase: Double) {
        time += dt
        // The last frame drew the birds where they are now, from where the camera was then.
        var sight = sight
        sight.slack = 2
        let shown = self.sight ?? sight
        self.sight = sight
        hiddenFrames = []
        let cam = camera.position
        let cam2 = SIMD2(cam.x, cam.z)
        let isles = islesAt().contains { $0.radius >= 40 && simd_length($0.pos - cam2) < $0.reach + 1500 }
        let roster = Wildlife.roster(sunY: sunY, phase: phase, isles: isles)
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
                return String(format: "%@%@%@ n=%d framed=%d d=%.0f tr=%.0f", "\(f.kind)", f.leaving ? "(L)" : "", f.orbit != nil ? "(O)" : "", f.birds.count, framed, d, max(0, f.transit))
            }
            print(String(format: "TRACE t=%.0f sunY=%.2f ", time, sunY) + desc.joined(separator: " | "))
        }
    }
    static var trace: Bool { Knobs.current.birdTrace }
}
