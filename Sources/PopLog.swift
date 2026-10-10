import Foundation
import simd

// VR_POPLOG=1: after every simulation step, any bird that starts or stops being drawn while a frame could show it
// (Sight.sees: in the frustum, not behind terrain, not lost in fog, directly or in a reflection). A bird that
// appears is judged with this step's camera; one that disappears with the previous step's camera and position, the
// last frame that drew it.
final class PopLog {
    static let enabled = ProcessInfo.processInfo.environment["VR_POPLOG"] != nil

    private struct Drawn { var pos: SIMD3<Double>; var size: Double; var kind: Wildlife.Kind; var flock: Int }
    private var drawn: [Int: Drawn] = [:]
    private var lastSight: Sight?
    private(set) var pops = 0
    private var byCause: [String: Int] = [:]
    private var printed = 0
    // Birds in frame (frustum and fog, not occlusion), integrated over time, by part of the day.
    private var birdSeconds: [String: Double] = [:]
    private var periodSeconds: [String: Double] = [:]
    private var spawned = 0, retired = 0

    static func period(sunY: Double, phase: Double) -> String {
        if sunY < -0.16 { return "night" }
        if sunY < 0.14 { return phase - floor(phase) > 0.25 && phase - floor(phase) < 0.75 ? "dusk" : "dawn" }
        return "day"
    }

    func check(_ w: Wildlife, sight: Sight, time: Double, dt: Double, period: String) {
        var now: [Int: Drawn] = [:]
        var inFrame = 0
        for (f, cs) in w.drawList() {
            let fid = f.serial
            for c in cs {
                for i in c {
                    let b = f.birds[i]
                    now[b.id] = Drawn(pos: b.pos, size: b.size, kind: f.kind, flock: fid)
                    if sight.inFrame(b.pos, size: b.size) {
                        let v = b.pos - sight.pos, t = simd_length(v)
                        if t < 1 || sight.transmittance(from: sight.pos, dir: v / t, t: t) >= Sight.fogHidden { inFrame += 1 }
                    }
                }
            }
        }
        birdSeconds[period, default: 0] += Double(inFrame) * dt
        periodSeconds[period, default: 0] += dt
        let liveFlocks = Set(w.flocks.map(\.serial))
        let oldFlocks = Set(drawn.values.map(\.flock))
        let newFlocks = Set(now.values.map(\.flock))
        spawned += newFlocks.subtracting(oldFlocks).count
        retired += oldFlocks.subtracting(liveFlocks).count
        for (id, d) in now where drawn[id] == nil {
            let cause = oldFlocks.contains(d.flock) ? "fade-in" : "spawn"
            if sight.sees(d.pos, size: d.size) { report(cause, d, sight: sight, time: time) }
        }
        // A bird that jumps further in one step than it can fly vanishes in one place and appears in another.
        for (id, d) in now {
            guard let o = drawn[id] else { continue }
            let jump = simd_length(d.pos - o.pos)
            if jump > Wildlife.flight(d.kind).maxSpeed * dt * 1.5 + 0.5,
               sight.sees(d.pos, size: d.size) || (lastSight?.sees(o.pos, size: o.size) ?? false) {
                report("teleport", d, sight: sight, time: time, note: String(format: " jump=%.1f dy=%.1f", jump, d.pos.y - o.pos.y))
            }
        }
        if let prev = lastSight {
            for (id, d) in drawn where now[id] == nil {
                let cause = liveFlocks.contains(d.flock) ? "fade-out/cull" : "retire"
                if prev.sees(d.pos, size: d.size) { report(cause, d, sight: prev, time: time) }
            }
        }
        drawn = now
        lastSight = sight
    }

    private func report(_ cause: String, _ d: Drawn, sight: Sight, time: Double, note: String = "") {
        pops += 1
        byCause["\(cause) \(d.kind)", default: 0] += 1
        guard printed < 60 else { return }
        printed += 1
        let v = d.pos - sight.pos
        let z = simd_dot(v, sight.fwd)
        print(String(format: "POP t=%.2f %@ %@ dist=%.0f screen=(%.2f,%.2f)%@", time, cause, "\(d.kind)", simd_length(v),
                     simd_dot(v, sight.right) / max(z, 1e-3) / (sight.tanV * sight.aspect), simd_dot(v, sight.up) / max(z, 1e-3) / sight.tanV,
                     (sight.direct(d.pos, size: d.size) ? "" : " (reflection)") + note))
    }

    func summary(_ w: Wildlife) -> String {
        let causes = byCause.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        let seen = ["dawn", "day", "dusk", "night"].compactMap { p -> String? in
            guard let s = periodSeconds[p], s > 0 else { return nil }
            return String(format: "%@ %.1f", p, (birdSeconds[p] ?? 0) / s)
        }.joined(separator: ", ")
        let spawns = w.spawnStats.sorted { $0.key < $1.key }.map {
            String(format: "%@ %d/%d %.1fs", $0.key, $0.value.made, $0.value.tries, $0.value.transit / Double(max($0.value.made, 1)))
        }.joined(separator: ", ")
        print("spawns made/tried, mean transit: \(spawns); failed starts: \(w.failWhy.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }); slowest spawn \(String(format: "%.1f", w.spawnMs)) ms")
        return "POPLOG pops=\(pops) [\(causes)] flocks spawned=\(spawned) retired=\(retired); mean birds in frame: \(seen)"
    }
}
