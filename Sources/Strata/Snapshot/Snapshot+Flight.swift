import Metal
import StrataCore
import simd

extension Snapshot {
    // VR_FLIGHT=minutes: flies the autopilot that long, logging altitude, clearance and climb every 12 s and
    // writing a frame each minute next to the screenshot path. VR_FTRACE=1 also prints every 0.25 s: time, x, z, y, ground,
    // yaw, the altitude terms (target, climb rate, need (-1: floor clamp), rim lift, regional floor, ahead, planned climb)
    // and the routing (blocked ahead, route offset).
    static func fly(_ r: Renderer, minutes mins: Double, output tex: MTLTexture, path: String) -> Bool {
        let env = Knobs.current.env
        var wrote = true
        r.camera.tracing = env["VR_FTRACE"] != nil
        var maxClimb = 0.0, minClear = Double.infinity, maxY = 0.0, lows = 0
        var walled = 0, walledRun = 0, maxWalledRun = 0, fastClimb = 0, prevY = r.camera.position.y
        var lastY = r.camera.position.y
        var underCrown = 0, giantsSeen = Set<Int>()
        // Ruined towers: the closest the flight came to a plinth's edge below its top, and the seconds a tower stood in
        // frame within 9k.
        var towerEdge = Double.infinity, towerSeen = 0, towerClose = 0
        var towersMet = Set<SIMD2<Int>>()
        // Islands: the closest the flight came to one's rock (from the CPU cells within 24), the seconds it was nearer
        // than 25, the seconds it spent beneath one, and the islands it passed under.
        var isleClear = Double.infinity, isleScrape = 0, isleUnder = 0, birdsInRock = 0, birdChecks = 0
        var islesUnder = Set<SIMD2<Int>>()
        // Seconds with an island's reach within 400 on either side, and flocks circling an island (by serial).
        var isleBetween = 0, circlingSeconds = 0
        var circlers = Set<Int>()
        for i in 1...Int(mins * 60) {
            for q in 0..<4 {
                r.simulate(0.25)
                if env["VR_FTRACE"] != nil {
                    let c = r.camera.position
                    print(String(format: "ft %.2f %.1f %.1f %.2f %.2f %.3f", Double(i - 1) + Double(q + 1) * 0.25, c.x, c.z, c.y, r.world.terrain.height(x: c.x, z: c.z) ?? -1, r.camera.yaw)
                          + " " + r.camera.terms.map { String(format: "%.1f", $0) }.joined(separator: " "))
                }
            }
            let p = r.camera.position
            let col = r.world.terrain.column(x: p.x, z: p.z)
            let g = col.map { c in c.under.map { p.y < $0 ? c.ground : c.top } ?? c.top } ?? r.world.terrain.height(x: p.x, z: p.z) ?? 0
            var near = Double.infinity
            for (dx, dz) in [(0.0, 0.0)] + (0..<16).map({ (k: Int) -> (Double, Double) in
                let a = Double(k % 8) * .pi / 4, rr = k < 8 ? 12.0 : 24.0
                return (sin(a) * rr, cos(a) * rr)
            }) {
                guard let c = r.world.terrain.column(x: p.x + dx, z: p.z + dz), let u = c.under else { continue }
                let v = p.y < u ? u - p.y : p.y > c.top ? p.y - c.top : 0
                near = min(near, (dx * dx + dz * dz + v * v).squareRoot())
            }
            isleClear = min(isleClear, near)
            // Birds inside an island's rock (its cell's span), which Sight cannot see as hidden.
            for f in r.wildlife.flocks {
                for b in f.birds {
                    birdChecks += 1
                    if let c = r.world.terrain.column(x: b.pos.x, z: b.pos.z), let u = c.under, b.pos.y >= u, b.pos.y < c.top { birdsInRock += 1 }
                }
            }
            if near < 25 { isleScrape += 1; print(String(format: "flight island scrape t=%d pos=(%.0f,%.0f) y=%.0f clearance %.0f", i, p.x, p.z, p.y, near)) }
            do {
                // Between two islands: one abreast on each side (its reach spans the camera along the track), within 400 of its rock.
                let side = SIMD2(cos(r.camera.yaw), -sin(r.camera.yaw)), fwd = SIMD2(sin(r.camera.yaw), cos(r.camera.yaw))
                var l = false, rt = false
                for s in r.world.terrain.isles where s.radius > 40 {
                    let v = s.pos - SIMD2(p.x, p.z), across = simd_dot(v, side), along = simd_dot(v, fwd)
                    if abs(along) < s.reach, abs(across) > s.reach * 0.5, abs(across) < s.reach + 400 { if across > 0 { rt = true } else { l = true } }
                }
                if l && rt { isleBetween += 1 }
                let circ = r.wildlife.flocks.filter { $0.orbit != nil && !$0.leaving }
                if !circ.isEmpty { circlingSeconds += 1 }
                for f in circ { circlers.insert(f.serial) }
            }
            if let u = col?.under, p.y < u {
                isleUnder += 1
                if let s = r.world.terrain.isles.first(where: { simd_length($0.pos - SIMD2(p.x, p.z)) < $0.reach }) { islesUnder.insert(SIMD2(Int(s.pos.x), Int(s.pos.y))) }
            }
            if p.y - lastY > 6 { print(String(format: "flight jump t=%d pos=(%.0f,%.0f) %.0f -> %.0f", i, p.x, p.z, lastY, p.y)) }
            maxClimb = max(maxClimb, p.y - lastY)
            lastY = p.y
            minClear = min(minClear, p.y - g)
            maxY = max(maxY, p.y)
            if p.y < g + 2 { lows += 1 }
            var seen = false
            for t in r.towers.list {
                let v = t.pos - SIMD2(p.x, p.z), d = simd_length(v)
                if p.y < t.top { towerEdge = min(towerEdge, d - t.radius * 1.42) }
                if d < 9000, abs(atan2(sin(atan2(v.x, v.y) - r.camera.yaw), cos(atan2(v.x, v.y) - r.camera.yaw))) < 0.6 {
                    seen = true
                    if towersMet.insert(SIMD2(Int(t.pos.x), Int(t.pos.y))).inserted {
                        print(String(format: "flight tower seen t=%d at %.0f,%.0f member %d dist %.0f", i, t.pos.x, t.pos.y, t.member, d))
                    }
                }
                if d < 1500 { towerClose += 1 }
            }
            if seen { towerSeen += 1 }
            let wall = r.camera.wallInFrame()
            if wall >= 0.4 { walled += 1; walledRun += 1; maxWalledRun = max(maxWalledRun, walledRun) } else { walledRun = 0 }
            if p.y - (i == 1 ? p.y : prevY) > 6 { fastClimb += 1 }
            for g in r.world.terrain.giants {
                let dg = simd_length(SIMD2(p.x, p.z) - g.pos)
                if dg < GiantInfo.reach && p.y < g.top { underCrown += 1 }
                if dg < 1500 { giantsSeen.insert(Int(g.pos.x) * 100_003 + Int(g.pos.y)) }
            }
            prevY = p.y
            if i % 12 == 0 {
                fflush(stdout)
                let nearG = r.world.terrain.giants.map { simd_length(SIMD2(p.x, p.z) - $0.pos) }.min() ?? -1
                let nearI = r.world.terrain.isles.map { simd_length(SIMD2(p.x, p.z) - $0.pos) - $0.reach }.min() ?? -1
                print(String(format: "flight t=%d pos=(%.0f,%.0f) y=%.0f ground=%.0f wall=%.2f far=%.0f yaw=%.0f giant=%.0f isle=%.0f", i, p.x, p.z, p.y, g, wall,
                             r.farHeight(x: p.x + sin(r.camera.yaw) * 600, z: p.z + cos(r.camera.yaw) * 600) ?? -1, r.camera.yaw * 180 / .pi, nearG, nearI))
            }
            if i % 60 == 0 {
                for _ in 0..<4 { render(r, into: tex) }
                wrote = save(tex, to: path.replacingOccurrences(of: ".png", with: "_m\(i / 60).png")) && wrote
            }
        }
        print(String(format: "flight towers: closest plinth edge %.0f, seconds with a tower ahead within 9k %d, tower-seconds within 1.5k %d", towerEdge, towerSeen, towerClose))
        print(String(format: "flight islands: closest %.0f, seconds nearer than 25 %d, seconds beneath one %d, islands passed under %d, on the table %d, bird-seconds in island cells %d of %d",
                     isleClear, isleScrape, isleUnder, islesUnder.count, r.world.terrain.isles.count, birdsInRock, birdChecks))
        print(String(format: "flight islands 2: seconds between two %d, seconds with a flock circling one %d, flocks that circled %d", isleBetween, circlingSeconds, circlers.count))
        print(String(format: "flight done: max climb %.2f/s, min clearance %.1f, max y %.0f (ceiling %.0f), low seconds %d, walled seconds %d (longest %d), seconds climbing over 6/s %d, seconds under a crown %d, giants passed within 1500 %d", maxClimb, minClear, maxY, FlightCamera.ceiling, lows, walled, maxWalledRun, fastClimb, underCrown, giantsSeen.count))
        if let log = r.popLog { print(log.summary(r.wildlife)) }
        return wrote
    }
}
