import Foundation
import simd

extension FlightCamera {
    // Elevation angle of the highest terrain 1-8k along a heading: a range or a tepui standing in the upper middle
    // of the frame is a vista; a wall close by is not.
    // (reward, wall penalty)
    private func vista(_ heading: Double) -> (Double, Double) {
        let dir = SIMD2(sin(heading), cos(heading))
        var best = -1.0, near = -1.0
        for dist in [60.0, 120.0, 200.0] + Array(stride(from: 300.0, through: 7500.0, by: 300.0)) {
            for off in [-0.2, 0.0, 0.2] {
                let d2 = SIMD2(sin(heading + off), cos(heading + off))
                let q = SIMD2(position.x, position.z) + (off == 0 ? dir : d2) * dist
                // An island the flight will pass beneath is no wall; far off it is a vista like any relief.
                guard let g = dist < 1000 ? (dist < 300 ? (world.height(x: q.x, z: q.y) != nil ? h(q.x, q.y) : nil) : hFar(q.x, q.y))
                                          : far.height(x: q.x, z: q.y) else { continue }
                let e = atan2(g - position.y, dist)
                if dist >= 1000 { best = max(best, e) } else { near = max(near, e) }
            }
        }
        let deg = best * 180 / .pi, nearDeg = near * 180 / .pi
        let reward = deg < 2 ? max(0, deg / 2) : deg < 12 ? 1 : max(0, 1 - (deg - 12) / 8)
        return (0.6 * reward, 1.2 * min(max((nearDeg - 15) / 10, 0), 1))
    }

    // Shot director: score a candidate heading by what the frame would hold. Relief across the middle
    // distance gives a subject; some water is welcome but a frame full of it is not; facing a low sun or
    // the moon puts the light in shot; large turns cost a little so motion stays calm; land rising past eye
    // height close ahead would fill the frame with a wall. Under a high sun close slopes read as blocks, so
    // the shot looks out: open sightlines and water count for more, and near land climbing into the lower
    // half of the frame (bottom edge 0.436 rad under the pitch) costs.
    // Interest of the sites a heading frames: by day the old stones and the villages, at night only what is lit.
    // Capped, so a site adds to a shot but never outvotes the land and the light.
    func siteInterest(_ heading: Double, night: Double) -> (Double, SiteInfo?) {
        let p = SIMD2(position.x, position.z)
        var sum = 0.0, best = 0.0
        var top: SiteInfo?
        for site in world.sites {
            if let t = framedAt[site.id], time - t < 240 { continue }
            let v = site.pos - p
            let d = simd_length(v)
            guard d > 90, d < 450 else { continue }
            if abs(FlightCamera.angleDiff(atan2(v.x, v.y), heading)) > 0.35 { continue }
            let day: [SiteInfo.Kind: Double] = [.henge: 1, .ruins: 0.7, .hamlet: 0.7, .jetty: 0.5, .camp: 0.3, .cairn: 0.2, .cave: 0.4]
            let w = (day[site.kind] ?? 0) * (1 - night) + litInterest(site) * night
            let f = w * min(max((d - 90) / 60, 0), 1) * (1 - min(max((d - 300) / 150, 0), 1))
            sum += f
            if f > best { best = f; top = site }
        }
        return (min(0.5 * sum, 0.35 + 0.15 * night), top)
    }

    // A ruined tower 1.2-9k off, framed when the heading passes about 900 off one of its flanks: by day the strongest
    // landmark there is, at night (a few lit rooms on some) less so. The flank bearings are offered as shots.
    func towerFlanks() -> [(bearing: Double, tower: TowerInfo, d: Double)] {
        let p = SIMD2(position.x, position.z)
        var out: [(Double, TowerInfo, Double)] = []
        for t in towersAt() where t.member == 0 || t.height > 500 {
            let v = t.pos - p, d = simd_length(v)
            guard d > 1200, d < 9000 else { continue }
            let b = atan2(v.x, v.y), off = asin(min(1, 900 / d))
            out.append((b + off, t, d))
            out.append((b - off, t, d))
        }
        return out
    }

    private func towerInterest(_ heading: Double, night: Double) -> Double {
        var best = 0.0
        for f in towerFlanks() {
            let a = abs(FlightCamera.angleDiff(f.bearing, heading))
            guard a < 0.35 else { continue }
            let w = (0.3 * (1 - night) + (f.tower.lit ? 0.2 : 0.06) * night) * (1 - a / 0.35) * min(1, (9000 - f.d) / 2000)
            best = max(best, w)
        }
        return best
    }

    // Hearths, camp fires and jetty lanterns burn every night; a henge is only worth a night shot when its
    // rite is lit, and a beacon adds to its hamlet. Mirrors lampLevel in Shaders/Saver.metal.
    func litInterest(_ site: SiteInfo) -> Double {
        func pcg(_ v: UInt32) -> UInt32 {
            let state = v &* 747796405 &+ 2891336453
            let word = ((state >> ((state >> 28) &+ 4)) ^ state) &* 277803737
            return (word >> 22) ^ word
        }
        func burns(_ l: SiteInfo.Lamp, _ chance: Double) -> Bool {
            let h = pcg(site.id &+ UInt32(l.index % MapLayout.maxLamps) &* 7717)
            let nightH = pcg(UInt32(bitPattern: Int32(truncatingIfNeeded: nightIndex)) &* 1597334677 ^ pcg(7 &* 3812015801 ^ h))
            return Double(nightH & 0xffffff) / 16777215.0 <= chance
        }
        switch site.kind {
        case .hamlet: return site.lamps.contains { $0.kind == 5 && burns($0, 0.35) } ? 1.0 : 0.9
        case .camp, .cave: return 0.8
        case .jetty: return 0.7
        case .henge: return site.lamps.contains { $0.kind == 6 && burns($0, 0.3) } ? 0.8 : 0
        default: return 0
        }
    }

    func shotScore(_ heading: Double, lightYaw: Double, lightW: Double, noon: Double, night: Double = 0) -> Double {
        let p = SIMD2(position.x, position.z)
        var rise = -Double.infinity, nearAngle = -Double.infinity
        for dist in stride(from: 30.0, through: 100.0, by: 14.0) {
            for off in [-0.25, 0.0, 0.25] {
                let dh = h(p.x + sin(heading + off) * dist, p.y + cos(heading + off) * dist) - position.y
                rise = max(rise, dh)
                nearAngle = max(nearAngle, atan2(dh, dist))
            }
        }
        let wall = -0.8 * min(max(rise / 15, 0), 1)
        let fill = -0.6 * noon * min(max((nearAngle - (pitch - 0.25)) / 0.25, 0), 1)
        var sight = 400.0
        for dist in stride(from: 40.0, through: 400.0, by: 20.0)
        where h(p.x + sin(heading) * dist, p.y + cos(heading) * dist) > position.y + dist * tan(pitch) {
            sight = dist
            break
        }
        let open = 0.4 * noon * sight / 400
        var sum = 0.0, sum2 = 0.0, wet = 0.0, n = 0.0
        for dist in stride(from: 80.0, through: 400.0, by: 40.0) {
            for off in [-0.3, -0.15, 0.0, 0.15, 0.3] {
                let a = heading + off
                let hq = h(p.x + sin(a) * dist, p.y + cos(a) * dist)
                sum += hq; sum2 += hq * hq; n += 1
                if hq <= 20.0 { wet += 1 }
            }
        }
        let mean = sum / n
        let relief = min(sqrt(max(sum2 / n - mean * mean, 0)) / 25, 1)
        let wf = wet / n
        let pool = 0.4 + 0.25 * noon
        let water = wf < 0.1 ? 0 : (wf < 0.45 ? pool : pool - 2.0 * (wf - 0.45))
        let light = lightW * max(0, cos(heading - lightYaw)) * 0.8
        let turn = -0.25 * abs(FlightCamera.angleDiff(heading, yaw))
        return relief + water + light + turn + wall + fill + open + siteInterest(heading, night: night).0 + towerInterest(heading, night: night) + { let v = vista(heading); return v.0 * (1 - night) - v.1 }()
            + giantInterest(heading) + isleInterest(heading)
    }

    // A big island 0.3-3 km ahead is a subject, framed head on: the flight goes on to pass beneath it.
    // So is a way between two islands, or beneath one, within the next 1.8 km.
    private func isleInterest(_ heading: Double) -> Double {
        let p = SIMD2(position.x, position.z)
        let fd = SIMD2(sin(heading), cos(heading)), side = SIMD2(fd.y, -fd.x)
        var best = 0.0, left = Double.infinity, right = Double.infinity, under = 0.0
        for s in islesAt() where s.radius > 40 {
            let v = s.pos - p, d = simd_length(v)
            guard d > 200, d < 3000 else { continue }
            if s.radius > 60 {
                let off = abs(FlightCamera.angleDiff(atan2(v.x, v.y), heading))
                best = max(best, 0.8 * max(0, 1 - off / 0.35) * min(1, (3000 - d) / 1000) * (s.passable ? 1 : 0.4))
            }
            let along = simd_dot(v, fd), across = simd_dot(v, side)
            guard along > 150, along < 1800 else { continue }
            // Clear of the island's rock by its reach either side, or under it where it is passable.
            let gap = abs(across) - s.reach
            if gap > 0 { if across > 0 { right = min(right, gap) } else { left = min(left, gap) } }
            else if s.passable && abs(across) < 0.5 * s.reach { under = max(under, 0.7) }
        }
        let between = left < 600 && right < 600 ? 0.6 * (1 - max(left, right) / 600) + 0.3 : 0
        return best + max(between, under)
    }

    // A giant tree is a landmark: a heading that frames one 0.4-3 km ahead scores, and one far off draws the flight
    // slowly toward its country. Passing within a crown's reach is the steering's business (step).
    private func giantInterest(_ heading: Double) -> Double {
        let p = SIMD2(position.x, position.z)
        var best = 0.0
        for g in world.giants {
            let v = g.pos - p, d = simd_length(v)
            // Not once close: framing it from a few hundred off would circle the flight in under its crown.
            guard d > 700, d < 9000 else { continue }
            let off = abs(FlightCamera.angleDiff(atan2(v.x, v.y), heading))
            if d < 3500 {
                // Framed to one side, so the flight passes it a few hundred voxels off.
                let aim = atan2(GiantInfo.reach + 260, d)
                best = max(best, 0.7 * max(0, 1 - abs(off - aim) / 0.3) * min(1, (3500 - d) / 1000))
            } else {
                best = max(best, 0.25 * max(0, 1 - off / 0.7))
            }
        }
        return best
    }
}
