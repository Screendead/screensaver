import Foundation
import simd

// Valley-following flight. Positions are world-space Doubles; the renderer rebases them onto the map.
final class FlightCamera {
    private(set) var position: SIMD3<Double>
    private(set) var yaw: Double
    private(set) var pitch: Double = -0.18
    private(set) var roll: Double = 0
    private var yawRate: Double = 0
    private var yawAccel: Double = 0
    private var climbRate: Double = 0
    private var time: Double = 0
    private(set) var surveyAmount: Double = 0
    private var facing: Double = 0
    private var shotYaw: Double?
    private var nextShot: Double = 0
    private var rimLift: Double = 0
    var sun = SIMD3<Double>(0, 1, 0)
    var pitchOverride: Double?
    // A fixed pose for showcase snapshots; step() leaves it alone.
    private(set) var frozen = false

    func freeze(position p: SIMD3<Double>, yaw y: Double, pitch pt: Double) {
        position = p
        yaw = y
        pitch = pt
        roll = 0
        frozen = true
    }

    // Height lookup in world coords; returns nil outside the loaded map.
    var heightAt: (Double, Double) -> Double? = { _, _ in nil }
    // Coarse (32-voxel max) heights out to about 7.7k; nil beyond or before the first readback.
    var farHeightAt: (Double, Double) -> Double? = { _, _ in nil }
    // Flight stays this far under the cloud base (CLOUD_Y0 in Shaders/Saver.metal, 1065).
    static let ceiling = 1065.0 - 80.0
    private var avoidRate: Double = 0
    private var nextRoute: Double = 0
    private var routeYaw: Double?
    private var blockedAhead: Double = 0
    // 0...1, eased: how much a slope close by fills the frame (looked down onto, or rising beside the flight).
    private var hug: Double = 0
    // Structure sites on the loaded map.
    var sitesAt: () -> [SiteInfo] = { [] }
    // Ruined towers round the camera (Sources/Towers.swift).
    var towersAt: () -> [TowerInfo] = { [] }
    // Index of the current night, as the lamps read it (beacons and rites burn only on some nights).
    var nightIndex = 0
    // Sites a shot was built around, and when; each rests for a while so one place can't own the flight.
    private var framedAt: [UInt32: Double] = [:]
    private var focus: SIMD2<Double>?

    init(seed: UInt32) {
        let s = Double(seed % 100_000)
        position = SIMD3(s * 7.31, 70, s * 3.17)
        yaw = s.truncatingRemainder(dividingBy: 6.283)
        // Decorrelates the wander/pitch cycles across seeds and displays.
        time = Double(seed % 997) * 37.0
    }

    var forward: SIMD3<Double> {
        SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    }

    // Unit right/up vectors including roll.
    var basis: (fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>) {
        let f = forward
        let r0 = simd_normalize(simd_cross(SIMD3<Double>(0, 1, 0), f))
        let u0 = simd_cross(f, r0)
        let r = r0 * cos(roll) + u0 * sin(roll)
        let u = simd_cross(f, r)
        return (f, r, u)
    }

    private func h(_ x: Double, _ z: Double) -> Double { heightAt(x, z) ?? farHeightAt(x, z) ?? 20 }

    // Starts the flight here (tests: VR_START).
    func start(at p: SIMD2<Double>, yaw y: Double) {
        position.x = p.x
        position.z = p.y
        yaw = y
    }

    // A ruined tower as the routing sees it: a solid drum to its full height, a little wider than its plinth, so the
    // sparse samples along a heading cannot pass between them and the flight turns off well before its walls.
    private func towerTop(_ x: Double, _ z: Double) -> Double {
        var m = -Double.infinity
        for t in towersAt() where simd_length(SIMD2(x, z) - t.pos) < t.radius * 1.5 + 60 { m = max(m, t.top) }
        return m
    }
    private func hSteer(_ x: Double, _ z: Double) -> Double { max(h(x, z), towerTop(x, z)) }
    // Far terrain: the 32-voxel max grid, else the fine map.
    private func hFar(_ x: Double, _ z: Double) -> Double { farHeightAt(x, z) ?? heightAt(x, z) ?? 20 }

    // How much terrain along a heading rises above what the flight could clear: the slope it may climb (capped at
    // the ceiling), weighted toward near walls. Gorges and passes are the open headings.
    private func blocked(_ heading: Double) -> Double {
        let dir = SIMD2(sin(heading), cos(heading))
        let side = SIMD2(dir.y, -dir.x)
        var b = 0.0
        for dist in [40.0, 60.0, 80.0, 105.0, 135.0, 170.0, 210.0, 260.0, 320.0, 400.0, 500.0, 650.0, 850.0, 1100.0, 1500.0] {
            let q = SIMD2(position.x, position.z) + dir * dist
            // The flight's width: a pillar between samples must still count.
            let w = min(dist * 0.12, 30)
            let l = q + side * w, r = q - side * w
            let g = dist < 400 ? max(hSteer(q.x, q.y), hSteer(l.x, l.y), hSteer(r.x, r.y)) : max(hFar(q.x, q.y), towerTop(q.x, q.y))
            let allow = min(FlightCamera.ceiling - 30, position.y + 0.18 * dist)
            b += max(0, g - allow) / (1 + dist / 250)
        }
        return b
    }

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
                guard let g = dist < 300 ? heightAt(q.x, q.y) : farHeightAt(q.x, q.y) else { continue }
                let e = atan2(g - position.y, dist)
                if dist >= 1000 { best = max(best, e) } else { near = max(near, e) }
            }
        }
        let deg = best * 180 / .pi, nearDeg = near * 180 / .pi
        let reward = deg < 2 ? max(0, deg / 2) : deg < 12 ? 1 : max(0, 1 - (deg - 12) / 8)
        return (0.6 * reward, 1.2 * min(max((nearDeg - 15) / 10, 0), 1))
    }

    // Share of the frame's width (11 bearings across the horizontal field) where land within 220 rises above the eye:
    // a wall filling the frame (tests).
    func wallInFrame() -> Double {
        var n = 0.0
        for i in -5...5 {
            let a = yaw + Double(i) * 0.12
            for dist in stride(from: 25.0, through: 220.0, by: 15.0) where h(position.x + sin(a) * dist, position.z + cos(a) * dist) > position.y {
                n += 1
                break
            }
        }
        return n / 11
    }

    private func footprintMax(_ x: Double, _ z: Double, radius: Double) -> Double {
        var m = 0.0
        for dx in stride(from: -radius, through: radius, by: radius / 2) {
            for dz in stride(from: -radius, through: radius, by: radius / 2) { m = max(m, h(x + dx, z + dz)) }
        }
        return m
    }

    // Slow wander from incommensurate sines: smooth, deterministic, effectively non-repeating.
    private func wander(_ t: Double, _ k: Double) -> Double {
        sin(t * 0.0131 * k + 1.3) * 0.5 + sin(t * 0.0077 * k + 4.1) * 0.3 + sin(t * 0.0031 * k + 2.7) * 0.2
    }

    private static func angleDiff(_ a: Double, _ b: Double) -> Double {
        atan2(sin(a - b), cos(a - b))
    }

    // Shot director: score a candidate heading by what the frame would hold. Relief across the middle
    // distance gives a subject; some water is welcome but a frame full of it is not; facing a low sun or
    // the moon puts the light in shot; large turns cost a little so motion stays calm; land rising past eye
    // height close ahead would fill the frame with a wall. Under a high sun close slopes read as blocks, so
    // the shot looks out: open sightlines and water count for more, and near land climbing into the lower
    // half of the frame (bottom edge 0.436 rad under the pitch) costs.
    // Interest of the sites a heading frames: by day the old stones and the villages, at night only what is lit.
    // Capped, so a site adds to a shot but never outvotes the land and the light.
    private func siteInterest(_ heading: Double, night: Double) -> (Double, SiteInfo?) {
        let p = SIMD2(position.x, position.z)
        var sum = 0.0, best = 0.0
        var top: SiteInfo?
        for site in sitesAt() {
            if let t = framedAt[site.id], time - t < 240 { continue }
            let v = site.pos - p
            let d = simd_length(v)
            guard d > 90, d < 450 else { continue }
            if abs(FlightCamera.angleDiff(atan2(v.x, v.y), heading)) > 0.35 { continue }
            let day: [SiteInfo.Kind: Double] = [.henge: 1, .ruins: 0.7, .hamlet: 0.7, .jetty: 0.5, .camp: 0.3, .cairn: 0.2]
            let w = (day[site.kind] ?? 0) * (1 - night) + litInterest(site) * night
            let f = w * min(max((d - 90) / 60, 0), 1) * (1 - min(max((d - 300) / 150, 0), 1))
            sum += f
            if f > best { best = f; top = site }
        }
        return (min(0.5 * sum, 0.35 + 0.15 * night), top)
    }

    // A ruined tower 1.2-9k off, framed when the heading passes about 900 off one of its flanks: by day the strongest
    // landmark there is, at night (a few lit rooms on some) less so. The flank bearings are offered as shots.
    private func towerFlanks() -> [(bearing: Double, tower: TowerInfo, d: Double)] {
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
            let w = (0.55 * (1 - night) + (f.tower.lit ? 0.3 : 0.1) * night) * (1 - a / 0.35) * min(1, (9000 - f.d) / 2000)
            best = max(best, w)
        }
        return best
    }

    // Hearths, camp fires and jetty lanterns burn every night; a henge is only worth a night shot when its
    // rite is lit, and a beacon adds to its hamlet. Mirrors lampLevel in Shaders/Saver.metal.
    private func litInterest(_ site: SiteInfo) -> Double {
        func pcg(_ v: UInt32) -> UInt32 {
            let state = v &* 747796405 &+ 2891336453
            let word = ((state >> ((state >> 28) &+ 4)) ^ state) &* 277803737
            return (word >> 22) ^ word
        }
        func burns(_ l: SiteInfo.Lamp, _ chance: Double) -> Bool {
            let h = pcg(site.id &+ UInt32(l.index % World.maxLamps) &* 7717)
            let nightH = pcg(UInt32(bitPattern: Int32(truncatingIfNeeded: nightIndex)) &* 1597334677 ^ pcg(7 &* 3812015801 ^ h))
            return Double(nightH & 0xffffff) / 16777215.0 <= chance
        }
        switch site.kind {
        case .hamlet: return site.lamps.contains { $0.kind == 5 && burns($0, 0.35) } ? 1.0 : 0.9
        case .camp: return 0.8
        case .jetty: return 0.7
        case .henge: return site.lamps.contains { $0.kind == 6 && burns($0, 0.3) } ? 0.8 : 0
        default: return 0
        }
    }

    private func shotScore(_ heading: Double, lightYaw: Double, lightW: Double, noon: Double, night: Double = 0) -> Double {
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
    }

    func placeAtStart() {
        // A start among or beside the massifs moves to the nearest open low ground, so the flight begins under the ceiling.
        if (0..<12).contains(where: { (farHeightAt(position.x + sin(Double($0) * .pi / 6) * 500, position.z + cos(Double($0) * .pi / 6) * 500) ?? 0) > 260 })
            || hFar(position.x, position.z) > 200 {
            search: for r in stride(from: 200.0, through: 6000.0, by: 200.0) {
                for k in 0..<24 {
                    let a = Double(k) * .pi / 12
                    let x = position.x + sin(a) * r, z = position.z + cos(a) * r
                    guard let g = farHeightAt(x, z), g < 120 else { continue }
                    // Clear of the massifs on every side, so the first seconds are not a climb up a wall.
                    var clear = true
                    for rr in [250.0, 500.0, 800.0] {
                        for j in 0..<12 where (farHeightAt(x + sin(Double(j) * .pi / 6) * rr, z + cos(Double(j) * .pi / 6) * rr) ?? 0) > 260 { clear = false }
                    }
                    if clear { position.x = x; position.z = z; break search }
                }
            }
        }
        position.y = max(position.y, footprintMax(position.x, position.z, radius: 24) + 30)
    }

    func step(dt: Double) {
        if frozen { return }
        time += dt
        let fwdFlat = SIMD2(sin(yaw), cos(yaw))
        let left = SIMD2(-fwdFlat.y, fwdFlat.x) * -1
        let p = SIMD2(position.x, position.z)

        // Steer toward the lower side of a look-ahead point, so the flight follows valleys.
        var steer = 0.0
        for (dist, weight) in [(90.0, 0.5), (180.0, 0.35), (280.0, 0.15)] {
            let a = p + fwdFlat * dist
            let l = a + left * dist * 0.35, r = a - left * dist * 0.35
            steer += weight * (h(r.x, r.y) - h(l.x, l.y))
        }
        // A slow mode: mostly follow valleys, but about a third of the time run ridges or cut across.
        let mode = wander(time + 2000, 0.35)
        let m01 = min(max((mode + 0.2) / 0.6, 0), 1)
        let steerSign = m01 * m01 * (3 - 2 * m01) * 2 - 1
        // When the sun or moon is low, turn toward it so its disc and water glitter come into frame.
        let key = sun.y > -0.05 ? sun : SIMD3(-sun.x, max(-sun.y * 0.7, 0.08), -sun.z)
        let keyYaw = atan2(key.x, key.z)
        let w = min(max(1 - abs(key.y - 0.15) / 0.2, 0), 1)
        let steerKey = sin(keyYaw - yaw) * 0.06 * w
        facing = w * max(0, cos(keyYaw - yaw))
        // Night: the moon (matching L.moon in the shader) is the focal light; turn toward it.
        let nightW = min(max((-sun.y - 0.2) / 0.1, 0), 1)
        let moon = simd_normalize(SIMD3(-sun.x, max(-sun.y * 0.7, 0.08), -sun.z))
        let moonYaw = atan2(moon.x, moon.z)
        let steerMoon = sin(moonYaw - yaw) * 0.05 * nightW
        let moonOff = abs(FlightCamera.angleDiff(moonYaw, yaw))
        // The moon sits fully in frame within ~0.5 rad of the heading (horizontal half-field ~0.62 rad).
        let facingMoon = nightW * min(max((1.0 - moonOff) / 0.5, 0), 1)
        let noon = min(max((sun.y - 0.28) / 0.12, 0), 1)
        if time >= nextShot {
            let lightW = max(w, nightW)
            let lightYaw = nightW > w ? moonYaw : keyYaw
            // At night the shots are only those that hold the moon; terrain picks among them. Free headings let
            // relief outvote the moon wherever the land happens to rise away from it.
            let (center, span, step) = nightW > 0.5 ? (moonYaw, 3, 0.15) : (yaw, 3, 0.5)
            var headings: [Double] = (-span...span).map { center + Double($0) * step }
            // At night lit settlements near the moon's bearing join the shots, so a flight passes them and the
            // moon still frames it.
            if nightW > 0.5 {
                let p = SIMD2(position.x, position.z)
                for site in sitesAt() where litInterest(site) > 0 {
                    let v = site.pos - p
                    let b = atan2(v.x, v.y)
                    if simd_length(v) < 450, abs(FlightCamera.angleDiff(b, moonYaw)) < 1.0 { headings.append(b) }
                }
            }
            headings += towerFlanks().map { $0.bearing }.filter { abs(FlightCamera.angleDiff($0, yaw)) < 1.6 }
            let scored = headings.map { ($0, shotScore($0, lightYaw: lightYaw, lightW: lightW, noon: noon, night: nightW)) }
            let pick = scored.max { $0.1 < $1.1 }!
            shotYaw = pick.0
            let (interest, site) = siteInterest(pick.0, night: nightW)
            focus = nil
            if interest > 0.15, let site {
                framedAt[site.id] = time
                focus = site.pos
            }
            nextShot = time + 12
        }
        let steerShot = shotYaw.map { sin(FlightCamera.angleDiff($0, yaw)) * 0.05 } ?? 0
        let valley: Double = steer * 0.004 * steerSign * (1 - 0.6 * w) * 0.4
        let drift: Double = (valley + wander(time, 1) * 0.02) * (1 - nightW)
        // Routing: every second, compare headings by how much terrain they would have the flight climb; when the way
        // ahead is blocked, turn toward the most open one (a gorge, a pass, the lowland) rather than climb the wall.
        if time >= nextRoute {
            nextRoute = time + 1
            let here = blocked(yaw)
            blockedAhead = here
            if here > 2 {
                var best = (yaw, here)
                for off in [0.25, -0.25, 0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 1.7, -1.7] {
                    let b = blocked(yaw + off) + abs(off) * 2
                    if b < best.1 { best = (yaw + off, b) }
                }
                routeYaw = best.0
            } else {
                routeYaw = nil
            }
        }
        // Walls close beside or ahead push the heading away, so the flight keeps a frame's width off a massif's face
        // (close up, a steep face is a lattice of single steps).
        var push = SIMD2<Double>.zero
        for k in 0..<12 {
            let a = Double(k) * .pi / 6
            let dv = SIMD2(sin(a), cos(a))
            for r in [30.0, 60.0, 110.0, 180.0, 260.0, 360.0] {
                let q = p + dv * r
                let over = hSteer(q.x, q.y) - (position.y - 10)
                if over > 0 { push -= dv * min(over / 60, 1) * (60 / r) }
            }
        }
        let pushLen = simd_length(push)
        var steerAway = 0.0
        let crowd = min(max((pushLen - 0.25) / 1.0, 0), 1)
        if crowd > 0 {
            let awayYaw = atan2(push.x, push.y)
            steerAway = sin(FlightCamera.angleDiff(awayYaw, yaw)) * 0.2 * crowd
        }
        let urgency = max(min(blockedAhead / 20, 1), 0.7 * crowd)
        avoidRate = routeYaw.map { sin(FlightCamera.angleDiff($0, yaw)) * 0.15 * urgency } ?? 0
        // The birds' sightline predictions (Wildlife.hidden) rely on turns of at most 0.09 rad/s.
        let targetRate = max(-0.09, min(0.09, (drift + steerShot + steerKey + steerMoon) * (1 - urgency) + avoidRate + steerAway))
        // Critically damped spring on yaw rate: turns take tens of seconds.
        let k = 0.25
        yawAccel += (k * k * (targetRate - yawRate) - 2 * k * yawAccel) * dt
        yawRate += yawAccel * dt
        yaw += yawRate * dt

        // Constant angular ground flow: faster when high, slower when skimming.
        let agl = position.y - h(position.x, position.z)
        let v = min(15, max(5.5, 3.5 + agl * 0.17))
        position.x += sin(yaw) * v * dt
        position.z += cos(yaw) * v * dt

        // Altitude: clear the forward footprint by a slowly breathing margin; lifts over ridges become reveals.
        var ahead = footprintMax(position.x, position.z, radius: 8)
        for dist in [20.0, 45.0, 75.0, 110.0] {
            ahead = max(ahead, footprintMax(position.x + fwdFlat.x * dist, position.z + fwdFlat.y * dist, radius: dist < 50 ? 5 : 8) - dist * 0.12)
        }
        // Dense slopes close up read as block fields; climb over rough ground so they read as contours.
        var lo = Double.infinity, hi = -Double.infinity
        for (dist, side) in [(25.0, 0.0), (50.0, 0.0), (35.0, 18.0), (35.0, -18.0), (0.0, 0.0)] {
            let q = p + fwdFlat * dist + left * side
            let hq = h(q.x, q.y)
            lo = min(lo, hq); hi = max(hi, hq)
        }
        let roughClimb = min(max((hi - lo - 6) * 1.5, 0), 30)
        var clearance = roughClimb + 12 + 12 * (wander(time + 500, 0.7) + 0.6) + 10 * surveyAmount + 40 * max(0, min(1, -mode / 0.6)) + 14 * noon
        // Over the great relief the flight keeps well up: close above, its stepped slopes read as a block lattice.
        let groundHere = h(position.x, position.z)
        clearance += 70 * min(max((groundHere - 150) / 200, 0), 1)
        // A slope close by filling the frame (the view ray meets it within ~150, or land within 220 rises above the eye
        // across much of the width) is a dull, dark framing: lift away from it and raise the view toward the far land.
        // Rays at the frame's centre and a quarter of the way up it.
        var meet = Double.infinity, meetUp = Double.infinity
        let b3 = basis
        let up3 = simd_normalize(b3.fwd + b3.up * tan(0.12))
        for dist in stride(from: 20.0, through: 240.0, by: 14.0) {
            let q = position + b3.fwd * dist, u = position + up3 * dist
            if meet.isInfinite, h(q.x, q.z) > q.y { meet = dist }
            if meetUp.isInfinite, h(u.x, u.z) > u.y { meetUp = dist }
        }
        let hugNow = max(max(min(max((150 - meet) / 90, 0), 1), min(max((240 - meetUp) / 120, 0), 1)),
                         min(max((wallInFrame() - 0.3) / 0.4, 0), 1))
        hug += (hugNow - hug) * min(1, dt * (hugNow > hug ? 0.3 : 0.08))
        clearance += 25 * hug
        // Passing the site a shot was built around: a little lower, so the windows read.
        if let f = focus, simd_length(f - p) < 150 { clearance = max(clearance - 8, min(clearance, 22)) }
        // Canyon and mesa country: down among the walls they fill the frame, so rise to look across the rims.
        // The rim level is the upper quartile of a 150-voxel neighbourhood, so a lone peak off to the side does
        // not lift the flight, or the highest rim ahead, out to where a wall would still fill the frame. Eased
        // in, so a rim coming into range is a climb, not a pop; let go slowly, so the flight does not sink back
        // between rims it has just cleared.
        var ring: [Double] = []
        var rimAhead = -Double.infinity
        for r in [50.0, 100.0, 150.0, 200.0, 250.0] {
            for k in 0..<8 where r <= 150 || k <= 1 || k == 7 {
                let a = yaw + Double(k) * .pi / 4
                let hq = h(p.x + sin(a) * r, p.y + cos(a) * r)
                if r <= 150 { ring.append(hq) }
                if k <= 1 || k == 7 { rimAhead = max(rimAhead, hq) }
            }
        }
        ring.sort()
        let rim = max(ring[ring.count * 3 / 4], rimAhead)
        let rough = min(max((ring.last! - ring.first! - 25) / 30, 0), 1)
        let base = max(ahead, 19.85) + clearance
        let lift = rough * max(0, rim + 18 - base)
        rimLift += (lift - rimLift) * min(1, dt * (lift > rimLift ? 0.3 : 0.04))
        // The lift stops well under the cloud slab (among lowland rims at 200), and the flight never targets above the
        // ceiling. Among the great walls it rises to look across them rather than fly at their feet.
        let liftTop = rim > 260 ? FlightCamera.ceiling - 30 : 200
        var target = min(base + min(rimLift, max(0, liftTop - base)), FlightCamera.ceiling)
        // Relief rising ahead is met early, at the slow climb, so the flight arrives over a massif's shoulder rather
        // than having to leap it: land over 280 within 1.5k ahead sets a floor, eased with distance.
        var regional = -Double.infinity
        for dist in stride(from: 250.0, through: 1500.0, by: 125.0) {
            for off in [-0.3, 0.0, 0.3] {
                let g = hFar(position.x + sin(yaw + off) * dist, position.z + cos(yaw + off) * dist)
                if g > 280 { regional = max(regional, g + 30 - 0.15 * (dist - 250)) }
            }
        }
        target = max(target, min(regional, FlightCamera.ceiling))
        let kc = 0.35
        climbRate += (kc * kc * (target - position.y) - 2 * kc * climbRate) * dt
        // Climbs are capped (3 voxels/s): mountains are flown around and through, not leapt. A wall the routing
        // could not turn from lifts the flight just fast enough to clear it by the time it arrives, still smoothly.
        var need = 0.0
        for dist in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 130.0, 180.0] {
            let rise = footprintMax(position.x + fwdFlat.x * dist, position.z + fwdFlat.y * dist, radius: dist == 0 ? 14 : 8) + 12 - position.y
            if rise > 0 { need = max(need, rise / max((dist - 8) / v, 0.5)) }
        }
        if need > 3 { climbRate = max(climbRate, min(need, 40)) }
        climbRate = min(climbRate, max(3, min(1.3 * need, 40)))
        position.y += climbRate * dt
        let floor = footprintMax(position.x, position.z, radius: 4) + 8
        if position.y < floor { position.y = floor; climbRate = max(climbRate, 0) }

        // Survey/vista cycle: the horizon sweeps the whole frame every few minutes so no region of the
        // screen stays sky (or land) for long. Smoothstep dwells at each extreme.
        var s = 0.5 + 0.5 * sin(time * 2 * .pi / 230 + 0.6 * sin(time * 0.0047))
        if sun.y < -0.2 { s *= 0.65 } // keep stars and moon in the upper frame at night
        // High sun flattens a straight-down view; favour horizon vistas with ridges and cloud shadows.
        if sun.y > 0.35 { s *= 0.55 }
        s *= 1 - 0.4 * noon
        // Facing a low sun or moon: lift the view so the light and its water path stay in frame.
        s *= 1 - 0.6 * facing
        // Golden and blue hour: keep the horizon in frame; the sky and aerial perspective carry these moments.
        if abs(sun.y) < 0.18 { s = min(s, 0.55) }
        // Over open water, look out at the far shore rather than down at a featureless mirror.
        let look = p + fwdFlat * 70
        if h(look.x, look.y) <= 20.0 { s *= 0.35 }
        var targetPitch = -0.08 + 0.03 * noon - 0.40 * s + 0.05 * facing + 0.03 * wander(time + 900, 1.3) + max(-0.04, min(0.06, climbRate * 0.03))
        // Facing the moon: it sits just inside the top edge (vertical half-field 0.436 rad), leaving the lower
        // frame to land and the moon's path across the water.
        let moonPitch = min(asin(moon.y) - 0.38, 0.13)
        targetPitch += (moonPitch - targetPitch) * facingMoon
        targetPitch += 0.12 * hug
        surveyAmount = s
        pitch += ((pitchOverride ?? targetPitch) - pitch) * min(1, dt * 0.25)
        let targetRoll = max(-0.07, min(0.07, -yawRate * 0.8))
        roll += (targetRoll - roll) * min(1, dt * 0.8)
    }
}
