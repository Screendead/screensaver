import Foundation
import simd

extension FlightCamera {
    // Slow wander from incommensurate sines: smooth, deterministic, effectively non-repeating.
    func wander(_ t: Double, _ k: Double) -> Double {
        sin(t * 0.0131 * k + 1.3) * 0.5 + sin(t * 0.0077 * k + 4.1) * 0.3 + sin(t * 0.0031 * k + 2.7) * 0.2
    }

    static func angleDiff(_ a: Double, _ b: Double) -> Double {
        atan2(sin(a - b), cos(a - b))
    }

    package func step(dt: Double) {
        if frozen { return }
        time += dt
        let fwdFlat = SIMD2(sin(yaw), cos(yaw))
        let left = SIMD2(-fwdFlat.y, fwdFlat.x) * -1
        let p = SIMD2(position.x, position.z)

        // Steer toward the lower side of a look-ahead point, so the flight follows valleys.
        var steer = 0.0, steerRidge = 0.0
        // Ridge running reads side heights no higher than 20 under the flight, so land above it does not pull the heading in.
        let ridgeTop = position.y - 20
        for (dist, weight) in [(90.0, 0.5), (180.0, 0.35), (280.0, 0.15)] {
            let a = p + fwdFlat * dist
            let l = a + left * dist * 0.35, r = a - left * dist * 0.35
            let hr = h(r.x, r.y), hl = h(l.x, l.y)
            steer += weight * (hr - hl)
            steerRidge += weight * (min(hr, ridgeTop) - min(hl, ridgeTop))
        }
        // A slow mode: mostly follow valleys, but about a third of the time run ridges or cut across.
        let mode = wander(time + 2000, 0.35)
        let m01 = min(max((mode + 0.2) / 0.6, 0), 1)
        let steerSign = m01 * m01 * (3 - 2 * m01) * 2 - 1
        if steerSign < 0 { steer = steerRidge }
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
                for site in terrain.sites where litInterest(site) > 0 {
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
                // A turn once begun is kept unless the other side is clearly more open: re-choosing each second between two
                // near-equal sides cancels the turn and leaves the flight climbing the face between them.
                var best = (yaw, here, 0.0)
                for off in [0.25, -0.25, 0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 1.7, -1.7] {
                    let b = blocked(yaw + off) + abs(off) * 2 + (off * routeSide < 0 ? 3 : 0)
                    if b < best.1 { best = (yaw + off, b, off > 0 ? 1 : -1) }
                }
                routeYaw = best.0
                if best.2 != 0 { routeSide = best.2 }
            } else {
                routeYaw = nil
                routeSide = 0
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
        // A giant whose crown the heading would pass under or climb over: turn for its nearer side early, while a
        // gentle turn still clears it (at most 0.09 rad/s, a radius of ~170 at top speed).
        var steerGiant = 0.0
        for g in terrain.giants where g.top > position.y - 25 {
            let v = g.pos - p
            let along = simd_dot(v, fwdFlat), across = v.x * fwdFlat.y - v.y * fwdFlat.x
            let clear = GiantInfo.reach + 140
            guard along > -clear, along < 1000, abs(across) < clear else { continue }
            let w = (1 - max(along, 0) / 1000) * (1 - abs(across) / clear)
            steerGiant += (across >= 0 ? -1 : 1) * 0.09 * min(1, 2.5 * w)
        }
        // The nearest giant not yet reached, up to 12k off, draws the heading gently toward a point beside it; so does a
        // big island within 3k, toward its middle, to pass beneath it; farther off one steers the flight into trees.
        var steerPilgrim = 0.0
        if time >= nextPilgrim {
            nextPilgrim = time + 2
            pilgrim = nil
            var best = 12000.0
            for g in terrain.giants {
                let key = SIMD2(Int(g.pos.x.rounded()), Int(g.pos.y.rounded()))
                if visitedGiants.contains(key) { continue }
                let v = g.pos - p, d = simd_length(v)
                // Within 1500 it is reached: framing it and passing beside it are the shots' and the steering's.
                if d < 1500 { visitedGiants.insert(key); continue }
                if d < best { best = d; pilgrim = g.pos; pilgrimIsle = false }
            }
            for s in islesAt() where s.radius > 120 && s.passable {
                let key = SIMD2(Int(s.pos.x.rounded()), Int(s.pos.y.rounded()))
                if visitedGiants.contains(key) { continue }
                let d = simd_length(s.pos - p)
                if d < 0.4 * s.reach { visitedGiants.insert(key); continue }
                if d < min(best, 3000) { best = d; pilgrim = s.pos; pilgrimIsle = true }
            }
        }
        if let g = pilgrim {
            let v = g - p
            let side = (v.x * fwdFlat.y - v.y * fwdFlat.x) >= 0 ? 1.0 : -1.0
            let n = simd_normalize(SIMD2(-v.y, v.x)) * side
            let aim = pilgrimIsle ? g : g + n * (GiantInfo.reach + 260)
            let bearing = atan2(aim.x - p.x, aim.y - p.y)
            // Not over a massif: where the land toward it rises well above the flight, the way round is the routing's.
            var rise = 0.0
            for d in [300.0, 600.0, 900.0] { rise = max(rise, hFar(p.x + sin(bearing) * d, p.y + cos(bearing) * d) - position.y) }
            // Nor while a slope fills the frame: getting clear of it comes first.
            steerPilgrim = sin(FlightCamera.angleDiff(bearing, yaw)) * (pilgrimIsle ? 0.09 : 0.06) * min(max((60 - rise) / 40, 0), 1) * (1 - hug)
        }
        let urgency = max(min(blockedAhead / 20, 1), 0.7 * crowd)
        avoidRate = routeYaw.map { sin(FlightCamera.angleDiff($0, yaw)) * 0.15 * urgency } ?? 0
        // The birds' sightline predictions (Wildlife.hidden) rely on turns of at most 0.09 rad/s.
        let targetRate = max(-0.09, min(0.09, (drift + steerShot + steerKey + steerMoon + steerPilgrim) * (1 - urgency) + avoidRate + steerAway + steerGiant))
        // Critically damped spring on yaw rate: turns take tens of seconds.
        let k = 0.25
        yawAccel += (k * k * (targetRate - yawRate) - 2 * k * yawAccel) * dt
        yawRate += yawAccel * dt
        yaw += yawRate * dt

        // Constant angular ground flow: faster when high, slower when skimming.
        let agl = position.y - h(position.x, position.z)
        let v = 2 * min(15, max(5.5, 3.5 + agl * 0.17))
        speed = v
        position.x += sin(yaw) * v * dt
        position.z += cos(yaw) * v * dt

        // Altitude: clear the forward footprint by a slowly breathing margin; lifts over ridges become reveals.
        var ahead = footprintMax(position.x, position.z, radius: 8)
        for dist in [20.0, 45.0, 75.0, 110.0] {
            ahead = max(ahead, footprintMax(position.x + fwdFlat.x * dist, position.z + fwdFlat.y * dist, radius: dist < 50 ? 5 : 8) - dist * 0.12)
        }
        // Further along the path the flight is turning onto, out to ~25 s ahead, land is met at a planned 2 voxels/s (the
        // 3/s cap is the reserve), so a pillar or needle field is climbed from far off and the flight holds its height
        // across the gaps rather than dipping into each. The corridor is sampled on the 4-voxel grid the heights are
        // maxima over, so no needle slips between samples. Out to ~40 s, the corridor widened by the path's uncertainty
        // gives the steady climb that clears it on arrival (plan): a spire field taller than the 3/s cap can reach is
        // climbed early at up to 5/s, not late by the wall override.
        let reach = min(max(v * 25, 150), 400)
        var plan = 0.0
        for dist in stride(from: 114.0, through: min(max(v * 40, 240), 600), by: 4.0) {
            let a = yaw + yawRate * dist / v * 0.5
            let c = p + SIMD2(sin(a), cos(a)) * dist, side = SIMD2(cos(a), -sin(a))
            let w = 8 + ((dist - 114) * 0.015).rounded() * 4
            var g = -Double.infinity, gw = -Double.infinity
            for o in stride(from: -w, through: w, by: 4.0) {
                let q = h(c.x + side.x * o, c.y + side.y * o)
                gw = max(gw, q)
                if abs(o) <= 8 { g = max(g, q) }
            }
            if dist <= reach { ahead = max(ahead, g - 13.2 - (dist - 110) * 2 / v) }
            plan = max(plan, (gw + 20 - position.y) / max(dist / v - 4, 1))
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
        // The second-highest sample sets it: one karst pillar alone in a 32-voxel cell does not switch it on.
        var regional = -Double.infinity, top = -Double.infinity
        for dist in stride(from: 250.0, through: 1500.0, by: 125.0) {
            for off in [-0.3, 0.0, 0.3] {
                let g = hFar(position.x + sin(yaw + off) * dist, position.z + cos(yaw + off) * dist)
                guard g > 280 else { continue }
                let f = g + 30 - 0.15 * (dist - 250)
                if f > top { regional = top; top = f } else { regional = max(regional, f) }
            }
        }
        target = max(target, min(regional, FlightCamera.ceiling))
        // At dawn the sea of cloud's top reaches up to 12 x its amount over the sea (at most 14, SEA_LID in Saver.metal):
        // fly well over it, high enough that it reads as a surface below, rising with the amount so the lift is gentle.
        target = max(target, 19.85 + 38 * cloudSea)
        // Under an island: below its underside all the way through, the keel's deepest ahead included.
        var under = Double.infinity
        for dist in [0.0, 30.0, 60.0, 100.0, 140.0, 180.0, 260.0, 360.0] {
            for side in [-16.0, 0.0, 16.0] {
                let q = p + fwdFlat * dist + left * side
                under = min(under, isleCeiling(q.x, q.y))
            }
        }
        target = min(target, under)
        // Climbs are capped (3 voxels/s): mountains are flown around and through, not leapt. A wall the routing
        // could not turn from lifts the flight just fast enough to clear it by the time it arrives, still smoothly.
        var need = 0.0
        for dist in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 130.0, 180.0] {
            let rise = footprintMax(position.x + fwdFlat.x * dist, position.z + fwdFlat.y * dist, radius: dist == 0 ? 14 : 8) + 12 - position.y
            if rise > 0 { need = max(need, rise / max((dist - 8) / v, 0.5)) }
        }
        // Vertical speed eases toward a rate that can still stop at the target, changing by at most 1.2 voxels/s each
        // second, so a climb or a dive never starts or stops with a kick; only a wall's need ramps faster. Sinks are
        // capped at 6/s near the ground and at a constant angular rate (0.04 of the height) higher up.
        let err = target - position.y
        let accel = need > 3 ? 6.0 : 1.2
        let sink = min(max(6, 0.04 * agl), 15)
        var want = err >= 0 ? min(0.3 * err, (2 * accel * err).squareRoot(), 3) : max(0.3 * err, -(2 * accel * -err).squareRoot(), -sink)
        if plan * 1.2 > 3 { want = max(want, min(plan * 1.2, 5)) }
        if need > 3 { want = max(want, min(need, 40)) }
        climbRate += max(-accel * dt, min(accel * dt, want - climbRate))
        position.y += climbRate * dt
        let floor = footprintMax(position.x, position.z, radius: 4) + 8
        if tracing { terms = [target, climbRate, need, rimLift, regional, ahead, plan, blockedAhead, routeYaw.map { FlightCamera.angleDiff($0, yaw) } ?? 0] }
        if position.y < floor { position.y = floor; climbRate = max(climbRate, 0); if tracing { terms[2] = -1 } }

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
