import Foundation
import simd

extension Wildlife {
    package func advance(_ f: Flock, dt: Double, cam: SIMD3<Double>) {
        let fl = Wildlife.flight(f.kind)
        f.age += dt
        f.transit -= dt
        let arrived = f.transit <= 0
        // Leaving, they go no faster than their top speed, even if they glided in faster.
        let target = f.leaving ? min(f.cruise, fl.maxSpeed) : (arrived || f.kind == .starling ? f.hold ?? f.cruise : f.cruise)
        f.speed += (target - f.speed) * min(1, dt * 0.3)

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
                if l < s.reach + 900 && l < bd { bd = l; best = s }
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
            // A flock gliding in may go over its top speed, and eases back to it as its speed falls to its own pace.
            let top = max(fl.maxSpeed, f.speed)
            if s > top { b.vel *= top / s; s = top }
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
    func wing(_ b: Bird, _ fl: Flight) -> (arm: Double, hand: Double, ext: Double, sweep: Double) {
        if b.beat > 0 {
            let arm = fl.bias + fl.amp * sin(b.flap)
            return (arm, arm + 0.55 * fl.amp * sin(b.flap - 1.1), 1, fl.sweep * (0.75 + 0.25 * cos(b.flap)))
        }
        let folded = 1 - fl.glideExt
        return (fl.glideAngle, fl.glideAngle + fl.handDroop, fl.glideExt, fl.sweep + 0.6 * folded)
    }
}
