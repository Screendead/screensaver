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
    var sun = SIMD3<Double>(0, 1, 0)
    var pitchOverride: Double?

    // Height lookup in world coords; returns nil outside the loaded map.
    var heightAt: (Double, Double) -> Double? = { _, _ in nil }

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

    private func h(_ x: Double, _ z: Double) -> Double { heightAt(x, z) ?? 20 }

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

    func placeAtStart() {
        position.y = max(position.y, footprintMax(position.x, position.z, radius: 24) + 30)
    }

    func step(dt: Double) {
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
        let key = sun.y > -0.05 ? sun : SIMD3(-sun.x, max(-sun.y, 0.08), -sun.z)
        let keyYaw = atan2(key.x, key.z)
        let w = min(max(1 - abs(key.y - 0.15) / 0.2, 0), 1)
        let steerKey = sin(keyYaw - yaw) * 0.06 * w
        facing = w * max(0, cos(keyYaw - yaw))
        let targetRate = max(-0.09, min(0.09, steer * 0.004 * steerSign * (1 - 0.6 * w) + wander(time, 1) * 0.035 + steerKey))
        // Critically damped spring on yaw rate: turns take tens of seconds.
        let k = 0.25
        yawAccel += (k * k * (targetRate - yawRate) - 2 * k * yawAccel) * dt
        yawRate += yawAccel * dt
        yaw += yawRate * dt

        // Constant angular ground flow: faster when high, slower when skimming.
        let agl = position.y - h(position.x, position.z)
        let v = min(11, max(4, 2.5 + agl * 0.12))
        position.x += sin(yaw) * v * dt
        position.z += cos(yaw) * v * dt

        // Altitude: clear the forward footprint by a slowly breathing margin; lifts over ridges become reveals.
        var ahead = footprintMax(position.x, position.z, radius: 8)
        for dist in [20.0, 45.0, 75.0, 110.0] {
            ahead = max(ahead, footprintMax(position.x + fwdFlat.x * dist, position.z + fwdFlat.y * dist, radius: dist < 50 ? 5 : 8) - dist * 0.12)
        }
        let clearance = 18 + 10 * (wander(time + 500, 0.7) + 1.0) + 10 * surveyAmount + 40 * max(0, min(1, -mode / 0.6))
        let target = max(ahead, 19.85) + clearance
        let kc = 0.35
        climbRate += (kc * kc * (target - position.y) - 2 * kc * climbRate) * dt
        position.y += climbRate * dt
        let floor = footprintMax(position.x, position.z, radius: 4) + 14
        if position.y < floor { position.y = floor; climbRate = max(climbRate, 0) }

        // Survey/vista cycle: the horizon sweeps the whole frame every few minutes so no region of the
        // screen stays sky (or land) for long. Smoothstep dwells at each extreme.
        var s = 0.5 + 0.5 * sin(time * 2 * .pi / 230 + 0.6 * sin(time * 0.0047))
        if sun.y < -0.2 { s *= 0.65 } // keep stars and moon in the upper frame at night
        // High sun flattens a straight-down view; favour horizon vistas with ridges and cloud shadows.
        if sun.y > 0.35 { s *= 0.55 }
        // Facing a low sun or moon: lift the view so the light and its water path stay in frame.
        s *= 1 - 0.6 * facing
        // Golden and blue hour: keep the horizon in frame; the sky and aerial perspective carry these moments.
        if abs(sun.y) < 0.18 { s = min(s, 0.55) }
        // Over open water, look out at the far shore rather than down at a featureless mirror.
        let look = p + fwdFlat * 70
        if h(look.x, look.y) <= 20.0 { s *= 0.35 }
        let targetPitch = -0.08 - 0.40 * s + 0.05 * facing + 0.03 * wander(time + 900, 1.3) + max(-0.04, min(0.06, climbRate * 0.03))
        surveyAmount = s
        pitch += ((pitchOverride ?? targetPitch) - pitch) * min(1, dt * 0.25)
        let targetRoll = max(-0.07, min(0.07, -yawRate * 0.8))
        roll += (targetRoll - roll) * min(1, dt * 0.8)
    }
}
