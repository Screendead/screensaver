import Foundation
import simd

extension Wildlife {
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
    func predicted(_ s: Sight, tau: Double, turnFor: Double = 3) -> Sight {
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
    func hidden(_ pts: [SIMD3<Double>], size: Double) -> Bool {
        guard let s = sight else { return true }
        if hiddenFrames.isEmpty { hiddenFrames = [0.0, 1.0, 2.0].map { (predicted(s, tau: $0), 0.04 + 0.03 * $0) } }
        let frames = hiddenFrames
        for (f, m) in frames {
            for p in pts where f.sees(p, size: size, margin: m) { return false }
        }
        return true
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
