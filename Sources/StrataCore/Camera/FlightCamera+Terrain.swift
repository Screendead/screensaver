import Foundation
import simd

extension FlightCamera {
    private func beneath(_ x: Double, _ z: Double, under u: Double) -> Bool {
        let d = simd_length(SIMD2(x - position.x, z - position.z))
        return position.y < u - FlightCamera.isleMargin + FlightCamera.isleSink * d / speed
    }
    package func h(_ x: Double, _ z: Double) -> Double {
        if let c = terrain.column(x: x, z: z) {
            if let u = c.under, u - c.ground >= 90, beneath(x, z, under: u) { return c.ground }
            return c.top
        }
        return isleFloor(x, z) ?? far.height(x: x, z: z) ?? 20
    }
    // Beyond the loaded map: inside the reach of an island the flight could pass beneath, the ground under it.
    private func isleFloor(_ x: Double, _ z: Double) -> Double? {
        let q = SIMD2(x, z)
        for s in islesAt() where s.passable && simd_length_squared(q - s.pos) < s.reach * s.reach && beneath(x, z, under: s.keel) {
            return s.base
        }
        return nil
    }
    // The highest the flight may climb near (x, z) while it passes under an island there.
    func isleCeiling(_ x: Double, _ z: Double) -> Double {
        guard let c = terrain.column(x: x, z: z), let u = c.under, u - c.ground >= 90, beneath(x, z, under: u) else { return .infinity }
        return u - FlightCamera.isleCap
    }

    // A ruined tower as the routing sees it: a solid drum to its full height, a little wider than its plinth, so the
    // sparse samples along a heading cannot pass between them and the flight turns off well before its walls.
    private func towerTop(_ x: Double, _ z: Double) -> Double {
        var m = -Double.infinity
        for t in towersAt() where simd_length(SIMD2(x, z) - t.pos) < t.radius * 1.5 + 60 { m = max(m, t.top) }
        return m
    }
    func hSteer(_ x: Double, _ z: Double) -> Double { max(h(x, z), towerTop(x, z)) }
    // Far terrain: the 32-voxel max grid, else the fine map.
    func hFar(_ x: Double, _ z: Double) -> Double { isleFloor(x, z) ?? far.height(x: x, z: z) ?? terrain.height(x: x, z: z) ?? 20 }

    // How much terrain along a heading rises above what the flight could clear: the slope it may climb (capped at
    // the ceiling), weighted toward near walls. Gorges and passes are the open headings.
    func blocked(_ heading: Double) -> Double {
        let dir = SIMD2(sin(heading), cos(heading))
        let side = SIMD2(dir.y, -dir.x)
        var b = 0.0
        for dist in [40.0, 60.0, 80.0, 105.0, 135.0, 170.0, 210.0, 260.0, 320.0, 400.0, 500.0, 650.0, 850.0, 1100.0, 1500.0] {
            let q = xz + dir * dist
            // The flight's width, every 4 voxels at most: a needle between samples must still count.
            let w = min(dist * 0.12, 30)
            let l = q + side * w, r = q - side * w
            var g = dist < 400 ? max(hSteer(q.x, q.y), hSteer(l.x, l.y), hSteer(r.x, r.y)) : max(hFar(q.x, q.y), towerTop(q.x, q.y))
            if dist < 400 {
                let k = (w / 4).rounded(.up)
                for i in stride(from: 1 - k, through: k - 1, by: 1) where i != 0 {
                    let o = q + side * (w * i / k)
                    g = max(g, h(o.x, o.y))
                }
            }
            let allow = min(FlightCamera.ceiling - 30, position.y + 0.18 * dist)
            b += max(0, g - allow) / (1 + dist / 250)
        }
        return b
    }

    // Share of the frame's width (11 bearings across the horizontal field) where land within 220 rises above the eye:
    // a wall filling the frame (tests).
    package func wallInFrame() -> Double {
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

    func footprintMax(_ x: Double, _ z: Double, radius: Double) -> Double {
        var m = 0.0
        for dx in stride(from: -radius, through: radius, by: radius / 2) {
            for dz in stride(from: -radius, through: radius, by: radius / 2) { m = max(m, h(x + dx, z + dz)) }
        }
        return m
    }
}
