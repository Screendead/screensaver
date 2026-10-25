import StrataCore
import simd

extension Snapshot {
    // VR_SHOW=kind,distance,height,bearing[,rank]: hold the camera on the rank-th nearest site of a kind
    // (1 henge .. 6 camp), `distance` from it along `bearing` degrees, `height` above the ground there.
    static func showSite(_ r: Renderer, spec: String, pitch: Double?) {
        let v = spec.split(separator: ",").compactMap { Double($0) }
        if v.count >= 4 {
            let here = r.camera.xz
            let pick = r.world.terrain.sites.filter { $0.kind.rawValue == Int(v[0]) }
                .sorted { simd_length($0.pos - here) < simd_length($1.pos - here) }
            let rank = v.count > 4 ? Int(v[4]) : 0
            if rank < pick.count {
                let site = pick[rank]
                let b = v[3] * .pi / 180
                let cam = site.pos + SIMD2(sin(b), cos(b)) * v[1]
                let g = r.world.terrain.height(x: cam.x, z: cam.y) ?? 20
                let sg = r.world.terrain.height(x: site.pos.x, z: site.pos.y) ?? 20
                let y = max(g, Sight.sea) + v[2]
                let yaw = atan2(site.pos.x - cam.x, site.pos.y - cam.y)
                let pitch = atan2(sg + 2 - y, v[1]) + (pitch ?? 0)
                r.camera.freeze(position: SIMD3(cam.x, y, cam.y), yaw: yaw, pitch: pitch)
                r.camera.pitchOverride = nil
                print(String(format: "showing %@ site %u at (%.0f, %.0f), %d of %d", "\(site.kind)", site.id, site.pos.x, site.pos.y, rank + 1, pick.count))
            } else {
                print("no site of kind \(Int(v[0])) rank \(rank) (\(pick.count) on this map)")
            }
        }
    }

    // VR_CAVE=distance,height,bearing[,rank[,homes]]: hold the camera on the rank-th nearest cave (with homes 1, of those
    // someone lives in), `distance` out from its mouth turned `bearing` degrees off its axis, `height` over the ground
    // there or the cave's floor, whichever is higher, looking at the mouth.
    static func showCave(_ r: Renderer, spec: String, pitch: Double?) {
        let v = spec.split(separator: ",").compactMap { Double($0) }
        if v.count >= 3 {
            let here = SIMD2<Float>(Float(r.camera.position.x), Float(r.camera.position.z))
            // homes: 1 lived-in caves only; 2 any cave of a good size seen clear across `distance` (by the CPU heights); 3 grottoes.
            let b0 = v[2] * .pi / 180
            func clear(_ c: CaveRec) -> Bool {
                let m = SIMD2(Double(c.m.x), Double(c.m.y)), u = SIMD2(Double(c.u.x), Double(c.u.y))
                let dir = SIMD2(u.x * cos(b0) - u.y * sin(b0), u.x * sin(b0) + u.y * cos(b0))
                guard c.w >= 6, c.hc >= 8, let g = r.world.terrain.height(x: m.x - dir.x * v[0], z: m.y - dir.y * v[0]) else { return false }
                let y0 = max(g, Double(c.yF)) + v[1], y1 = Double(c.yF) + 0.5 * Double(c.hc)
                for t in stride(from: 12.0, to: v[0], by: 6.0) {
                    let q = m - dir * t
                    guard let h = r.world.terrain.height(x: q.x, z: q.y) else { return false }
                    if h > y1 + (y0 - y1) * t / v[0] { return false }
                }
                return true
            }
            let mode = v.count > 4 ? Int(v[4]) : 0
            let pick = r.world.terrain.caves.filter { mode == 0 || (mode == 1 && $0.flags & 2 != 0) || (mode == 2 && clear($0)) || (mode == 3 && $0.flags & 4 != 0) }
                .sorted { simd_length($0.m - here) < simd_length($1.m - here) }
            let rank = v.count > 3 ? Int(v[3]) : 0
            if rank < pick.count {
                let c = pick[rank]
                let m = SIMD2(Double(c.m.x), Double(c.m.y)), u = SIMD2(Double(c.u.x), Double(c.u.y))
                let b = v[2] * .pi / 180
                let dir = SIMD2(u.x * cos(b) - u.y * sin(b), u.x * sin(b) + u.y * cos(b))
                let cam = m - dir * v[0]
                let g = r.world.terrain.height(x: cam.x, z: cam.y) ?? 20
                let y = max(max(g, Sight.sea), Double(c.yF)) + v[1]
                let yaw = atan2(m.x - cam.x, m.y - cam.y)
                let pitch = atan2(Double(c.yF) + 0.4 * Double(c.hc) - y, v[0]) + (pitch ?? 0)
                r.camera.freeze(position: SIMD3(cam.x, y, cam.y), yaw: yaw, pitch: pitch)
                r.camera.pitchOverride = nil
                print(String(format: "showing cave at (%.0f, %.0f) floor %.0f W %.1f H %.0f L %.0f%@, %d of %d; VR_CAM=%.1f,%.1f,%.1f,%.2f,%.2f,1", c.m.x, c.m.y, c.yF, c.w, c.hc, c.l,
                             c.flags & 2 != 0 ? " home" : "", rank + 1, pick.count, cam.x, cam.y, y, yaw * 180 / .pi, pitch * 180 / .pi))
            } else {
                print("no cave rank \(rank) (\(pick.count) in the table)")
            }
        }
    }
}
