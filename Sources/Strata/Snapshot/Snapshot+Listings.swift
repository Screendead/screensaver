import Metal
import StrataCore
import simd

extension Snapshot {
    // VR_TOWERS=n: lists the towers of the n-cell square of tower cells round the camera's start, nearest first.
    static func listTowers(_ r: Renderer, cells n: Int) {
        let here = r.camera.xz
        for t in r.towers.scan(around: here, cells: n, queue: r.queue).sorted(by: { simd_length($0.pos - here) < simd_length($1.pos - here) }) {
            print(String(format: "tower %.0f,%.0f dist %.0f kind %d member %d R0 %.0f base %.0f height %.0f top %.0f lit %d moist %.2f temp %.2f",
                         t.pos.x, t.pos.y, simd_length(t.pos - here), t.kind, t.member, t.radius, t.base, t.height, t.top, t.lit ? 1 : 0, t.moist, t.temp))
        }
    }

    // VR_ISLES=n: lists the n floating islands nearest the camera's start on its map's table.
    static func listIsles(_ r: Renderer, count n: Int) {
        let here = r.camera.xz
        let list = r.world.terrain.isles.sorted { simd_length($0.pos - here) < simd_length($1.pos - here) }
        print("islands on the table: \(list.count) (massifs \(list.filter(\.massif).count))")
        for s in list.prefix(n) {
            print(String(format: "island %.0f,%.0f dist %.0f R %.0f reach %.0f top %.0f keel %.0f base %.0f%@", s.pos.x, s.pos.y,
                         simd_length(s.pos - here), s.radius, s.reach, s.top, s.keel, s.base, (s.massif ? " massif" : "")
                         + (s.fall.map { String(format: " fall lip %.0f,%.0f top %.0f foot %.0f", $0.x, $0.y, $0.z, $0.w) } ?? "")))
        }
    }

    // Sites within 700 voxels whose centre falls inside the frame (not tested for occlusion).
    static func sitesInView(_ r: Renderer) -> String {
        let b = r.camera.basis
        let o = r.camera.position
        let tanV = Renderer.tanV, tanH = tanV * Double(r.aspect)
        var seen: [String] = []
        for site in r.world.terrain.sites {
            let g = r.world.terrain.height(x: site.pos.x, z: site.pos.y) ?? 20
            let v = SIMD3(site.pos.x, g + 1, site.pos.y) - o
            let z = simd_dot(v, b.fwd), dist = simd_length(v)
            guard z > 0, dist < 700 else { continue }
            if abs(simd_dot(v, b.right)) / z < tanH && abs(simd_dot(v, b.up)) / z < tanV {
                seen.append("\(site.kind)@\(Int(dist))")
            }
        }
        return "sites=\(seen.count)[\(seen.joined(separator: " "))]"
    }
}
