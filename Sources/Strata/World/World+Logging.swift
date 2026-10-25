import Foundation
import StrataCore
import simd

extension World {
    static var logSites: Bool { Knobs.current.sites }

    // VR_CAVES=r: the caves within r of the new map's centre, nearest first, and the table's counts.
    func logCaves(near o: SIMD2<Int32>, within r: Float) {
        let mid = SIMD2<Float>(Float(o.x), Float(o.y)) + Float(MapLayout.size / 2)
        for c in terrain.caves.filter({ simd_length($0.m - mid) < r }).sorted(by: { simd_length($0.m - mid) < simd_length($1.m - mid) }) {
            print(String(format: "cave at (%.0f, %.0f) floor %.0f in (%.2f, %.2f) W %.1f L %.0f H %.0f%@%@%@", c.m.x, c.m.y, c.yF, c.u.x, c.u.y, c.w, c.l, c.hc,
                         c.flags & 2 != 0 ? " home" : "", c.flags & 4 != 0 ? " grotto" : "", c.flags & 8 != 0 ? " lowland" : ""))
        }
        print("caves in table: \(terrain.caves.count) (lowland \(terrain.caves.filter { $0.flags & 8 != 0 }.count)), within \(Int(r)): \(terrain.caves.filter { simd_length($0.m - mid) < r }.count)")
    }

    // VR_SITES: the map's giants and its sites by kind; VR_SITES=3 lists each site's parts too.
    func logSites(in buf: Buffer, origin o: SIMD2<Int32>) {
        for g in terrain.giants { print(String(format: "giant at (%.0f, %.0f) ground %.0f top %.0f kind %d", g.pos.x, g.pos.y, g.ground, g.top, g.kind)) }
        var counts = [Int](repeating: 0, count: 8)
        for site in terrain.sites { counts[site.kind.rawValue] += 1 }
        if Knobs.current.env["VR_SITES"] == "3" {
            // Parts as kind (2 house, 10 round) and dwelling style.
            let raw = buf.sites.contents()
            for site in terrain.sites {
                let base = raw + site.index * MapLayout.siteStride
                let np = Int((base + 16).load(as: SIMD4<Float>.self).z)
                var desc = ""
                for k in 0..<np {
                    let v = (base + 32 + 16 * MapLayout.maxLamps + 16 * k).load(as: SIMD4<Int32>.self)
                    let kind = (v.y >> 16) & 0xff, style = (v.w >> 28) & 15
                    desc += kind == 2 || kind == 10 ? " p\(kind)s\(style)" : kind == 11 ? " p11t\(v.z >> 16)" : " p\(kind)"
                }
                print("  site \(site.kind) \(Int(site.pos.x)),\(Int(site.pos.y)) r\(Int(site.radius)) lamps \(site.lamps.count)\(desc)")
            }
        }
        print("sites at (\(o.x),\(o.y)): henge \(counts[1]) cairn \(counts[2]) ruins \(counts[3]) hamlet \(counts[4]) jetty \(counts[5]) camp \(counts[6]) cave \(counts[7]) total \(terrain.sites.count)")
    }
}
