import Foundation
import Metal
import simd

extension World {
    static var logSites: Bool { Knobs.current.sites }

    static func readSites(_ buf: MTLBuffer) -> [SiteInfo] {
        let raw = buf.contents()
        var list: [SiteInfo] = []
        for i in 0..<(MapLayout.siteGrid * MapLayout.siteGrid) {
            let base = raw + i * MapLayout.siteStride
            let head = base.load(as: SIMD4<Int32>.self)
            guard let kind = SiteInfo.Kind(rawValue: Int(head.z)), kind != .none else { continue }
            let info = (base + 16).load(as: SIMD4<Float>.self)
            let c = SIMD2(Double(head.x), Double(head.y))
            var lamps: [SiteInfo.Lamp] = []
            for l in 0..<Int(info.w) {
                let v = (base + 32 + 16 * l).load(as: SIMD4<Float>.self)
                let w = Int(v.w), k = w % 16, rise = Double((w / 16) % 64), code = w / 1024
                let pos = SIMD3(c.x + Double(v.x), Double(v.y), c.y + Double(v.z))
                let off = code > 0 ? SIMD3(Double(code % 32 - 16), rise, Double(code / 32 - 16)) : SIMD3(0, rise, 0)
                lamps.append(.init(pos: pos, kind: k, rise: rise, index: i * MapLayout.maxLamps + l, smoke: pos + off))
            }
            list.append(SiteInfo(index: i, id: UInt32(bitPattern: head.w), kind: kind, pos: c + 0.5, radius: Double(info.x), lamps: lamps))
        }
        return list
    }

    static func readCaves(_ buf: MTLBuffer) -> [CaveRec] {
        let p = (buf.contents() + MapLayout.caveOffset).bindMemory(to: CaveRec.self, capacity: MapLayout.caveWin * MapLayout.caveWin)
        return (0..<(MapLayout.caveWin * MapLayout.caveWin)).compactMap { p[$0].flags & 1 != 0 ? p[$0] : nil }
    }

    static func readGiants(_ buf: MTLBuffer) -> [GiantInfo] {
        let p = buf.contents().bindMemory(to: GiantRec.self, capacity: MapLayout.giantWin * MapLayout.giantWin)
        return (0..<(MapLayout.giantWin * MapLayout.giantWin)).compactMap { i in
            let r = p[i]
            // The crown's top: giantPad's dome in Shaders/Saver.metal rises about yT + 36 + 0.55 * 56 + 9.5, the high limbs'
            // clumps to about yT + 84.
            return r.g > 0 ? GiantInfo(pos: SIMD2(Double(r.c.x), Double(r.c.y)), ground: Double(r.g), top: Double(r.g + r.yT) + 85, kind: Int(r.kind), yT: Double(r.yT), h: r.h) : nil
        }
    }

    // The CPU copy's heights are 4x4 maxima of the column tops, so their maximum is the map's.
    static func readBack(_ cpu: MTLTexture) -> (heights: [Float], floors: [Float], ground: [Float], under: [Float], top: Float) {
        let s = MapLayout.size >> MapLayout.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 4)
        cpu.getBytes(&raw, bytesPerRow: s * 8, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: 0)
        let heights = (0..<(s * s)).map { Float(raw[$0 * 4]) }
        return (heights, (0..<(s * s)).map { Float(raw[$0 * 4 + 1]) }, (0..<(s * s)).map { Float(raw[$0 * 4 + 2]) },
                (0..<(s * s)).map { Float(raw[$0 * 4 + 3]) }, heights.max() ?? 0)
    }
}
