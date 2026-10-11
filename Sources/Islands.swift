import Metal
import simd

// One table slot (Shaders/Saver.metal IsleRec): r > 0 an island standing.
struct IsleRec {
    var cell = SIMD2<Int32>(Int32.min, Int32.min)
    var c = SIMD2<Float>()
    var r: Float = 0
    var a: Float = 0
    var d: Float = 0
    var b: Float = 0
    var k: Float = 0
    var h: UInt32 = 0
    var pad = SIMD2<Float>()
}

// A floating island as the camera, the birds and the shading see it, in world coordinates: nothing of it lies further
// than `reach` from `pos`; its top stands about `top` at the centre, its underside nowhere under `keel`, over ground no
// higher than `base`.
struct IslandInfo {
    var pos: SIMD2<Double>
    var radius: Double
    var reach: Double
    var top: Double
    var keel: Double
    var base: Double
    var hash: UInt32
    var massif: Bool

    // Room under the keel for the flight to pass beneath.
    var passable: Bool { keel - base >= 90 }
}

enum Islands {
    // Must match ISLE_WIN, ISLE_MWIN and ISLE_SLOTS (and IsleRec's 48 bytes) in Shaders/Saver.metal.
    static let win = 60, mwin = 13
    static let slots = win * win + mwin * mwin
    static let recordSize = 48
    // Must match ISLE_TABLE and IsleShade in Shaders/Saver.metal: the islands the shading reads, nearest the camera first.
    static let tableSize = 16
    static let tableBytes = 16 + tableSize * 32
    // VR_ISLE_FIELD=1: the test field round (-3500, -2500) in place of island country; 2: no islands (tests).
    static let field = Int32(ProcessInfo.processInfo.environment["VR_ISLE_FIELD"] ?? "") ?? 0

    static func stretch(_ r: Double, _ h: UInt32) -> Double {
        (r > 40 && (r <= 210 || r >= 420) && Provinces.hashf(Provinces.pcg(h &+ 10)) < 0.33) ? 1.5 : 1.0
    }

    static func parse(_ buf: MTLBuffer) -> [IslandInfo] {
        let p = buf.contents().bindMemory(to: IsleRec.self, capacity: slots)
        var out: [IslandInfo] = []
        for i in 0..<slots {
            let r = p[i]
            guard r.r > 0, r.cell.x != Int32.min else { continue }
            let R = Double(r.r)
            out.append(IslandInfo(pos: SIMD2(Double(r.c.x), Double(r.c.y)), radius: R, reach: stretch(R, r.h) * (1.45 * R + 8),
                                  top: Double(r.a) + 0.06 * R + 10, keel: Double(r.k), base: Double(r.b), hash: r.h, massif: i >= win * win))
        }
        return out
    }

    // The shading table: the islands nearest p within reach of the view (count, then per island centre xz, reach,
    // underside's mean height and radius), as bytes for setFragmentBytes.
    static func table(_ list: [IslandInfo], near p: SIMD2<Double>, within d: Double) -> [UInt8] {
        let near = list.filter { simd_length($0.pos - p) < d + $0.reach }.sorted { simd_length($0.pos - p) < simd_length($1.pos - p) }
        var bytes = [UInt8](repeating: 0, count: tableBytes)
        let n = min(near.count, tableSize)
        bytes.withUnsafeMutableBytes { b in
            b.storeBytes(of: Int32(n), as: Int32.self)
            for i in 0..<n {
                let s = near[i]
                let o = 16 + i * 32
                b.storeBytes(of: SIMD4<Float>(Float(s.pos.x), Float(s.pos.y), Float(s.reach), Float(0.5 * (s.keel + s.top))), toByteOffset: o, as: SIMD4<Float>.self)
                b.storeBytes(of: SIMD4<Float>(Float(s.radius), Float(s.top), Float(s.keel), 0), toByteOffset: o + 16, as: SIMD4<Float>.self)
            }
        }
        return bytes
    }
}
