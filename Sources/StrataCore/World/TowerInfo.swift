import simd

// One ruined tower (Shaders/Saver.metal, MARK: Ruined towers), as placeTowers records it.
struct TowerInfo {
    var pos: SIMD2<Double>
    var radius: Double      // the base shaft's apothem
    var base: Double
    var height: Double      // above base
    var kind: Int           // 0 stepped spire, 1 drum, 2 needle
    var member: Int         // 0: the tallest of its group
    var lit: Bool
    var moist: Double, temp: Double
    var record: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<UInt32>)
    var top: Double { base + height }

    // Must match TowerRec's 64 bytes in Shaders/Saver.metal.
    static let recordSize = 64

    // The towers among `count` records (a slot without one has height 0).
    static func decode(_ raw: UnsafeRawPointer, count: Int) -> [TowerInfo] {
        var out: [TowerInfo] = []
        for i in 0..<count {
            let r = raw + i * recordSize
            let a = r.load(as: SIMD4<Float>.self), b = (r + 16).load(as: SIMD4<Float>.self)
            let c = (r + 32).load(as: SIMD4<Float>.self), d = (r + 48).load(as: SIMD4<UInt32>.self)
            guard b.x > 0 else { continue }
            out.append(TowerInfo(pos: SIMD2(Double(a.x), Double(a.y)), radius: Double(a.z), base: Double(a.w), height: Double(b.x),
                                 kind: Int(d.y), member: Int(d.w & 255), lit: d.z != 0, moist: Double((d.w >> 8) & 255) / 255,
                                 temp: Double((d.w >> 16) & 255) / 255, record: (a, b, c, d)))
        }
        return out
    }
}
