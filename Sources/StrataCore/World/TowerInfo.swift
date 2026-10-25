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
}
