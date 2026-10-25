import simd

// A structure site as the camera and the smoke and halo lists see it, in world coordinates.
struct SiteInfo {
    enum Kind: Int { case none = 0, henge, cairn, ruins, hamlet, jetty, camp, cave }
    // smoke: where its plume starts (rise > 0 only): a window lamp stands at its door, its chimney a few columns off.
    struct Lamp { var pos: SIMD3<Double>; var kind: Int; var rise: Double; var index: Int; var smoke: SIMD3<Double> }
    var index: Int
    var id: UInt32
    var kind: Kind
    var pos: SIMD2<Double>
    var radius: Double
    var lamps: [Lamp]
}
