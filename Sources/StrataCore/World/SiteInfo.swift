import simd

// A structure site as the camera and the smoke and halo lists see it, in world coordinates.
package struct SiteInfo {
    package enum Kind: Int { case none = 0, henge, cairn, ruins, hamlet, jetty, camp, cave }
    // smoke: where its plume starts (rise > 0 only): a window lamp stands at its door, its chimney a few columns off.
    package struct Lamp { package var pos: SIMD3<Double>; package var kind: Int; package var rise: Double; package var index: Int; package var smoke: SIMD3<Double> }
    package var index: Int
    package var id: UInt32
    package var kind: Kind
    package var pos: SIMD2<Double>
    package var radius: Double
    package var lamps: [Lamp]
}
