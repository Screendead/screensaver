import Foundation
import simd

// Valley-following flight. Positions are world-space Doubles; the renderer rebases them onto the map.
final class FlightCamera {
    var position: SIMD3<Double>
    var yaw: Double
    var pitch: Double = -0.18
    var roll: Double = 0
    var yawRate: Double = 0
    var yawAccel: Double = 0
    var climbRate: Double = 0
    var tracing = false
    var terms: [Double] = []
    var time: Double = 0
    var surveyAmount: Double = 0
    var facing: Double = 0
    var shotYaw: Double?
    var nextShot: Double = 0
    var nextPilgrim: Double = 0
    var rimLift: Double = 0
    var sun = SIMD3<Double>(0, 1, 0)
    var cloudSea = 0.0
    var pitchOverride: Double?
    // A fixed pose for showcase snapshots; step() leaves it alone.
    private(set) var frozen = false

    func freeze(position p: SIMD3<Double>, yaw y: Double, pitch pt: Double, roll rl: Double = 0) {
        position = p
        yaw = y
        pitch = pt
        roll = rl
        frozen = true
    }

    // Heights in world coords: the loaded map's (nil outside it), and the coarse 32-voxel max heights out to about 7.7k
    // (nil beyond, or before the first readback). The Renderer owns both and the camera, and drops them together.
    unowned(unsafe) let world: World
    unowned(unsafe) let far: FarHeights
    // Flight stays this far under the cloud base (CLOUD_Y0 in Shaders/Saver.metal, 1065).
    static let ceiling = 1065.0 - 80.0
    var avoidRate: Double = 0
    var nextRoute: Double = 0
    var routeYaw: Double?
    var routeSide = 0.0
    var blockedAhead: Double = 0
    // 0...1, eased: how much a slope close by fills the frame (looked down onto, or rising beside the flight).
    var hug: Double = 0
    // Ruined towers round the camera (Sources/Towers.swift).
    var towersAt: () -> [TowerInfo] = { [] }
    // Floating islands round the camera (Sources/Islands.swift).
    var islesAt: () -> [IslandInfo] = { [] }
    // Index of the current night, as the lamps read it (beacons and rites burn only on some nights).
    var nightIndex = 0
    // Sites a shot was built around, and when; each rests for a while so one place can't own the flight.
    var framedAt: [UInt32: Double] = [:]
    var focus: SIMD2<Double>?
    // Giants already passed (by axis, rounded), and the one the flight is slowly heading for.
    var visitedGiants = Set<SIMD2<Int>>()
    var pilgrim: SIMD2<Double>?
    // The pilgrim is a big island, flown beneath rather than beside.
    var pilgrimIsle = false

    init(seed: UInt32, world: World, far: FarHeights) {
        self.world = world
        self.far = far
        let s = Double(seed % 100_000)
        position = SIMD3(s * 7.31, 70, s * 3.17)
        yaw = s.truncatingRemainder(dividingBy: 6.283)
        // Decorrelates the wander/pitch cycles across seeds and displays.
        time = Double(seed % 997) * 37.0
    }

    var forward: SIMD3<Double> {
        SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    }

    // Unit right/up vectors including roll.
    var basis: (fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>) {
        let f = forward
        let r0 = simd_normalize(simd_cross(SIMD3<Double>(0, 1, 0), f))
        let u0 = simd_cross(f, r0)
        let r = r0 * cos(roll) + u0 * sin(roll)
        let u = simd_cross(f, r)
        return (f, r, u)
    }

    // The flight passes beneath an island it is under by isleMargin, or can sink to that by the time it gets there (at
    // most isleSink a second), with room under the keel; it steers by the ground there. Any other island is terrain to
    // its top. Kept under isleCap below the underside, so the margin holds.
    static let isleMargin = 30.0, isleCap = 45.0, isleSink = 3.0
    var speed = 20.0

    // Starts the flight here (tests: VR_START).
    func start(at p: SIMD2<Double>, yaw y: Double) {
        position.x = p.x
        position.z = p.y
        yaw = y
    }
}
