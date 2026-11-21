@testable import StrataCore
import simd

// Rolling hills with valleys under the sea, standing in for the GPU's map: the loaded map's CPU copy and the far heights,
// re-centred on the camera as World and Renderer keep them.
final class SyntheticWorld {
    let terrain = Terrain(), far = FarHeights()

    func ground(_ x: Double, _ z: Double) -> Double {
        45 + 70 * sin(x / 310) * sin(z / 270) + 35 * sin((x - 2 * z) / 530) + 20 * cos(z / 150 + x / 900)
    }

    init(around p: SIMD2<Double>) { fill(p) }

    // Moves the map once the camera is a quarter of it from its centre.
    func follow(_ p: SIMD2<Double>) {
        let c = SIMD2(Double(terrain.origin.x), Double(terrain.origin.y)) + Double(MapLayout.size / 2)
        if simd_length(p - c) > Double(MapLayout.size / 4) { fill(p) }
    }

    private func fill(_ p: SIMD2<Double>) {
        let s = MapLayout.size >> MapLayout.cpuLevel, cell = Double(1 << MapLayout.cpuLevel)
        let o = SIMD2<Int32>(Int32(floor((p.x - Double(MapLayout.size / 2)) / 64)) * 64, Int32(floor((p.y - Double(MapLayout.size / 2)) / 64)) * 64)
        terrain.origin = o
        // A cell's height is the most of its corners and centre: conservative, as the GPU's max mip.
        terrain.heights = (0..<(s * s)).map { i in
            let x = Double(o.x) + Double(i % s) * cell, z = Double(o.y) + Double(i / s) * cell
            return Float([(0.0, 0.0), (cell, 0), (0, cell), (cell, cell), (cell / 2, cell / 2)].map { ground(x + $0.0, z + $0.1) }.max()!)
        }
        terrain.floors = terrain.heights.map { $0 - 6 }
        terrain.ground = terrain.heights
        terrain.under = [Float](repeating: 4096, count: s * s)
        let n = FarHeights.n
        far.centre = SIMD2(Int(floor(p.x / FarHeights.cell)), Int(floor(p.y / FarHeights.cell)))
        var h = [Float16](repeating: 0, count: n * n)
        for cz in (far.centre.y - n / 2)..<(far.centre.y + n / 2) {
            for cx in (far.centre.x - n / 2)..<(far.centre.x + n / 2) {
                let x = Double(cx) * FarHeights.cell, z = Double(cz) * FarHeights.cell
                h[(cz & (n - 1)) * n + (cx & (n - 1))] = Float16(ground(x + 16, z + 16) + 30)
            }
        }
        far.heights = h
    }
}

// The camera and the birds flown over a SyntheticWorld through the sky clock, stepped as Renderer.simulate steps them
// (the birds after the camera, with an 864x558 frame's Sight); each step's pose.
final class Flight {
    struct Pose: Equatable { var position: SIMD3<Double>; var yaw: Double, pitch: Double, roll: Double }
    let world: SyntheticWorld
    let camera: FlightCamera
    let wildlife: Wildlife
    var clock = SkyClock()
    private(set) var poses: [Pose] = []

    init(seed: UInt32, phase: Double) {
        let probe = FlightCamera(seed: seed, terrain: Terrain(), far: FarHeights())
        world = SyntheticWorld(around: probe.xz)
        camera = FlightCamera(seed: seed, terrain: world.terrain, far: world.far)
        camera.moveToOpenGround()
        world.follow(camera.xz)
        camera.liftOverGround()
        clock.phase = phase
        wildlife = Wildlife(seed: seed)
        let terrain = world.terrain
        wildlife.heightAt = { terrain.height(x: $0, z: $1) }
        wildlife.columnAt = { terrain.column(x: $0, z: $1) }
    }

    private func sight() -> Sight {
        let b = camera.basis, tanV = tan(25.0 * .pi / 180)
        return Sight(pos: camera.position, fwd: b.fwd, right: b.right, up: b.up, tanV: tanV, aspect: 864.0 / 558, pixelAngle: 2 * tanV / 558,
                     sunY: clock.sunDirection.y, worldEdge: 860 * 16, floorReach: 880 - 146, terrain: world.terrain)
    }

    func fly(seconds: Double, dt: Double = 1.0 / 30) {
        for _ in 0..<Int(seconds / dt) {
            camera.sun = clock.sunDirection
            camera.nightIndex = clock.nightIndex
            camera.cloudSea = clock.cloudSea
            camera.step(dt: dt)
            wildlife.step(dt: dt, camera: camera, sight: sight(), sunY: clock.sunDirection.y, phase: clock.phase)
            clock.advance(dt)
            world.follow(camera.xz)
            poses.append(Pose(position: camera.position, yaw: camera.yaw, pitch: camera.pitch, roll: camera.roll))
        }
    }
}
