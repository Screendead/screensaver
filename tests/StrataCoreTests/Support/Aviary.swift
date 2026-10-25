@testable import StrataCore
import simd

// The birds round a Flight, stepped after the camera with the frame's Sight as Renderer.simulate does (an 864x558 frame).
final class Aviary {
    let flight: Flight
    let wildlife: Wildlife

    init(seed: UInt32, phase: Double) {
        flight = Flight(seed: seed, phase: phase)
        wildlife = Wildlife(seed: seed)
        let terrain = flight.world.terrain
        wildlife.heightAt = { terrain.height(x: $0, z: $1) }
        wildlife.columnAt = { terrain.column(x: $0, z: $1) }
    }

    func sight() -> Sight {
        let c = flight.camera, b = c.basis, tanV = tan(25.0 * .pi / 180)
        return Sight(pos: c.position, fwd: b.fwd, right: b.right, up: b.up, tanV: tanV, aspect: 864.0 / 558, pixelAngle: 2 * tanV / 558,
                     sunY: flight.clock.sunDirection.y, worldEdge: 860 * 16, floorReach: 880 - 146, terrain: flight.world.terrain)
    }

    func fly(seconds: Double, each: (Wildlife) -> Void = { _ in }) {
        flight.fly(seconds: seconds) { camera, dt in
            wildlife.step(dt: dt, camera: camera, sight: sight(), sunY: flight.clock.sunDirection.y, phase: flight.clock.phase)
            each(wildlife)
        }
    }
}
