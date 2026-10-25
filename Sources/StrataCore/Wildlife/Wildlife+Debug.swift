import Foundation
import simd

extension Wildlife {
    // Debug: replaces the flocks with one murmuration `distance` ahead of the camera, gathered, for timing the worst case.
    package func testMurmuration(camera: FlightCamera, distance: Double) {
        flocks = []
        let n = Int(rand(320...420))
        flocks = [makeMurmuration(slot: 0, anchor: camera.position + camera.forward * distance, n: n, life: rand(70...150), seedPhase: rand(0...1000))]
    }

    // Debug: replaces the flocks with a grid of birds of every kind close in front of the camera, at varied headings
    // and wingbeat phases, for judging their shapes.
    package func testPattern(camera: FlightCamera, distance: Double) {
        let b = camera.basis
        let kinds: [Kind] = [.songbird, .raptor, .gull, .starling, .bat, .goose]
        flocks = kinds.map { Flock(kind: $0, slot: 99, anchor: camera.position, heading: 0, speed: 0, altitude: 0, life: 1e9, seedPhase: 0) }
        for row in 0..<3 {
            for col in 0..<6 {
                let k = kinds[col]
                let f = flocks[col]
                let fl = Wildlife.flight(k)
                let p = camera.position + b.fwd * distance + b.right * (Double(col) - 2.5) * distance * 0.22
                    + b.up * (Double(row) - 1) * distance * 0.22
                let yaw = Double(row) * 2.1 + Double(col) * 0.4
                var bird = Bird(pos: p, vel: SIMD3(sin(yaw), 0.1 * Double(row - 1), cos(yaw)) * fl.speed, size: fl.span, tint: 0.5, phase: 0.5)
                bird.beat = row == 1 ? -1 : 1
                bird.flap = Double(row) * 2.0 + Double(col)
                f.birds.append(bird)
            }
        }
    }
}
