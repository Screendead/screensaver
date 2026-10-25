import Foundation
import simd

extension Wildlife {
    enum Kind: Int { case songbird, raptor, gull, starling, bat, goose }

    struct Bird {
        var pos: SIMD3<Double>
        var vel: SIMD3<Double>
        var flap = 0.0
        var beat = 0.0 // > 0: seconds of flapping left; < 0: of gliding
        var bank = 0.0
        var size: Double
        var tint: Double
        var phase: Double
        var sheet = SIMD2<Double>.zero // starlings: place on the murmuration's sheet
        var id = 0
    }

    struct Flight {
        let speed: Double, minSpeed: Double, maxSpeed: Double
        let span: Double
        let flapHz: Double, amp: Double, bias: Double
        let flapFor: ClosedRange<Double>, glideFor: ClosedRange<Double>
        let glideAngle: Double, glideExt: Double
        let handDroop: Double, sweep: Double // gliding hand angle relative to the arm; hand swept back (rad)
        let reach: Double, separation: Double, cohesion: Double, alignment: Double, pull: Double, wander: Double
        let clearance: Double
    }

    static func flight(_ k: Kind) -> Flight {
        switch k {
        case .songbird: // bounding: bursts of wingbeats between folded-wing swoops
            return Flight(speed: 9, minSpeed: 6, maxSpeed: 12, span: 0.85, flapHz: 11, amp: 0.85, bias: 0.1,
                          flapFor: 0.3...0.7, glideFor: 0.2...0.45, glideAngle: -0.1, glideExt: 0.35, handDroop: 0.0, sweep: 0.45,
                          reach: 9, separation: 1.8, cohesion: 0.25, alignment: 0.9, pull: 0.05, wander: 3.0, clearance: 6)
        case .raptor:
            return Flight(speed: 7, minSpeed: 5, maxSpeed: 9, span: 2.3, flapHz: 2.2, amp: 0.45, bias: 0.05,
                          flapFor: 0.8...1.5, glideFor: 5...11, glideAngle: 0.17, glideExt: 1, handDroop: 0.12, sweep: 0.1,
                          reach: 0, separation: 14, cohesion: 0, alignment: 0, pull: 0, wander: 0.6, clearance: 18)
        case .gull:
            return Flight(speed: 8, minSpeed: 6, maxSpeed: 11, span: 1.7, flapHz: 3, amp: 0.65, bias: 0.05,
                          flapFor: 1...2.6, glideFor: 1.5...4, glideAngle: 0.12, glideExt: 1, handDroop: -0.42, sweep: 0.45,
                          reach: 16, separation: 4.5, cohesion: 0.08, alignment: 0.35, pull: 0.03, wander: 2.0, clearance: 4)
        case .starling:
            return Flight(speed: 12, minSpeed: 4, maxSpeed: 22, span: 0.7, flapHz: 8, amp: 0.8, bias: 0.05,
                          flapFor: 0.6...1.8, glideFor: 0.2...0.5, glideAngle: 0, glideExt: 0.9, handDroop: -0.05, sweep: 0.6,
                          reach: 4, separation: 0.9, cohesion: 0, alignment: 0.3, pull: 0, wander: 1.0, clearance: 8)
        case .bat: // never glides; jinks constantly
            return Flight(speed: 6, minSpeed: 3.5, maxSpeed: 9, span: 0.65, flapHz: 10, amp: 1.0, bias: 0,
                          flapFor: 2...4, glideFor: 0.05...0.1, glideAngle: 0, glideExt: 1, handDroop: -0.2, sweep: 0.25,
                          reach: 6, separation: 2.5, cohesion: 0.05, alignment: 0.1, pull: 0.12, wander: 14, clearance: 4)
        case .goose: // formation flyers: slots in a V behind the leader
            return Flight(speed: 11, minSpeed: 8, maxSpeed: 14, span: 2.0, flapHz: 3.2, amp: 0.55, bias: 0.05,
                          flapFor: 4...9, glideFor: 0.5...1.2, glideAngle: 0.05, glideExt: 1, handDroop: -0.1, sweep: 0.15,
                          reach: 0, separation: 1.6, cohesion: 0, alignment: 0, pull: 0, wander: 0.4, clearance: 15)
        }
    }

    final class Flock {
        let kind: Kind
        let slot: Int
        var birds: [Bird] = []
        var anchor: SIMD3<Double>
        var heading: Double
        var speed: Double
        var altitude: Double
        var age = 0.0
        let life: Double
        var leaving = false
        var unseen = 0.0 // seconds since any bird was in frame
        // Seconds until the flock, flying in from where nothing could see it, reaches the part of the frame it was
        // aimed at; until then it holds its course and speed.
        var transit = 0.0
        var cruise = 0.0
        var climb = 0.0 // while flying in
        // Speed once arrived, for kinds that then hold to one spot.
        var hold: Double?
        // Murmuration: a slow drift along with the camera, so it stays in the frame while it gathers and wheels.
        var drift = SIMD3<Double>.zero
        let seedPhase: Double
        var serial = 0
        var misses = 0 // murmuration: tries in a row to gather again ahead that found no place
        // Circling an island: its centre (x, z), the ring's radius and height, and which way round.
        var orbit: SIMD4<Double>?
        var orbitSense = 1.0

        init(kind: Kind, slot: Int, anchor: SIMD3<Double>, heading: Double, speed: Double, altitude: Double, life: Double, seedPhase: Double) {
            self.kind = kind
            self.slot = slot
            self.anchor = anchor
            self.heading = heading
            self.speed = speed
            self.altitude = altitude
            self.life = life
            self.seedPhase = seedPhase
        }

        var dir: SIMD3<Double> { SIMD3(sin(heading), 0, cos(heading)) }
    }
}
