import Foundation
import simd

extension FlightCamera {
    // A flight starting over 3.5k from every giant moves to 2.5k short of the nearest (on the line from it to the start, so
    // the start stays as near its own country as it can), facing it: it reaches the landmark within minutes. True if
    // it moved. terrain.giants is the table round the start's map (+-18k).
    func startNearGiant() -> Bool {
        let p = SIMD2(position.x, position.z)
        guard let g = terrain.giants.min(by: { simd_length($0.pos - p) < simd_length($1.pos - p) }) else { return false }
        let d = simd_length(g.pos - p)
        guard d > 3500 else { return false }
        let q = g.pos + (p - g.pos) / d * 2500
        position.x = q.x
        position.z = q.y
        yaw = atan2(g.pos.x - q.x, g.pos.y - q.y) + 0.35
        return true
    }

    // A start among or beside the massifs moves to the nearest open low ground, so the flight begins under the ceiling.
    // Reads only the far heights.
    func moveToOpenGround() {
        if (0..<12).contains(where: { hFar(position.x + sin(Double($0) * .pi / 6) * 500, position.z + cos(Double($0) * .pi / 6) * 500) > 260 })
            || hFar(position.x, position.z) > 200 {
            search: for r in stride(from: 200.0, through: 6000.0, by: 200.0) {
                for k in 0..<24 {
                    let a = Double(k) * .pi / 12
                    let x = position.x + sin(a) * r, z = position.z + cos(a) * r
                    guard far.height(x: x, z: z) != nil, hFar(x, z) < 120 else { continue }
                    // Clear of the massifs on every side, so the first seconds are not a climb up a wall.
                    var clear = true
                    for rr in [250.0, 500.0, 800.0] {
                        for j in 0..<12 where hFar(x + sin(Double(j) * .pi / 6) * rr, z + cos(Double(j) * .pi / 6) * rr) > 260 { clear = false }
                    }
                    if clear { position.x = x; position.z = z; break search }
                }
            }
        }
    }

    static let liftRadius = 24.0

    func liftOverGround() {
        position.y = max(position.y, footprintMax(position.x, position.z, radius: FlightCamera.liftRadius) + 30)
        // Spires just ahead are already cleared at a 3/s climb, so the first seconds are not a scramble over them.
        for d in stride(from: 32.0, through: 320.0, by: 32.0) {
            position.y = max(position.y, hFar(position.x + sin(yaw) * d, position.z + cos(yaw) * d) + 20 - 0.2 * d)
        }
    }
}
