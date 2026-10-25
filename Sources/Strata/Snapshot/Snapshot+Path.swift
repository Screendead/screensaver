import Foundation
import Metal
import StrataCore

extension Snapshot {
    // VR_PATH=file: renders each line "simTime x z y yawDeg pitchDeg rollDeg exposure [phase day]" (a bug report's
    // recorded frames, bugclip.sh) in turn as <screenshot>_pNNNN.png, carrying the temporal history across them as live.
    // Phase and day hold the sky clock where the report's frame had it: the menu's Time of Day pins the phase in the
    // saver, and the day picks each night's lit beacons and rites.
    // `simulated`: the sim time already run.
    static func replay(_ r: Renderer, path text: String, from simulated: Double, output tex: MTLTexture, to path: String) -> Bool {
        var simulated = simulated, wrote = true
        var n = 0
        for line in text.split(separator: "\n") {
            let v = line.split(separator: " ").compactMap { Double($0) }
            guard v.count == 8 || v.count == 10 else { continue }
            if v.count == 10 { r.pinnedPhase = v[8] }
            r.simulate(max(v[0] - simulated, 0))
            simulated = v[0]
            if v.count == 10 {
                r.clock.phase = v[8]
                r.clock.day = Int(v[9])
                r.camera.sun = r.clock.sunDirection
                r.camera.nightIndex = r.clock.nightIndex
                r.camera.cloudSea = r.clock.cloudSea
            }
            r.camera.freeze(position: SIMD3(v[1], v[3], v[2]), yaw: v[4] * .pi / 180, pitch: v[5] * .pi / 180, roll: v[6] * .pi / 180)
            r.exposure = v[7]
            render(r, into: tex)
            wrote = save(tex, to: path.replacingOccurrences(of: ".png", with: String(format: "_p%04d.png", n))) && wrote
            n += 1
        }
        return wrote
    }
}
