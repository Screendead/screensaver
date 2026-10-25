import Foundation
import StrataCore
import simd

extension Renderer {
    // shadeHeight sets the shading prefilter's pixel: the output's without the dither, so a scene smaller than the
    // output keeps the output's surface detail for the temporal resolve to gather.
    func makeFrame(size: CGSize, cells: CGSize, shadeHeight: Int) -> Frame {
        var f = Frame()
        let b = camera.basis
        let o = SIMD3(Double(world.origin.x), 0, Double(world.origin.y))
        let p = camera.position - o
        f.camPos = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(Renderer.tanV))
        f.camFwd = SIMD4(SIMD3<Float>(b.fwd), Float(size.width / size.height))
        f.camRight = SIMD4(SIMD3<Float>(b.right), Float(simTime))
        f.camUp = SIMD4(SIMD3<Float>(b.up), battery ? 0 : 1)
        f.sunDir = SIMD4(SIMD3<Float>(clock.sunDirection), Float(clock.phase))
        f.planet = SIMD4(SIMD3<Float>(clock.planetDirection), 0)
        // The cell corner nearest the planet, projected as the shader casts its rays (without the jitter); off screen
        // while the sun is up (planetAmt in Shaders/Saver.metal).
        let pv = clock.planetDirection, pz = simd_dot(pv, b.fwd), tanH = Double(f.camPos.w)
        f.planetPx = SIMD4(-1e4, -1e4, Float(size.width / cells.width), 0)
        if pz > 0.01 && clock.sunDirection.y < 0.02 {
            let pn = SIMD2(simd_dot(pv, b.right) / (pz * tanH * Double(f.camFwd.w)), simd_dot(pv, b.up) / (pz * tanH))
            let uv = SIMD2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5)
            let cx = (uv.x * cells.width).rounded(), cy = (uv.y * cells.height).rounded()
            f.planetPx.x = Float(cx)
            f.planetPx.y = Float(cy)
        }
        f.resolution = SIMD4(Float(size.width), Float(size.height), Float(1 / Double(shadeHeight)), Float(1 / size.height))
        f.mapInfo = SIMD4(world.origin.x, world.origin.y, Int32(MapLayout.size), Int32(bitPattern: world.seed))
        f.misc = SIMD4(Float(exposure), world.globalMax, 0.6, Float(cloudCover))
        f.misc2 = SIMD4(frameIndex, 1, debugMode, headroom)
        let sunY = clock.sunDirection.y
        f.lights = SIMD4(Float(sunY), Float(clock.nightProgress), Float(clock.nightIndex), Float(min(max((0.08 - sunY) / 0.1, 0), 1)))
        f.refs = SIMD4(Int32(f.lights.w > 0 ? hazeCount : 0), 0, 0, 0)
        pickSmoke()
        f.refs.y = Int32(smokeCount)
        if let pc = prevCamera {
            let pp = pc.pos - o
            f.prevPos = SIMD4(Float(pp.x), Float(pp.y), Float(pp.z), 1)
            f.prevFwd = SIMD4(SIMD3<Float>(pc.fwd), 0)
            f.prevRight = SIMD4(SIMD3<Float>(pc.right), 0)
            f.prevUp = SIMD4(SIMD3<Float>(pc.up), 0)
        }
        // Halton(2,3) subpixel jitter, accumulated by the temporal resolve; upsampling needs more phases to
        // cover every output pixel.
        let phases = min(64, Int((8 / min(renderScale * renderScale, 1)).rounded(.up)))
        let k = Int(frameIndex) % phases + 1
        func halton(_ i: Int, _ b: Int) -> Double { var f = 1.0, r = 0.0, i = i; while i > 0 { f /= Double(b); r += f * Double(i % b); i /= b }; return r }
        f.jitter = SIMD4(Float((halton(k, 2) - 0.5) * 2 / size.width), Float((halton(k, 3) - 0.5) * 2 / size.height), 0, 0)
        // LOD rings. The 1-voxel map hands off at 880, inside the 1024 - 8 (neighbour reads) - 100 (recentre drift)
        // - ~21 (drift while a regeneration is in flight: 41 pieces at 30 fps) it covers; nearer its edge only if the camera outran
        // regeneration. Each level morphs toward the next over a band, reaching it 1.5 parent columns before the
        // handoff circle.
        let half = Double(MapLayout.size / 2)
        let edge = half - 8 - max(abs(p.x - half), abs(p.z - half)) - abs(Double(Knobs.current.lodShift))
        let b01 = min(880, edge)
        var lods = [SIMD4<Float>(Float(b01 - 140), Float(b01), Float(b01 - 3), 1)]
        // Levels drawn: those built, out to the edge. Past the battery edge the last level's band eases in with the
        // edge, so no ring pops when the level count changes.
        let built = (1...Clipmap.levels).prefix { clipmap.ready($0) }.count
        let levels = min(built, edgeRadius > Clipmap.handoff(Clipmap.levels - 1) ? Clipmap.levels : Clipmap.levels - 1)
        let rMax = levels == Clipmap.levels ? edgeRadius : Clipmap.handoff(levels)
        let grow = levels == Clipmap.levels ? (edgeRadius - Clipmap.handoff(levels - 1)) / (Clipmap.handoff(levels) - Clipmap.handoff(levels - 1)) : 1
        for k in 1...max(levels, 1) {
            let b1 = Clipmap.handoff(k), s = Double(1 << k)
            if k == levels {
                lods.append(SIMD4(Float(rMax), Float(rMax), Float(rMax), Float(s)))
            } else {
                let u = k == levels - 1 ? grow : 1
                lods.append(SIMD4(Float(b1 - u * 0.2 * b1), Float(b1), Float(b1 - u * 3 * s), Float(s)))
            }
        }
        while lods.count < 5 { lods.append(lods.last!) }
        (f.lod0, f.lod1, f.lod2, f.lod3, f.lod4) = (lods[0], lods[1], lods[2], lods[3], lods[4])
        f.fade = SIMD4(Float(0.7 * rMax), Float(rMax), Float(levels + 1), Knobs.current.lodShift)
        return f
    }
}
