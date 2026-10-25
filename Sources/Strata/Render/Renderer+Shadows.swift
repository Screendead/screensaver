import Foundation
import Metal
import StrataCore
import simd

extension Renderer {
    // A band of each shadow-height map per frame, around the camera, for the current key light; everything at once
    // when the camera or the light has jumped (startup, snapshots), so the maps never hold another place's shadows.
    func bakeShadows(into enc: MTLComputeCommandEncoder, frame f: inout Frame) {
        let (key, casts) = SkyClock.keyLight(clock.sunDirection)
        let levels = (1...Clipmap.levels).prefix { clipmap.ready($0) }.count
        // VR_PROFILE (tests): "noshadow" drops terrain shadows, "nobake" keeps the maps but skips their rebaking.
        // "bitsN" sets profiling flags directly: 8 reflections unshaded, 16 no sky visibility, 32 no cloud shadow or veil,
        // 64 primary hits unshaded, 4096 no glow on the land, 8192 the land's glow at its far means only.
        f.shadow.y = Knobs.current.profile == "nearclouds" ? 1 : Knobs.current.profile == "lowmax" ? 2 : Knobs.current.profile == "norefl" ? 4 : Float(Knobs.current.profileBits)
        if Knobs.current.profile == "noshadow" { f.shadow.x = 3; return }
        if Knobs.current.profile == "oldshadow" { f.shadow.x = 0; return }
        if Knobs.current.profile == "nobake" && shadowValid { f.shadow.x = 1; return }
        guard casts, levels > 0 else { shadowValid = false; f.shadow.x = 0; return }
        let cam = camera.xz
        let jumped = shadowFrom.map { simd_length($0.cam - cam) > 300 || simd_dot($0.key, key) < 0.995 } ?? true
        if !shadowValid || jumped { shadowFrom = (cam, key); shadowBand = [0, 0] }
        let full = !shadowValid || jumped
        let cosE = simd_length(SIMD2(key.x, key.z))
        enc.setComputePipelineState(shadowPipeline)
        enc.setTexture(clipmap.height, index: 1)
        enc.setTexture(clipmap.band, index: 2)
        enc.setTexture(clipmap.gaps, index: 3)
        for c in 0..<2 {
            let cell = Renderer.shadowCell * (c == 0 ? 1 : 4)
            let bands = Renderer.shadowBands[c], rows = Renderer.shadowN / bands
            let todo = full ? bands : 1
            for _ in 0..<todo {
                // 96: SHADOW_NEAR in Shaders/Light/Shadows.metal, the traced ray's reach.
                var b = ShadowBake(key: SIMD4(Float(key.x), Float(key.y), Float(key.z), Float(max(0.8 * 96 * cosE, cell))),
                                   cam: SIMD4(Float(cam.x), Float(cam.y), Float(cell), Float(levels)),
                                   rows: SIMD4(Int32(shadowBand[c] * rows), Int32(floor(cam.x / cell)), Int32(floor(cam.y / cell)), 0))
                enc.setTexture(shadowMaps[c], index: 0)
                enc.setBytes(&b, length: MemoryLayout<ShadowBake>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Renderer.shadowN, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                shadowBand[c] = (shadowBand[c] + 1) % bands
                if c == 0 && shadowBand[0] == 0 { shadowFrom = (cam, key) }
            }
        }
        shadowValid = true
        f.shadow.x = 1
    }
}
