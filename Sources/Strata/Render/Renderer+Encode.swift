import Foundation
import Metal
import StrataCore
import simd

extension Renderer {
    func encode(into cb: MTLCommandBuffer, output: MTLTexture, outputSize: CGSize) {
        let (ow, oh, w, h) = sizes(outputSize)
        if sceneTex?.width != w || sceneTex?.height != h || history.first?.width != ow || history.first?.height != oh {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]
            td.storageMode = .private
            sceneTex = device.makeTexture(descriptor: td)
            td.width = ow
            td.height = oh
            history = (0..<2).compactMap { _ in device.makeTexture(descriptor: td) }
            prevCamera = nil
        }
        guard let sceneTex else { return }
        sceneSize = (w, h)
        aspect = Float(w) / Float(h)
        var f = makeFrame(size: CGSize(width: w, height: h), cells: CGSize(width: ow, height: oh), shadeHeight: dither ? h : oh)
        frameIndex = (frameIndex + 1).truncatingRemainder(dividingBy: 4096)
        bakeClouds(into: cb, frame: &f)
        if Renderer.fullCoarse { clipmap.invalidate() }
        clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: world.synchronous && !Renderer.chunkedCoarse,
                       active: activeLevels, giants: world.giantBuffer, isles: world.isleBuffer)
        bakeLamps(into: cb, frame: &f)
        fillLampTiles(frame: f, width: w, height: h)
        encodeProvinces(into: cb)
        // The shadow maps and the horizon share no data: one concurrent pass, so their short dispatches overlap.
        let bakes = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        if let bakes { bakeShadows(into: bakes, frame: &f) }
        if Renderer.profile != "nohorizon", let enc = bakes { encodeHorizon(into: enc, frame: &f) }
        bakes?.endEncoding()
        // The giants within the world's edge: their strand maps (baked on arrival) and the frame's table of them.
        vines.update(giants: world.terrain.giants, eye: SIMD2(camera.position.x, camera.position.z), reach: Double(f.fade.y), origin: world.origin, seed: world.seed, into: cb)
        // Snapshots refresh the far heights synchronously as they simulate.
        if !world.synchronous { updateFar(into: cb) }
        birdSlots.wait()
        cb.addCompletedHandler { [birdSlots] _ in birdSlots.signal() }
        let birdBuf = birdBuffers[birdIndex]
        birdIndex = (birdIndex + 1) % birdBuffers.count
        // Caves are carved out to level 3's handoff: shading looks them up only while one lies within it.
        let camXZ = SIMD2<Float>(Float(camera.position.x), Float(camera.position.z))
        let caveR = Float(Clipmap.handoff(3)) + Float(MapLayout.caveReach)
        f.birds.w = world.terrain.caves.contains { simd_length_squared($0.m - camXZ) < caveR * caveR } ? 1 : 0
        f.birds.x = Float(wildlife.upload(into: birdBuf.contents(), origin: SIMD3(Double(world.origin.x), 0, Double(world.origin.y))))
        let b = camera.basis
        prevCamera = (camera.position, b.fwd, b.right, b.up)
        let histIn = history[historyIndex], histOut = history[1 - historyIndex]
        historyIndex = 1 - historyIndex
        encodeScene(into: cb, target: sceneTex, frame: &f, birds: birdBuf, width: w, height: h)
        // The giants' hanging strands over the scene (vineFragment): the scene's bindings less the birds and the lamp
        // references, plus the strand maps. VR_PROFILE=bits2048 leaves them out.
        if vines.tableCount > 0 && !(Renderer.profile.hasPrefix("bits") && (Int(Renderer.profile.dropFirst(4)) ?? 0) & 2048 != 0) {
            encodeVines(into: cb, target: sceneTex, frame: &f)
        }
        // In its own pass: drawn after the scene in the scene's pass, even a draw doing nothing cost a twilight scene ~0.4 ms.
        if f.sunDir.y < Renderer.bioSun { encodeBio(into: cb, target: sceneTex, frame: &f) }
        encodeResolve(into: cb, scene: sceneTex, history: histIn, into: histOut, frame: &f, cells: (ow, oh))
        encodePost(into: cb, history: histOut, output: output, frame: &f)
    }

    // The province fields round the camera, rebaked whole once it has moved an eighth of their window.
    private func encodeProvinces(into cb: MTLCommandBuffer) {
        let cam2 = SIMD2(camera.position.x, camera.position.z)
        guard provCentre.map({ simd_length($0 - cam2) > Double(Renderer.provN) * Renderer.provCell / 8 }) ?? true,
              let enc = cb.makeComputeCommandEncoder() else { return }
        provCentre = cam2
        var info = SIMD4<Int32>(Int32(floor(cam2.x / Renderer.provCell)) - Int32(Renderer.provN / 2),
                                Int32(floor(cam2.y / Renderer.provCell)) - Int32(Renderer.provN / 2), 0, Int32(bitPattern: world.seed))
        enc.setComputePipelineState(provPipeline)
        enc.setTexture(provTex, index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: Renderer.provN, height: Renderer.provN, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

    // The terrain horizon round the camera and the islands' tangents (bakeHorizon).
    private func encodeHorizon(into enc: MTLComputeCommandEncoder, frame f: inout Frame) {
        f.shadow.z = 1
        enc.setComputePipelineState(horizonPipeline)
        enc.setBuffer(horizonBuf, offset: 0, index: 0)
        enc.setBytes(&f, length: MemoryLayout<Frame>.stride, index: 1)
        enc.setTexture(world.heightTex, index: 0)
        enc.setTexture(clipmap.height, index: 1)
        enc.setTexture(world.maxTex, index: 2)
        enc.setTexture(world.gapTex, index: 3)
        enc.setTexture(clipmap.gaps, index: 4)
        enc.setTexture(clipmap.gaps2, index: 5)
        enc.setTexture(world.bandTex, index: 6)
        enc.setTexture(clipmap.band, index: 7)
        enc.setBuffer(world.giantBuffer, offset: 0, index: 2)
        // HORIZON_SEGS (64) threads per bin, one threadgroup each.
        enc.dispatchThreads(MTLSize(width: 64, height: 1024, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    static func pass(_ target: MTLTexture, load: Bool) -> MTLRenderPassDescriptor {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = target
        d.colorAttachments[0].loadAction = load ? .load : .dontCare
        d.colorAttachments[0].storeAction = .store
        return d
    }
}
