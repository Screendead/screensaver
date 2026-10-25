import Foundation
import Metal
import simd

extension Renderer {
    func bakeClouds(into cb: MTLCommandBuffer, frame f: inout Frame) {
        let n = Renderer.cloudTexels, span = Double(n) * Renderer.cloudTexel
        if cloudTex == nil {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: n, height: n, mipmapped: false)
            td.usage = [.shaderRead, .shaderWrite]
            td.storageMode = .private
            cloudTex = device.makeTexture(descriptor: td)
        }
        guard let cloudTex else { return }
        // Must match CLOUD_DRIFT in Shaders/Saver.metal.
        let drift = SIMD2(cos(0.6), sin(0.6)) * 4.8 * Double(Float(simTime))
        let cam = SIMD2(camera.position.x, camera.position.z) - drift
        if cloudCenter.map({ simd_length(cam - $0) > span * 0.2 }) ?? true,
           let enc = cb.makeComputeCommandEncoder() {
            cloudCenter = (cam / Renderer.cloudTexel).rounded(.down) * Renderer.cloudTexel
            var info = SIMD4<Float>(Float(cloudCenter!.x - span / 2), Float(cloudCenter!.y - span / 2), Float(Renderer.cloudTexel),
                                    Float(bitPattern: world.seed))
            enc.setComputePipelineState(cloudPipeline)
            enc.setTexture(cloudTex, index: 0)
            enc.setBytes(&info, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        if let c = cloudCenter { f.cloud = SIMD4(Float(c.x - span / 2), Float(c.y - span / 2), Float(Renderer.cloudTexel), 0) }
        // The slow part of the cloud cover evolves and drifts: baked every frame around the camera, in world coordinates.
        let sn = Renderer.slowTexels, sspan = Double(sn) * Renderer.slowTexel
        if slowTex == nil {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: sn, height: sn, mipmapped: false)
            td.usage = [.shaderRead, .shaderWrite]
            td.storageMode = .private
            slowTex = device.makeTexture(descriptor: td)
        }
        guard let slowTex, let enc = cb.makeComputeCommandEncoder() else { return }
        let sc = (SIMD2(camera.position.x, camera.position.z) / Renderer.slowTexel).rounded(.down) * Renderer.slowTexel
        var info = SIMD4<Float>(Float(sc.x - sspan / 2), Float(sc.y - sspan / 2), Float(Renderer.slowTexel), Float(simTime))
        var seed = world.seed
        enc.setComputePipelineState(slowPipeline)
        enc.setTexture(slowTex, index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        enc.setBytes(&seed, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: sn, height: sn, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        f.slowCloud = info
    }
}
