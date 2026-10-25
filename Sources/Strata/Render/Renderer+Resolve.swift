import Metal

extension Renderer {
    // The temporal resolve into this frame's history, and again round the evening star in a scissor (STAR_FC).
    func encodeResolve(into cb: MTLCommandBuffer, scene sceneTex: MTLTexture, history histIn: MTLTexture, into histOut: MTLTexture,
                               frame f: inout Frame, cells: (Int, Int)) {
        let (ow, oh) = cells
        guard let enc = cb.makeRenderCommandEncoder(descriptor: Renderer.pass(histOut, load: false)) else { return }
        enc.setRenderPipelineState(taaPipeline)
        enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
        enc.setFragmentTexture(sceneTex, index: 0)
        enc.setFragmentTexture(histIn, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        let px = Int(f.planetPx.x), py = Int(f.planetPx.y)
        if let sp = taaStarPipeline, px >= -4, py >= -4, px <= ow + 4, py <= oh + 4 {
            let x0 = max(px - 5, 0), y0 = max(py - 5, 0)
            enc.setRenderPipelineState(sp)
            enc.setScissorRect(MTLScissorRect(x: x0, y: y0, width: min(px + 5, ow) - x0, height: min(py + 5, oh) - y0))
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
    }

    func encodePost(into cb: MTLCommandBuffer, history histOut: MTLTexture, output: MTLTexture, frame f: inout Frame) {
        guard let enc = cb.makeRenderCommandEncoder(descriptor: Renderer.pass(output, load: false)) else { return }
        enc.setRenderPipelineState(postPipeline)
        enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
        enc.setFragmentTexture(histOut, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }
}
