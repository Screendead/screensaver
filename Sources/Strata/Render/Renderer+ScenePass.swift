import Foundation
import Metal
import StrataCore
import simd

extension Renderer {
    func encodeScene(into cb: MTLCommandBuffer, target sceneTex: MTLTexture, frame f: inout Frame, birds birdBuf: MTLBuffer,
                             width w: Int, height h: Int) {
        guard let enc = cb.makeRenderCommandEncoder(descriptor: Renderer.pass(sceneTex, load: false)) else { return }
        // The canopies' band test only skips through cells inside the 1-voxel map's unmorphed disc (trace in
        // Shaders/Saver.metal), so it is compiled in only while bands lie that close; islands' bands hold on every
        // ring, needed out to isleShadeReach.
        let cam2 = camera.xz
        let tb = camera.basis, tanV = Renderer.tanV * 1.05
        let half = atan(tanV * (1 + Double(aspect) * Double(aspect)).squareRoot())
        // On while an island within the reach could show, off a little past it, so a camera hovering at the edge does
        // not flick it.
        let key = SkyClock.keyLight(clock.sunDirection).0
        let isleSpheres = Islands.spheres(world.terrain.isles, eye: camera.position, within: Renderer.isleShadeReach + (isleVariant ? 400 : 0), key: key)
        isleVariant = Islands.inView(isleSpheres, eye: camera.position, fwd: tb.fwd, half: half, margin: isleVariant ? 0.12 : 0.06)
        let isles = isleVariant
        isleShapeVariant = isles || Islands.inView(Islands.spheres(world.terrain.isles, eye: camera.position, within: edgeRadius + 1000, key: key),
                                                   eye: camera.position, fwd: tb.fwd, half: half, margin: isleShapeVariant ? 0.12 : 0.06)
        let canopyBands = world.terrain.bandsNear(cam2, radius: Clipmap.handoff(0) + 40)
        let noband = Knobs.current.profile == "noband"
        f.shadow.w = isles && !noband ? 1 : 0
        let isleTable = Islands.table(world.terrain.isles, near: cam2, within: 3000, view: (camera.position, tb.fwd, half))
        var isleBytes = isleTable.bytes
        towers.frameTable(eye: camera.position, fwd: tb.fwd, right: tb.right, up: tb.up, tanH: tanV * Double(aspect), tanV: tanV)
        // Per tile, the need bits SceneTiles.variant reads.
        let T = Renderer.sceneTile, nx = (w + T - 1) / T, ny = (h + T - 1) / T
        var tileNeed = [UInt8](repeating: 0, count: nx * ny)
        func mark(_ list: [Islands.Box], _ bit: UInt8) {
            for b in list {
                guard let t = tileRange(box: b.lo, b.hi, nx: nx, ny: ny, tw: Double(T), th: Double(T), width: w, height: h, pad: 6) else { continue }
                for ty in t.y0...t.y1 { for tx in t.x0...t.x1 { tileNeed[ty * nx + tx] |= bit } }
            }
        }
        if isleShapeVariant {
            mark(Islands.boxes(world.terrain.isles, eye: camera.position, within: edgeRadius + 1000, key: key, columns: true, shade: false), 1)
        }
        if isles {
            let within = Renderer.isleShadeReach + 400
            mark(Islands.boxes(world.terrain.isles, eye: camera.position, within: within, key: key, columns: true, shade: false), 8)
            mark(Islands.boxes(world.terrain.isles, eye: camera.position, within: within, key: key, columns: false, shade: true), 2)
            mark(isleTable.mist, 12)
            // The sky occlusion reaches no farther than the shading table, which holds every island it darkens.
            mark(world.terrain.isles.filter { simd_length($0.pos - cam2) < 3000 + $0.reach }.map(Islands.occlusionBox), 2)
        }
        // Island bands only where an island's work does: elsewhere its air is walked as any rock gap's.
        let bandsT = canopyBands && !noband
        let rects = SceneTiles.rects(tileNeed.map { SceneTiles.variant(need: $0, isles: isles, shapes: isleShapeVariant, bands: bandsT) }, nx: nx, ny: ny)
        bindShared(enc, frame: &f)
        enc.setFragmentBuffer(birdBuf, offset: 0, index: 5)
        enc.setFragmentBuffer(horizonBuf, offset: 0, index: 6)
        enc.setFragmentTexture(clipmap.band, index: 18)
        enc.setFragmentBytes(&hazeRefs, length: 128, index: 4)
        enc.setFragmentBytes(&isleBytes, length: Islands.tableBytes, index: 10)
        for v in 0..<GPU.sceneVariants.count {
            let mine = rects.filter { $0.v == v }
            guard !mine.isEmpty else { continue }
            enc.setRenderPipelineState(scenePipelines[clock.cloudSea > 0 ? 1 : 0][towers.tableCount > 0 ? 1 : 0][v])
            for r in mine {
                let x0 = r.x0 * T, y0 = r.y0 * T
                enc.setScissorRect(MTLScissorRect(x: x0, y: y0, width: min((r.x1 + 1) * T, w) - x0, height: min((r.y1 + 1) * T, h) - y0))
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            }
        }
        enc.endEncoding()
    }

    // The fragment bindings the scene and strand passes share.
    private func bindShared(_ enc: MTLRenderCommandEncoder, frame f: inout Frame) {
        enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
        enc.setFragmentTexture(world.heightTex, index: 0)
        enc.setFragmentTexture(cloudTex, index: 1)
        enc.setFragmentTexture(world.climateTex, index: 2)
        enc.setFragmentBuffer(world.siteBuffer, offset: 0, index: 1)
        enc.setFragmentTexture(world.structTex, index: 3)
        enc.setFragmentTexture(lightMap, index: 4)
        enc.setFragmentTexture(lightDir, index: 5)
        enc.setFragmentTexture(clipmap.height, index: 6)
        enc.setFragmentTexture(clipmap.climate, index: 7)
        enc.setFragmentTexture(clipmap.cover, index: 8)
        enc.setFragmentTexture(shadowMaps[0], index: 9)
        enc.setFragmentTexture(shadowMaps[1], index: 10)
        enc.setFragmentTexture(slowTex, index: 11)
        enc.setFragmentTexture(world.maxTex, index: 12)
        enc.setFragmentTexture(world.gapTex, index: 13)
        enc.setFragmentTexture(clipmap.gaps, index: 14)
        enc.setFragmentTexture(clipmap.gaps2, index: 17)
        enc.setFragmentTexture(world.bandTex, index: 15)
        enc.setFragmentTexture(provTex, index: 16)
        enc.setFragmentBuffer(lampPos, offset: 0, index: 2)
        enc.setFragmentBuffer(lampState, offset: 0, index: 3)
        enc.setFragmentBytes(&tileMasks, length: tileMasks.count * 4, index: 7)
        enc.setFragmentBuffer(world.giantBuffer, offset: 0, index: 8)
        enc.setFragmentBytes(towers.table, length: Towers.tableBytes, index: 9)
    }

    func encodeVines(into cb: MTLCommandBuffer, target sceneTex: MTLTexture, frame f: inout Frame) {
        guard let enc = cb.makeRenderCommandEncoder(descriptor: Renderer.pass(sceneTex, load: true)) else { return }
        enc.setRenderPipelineState(vinePipelines[clock.cloudSea > 0 ? 1 : 0])
        bindShared(enc, frame: &f)
        enc.setFragmentBytes(vines.table, length: Vines.tableBytes, index: 10)
        enc.setFragmentTexture(vines.fine, index: 18)
        enc.setFragmentTexture(vines.coarse, index: 19)
        enc.setFragmentTexture(vines.sup, index: 20)
        enc.setFragmentTexture(vines.bands, index: 21)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    func encodeBio(into cb: MTLCommandBuffer, target sceneTex: MTLTexture, frame f: inout Frame) {
        guard let enc = cb.makeRenderCommandEncoder(descriptor: Renderer.pass(sceneTex, load: true)) else { return }
        enc.setRenderPipelineState(bioPipeline)
        enc.setVertexBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
        enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
        enc.setVertexTexture(world.heightTex, index: 0)
        enc.setVertexTexture(world.climateTex, index: 2)
        enc.setVertexTexture(provTex, index: 16)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: Renderer.moteCount)
        enc.endEncoding()
    }

    // The tiles (nx x ny of tw x th pixels over a width x height frame, rows down the screen) whose rays may cross the box
    // lo...hi, padded by pad pixels for the subpixel jitter: the part of it in front of the eye, by its corners and where its
    // edges cross the near plane; nil: none.
    private func tileRange(box lo: SIMD3<Double>, _ hi: SIMD3<Double>, nx: Int, ny: Int, tw: Double, th: Double, width: Int, height: Int,
                           pad: Double) -> (x0: Int, x1: Int, y0: Int, y1: Int)? {
        let b = camera.basis
        let tanV = Renderer.tanV, aspect = Double(width) / Double(height), near = 0.5
        let cs = (0..<8).map { i -> SIMD3<Double> in
            let v = SIMD3(i & 1 == 0 ? lo.x : hi.x, i & 2 == 0 ? lo.y : hi.y, i & 4 == 0 ? lo.z : hi.z) - camera.position
            return SIMD3(simd_dot(v, b.right), simd_dot(v, b.up), simd_dot(v, b.fwd))
        }
        var pts = cs.filter { $0.z > near }
        for i in 0..<8 { for bit in [1, 2, 4] where i & bit == 0 {
            let a = cs[i], c = cs[i | bit]
            if (a.z > near) != (c.z > near) { pts.append(a + (c - a) * ((near - a.z) / (c.z - a.z))) }
        } }
        guard !pts.isEmpty else { return nil }
        var nx0 = Double.infinity, nx1 = -Double.infinity, ny0 = Double.infinity, ny1 = -Double.infinity
        for q in pts {
            let u = q.x / (q.z * tanV * aspect), v = q.y / (q.z * tanV)
            nx0 = min(nx0, u); nx1 = max(nx1, u); ny0 = min(ny0, v); ny1 = max(ny1, v)
        }
        let padX = 2 * pad / Double(width), padY = 2 * pad / Double(height)
        nx0 -= padX; nx1 += padX; ny0 -= padY; ny1 += padY
        if nx1 < -1 || nx0 > 1 || ny1 < -1 || ny0 > 1 { return nil }
        let W = Double(width), H = Double(height)
        return (max(0, Int(floor((nx0 + 1) / 2 * W / tw))), min(nx - 1, Int(floor((nx1 + 1) / 2 * W / tw))),
                max(0, Int(floor((1 - ny1) / 2 * H / th))), min(ny - 1, Int(floor((1 - ny0) / 2 * H / th))))
    }
}
