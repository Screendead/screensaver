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
        let cam2 = SIMD2(camera.position.x, camera.position.z)
        if provCentre.map({ simd_length($0 - cam2) > Double(Renderer.provN) * Renderer.provCell / 8 }) ?? true,
           let enc = cb.makeComputeCommandEncoder() {
            provCentre = cam2
            var info = SIMD4<Int32>(Int32(floor(cam2.x / Renderer.provCell)) - Int32(Renderer.provN / 2),
                                    Int32(floor(cam2.y / Renderer.provCell)) - Int32(Renderer.provN / 2), 0, Int32(bitPattern: world.seed))
            enc.setComputePipelineState(provPipeline)
            enc.setTexture(provTex, index: 0)
            enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            enc.dispatchThreads(MTLSize(width: Renderer.provN, height: Renderer.provN, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        // The shadow maps and the horizon share no data: one concurrent pass, so their short dispatches overlap.
        let bakes = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        if let bakes { bakeShadows(into: bakes, frame: &f) }
        if Renderer.profile != "nohorizon", let enc = bakes {
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

        let sp = MTLRenderPassDescriptor()
        sp.colorAttachments[0].texture = sceneTex
        sp.colorAttachments[0].loadAction = .dontCare
        sp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: sp) {
            // The canopies' band test only skips through cells inside the 1-voxel map's unmorphed disc (trace in
            // Shaders/Saver.metal), so it is compiled in only while bands lie that close; islands' bands hold on every
            // ring, needed out to isleShadeReach.
            let cam2 = SIMD2(camera.position.x, camera.position.z)
            let tb = camera.basis, tanV = tan(25.0 * .pi / 180) * 1.05
            let half = atan(tanV * (1 + Double(aspect) * Double(aspect)).squareRoot())
            // On while an island within the reach could show, off a little past it, so a camera hovering at the edge does
            // not flick it.
            let key = Renderer.keyLight(clock.sunDirection).0
            let isleSpheres = Islands.spheres(world.terrain.isles, eye: camera.position, within: Renderer.isleShadeReach + (isleVariant ? 400 : 0), key: key)
            isleVariant = Islands.inView(isleSpheres, eye: camera.position, fwd: tb.fwd, half: half, margin: isleVariant ? 0.12 : 0.06)
            let isles = isleVariant
            isleShapeVariant = isles || Islands.inView(Islands.spheres(world.terrain.isles, eye: camera.position, within: edgeRadius + 1000, key: key),
                                                       eye: camera.position, fwd: tb.fwd, half: half, margin: isleShapeVariant ? 0.12 : 0.06)
            let canopyBands = world.terrain.bandsNear(cam2, radius: Clipmap.handoff(0) + 40)
            let noband = Renderer.profile == "noband"
            f.shadow.w = isles && !noband ? 1 : 0
            let isleTable = Islands.table(world.terrain.isles, near: cam2, within: 3000, view: (camera.position, tb.fwd, half))
            var isleBytes = isleTable.bytes
            towers.frameTable(eye: camera.position, fwd: tb.fwd, right: tb.right, up: tb.up, tanH: tanV * Double(aspect), tanV: tanV)
            // Per tile: 1 an island's surfaces may show, 2 its shade or sky occlusion, 8 close by one within isleShadeReach,
            // 4 its mist.
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
            let variant = tileNeed.map { need -> Int in
                let shapeT = isleShapeVariant && need & 1 != 0
                if isles && (need & 8 != 0 || (need & 2 != 0 && shapeT)) { return need & 4 != 0 ? 6 : 5 }
                if isles && need & 2 != 0 { return 4 }
                return (bandsT ? 1 : 0) + (shapeT ? 2 : 0)
            }
            // Rectangles of tiles of one variant, each drawn as the whole frame's triangle scissored to it: a pair of
            // triangles per tile would shade the 2x2 quads along every diagonal twice.
            var rects: [(v: Int, x0: Int, x1: Int, y0: Int, y1: Int)] = []
            var open: [Int: Int] = [:]
            for ty in 0..<ny {
                var next: [Int: Int] = [:]
                var tx = 0
                while tx < nx {
                    let v = variant[ty * nx + tx]
                    var x1 = tx
                    while x1 + 1 < nx && variant[ty * nx + x1 + 1] == v { x1 += 1 }
                    if let i = open[tx], rects[i].x1 == x1, rects[i].v == v {
                        rects[i].y1 = ty
                        next[tx] = i
                    } else {
                        rects.append((v, tx, x1, ty, ty))
                        next[tx] = rects.count - 1
                    }
                    tx = x1 + 1
                }
                open = next
            }
            enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
            enc.setFragmentBuffer(birdBuf, offset: 0, index: 5)
            enc.setFragmentBuffer(horizonBuf, offset: 0, index: 6)
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
            enc.setFragmentTexture(clipmap.band, index: 18)
            enc.setFragmentBuffer(lampPos, offset: 0, index: 2)
            enc.setFragmentBuffer(lampState, offset: 0, index: 3)
            enc.setFragmentBytes(&hazeRefs, length: 128, index: 4)
            enc.setFragmentBytes(&tileMasks, length: tileMasks.count * 4, index: 7)
            enc.setFragmentBuffer(world.giantBuffer, offset: 0, index: 8)
            enc.setFragmentBytes(towers.table, length: Towers.tableBytes, index: 9)
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
        // The giants' hanging strands over the scene (vineFragment): the scene's bindings less the birds and the lamp
        // references, plus the strand maps. VR_PROFILE=bits2048 leaves them out.
        if vines.tableCount > 0 && !(Renderer.profile.hasPrefix("bits") && (Int(Renderer.profile.dropFirst(4)) ?? 0) & 2048 != 0) {
            let vp = MTLRenderPassDescriptor()
            vp.colorAttachments[0].texture = sceneTex
            vp.colorAttachments[0].loadAction = .load
            vp.colorAttachments[0].storeAction = .store
            if let enc = cb.makeRenderCommandEncoder(descriptor: vp) {
                enc.setRenderPipelineState(vinePipelines[clock.cloudSea > 0 ? 1 : 0])
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
                enc.setFragmentBytes(vines.table, length: Vines.tableBytes, index: 10)
                enc.setFragmentTexture(vines.fine, index: 18)
                enc.setFragmentTexture(vines.coarse, index: 19)
                enc.setFragmentTexture(vines.sup, index: 20)
                enc.setFragmentTexture(vines.bands, index: 21)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
        }
        // In its own pass: drawn after the scene in the scene's pass, even a draw doing nothing cost a twilight scene ~0.4 ms.
        if f.sunDir.y < Renderer.bioSun {
            let bp = MTLRenderPassDescriptor()
            bp.colorAttachments[0].texture = sceneTex
            bp.colorAttachments[0].loadAction = .load
            bp.colorAttachments[0].storeAction = .store
            if let enc = cb.makeRenderCommandEncoder(descriptor: bp) {
                enc.setRenderPipelineState(bioPipeline)
                enc.setVertexBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
                enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
                enc.setVertexTexture(world.heightTex, index: 0)
                enc.setVertexTexture(world.climateTex, index: 2)
                enc.setVertexTexture(provTex, index: 16)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: Renderer.moteCount)
                enc.endEncoding()
            }
        }
        let tpd = MTLRenderPassDescriptor()
        tpd.colorAttachments[0].texture = histOut
        tpd.colorAttachments[0].loadAction = .dontCare
        tpd.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: tpd) {
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
        let pp = MTLRenderPassDescriptor()
        pp.colorAttachments[0].texture = output
        pp.colorAttachments[0].loadAction = .dontCare
        pp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: pp) {
            enc.setRenderPipelineState(postPipeline)
            enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
            enc.setFragmentTexture(histOut, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
    }

    // The tiles (nx x ny of tw x th pixels over a width x height frame, rows down the screen) whose rays may cross the box
    // lo...hi, padded by pad pixels for the subpixel jitter: the part of it in front of the eye, by its corners and where its
    // edges cross the near plane; nil: none.
    private func tileRange(box lo: SIMD3<Double>, _ hi: SIMD3<Double>, nx: Int, ny: Int, tw: Double, th: Double, width: Int, height: Int,
                           pad: Double) -> (x0: Int, x1: Int, y0: Int, y1: Int)? {
        let b = camera.basis
        let tanV = tan(25.0 * .pi / 180), aspect = Double(width) / Double(height), near = 0.5
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
