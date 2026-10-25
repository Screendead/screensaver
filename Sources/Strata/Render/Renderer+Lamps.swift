import Foundation
import Metal
import simd

extension Renderer {
    // The lamps nearest the camera that glow in their own haze; the shader reads their state, so the list
    // needs no schedule.
    private func pickHazeLamps() {
        let cam = SIMD2(camera.position.x, camera.position.z)
        var near: [(Double, Int)] = []
        for site in world.sites {
            for l in site.lamps where l.kind != 3 && l.kind != 7 {
                let d = simd_length(SIMD2(l.pos.x, l.pos.z) - cam)
                if d < 1200 { near.append((d, l.index)) }
            }
        }
        near.sort { $0.0 < $1.0 }
        hazeCount = min(near.count, Renderer.hazeLamps)
        for i in 0..<hazeCount { hazeRefs[i] = Int32(near[i].1) }
    }

    // Fires and chimneys within 600 that the camera can see, nearest first.
    func pickSmoke() {
        let cam = camera.position
        let fwd = camera.basis.fwd
        var near: [(Double, Int)] = []
        for site in world.sites {
            for l in site.lamps where l.rise > 0 {
                let v = l.smoke - cam
                let d = simd_length(v)
                if d < 600, d < 40 || simd_dot(v, fwd) > -0.2 * d { near.append((d, l.index)) }
            }
        }
        near.sort { $0.0 < $1.0 }
        smokeCount = min(near.count, Renderer.smokePlumes)
        for i in 0..<smokeCount { hazeRefs[16 + i] = Int32(near[i].1) }
    }

    // Marks the tiles a sphere's projection may touch: its bounding box's corners bound the projection, padded by a few
    // pixels for the subpixel jitter. A sphere reaching behind the near plane marks every tile.
    func fillLampTiles(frame f: Frame, width: Int, height: Int) {
        let n = Renderer.lampTiles
        for i in 0..<tileMasks.count { tileMasks[i] = 0 }
        let b = camera.basis, cam = camera.position
        let tanV = tan(25.0 * .pi / 180), aspect = Double(width) / Double(height)
        let padX = 6.0 / Double(width), padY = 6.0 / Double(height)
        func mark(_ c: SIMD3<Double>, _ r: Double, _ bit: UInt32) {
            let v = c - cam
            let x = simd_dot(v, b.right), y = simd_dot(v, b.up), z = simd_dot(v, b.fwd)
            var x0 = 0, x1 = n - 1, y0 = 0, y1 = n - 1
            if z - r > 0.5 {
                var nx0 = Double.infinity, nx1 = -Double.infinity, ny0 = Double.infinity, ny1 = -Double.infinity
                for zz in [z - r, z + r] {
                    for xx in [x - r, x + r] { let q = xx / (zz * tanV * aspect); nx0 = min(nx0, q); nx1 = max(nx1, q) }
                    for yy in [y - r, y + r] { let q = yy / (zz * tanV); ny0 = min(ny0, q); ny1 = max(ny1, q) }
                }
                nx0 -= padX; nx1 += padX; ny0 -= padY; ny1 += padY
                if nx1 < -1 || nx0 > 1 || ny1 < -1 || ny0 > 1 { return }
                x0 = max(0, Int(floor((nx0 + 1) / 2 * Double(n)))); x1 = min(n - 1, Int(floor((nx1 + 1) / 2 * Double(n))))
                // Tile rows run down the screen, NDC y up.
                y0 = max(0, Int(floor((1 - ny1) / 2 * Double(n)))); y1 = min(n - 1, Int(floor((1 - ny0) / 2 * Double(n))))
            }
            for ty in y0...y1 { for tx in x0...x1 { tileMasks[ty * n + tx] |= bit } }
        }
        let lamps = world.lamps
        for i in 0..<Int(f.refs.x) {
            guard let l = lamps[Int(hazeRefs[i])] else { continue }
            mark(l.pos, 3 * Renderer.lampReach[min(l.kind, 7)] + 0.5, 1 << UInt32(i))
        }
        let wind = SIMD2(0.825336, 0.564642)
        for i in 0..<Int(f.refs.y) {
            guard let l = lamps[Int(hazeRefs[16 + i])] else { continue }
            mark(l.smoke + SIMD3(wind.x * 6, 14, wind.y * 6), 16.5, 1 << UInt32(16 + i))
        }
    }

    func bakeLamps(into cb: MTLCommandBuffer, frame f: inout Frame) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        let n = MapLayout.siteGrid * MapLayout.siteGrid * MapLayout.maxLamps
        enc.setComputePipelineState(lampPipeline)
        enc.setBuffer(world.siteBuffer, offset: 0, index: 0)
        enc.setBuffer(lampPos, offset: 0, index: 1)
        enc.setBuffer(lampState, offset: 0, index: 2)
        enc.setBytes(&f, length: MemoryLayout<Frame>.stride, index: 3)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if f.lights.w > 0 {
            pickHazeLamps()
            f.refs.x = Int32(hazeCount)
            enc.setComputePipelineState(lightPipeline)
            enc.setTexture(lightMap, index: 0)
            enc.setTexture(lightDir, index: 1)
            let t = Renderer.lightTexels
            enc.dispatchThreads(MTLSize(width: t, height: t, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
        enc.endEncoding()
    }
}
