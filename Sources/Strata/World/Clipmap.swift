import Foundation
import Metal
import simd

// Coarse LOD rings beyond the 1-voxel map. Level k (1...4) has columns 2^k voxels wide, 2048 to a side, kept
// toroidally (texel = world cell & 2047) around the camera, so re-centring rewrites only the 64-cell strip that
// enters the window, a chunk per frame inside the frame's own command buffer.
final class Clipmap {
    static let levels = 4, size = 2048, snap = 64, chunk = 128
    // Columns of a whole-window build per piece, so other queues' frames interleave between its dispatches.
    static let fullChunk = 32
    // Must match Frame.lod in Shaders/Saver.metal: a level hands off to the next at 860 of its own columns,
    // which keeps every read inside the window while a strip is rewritten.
    static func handoff(_ k: Int) -> Double { 860.0 * Double(1 << k) }

    let height: MTLTexture   // r16Float array: height; max mips (from mip 1 also bounding the next level)
    let cover: MTLTexture    // rg8Unorm array: canopy cover, depth of leaves under the top / 64 (coarseVeg)
    let climate: MTLTexture  // rgba16Float array at half resolution
    // Must match GAP_LEVELS in Shaders/Saver.metal: levels 1...gapLevels carry one gap per column.
    static let gapLevels = 4
    // Must match GAP2_LEVELS: levels 1...gap2Levels also carry a second gap (giants' crowns).
    static let gap2Levels = 1
    let gaps: MTLTexture     // rg16Float array, levels 1...gapLevels: the column's gap (lo, hi)
    let gaps2: MTLTexture    // rg16Float array, levels 1...gap2Levels: a second gap below it (a giant crown's air)
    private let gapViews: [MTLTexture]
    // rg16Float array at half size with mips: band mip m - 1 holds the air every column of a height mip m cell leaves
    // open (under islands), whose height texels are then negated.
    let band: MTLTexture
    private let bandViews: [[MTLTexture]]
    private let bandTopPipeline: MTLComputePipelineState
    private let bandReducePipeline: MTLComputePipelineState
    private let bandFlagPipeline: MTLComputePipelineState
    private let gap2Views: [MTLTexture]
    private let heightViews: [[MTLTexture]]
    private let coverViews: [MTLTexture]
    private let climateViews: [MTLTexture]
    private let genPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let parentPipeline: MTLComputePipelineState
    private let seed: UInt32
    private var lo: [SIMD2<Int32>?] = Array(repeating: nil, count: Clipmap.levels)
    private struct Job { var level: Int; var x0: Int32; var z0: Int32; var w: Int32; var h: Int32; var done: Int32; var newLo: SIMD2<Int32> }
    private var job: Job?
    private var completed = [Int](repeating: 0, count: Clipmap.levels)

    init?(gpu: GPU, seed: UInt32) {
        let device = gpu.device
        let pipeline = gpu.pipeline as (String) -> MTLComputePipelineState?
        func array(_ format: MTLPixelFormat, _ n: Int, mips: Bool, slices: Int = Clipmap.levels) -> MTLTexture? {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = format
            d.width = n
            d.height = n
            d.arrayLength = slices
            d.mipmapLevelCount = mips ? Int(log2(Double(n))) + 1 : 1
            d.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        guard let h = array(.r16Float, Clipmap.size, mips: true), let v = array(.rg8Unorm, Clipmap.size, mips: false),
              let g = array(.rg16Float, Clipmap.size, mips: false, slices: Clipmap.gapLevels),
              let g2 = array(.rg16Float, Clipmap.size, mips: false, slices: Clipmap.gap2Levels),
              let c = array(.rgba16Float, Clipmap.size / 2, mips: false),
              let bnd = array(.rg16Float, Clipmap.size / 2, mips: true),
              let bTop = pipeline("coarseBandTop"), let bRed = pipeline("coarseBandReduce"), let bFlag = pipeline("coarseBandFlag"),
              let gen = pipeline("generateCoarse"), let red = pipeline("reduceMaxRegion"), let par = pipeline("reduceMaxParent") else { return nil }
        func slice(_ t: MTLTexture, _ k: Int, _ m: Int) -> MTLTexture? {
            t.makeTextureView(pixelFormat: t.pixelFormat, textureType: .type2D, levels: m..<(m + 1), slices: k..<(k + 1))
        }
        var hv: [[MTLTexture]] = [], vv: [MTLTexture] = [], cv: [MTLTexture] = []
        for k in 0..<Clipmap.levels {
            var mips: [MTLTexture] = []
            for m in 0..<h.mipmapLevelCount {
                guard let t = slice(h, k, m) else { return nil }
                mips.append(t)
            }
            hv.append(mips)
            guard let tv = slice(v, k, 0), let tc = slice(c, k, 0) else { return nil }
            vv.append(tv)
            cv.append(tc)
        }
        var bv: [[MTLTexture]] = []
        for k in 0..<Clipmap.levels {
            var mips: [MTLTexture] = []
            for m in 0..<bnd.mipmapLevelCount {
                guard let t = slice(bnd, k, m) else { return nil }
                mips.append(t)
            }
            bv.append(mips)
        }
        band = bnd
        bandViews = bv
        bandTopPipeline = bTop
        bandReducePipeline = bRed
        bandFlagPipeline = bFlag
        var gv: [MTLTexture] = [], g2v: [MTLTexture] = []
        for k in 0..<Clipmap.gapLevels {
            guard let t = slice(g, k, 0) else { return nil }
            gv.append(t)
        }
        for k in 0..<Clipmap.gap2Levels {
            guard let t = slice(g2, k, 0) else { return nil }
            g2v.append(t)
        }
        gaps = g
        gaps2 = g2
        gapViews = gv
        gap2Views = g2v
        height = h
        cover = v
        climate = c
        heightViews = hv
        coverViews = vv
        climateViews = cv
        genPipeline = gen
        reducePipeline = red
        parentPipeline = par
        self.seed = seed
    }

    // Window origin (in level cells) that centres level k on world xz `center`.
    private static func target(_ center: SIMD2<Double>, level k: Int) -> SIMD2<Int32> {
        let s = Double(1 << k)
        func f(_ v: Double) -> Int32 { Int32(((floor(v / s) - Double(size / 2)) / Double(snap)).rounded(.down)) * Int32(snap) }
        return SIMD2(f(center.x), f(center.y))
    }

    var idle: Bool { job == nil }

    // Level k (1...levels) holds a whole window.
    func ready(_ k: Int) -> Bool { lo[k - 1] != nil }

    // Counts the jobs encoded for level k: its contents change only when this does.
    func generation(_ k: Int) -> Int { completed[k - 1] }

    // Forgets the windows, so the next encode rebuilds every level whole (VR_COARSE_FULL: checks strips against it).
    func invalidate() { lo = Array(repeating: nil, count: Clipmap.levels); job = nil }

    private func nextJob(_ center: SIMD2<Double>, active: Int, only: Int?) -> Job? {
        let n = Int32(Clipmap.size), step = Int32(Clipmap.snap)
        for i in only.map({ [$0 - 1] }) ?? Array(0..<active) {
            let t = Clipmap.target(center, level: i + 1)
            guard let cur = lo[i], abs(t.x - cur.x) <= 8 * step, abs(t.y - cur.y) <= 8 * step else {
                lo[i] = nil
                return Job(level: i, x0: t.x, z0: t.y, w: n, h: n, done: 0, newLo: t)
            }
            if t.x != cur.x {
                let dir: Int32 = t.x > cur.x ? 1 : -1
                return Job(level: i, x0: dir > 0 ? cur.x + n : cur.x - step, z0: cur.y, w: step, h: n, done: 0,
                           newLo: SIMD2(cur.x + dir * step, cur.y))
            }
            if t.y != cur.y {
                let dir: Int32 = t.y > cur.y ? 1 : -1
                return Job(level: i, x0: cur.x, z0: dir > 0 ? cur.y + n : cur.y - step, w: n, h: step, done: 0,
                           newLo: SIMD2(cur.x, cur.y + dir * step))
            }
        }
        return nil
    }

    // Encodes pending generation for levels 1...active (or level `only` alone): one piece, or with `drain` everything
    // the camera position needs. Levels past `active` are dropped, to be rebuilt whole once they are active again.
    // giants, isles: the giant and island tables of the world's active map, which cover every window here.
    func encode(into cb: MTLCommandBuffer, center: SIMD2<Double>, drain: Bool, active: Int, giants: MTLBuffer, isles: MTLBuffer, only: Int? = nil) {
        for i in active..<Clipmap.levels { lo[i] = nil }
        if let j = job, j.level >= active || only.map({ j.level != $0 - 1 }) ?? false { job = nil }
        // A level more than a strip behind the camera could soon be read past its window: catch it up at once. In
        // flight that never happens: at 15 voxels/s a level-1 strip falls due at most every 8.5 s and takes 17
        // frames. Only the outermost level, which the edge waits for, is ever built whole in pieces.
        let lagging = (0..<active).contains { i in
            let t = Clipmap.target(center, level: i + 1)
            return lo[i].map { abs($0.x - t.x) > Int32(Clipmap.snap) || abs($0.y - t.y) > Int32(Clipmap.snap) } ?? (i < active - 1)
        }
        var budget = drain || lagging ? Int.max : 1
        if job == nil { job = nextJob(center, active: active, only: only) }
        guard job != nil, let enc = cb.makeComputeCommandEncoder() else { return }
        defer { enc.endEncoding() }
        while budget > 0 {
            if job == nil { job = nextJob(center, active: active, only: only) }
            guard var j = job else { return }
            let full = j.w == Int32(Clipmap.size) && j.h == Int32(Clipmap.size)
            let long = max(j.w, j.h)
            let piece = min(Int32(full ? Clipmap.fullChunk : Clipmap.chunk), long - j.done)
            var region = j.w >= j.h ? SIMD4<Int32>(j.x0 + j.done, j.z0, piece, j.h) : SIMD4<Int32>(j.x0, j.z0 + j.done, j.w, piece)
            var info = SIMD4<Int32>(Int32(j.level + 1), Int32(bitPattern: seed), 0, 0)
            enc.setComputePipelineState(genPipeline)
            enc.setTexture(heightViews[j.level][0], index: 0)
            enc.setTexture(climateViews[j.level], index: 1)
            enc.setTexture(coverViews[j.level], index: 2)
            enc.setTexture(gapViews[min(j.level, Clipmap.gapLevels - 1)], index: 3)
            // Written only on levels 1...gap2Levels.
            enc.setTexture(gap2Views[min(j.level, Clipmap.gap2Levels - 1)], index: 4)
            enc.setBytes(&region, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
            enc.setBuffer(giants, offset: 0, index: 2)
            enc.setBuffer(isles, offset: 0, index: 3)
            enc.dispatchThreads(MTLSize(width: Int(region.z), height: Int(region.w), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            j.done += piece
            budget -= 1
            if j.done < long { job = j; continue }
            let views = heightViews[j.level]
            // Mip 1 over the rewritten region, in world mip-1 cells: each is one column of the next level, whose
            // height it also bounds. The top level has no next level to morph toward.
            var r1 = SIMD4<Int32>(j.x0 >> 1, j.z0 >> 1, j.w >> 1, j.h >> 1)
            enc.setComputePipelineState(j.level + 1 < Clipmap.levels ? parentPipeline : reducePipeline)
            enc.setTexture(views[0], index: 0)
            enc.setTexture(views[1], index: 1)
            if j.level + 1 < Clipmap.levels {
                enc.setTexture(gapViews[min(j.level, Clipmap.gapLevels - 1)], index: 2)
                enc.setTexture(bandViews[j.level][0], index: 3)
                enc.setBytes(&r1, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                enc.setBuffer(isles, offset: 0, index: 3)
            } else {
                let mask = Int32(Clipmap.size / 2 - 1)
                var t1 = SIMD4<Int32>(j.w == Int32(Clipmap.size) ? 0 : r1.x & mask, j.h == Int32(Clipmap.size) ? 0 : r1.y & mask, r1.z, r1.w)
                enc.setBytes(&t1, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            }
            enc.dispatchThreads(MTLSize(width: Int(r1.z), height: Int(r1.w), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            // Coarser mips: the window is 64-aligned, so the region is contiguous in storage unless it spans the
            // whole axis. Coarser than the strip: rebuild the whole (tiny) mip. Only the top one is read past mip 6 (the
            // trace's MAX_LEVEL), for the global height bound; the rest cost a few tiny dispatches.
            let n = Int32(Clipmap.size), mask = n - 1
            let sx = j.w == n ? 0 : j.x0 & mask, sz = j.h == n ? 0 : j.z0 & mask
            func mipRegion(_ m: Int) -> SIMD4<Int32> {
                let dim = Int32(Clipmap.size >> m)
                return m <= 6 ? SIMD4<Int32>(sx >> Int32(m), sz >> Int32(m), max(j.w >> Int32(m), 1), max(j.h >> Int32(m), 1)) : SIMD4<Int32>(0, 0, dim, dim)
            }
            let bviews = bandViews[j.level]
            if j.level + 1 == Clipmap.levels {
                enc.setComputePipelineState(bandTopPipeline)
                enc.setTexture(views[0], index: 0)
                enc.setTexture(gapViews[min(j.level, Clipmap.gapLevels - 1)], index: 1)
                enc.setTexture(bviews[0], index: 2)
                enc.setBytes(&r1, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Int(r1.z), height: Int(r1.w), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            }
            for m in 2..<views.count {
                var r = mipRegion(m)
                enc.setComputePipelineState(bandReducePipeline)
                enc.setTexture(bviews[m - 2], index: 0)
                enc.setTexture(bviews[m - 1], index: 1)
                enc.setBytes(&r, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Int(r.z), height: Int(r.w), depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(16, Int(r.z)), height: min(16, Int(r.w)), depth: 1))
            }
            enc.setComputePipelineState(reducePipeline)
            for m in 2..<views.count {
                var r = mipRegion(m)
                enc.setTexture(views[m - 1], index: 0)
                enc.setTexture(views[m], index: 1)
                enc.setBytes(&r, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Int(r.z), height: Int(r.w), depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(16, Int(r.z)), height: min(16, Int(r.w)), depth: 1))
            }
            enc.setComputePipelineState(bandFlagPipeline)
            for m in 1..<views.count {
                var r = mipRegion(m)
                enc.setTexture(views[m], index: 0)
                enc.setTexture(bviews[m - 1], index: 1)
                enc.setBytes(&r, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Int(r.z), height: Int(r.w), depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(16, Int(r.z)), height: min(16, Int(r.w)), depth: 1))
            }
            lo[j.level] = j.newLo
            completed[j.level] += 1
            if Renderer.logTiming { print("coarse level \(j.level + 1) window \(j.newLo) (\(j.w)x\(j.h))") }
            job = nil
        }
    }
}
