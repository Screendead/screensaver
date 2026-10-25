import Foundation
import Metal
import QuartzCore
import StrataCore
import simd

extension World {
    private static let passes = 7
    // Pieces per pass: the tables (giants in two halves, then the islands, which keep clear of them), then bands of rows;
    // planting (giants, the jungle) runs in half bands, the costliest columns of any pass; the band mips' first level
    // (under islands it works out the level-1 columns) a band of rows at a time too.
    private static let passBands = [3, MapLayout.size / bandRows, MapLayout.size / bandRows, MapLayout.size / bandRows, 2 * MapLayout.size / bandRows, MapLayout.size / bandRows, MapLayout.size / bandRows]
    private static var pieceCount: Int { passBands.reduce(0, +) + 1 }
    // Piece k's pass, band and rows per band (the max mips' piece is pass `passes`).
    static func piece(_ k: Int) -> (pass: Int, band: Int, rows: Int) {
        var k = k
        for (p, n) in passBands.enumerated() {
            if k < n { return (p, k, MapLayout.size / n) }
            k -= n
        }
        return (passes, 0, 0)
    }

    // Encodes the next `pieces` pieces of the job: the giant and island tables, then each pass (heights, cleanup,
    // structures, trees, climate) a band of rows at a time, finishing every band of a pass before the next pass reads its
    // neighbours, then the max mips.
    // Sites are placed once, ahead of the first structure band: they read the cleaned map, and every band stamps them.
    @discardableResult
    func advance(pieces: Int, wait: Bool, timed: Bool = false) -> Double {
        guard var j = job, let buf = buffers[j.index], let scratch, let cb = genQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return 0 }
        var info = SIMD4<Int32>(j.origin.x, j.origin.y, Int32(MapLayout.size), Int32(bitPattern: seed))
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        var n = pieces
        while n > 0 && j.next < World.pieceCount {
            let (pass, band, rows) = World.piece(j.next)
            if pass < World.passes {
                var row0 = UInt32(band * rows)
                switch pass {
                case 0:
                    // The tables, centred on the map's middle: the giants (half in each of two pieces; sites and trees read
                    // them), then the islands, which keep clear of them and which every column reads.
                    if band < 2 {
                        encodeGiants(enc, into: buf, origin: j.origin, half: band)
                    } else {
                        let mid = SIMD2<Double>(Double(j.origin.x), Double(j.origin.y)) + Double(MapLayout.size / 2)
                        var ii = SIMD4<Int32>(Int32(floor(mid.x)), Int32(floor(mid.y)), Knobs.current.isleField, Int32(bitPattern: seed))
                        enc.setComputePipelineState(islePipeline)
                        enc.setBuffer(buf.isles, offset: 0, index: 0)
                        enc.setBytes(&ii, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                        enc.setBuffer(buf.giants, offset: 0, index: 2)
                        enc.dispatchThreads(MTLSize(width: Islands.slots, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                        enc.setComputePipelineState(isleResolvePipeline)
                        enc.dispatchThreads(MTLSize(width: Islands.slots, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    }
                    j.next += 1
                    n -= 1
                    continue
                case 1:
                    // The cave table first: the heights carve its caves.
                    if band == 0 { encodeCaves(enc, into: buf, origin: j.origin) }
                    enc.setComputePipelineState(genPipeline)
                    enc.setTexture(scratch, index: 0)
                    enc.setTexture(buf.rawGaps, index: 1)
                    enc.setBuffer(buf.isles, offset: 0, index: 2)
                    enc.setBuffer(buf.giants, offset: MapLayout.caveOffset, index: 3)
                case 2:
                    enc.setComputePipelineState(cleanupPipeline)
                    enc.setTexture(scratch, index: 0)
                    enc.setTexture(buf.tex, index: 1)
                    enc.setTexture(buf.rawGaps, index: 2)
                    enc.setTexture(buf.gx, index: 3)
                case 3:
                    if band == 0 {
                        enc.setComputePipelineState(sitePipeline)
                        enc.setTexture(buf.tex, index: 0)
                        enc.setTexture(buf.gx, index: 1)
                        enc.setBuffer(buf.sites, offset: 0, index: 0)
                        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                        enc.setBuffer(buf.giants, offset: 0, index: 2)
                        enc.dispatchThreads(MTLSize(width: MapLayout.siteGrid, height: MapLayout.siteGrid, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 6, height: 6, depth: 1))
                    }
                    // Stamping follows cleanup, which would strip lintel and eave gaps, and precedes planting, so
                    // trees avoid it. It writes into the climate texture, free until the climate pass: the scratch must
                    // keep the raw columns for the max mips.
                    enc.setComputePipelineState(stampPipeline)
                    enc.setTexture(buf.tex, index: 0)
                    enc.setTexture(buf.climate, index: 1)
                    enc.setTexture(buf.structTex, index: 2)
                    enc.setBuffer(buf.sites, offset: 0, index: 0)
                    enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                    enc.setBytes(&row0, length: MemoryLayout<UInt32>.stride, index: 2)
                    enc.dispatchThreads(MTLSize(width: MapLayout.size, height: rows, depth: 1), threadsPerThreadgroup: tg)
                    j.next += 1
                    n -= 1
                    continue
                case 6:
                    var brow = UInt32(band * rows / 2)
                    enc.setComputePipelineState(bandTopPipeline)
                    enc.setTexture(buf.levels[0], index: 0)
                    enc.setTexture(buf.gx, index: 1)
                    enc.setTexture(buf.bandViews[0], index: 2)
                    enc.setTexture(buf.levels[1], index: 3)
                    enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                    enc.setBuffer(buf.giants, offset: 0, index: 1)
                    enc.setBuffer(buf.isles, offset: 0, index: 2)
                    enc.setBytes(&brow, length: MemoryLayout<UInt32>.stride, index: 3)
                    enc.dispatchThreads(MTLSize(width: MapLayout.size / 2, height: rows / 2, depth: 1), threadsPerThreadgroup: tg)
                    j.next += 1
                    n -= 1
                    continue
                case 4:
                    enc.setComputePipelineState(plantPipeline)
                    enc.setTexture(buf.climate, index: 0)
                    enc.setTexture(buf.levels[0], index: 1)
                    enc.setTexture(buf.gx, index: 2)
                    enc.setTexture(buf.structTex, index: 3)
                    enc.setBuffer(buf.giants, offset: 0, index: 2)
                default:
                    // Mip 1 of the finished columns, a band of its rows per climate piece (it builds the level-1 forest
                    // it bounds, too costly for one piece). The scratch still holds the raw columns that level 1 averages.
                    var rows1 = UInt32(band * rows / 2)
                    enc.setComputePipelineState(reduceBasePipeline)
                    enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                    enc.setBuffer(buf.giants, offset: 0, index: 1)
                    enc.setBytes(&rows1, length: MemoryLayout<UInt32>.stride, index: 2)
                    enc.setTexture(buf.levels[0], index: 0)
                    enc.setTexture(scratch, index: 1)
                    enc.setTexture(buf.levels[1], index: 2)
                    enc.dispatchThreads(MTLSize(width: MapLayout.size >> 1, height: rows / 2, depth: 1), threadsPerThreadgroup: tg)
                    enc.setComputePipelineState(climatePipeline)
                    enc.setTexture(buf.climate, index: 0)
                }
                enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.setBytes(&row0, length: MemoryLayout<UInt32>.stride, index: 1)
                enc.dispatchThreads(MTLSize(width: MapLayout.size, height: rows, depth: 1), threadsPerThreadgroup: tg)
            } else {
                enc.setComputePipelineState(cpuReducePipeline)
                enc.setTexture(buf.gx, index: 2)
                for (l, t) in cpuMips.enumerated() {
                    var fromColumns: Int32 = l == 0 ? 1 : 0
                    enc.setBytes(&fromColumns, length: 4, index: 0)
                    enc.setTexture(l == 0 ? buf.levels[0] : cpuMips[l - 1], index: 0)
                    enc.setTexture(t, index: 1)
                    enc.dispatchThreads(MTLSize(width: t.width, height: t.height, depth: 1), threadsPerThreadgroup: tg)
                }
                enc.setComputePipelineState(copyTopPipeline)
                enc.setTexture(buf.levels[0], index: 0)
                enc.setTexture(buf.hmax0, index: 1)
                enc.setTexture(buf.gx, index: 2)
                enc.dispatchThreads(MTLSize(width: MapLayout.size, height: MapLayout.size, depth: 1), threadsPerThreadgroup: tg)
                enc.setComputePipelineState(bandReducePipeline)
                for l in 1..<buf.bandViews.count {
                    enc.setTexture(buf.bandViews[l - 1], index: 0)
                    enc.setTexture(buf.bandViews[l], index: 1)
                    let s = max(1, MapLayout.size >> (l + 1))
                    enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
                }
                enc.setComputePipelineState(reducePipeline)
                for l in 2..<buf.levels.count {
                    enc.setTexture(buf.levels[l - 1], index: 0)
                    enc.setTexture(buf.levels[l], index: 1)
                    let s = max(1, MapLayout.size >> l)
                    enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
                }
                enc.setComputePipelineState(bandFlagPipeline)
                for l in 1..<buf.levels.count {
                    enc.setTexture(buf.levels[l], index: 0)
                    enc.setTexture(buf.bandViews[l - 1], index: 1)
                    let s = max(1, MapLayout.size >> l)
                    enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
                }
            }
            j.next += 1
            n -= 1
        }
        enc.endEncoding()
        let done = j.next == World.pieceCount
        job = done ? nil : j
        if timed && !done { cb.commit(); cb.waitUntilCompleted(); return (cb.gpuEndTime - cb.gpuStartTime) * 1000 }
        guard done else { cb.commit(); return 0 }

        let index = j.index, o = j.origin, t0 = j.t0
        if wait {
            cb.commit()
            cb.waitUntilCompleted()
            land(World.readBack(cpuMips.last!), into: buf, index: index, origin: o, started: t0, sync: true)
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        } else {
            // Strong captures: a weak reference would give World a side table, which slows every retain of it.
            cb.addCompletedHandler { cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self.pending = false }; return }
                let r = World.readBack(self.cpuMips.last!)
                DispatchQueue.main.async { self.land(r, into: buf, index: index, origin: o, started: t0, sync: false) }
            }
            cb.commit()
        }
        return 0
    }

    // Half the giant table round the map at `origin`, centred on its middle cell.
    func encodeGiants(_ enc: MTLComputeCommandEncoder, into buf: Buffer, origin: SIMD2<Int32>, half: Int) {
        let mid = SIMD2<Double>(Double(origin.x), Double(origin.y)) + Double(MapLayout.size / 2)
        var gi = SIMD4<Int32>(Int32(floor(mid.x / MapLayout.giantCell)) - Int32(MapLayout.giantWin / 2),
                              Int32(floor(mid.y / MapLayout.giantCell)) - Int32(MapLayout.giantWin / 2), Int32(half * MapLayout.giantWin / 2), Int32(bitPattern: seed))
        enc.setComputePipelineState(giantPipeline)
        enc.setBuffer(buf.giants, offset: 0, index: 0)
        enc.setBytes(&gi, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: MapLayout.giantWin, height: MapLayout.giantWin / 2, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 4, depth: 1))
    }

    // The cave table round the map at `origin`, centred on its middle cell.
    func encodeCaves(_ enc: MTLComputeCommandEncoder, into buf: Buffer, origin: SIMD2<Int32>) {
        if Knobs.current.noCaves { return }
        let mid = SIMD2<Double>(Double(origin.x), Double(origin.y)) + Double(MapLayout.size / 2)
        var ci = SIMD4<Int32>(Int32(floor(mid.x / MapLayout.caveCell)) - Int32(MapLayout.caveWin / 2),
                              Int32(floor(mid.y / MapLayout.caveCell)) - Int32(MapLayout.caveWin / 2), 0, Int32(bitPattern: seed))
        enc.setComputePipelineState(cavePipeline)
        enc.setBuffer(buf.giants, offset: MapLayout.caveOffset, index: 0)
        enc.setBytes(&ci, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: MapLayout.caveWin, height: MapLayout.caveWin, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }
}
