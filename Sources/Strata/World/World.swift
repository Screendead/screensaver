import Foundation
import Metal
import QuartzCore
import StrataCore
import simd

final class World {
    // Recentred once the camera drifts this far from the map centre. The LOD handoff (Renderer.makeFrame) relies
    // on the map reaching 1024 - 8 - (this + the drift while a regeneration is in flight) around the camera.
    static let recentre = 100.0
    // Background regeneration runs a band of rows of one pass per frame, so no frame carries a whole map's work.
    static let bandRows = 256

    // One generated map: the column texture, the max-mip chain of its tops (r16Float: the DDA's skips fetch 2 bytes,
    // not the column's 8), and the CPU copy the camera steers by.
    final class Buffer {
        let tex: MTLTexture
        // Up to two more gaps per column below the column texture's (packGaps in Shaders/World/Column.metal).
        let gx: MTLTexture
        let hmax: MTLTexture
        let hmax0: MTLTexture
        // levels[0] is the column texture, levels[m] for m >= 1 the max mips.
        let levels: [MTLTexture]
        let climate: MTLTexture
        // The climate texture's memory seen as rgba16Uint: the raw extra gaps between generation and cleanup, which run
        // before anything writes the climate (structures are stamped into it next).
        let rawGaps: MTLTexture
        let structTex: MTLTexture
        let sites: MTLBuffer
        // The giant table around this map (placeGiants).
        let giants: MTLBuffer
        // The island table around this map (placeIslands).
        let isles: MTLBuffer
        // Band mips (bandTop in Shaders/Maps/Mips.metal), one view per mip.
        let band: MTLTexture
        let bandViews: [MTLTexture]
        var origin = SIMD2<Int32>(Int32.min, Int32.min)
        var globalMax: Float = 255

        init?(device: MTLDevice) {
            let td = MTLTextureDescriptor.kernel(.rgba16Float, width: MapLayout.size, height: MapLayout.size)
            let xd = MTLTextureDescriptor.kernel(.r16Float, width: MapLayout.size, height: MapLayout.size, mipmapped: true, views: true)
            let cd = MTLTextureDescriptor.kernel(.rgba16Float, width: MapLayout.size, height: MapLayout.size, views: true)
            let sd = MTLTextureDescriptor.kernel(.r32Uint, width: MapLayout.size, height: MapLayout.size)
            let gd = MTLTextureDescriptor.kernel(.rgba16Uint, width: MapLayout.size, height: MapLayout.size)
            // From the column mips' level 1 up: mip m here is their level m + 1.
            let bd = MTLTextureDescriptor.kernel(.rg16Float, width: MapLayout.size / 2, height: MapLayout.size / 2, mipmapped: true, views: true)
            guard let bt = device.makeTexture(descriptor: bd),
                  let gb = device.makeBuffer(length: MapTables.giantTableBytes, options: .storageModeShared),
                  let ib = device.makeBuffer(length: Islands.slots * Islands.recordSize, options: .storageModeShared) else { return nil }
            MapTables.clearGiantsAndCaves(gb.contents())
            Islands.clear(ib.contents())
            giants = gb
            isles = ib
            band = bt
            bandViews = (0..<bt.mipmapLevelCount).map { bt.makeTextureView(pixelFormat: .rg16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)! }
            guard let t = device.makeTexture(descriptor: td), let x = device.makeTexture(descriptor: xd), let g = device.makeTexture(descriptor: gd),
                  let c = device.makeTexture(descriptor: cd), let st = device.makeTexture(descriptor: sd),
                  let sb = device.makeBuffer(length: MapLayout.siteGrid * MapLayout.siteGrid * MapLayout.siteStride, options: .storageModeShared) else { return nil }
            tex = t
            climate = c
            rawGaps = c.makeTextureView(pixelFormat: .rgba16Uint)!
            structTex = st
            sites = sb
            hmax = x
            gx = g
            hmax0 = x.makeTextureView(pixelFormat: .r16Float, textureType: .type2D, levels: 0..<1, slices: 0..<1)!
            levels = [t] + (1..<x.mipmapLevelCount).map {
                x.makeTextureView(pixelFormat: .r16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)!
            }
        }
    }

    let device: MTLDevice
    let seed: UInt32
    // Regeneration runs on its own queue so frames never wait behind it.
    let genQueue: MTLCommandQueue
    let genPipeline: MTLComputePipelineState
    let reducePipeline: MTLComputePipelineState
    let reduceBasePipeline: MTLComputePipelineState
    let cpuReducePipeline: MTLComputePipelineState
    let cleanupPipeline: MTLComputePipelineState
    let copyTopPipeline: MTLComputePipelineState
    let plantPipeline: MTLComputePipelineState
    let climatePipeline: MTLComputePipelineState
    let sitePipeline: MTLComputePipelineState
    let stampPipeline: MTLComputePipelineState
    let giantPipeline: MTLComputePipelineState
    let cavePipeline: MTLComputePipelineState
    let bandTopPipeline: MTLComputePipelineState
    let bandReducePipeline: MTLComputePipelineState
    let bandFlagPipeline: MTLComputePipelineState
    let islePipeline: MTLComputePipelineState
    let isleResolvePipeline: MTLComputePipelineState
    // The raw columns while a regeneration runs; allocated for each, as is the map it builds, so between regenerations only the
    // map shown holds memory.
    var scratch: MTLTexture?
    private let scratchDescriptor: MTLTextureDescriptor
    // Max mips 1 and 2 of the plain heights (r) and solid floors (g) for the CPU readback: the map's own mips from 1 up also bound the
    // level-1 columns, which the camera must not steer by.
    let cpuMips: [MTLTexture]
    var buffers: [Buffer?]
    var active = 0
    // The map shown, as the CPU reads it.
    let terrain = Terrain()
    var job: (index: Int, origin: SIMD2<Int32>, next: Int, t0: CFTimeInterval)?
    var pending = false
    var synchronous = false

    private var shown: Buffer { buffers[active]! }
    var heightTex: MTLTexture { shown.tex }
    var maxTex: MTLTexture { shown.hmax }
    var gapTex: MTLTexture { shown.gx }
    var climateTex: MTLTexture { shown.climate }
    var origin: SIMD2<Int32> { shown.origin }
    var globalMax: Float { shown.globalMax }
    var siteBuffer: MTLBuffer { shown.sites }
    var structTex: MTLTexture { shown.structTex }
    var bandTex: MTLTexture { shown.band }
    var giantBuffer: MTLBuffer { shown.giants }
    var isleBuffer: MTLBuffer { shown.isles }
    init?(gpu: GPU, seed: UInt32) {
        let device = gpu.device
        let pipeline = gpu.pipeline as (String) -> MTLComputePipelineState?
        let sd = MTLTextureDescriptor.kernel(.rgba16Float, width: MapLayout.size, height: MapLayout.size)
        guard let gen = pipeline("generateHeight"), let red = pipeline("reduceMax"), let redBase = pipeline("reduceMaxBase"), let redCpu = pipeline("reduceCpu"), let cpt = pipeline("copyTop"),
              let clean = pipeline("cleanupColumns"),
              let plant = pipeline("plantTrees"), let clim = pipeline("bakeClimate"), let place = pipeline("placeSites"), let stamp = pipeline("stampStructures"),
              let bTop = pipeline("bandTop"), let bRed = pipeline("bandReduce"), let bFlag = pipeline("bandFlag"), let pg = pipeline("placeGiants"), let pc = pipeline("placeCaves"),
              let pi = pipeline("placeIslands"), let ri = pipeline("resolveIslands"),
              let q = device.makeCommandQueue(), let a = Buffer(device: device) else { return nil }
        var mips: [MTLTexture] = []
        for l in 1...MapLayout.cpuLevel {
            let md = MTLTextureDescriptor.kernel(.rgba16Float, width: MapLayout.size >> l, height: MapLayout.size >> l)
            md.storageMode = l == MapLayout.cpuLevel ? .shared : .private
            guard let t = device.makeTexture(descriptor: md) else { return nil }
            mips.append(t)
        }
        cpuMips = mips
        self.device = device
        self.seed = seed
        genQueue = q
        genPipeline = gen
        reducePipeline = red
        reduceBasePipeline = redBase
        cpuReducePipeline = redCpu
        cleanupPipeline = clean
        copyTopPipeline = cpt
        plantPipeline = plant
        climatePipeline = clim
        sitePipeline = place
        stampPipeline = stamp
        giantPipeline = pg
        cavePipeline = pc
        bandTopPipeline = bTop
        bandReducePipeline = bRed
        bandFlagPipeline = bFlag
        islePipeline = pi
        isleResolvePipeline = ri
        scratchDescriptor = sd
        buffers = [a, nil]
    }

    // Recentres the map when the camera drifts. The swap happens only once the new map is complete;
    // a synchronous fallback covers startup and the case where the camera outruns generation.
    func ensureCentered(on center: SIMD2<Double>) {
        let half = Double(MapLayout.size / 2)
        let cur = shown
        let local = center - SIMD2(Double(cur.origin.x), Double(cur.origin.y))
        let drift = max(abs(local.x - half), abs(local.y - half))
        let o = World.mapOrigin(center)
        if cur.origin.x == Int32.min || drift > 340 || (synchronous && drift > World.recentre) {
            if job != nil { advance(pieces: Int.max, wait: true); return }
            if pending { return } // the in-flight job will land momentarily
            // Built beside the map shown, which frames in flight may still read; only the first is built in place.
            let target = cur.origin.x == Int32.min ? active : 1 - active
            guard begin(target), let next = buffers[target] else { return }
            if target != active {
                memcpy(next.giants.contents(), cur.giants.contents(), cur.giants.length)
                memcpy(next.isles.contents(), cur.isles.contents(), cur.isles.length)
            }
            job = (target, o, 0, CACurrentMediaTime())
            // VR_PIECES: one command buffer per piece, each timed (the regeneration budget is per piece).
            if Knobs.current.pieces {
                while let j = job {
                    let k = j.next
                    let t = advance(pieces: 1, wait: true, timed: true)
                    print(String(format: "regen piece %d (pass %d) gpu %.2f ms", k, World.piece(k).pass, t))
                }
                return
            }
            advance(pieces: Int.max, wait: true)
        } else if job != nil {
            advance(pieces: 1, wait: false)
        } else if drift > World.recentre && !pending, begin(1 - active), let next = buffers[1 - active] {
            pending = true
            // The new map's giant and island tables start from this one's, so placing fills only the cells that changed.
            memcpy(next.giants.contents(), cur.giants.contents(), cur.giants.length)
            memcpy(next.isles.contents(), cur.isles.contents(), cur.isles.length)
            job = (1 - active, o, 0, CACurrentMediaTime())
            advance(pieces: 1, wait: false)
        }
    }

    // Allocates what a regeneration into map i needs.
    private func begin(_ i: Int) -> Bool {
        if buffers[i] == nil { buffers[i] = Buffer(device: device) }
        if scratch == nil { scratch = device.makeTexture(descriptor: scratchDescriptor) }
        return buffers[i] != nil && scratch != nil
    }

    // Before the first map: the giant table (and `giants`) the map centred on `center` would have, so the start can be
    // chosen by it without building a map there. placeGiants depends on the cells alone; the map keeps the slots it fills.
    func placeGiants(around center: SIMD2<Double>) {
        guard shown.origin.x == Int32.min, let cb = genQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        let buf = shown
        for half in 0..<2 { encodeGiants(enc, into: buf, origin: World.mapOrigin(center), half: half) }
        // The coarse rings may be built from this table before the first map: they carve its caves.
        encodeCaves(enc, into: buf, origin: World.mapOrigin(center))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        terrain.giants = MapTables.giants(in: buf.giants.contents())
    }

    static func mapOrigin(_ center: SIMD2<Double>) -> SIMD2<Int32> {
        let half = Double(MapLayout.size / 2)
        return SIMD2<Int32>(Int32(floor((center.x - half) / 64)) * 64, Int32(floor((center.y - half) / 64)) * 64)
    }
}
