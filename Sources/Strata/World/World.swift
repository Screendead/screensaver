import Foundation
import Metal
import QuartzCore
import simd

final class World {
    static let size = 2048
    static let cpuLevel = 2
    // Must match SITE_GRID, MAX_LAMPS and MAX_PARTS (the Site struct) in Shaders/Saver.metal.
    static let siteGrid = 18, siteCell = 128, maxLamps = 8, maxParts = 32
    static let siteStride = 32 + 16 * maxLamps + 16 * maxParts
    // Recentred once the camera drifts this far from the map centre. The LOD handoff (Renderer.makeFrame) relies
    // on the map reaching 1024 - 8 - (this + the drift while a regeneration is in flight) around the camera.
    static let recentre = 100.0
    // Background regeneration runs a band of rows of one pass per frame, so no frame carries a whole map's work.
    static let bandRows = 256
    // Must match GIANT_WIN and GIANT_CELL in Shaders/Saver.metal.
    static let giantWin = 24, giantCell = 1536.0
    // Must match CAVE_WIN and CAVE_CELL in Shaders/Saver.metal; the cave table follows the giant table in its buffer.
    static let caveWin = 96, caveCell = 256.0, caveReach = 64.0
    static let caveOffset = giantWin * giantWin * MemoryLayout<GiantRec>.stride

    // One generated map: the column texture, the max-mip chain of its tops (r16Float: the DDA's skips fetch 2 bytes,
    // not the column's 8), and the CPU copy the camera steers by.
    final class Buffer {
        let tex: MTLTexture
        // Up to two more gaps per column below the column texture's (Shaders/Saver.metal packGaps).
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
        // Band mips (bandTop in Shaders/Saver.metal), one view per mip.
        let band: MTLTexture
        let bandViews: [MTLTexture]
        var origin = SIMD2<Int32>(Int32.min, Int32.min)
        var globalMax: Float = 255
        var siteList: [SiteInfo] = []
        // siteList's lamps by Lamp.index.
        var lamps: [Int: SiteInfo.Lamp] = [:]
        var giantList: [GiantInfo] = []
        var isleList: [IslandInfo] = []
        // The caves of the cave table (placeCaves), world xz of each mouth.
        var caveList: [CaveRec] = []
        // Whether the jungle's roof (Shaders/Saver.metal jungleCanopy) may stand on this map.
        var jungle = false
        // The jungle's extent on this map: whether the province reaches each point of a 64-voxel grid over it.
        var jungleGrid: [Bool] = []

        init?(device: MTLDevice) {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
            td.usage = [.shaderRead, .shaderWrite]
            td.storageMode = .private
            let xd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: World.size, height: World.size, mipmapped: true)
            xd.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            xd.storageMode = .private
            let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
            cd.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            cd.storageMode = .private
            let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: World.size, height: World.size, mipmapped: false)
            sd.usage = [.shaderRead, .shaderWrite]
            sd.storageMode = .private
            let gd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Uint, width: World.size, height: World.size, mipmapped: false)
            gd.usage = [.shaderRead, .shaderWrite]
            gd.storageMode = .private
            // From the column mips' level 1 up: mip m here is their level m + 1.
            let bd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: World.size / 2, height: World.size / 2, mipmapped: true)
            bd.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            bd.storageMode = .private
            guard let bt = device.makeTexture(descriptor: bd),
                  let gb = device.makeBuffer(length: World.caveOffset + World.caveWin * World.caveWin * MemoryLayout<CaveRec>.stride, options: .storageModeShared),
                  let ib = device.makeBuffer(length: Islands.slots * Islands.recordSize, options: .storageModeShared) else { return nil }
            // No slot holds a cell yet (placeGiants and placeIslands fill a slot only when its cell changes).
            let none = GiantRec(cell: SIMD2(Int32.min, Int32.min))
            let slots = gb.contents().bindMemory(to: GiantRec.self, capacity: World.giantWin * World.giantWin)
            for i in 0..<(World.giantWin * World.giantWin) { slots[i] = none }
            let islots = ib.contents().bindMemory(to: IsleRec.self, capacity: Islands.slots)
            for i in 0..<Islands.slots { islots[i] = IsleRec() }
            let caveSlots = (gb.contents() + World.caveOffset).bindMemory(to: CaveRec.self, capacity: World.caveWin * World.caveWin)
            for i in 0..<(World.caveWin * World.caveWin) { caveSlots[i] = CaveRec(cell: SIMD2(Int32.min, Int32.min)) }
            giants = gb
            isles = ib
            band = bt
            bandViews = (0..<bt.mipmapLevelCount).map { bt.makeTextureView(pixelFormat: .rg16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)! }
            guard let t = device.makeTexture(descriptor: td), let x = device.makeTexture(descriptor: xd), let g = device.makeTexture(descriptor: gd),
                  let c = device.makeTexture(descriptor: cd), let st = device.makeTexture(descriptor: sd),
                  let sb = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.siteStride, options: .storageModeShared) else { return nil }
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
    // The active map's CPU copy, held here so the camera's many lookups read it directly.
    var cpuOrigin = SIMD2<Int32>(Int32.min, Int32.min)
    var cpuHeights: [Float] = []
    var cpuFloors: [Float] = []
    // Per CPU cell, the highest ground islands aside and the lowest island underside (4096: none).
    var cpuGround: [Float] = []
    var cpuUnder: [Float] = []
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
    var caves: [CaveRec] { shown.caveList }
    var sites: [SiteInfo] { shown.siteList }
    var lamps: [Int: SiteInfo.Lamp] { shown.lamps }
    var giants: [GiantInfo] { shown.giantList }
    var isles: [IslandInfo] { shown.isleList }
    // Whether a flagged cell of the active map's band mips (under a giant's crown or the jungle's roof) may lie within r
    // of world xz p.
    func bandsNear(_ p: SIMD2<Double>, radius r: Double) -> Bool {
        let b = shown
        if b.giantList.contains(where: { simd_length($0.pos - p) < r + GiantInfo.reach }) { return true }
        guard b.jungle else { return false }
        let n = World.size / 64 + 1
        for z in 0..<n {
            for x in 0..<n where b.jungleGrid[z * n + x] {
                let q = SIMD2(Double(Int(b.origin.x) + x * 64), Double(Int(b.origin.y) + z * 64))
                if simd_length(q - p) < r + 64 { return true }
            }
        }
        return false
    }

    init?(gpu: GPU, seed: UInt32) {
        let device = gpu.device
        let pipeline = gpu.pipeline as (String) -> MTLComputePipelineState?
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
        sd.usage = [.shaderRead, .shaderWrite]
        sd.storageMode = .private
        guard let gen = pipeline("generateHeight"), let red = pipeline("reduceMax"), let redBase = pipeline("reduceMaxBase"), let redCpu = pipeline("reduceCpu"), let cpt = pipeline("copyTop"),
              let clean = pipeline("cleanupColumns"),
              let plant = pipeline("plantTrees"), let clim = pipeline("bakeClimate"), let place = pipeline("placeSites"), let stamp = pipeline("stampStructures"),
              let bTop = pipeline("bandTop"), let bRed = pipeline("bandReduce"), let bFlag = pipeline("bandFlag"), let pg = pipeline("placeGiants"), let pc = pipeline("placeCaves"),
              let pi = pipeline("placeIslands"), let ri = pipeline("resolveIslands"),
              let q = device.makeCommandQueue(), let a = Buffer(device: device) else { return nil }
        var mips: [MTLTexture] = []
        for l in 1...World.cpuLevel {
            let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size >> l, height: World.size >> l, mipmapped: false)
            md.usage = [.shaderRead, .shaderWrite]
            md.storageMode = l == World.cpuLevel ? .shared : .private
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
        let half = Double(World.size / 2)
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
            if World.timePieces {
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
        buf.giantList = World.readGiants(buf.giants)
    }

    static func mapOrigin(_ center: SIMD2<Double>) -> SIMD2<Int32> {
        let half = Double(World.size / 2)
        return SIMD2<Int32>(Int32(floor((center.x - half) / 64)) * 64, Int32(floor((center.y - half) / 64)) * 64)
    }

    // Conservative (4x4 max) height in world coords.
    func height(x: Double, z: Double) -> Double? {
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(cpuOrigin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(cpuOrigin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !cpuHeights.isEmpty else { return nil }
        return Double(cpuHeights[cz * s + cx])
    }

    // The 4x4 cell under world (x, z) as islands shape it: its highest top, its highest ground islands aside, and the
    // lowest island underside over it (nil: no island).
    func column(x: Double, z: Double) -> (top: Double, ground: Double, under: Double?)? {
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(cpuOrigin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(cpuOrigin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !cpuUnder.isEmpty else { return nil }
        let i = cz * s + cx
        let u = Double(cpuUnder[i])
        return (Double(cpuHeights[i]), Double(cpuGround[i]), u < 4000 ? u : nil)
    }

    // Lower bound in world coords: every column of the 4x4 cell is solid from the ground up to at least this.
    func solidFloor(x: Double, z: Double) -> Double? {
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(cpuOrigin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(cpuOrigin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !cpuFloors.isEmpty else { return nil }
        return Double(cpuFloors[cz * s + cx])
    }
}
