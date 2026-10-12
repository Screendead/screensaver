import AppKit
import MetalKit

// Must match `Frame` in Shaders/Saver.metal.
struct Frame {
    var camPos = SIMD4<Float>()
    var camFwd = SIMD4<Float>()
    var camRight = SIMD4<Float>()
    var camUp = SIMD4<Float>()
    var sunDir = SIMD4<Float>()
    var resolution = SIMD4<Float>()
    var mapInfo = SIMD4<Int32>()
    var misc = SIMD4<Float>()
    var misc2 = SIMD4<Float>()
    var prevPos = SIMD4<Float>()
    var prevFwd = SIMD4<Float>()
    var prevRight = SIMD4<Float>()
    var prevUp = SIMD4<Float>()
    var jitter = SIMD4<Float>()
    var cloud = SIMD4<Float>()
    var lights = SIMD4<Float>()
    var refs = SIMD4<Int32>()
    var birds = SIMD4<Float>()
    // lod0...lod4 are `lod[5]` in the shader.
    var lod0 = SIMD4<Float>()
    var lod1 = SIMD4<Float>()
    var lod2 = SIMD4<Float>()
    var lod3 = SIMD4<Float>()
    var lod4 = SIMD4<Float>()
    var fade = SIMD4<Float>()
    var shadow = SIMD4<Float>()
    var slowCloud = SIMD4<Float>()
}

// The shader library and its pipeline states, shared by the renderers of one activation (a screen each).
final class GPU {
    let device: MTLDevice
    let library: MTLLibrary
    private var compute: [String: MTLComputePipelineState] = [:]
    private var render: [String: MTLRenderPipelineState] = [:]

    init?(device: MTLDevice) {
        self.device = device
        if let url = Bundle.main.url(forResource: "Saver", withExtension: "metallib"), let lib = try? device.makeLibrary(URL: url) {
            library = lib
            return
        }
        // Loose build: metallib next to the binary.
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        guard let lib = try? device.makeLibrary(URL: exe.appendingPathComponent("Saver.metallib")) else { return nil }
        library = lib
    }

    func pipeline(_ name: String) -> MTLComputePipelineState? {
        if let p = compute[name] { return p }
        let p = library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        compute[name] = p
        return p
    }

    func pipeline(_ key: String, _ descriptor: () -> MTLRenderPipelineDescriptor) -> MTLRenderPipelineState? {
        if let p = render[key] { return p }
        let p = try? device.makeRenderPipelineState(descriptor: descriptor())
        render[key] = p
        return p
    }

    func fullscreen(_ fragment: MTLFunction?, _ format: MTLPixelFormat) -> MTLRenderPipelineDescriptor {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: "fullscreenVertex")
        d.fragmentFunction = fragment
        d.colorAttachments[0].pixelFormat = format
        return d
    }

    func scene(towers: Bool, bands: Bool) -> MTLRenderPipelineState? {
        pipeline(GPU.sceneKey(towers, bands)) { sceneDescriptor(towers, bands) }
    }

    private static func sceneKey(_ towers: Bool, _ bands: Bool) -> String { "scene \(towers) \(bands)" }

    private func sceneDescriptor(_ towers: Bool, _ bands: Bool) -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var b = bands, t = towers
        fc.setConstantValue(&b, type: .bool, index: 0)
        fc.setConstantValue(&t, type: .bool, index: 1)
        return fullscreen(try? library.makeFunction(name: "sceneFragment", constantValues: fc), .rgba16Float)
    }

    // The giants' hanging strands, drawn over the scene (vineFragment): the towers and the band test compiled out.
    func vine() -> MTLRenderPipelineState? {
        pipeline("vine") { vineDescriptor() }
    }

    private func vineDescriptor() -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var no = false
        fc.setConstantValue(&no, type: .bool, index: 0)
        fc.setConstantValue(&no, type: .bool, index: 1)
        return fullscreen(try? library.makeFunction(name: "vineFragment", constantValues: fc), .rgba16Float)
    }

    func taa() -> MTLRenderPipelineState? {
        pipeline("taa") { fullscreen(library.makeFunction(name: "taaFragment"), .rgba16Float) }
    }

    func bio() -> MTLRenderPipelineState? {
        pipeline("bio") { bioDescriptor() }
    }

    private func bioDescriptor() -> MTLRenderPipelineDescriptor {
        let d = fullscreen(library.makeFunction(name: "bioFragment"), .rgba16Float)
        d.vertexFunction = library.makeFunction(name: "bioVertex")
        return d
    }

    func post(_ format: MTLPixelFormat, dither: Bool) -> MTLRenderPipelineState? {
        pipeline(GPU.postKey(format, dither)) { postDescriptor(format, dither) }
    }

    static func postKey(_ format: MTLPixelFormat, _ dither: Bool) -> String { "post \(format.rawValue) \(dither)" }

    private func postDescriptor(_ format: MTLPixelFormat, _ dither: Bool) -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var d = dither
        fc.setConstantValue(&d, type: .bool, index: 2)
        return fullscreen(try? library.makeFunction(name: "postFragment", constantValues: fc), format)
    }

    // Creates every kernel's pipeline and the render pipelines for `output` at once. On a cold shader cache (any shader
    // edit) creating them one by one takes ~17 s; the compiler service builds concurrent requests in parallel.
    func prewarm(output: MTLPixelFormat) {
        let group = DispatchGroup(), lock = NSLock()
        var kernels: [String: MTLComputePipelineState] = [:], renders: [String: MTLRenderPipelineState] = [:]
        let functions = library.functionNames.filter { compute[$0] == nil }.compactMap { library.makeFunction(name: $0) }
        var descriptors = ["taa": fullscreen(library.makeFunction(name: "taaFragment"), .rgba16Float),
                           "bio": bioDescriptor(),
                           GPU.postKey(output, true): postDescriptor(output, true)]
        for t in [false, true] { for b in [false, true] { descriptors[GPU.sceneKey(t, b)] = sceneDescriptor(t, b) } }
        descriptors["vine"] = vineDescriptor()
        for f in functions where f.functionType == .kernel {
            group.enter()
            device.makeComputePipelineState(function: f) { p, _ in
                lock.lock(); kernels[f.name] = p; lock.unlock()
                group.leave()
            }
        }
        for (key, d) in descriptors where render[key] == nil {
            group.enter()
            device.makeRenderPipelineState(descriptor: d) { p, _ in
                lock.lock(); renders[key] = p; lock.unlock()
                group.leave()
            }
        }
        group.wait()
        compute.merge(kernels) { a, _ in a }
        render.merge(renders) { a, _ in a }
    }
}

// Must match ShadowBake in Shaders/Saver.metal.
struct ShadowBake {
    var key: SIMD4<Float>
    var cam: SIMD4<Float>
    var rows: SIMD4<Int32>
}

// Day cycle with a time-warp that lingers in golden and blue hour.
struct SkyClock {
    var phase: Double = -0.035 // just before sunrise: open on blue hour
    var day = 0
    static let dayLength: Double = 12 * 60

    // Peak day is held to a long-afternoon sun (flat high noon judged weakest); night keeps its full depth
    // (a shallower night judged worse). Both halves meet at the horizon.
    static func elevation(_ phase: Double) -> Double {
        let w = sin(2 * .pi * phase)
        return w * (w > 0 ? 0.42 : 0.56) + 0.07
    }

    var sunDirection: SIMD3<Double> {
        let el = asin(SkyClock.elevation(phase))
        let az = 2 * .pi * phase * 0.5 + 0.9
        return SIMD3(cos(el) * sin(az), sin(el), cos(el) * cos(az))
    }

    // Steps at mid-morning rather than at the dawn wrap, so a night keeps its beacons and rites until daylight.
    var nightIndex: Int {
        let n = day + Int(floor(phase))
        return phase - floor(phase) < 0.25 ? n - 1 : n
    }

    mutating func advance(_ dt: Double) {
        let y = SkyClock.elevation(phase)
        let s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
        // Linger in golden and blue hour, hurry through midday and deep night.
        let warp = y < -0.25 ? 5.0 : 0.5 + 1.5 * s * s * (3 - 2 * s)
        phase += dt / SkyClock.dayLength * warp * 1.05
        if phase > 1 { phase -= 1; day += 1 }
    }
}

// One giant-table entry (Shaders/Saver.metal GiantRec): g <= 0 marks a cell without a giant.
struct GiantRec {
    var cell = SIMD2<Int32>()
    var c = SIMD2<Float>()
    var g: Float = 0
    var yT: Float = 0
    var h: UInt32 = 0
    var kind: UInt32 = 0
}

// A giant tree as the camera and the birds see it, in world coordinates: a cylinder of `reach` round its axis, its crown
// top at `top`.
struct GiantInfo {
    static let reach = 210.0
    var pos: SIMD2<Double>
    var ground: Double
    var top: Double
    // 0 banyan, 1 kapok, 2 weeping (placeGiants).
    var kind = 0
    // The record's trunk height and hash (Vines bakes its strands from them).
    var yT = 0.0
    var h: UInt32 = 0
}

// A structure site as the camera and the smoke and halo lists see it, in world coordinates.
struct SiteInfo {
    enum Kind: Int { case none = 0, henge, cairn, ruins, hamlet, jetty, camp }
    // smoke: where its plume starts (rise > 0 only): a window lamp stands at its door, its chimney a few columns off.
    struct Lamp { var pos: SIMD3<Double>; var kind: Int; var rise: Double; var index: Int; var smoke: SIMD3<Double> }
    var index: Int
    var id: UInt32
    var kind: Kind
    var pos: SIMD2<Double>
    var radius: Double
    var lamps: [Lamp]
}

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

    // One generated map: the column texture, the max-mip chain of its tops (r16Float: the DDA's skips fetch 2 bytes,
    // not the column's 8), and the CPU copy the camera steers by.
    private final class Buffer {
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
        // Band mips (bandTop in Shaders/Saver.metal), one view per mip.
        let band: MTLTexture
        let bandViews: [MTLTexture]
        var origin = SIMD2<Int32>(Int32.min, Int32.min)
        var globalMax: Float = 255
        var siteList: [SiteInfo] = []
        // siteList's lamps by Lamp.index.
        var lamps: [Int: SiteInfo.Lamp] = [:]
        var giantList: [GiantInfo] = []
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
                  let gb = device.makeBuffer(length: World.giantWin * World.giantWin * MemoryLayout<GiantRec>.stride, options: .storageModeShared) else { return nil }
            // No slot holds a cell yet (placeGiants fills a slot only when its cell changes).
            let none = GiantRec(cell: SIMD2(Int32.min, Int32.min))
            let slots = gb.contents().bindMemory(to: GiantRec.self, capacity: World.giantWin * World.giantWin)
            for i in 0..<(World.giantWin * World.giantWin) { slots[i] = none }
            giants = gb
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
    private let genPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let reduceBasePipeline: MTLComputePipelineState
    private let cpuReducePipeline: MTLComputePipelineState
    private let cleanupPipeline: MTLComputePipelineState
    private let copyTopPipeline: MTLComputePipelineState
    private let plantPipeline: MTLComputePipelineState
    private let climatePipeline: MTLComputePipelineState
    private let sitePipeline: MTLComputePipelineState
    private let stampPipeline: MTLComputePipelineState
    private let giantPipeline: MTLComputePipelineState
    private let bandTopPipeline: MTLComputePipelineState
    private let bandReducePipeline: MTLComputePipelineState
    private let bandFlagPipeline: MTLComputePipelineState
    // The raw columns while a regeneration runs; allocated for each, as is the map it builds, so between regenerations only the
    // map shown holds memory.
    private var scratch: MTLTexture?
    private let scratchDescriptor: MTLTextureDescriptor
    // Max mips 1 and 2 of the plain heights (r) and solid floors (g) for the CPU readback: the map's own mips from 1 up also bound the
    // level-1 columns, which the camera must not steer by.
    private let cpuMips: [MTLTexture]
    private var buffers: [Buffer?]
    private var active = 0
    // The active map's CPU copy, held here so the camera's many lookups read it directly.
    private var cpuOrigin = SIMD2<Int32>(Int32.min, Int32.min)
    private var cpuHeights: [Float] = []
    private var cpuFloors: [Float] = []
    private var job: (index: Int, origin: SIMD2<Int32>, next: Int, t0: CFTimeInterval)?
    private var pending = false
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
    var sites: [SiteInfo] { shown.siteList }
    var lamps: [Int: SiteInfo.Lamp] { shown.lamps }
    var giants: [GiantInfo] { shown.giantList }
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
              let bTop = pipeline("bandTop"), let bRed = pipeline("bandReduce"), let bFlag = pipeline("bandFlag"), let pg = pipeline("placeGiants"),
              let q = device.makeCommandQueue(), let a = Buffer(device: device) else { return nil }
        var mips: [MTLTexture] = []
        for l in 1...World.cpuLevel {
            let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: World.size >> l, height: World.size >> l, mipmapped: false)
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
        bandTopPipeline = bTop
        bandReducePipeline = bRed
        bandFlagPipeline = bFlag
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
            if target != active { memcpy(next.giants.contents(), cur.giants.contents(), cur.giants.length) }
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
            // The new map's giant table starts from this one's, so placeGiants fills only the cells that changed.
            memcpy(next.giants.contents(), cur.giants.contents(), cur.giants.length)
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

    private static let passes = 5
    // Bands per pass: planting (giants, the jungle) runs in half bands, the costliest columns of any pass.
    private static let passBands = [size / bandRows, size / bandRows, size / bandRows, 2 * size / bandRows, size / bandRows]
    private static var pieceCount: Int { passBands.reduce(0, +) + 1 }
    // Piece k's pass, band and rows per band (the max mips' piece is pass `passes`).
    static func piece(_ k: Int) -> (pass: Int, band: Int, rows: Int) {
        var k = k
        for (p, n) in passBands.enumerated() {
            if k < n { return (p, k, size / n) }
            k -= n
        }
        return (passes, 0, 0)
    }

    // Encodes the next `pieces` pieces of the job: each pass (heights, cleanup, structures, trees, climate) a band of
    // rows at a time, finishing every band of a pass before the next pass reads its neighbours, then the max mips.
    // Sites are placed once, ahead of the first structure band: they read the cleaned map, and every band stamps them.
    static var timePieces: Bool { Knobs.current.pieces }

    @discardableResult
    private func advance(pieces: Int, wait: Bool, timed: Bool = false) -> Double {
        guard var j = job, let buf = buffers[j.index], let scratch, let cb = genQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return 0 }
        var info = SIMD4<Int32>(j.origin.x, j.origin.y, Int32(World.size), Int32(bitPattern: seed))
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        var n = pieces
        while n > 0 && j.next < World.pieceCount {
            let (pass, band, rows) = World.piece(j.next)
            if pass < World.passes {
                var row0 = UInt32(band * rows)
                switch pass {
                case 0:
                    enc.setComputePipelineState(genPipeline)
                    enc.setTexture(scratch, index: 0)
                    enc.setTexture(buf.rawGaps, index: 1)
                case 1:
                    // The giant table, with the cheap cleanup pass (half in each of its first two pieces): sites and
                    // trees read it.
                    if band < 2 { encodeGiants(enc, into: buf, origin: j.origin, half: band) }
                    enc.setComputePipelineState(cleanupPipeline)
                    enc.setTexture(scratch, index: 0)
                    enc.setTexture(buf.tex, index: 1)
                    enc.setTexture(buf.rawGaps, index: 2)
                    enc.setTexture(buf.gx, index: 3)
                case 2:
                    if band == 0 {
                        enc.setComputePipelineState(sitePipeline)
                        enc.setTexture(buf.tex, index: 0)
                        enc.setTexture(buf.gx, index: 1)
                        enc.setBuffer(buf.sites, offset: 0, index: 0)
                        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                        enc.setBuffer(buf.giants, offset: 0, index: 2)
                        enc.dispatchThreads(MTLSize(width: World.siteGrid, height: World.siteGrid, depth: 1),
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
                    enc.dispatchThreads(MTLSize(width: World.size, height: rows, depth: 1), threadsPerThreadgroup: tg)
                    j.next += 1
                    n -= 1
                    continue
                case 3:
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
                    enc.dispatchThreads(MTLSize(width: World.size >> 1, height: rows / 2, depth: 1), threadsPerThreadgroup: tg)
                    enc.setComputePipelineState(climatePipeline)
                    enc.setTexture(buf.climate, index: 0)
                }
                enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.setBytes(&row0, length: MemoryLayout<UInt32>.stride, index: 1)
                enc.dispatchThreads(MTLSize(width: World.size, height: rows, depth: 1), threadsPerThreadgroup: tg)
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
                enc.dispatchThreads(MTLSize(width: World.size, height: World.size, depth: 1), threadsPerThreadgroup: tg)
                                enc.setComputePipelineState(bandTopPipeline)
                enc.setTexture(buf.levels[0], index: 0)
                enc.setTexture(buf.gx, index: 1)
                enc.setTexture(buf.bandViews[0], index: 2)
                enc.dispatchThreads(MTLSize(width: World.size / 2, height: World.size / 2, depth: 1), threadsPerThreadgroup: tg)
                enc.setComputePipelineState(bandReducePipeline)
                for l in 1..<buf.bandViews.count {
                    enc.setTexture(buf.bandViews[l - 1], index: 0)
                    enc.setTexture(buf.bandViews[l], index: 1)
                    let s = max(1, World.size >> (l + 1))
                    enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
                }
                enc.setComputePipelineState(reducePipeline)
                for l in 2..<buf.levels.count {
                    enc.setTexture(buf.levels[l - 1], index: 0)
                    enc.setTexture(buf.levels[l], index: 1)
                    let s = max(1, World.size >> l)
                    enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
                }
                enc.setComputePipelineState(bandFlagPipeline)
                for l in 1..<buf.levels.count {
                    enc.setTexture(buf.levels[l], index: 0)
                    enc.setTexture(buf.bandViews[l - 1], index: 1)
                    let s = max(1, World.size >> l)
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
        // Strong captures: a weak reference would give World a side table, which slows every retain of it (the camera
        // and the birds look heights up thousands of times a frame).
        let finish = { (heights: [Float], floors: [Float], top: Float) in
            buf.origin = o
            buf.globalMax = top + 1
            buf.siteList = World.readSites(buf.sites)
            buf.lamps = Dictionary(buf.siteList.flatMap { $0.lamps.map { ($0.index, $0) } }, uniquingKeysWith: { $1 })
            buf.giantList = World.readGiants(buf.giants)
            buf.jungleGrid = stride(from: 0, through: World.size, by: 64).flatMap { z in
                stride(from: 0, through: World.size, by: 64).map { x in
                    Provinces.at(Double(Int(o.x) + x), Double(Int(o.y) + z), seed: self.seed).0 > 0.1
                }
            }
            buf.jungle = buf.jungleGrid.contains(true)
            if World.logSites {
                for g in buf.giantList { print(String(format: "giant at (%.0f, %.0f) ground %.0f top %.0f kind %d", g.pos.x, g.pos.y, g.ground, g.top, g.kind)) }
            }
            if World.logSites {
                var counts = [Int](repeating: 0, count: 7)
                for site in buf.siteList { counts[site.kind.rawValue] += 1 }
                if Knobs.current.env["VR_SITES"] == "3" {
                    // Parts as kind (2 house, 10 round) and dwelling style.
                    let raw = buf.sites.contents()
                    for site in buf.siteList {
                        let base = raw + site.index * World.siteStride
                        let np = Int((base + 16).load(as: SIMD4<Float>.self).z)
                        var desc = ""
                        for k in 0..<np {
                            let v = (base + 32 + 16 * World.maxLamps + 16 * k).load(as: SIMD4<Int32>.self)
                            let kind = (v.y >> 16) & 0xff, style = (v.w >> 28) & 15
                            desc += kind == 2 || kind == 10 ? " p\(kind)s\(style)" : kind == 11 ? " p11t\(v.z >> 16)" : " p\(kind)"
                        }
                        print("  site \(site.kind) \(Int(site.pos.x)),\(Int(site.pos.y)) r\(Int(site.radius)) lamps \(site.lamps.count)\(desc)")
                    }
                }
                print("sites at (\(o.x),\(o.y)): henge \(counts[1]) cairn \(counts[2]) ruins \(counts[3]) hamlet \(counts[4]) jetty \(counts[5]) camp \(counts[6]) total \(buf.siteList.count)")
            }
            // Frames in flight hold the map they read until they complete.
            if index != self.active { self.buffers[self.active] = nil }
            self.active = index
            self.scratch = nil
            self.cpuOrigin = o
            self.cpuHeights = heights
            self.cpuFloors = floors
            self.pending = false
            if Renderer.logTiming { print(String(format: "world regen %.1f ms (%@)", (CACurrentMediaTime() - t0) * 1000, wait ? "sync" : "async")) }
        }
        if wait {
            cb.commit()
            cb.waitUntilCompleted()
            let r = World.readBack(cpuMips.last!)
            finish(r.0, r.1, r.2)
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        } else {
            cb.addCompletedHandler { cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self.pending = false }; return }
                let r = World.readBack(self.cpuMips.last!)
                DispatchQueue.main.async { finish(r.0, r.1, r.2) }
            }
            cb.commit()
        }
        return 0
    }

    // Half the giant table round the map at `origin`, centred on its middle cell.
    private func encodeGiants(_ enc: MTLComputeCommandEncoder, into buf: Buffer, origin: SIMD2<Int32>, half: Int) {
        let mid = SIMD2<Double>(Double(origin.x), Double(origin.y)) + Double(World.size / 2)
        var gi = SIMD4<Int32>(Int32(floor(mid.x / World.giantCell)) - Int32(World.giantWin / 2),
                              Int32(floor(mid.y / World.giantCell)) - Int32(World.giantWin / 2), Int32(half * World.giantWin / 2), Int32(bitPattern: seed))
        enc.setComputePipelineState(giantPipeline)
        enc.setBuffer(buf.giants, offset: 0, index: 0)
        enc.setBytes(&gi, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: World.giantWin, height: World.giantWin / 2, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 8, height: 4, depth: 1))
    }

    // Before the first map: the giant table (and `giants`) the map centred on `center` would have, so the start can be
    // chosen by it without building a map there. placeGiants depends on the cells alone; the map keeps the slots it fills.
    func placeGiants(around center: SIMD2<Double>) {
        guard shown.origin.x == Int32.min, let cb = genQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        let buf = shown
        for half in 0..<2 { encodeGiants(enc, into: buf, origin: World.mapOrigin(center), half: half) }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        buf.giantList = World.readGiants(buf.giants)
    }

    static func mapOrigin(_ center: SIMD2<Double>) -> SIMD2<Int32> {
        let half = Double(World.size / 2)
        return SIMD2<Int32>(Int32(floor((center.x - half) / 64)) * 64, Int32(floor((center.y - half) / 64)) * 64)
    }

    static var logSites: Bool { Knobs.current.sites }

    private static func readSites(_ buf: MTLBuffer) -> [SiteInfo] {
        let raw = buf.contents()
        var list: [SiteInfo] = []
        for i in 0..<(World.siteGrid * World.siteGrid) {
            let base = raw + i * World.siteStride
            let head = base.load(as: SIMD4<Int32>.self)
            guard let kind = SiteInfo.Kind(rawValue: Int(head.z)), kind != .none else { continue }
            let info = (base + 16).load(as: SIMD4<Float>.self)
            let c = SIMD2(Double(head.x), Double(head.y))
            var lamps: [SiteInfo.Lamp] = []
            for l in 0..<Int(info.w) {
                let v = (base + 32 + 16 * l).load(as: SIMD4<Float>.self)
                let w = Int(v.w), k = w % 16, rise = Double((w / 16) % 64), code = w / 1024
                let pos = SIMD3(c.x + Double(v.x), Double(v.y), c.y + Double(v.z))
                let off = code > 0 ? SIMD3(Double(code % 32 - 16), rise, Double(code / 32 - 16)) : SIMD3(0, rise, 0)
                lamps.append(.init(pos: pos, kind: k, rise: rise, index: i * World.maxLamps + l, smoke: pos + off))
            }
            list.append(SiteInfo(index: i, id: UInt32(bitPattern: head.w), kind: kind, pos: c + 0.5, radius: Double(info.x), lamps: lamps))
        }
        return list
    }

    private static func readGiants(_ buf: MTLBuffer) -> [GiantInfo] {
        let p = buf.contents().bindMemory(to: GiantRec.self, capacity: World.giantWin * World.giantWin)
        return (0..<(World.giantWin * World.giantWin)).compactMap { i in
            let r = p[i]
            // The crown's top: giantPad's dome in Shaders/Saver.metal rises about yT + 36 + 0.55 * 56 + 9.5, the high limbs'
            // clumps to about yT + 84.
            return r.g > 0 ? GiantInfo(pos: SIMD2(Double(r.c.x), Double(r.c.y)), ground: Double(r.g), top: Double(r.g + r.yT) + 85, kind: Int(r.kind), yT: Double(r.yT), h: r.h) : nil
        }
    }

    // The CPU copy's heights are 4x4 maxima of the column tops, so their maximum is the map's.
    private static func readBack(_ cpu: MTLTexture) -> ([Float], [Float], Float) {
        let s = World.size >> World.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 2)
        cpu.getBytes(&raw, bytesPerRow: s * 4, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: 0)
        let heights = (0..<(s * s)).map { Float(raw[$0 * 2]) }
        return (heights, (0..<(s * s)).map { Float(raw[$0 * 2 + 1]) }, heights.max() ?? 0)
    }

    // Conservative (4x4 max) height in world coords.
    func height(x: Double, z: Double) -> Double? {
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(cpuOrigin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(cpuOrigin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !cpuHeights.isEmpty else { return nil }
        return Double(cpuHeights[cz * s + cx])
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

// Far heights for the camera: level 3's max mip 2 (32-voxel cells, toroidal, about +-7.7k around the camera),
// read back whenever level 3 changes; conservative, being a max.
final class FarHeights {
    static let n = 512, cell = 32.0
    fileprivate var heights: [Float16] = []
    fileprivate var centre = SIMD2<Int>(Int.min, Int.min)

    // Conservative terrain height at world (x, z) out to about 7.7k, or nil.
    func height(x: Double, z: Double) -> Double? {
        guard !heights.isEmpty else { return nil }
        let cx = Int(floor(x / FarHeights.cell)), cz = Int(floor(z / FarHeights.cell))
        guard abs(cx - centre.x) < 236, abs(cz - centre.y) < 236 else { return nil }
        let n = FarHeights.n
        return Double(heights[(cz & (n - 1)) * n + (cx & (n - 1))])
    }
}

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
    // giants: the giant table of the world's active map, which covers every window here.
    func encode(into cb: MTLCommandBuffer, center: SIMD2<Double>, drain: Bool, active: Int, giants: MTLBuffer, only: Int? = nil) {
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
                enc.setBytes(&r1, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
            } else {
                let mask = Int32(Clipmap.size / 2 - 1)
                var t1 = SIMD4<Int32>(j.w == Int32(Clipmap.size) ? 0 : r1.x & mask, j.h == Int32(Clipmap.size) ? 0 : r1.y & mask, r1.z, r1.w)
                enc.setBytes(&t1, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            }
            enc.dispatchThreads(MTLSize(width: Int(r1.z), height: Int(r1.w), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            // Coarser mips: the window is 64-aligned, so the region is contiguous in storage unless it spans the
            // whole axis.
            let n = Int32(Clipmap.size), mask = n - 1
            let sx = j.w == n ? 0 : j.x0 & mask, sz = j.h == n ? 0 : j.z0 & mask
            enc.setComputePipelineState(reducePipeline)
            for m in 2..<views.count {
                let dim = Int32(Clipmap.size >> m)
                // Coarser than the strip: rebuild the whole (tiny) mip. Only the top one is read past mip 6 (the
                // trace's MAX_LEVEL), for the global height bound; the rest cost a few tiny dispatches.
                var r = m <= 6 ? SIMD4<Int32>(sx >> Int32(m), sz >> Int32(m), max(j.w >> Int32(m), 1), max(j.h >> Int32(m), 1))
                               : SIMD4<Int32>(0, 0, dim, dim)
                enc.setTexture(views[m - 1], index: 0)
                enc.setTexture(views[m], index: 1)
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

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    // sceneFragment specialised by [towers][bands]: function constant 1 compiles the towers in (only frames with one in
    // view need them), constant 0 the band test in the DDA (BAND_FC, only while a giant or the jungle is near). Each
    // costs every pixel a little even unused.
    let scenePipelines: [[MTLRenderPipelineState]]
    // The giants' hanging strands, a pass over the scene while a giant stands within the world's edge (vineFragment).
    private let vinePipeline: MTLRenderPipelineState
    let postPipeline: MTLRenderPipelineState
    let taaPipeline: MTLRenderPipelineState
    // The glowing bugs (bioFragment), drawn over the scene in a pass of their own from sunset.
    private let bioPipeline: MTLRenderPipelineState
    private let cloudPipeline: MTLComputePipelineState
    // Shadow-height maps (see bakeShadowHeight in Shaders/Saver.metal): near and far cascades, a band rebaked per frame.
    private let shadowPipeline: MTLComputePipelineState
    private let shadowMaps: [MTLTexture]
    static let shadowN = 1024, shadowCell = 8.0
    private static let shadowBands = [256, 512]
    private var shadowBand = [0, 0]
    private var shadowValid = false
    private var shadowFrom: (cam: SIMD2<Double>, key: SIMD3<Double>)?
    // The terrain horizon around the camera (bakeHorizon): sky pixels above it skip the trace.
    private let horizonPipeline: MTLComputePipelineState
    private let horizonBuf: MTLBuffer
    // The forest-province fields around the camera (bakeProvinces in Shaders/Saver.metal), which shading samples rather
    // than evaluating per pixel; rebaked whole once the camera has moved an eighth of the window (+-24.6k), so the
    // coarsest ring (13.8k) never reads past it.
    private let provPipeline: MTLComputePipelineState
    private let provTex: MTLTexture
    private var provCentre: SIMD2<Double>?
    static let provN = 512, provCell = 96.0
    let far = FarHeights()
    private var farAge = 0
    // Level 3's generation last read back, and the one far.heights holds.
    private var farRequested = -1, farHeld = -1
    private var farSimClock = -1e9
    // The fine cloud field only drifts with the wind, so it is baked once around the camera (in drifted
    // coordinates) and re-baked when the camera nears its edge.
    private var cloudTex: MTLTexture?
    private var cloudCenter: SIMD2<Double>?
    static let cloudTexels = 2048, cloudTexel = 20.0
    private var slowTex: MTLTexture?
    private let slowPipeline: MTLComputePipelineState
    static let slowTexels = 512, slowTexel = 100.0
    private var history: [MTLTexture] = []
    // Lamps: per-frame position and state (schedule, flicker), and their light baked onto the map.
    private let lampPipeline: MTLComputePipelineState
    private let lightPipeline: MTLComputePipelineState
    private let lampPos: MTLBuffer
    private let lampState: MTLBuffer
    private let lightMap: MTLTexture
    private let lightDir: MTLTexture
    static let lightTexels = 512, hazeLamps = 16
    // Lamp indices: 16 haze lamps, then 12 smoke sources (and 4 spare).
    private var hazeRefs = [Int32](repeating: 0, count: 32)
    static let smokePlumes = 12
    private var smokeCount = 0
    // Per screen tile (LAMP_TILES x LAMP_TILES in Shaders/Saver.metal), the haze lamps (bits 0-15) and smoke plumes
    // (bits 16-27) whose bounding spheres the tile's rays may cross; the shader loops only over those.
    static let lampTiles = 16
    private static let lampReach: [Double] = [1, 20, 16, 7, 10, 24, 22, 6]   // LAMP_R in Shaders/Saver.metal
    private var tileMasks = [UInt32](repeating: 0, count: 16 * 16)
    static let ditherCells = 864.0
    // Sun height the glowing bugs start from (BIO_SUN in Shaders/Saver.metal).
    static let bioSun: Float = 0.07
    // Bug sprites: one per cube of the MOTE_N^3 block round the camera (bioVertex in Shaders/Saver.metal).
    static let moteCount = 18 * 18 * 18
    private var historyIndex = 0
    private var prevCamera: (pos: SIMD3<Double>, fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)?
    let world: World
    let clipmap: Clipmap
    let towers: Towers
    // The giants' hanging strands: their slots of baked maps and the frame's table.
    let vines: Vines
    let camera: FlightCamera
    let wildlife: Wildlife
    // Ring of bird buffers, one per frame in flight; the semaphore keeps the CPU from overwriting one the GPU reads.
    private var birdBuffers: [MTLBuffer] = []
    private var birdIndex = 0
    private let birdSlots = DispatchSemaphore(value: 3)
    var clock = SkyClock()
    var battery: Bool
    // The world's edge, a circle around the camera: it eases between the battery and AC radii rather than snapping
    // when the power source changes.
    private var edgeRadius: Double
    var timeScale: Double = 1
    var debugMode: Float = 0
    var headroom: Float = 0
    private var sceneTex: MTLTexture?
    private(set) var simTime: Double = 0
    private var lastFrame: CFTimeInterval?
    private var frameIndex: Float = 0
    private var gpuTimes: [Double] = []
    private var sceneSize = (width: 0, height: 0)
    let popLog: PopLog? = PopLog.enabled ? PopLog() : nil
    private(set) var aspect: Float = 16.0 / 10.0
    static var logTiming: Bool { Knobs.current.timing }
    static var profile: String { Knobs.current.profile }
    static var fullCoarse: Bool { Knobs.current.fullCoarse }
    // VR_CHUNKED: snapshots update the coarse rings a chunk per frame, as live.
    static var chunkedCoarse: Bool { Knobs.current.chunkedCoarse }
    // VR_LODSHIFT=v moves the LOD rings' centre v voxels ahead of the camera: frames differing only in it isolate pops.
    static var lodShift: Float { Knobs.current.lodShift }
    fileprivate(set) var exposure: Double = 1.5
    // Holds the time of day fixed for snapshots, so exposure and the shot director see the rendered sun.
    var pinnedPhase: Double?
    var renderScale: Double
    let dither: Bool
    // The screen's size in pixels: the dither grid follows it, while the drawable holds just one pixel per cell.
    private(set) var screenPixels = CGSize.zero
    var recorder: BugRecorder?

    // Cover drifts between clear and broken skies over tens of minutes.
    var cloudCover: Double {
        0.52 + 0.12 * sin(simTime * 0.0021 + 1.0) + 0.06 * sin(simTime * 0.0057 + 4.0)
    }

    init?(gpu: GPU, outputFormat: MTLPixelFormat, seed: UInt32, screenIndex: Int, battery: Bool, dither: Bool = true) {
        let device = gpu.device
        guard let queue = device.makeCommandQueue() else { return nil }
        let sceneStates = [false, true].map { t in [false, true].compactMap { b in gpu.scene(towers: t, bands: b) } }
        guard sceneStates.allSatisfy({ $0.count == 2 }),
              let vp = gpu.vine(),
              let tp = gpu.taa(),
              let bp = gpu.bio(),
              let pp = gpu.post(outputFormat, dither: dither),
              let cf = gpu.pipeline("cloudField"),
              let csf = gpu.pipeline("cloudSlowField"),
              let shP = gpu.pipeline("bakeShadowHeight"),
              let hzP = gpu.pipeline("bakeHorizon"),
              let hzB = device.makeBuffer(length: 1024 * 4, options: .storageModePrivate),
              let pvP = gpu.pipeline("bakeProvinces"),
              let lampP = gpu.pipeline("updateLamps"),
              let lightP = gpu.pipeline("bakeLights"),
              let lpos = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let lstate = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let world = World(gpu: gpu, seed: seed),
              let clipmap = Clipmap(gpu: gpu, seed: seed),
              let towers = Towers(gpu: gpu, seed: seed),
              let vines = Vines(gpu: gpu) else { return nil }
        self.device = device
        self.queue = queue
        scenePipelines = sceneStates
        postPipeline = pp
        taaPipeline = tp
        bioPipeline = bp
        cloudPipeline = cf
        slowPipeline = csf
        shadowPipeline = shP
        horizonPipeline = hzP
        horizonBuf = hzB
        provPipeline = pvP
        let pd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: Renderer.provN, height: Renderer.provN, mipmapped: false)
        pd.usage = [.shaderRead, .shaderWrite]
        pd.storageMode = .private
        guard let pt = device.makeTexture(descriptor: pd) else { return nil }
        provTex = pt
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: Renderer.shadowN, height: Renderer.shadowN, mipmapped: false)
        sd.usage = [.shaderRead, .shaderWrite]
        sd.storageMode = .private
        guard let sm0 = device.makeTexture(descriptor: sd), let sm1 = device.makeTexture(descriptor: sd) else { return nil }
        shadowMaps = [sm0, sm1]
        lampPipeline = lampP
        lightPipeline = lightP
        lampPos = lpos
        lampState = lstate
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Renderer.lightTexels, height: Renderer.lightTexels, mipmapped: false)
        ld.usage = [.shaderRead, .shaderWrite]
        ld.storageMode = .private
        guard let lmT = device.makeTexture(descriptor: ld), let ldT = device.makeTexture(descriptor: ld) else { return nil }
        lightMap = lmT
        lightDir = ldT
        self.world = world
        self.clipmap = clipmap
        self.towers = towers
        self.vines = vines
        vinePipeline = vp
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919, world: world, far: far)
        wildlife = Wildlife(seed: seed &+ UInt32(screenIndex) &* 7919)
        birdBuffers = (0..<3).compactMap { _ in device.makeBuffer(length: Wildlife.bufferLength, options: .storageModeShared) }
        guard birdBuffers.count == 3 else { return nil }
        self.battery = battery
        self.dither = dither
        renderScale = battery ? 0.7 : 1.0
        edgeRadius = Clipmap.handoff(battery ? Clipmap.levels - 1 : Clipmap.levels)
        super.init()
        wildlife.heightAt = { [unowned(unsafe) world] x, z in world.height(x: x, z: z) }
        wildlife.giantsAt = { [unowned(unsafe) world] in world.giants }
        camera.towersAt = { [unowned(unsafe) towers] in towers.list }
        // VR_START=x,z,yawDeg: the flight starts there (tests).
        if let v = Knobs.current.env["VR_START"]?.split(separator: ",").compactMap({ Double($0) }), v.count >= 3 {
            camera.start(at: SIMD2(v[0], v[1]), yaw: v[2] * .pi / 180)
        }
        world.placeGiants(around: SIMD2(camera.position.x, camera.position.z))
        if Knobs.current.env["VR_START"] == nil, camera.startNearGiant() {
            world.placeGiants(around: SIMD2(camera.position.x, camera.position.z))
        }
        let pre = SIMD2(camera.position.x, camera.position.z)
        towers.update(center: pre, queue: queue, wait: true)
        // Placing the camera reads only the far heights, which level 3 holds: the CPU waits for that level alone.
        refreshFar(only: 3)
        camera.moveToOpenGround()
        let post = SIMD2(camera.position.x, camera.position.z)
        // The lift reads the map centred on `pre` where it reaches round `post`, else the far heights: that map is built only
        // if it reaches.
        let o = World.mapOrigin(pre), r = FlightCamera.liftRadius
        if post == pre || (post.x + r >= Double(o.x) && post.x - r < Double(o.x) + Double(World.size)
                           && post.y + r >= Double(o.y) && post.y - r < Double(o.y) + Double(World.size)) {
            world.ensureCentered(on: pre)
        }
        camera.liftOverGround()
        // A start moved off the massifs needs its own map now: on the first frame it would wait behind the coarse rings.
        world.ensureCentered(on: post)
        // The other rings are built ahead of the first frame on the same queue, which orders that frame after them;
        // nothing waits on the CPU. Afterwards only strips are rewritten.
        if let cb = queue.makeCommandBuffer() {
            let t0 = CACurrentMediaTime()
            clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: true, active: activeLevels, giants: world.giantBuffer)
            if Renderer.logTiming {
                cb.addCompletedHandler { cb in
                    print(String(format: "coarse build %.1f ms (gpu %.1f)", (CACurrentMediaTime() - t0) * 1000, (cb.gpuEndTime - cb.gpuStartTime) * 1000))
                }
            }
            cb.commit()
        }
    }

    // Coarse levels kept up to date: on battery the outermost is dropped once the edge has eased inside it.
    private var activeLevels: Int {
        battery && edgeRadius <= Clipmap.handoff(Clipmap.levels - 1) ? Clipmap.levels - 1 : Clipmap.levels
    }

    convenience init?(view: MTKView, gpu: GPU, screen: NSScreen, seed: UInt32, screenIndex: Int, battery: Bool, dither: Bool = true) {
        let backingScale = screen.backingScaleFactor
        let edr = screen.maximumPotentialExtendedDynamicRangeColorComponentValue > 1
        if edr {
            view.colorPixelFormat = .rgba16Float
            view.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
            (view.layer as? CAMetalLayer)?.wantsExtendedDynamicRangeContent = true
        } else {
            view.colorPixelFormat = .bgra8Unorm
        }
        view.framebufferOnly = true
        view.preferredFramesPerSecond = battery ? 30 : 60
        self.init(gpu: gpu, outputFormat: view.colorPixelFormat, seed: seed, screenIndex: screenIndex, battery: battery, dither: dither)
        view.autoResizeDrawable = false
        screenPixels = CGSize(width: view.bounds.width * backingScale, height: view.bounds.height * backingScale)
        // postFragment gives every pixel of a cell the cell's colour, so presenting the cell grid and letting the layer
        // scale it up with nearest filtering shows the same pixels (without the dither a cell is one pixel).
        let s = sizes(screenPixels)
        view.drawableSize = CGSize(width: s.ow, height: s.oh)
        (view.layer as? CAMetalLayer)?.magnificationFilter = .nearest
        prepare(outputSize: screenPixels)
        view.delegate = self
    }

    func setBattery(_ b: Bool, view: MTKView) {
        battery = b
        renderScale = b ? 0.7 : 1.0
        view.preferredFramesPerSecond = b ? 30 : 60
    }

    // Advances the simulation; large steps are subdivided so steering stays stable.
    func simulate(_ dt: Double) {
        var remaining = dt * timeScale
        while remaining > 0 {
            let step = min(remaining, 1.0 / 30.0)
            if let p = pinnedPhase { clock.phase = p }
            camera.sun = clock.sunDirection
            camera.nightIndex = clock.nightIndex
            camera.step(dt: step)
            wildlife.step(dt: step, camera: camera, sight: sight(), sunY: clock.sunDirection.y, phase: clock.phase)
            popLog?.check(wildlife, sight: sight(), time: simTime + dt * timeScale - remaining + step, dt: step,
                          period: PopLog.period(sunY: clock.sunDirection.y, phase: clock.phase))
            if Renderer.logTiming, let g = world.height(x: camera.position.x, z: camera.position.z), camera.position.y < g + 2 {
                print(String(format: "camera low: y=%.1f ground=%.1f", camera.position.y, g))
            }
            clock.advance(step)
            if let p = pinnedPhase { clock.phase = p }
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
            towers.update(center: SIMD2(camera.position.x, camera.position.z), queue: queue, wait: world.synchronous)
            // Snapshots simulate long stretches without frames: keep the far heights following the camera.
            if world.synchronous && simTime + (dt * timeScale - remaining) - farSimClock > 2 {
                farSimClock = simTime + (dt * timeScale - remaining)
                refreshFar()
            }
            remaining -= step
        }
        simTime += dt * timeScale
        // Out to the AC edge only once its level is built (after battery mode it is rebuilt in pieces).
        let full = Clipmap.handoff(Clipmap.levels), part = Clipmap.handoff(Clipmap.levels - 1)
        let edgeTarget = battery || !clipmap.ready(Clipmap.levels) ? part : full
        let edgeStep = (full - part) / 4 * dt * timeScale
        edgeRadius = edgeRadius < edgeTarget ? min(edgeTarget, edgeRadius + edgeStep) : max(edgeTarget, edgeRadius - edgeStep)
        let e = clock.sunDirection.y
        let target = e > 0 ? 0.62 + 1.0 * max(0, 0.3 - e) / 0.3 + 0.08 * min(1, max(0, (e - 0.3) / 0.2)) : (e > -0.1 ? 2.02 + 0.4 * (-e / 0.1) : 1.6 + 0.4 * max(0, 0.3 + e) / 0.2)
        exposure += (target - exposure) * min(1, dt * timeScale * 0.2)
    }

    // What a frame rendered now could show; the frustum follows the scene size the last encode (or prepare) set.
    func sight() -> Sight {
        let b = camera.basis
        let tanV = tan(25.0 * .pi / 180)
        // As makeFrame's level-0 ring: its morph band starts 140 inside the handoff, less a cell for the floor's 4x4 grain.
        let p = camera.position - SIMD3(Double(world.origin.x), 0, Double(world.origin.y)), half = Double(World.size / 2)
        let edge = half - 8 - max(abs(p.x - half), abs(p.z - half)) - abs(Double(Renderer.lodShift))
        return Sight(pos: camera.position, fwd: b.fwd, right: b.right, up: b.up, tanV: tanV, aspect: Double(aspect),
                     pixelAngle: 2 * tanV / Double(max(sceneSize.height, 1)), sunY: clock.sunDirection.y,
                     worldEdge: Clipmap.handoff(Clipmap.levels), floorReach: min(880, edge) - 140 - 6 - abs(Double(Renderer.lodShift)),
                     world: world)
    }

    // Scene and history sizes for an output size: the retro dither shows one flat colour per cell, so the temporal
    // history only needs the cell grid, and the scene renders at (a fraction of) it. Without the dither the history
    // is every pixel.
    private func sizes(_ outputSize: CGSize) -> (ow: Int, oh: Int, w: Int, h: Int) {
        let cell = dither ? max(1.0, (outputSize.width / Renderer.ditherCells).rounded()) : 1
        let ow = max(16, Int((outputSize.width / cell).rounded())), oh = max(16, Int((outputSize.height / cell).rounded()))
        return (ow, oh, max(16, Int(Double(ow) * renderScale)), max(16, Int(Double(oh) * renderScale)))
    }

    // Sets the frustum the simulation sees before the first frame is encoded.
    func prepare(outputSize: CGSize) {
        let s = sizes(outputSize)
        sceneSize = (s.w, s.h)
        aspect = Float(s.w) / Float(s.h)
    }

    // shadeHeight sets the shading prefilter's pixel: the output's without the dither, so a scene smaller than the
    // output keeps the output's surface detail for the temporal resolve to gather.
    func makeFrame(size: CGSize, shadeHeight: Int) -> Frame {
        var f = Frame()
        let b = camera.basis
        let o = SIMD3(Double(world.origin.x), 0, Double(world.origin.y))
        let p = camera.position - o
        f.camPos = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(tan(25.0 * .pi / 180)))
        f.camFwd = SIMD4(SIMD3<Float>(b.fwd), Float(size.width / size.height))
        f.camRight = SIMD4(SIMD3<Float>(b.right), Float(simTime))
        f.camUp = SIMD4(SIMD3<Float>(b.up), battery ? 0 : 1)
        f.sunDir = SIMD4(SIMD3<Float>(clock.sunDirection), Float(clock.phase))
        f.resolution = SIMD4(Float(size.width), Float(size.height), Float(1 / Double(shadeHeight)), Float(1 / size.height))
        f.mapInfo = SIMD4(world.origin.x, world.origin.y, Int32(World.size), Int32(bitPattern: world.seed))
        f.misc = SIMD4(Float(exposure), world.globalMax, 0.6, Float(cloudCover))
        f.misc2 = SIMD4(frameIndex, 1, debugMode, headroom)
        let sunY = clock.sunDirection.y
        let ph = clock.phase - floor(clock.phase)
        // Night progress runs 0 at dusk to 1 at dawn and eases back to 0 through the morning, so the burn-down
        // schedule never steps. A night keeps one index from dusk through to the morning after.
        let progress = ph < 0.25 ? 1 - { let s = min(max((ph - 0.08) / 0.12, 0), 1); return s * s * (3 - 2 * s) }() : min(max((ph - 0.52) / 0.46, 0), 1)
        f.lights = SIMD4(Float(sunY), Float(progress), Float(clock.nightIndex), Float(min(max((0.08 - sunY) / 0.1, 0), 1)))
        f.refs = SIMD4(Int32(f.lights.w > 0 ? hazeCount : 0), 0, 0, 0)
        pickSmoke()
        f.refs.y = Int32(smokeCount)
        if let pc = prevCamera {
            let pp = pc.pos - o
            f.prevPos = SIMD4(Float(pp.x), Float(pp.y), Float(pp.z), 1)
            f.prevFwd = SIMD4(SIMD3<Float>(pc.fwd), 0)
            f.prevRight = SIMD4(SIMD3<Float>(pc.right), 0)
            f.prevUp = SIMD4(SIMD3<Float>(pc.up), 0)
        }
        // Halton(2,3) subpixel jitter, accumulated by the temporal resolve; upsampling needs more phases to
        // cover every output pixel.
        let phases = min(64, Int((8 / min(renderScale * renderScale, 1)).rounded(.up)))
        let k = Int(frameIndex) % phases + 1
        func halton(_ i: Int, _ b: Int) -> Double { var f = 1.0, r = 0.0, i = i; while i > 0 { f /= Double(b); r += f * Double(i % b); i /= b }; return r }
        f.jitter = SIMD4(Float((halton(k, 2) - 0.5) * 2 / size.width), Float((halton(k, 3) - 0.5) * 2 / size.height), 0, 0)
        // LOD rings. The 1-voxel map hands off at 880, inside the 1024 - 8 (neighbour reads) - 100 (recentre drift)
        // - ~21 (drift while a regeneration is in flight: 41 pieces at 30 fps) it covers; nearer its edge only if the camera outran
        // regeneration. Each level morphs toward the next over a band, reaching it 1.5 parent columns before the
        // handoff circle.
        let half = Double(World.size / 2)
        let edge = half - 8 - max(abs(p.x - half), abs(p.z - half)) - abs(Double(Renderer.lodShift))
        let b01 = min(880, edge)
        var lods = [SIMD4<Float>(Float(b01 - 140), Float(b01), Float(b01 - 3), 1)]
        // Levels drawn: those built, out to the edge. Past the battery edge the last level's band eases in with the
        // edge, so no ring pops when the level count changes.
        let built = (1...Clipmap.levels).prefix { clipmap.ready($0) }.count
        let levels = min(built, edgeRadius > Clipmap.handoff(Clipmap.levels - 1) ? Clipmap.levels : Clipmap.levels - 1)
        let rMax = levels == Clipmap.levels ? edgeRadius : Clipmap.handoff(levels)
        let grow = levels == Clipmap.levels ? (edgeRadius - Clipmap.handoff(levels - 1)) / (Clipmap.handoff(levels) - Clipmap.handoff(levels - 1)) : 1
        for k in 1...max(levels, 1) {
            let b1 = Clipmap.handoff(k), s = Double(1 << k)
            if k == levels {
                lods.append(SIMD4(Float(rMax), Float(rMax), Float(rMax), Float(s)))
            } else {
                let u = k == levels - 1 ? grow : 1
                lods.append(SIMD4(Float(b1 - u * 0.2 * b1), Float(b1), Float(b1 - u * 3 * s), Float(s)))
            }
        }
        while lods.count < 5 { lods.append(lods.last!) }
        (f.lod0, f.lod1, f.lod2, f.lod3, f.lod4) = (lods[0], lods[1], lods[2], lods[3], lods[4])
        f.fade = SIMD4(Float(0.7 * rMax), Float(rMax), Float(levels + 1), Renderer.lodShift)
        return f
    }

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
        var f = makeFrame(size: CGSize(width: w, height: h), shadeHeight: dither ? h : oh)
        frameIndex = (frameIndex + 1).truncatingRemainder(dividingBy: 4096)
        bakeClouds(into: cb, frame: &f)
        if Renderer.fullCoarse { clipmap.invalidate() }
        clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: world.synchronous && !Renderer.chunkedCoarse,
                       active: activeLevels, giants: world.giantBuffer)
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
            enc.setBuffer(world.giantBuffer, offset: 0, index: 2)
            // HORIZON_SEGS (16) threads per bin, one threadgroup each.
            enc.dispatchThreads(MTLSize(width: 16, height: 1024, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        }
        bakes?.endEncoding()
        // The giants within the world's edge: their strand maps (baked on arrival) and the frame's table of them.
        vines.update(giants: world.giants, eye: SIMD2(camera.position.x, camera.position.z), reach: Double(f.fade.y), origin: world.origin, seed: world.seed, into: cb)
        // Snapshots refresh the far heights synchronously as they simulate.
        if !world.synchronous { updateFar(into: cb) }
        birdSlots.wait()
        cb.addCompletedHandler { [birdSlots] _ in birdSlots.signal() }
        let birdBuf = birdBuffers[birdIndex]
        birdIndex = (birdIndex + 1) % birdBuffers.count
        f.birds.x = Float(wildlife.upload(into: birdBuf, origin: SIMD3(Double(world.origin.x), 0, Double(world.origin.y))))
        let b = camera.basis
        prevCamera = (camera.position, b.fwd, b.right, b.up)
        let histIn = history[historyIndex], histOut = history[1 - historyIndex]
        historyIndex = 1 - historyIndex

        let sp = MTLRenderPassDescriptor()
        sp.colorAttachments[0].texture = sceneTex
        sp.colorAttachments[0].loadAction = .dontCare
        sp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: sp) {
            // The band test only skips through cells inside the 1-voxel map's unmorphed disc (trace in
            // Shaders/Saver.metal), so it is compiled in only while bands lie that close.
            let bands = world.bandsNear(SIMD2(camera.position.x, camera.position.z), radius: Clipmap.handoff(0) + 40)
            let tb = camera.basis, tanV = tan(25.0 * .pi / 180) * 1.05
            towers.frameTable(eye: camera.position, fwd: tb.fwd, right: tb.right, up: tb.up, tanH: tanV * Double(aspect), tanV: tanV)
            enc.setRenderPipelineState(scenePipelines[towers.tableCount > 0 ? 1 : 0][bands && Renderer.profile != "noband" ? 1 : 0])
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
            enc.setFragmentBuffer(lampPos, offset: 0, index: 2)
            enc.setFragmentBuffer(lampState, offset: 0, index: 3)
            enc.setFragmentBytes(&hazeRefs, length: 128, index: 4)
            enc.setFragmentBytes(&tileMasks, length: tileMasks.count * 4, index: 7)
            enc.setFragmentBuffer(world.giantBuffer, offset: 0, index: 8)
            enc.setFragmentBytes(towers.table, length: Towers.tableBytes, index: 9)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
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
                enc.setRenderPipelineState(vinePipeline)
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

    private var hazeCount = 0

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
    private func pickSmoke() {
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
    private func fillLampTiles(frame f: Frame, width: Int, height: Int) {
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

    private func bakeLamps(into cb: MTLCommandBuffer, frame f: inout Frame) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        let n = World.siteGrid * World.siteGrid * World.maxLamps
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

    func farHeight(x: Double, z: Double) -> Double? { far.height(x: x, z: z) }

    private var farCameraCell: SIMD2<Int> {
        SIMD2(Int(floor(camera.position.x / FarHeights.cell)), Int(floor(camera.position.z / FarHeights.cell)))
    }

    // Copies the far heights into a buffer the CPU reads once `cb` completes.
    private func encodeFar(into cb: MTLCommandBuffer) -> MTLBuffer? {
        let n = FarHeights.n
        guard clipmap.ready(3), let buf = device.makeBuffer(length: n * n * 2, options: .storageModeShared),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: clipmap.height, sourceSlice: 2, sourceLevel: 2, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: n, height: n, depth: 1),
                  to: buf, destinationOffset: 0, destinationBytesPerRow: n * 2, destinationBytesPerImage: n * n * 2)
        blit.endEncoding()
        return buf
    }

    private static func readFar(_ buf: MTLBuffer) -> [Float16] {
        Array(UnsafeBufferPointer(start: buf.contents().assumingMemoryBound(to: Float16.self), count: FarHeights.n * FarHeights.n))
    }

    // Level 3 changes only when its window moves a strip, every 512 voxels of flight: its heights are read back then,
    // off the main thread. The range farHeight trusts still follows the camera every 20 frames, as the window does.
    private func updateFar(into cb: MTLCommandBuffer) {
        let gen = clipmap.generation(3)
        if gen != farRequested, let buf = encodeFar(into: cb) {
            farRequested = gen
            cb.addCompletedHandler { [weak self] cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self?.farRequested = -1 }; return }
                let heights = Renderer.readFar(buf)
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.far.heights = heights
                    self.farHeld = gen
                    self.far.centre = self.farCameraCell
                }
            }
        }
        farAge += 1
        if farAge >= 20 {
            farAge = 0
            if farHeld == gen && clipmap.ready(3) { far.centre = farCameraCell }
        }
    }

    // Brings the rings to the camera and reads the far heights back at once (startup and snapshots).
    private func refreshFar(only: Int? = nil) {
        guard let cb = queue.makeCommandBuffer() else { return }
        clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: true, active: activeLevels, giants: world.giantBuffer, only: only)
        let buf = encodeFar(into: cb)
        cb.commit()
        cb.waitUntilCompleted()
        guard let buf else { return }
        far.heights = Renderer.readFar(buf)
        farRequested = clipmap.generation(3)
        farHeld = farRequested
        far.centre = farCameraCell
    }

    // The key light as the shader's makeLight picks it, and whether it casts shadows.
    static func keyLight(_ sun: SIMD3<Double>) -> (SIMD3<Double>, Bool) {
        let e = sun.y
        if e > -0.02 { return (sun, true) }
        let glow = simd_normalize(SIMD3(sun.x, 0.05, sun.z))
        let moon = simd_normalize(SIMD3(-sun.x, max(-sun.y * 0.7, 0.08), -sun.z))
        let t = min(max((e + 0.09) / (-0.16 + 0.09), 0), 1)
        let m = t * t * (3 - 2 * t)
        return (simd_normalize(glow + (moon - glow) * m), m > 0.5)
    }

    // A band of each shadow-height map per frame, around the camera, for the current key light; everything at once
    // when the camera or the light has jumped (startup, snapshots), so the maps never hold another place's shadows.
    private func bakeShadows(into enc: MTLComputeCommandEncoder, frame f: inout Frame) {
        let (key, casts) = Renderer.keyLight(clock.sunDirection)
        let levels = (1...Clipmap.levels).prefix { clipmap.ready($0) }.count
        // VR_PROFILE (tests): "noshadow" drops terrain shadows, "nobake" keeps the maps but skips their rebaking.
        // "bitsN" sets profiling flags directly: 8 reflections unshaded, 16 no sky visibility, 32 no cloud shadow or veil,
        // 64 primary hits unshaded.
        f.shadow.y = Renderer.profile == "nearclouds" ? 1 : Renderer.profile == "lowmax" ? 2 : Renderer.profile == "norefl" ? 4 : Float(Renderer.profile.hasPrefix("bits") ? Int(Renderer.profile.dropFirst(4)) ?? 0 : 0)
        if Renderer.profile == "noshadow" { f.shadow.x = 3; return }
        if Renderer.profile == "oldshadow" { f.shadow.x = 0; return }
        if Renderer.profile == "nobake" && shadowValid { f.shadow.x = 1; return }
        guard casts, levels > 0 else { shadowValid = false; f.shadow.x = 0; return }
        let cam = SIMD2(camera.position.x, camera.position.z)
        let jumped = shadowFrom.map { simd_length($0.cam - cam) > 300 || simd_dot($0.key, key) < 0.995 } ?? true
        if !shadowValid || jumped { shadowFrom = (cam, key); shadowBand = [0, 0] }
        let full = !shadowValid || jumped
        let cosE = simd_length(SIMD2(key.x, key.z))
        enc.setComputePipelineState(shadowPipeline)
        enc.setTexture(clipmap.height, index: 1)
        for c in 0..<2 {
            let cell = Renderer.shadowCell * (c == 0 ? 1 : 4)
            let bands = Renderer.shadowBands[c], rows = Renderer.shadowN / bands
            let todo = full ? bands : 1
            for _ in 0..<todo {
                // 96: SHADOW_NEAR in Shaders/Saver.metal, the traced ray's reach.
                var b = ShadowBake(key: SIMD4(Float(key.x), Float(key.y), Float(key.z), Float(max(0.8 * 96 * cosE, cell))),
                                   cam: SIMD4(Float(cam.x), Float(cam.y), Float(cell), Float(levels)),
                                   rows: SIMD4(Int32(shadowBand[c] * rows), Int32(floor(cam.x / cell)), Int32(floor(cam.y / cell)), 0))
                enc.setTexture(shadowMaps[c], index: 0)
                enc.setBytes(&b, length: MemoryLayout<ShadowBake>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Renderer.shadowN, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                shadowBand[c] = (shadowBand[c] + 1) % bands
                if c == 0 && shadowBand[0] == 0 { shadowFrom = (cam, key) }
            }
        }
        shadowValid = true
        f.shadow.x = 1
    }

    private func bakeClouds(into cb: MTLCommandBuffer, frame f: inout Frame) {
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

    // A queue runs its command buffers in order, so an empty one completing means all committed before it have.
    func drain() {
        for q in [queue, world.genQueue] {
            guard let cb = q.makeCommandBuffer() else { continue }
            cb.commit()
            cb.waitUntilCompleted()
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let dt = min(0.1, now - (lastFrame ?? now))
        lastFrame = now
        simulate(dt)
        if view.colorPixelFormat == .rgba16Float {
            headroom = Float(view.window?.screen?.maximumExtendedDynamicRangeColorComponentValue ?? 1)
            // Capped: past ~4x, glints and glitter stop reading as light and start reading as effects.
            headroom = min(max(headroom, 1.0001), 4)
            if frameIndex == 30 { dlog("EDR headroom now \(headroom), potential \(view.window?.screen?.maximumPotentialExtendedDynamicRangeColorComponentValue ?? 0), format \(view.colorPixelFormat.rawValue)") }
        }
        guard let drawable = view.currentDrawable, let cb = queue.makeCommandBuffer() else { return }
        encode(into: cb, output: drawable.texture, outputSize: screenPixels)
        recorder?.capture(self, into: cb, source: drawable.texture, time: now)
        if Renderer.logTiming {
            cb.addCompletedHandler { [weak self] cb in
                guard let self else { return }
                self.gpuTimes.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if self.gpuTimes.count >= 120 {
                    let sorted = self.gpuTimes.sorted()
                    print(String(format: "gpu ms: median %.2f  p95 %.2f  (%dx%d scene)", sorted[60], sorted[113],
                                 self.sceneSize.width, self.sceneSize.height))
                    self.gpuTimes.removeAll()
                }
            }
        }
        cb.present(drawable)
        cb.commit()
    }
}

// Offscreen render to PNG, for checking visuals without a display.
enum Snapshot {
    // False if the renderer could not start or a PNG could not be written.
    static func run(settings: Settings, path: String, gpu shared: GPU? = nil) -> Bool {
        guard let gpu = shared ?? MTLCreateSystemDefaultDevice().flatMap({ GPU(device: $0) }),
              let r = Renderer(gpu: gpu, outputFormat: .rgba8Unorm, seed: settings.seed == 0 ? 1 : settings.seed,
                               screenIndex: 0, battery: settings.forceBattery, dither: !settings.noDither) else {
            FileHandle.standardError.write("renderer init failed\n".data(using: .utf8)!)
            return false
        }
        let device = gpu.device
        var wrote = true
        r.debugMode = settings.debugMode
        r.timeScale = settings.timeScale
        r.world.synchronous = true
        r.camera.pitchOverride = settings.pitch
        let w = Int(settings.screenshotSize.width), h = Int(settings.screenshotSize.height)
        r.renderScale = settings.renderScale ?? r.renderScale
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        let tex = device.makeTexture(descriptor: td)!
        r.pinnedPhase = settings.phase
        r.prepare(outputSize: CGSize(width: w, height: h))
        r.headroom = settings.headroom ?? 0
        let env = Knobs.current.env
        // VR_TOWERS=n: lists the towers of the n-cell square of tower cells round the camera's start, nearest first.
        if let n = Int(env["VR_TOWERS"] ?? "") {
            let here = SIMD2(r.camera.position.x, r.camera.position.z)
            for t in r.towers.scan(around: here, cells: n, queue: r.queue).sorted(by: { simd_length($0.pos - here) < simd_length($1.pos - here) }) {
                print(String(format: "tower %.0f,%.0f dist %.0f kind %d member %d R0 %.0f base %.0f height %.0f top %.0f lit %d moist %.2f temp %.2f",
                             t.pos.x, t.pos.y, simd_length(t.pos - here), t.kind, t.member, t.radius, t.base, t.height, t.top, t.lit ? 1 : 0, t.moist, t.temp))
            }
        }
        // VR_SHOW=kind,distance,height,bearing[,rank]: hold the camera on the rank-th nearest site of a kind
        // (1 henge .. 6 camp), `distance` from it along `bearing` degrees, `height` above the ground there.
        if let spec = env["VR_SHOW"] {
            let v = spec.split(separator: ",").compactMap { Double($0) }
            if v.count >= 4 {
                let here = SIMD2(r.camera.position.x, r.camera.position.z)
                let pick = r.world.sites.filter { $0.kind.rawValue == Int(v[0]) }
                    .sorted { simd_length($0.pos - here) < simd_length($1.pos - here) }
                let rank = v.count > 4 ? Int(v[4]) : 0
                if rank < pick.count {
                    let site = pick[rank]
                    let b = v[3] * .pi / 180
                    let cam = site.pos + SIMD2(sin(b), cos(b)) * v[1]
                    let g = r.world.height(x: cam.x, z: cam.y) ?? 20
                    let sg = r.world.height(x: site.pos.x, z: site.pos.y) ?? 20
                    let y = max(g, 19.85) + v[2]
                    let yaw = atan2(site.pos.x - cam.x, site.pos.y - cam.y)
                    let pitch = atan2(sg + 2 - y, v[1]) + (settings.pitch ?? 0)
                    r.camera.freeze(position: SIMD3(cam.x, y, cam.y), yaw: yaw, pitch: pitch)
                    r.camera.pitchOverride = nil
                    print(String(format: "showing %@ site %u at (%.0f, %.0f), %d of %d", "\(site.kind)", site.id, site.pos.x, site.pos.y, rank + 1, pick.count))
                } else {
                    print("no site of kind \(Int(v[0])) rank \(rank) (\(pick.count) on this map)")
                }
            }
        }
        // VR_CAM=x,z,height,yawDeg,pitchDeg[,abs[,rollDeg]]: holds the camera at world (x, z), `height` above the ground
        // there (absolute when abs is 1).
        let camSpec = env["VR_CAM"].map { $0.split(separator: ",").compactMap { Double($0) } }
        if let v = camSpec, v.count >= 5 {
            r.camera.freeze(position: SIMD3(v[0], 400, v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            r.camera.pitchOverride = nil
        }
        // VR_FLIGHT=minutes: flies the autopilot that long, logging altitude, clearance and climb every 12 s and
        // writing a frame each minute next to the screenshot path. VR_FTRACE=1 also prints every 0.25 s: time, x, z, y, ground,
        // yaw, the altitude terms (target, climb rate, need (-1: floor clamp), rim lift, regional floor, ahead, planned climb)
        // and the routing (blocked ahead, route offset).
        if let mins = Double(env["VR_FLIGHT"] ?? "") {
            r.camera.tracing = env["VR_FTRACE"] != nil
            var maxClimb = 0.0, minClear = Double.infinity, maxY = 0.0, lows = 0
            var walled = 0, walledRun = 0, maxWalledRun = 0, fastClimb = 0, prevY = r.camera.position.y
            var lastY = r.camera.position.y
            var underCrown = 0, giantsSeen = Set<Int>()
            // Ruined towers: the closest the flight came to a plinth's edge below its top, and the seconds a tower stood in
            // frame within 9k.
            var towerEdge = Double.infinity, towerSeen = 0, towerClose = 0
            var towersMet = Set<SIMD2<Int>>()
            for i in 1...Int(mins * 60) {
                for q in 0..<4 {
                    r.simulate(0.25)
                    if env["VR_FTRACE"] != nil {
                        let c = r.camera.position
                        print(String(format: "ft %.2f %.1f %.1f %.2f %.2f %.3f", Double(i - 1) + Double(q + 1) * 0.25, c.x, c.z, c.y, r.world.height(x: c.x, z: c.z) ?? -1, r.camera.yaw)
                              + " " + r.camera.terms.map { String(format: "%.1f", $0) }.joined(separator: " "))
                    }
                }
                let p = r.camera.position
                let g = r.world.height(x: p.x, z: p.z) ?? 0
                if p.y - lastY > 6 { print(String(format: "flight jump t=%d pos=(%.0f,%.0f) %.0f -> %.0f", i, p.x, p.z, lastY, p.y)) }
                maxClimb = max(maxClimb, p.y - lastY)
                lastY = p.y
                minClear = min(minClear, p.y - g)
                maxY = max(maxY, p.y)
                if p.y < g + 2 { lows += 1 }
                var seen = false
                for t in r.towers.list {
                    let v = t.pos - SIMD2(p.x, p.z), d = simd_length(v)
                    if p.y < t.top { towerEdge = min(towerEdge, d - t.radius * 1.42) }
                    if d < 9000, abs(atan2(sin(atan2(v.x, v.y) - r.camera.yaw), cos(atan2(v.x, v.y) - r.camera.yaw))) < 0.6 {
                        seen = true
                        if towersMet.insert(SIMD2(Int(t.pos.x), Int(t.pos.y))).inserted {
                            print(String(format: "flight tower seen t=%d at %.0f,%.0f member %d dist %.0f", i, t.pos.x, t.pos.y, t.member, d))
                        }
                    }
                    if d < 1500 { towerClose += 1 }
                }
                if seen { towerSeen += 1 }
                let wall = r.camera.wallInFrame()
                if wall >= 0.4 { walled += 1; walledRun += 1; maxWalledRun = max(maxWalledRun, walledRun) } else { walledRun = 0 }
                if p.y - (i == 1 ? p.y : prevY) > 6 { fastClimb += 1 }
                for g in r.world.giants {
                    let dg = simd_length(SIMD2(p.x, p.z) - g.pos)
                    if dg < GiantInfo.reach && p.y < g.top { underCrown += 1 }
                    if dg < 1500 { giantsSeen.insert(Int(g.pos.x) * 100_003 + Int(g.pos.y)) }
                }
                prevY = p.y
                if i % 12 == 0 {
                    fflush(stdout)
                    let nearG = r.world.giants.map { simd_length(SIMD2(p.x, p.z) - $0.pos) }.min() ?? -1
                    print(String(format: "flight t=%d pos=(%.0f,%.0f) y=%.0f ground=%.0f wall=%.2f far=%.0f yaw=%.0f giant=%.0f", i, p.x, p.z, p.y, g, wall,
                                 r.farHeight(x: p.x + sin(r.camera.yaw) * 600, z: p.z + cos(r.camera.yaw) * 600) ?? -1, r.camera.yaw * 180 / .pi, nearG))
                }
                if i % 60 == 0 {
                    for _ in 0..<4 {
                        let cb = r.queue.makeCommandBuffer()!
                        r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                        cb.commit()
                        cb.waitUntilCompleted()
                    }
                    var bytes = [UInt8](repeating: 0, count: w * h * 4)
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    wrote = write(bytes: bytes, width: w, height: h, to: path.replacingOccurrences(of: ".png", with: "_m\(i / 60).png")) && wrote
                }
            }
            print(String(format: "flight towers: closest plinth edge %.0f, seconds with a tower ahead within 9k %d, tower-seconds within 1.5k %d", towerEdge, towerSeen, towerClose))
            print(String(format: "flight done: max climb %.2f/s, min clearance %.1f, max y %.0f (ceiling %.0f), low seconds %d, walled seconds %d (longest %d), seconds climbing over 6/s %d, seconds under a crown %d, giants passed within 1500 %d", maxClimb, minClear, maxY, FlightCamera.ceiling, lows, walled, maxWalledRun, fastClimb, underCrown, giantsSeen.count))
            if let log = r.popLog { print(log.summary(r.wildlife)) }
            return wrote
        }
        // VR_BUGTEST=seconds: flies that long past the first time, recording as the saver does, then writes a bug report
        // beside the screenshot path, whose last frame is the screenshot.
        if let secs = Double(env["VR_BUGTEST"] ?? "") {
            BugReport.test(r, gpu: gpu, settings: settings, output: tex, path: path, seconds: secs)
            return wrote
        }
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            r.simulate(t - simulated)
            if let v = camSpec, v.count >= 5 {
                let g = (v.count > 5 && v[5] == 1) ? 0 : max(r.world.height(x: v[0], z: v[1]) ?? 20, 19.85)
                r.camera.freeze(position: SIMD3(v[0], g + v[2], v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180,
                                roll: v.count > 6 ? v[6] * .pi / 180 : 0)
            }
            if let phase = settings.phase { r.clock.phase = phase }
            // VR_EXPOSURE=e: the exposure a bug report recorded (snapshots jump straight to the target the saver eases toward).
            if let e = Double(env["VR_EXPOSURE"] ?? "") { r.exposure = e }
            simulated = t
            if let d = Double(env["VR_BIRDTEST"] ?? "") { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = Double(env["VR_MURMUR"] ?? "") { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            // VR_BENCH=n renders n still frames (gpu= is their minimum): more frames ride out the GPU's clock ramp.
            let frames = env["VR_MOTION"] != nil ? Int(env["VR_MOTION_FRAMES"] ?? "") ?? 48 : Int(env["VR_BENCH"] ?? "") ?? 12
            // VR_FRAMES=n also writes each of the last n frames, for measuring frame-to-frame crawl.
            let dump = min(Int(env["VR_FRAMES"] ?? "") ?? 0, frames)
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
            let out = settings.screenshotTimes.count > 1 ? path.replacingOccurrences(of: ".png", with: "_t\(Int(t)).png") : path
            for i in 0..<frames {
                // VR_POWER_FLIP=i switches between battery and AC power at frame i (render scale kept).
                if Int(env["VR_POWER_FLIP"] ?? "") == i { r.battery.toggle() }
                if env["VR_MOTION"] != nil { r.simulate(1.0 / 60) }
                let cb = r.queue.makeCommandBuffer()!
                r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                cb.commit()
                cb.waitUntilCompleted()
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if Renderer.logTiming { print(String(format: "frame %d gpu %.2f ms cam (%.1f, %.1f, %.1f)%@", i, times.last!, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.clipmap.idle ? "" : " (coarse job pending)")) }
                if i >= frames - dump {
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    wrote = write(bytes: bytes, width: w, height: h, to: out.replacingOccurrences(of: ".png", with: "_f\(i - frames + dump).png")) && wrote
                }
            }
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            wrote = write(bytes: bytes, width: w, height: h, to: out) && wrote
            let seen = r.wildlife.inView(camera: r.camera, aspect: Double(w) / Double(h), tanHalf: tan(25.0 * .pi / 180))
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) yaw=%.1f pitch=%.1f gpu=%.2fms mem=%.0fMB birds=%d flocks=%d near=%.0f %@ -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.camera.yaw * 180 / .pi, r.camera.pitch * 180 / .pi, times.min()!,
                         Double(device.currentAllocatedSize) / 1048576, seen.count, seen.flocks, seen.nearest.isFinite ? seen.nearest : -1, Snapshot.sitesInView(r), out))
        }
        if let log = r.popLog { print(log.summary(r.wildlife)) }
        return wrote
    }

    // Sites within 700 voxels whose centre falls inside the frame (not tested for occlusion).
    static func sitesInView(_ r: Renderer) -> String {
        let b = r.camera.basis
        let o = r.camera.position
        let tanV = tan(25.0 * .pi / 180), tanH = tanV * Double(r.aspect)
        var seen: [String] = []
        for site in r.world.sites {
            let g = r.world.height(x: site.pos.x, z: site.pos.y) ?? 20
            let v = SIMD3(site.pos.x, g + 1, site.pos.y) - o
            let z = simd_dot(v, b.fwd), dist = simd_length(v)
            guard z > 0, dist < 700 else { continue }
            if abs(simd_dot(v, b.right)) / z < tanH && abs(simd_dot(v, b.up)) / z < tanV {
                seen.append("\(site.kind)@\(Int(dist))")
            }
        }
        return "sites=\(seen.count)[\(seen.joined(separator: " "))]"
    }

    static func write(bytes: [UInt8], width: Int, height: Int, to path: String) -> Bool {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: cs,
                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                          decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let rep = NSBitmapImageRep(cgImage: img)
        return (try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))) != nil
    }
}
