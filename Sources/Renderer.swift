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
}

enum GPU {
    static func library(device: MTLDevice) -> MTLLibrary? {
        if let url = Bundle.main.url(forResource: "Saver", withExtension: "metallib"),
           let lib = try? device.makeLibrary(URL: url) { return lib }
        // Loose build: metallib next to the binary.
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        return try? device.makeLibrary(URL: exe.appendingPathComponent("Saver.metallib"))
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
    static let dayLength: Double = 24 * 60

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

// A structure site as the camera and the smoke and halo lists see it, in world coordinates.
struct SiteInfo {
    enum Kind: Int { case none = 0, henge, cairn, ruins, hamlet, jetty, camp }
    struct Lamp { var pos: SIMD3<Double>; var kind: Int; var rise: Double; var index: Int }
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
    static let siteGrid = 18, siteCell = 128, maxLamps = 8, maxParts = 16
    static let siteStride = 32 + 16 * maxLamps + 16 * maxParts
    // Recentred once the camera drifts this far from the map centre. The LOD handoff (Renderer.makeFrame) relies
    // on the map reaching 1024 - 8 - (this + the drift while a regeneration is in flight) around the camera.
    static let recentre = 100.0
    // Background regeneration runs a band of rows of one pass per frame, so no frame carries a whole map's work.
    static let bandRows = 256

    // One generated map: the column texture with its max-mip chain, and the CPU copy the camera steers by.
    private final class Buffer {
        let tex: MTLTexture
        let levels: [MTLTexture]
        let climate: MTLTexture
        let structTex: MTLTexture
        let sites: MTLBuffer
        var origin = SIMD2<Int32>(Int32.min, Int32.min)
        var globalMax: Float = 255
        var cpuHeights: [Float] = []
        var siteList: [SiteInfo] = []

        init?(device: MTLDevice) {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: true)
            td.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            td.storageMode = .shared
            let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
            cd.usage = [.shaderRead, .shaderWrite]
            cd.storageMode = .private
            let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: World.size, height: World.size, mipmapped: false)
            sd.usage = [.shaderRead, .shaderWrite]
            sd.storageMode = .private
            guard let t = device.makeTexture(descriptor: td), let c = device.makeTexture(descriptor: cd), let st = device.makeTexture(descriptor: sd),
                  let sb = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.siteStride, options: .storageModeShared) else { return nil }
            tex = t
            climate = c
            structTex = st
            sites = sb
            levels = (0..<t.mipmapLevelCount).map {
                t.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)!
            }
        }
    }

    let device: MTLDevice
    let seed: UInt32
    // Regeneration runs on its own queue so frames never wait behind it.
    private let genQueue: MTLCommandQueue
    private let genPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let reduceBasePipeline: MTLComputePipelineState
    private let cleanupPipeline: MTLComputePipelineState
    private let plantPipeline: MTLComputePipelineState
    private let climatePipeline: MTLComputePipelineState
    private let sitePipeline: MTLComputePipelineState
    private let stampPipeline: MTLComputePipelineState
    private let scratch: [MTLTexture]
    // Max mips 1 and 2 of the plain heights for the camera's readback: the map's own mips from 1 up also bound the
    // level-1 columns, which the camera must not steer by.
    private let cpuMips: [MTLTexture]
    private var buffers: [Buffer]
    private var active = 0
    private var job: (index: Int, origin: SIMD2<Int32>, next: Int, t0: CFTimeInterval)?
    private var pending = false
    var synchronous = false

    var heightTex: MTLTexture { buffers[active].tex }
    var climateTex: MTLTexture { buffers[active].climate }
    var origin: SIMD2<Int32> { buffers[active].origin }
    var globalMax: Float { buffers[active].globalMax }
    var siteBuffer: MTLBuffer { buffers[active].sites }
    var structTex: MTLTexture { buffers[active].structTex }
    var sites: [SiteInfo] { buffers[active].siteList }

    init?(device: MTLDevice, library: MTLLibrary, seed: UInt32) {
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        }
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
        sd.usage = [.shaderRead, .shaderWrite]
        sd.storageMode = .private
        guard let gen = pipeline("generateHeight"), let red = pipeline("reduceMax"), let redBase = pipeline("reduceMaxBase"),
              let clean = pipeline("cleanupColumns"),
              let plant = pipeline("plantTrees"), let clim = pipeline("bakeClimate"), let place = pipeline("placeSites"), let stamp = pipeline("stampStructures"),
              let s0 = device.makeTexture(descriptor: sd), let s1 = device.makeTexture(descriptor: sd), let q = device.makeCommandQueue(),
              let a = Buffer(device: device), let b = Buffer(device: device) else { return nil }
        var mips: [MTLTexture] = []
        for l in 1...World.cpuLevel {
            let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: World.size >> l, height: World.size >> l, mipmapped: false)
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
        cleanupPipeline = clean
        plantPipeline = plant
        climatePipeline = clim
        sitePipeline = place
        stampPipeline = stamp
        scratch = [s0, s1]
        buffers = [a, b]
    }

    // Recentres the map when the camera drifts. The swap happens only once the new map is complete;
    // a synchronous fallback covers startup and the case where the camera outruns generation.
    func ensureCentered(on center: SIMD2<Double>) {
        let half = Double(World.size / 2)
        let cur = buffers[active]
        let local = center - SIMD2(Double(cur.origin.x), Double(cur.origin.y))
        let drift = max(abs(local.x - half), abs(local.y - half))
        let o = SIMD2<Int32>(Int32(floor((center.x - half) / 64)) * 64, Int32(floor((center.y - half) / 64)) * 64)
        if cur.origin.x == Int32.min || drift > 340 || (synchronous && drift > World.recentre) {
            if job != nil { advance(pieces: Int.max, wait: true); return }
            if pending { return } // the in-flight job will land momentarily
            job = (active, o, 0, CACurrentMediaTime())
            advance(pieces: Int.max, wait: true)
        } else if job != nil {
            advance(pieces: 1, wait: false)
        } else if drift > World.recentre && !pending {
            pending = true
            job = (1 - active, o, 0, CACurrentMediaTime())
            advance(pieces: 1, wait: false)
        }
    }

    private static let passes = 5
    private static var pieceCount: Int { passes * (size / bandRows) + 1 }

    // Encodes the next `pieces` pieces of the job: each pass (heights, cleanup, structures, trees, climate) a band of
    // rows at a time, finishing every band of a pass before the next pass reads its neighbours, then the max mips.
    // Sites are placed once, ahead of the first structure band: they read the cleaned map, and every band stamps them.
    private func advance(pieces: Int, wait: Bool) {
        guard var j = job, let cb = genQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        let buf = buffers[j.index]
        var info = SIMD4<Int32>(j.origin.x, j.origin.y, Int32(World.size), Int32(bitPattern: seed))
        let bands = World.size / World.bandRows, tg = MTLSize(width: 16, height: 16, depth: 1)
        var n = pieces
        while n > 0 && j.next < World.pieceCount {
            let pass = j.next / bands, band = j.next % bands
            if pass < World.passes {
                var row0 = UInt32(band * World.bandRows)
                switch pass {
                case 0:
                    enc.setComputePipelineState(genPipeline)
                    enc.setTexture(scratch[0], index: 0)
                case 1:
                    enc.setComputePipelineState(cleanupPipeline)
                    enc.setTexture(scratch[0], index: 0)
                    enc.setTexture(scratch[1], index: 1)
                case 2:
                    if band == 0 {
                        enc.setComputePipelineState(sitePipeline)
                        enc.setTexture(scratch[1], index: 0)
                        enc.setBuffer(buf.sites, offset: 0, index: 0)
                        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                        enc.dispatchThreads(MTLSize(width: World.siteGrid, height: World.siteGrid, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 6, height: 6, depth: 1))
                    }
                    // Stamping follows cleanup, which would strip lintel and eave gaps, and precedes planting, so
                    // trees avoid it. It writes into the climate texture, free until the climate pass: scratch 0 must
                    // keep the raw columns for the max mips.
                    enc.setComputePipelineState(stampPipeline)
                    enc.setTexture(scratch[1], index: 0)
                    enc.setTexture(buf.climate, index: 1)
                    enc.setTexture(buf.structTex, index: 2)
                    enc.setBuffer(buf.sites, offset: 0, index: 0)
                    enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
                    enc.setBytes(&row0, length: MemoryLayout<UInt32>.stride, index: 2)
                    enc.dispatchThreads(MTLSize(width: World.size, height: World.bandRows, depth: 1), threadsPerThreadgroup: tg)
                    j.next += 1
                    n -= 1
                    continue
                case 3:
                    enc.setComputePipelineState(plantPipeline)
                    enc.setTexture(buf.climate, index: 0)
                    enc.setTexture(buf.levels[0], index: 1)
                default:
                    enc.setComputePipelineState(climatePipeline)
                    enc.setTexture(buf.climate, index: 0)
                }
                enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
                enc.setBytes(&row0, length: MemoryLayout<UInt32>.stride, index: 1)
                enc.dispatchThreads(MTLSize(width: World.size, height: World.bandRows, depth: 1), threadsPerThreadgroup: tg)
            } else {
                enc.setComputePipelineState(reducePipeline)
                for (l, t) in cpuMips.enumerated() {
                    enc.setTexture(l == 0 ? buf.levels[0] : cpuMips[l - 1], index: 0)
                    enc.setTexture(t, index: 1)
                    enc.dispatchThreads(MTLSize(width: t.width, height: t.height, depth: 1), threadsPerThreadgroup: tg)
                }
                // Scratch 0 still holds the raw columns that level 1 averages.
                enc.setComputePipelineState(reduceBasePipeline)
                enc.setTexture(buf.levels[0], index: 0)
                enc.setTexture(scratch[0], index: 1)
                enc.setTexture(buf.levels[1], index: 2)
                let s1 = World.size >> 1
                enc.dispatchThreads(MTLSize(width: s1, height: s1, depth: 1), threadsPerThreadgroup: tg)
                enc.setComputePipelineState(reducePipeline)
                for l in 2..<buf.levels.count {
                    enc.setTexture(buf.levels[l - 1], index: 0)
                    enc.setTexture(buf.levels[l], index: 1)
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
        guard done else { cb.commit(); return }

        let index = j.index, o = j.origin, t0 = j.t0
        let finish = { [weak self] (heights: [Float], top: Float) in
            guard let self else { return }
            buf.origin = o
            buf.cpuHeights = heights
            buf.globalMax = top + 1
            buf.siteList = World.readSites(buf.sites)
            if World.logSites {
                var counts = [Int](repeating: 0, count: 7)
                for site in buf.siteList { counts[site.kind.rawValue] += 1 }
                if ProcessInfo.processInfo.environment["VR_SITES"] == "3" {
                    for site in buf.siteList { print("  site \(site.kind) \(Int(site.pos.x)),\(Int(site.pos.y)) r\(Int(site.radius)) lamps \(site.lamps.count)") }
                }
                print("sites at (\(o.x),\(o.y)): henge \(counts[1]) cairn \(counts[2]) ruins \(counts[3]) hamlet \(counts[4]) jetty \(counts[5]) camp \(counts[6]) total \(buf.siteList.count)")
            }
            self.active = index
            self.pending = false
            if Renderer.logTiming { print(String(format: "world regen %.1f ms (%@)", (CACurrentMediaTime() - t0) * 1000, wait ? "sync" : "async")) }
        }
        if wait {
            cb.commit()
            cb.waitUntilCompleted()
            let r = World.readBack(buf.tex, cpuMips.last!)
            finish(r.0, r.1)
        } else {
            cb.addCompletedHandler { [weak self] cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self?.pending = false }; return }
                guard let self else { return }
                let r = World.readBack(buf.tex, self.cpuMips.last!)
                DispatchQueue.main.async { finish(r.0, r.1) }
            }
            cb.commit()
        }
    }

    static let logSites = ProcessInfo.processInfo.environment["VR_SITES"] != nil

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
                let k = Int(v.w) % 16
                lamps.append(.init(pos: SIMD3(c.x + Double(v.x), Double(v.y), c.y + Double(v.z)), kind: k, rise: Double(Int(v.w) / 16), index: i * World.maxLamps + l))
            }
            list.append(SiteInfo(index: i, id: UInt32(bitPattern: head.w), kind: kind, pos: c + 0.5, radius: Double(info.x), lamps: lamps))
        }
        return list
    }

    private static func readBack(_ tex: MTLTexture, _ cpu: MTLTexture) -> ([Float], Float) {
        let s = World.size >> World.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s)
        cpu.getBytes(&raw, bytesPerRow: s * 2, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: 0)
        var top = [Float16](repeating: 0, count: 4)
        tex.getBytes(&top, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: tex.mipmapLevelCount - 1)
        return (raw.map { Float($0) }, Float(top[0]))
    }

    // Conservative (4x4 max) height in world coords.
    func height(x: Double, z: Double) -> Double? {
        let b = buffers[active]
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(b.origin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(b.origin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !b.cpuHeights.isEmpty else { return nil }
        return Double(b.cpuHeights[cz * s + cx])
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
    let cover: MTLTexture    // r8Unorm array: canopy cover
    let climate: MTLTexture  // rgba16Float array at half resolution
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

    init?(device: MTLDevice, library: MTLLibrary, seed: UInt32) {
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        }
        func array(_ format: MTLPixelFormat, _ n: Int, mips: Bool) -> MTLTexture? {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = format
            d.width = n
            d.height = n
            d.arrayLength = Clipmap.levels
            d.mipmapLevelCount = mips ? Int(log2(Double(n))) + 1 : 1
            d.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        guard let h = array(.r16Float, Clipmap.size, mips: true), let v = array(.r8Unorm, Clipmap.size, mips: false),
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

    // Forgets the windows, so the next encode rebuilds every level whole (VR_COARSE_FULL: checks strips against it).
    func invalidate() { lo = Array(repeating: nil, count: Clipmap.levels); job = nil }

    private func nextJob(_ center: SIMD2<Double>, active: Int) -> Job? {
        let n = Int32(Clipmap.size), step = Int32(Clipmap.snap)
        for i in 0..<active {
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

    // Encodes pending generation for levels 1...active: one piece, or with `drain` everything the camera position
    // needs. Levels past `active` are dropped, to be rebuilt whole once they are active again.
    func encode(into cb: MTLCommandBuffer, center: SIMD2<Double>, drain: Bool, active: Int) {
        for i in active..<Clipmap.levels { lo[i] = nil }
        if let j = job, j.level >= active { job = nil }
        // A level more than a strip behind the camera could soon be read past its window: catch it up at once. In
        // flight that never happens: at 15 voxels/s a level-1 strip falls due at most every 8.5 s and takes 17
        // frames. Only the outermost level, which the edge waits for, is ever built whole in pieces.
        let lagging = (0..<active).contains { i in
            let t = Clipmap.target(center, level: i + 1)
            return lo[i].map { abs($0.x - t.x) > Int32(Clipmap.snap) || abs($0.y - t.y) > Int32(Clipmap.snap) } ?? (i < active - 1)
        }
        var budget = drain || lagging ? Int.max : 1
        if job == nil { job = nextJob(center, active: active) }
        guard job != nil, let enc = cb.makeComputeCommandEncoder() else { return }
        defer { enc.endEncoding() }
        while budget > 0 {
            if job == nil { job = nextJob(center, active: active) }
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
            enc.setBytes(&region, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
            enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
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
            if Renderer.logTiming { print("coarse level \(j.level + 1) window \(j.newLo) (\(j.w)x\(j.h))") }
            job = nil
        }
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scenePipeline: MTLRenderPipelineState
    let postPipeline: MTLRenderPipelineState
    let taaPipeline: MTLRenderPipelineState
    private let cloudPipeline: MTLComputePipelineState
    // Shadow-height maps (see bakeShadowHeight in Shaders/Saver.metal): near and far cascades, a band rebaked per frame.
    private let shadowPipeline: MTLComputePipelineState
    private let shadowMaps: [MTLTexture]
    static let shadowN = 1024, shadowCell = 8.0
    private static let shadowBands = [16, 32]
    private var shadowBand = [0, 0]
    private var shadowValid = false
    private var shadowFrom: (cam: SIMD2<Double>, key: SIMD3<Double>)?
    // The fine cloud field only drifts with the wind, so it is baked once around the camera (in drifted
    // coordinates) and re-baked when the camera nears its edge.
    private var cloudTex: MTLTexture?
    private var cloudCenter: SIMD2<Double>?
    static let cloudTexels = 2048, cloudTexel = 4.0
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
    static let ditherCells = 864.0
    private var historyIndex = 0
    private var prevCamera: (pos: SIMD3<Double>, fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)?
    let world: World
    let clipmap: Clipmap
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
    private var simTime: Double = 0
    private var lastFrame: CFTimeInterval?
    private var frameIndex: Float = 0
    private var gpuTimes: [Double] = []
    private var sceneSize = (width: 0, height: 0)
    private(set) var aspect: Float = 16.0 / 10.0
    static let logTiming = ProcessInfo.processInfo.environment["VR_TIMING"] != nil
    static let fullCoarse = ProcessInfo.processInfo.environment["VR_COARSE_FULL"] != nil
    // VR_CHUNKED: snapshots update the coarse rings a chunk per frame, as live.
    static let chunkedCoarse = ProcessInfo.processInfo.environment["VR_CHUNKED"] != nil
    // VR_LODSHIFT=v moves the LOD rings' centre v voxels ahead of the camera: frames differing only in it isolate pops.
    static let lodShift = Float(ProcessInfo.processInfo.environment["VR_LODSHIFT"] ?? "") ?? 0
    fileprivate var exposure: Double = 1.5
    // Holds the time of day fixed for snapshots, so exposure and the shot director see the rendered sun.
    var pinnedPhase: Double?
    var renderScale: Double

    // Cover drifts between clear and broken skies over tens of minutes.
    var cloudCover: Double {
        0.52 + 0.12 * sin(simTime * 0.0021 + 1.0) + 0.06 * sin(simTime * 0.0057 + 4.0)
    }

    init?(device: MTLDevice, outputFormat: MTLPixelFormat, seed: UInt32, screenIndex: Int, battery: Bool) {
        guard let queue = device.makeCommandQueue(), let lib = GPU.library(device: device) else { return nil }
        let scene = MTLRenderPipelineDescriptor()
        scene.vertexFunction = lib.makeFunction(name: "fullscreenVertex")
        scene.fragmentFunction = lib.makeFunction(name: "sceneFragment")
        scene.colorAttachments[0].pixelFormat = .rgba16Float
        let post = MTLRenderPipelineDescriptor()
        post.vertexFunction = lib.makeFunction(name: "fullscreenVertex")
        post.fragmentFunction = lib.makeFunction(name: "postFragment")
        post.colorAttachments[0].pixelFormat = outputFormat
        let taa = MTLRenderPipelineDescriptor()
        taa.vertexFunction = lib.makeFunction(name: "fullscreenVertex")
        taa.fragmentFunction = lib.makeFunction(name: "taaFragment")
        taa.colorAttachments[0].pixelFormat = .rgba16Float
        guard let sp = try? device.makeRenderPipelineState(descriptor: scene),
              let tp = try? device.makeRenderPipelineState(descriptor: taa),
              let pp = try? device.makeRenderPipelineState(descriptor: post),
              let cf = lib.makeFunction(name: "cloudField").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let shP = lib.makeFunction(name: "bakeShadowHeight").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let lampP = lib.makeFunction(name: "updateLamps").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let lightP = lib.makeFunction(name: "bakeLights").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let lpos = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let lstate = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let world = World(device: device, library: lib, seed: seed),
              let clipmap = Clipmap(device: device, library: lib, seed: seed) else { return nil }
        self.device = device
        self.queue = queue
        scenePipeline = sp
        postPipeline = pp
        taaPipeline = tp
        cloudPipeline = cf
        shadowPipeline = shP
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: Renderer.shadowN, height: Renderer.shadowN, mipmapped: false)
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
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919)
        wildlife = Wildlife(seed: seed &+ UInt32(screenIndex) &* 7919)
        birdBuffers = (0..<3).compactMap { _ in device.makeBuffer(length: Wildlife.bufferLength, options: .storageModeShared) }
        guard birdBuffers.count == 3 else { return nil }
        self.battery = battery
        renderScale = battery ? 0.7 : 1.0
        edgeRadius = Clipmap.handoff(battery ? Clipmap.levels - 1 : Clipmap.levels)
        super.init()
        camera.heightAt = { [unowned world] x, z in world.height(x: x, z: z) }
        camera.sitesAt = { [unowned world] in world.sites }
        wildlife.heightAt = camera.heightAt
        world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
        camera.placeAtStart()
        // The coarse rings are built ahead of the first frame on the same queue, which orders that frame after them;
        // nothing waits on the CPU. Afterwards only strips are rewritten.
        if let cb = queue.makeCommandBuffer() {
            let t0 = CACurrentMediaTime()
            clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: true, active: activeLevels)
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

    convenience init?(view: MTKView, screen: NSScreen, seed: UInt32, screenIndex: Int, battery: Bool) {
        let backingScale = screen.backingScaleFactor
        guard let device = view.device else { return nil }
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
        self.init(device: device, outputFormat: view.colorPixelFormat, seed: seed, screenIndex: screenIndex, battery: battery)
        view.autoResizeDrawable = false
        view.drawableSize = CGSize(width: view.bounds.width * backingScale, height: view.bounds.height * backingScale)
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
            wildlife.step(dt: step, camera: camera, sunY: clock.sunDirection.y, phase: clock.phase)
            if Renderer.logTiming, let g = world.height(x: camera.position.x, z: camera.position.z), camera.position.y < g + 2 {
                print(String(format: "camera low: y=%.1f ground=%.1f", camera.position.y, g))
            }
            clock.advance(step)
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
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

    func makeFrame(size: CGSize) -> Frame {
        var f = Frame()
        let b = camera.basis
        let o = SIMD3(Double(world.origin.x), 0, Double(world.origin.y))
        let p = camera.position - o
        f.camPos = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(tan(25.0 * .pi / 180)))
        f.camFwd = SIMD4(SIMD3<Float>(b.fwd), Float(size.width / size.height))
        f.camRight = SIMD4(SIMD3<Float>(b.right), Float(simTime))
        f.camUp = SIMD4(SIMD3<Float>(b.up), battery ? 0 : 1)
        f.sunDir = SIMD4(SIMD3<Float>(clock.sunDirection), Float(clock.phase))
        f.resolution = SIMD4(Float(size.width), Float(size.height), Float(1 / size.width), Float(1 / size.height))
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
        // The retro dither shows one flat colour per cell, so the temporal history only needs the cell grid,
        // and the scene renders at (a fraction of) it.
        let cell = max(1.0, (outputSize.width / Renderer.ditherCells).rounded())
        let ow = max(16, Int((outputSize.width / cell).rounded())), oh = max(16, Int((outputSize.height / cell).rounded()))
        let w = max(16, Int(Double(ow) * renderScale)), h = max(16, Int(Double(oh) * renderScale))
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
        var f = makeFrame(size: CGSize(width: w, height: h))
        frameIndex = (frameIndex + 1).truncatingRemainder(dividingBy: 4096)
        bakeClouds(into: cb, frame: &f)
        if Renderer.fullCoarse { clipmap.invalidate() }
        clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: world.synchronous && !Renderer.chunkedCoarse,
                       active: activeLevels)
        bakeLamps(into: cb, frame: &f)
        bakeShadows(into: cb, frame: &f)
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
            enc.setRenderPipelineState(scenePipeline)
            enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
            enc.setFragmentBuffer(birdBuf, offset: 0, index: 5)
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
            enc.setFragmentBuffer(lampPos, offset: 0, index: 2)
            enc.setFragmentBuffer(lampState, offset: 0, index: 3)
            enc.setFragmentBytes(&hazeRefs, length: 128, index: 4)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
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
            for l in site.lamps where l.kind != 3 {
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
                let v = l.pos + SIMD3(0, l.rise, 0) - cam
                let d = simd_length(v)
                if d < 600, d < 40 || simd_dot(v, fwd) > -0.2 * d { near.append((d, l.index)) }
            }
        }
        near.sort { $0.0 < $1.0 }
        smokeCount = min(near.count, Renderer.smokePlumes)
        for i in 0..<smokeCount { hazeRefs[16 + i] = Int32(near[i].1) }
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
    private func bakeShadows(into cb: MTLCommandBuffer, frame f: inout Frame) {
        let (key, casts) = Renderer.keyLight(clock.sunDirection)
        let levels = (1...Clipmap.levels).prefix { clipmap.ready($0) }.count
        guard casts, levels > 0, let enc = cb.makeComputeCommandEncoder() else { shadowValid = false; f.shadow.x = 0; return }
        let cam = SIMD2(camera.position.x, camera.position.z)
        let jumped = shadowFrom.map { simd_length($0.cam - cam) > 300 || simd_dot($0.key, key) < 0.9998 } ?? true
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
                var b = ShadowBake(key: SIMD4(Float(key.x), Float(key.y), Float(key.z), Float(max(0.8 * 160 * cosE, cell))),
                                   cam: SIMD4(Float(cam.x), Float(cam.y), Float(cell), Float(levels)),
                                   rows: SIMD4(Int32(shadowBand[c] * rows), Int32(floor(cam.x / cell)), Int32(floor(cam.y / cell)), 0))
                enc.setTexture(shadowMaps[c], index: 0)
                enc.setBytes(&b, length: MemoryLayout<ShadowBake>.stride, index: 0)
                enc.dispatchThreads(MTLSize(width: Renderer.shadowN, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                shadowBand[c] = (shadowBand[c] + 1) % bands
                if c == 0 && shadowBand[0] == 0 { shadowFrom = (cam, key) }
            }
        }
        enc.endEncoding()
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
        let drift = SIMD2(cos(0.6), sin(0.6)) * 1.6 * Double(Float(simTime))
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
        encode(into: cb, output: drawable.texture, outputSize: view.drawableSize)
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
    static func run(settings: Settings, path: String) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let r = Renderer(device: device, outputFormat: .rgba8Unorm, seed: settings.seed == 0 ? 1 : settings.seed,
                               screenIndex: 0, battery: settings.forceBattery) else {
            FileHandle.standardError.write("renderer init failed\n".data(using: .utf8)!)
            return
        }
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
        r.headroom = settings.headroom ?? 0
        let env = ProcessInfo.processInfo.environment
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
        // VR_CAM=x,z,height,yawDeg,pitchDeg[,abs]: holds the camera at world (x, z), `height` above the ground there
        // (absolute when abs is 1).
        let camSpec = env["VR_CAM"].map { $0.split(separator: ",").compactMap { Double($0) } }
        if let v = camSpec, v.count >= 5 {
            r.camera.freeze(position: SIMD3(v[0], 400, v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            r.camera.pitchOverride = nil
        }
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            r.simulate(t - simulated)
            if let v = camSpec, v.count >= 5 {
                let g = (v.count > 5 && v[5] == 1) ? 0 : max(r.world.height(x: v[0], z: v[1]) ?? 20, 19.85)
                r.camera.freeze(position: SIMD3(v[0], g + v[2], v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            }
            if let phase = settings.phase { r.clock.phase = phase }
            simulated = t
            if let d = Double(ProcessInfo.processInfo.environment["VR_BIRDTEST"] ?? "") { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = Double(ProcessInfo.processInfo.environment["VR_MURMUR"] ?? "") { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            let frames = env["VR_MOTION"] != nil ? Int(env["VR_MOTION_FRAMES"] ?? "") ?? 48 : 12
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
                if Renderer.logTiming { print(String(format: "frame %d gpu %.2f ms%@", i, times.last!, r.clipmap.idle ? "" : " (coarse job pending)")) }
                if i >= frames - dump {
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    write(bytes: bytes, width: w, height: h, to: out.replacingOccurrences(of: ".png", with: "_f\(i - frames + dump).png"))
                }
            }
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            write(bytes: bytes, width: w, height: h, to: out)
            let seen = r.wildlife.inView(camera: r.camera, aspect: Double(w) / Double(h), tanHalf: tan(25.0 * .pi / 180))
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) yaw=%.1f pitch=%.1f gpu=%.2fms mem=%.0fMB birds=%d flocks=%d near=%.0f %@ -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.camera.yaw * 180 / .pi, r.camera.pitch * 180 / .pi, times.min()!,
                         Double(device.currentAllocatedSize) / 1048576, seen.count, seen.flocks, seen.nearest.isFinite ? seen.nearest : -1, Snapshot.sitesInView(r), out))
        }
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

    static func write(bytes: [UInt8], width: Int, height: Int, to path: String) {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: cs,
                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                          decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let rep = NSBitmapImageRep(cgImage: img)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
    }
}
