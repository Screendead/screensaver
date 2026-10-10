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
        var cpuFloors: [Float] = []
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
    private let cleanupPipeline: MTLComputePipelineState
    private let plantPipeline: MTLComputePipelineState
    private let climatePipeline: MTLComputePipelineState
    private let sitePipeline: MTLComputePipelineState
    private let stampPipeline: MTLComputePipelineState
    private let scratch: [MTLTexture]
    private var buffers: [Buffer]
    private var active = 0
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
        guard let gen = pipeline("generateHeight"), let red = pipeline("reduceMax"), let clean = pipeline("cleanupColumns"),
              let plant = pipeline("plantTrees"), let clim = pipeline("bakeClimate"), let place = pipeline("placeSites"), let stamp = pipeline("stampStructures"),
              let s0 = device.makeTexture(descriptor: sd), let s1 = device.makeTexture(descriptor: sd), let q = device.makeCommandQueue(),
              let a = Buffer(device: device), let b = Buffer(device: device) else { return nil }
        self.device = device
        self.seed = seed
        genQueue = q
        genPipeline = gen
        reducePipeline = red
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
        if cur.origin.x == Int32.min || drift > 340 || (synchronous && drift > 220) {
            if pending { return } // the in-flight job will land momentarily
            generate(into: active, origin: o, wait: true)
        } else if drift > 220 && !pending {
            pending = true
            generate(into: 1 - active, origin: o, wait: false)
        }
    }

    private func generate(into index: Int, origin o: SIMD2<Int32>, wait: Bool) {
        let buf = buffers[index]
        guard let cb = genQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { pending = false; return }
        var info = SIMD4<Int32>(o.x, o.y, Int32(World.size), Int32(bitPattern: seed))
        let full = MTLSize(width: World.size, height: World.size, depth: 1), tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.setComputePipelineState(genPipeline)
        enc.setTexture(scratch[0], index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(cleanupPipeline)
        enc.setTexture(scratch[0], index: 0)
        enc.setTexture(scratch[1], index: 1)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(sitePipeline)
        enc.setTexture(scratch[1], index: 0)
        enc.setBuffer(buf.sites, offset: 0, index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: World.siteGrid, height: World.siteGrid, depth: 1), threadsPerThreadgroup: MTLSize(width: 6, height: 6, depth: 1))
        // Stamping follows cleanup, which would strip lintel and eave gaps, and precedes planting, so trees avoid it.
        enc.setComputePipelineState(stampPipeline)
        enc.setTexture(scratch[1], index: 0)
        enc.setTexture(scratch[0], index: 1)
        enc.setTexture(buf.structTex, index: 2)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
        enc.setComputePipelineState(plantPipeline)
        enc.setTexture(scratch[0], index: 0)
        enc.setTexture(buf.levels[0], index: 1)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(climatePipeline)
        enc.setTexture(buf.climate, index: 0)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(reducePipeline)
        for l in 1..<buf.levels.count {
            var fromColumns: Int32 = l == 1 ? 1 : 0
            enc.setBytes(&fromColumns, length: 4, index: 0)
            enc.setTexture(buf.levels[l - 1], index: 0)
            enc.setTexture(buf.levels[l], index: 1)
            let s = max(1, World.size >> l)
            enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
        }
        enc.endEncoding()

        let t0 = CACurrentMediaTime()
        let finish = { [weak self] (heights: [Float], floors: [Float], top: Float) in
            guard let self else { return }
            buf.origin = o
            buf.cpuHeights = heights
            buf.cpuFloors = floors
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
            let r = World.readBack(buf.tex)
            finish(r.0, r.1, r.2)
        } else {
            cb.addCompletedHandler { [weak self] cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self?.pending = false }; return }
                let r = World.readBack(buf.tex)
                DispatchQueue.main.async { finish(r.0, r.1, r.2) }
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

    private static func readBack(_ tex: MTLTexture) -> ([Float], [Float], Float) {
        let s = World.size >> World.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 4)
        tex.getBytes(&raw, bytesPerRow: s * 8, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: World.cpuLevel)
        var top = [Float16](repeating: 0, count: 4)
        tex.getBytes(&top, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: tex.mipmapLevelCount - 1)
        return ((0..<(s * s)).map { Float(raw[$0 * 4]) }, (0..<(s * s)).map { Float(raw[$0 * 4 + 1]) }, Float(top[0]))
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

    // Lower bound in world coords: every column of the 4x4 cell is solid from the ground up to at least this.
    func solidFloor(x: Double, z: Double) -> Double? {
        let b = buffers[active]
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(b.origin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(b.origin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !b.cpuFloors.isEmpty else { return nil }
        return Double(b.cpuFloors[cz * s + cx])
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scenePipeline: MTLRenderPipelineState
    let postPipeline: MTLRenderPipelineState
    let taaPipeline: MTLRenderPipelineState
    private let cloudPipeline: MTLComputePipelineState
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
    let camera: FlightCamera
    let wildlife: Wildlife
    // Ring of bird buffers, one per frame in flight; the semaphore keeps the CPU from overwriting one the GPU reads.
    private var birdBuffers: [MTLBuffer] = []
    private var birdIndex = 0
    private let birdSlots = DispatchSemaphore(value: 3)
    var clock = SkyClock()
    private(set) var battery: Bool
    var timeScale: Double = 1
    var debugMode: Float = 0
    var headroom: Float = 0
    private var sceneTex: MTLTexture?
    private var simTime: Double = 0
    private var lastFrame: CFTimeInterval?
    private var frameIndex: Float = 0
    private var gpuTimes: [Double] = []
    private var sceneSize = (width: 0, height: 0)
    let popLog: PopLog? = PopLog.enabled ? PopLog() : nil
    private(set) var aspect: Float = 16.0 / 10.0
    static let logTiming = ProcessInfo.processInfo.environment["VR_TIMING"] != nil
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
              let lampP = lib.makeFunction(name: "updateLamps").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let lightP = lib.makeFunction(name: "bakeLights").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let lpos = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let lstate = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let world = World(device: device, library: lib, seed: seed) else { return nil }
        self.device = device
        self.queue = queue
        scenePipeline = sp
        postPipeline = pp
        taaPipeline = tp
        cloudPipeline = cf
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
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919)
        wildlife = Wildlife(seed: seed &+ UInt32(screenIndex) &* 7919)
        birdBuffers = (0..<3).compactMap { _ in device.makeBuffer(length: Wildlife.bufferLength, options: .storageModeShared) }
        guard birdBuffers.count == 3 else { return nil }
        self.battery = battery
        renderScale = battery ? 0.7 : 1.0
        super.init()
        camera.heightAt = { [unowned world] x, z in world.height(x: x, z: z) }
        camera.sitesAt = { [unowned world] in world.sites }
        wildlife.heightAt = camera.heightAt
        world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
        camera.placeAtStart()
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
        prepare(outputSize: view.drawableSize)
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
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
            remaining -= step
        }
        simTime += dt * timeScale
        let e = clock.sunDirection.y
        let target = e > 0 ? 0.62 + 1.0 * max(0, 0.3 - e) / 0.3 + 0.08 * min(1, max(0, (e - 0.3) / 0.2)) : (e > -0.1 ? 2.02 + 0.4 * (-e / 0.1) : 1.6 + 0.4 * max(0, 0.3 + e) / 0.2)
        exposure += (target - exposure) * min(1, dt * timeScale * 0.2)
    }

    // What a frame rendered now could show; the frustum follows the scene size the last encode (or prepare) set.
    func sight() -> Sight {
        let b = camera.basis
        let tanV = tan(25.0 * .pi / 180)
        let o = world.origin
        return Sight(pos: camera.position, fwd: b.fwd, right: b.right, up: b.up, tanV: tanV, aspect: Double(aspect),
                     pixelAngle: 2 * tanV / Double(max(sceneSize.height, 1)), sunY: clock.sunDirection.y,
                     mapOrigin: SIMD2(Double(o.x), Double(o.y)), mapSize: Double(World.size),
                     floorAt: { [unowned world] x, z in world.solidFloor(x: x, z: z) })
    }

    // Scene and history sizes for an output size: the retro dither shows one flat colour per cell, so the temporal
    // history only needs the cell grid, and the scene renders at (a fraction of) it.
    private func sizes(_ outputSize: CGSize) -> (ow: Int, oh: Int, w: Int, h: Int) {
        let cell = max(1.0, (outputSize.width / Renderer.ditherCells).rounded())
        let ow = max(16, Int((outputSize.width / cell).rounded())), oh = max(16, Int((outputSize.height / cell).rounded()))
        return (ow, oh, max(16, Int(Double(ow) * renderScale)), max(16, Int(Double(oh) * renderScale)))
    }

    // Sets the frustum the simulation sees before the first frame is encoded.
    func prepare(outputSize: CGSize) {
        let s = sizes(outputSize)
        sceneSize = (s.w, s.h)
        aspect = Float(s.w) / Float(s.h)
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
        var f = makeFrame(size: CGSize(width: w, height: h))
        frameIndex = (frameIndex + 1).truncatingRemainder(dividingBy: 4096)
        bakeClouds(into: cb, frame: &f)
        bakeLamps(into: cb, frame: &f)
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
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            r.simulate(t - simulated)
            if let phase = settings.phase { r.clock.phase = phase }
            simulated = t
            if let d = Double(ProcessInfo.processInfo.environment["VR_BIRDTEST"] ?? "") { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = Double(ProcessInfo.processInfo.environment["VR_MURMUR"] ?? "") { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            let frames = env["VR_MOTION"] != nil ? 48 : 12
            // VR_FRAMES=n also writes each of the last n frames, for measuring frame-to-frame crawl.
            let dump = min(Int(env["VR_FRAMES"] ?? "") ?? 0, frames)
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
            let out = settings.screenshotTimes.count > 1 ? path.replacingOccurrences(of: ".png", with: "_t\(Int(t)).png") : path
            for i in 0..<frames {
                if env["VR_MOTION"] != nil { r.simulate(1.0 / 60) }
                let cb = r.queue.makeCommandBuffer()!
                r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                cb.commit()
                cb.waitUntilCompleted()
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if i >= frames - dump {
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    write(bytes: bytes, width: w, height: h, to: out.replacingOccurrences(of: ".png", with: "_f\(i - frames + dump).png"))
                }
            }
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            write(bytes: bytes, width: w, height: h, to: out)
            let seen = r.wildlife.inView(camera: r.camera, aspect: Double(w) / Double(h), tanHalf: tan(25.0 * .pi / 180))
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) gpu=%.1fms birds=%d flocks=%d near=%.0f %@ -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, times.min()!,
                         seen.count, seen.flocks, seen.nearest.isFinite ? seen.nearest : -1, Snapshot.sitesInView(r), out))
        }
        if let log = r.popLog { print(log.summary(r.wildlife)) }
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
