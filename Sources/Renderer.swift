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

    mutating func advance(_ dt: Double) {
        let y = SkyClock.elevation(phase)
        let s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
        // Linger in golden and blue hour, hurry through midday and deep night.
        let warp = y < -0.25 ? 5.0 : 0.5 + 1.5 * s * s * (3 - 2 * s)
        phase += dt / SkyClock.dayLength * warp * 1.05
        if phase > 1 { phase -= 1 }
    }
}

final class World {
    static let size = 2048
    static let cpuLevel = 2

    // One generated map: the column texture with its max-mip chain, and the CPU copy the camera steers by.
    private final class Buffer {
        let tex: MTLTexture
        let levels: [MTLTexture]
        var origin = SIMD2<Int32>(Int32.min, Int32.min)
        var globalMax: Float = 255
        var cpuHeights: [Float] = []

        init?(device: MTLDevice) {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: true)
            td.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
            td.storageMode = .shared
            guard let t = device.makeTexture(descriptor: td) else { return nil }
            tex = t
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
    private let scratch: MTLTexture
    private var buffers: [Buffer]
    private var active = 0
    private var pending = false
    var synchronous = false

    var heightTex: MTLTexture { buffers[active].tex }
    var origin: SIMD2<Int32> { buffers[active].origin }
    var globalMax: Float { buffers[active].globalMax }

    init?(device: MTLDevice, library: MTLLibrary, seed: UInt32) {
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        }
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: false)
        sd.usage = [.shaderRead, .shaderWrite]
        sd.storageMode = .private
        guard let gen = pipeline("generateHeight"), let red = pipeline("reduceMax"), let clean = pipeline("cleanupColumns"),
              let scratchTex = device.makeTexture(descriptor: sd), let q = device.makeCommandQueue(),
              let a = Buffer(device: device), let b = Buffer(device: device) else { return nil }
        self.device = device
        self.seed = seed
        genQueue = q
        genPipeline = gen
        reducePipeline = red
        cleanupPipeline = clean
        scratch = scratchTex
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
        enc.setTexture(scratch, index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(cleanupPipeline)
        enc.setTexture(scratch, index: 0)
        enc.setTexture(buf.levels[0], index: 1)
        enc.dispatchThreads(full, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(reducePipeline)
        for l in 1..<buf.levels.count {
            enc.setTexture(buf.levels[l - 1], index: 0)
            enc.setTexture(buf.levels[l], index: 1)
            let s = max(1, World.size >> l)
            enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
        }
        enc.endEncoding()

        let t0 = CACurrentMediaTime()
        let finish = { [weak self] (heights: [Float], top: Float) in
            guard let self else { return }
            buf.origin = o
            buf.cpuHeights = heights
            buf.globalMax = top + 1
            self.active = index
            self.pending = false
            if Renderer.logTiming { print(String(format: "world regen %.1f ms (%@)", (CACurrentMediaTime() - t0) * 1000, wait ? "sync" : "async")) }
        }
        if wait {
            cb.commit()
            cb.waitUntilCompleted()
            let r = World.readBack(buf.tex)
            finish(r.0, r.1)
        } else {
            cb.addCompletedHandler { [weak self] cb in
                guard cb.status == .completed else { DispatchQueue.main.async { self?.pending = false }; return }
                let r = World.readBack(buf.tex)
                DispatchQueue.main.async { finish(r.0, r.1) }
            }
            cb.commit()
        }
    }

    private static func readBack(_ tex: MTLTexture) -> ([Float], Float) {
        let s = World.size >> World.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 4)
        tex.getBytes(&raw, bytesPerRow: s * 8, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: World.cpuLevel)
        var top = [Float16](repeating: 0, count: 4)
        tex.getBytes(&top, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: tex.mipmapLevelCount - 1)
        return ((0..<(s * s)).map { Float(raw[$0 * 4]) }, Float(top[0]))
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

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scenePipeline: MTLRenderPipelineState
    let postPipeline: MTLRenderPipelineState
    let taaPipeline: MTLRenderPipelineState
    private var history: [MTLTexture] = []
    private var historyIndex = 0
    private var prevCamera: (pos: SIMD3<Double>, fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)?
    let world: World
    let camera: FlightCamera
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
              let world = World(device: device, library: lib, seed: seed) else { return nil }
        self.device = device
        self.queue = queue
        scenePipeline = sp
        postPipeline = pp
        taaPipeline = tp
        self.world = world
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919)
        self.battery = battery
        renderScale = battery ? 0.35 : 0.5
        super.init()
        camera.heightAt = { [unowned world] x, z in world.height(x: x, z: z) }
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
        view.delegate = self
    }

    func setBattery(_ b: Bool, view: MTKView) {
        battery = b
        renderScale = b ? 0.35 : 0.5
        view.preferredFramesPerSecond = b ? 30 : 60
    }

    // Advances the simulation; large steps are subdivided so steering stays stable.
    func simulate(_ dt: Double) {
        var remaining = dt * timeScale
        while remaining > 0 {
            let step = min(remaining, 1.0 / 30.0)
            if let p = pinnedPhase { clock.phase = p }
            camera.sun = clock.sunDirection
            camera.step(dt: step)
            if Renderer.logTiming, let g = world.height(x: camera.position.x, z: camera.position.z), camera.position.y < g + 2 {
                print(String(format: "camera low: y=%.1f ground=%.1f", camera.position.y, g))
            }
            clock.advance(step)
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
            remaining -= step
        }
        simTime += dt * timeScale
        let e = clock.sunDirection.y
        let target = e > 0 ? 0.62 + 1.0 * max(0, 0.3 - e) / 0.3 - 0.10 * min(1, max(0, (e - 0.3) / 0.2)) : (e > -0.1 ? 2.02 + 0.4 * (-e / 0.1) : 1.6 + 0.4 * max(0, 0.3 + e) / 0.2)
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
        if let pc = prevCamera {
            let pp = pc.pos - o
            f.prevPos = SIMD4(Float(pp.x), Float(pp.y), Float(pp.z), 1)
            f.prevFwd = SIMD4(SIMD3<Float>(pc.fwd), 0)
            f.prevRight = SIMD4(SIMD3<Float>(pc.right), 0)
            f.prevUp = SIMD4(SIMD3<Float>(pc.up), 0)
        }
        // Halton(2,3) subpixel jitter, accumulated by the temporal resolve.
        let k = Int(frameIndex) % 8 + 1
        func halton(_ i: Int, _ b: Int) -> Double { var f = 1.0, r = 0.0, i = i; while i > 0 { f /= Double(b); r += f * Double(i % b); i /= b }; return r }
        f.jitter = SIMD4(Float((halton(k, 2) - 0.5) * 2 / size.width), Float((halton(k, 3) - 0.5) * 2 / size.height), 0, 0)
        return f
    }

    func encode(into cb: MTLCommandBuffer, output: MTLTexture, outputSize: CGSize) {
        let w = max(16, Int(outputSize.width * renderScale)), h = max(16, Int(outputSize.height * renderScale))
        if sceneTex?.width != w || sceneTex?.height != h {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]
            td.storageMode = .private
            sceneTex = device.makeTexture(descriptor: td)
            history = (0..<2).compactMap { _ in device.makeTexture(descriptor: td) }
            prevCamera = nil
        }
        guard let sceneTex else { return }
        sceneSize = (w, h)
        var f = makeFrame(size: CGSize(width: w, height: h))
        frameIndex = (frameIndex + 1).truncatingRemainder(dividingBy: 4096)
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
            enc.setFragmentTexture(world.heightTex, index: 0)
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
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            r.simulate(t - simulated)
            if let phase = settings.phase { r.clock.phase = phase }
            simulated = t
            var times: [Double] = []
            for _ in 0..<12 {
                let cb = r.queue.makeCommandBuffer()!
                let t0 = CACurrentMediaTime()
                r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                cb.commit()
                cb.waitUntilCompleted()
                times.append((CACurrentMediaTime() - t0) * 1000)
            }
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            let out = settings.screenshotTimes.count > 1 ? path.replacingOccurrences(of: ".png", with: "_t\(Int(t)).png") : path
            write(bytes: bytes, width: w, height: h, to: out)
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) gpu=%.1fms -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, times.min()!, out))
        }
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
