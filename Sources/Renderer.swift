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

    var sunDirection: SIMD3<Double> {
        let el = asin(sin(2 * .pi * phase) * 0.72)
        let az = 2 * .pi * phase * 0.5 + 0.9
        return SIMD3(cos(el) * sin(az), sin(el), cos(el) * cos(az))
    }

    mutating func advance(_ dt: Double) {
        let e = abs(sin(2 * .pi * phase) * 0.72)
        let s = min(max((e - 0.22) / 0.35, 0), 1)
        let warp = 0.6 + 0.9 * s * s * (3 - 2 * s)
        phase += dt / SkyClock.dayLength * warp * 1.05
        if phase > 1 { phase -= 1 }
    }
}

final class World {
    static let size = 2048
    let device: MTLDevice
    let heightTex: MTLTexture
    private let levelViews: [MTLTexture]
    private let genPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private(set) var origin = SIMD2<Int32>(Int32.min, Int32.min)
    private(set) var globalMax: Float = 255
    private var cpuLevel2: [Float] = []
    let seed: UInt32
    static let cpuLevel = 2

    init?(device: MTLDevice, library: MTLLibrary, seed: UInt32) {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: World.size, height: World.size, mipmapped: true)
        td.usage = [.shaderRead, .shaderWrite, .pixelFormatView]
        td.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: td),
              let gen = library.makeFunction(name: "generateHeight").flatMap({ try? device.makeComputePipelineState(function: $0) }),
              let red = library.makeFunction(name: "reduceMax").flatMap({ try? device.makeComputePipelineState(function: $0) })
        else { return nil }
        self.device = device
        heightTex = tex
        levelViews = (0..<tex.mipmapLevelCount).map {
            tex.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)!
        }
        genPipeline = gen
        reducePipeline = red
        self.seed = seed
    }

    // Recentres the map on `center` when the camera drifts too far; synchronous, a few ms.
    func ensureCentered(on center: SIMD2<Double>, queue: MTLCommandQueue) {
        let half = Double(World.size / 2)
        let local = center - SIMD2(Double(origin.x), Double(origin.y))
        if origin.x != Int32.min && abs(local.x - half) < 300 && abs(local.y - half) < 300 { return }
        let o = SIMD2<Int32>(Int32(floor((center.x - half) / 64)) * 64, Int32(floor((center.y - half) / 64)) * 64)
        regenerate(origin: o, queue: queue)
    }

    func regenerate(origin o: SIMD2<Int32>, queue: MTLCommandQueue) {
        origin = o
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        var info = SIMD4<Int32>(o.x, o.y, Int32(World.size), Int32(bitPattern: seed))
        enc.setComputePipelineState(genPipeline)
        enc.setTexture(levelViews[0], index: 0)
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: World.size, height: World.size, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.setComputePipelineState(reducePipeline)
        for l in 1..<levelViews.count {
            enc.setTexture(levelViews[l - 1], index: 0)
            enc.setTexture(levelViews[l], index: 1)
            let s = max(1, World.size >> l)
            enc.dispatchThreads(MTLSize(width: s, height: s, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, s), height: min(16, s), depth: 1))
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let s = World.size >> World.cpuLevel
        var raw = [Float16](repeating: 0, count: s * s * 4)
        heightTex.getBytes(&raw, bytesPerRow: s * 8, from: MTLRegionMake2D(0, 0, s, s), mipmapLevel: World.cpuLevel)
        cpuLevel2 = (0..<(s * s)).map { Float(raw[$0 * 4]) }
        var top = [Float16](repeating: 0, count: 4)
        heightTex.getBytes(&top, bytesPerRow: 8, from: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: heightTex.mipmapLevelCount - 1)
        globalMax = Float(top[0]) + 1
    }

    // Conservative (4x4 max) height in world coords.
    func height(x: Double, z: Double) -> Double? {
        let s = World.size >> World.cpuLevel
        let cx = Int(floor((x - Double(origin.x)) / Double(1 << World.cpuLevel)))
        let cz = Int(floor((z - Double(origin.y)) / Double(1 << World.cpuLevel)))
        guard cx >= 0, cz >= 0, cx < s, cz < s, !cpuLevel2.isEmpty else { return nil }
        return Double(cpuLevel2[cz * s + cx])
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let scenePipeline: MTLRenderPipelineState
    let postPipeline: MTLRenderPipelineState
    let world: World
    let camera: FlightCamera
    var clock = SkyClock()
    let battery: Bool
    var timeScale: Double = 1
    var debugMode: Float = 0
    var headroom: Float = 0
    private var sceneTex: MTLTexture?
    private var simTime: Double = 0
    private var lastFrame: CFTimeInterval?
    private var frameIndex: Float = 0
    private var exposure: Double = 1.5
    var renderScale: Double

    // Cover drifts between clear and broken skies over tens of minutes.
    var cloudCover: Double {
        0.42 + 0.14 * sin(simTime * 0.0021 + 1.0) + 0.08 * sin(simTime * 0.0057 + 4.0)
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
        guard let sp = try? device.makeRenderPipelineState(descriptor: scene),
              let pp = try? device.makeRenderPipelineState(descriptor: post),
              let world = World(device: device, library: lib, seed: seed) else { return nil }
        self.device = device
        self.queue = queue
        scenePipeline = sp
        postPipeline = pp
        self.world = world
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919)
        self.battery = battery
        renderScale = battery ? 0.4 : 0.6
        super.init()
        camera.heightAt = { [unowned world] x, z in world.height(x: x, z: z) }
        world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z), queue: queue)
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

    // Advances the simulation; large steps are subdivided so steering stays stable.
    func simulate(_ dt: Double) {
        var remaining = dt * timeScale
        while remaining > 0 {
            let step = min(remaining, 1.0 / 30.0)
            camera.step(dt: step)
            clock.advance(step)
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z), queue: queue)
            remaining -= step
        }
        simTime += dt * timeScale
        let e = clock.sunDirection.y
        let target = e > 0 ? 0.8 + 0.5 * max(0, 0.25 - e) * 4 : 1.3 + min(1.0, -e * 5) * 0.8
        exposure += (target - exposure) * min(1, dt * timeScale * 0.2)
    }

    func makeFrame(size: CGSize) -> Frame {
        var f = Frame()
        let b = camera.basis
        let o = SIMD3(Double(world.origin.x), 0, Double(world.origin.y))
        let p = camera.position - o
        f.camPos = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(tan(25.0 * .pi / 180)))
        f.camFwd = SIMD4(SIMD3<Float>(b.fwd), Float(size.width / size.height))
        f.camRight = SIMD4(SIMD3<Float>(b.right), Float(simTime.truncatingRemainder(dividingBy: 3600)))
        f.camUp = SIMD4(SIMD3<Float>(b.up), battery ? 0 : 1)
        f.sunDir = SIMD4(SIMD3<Float>(clock.sunDirection), Float(clock.phase))
        f.resolution = SIMD4(Float(size.width), Float(size.height), Float(1 / size.width), Float(1 / size.height))
        f.mapInfo = SIMD4(world.origin.x, world.origin.y, Int32(World.size), Int32(bitPattern: world.seed))
        f.misc = SIMD4(Float(exposure), world.globalMax, 0.6, Float(cloudCover))
        f.misc2 = SIMD4(frameIndex, 1, debugMode, headroom)
        return f
    }

    func encode(into cb: MTLCommandBuffer, output: MTLTexture, outputSize: CGSize) {
        let w = max(16, Int(outputSize.width * renderScale)), h = max(16, Int(outputSize.height * renderScale))
        if sceneTex?.width != w || sceneTex?.height != h {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]
            td.storageMode = .private
            sceneTex = device.makeTexture(descriptor: td)
        }
        guard let sceneTex else { return }
        var f = makeFrame(size: CGSize(width: w, height: h))
        frameIndex += 1

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
        let pp = MTLRenderPassDescriptor()
        pp.colorAttachments[0].texture = output
        pp.colorAttachments[0].loadAction = .dontCare
        pp.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: pp) {
            enc.setRenderPipelineState(postPipeline)
            enc.setFragmentBytes(&f, length: MemoryLayout<Frame>.stride, index: 0)
            enc.setFragmentTexture(sceneTex, index: 0)
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
            headroom = max(headroom, 1.0001)
        }
        guard let drawable = view.currentDrawable, let cb = queue.makeCommandBuffer() else { return }
        encode(into: cb, output: drawable.texture, outputSize: view.drawableSize)
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
        let w = Int(settings.screenshotSize.width), h = Int(settings.screenshotSize.height)
        r.renderScale = settings.renderScale ?? r.renderScale
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        let tex = device.makeTexture(descriptor: td)!
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            if let phase = settings.phase { r.clock.phase = phase }
            r.simulate(t - simulated)
            if let phase = settings.phase { r.clock.phase = phase }
            simulated = t
            var times: [Double] = []
            for _ in 0..<3 {
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
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f cam=(%.0f,%.0f,%.0f) gpu=%.1fms -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.camera.position.x, r.camera.position.y, r.camera.position.z, times.min()!, out))
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
