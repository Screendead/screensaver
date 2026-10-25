import Foundation
import Metal

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

    // A scene variant: the band test, the islands' surfaces, their shade and sky occlusion, the work close by them, their
    // mist (function constants 0, 4, 3, 6, 5 of sceneFragment).
    typealias SceneVariant = (bands: Bool, shape: Bool, isles: Bool, near: Bool, mist: Bool)
    // The scene variants for one towers setting, as Renderer indexes them: none, bands, island surfaces, both, the
    // islands' shade, close by an island, and with its mist.
    static let sceneVariants: [SceneVariant] = [(false, false, false, false, false), (true, false, false, false, false),
                                                (false, true, false, false, false), (true, true, false, false, false),
                                                (true, false, true, false, false), (true, true, true, true, false),
                                                (true, true, true, true, true)]

    // sea: the sea of cloud compiled in (function constant 8), only while SkyClock.cloudSea is above 0.
    func scene(towers: Bool, sea: Bool, _ v: SceneVariant) -> MTLRenderPipelineState? {
        pipeline(GPU.sceneKey(towers, sea, v)) { sceneDescriptor(towers, sea, v) }
    }

    private static func sceneKey(_ towers: Bool, _ sea: Bool, _ v: SceneVariant) -> String {
        "scene \(towers) \(sea) \(v.bands) \(v.shape) \(v.isles) \(v.near) \(v.mist)"
    }

    private func sceneDescriptor(_ towers: Bool, _ sea: Bool, _ v: SceneVariant) -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var b = v.bands, t = towers, i = v.isles, sh = v.shape, m = v.mist, n = v.near, s = sea
        fc.setConstantValue(&s, type: .bool, index: 8)
        fc.setConstantValue(&b, type: .bool, index: 0)
        fc.setConstantValue(&t, type: .bool, index: 1)
        fc.setConstantValue(&i, type: .bool, index: 3)
        fc.setConstantValue(&sh, type: .bool, index: 4)
        fc.setConstantValue(&m, type: .bool, index: 5)
        fc.setConstantValue(&n, type: .bool, index: 6)
        return fullscreen(try? library.makeFunction(name: "sceneFragment", constantValues: fc), .rgba16Float)
    }

    // The giants' hanging strands, drawn over the scene (vineFragment): the towers and the band test compiled out.
    func vine(sea: Bool) -> MTLRenderPipelineState? {
        pipeline("vine \(sea)") { vineDescriptor(sea) }
    }

    private func vineDescriptor(_ sea: Bool) -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var no = false, s = sea
        fc.setConstantValue(&no, type: .bool, index: 0)
        fc.setConstantValue(&no, type: .bool, index: 1)
        fc.setConstantValue(&s, type: .bool, index: 8)
        return fullscreen(try? library.makeFunction(name: "vineFragment", constantValues: fc), .rgba16Float)
    }

    // star: the resolve round the evening star (STAR_FC), drawn over the plain one in a scissor.
    func taa(star: Bool = false) -> MTLRenderPipelineState? {
        pipeline(star ? "taa star" : "taa") { taaDescriptor(star) }
    }

    private func taaDescriptor(_ star: Bool) -> MTLRenderPipelineDescriptor {
        let fc = MTLFunctionConstantValues()
        var s = star
        fc.setConstantValue(&s, type: .bool, index: 7)
        return fullscreen(try? library.makeFunction(name: "taaFragment", constantValues: fc), .rgba16Float)
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
        var descriptors = ["taa": taaDescriptor(false), "taa star": taaDescriptor(true),
                           "bio": bioDescriptor(),
                           GPU.postKey(output, true): postDescriptor(output, true)]
        for s in [false, true] {
            for t in [false, true] {
                for v in GPU.sceneVariants { descriptors[GPU.sceneKey(t, s, v)] = sceneDescriptor(t, s, v) }
            }
            descriptors["vine \(s)"] = vineDescriptor(s)
        }
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
