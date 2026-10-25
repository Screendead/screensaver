import Foundation
import Metal
import XCTest

// ShaderProbes.metal built against the repo's Shaders/ (the shader as the app ships it, its includes resolved there),
// run one dispatch at a time. Built once per test run.
final class ShaderProbes {
    let device: MTLDevice
    let library: MTLLibrary
    private let queue: MTLCommandQueue

    private static let shared: Result<ShaderProbes, Error> = Result { try ShaderProbes() }

    // Skips the test without a GPU; fails it when the probes don't build.
    static func get() throws -> ShaderProbes {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try shared.get()
    }

    private init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("no Metal device") }
        let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let shaders = here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Shaders")
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("strata-probes-\(getpid()).metallib")
        let xcrun = Process()
        xcrun.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        // The app builds with -O3; -O0 compiles in a second, and every comparison allows float rounding.
        xcrun.arguments = ["metal", "-O0", "-std=metal3.1", "-w", "-I", shaders.path, here.appendingPathComponent("ShaderProbes.metal").path, "-o", out.path]
        let errors = Pipe()
        xcrun.standardError = errors
        try xcrun.run()
        let message = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        xcrun.waitUntilExit()
        guard xcrun.terminationStatus == 0 else { throw NSError(domain: "ShaderProbes", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
        defer { try? FileManager.default.removeItem(at: out) }
        self.device = device
        self.queue = queue
        library = try device.makeLibrary(URL: out)
    }

    // Runs kernel `name` over `input` (buffer 0) into `count` outputs (buffer 1), or with no input into buffer 0;
    // `bytes` go to the next index.
    func run<In, Out>(_ name: String, _ input: [In]?, count: Int, as: Out.Type, bytes: [UInt32] = [],
                      constants: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> [Out] {
        let fn = try library.makeFunction(name: name, constantValues: constants)
        let pipeline = try device.makeComputePipelineState(function: fn)
        let out = device.makeBuffer(length: max(count, 1) * MemoryLayout<Out>.stride, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        var index = 0
        if let input {
            enc.setBuffer(input.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }, offset: 0, index: 0)
            index = 1
        }
        enc.setBuffer(out, offset: 0, index: index)
        if !bytes.isEmpty { enc.setBytes(bytes, length: bytes.count * 4, index: index + 1) }
        let n = input?.count ?? 1
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        let p = out.contents().bindMemory(to: Out.self, capacity: count)
        return (0..<count).map { p[$0] }
    }
}
