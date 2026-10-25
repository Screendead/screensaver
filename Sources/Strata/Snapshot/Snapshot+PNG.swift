import AppKit
import Metal

extension Snapshot {
    // Encodes one frame into `output` and waits for it; the command buffer holds its GPU times.
    @discardableResult
    static func render(_ r: Renderer, into output: MTLTexture) -> MTLCommandBuffer {
        let cb = r.queue.makeCommandBuffer()!
        r.encode(into: cb, output: output, outputSize: CGSize(width: output.width, height: output.height))
        cb.commit()
        cb.waitUntilCompleted()
        return cb
    }

    // `tex` is rgba8Unorm and CPU-readable.
    static func save(_ tex: MTLTexture, to path: String) -> Bool {
        let w = tex.width, h = tex.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return write(bytes: bytes, width: w, height: h, to: path)
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
