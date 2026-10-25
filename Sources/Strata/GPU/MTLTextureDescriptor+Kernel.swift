import Metal

extension MTLTextureDescriptor {
    // A private 2D texture kernels read and write; views: it is also seen through views of other formats or mips.
    static func kernel(_ format: MTLPixelFormat, width: Int, height: Int, mipmapped: Bool = false, views: Bool = false) -> MTLTextureDescriptor {
        let d = texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: mipmapped)
        d.usage = views ? [.shaderRead, .shaderWrite, .pixelFormatView] : [.shaderRead, .shaderWrite]
        d.storageMode = .private
        return d
    }
}
