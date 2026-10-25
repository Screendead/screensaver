import AppKit
import MetalKit

extension Renderer {
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
        if let path = shotPath {
            shotPath = nil
            saveShot(drawable.texture, into: cb, path: path)
        }
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

    // The drawable is the dither-cell grid, which the layer scales up with nearest filtering, so the PNG does the same.
    private func saveShot(_ src: MTLTexture, into cb: MTLCommandBuffer, path: String) {
        let w = src.width, h = src.height
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: src.pixelFormat, width: w, height: h, mipmapped: false)
        d.storageMode = .shared
        guard let copy = queue.device.makeTexture(descriptor: d), let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, to: copy)
        blit.endEncoding()
        let W = Int(screenPixels.width), H = Int(screenPixels.height)
        cb.addCompletedHandler { _ in
            DispatchQueue.global(qos: .utility).async {
                var cells = [UInt8](repeating: 255, count: w * h * 4)
                if copy.pixelFormat == .rgba16Float {
                    // Extended linear sRGB; light above 1 clips, as on a screen without headroom.
                    var half = [Float16](repeating: 0, count: w * h * 4)
                    copy.getBytes(&half, bytesPerRow: w * 8, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    for i in 0..<w * h {
                        for c in 0..<3 {
                            let x = min(max(Float(half[i * 4 + c]), 0), 1)
                            let e = x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055
                            cells[i * 4 + c] = UInt8(e * 255 + 0.5)
                        }
                    }
                } else {
                    copy.getBytes(&cells, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    for i in 0..<w * h { cells.swapAt(i * 4, i * 4 + 2) }
                }
                var out = [UInt8](repeating: 255, count: W * H * 4)
                for y in 0..<H {
                    let sy = y * h / H
                    for x in 0..<W {
                        let s = (sy * w + x * w / W) * 4, o = (y * W + x) * 4
                        out[o] = cells[s]; out[o + 1] = cells[s + 1]; out[o + 2] = cells[s + 2]
                    }
                }
                let ok = Snapshot.write(bytes: out, width: W, height: H, to: path)
                BugLog.note("screenshot \(ok ? "saved" : "failed"): \(path)")
            }
        }
    }
}
