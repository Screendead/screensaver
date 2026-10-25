import Foundation
import Metal
import StrataCore
import simd

extension Renderer {
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

    // Through abs: a negated texel holds an island's band (coarseBandFlag).
    private static func readFar(_ buf: MTLBuffer) -> [Float16] {
        UnsafeBufferPointer(start: buf.contents().assumingMemoryBound(to: Float16.self), count: FarHeights.n * FarHeights.n).map { abs($0) }
    }

    // Level 3 changes only when its window moves a strip, every 512 voxels of flight: its heights are read back then,
    // off the main thread. The range farHeight trusts still follows the camera every 20 frames, as the window does.
    func updateFar(into cb: MTLCommandBuffer) {
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
    func refreshFar(only: Int? = nil) {
        guard let cb = queue.makeCommandBuffer() else { return }
        clipmap.encode(into: cb, center: camera.xz, drain: true, active: activeLevels, giants: world.giantBuffer, isles: world.isleBuffer, only: only)
        let buf = encodeFar(into: cb)
        cb.commit()
        cb.waitUntilCompleted()
        guard let buf else { return }
        far.heights = Renderer.readFar(buf)
        farRequested = clipmap.generation(3)
        farHeld = farRequested
        far.centre = farCameraCell
    }
}
