import Foundation
import Metal
import StrataCore
import simd

// Rare ruined towers. Whether a group stands turns on the climate and the provinces at its centre, so placeTowers
// decides it on the GPU for a square of world cells round the camera's; the list feeds the camera, and the records
// nearest the camera are the shader's table (fragment buffer 9; stage 3's giants hold 8).
final class Towers {
    // Must match TOWER_CELL, TOWER_MEMBERS and TOWER_TABLE in Shaders/World/Towers/Layout.metal.
    static let cell = 9000.0, members = 4, tableSize = 16
    // Cells on a side of the square round the camera's cell: every tower out past the world's edge (13.8k).
    static let grid = 5
    static let tableBytes = 16 + tableSize * TowerInfo.recordSize

    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let buffer: MTLBuffer
    private let seed: UInt32
    private(set) var list: [TowerInfo] = []
    private var cellAt: SIMD2<Int32>?
    private var inFlight = false
    private(set) var table = [UInt8](repeating: 0, count: Towers.tableBytes)
    private(set) var tableCount = 0

    init?(gpu: GPU, seed: UInt32) {
        let device = gpu.device
        guard let p = gpu.pipeline("placeTowers"),
              let b = device.makeBuffer(length: Towers.grid * Towers.grid * Towers.members * TowerInfo.recordSize, options: .storageModeShared)
        else { return nil }
        self.device = device
        pipeline = p
        buffer = b
        self.seed = seed
    }

    private func encode(_ cb: MTLCommandBuffer, into buf: MTLBuffer, first: SIMD2<Int32>, n: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buf, offset: 0, index: 0)
        var info = SIMD4<Int32>(first.x, first.y, Int32(n), Int32(bitPattern: seed))
        enc.setBytes(&info, length: MemoryLayout<SIMD4<Int32>>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 8), height: min(n, 8), depth: 1))
        enc.endEncoding()
    }

    private static func cellOf(_ p: SIMD2<Double>) -> SIMD2<Int32> {
        SIMD2(Int32(floor(p.x / cell)), Int32(floor(p.y / cell)))
    }

    // Keeps the list on the square round the camera's cell; wait: place them before returning (startup, snapshots).
    func update(center: SIMD2<Double>, queue: MTLCommandQueue, wait: Bool) {
        let c = Towers.cellOf(center)
        if c != cellAt, !inFlight, let cb = queue.makeCommandBuffer() {
            inFlight = true
            let n = Towers.grid
            encode(cb, into: buffer, first: c &- Int32(n / 2), n: n)
            // A failed placement keeps the old list and is tried again on the next update.
            // Strong: a weak reference would give Towers a side table, slowing the camera's many reads of the list.
            let done = { (ok: Bool) in
                if ok {
                    self.list = TowerInfo.decode(self.buffer.contents(), count: n * n * Towers.members)
                    self.cellAt = c
                }
                self.inFlight = false
            }
            if wait {
                cb.commit()
                cb.waitUntilCompleted()
                done(cb.status == .completed)
            } else {
                cb.addCompletedHandler { b in
                    let ok = b.status == .completed
                    DispatchQueue.main.async { done(ok) }
                }
                cb.commit()
            }
        }
    }

    // The table for a frame: the towers whose drum (or its mirror image in the water) can stand inside the view frustum
    // (fwd, right, up and the tangents of its half angles) within the world's edge. Out of view, the shader skips the
    // masonry at the cost of one test.
    func frameTable(eye: SIMD3<Double>, fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>, tanH: Double, tanV: Double) {
        let planes = [right - fwd * tanH, -right - fwd * tanH, up - fwd * tanV, -up - fwd * tanV, -fwd].map { simd_normalize($0) }
        let sea = 19.85
        func inView(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ r: Double) -> Bool {
            !planes.contains { simd_dot(a - eye, $0) > r && simd_dot(b - eye, $0) > r }
        }
        let flat = SIMD2(eye.x, eye.z)
        var near: [TowerInfo] = []
        for t in list where simd_length(t.pos - flat) < Clipmap.handoff(Clipmap.levels) + t.radius * 1.5 {
            let r = t.radius * 1.5 + 40
            let lo = SIMD3(t.pos.x, t.base, t.pos.y), hi = SIMD3(t.pos.x, t.top, t.pos.y)
            let mlo = SIMD3(t.pos.x, 2 * sea - t.base, t.pos.y), mhi = SIMD3(t.pos.x, 2 * sea - t.top, t.pos.y)
            if inView(lo, hi, r) || inView(mlo, mhi, r) { near.append(t) }
        }
        near.sort { simd_length($0.pos - flat) < simd_length($1.pos - flat) }
        let n = min(near.count, Towers.tableSize)
        tableCount = n
        table.withUnsafeMutableBytes { p in
            p.storeBytes(of: Int32(n), as: Int32.self)
            for i in 0..<n {
                let t = near[i], o = 16 + i * TowerInfo.recordSize
                p.storeBytes(of: t.record.0, toByteOffset: o, as: SIMD4<Float>.self)
                p.storeBytes(of: t.record.1, toByteOffset: o + 16, as: SIMD4<Float>.self)
                p.storeBytes(of: t.record.2, toByteOffset: o + 32, as: SIMD4<Float>.self)
                p.storeBytes(of: t.record.3, toByteOffset: o + 48, as: SIMD4<UInt32>.self)
            }
        }
    }

    // Every tower of an n-cell square round p (VR_TOWERS listings).
    func scan(around p: SIMD2<Double>, cells n: Int, queue: MTLCommandQueue) -> [TowerInfo] {
        guard let buf = device.makeBuffer(length: n * n * Towers.members * TowerInfo.recordSize, options: .storageModeShared),
              let cb = queue.makeCommandBuffer() else { return [] }
        encode(cb, into: buf, first: Towers.cellOf(p) &- Int32(n / 2), n: n)
        cb.commit()
        cb.waitUntilCompleted()
        return TowerInfo.decode(buf.contents(), count: n * n * Towers.members)
    }
}
