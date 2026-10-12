import Foundation
import Metal
import simd

// A giant's hanging strands (Shaders/Saver.metal, MARK: Hanging strands): a slot per giant near the camera holding its
// baked strand maps, and the frame's table of them (vineFragment's buffer 10; the maps its textures 18 to 21).
final class Vines {
    // Must match VINE_SLOTS and VINE_W in Shaders/Saver.metal.
    static let slots = 6, window = 512
    static let recordSize = 64
    static let tableBytes = 16 + slots * recordSize

    private let bakePipeline: MTLComputePipelineState
    private let coarsePipeline: MTLComputePipelineState
    private let superPipeline: MTLComputePipelineState
    private let rangePipeline: MTLComputePipelineState
    let fine: MTLTexture, coarse: MTLTexture, sup: MTLTexture, bands: MTLTexture
    // The axis of the giant each slot holds, and whether its maps are baked (then `ranges` holds its strands' bounds:
    // lowest bottom, highest top, and the super cells they lie in, the trace's box).
    private var held: [SIMD2<Double>?] = Array(repeating: nil, count: Vines.slots)
    private var baked = [Bool](repeating: false, count: Vines.slots)
    // Counts a slot's assignments, so a bake's completion marks only the giant it baked.
    private var generation = [Int](repeating: 0, count: Vines.slots)
    private let bakedLock = NSLock()
    private let ranges: MTLBuffer
    private(set) var table = [UInt8](repeating: 0, count: Vines.tableBytes)
    private(set) var tableCount = 0

    init?(gpu: GPU) {
        func array(_ format: MTLPixelFormat, _ size: Int, wide: Int = 1) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: size * wide, height: size, mipmapped: false)
            d.textureType = .type2DArray
            d.arrayLength = Vines.slots
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return gpu.device.makeTexture(descriptor: d)
        }
        guard let b = gpu.pipeline("bakeVines"), let c = gpu.pipeline("vineCoarse"), let s = gpu.pipeline("vineSuper"),
              let r = gpu.pipeline("vineRange"), let rb = gpu.device.makeBuffer(length: Vines.slots * 32, options: .storageModeShared),
              let f = array(.rg16Float, Vines.window / 2), let co = array(.rgba32Uint, Vines.window / 8), let su = array(.rgba32Uint, Vines.window / 32),
              let ba = array(.rgba32Uint, Vines.window / 32, wide: 2)
        else { return nil }
        bakePipeline = b
        coarsePipeline = c
        superPipeline = s
        rangePipeline = r
        ranges = rb
        fine = f
        coarse = co
        sup = su
        bands = ba
    }

    // Keeps a slot for each giant within `reach` of the eye (the nearest first), baking a newcomer's maps into cb, and
    // writes the frame's table in map-local coordinates (the map's origin).
    func update(giants: [GiantInfo], eye: SIMD2<Double>, reach: Double, origin: SIMD2<Int32>, seed: UInt32, into cb: MTLCommandBuffer) {
        let near = giants.filter { simd_length($0.pos - eye) < reach + GiantInfo.reach }
            .sorted { simd_length($0.pos - eye) < simd_length($1.pos - eye) }
            .prefix(Vines.slots)
        var slotOf = near.map { g in held.firstIndex(where: { $0 == g.pos }) }
        var free = (0..<Vines.slots).filter { s in !slotOf.contains(s) }
        var enc: MTLComputeCommandEncoder?
        var fresh: [(slot: Int, generation: Int)] = []
        for (i, g) in near.enumerated() where slotOf[i] == nil {
            let s = free.removeFirst()
            slotOf[i] = s
            held[s] = g.pos
            bakedLock.lock(); generation[s] += 1; baked[s] = false; bakedLock.unlock()
            fresh.append((s, generation[s]))
            if enc == nil { enc = cb.makeComputeCommandEncoder() }
            guard let enc else { break }
            let w = Vines.windowOrigin(g)
            var b = VineBake(a: SIMD4(Float(g.pos.x), Float(g.pos.y), Float(g.ground), Float(g.yT)), b: SIMD4(w.x, w.y, 0, 0),
                             c: SIMD4(g.h, UInt32(g.kind), seed, UInt32(s)))
            enc.setBytes(&b, length: MemoryLayout<VineBake>.stride, index: 0)
            enc.setComputePipelineState(bakePipeline)
            enc.setTexture(fine, index: 0)
            let n = Vines.window / 2
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.setComputePipelineState(coarsePipeline)
            enc.setTexture(fine, index: 0)
            enc.setTexture(coarse, index: 1)
            enc.dispatchThreads(MTLSize(width: n / 4, height: n / 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.setComputePipelineState(superPipeline)
            enc.setTexture(coarse, index: 0)
            enc.setTexture(sup, index: 1)
            enc.setTexture(bands, index: 2)
            enc.dispatchThreads(MTLSize(width: n / 16, height: n / 16, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.setComputePipelineState(rangePipeline)
            enc.setTexture(sup, index: 0)
            enc.setBuffer(ranges, offset: 0, index: 0)
            enc.setBytes(&b, length: MemoryLayout<VineBake>.stride, index: 1)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        enc?.endEncoding()
        if !fresh.isEmpty {
            // Until the bake lands the slab is the loose one below.
            cb.addCompletedHandler { _ in
                self.bakedLock.lock()
                for f in fresh where self.generation[f.slot] == f.generation { self.baked[f.slot] = true }
                self.bakedLock.unlock()
            }
        }
        // Slots no giant in the list holds stay baked: the giant may come back.
        tableCount = near.count
        table.withUnsafeMutableBytes { p in
            p.storeBytes(of: Int32(near.count), as: Int32.self)
            for (i, g) in near.enumerated() {
                let o = 16 + i * Vines.recordSize, w = Vines.windowOrigin(g), s = slotOf[i] ?? 0
                // Until the bake lands: the whole window, and heights from the ground to over the crown.
                var slab = SIMD2<Float>(Float(g.ground + 2), Float(g.ground + g.yT + 42))
                var boxLo = SIMD2<Int32>(w.x &- origin.x, w.y &- origin.y), boxHi = boxLo &+ Int32(Vines.window)
                bakedLock.lock(); let isBaked = baked[s]; bakedLock.unlock()
                if isBaked {
                    let r0 = ranges.contents().load(fromByteOffset: s * 32, as: SIMD4<Float>.self)
                    let r1 = ranges.contents().load(fromByteOffset: s * 32 + 16, as: SIMD4<Float>.self)
                    if r0.y > r0.x {
                        slab = SIMD2(r0.x - 1, r0.y + 1)
                        boxLo = SIMD2(boxLo.x + Int32(r0.z) * 32, boxLo.y + Int32(r0.w) * 32)
                        boxHi = SIMD2(w.x &- origin.x + (Int32(r1.x) + 1) * 32, w.y &- origin.y + (Int32(r1.y) + 1) * 32)
                    }
                }
                p.storeBytes(of: SIMD4<Float>(Float(g.pos.x - Double(origin.x)), Float(g.pos.y - Double(origin.y)), Float(g.ground), Float(g.yT)), toByteOffset: o, as: SIMD4<Float>.self)
                p.storeBytes(of: SIMD4<Float>(Float(boxLo.x), Float(boxLo.y), slab.x, slab.y), toByteOffset: o + 16, as: SIMD4<Float>.self)
                p.storeBytes(of: SIMD4<UInt32>(g.h, UInt32(g.kind), UInt32(s), 0), toByteOffset: o + 32, as: SIMD4<UInt32>.self)
                p.storeBytes(of: SIMD4<Int32>(w.x, w.y, boxHi.x, boxHi.y), toByteOffset: o + 48, as: SIMD4<Int32>.self)
            }
        }
    }

    // The window's origin: 8-aligned, so the strand cells stay on the world's even grid.
    private static func windowOrigin(_ g: GiantInfo) -> SIMD2<Int32> {
        SIMD2(Int32(floor((g.pos.x - Double(window / 2)) / 8)) * 8, Int32(floor((g.pos.y - Double(window / 2)) / 8)) * 8)
    }
}

// Must match VineBake in Shaders/Saver.metal.
struct VineBake {
    var a: SIMD4<Float>
    var b: SIMD4<Int32>
    var c: SIMD4<UInt32>
}
