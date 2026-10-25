import Foundation
import simd

extension Wildlife {
    // The birds the renderer is given, flock by flock and cluster by cluster, within the bird buffer's caps.
    func drawList() -> [(flock: Flock, clusters: [Range<Int>])] {
        var list: [(flock: Flock, clusters: [Range<Int>])] = []
        var nc = 0, nb = 0
        for f in flocks where list.count < Wildlife.maxFlocks {
            var cs: [Range<Int>] = []
            for start in stride(from: 0, to: f.birds.count, by: Wildlife.clusterSize) where nc < Wildlife.maxClusters {
                let n = min(min(start + Wildlife.clusterSize, f.birds.count) - start, Wildlife.maxBirds - nb)
                if n <= 0 { continue }
                cs.append(start..<(start + n))
                nb += n
                nc += 1
            }
            if !cs.isEmpty { list.append((f, cs)) }
        }
        return list
    }

    // Writes cluster spheres then birds into `buffer` (bufferLength bytes), map-local (relative to `origin`). Returns the
    // flock count.
    package func upload(into buffer: UnsafeMutableRawPointer, origin o: SIMD3<Double>) -> Int {
        let out = buffer.bindMemory(to: SIMD4<Float>.self, capacity: Wildlife.bufferLength / 16)
        var nc = 0, nb = 0
        let clusters = Wildlife.maxFlocks * 2
        let birds = (Wildlife.maxFlocks + Wildlife.maxClusters) * 2
        var nf = 0
        for (f, cs) in drawList() {
            let fl = Wildlife.flight(f.kind)
            let firstCluster = nc
            var flo = SIMD3<Double>(repeating: .infinity), fhi = SIMD3<Double>(repeating: -.infinity)
            for c in cs {
                let first = nb
                var lo = SIMD3<Double>(repeating: .infinity), hi = SIMD3<Double>(repeating: -.infinity)
                for i in c {
                    let b = f.birds[i]
                    let p = b.pos - o
                    lo = simd_min(lo, p); hi = simd_max(hi, p)
                    let v = simd_normalize(b.vel)
                    let w = wing(b, fl)
                    let base = birds + nb * 4
                    out[base] = SIMD4(Float(p.x), Float(p.y), Float(p.z), Float(b.size))
                    out[base + 1] = SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w.arm))
                    out[base + 2] = SIMD4(Float(b.bank), Float(f.kind.rawValue), Float(b.tint), Float(w.ext))
                    out[base + 3] = SIMD4(Float(w.hand), Float(w.sweep), 0, 0)
                    nb += 1
                }
                flo = simd_min(flo, lo); fhi = simd_max(fhi, hi)
                let c = (lo + hi) / 2
                let r = simd_length(hi - lo) / 2 + fl.span
                out[clusters + nc * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
                out[clusters + nc * 2 + 1] = SIMD4(Float(first), Float(nb - first), 0, 0)
                nc += 1
            }
            let c = (flo + fhi) / 2
            let r = simd_length(fhi - flo) / 2 + fl.span
            out[nf * 2] = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
            out[nf * 2 + 1] = SIMD4(Float(firstCluster), Float(nc - firstCluster), 0, 0)
            nf += 1
        }
        return nf
    }
}
