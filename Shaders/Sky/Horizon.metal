// The terrain horizon around the camera: per azimuth bin, the steepest tangent of elevation any terrain reaches,
// from the max mips (cells about a percent of the distance wide, so a bin also bounds its neighbours' gaps); the ground
// under an island's band counts, and the island over it gives the bin a range of tangents of its own (bins
// HORIZON_BINS on: its lowest, 2 HORIZON_BINS on: its highest). A primary ray above its bin's horizon (and the
// neighbours') and outside their island ranges can hit nothing and skips the trace.
constant int HORIZON_BINS = 1024;
constant int HORIZON_SEGS = 64;

kernel void bakeHorizon(device float* out [[buffer(0)]],
                        constant Frame& f [[buffer(1)]],
                        texture2d<float> hm [[texture(0)]],
                        texture2d_array<float> coarse [[texture(1)]],
                        texture2d<float> hx [[texture(2)]],
                        texture2d<uint> gx [[texture(3)]],
                        texture2d_array<float> cg [[texture(4)]], texture2d_array<float> cg2 [[texture(5)]],
                        texture2d<float> hb [[texture(6)]], texture2d_array<float> cb [[texture(7)]],
                        device const GiantRec* giants [[buffer(2)]],
                        uint2 tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    // HORIZON_SEGS threads per bin walk geometric slices of the distance and reduce, so the bake is not one long chain.
    threadgroup float3 part[HORIZON_SEGS];
    uint gid = tid.y;
    int levels = int(f.fade.z);
    Terrain tr { hm, coarse, coarse, coarse, hm, f.lod, f.mapInfo.xy, f.mapInfo.z, levels,
                 f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w, f.fade.y, f.fade.x, 0.0, hx, gx, cg, cg2, hb, uint(f.mapInfo.w), giants, hx, nullptr, 0, cb };
    float a = (float(gid) + 0.5) * TAU / float(HORIZON_BINS);
    float2 dir = float2(sin(a), cos(a));
    float3 o = f.camPos.xyz;
    float best = -1e9, iLo = 1e9, iHi = -1e9;
    float d = lane == 0 ? 1.0 : pow(tr.rmax, float(lane) / float(HORIZON_SEGS));
    float dEnd = pow(tr.rmax, float(lane + 1) / float(HORIZON_SEGS));
    while (d < dEnd) {
        float2 X = o.xz + dir * d;
        int k = levelAt(tr, length(X - tr.cam));
        if (k >= levels) break;
        float cs = tr.lod[k].w;
        // Cell about 1% of the distance (and at least two columns) wide, from mip 1 so morph targets are bounded: at least
        // the spacing of the bins, so a ray between two bins' lines lies in a cell one of them reads.
        int m = clamp(int(ceil(log2(max(0.01 * d, 2.0 * cs) / cs))), 1, MAX_LEVEL);
        float w = cs * float(1 << m);
        int2 mc = (int2(floor(X / cs)) + levelOrigin(tr, k)) >> m;
        float hmax = k == 0 ? tr.hx.read(uint2(clamp(mc, int2(0), int2((tr.size >> m) - 1))), m).r
                            : tr.ch.read(uint2(mc & ((COARSE_N >> m) - 1)), uint(k - 1), uint(m)).r;
        float s0 = max(d - 0.75 * w, 0.5), s1 = d + 0.75 * w;
        float top = abs(hmax);
        // The bands the trace walks through whatever the morph (trace): the ground stops at the floor, the island
        // stands from the roof to the top.
        if (hmax < 0.0) {
            float2 bd = k == 0 ? tr.hb.read(uint2(clamp(mc, int2(0), int2((tr.size >> m) - 1))), m - 1).rg
                               : tr.cb.read(uint2(mc & ((COARSE_N >> m) - 1)), uint(k - 1), uint(m - 1)).rg;
            float2 lo = bandLo(bd.x);
            if (k > 0 || lo.y > 0.0) {
                float kl = bd.y - o.y, tp = top - o.y;
                iLo = min(iLo, kl / (kl >= 0.0 ? s1 : s0));
                iHi = max(iHi, tp / (tp >= 0.0 ? s0 : s1));
                top = lo.x;
            }
        }
        best = max(best, (top - o.y) / s0);
        d += 0.5 * w;
    }
    part[lane] = float3(best, iLo, iHi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (int i = 1; i < HORIZON_SEGS; i++) { best = max(best, part[i].x); iLo = min(iLo, part[i].y); iHi = max(iHi, part[i].z); }
        out[gid] = best;
        out[HORIZON_BINS + gid] = iLo;
        out[2 * HORIZON_BINS + gid] = iHi;
    }
}

static inline bool aboveHorizon(device const float* hz, float3 d) {
    float l = length(d.xz);
    if (l < 1e-4) return d.y > 0.0;
    float a = atan2(d.x, d.z);
    int i = int(floor((a < 0.0 ? a + TAU : a) / TAU * float(HORIZON_BINS)));
    int i0 = (i + HORIZON_BINS - 1) % HORIZON_BINS, i1 = i % HORIZON_BINS, i2 = (i + 1) % HORIZON_BINS;
    float t = max(max(hz[i0], hz[i1]), hz[i2]);
    float e = d.y / l;
    if (e <= t + 0.002) return false;
    float lo = min(min(hz[HORIZON_BINS + i0], hz[HORIZON_BINS + i1]), hz[HORIZON_BINS + i2]);
    float hi = max(max(hz[2 * HORIZON_BINS + i0], hz[2 * HORIZON_BINS + i1]), hz[2 * HORIZON_BINS + i2]);
    return e < lo - 0.002 || e > hi + 0.002;
}
