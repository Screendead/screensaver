kernel void generateHeight(texture2d<float, access::write> out [[texture(0)]],
                           texture2d<uint, access::write> gxOut [[texture(1)]],
                           constant int4& info [[buffer(0)]],
                           constant uint& row0 [[buffer(1)]],
                           device const IsleRec* isles [[buffer(2)]],
                           device const CaveRec* caves [[buffer(3)]],
                           uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    int2 w = int2(gid) + info.xy;
    Col c = terrainColumnC(w, uint(info.w), 1, caves, isles);
    out.write(float4(c.h, c.gl, c.gh, isleAlpha(c)), gid);
    gxOut.write(c.h < SEA && c.gh <= c.gl ? reefPack(packGaps(c), c.reef) : packGaps(c), gid);
}

// Solid at height y in a column (H, gapLo, gapHi) with extra gaps e (lo1, hi1, lo2, hi2).
static inline bool solidIn(float3 c, float4 e, float y) {
    return y < c.x && !(y >= c.y && y < c.z) && !(y >= e.x && y < e.y) && !(y >= e.z && y < e.w);
}

// Clamps each column into its 4-neighbour height range: removes single-voxel spikes, pits and slots.
kernel void cleanupColumns(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           texture2d<uint, access::read> gxSrc [[texture(2)]],
                           texture2d<uint, access::write> gxDst [[texture(3)]],
                           constant uint& row0 [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 c = src.read(gid);
    uint4 gx = gxSrc.read(gid);
    uint2 a = uint2(max(int(gid.x) - 1, 0), gid.y), b = uint2(min(gid.x + 1, n - 1), gid.y);
    uint2 e = uint2(gid.x, max(int(gid.y) - 1, 0)), f = uint2(gid.x, min(gid.y + 1, n - 1));
    float4 c0 = src.read(a), c1 = src.read(b), c2 = src.read(e), c3 = src.read(f);
    float h0 = c0.r, h1 = c1.r, h2 = c2.r, h3 = c3.r;
    float r0 = c.r;
    c.r = clamp(c.r, min(min(h0, h1), min(h2, h3)), max(max(h0, h1), max(h2, h3)));
    // A lone coral spike clamped down to the bed round it is bed again (reefPack: colony in x bit 15, tip in y bit 13);
    // a colony's crown clamped to its shoulders stays coral.
    if (c.r < r0 && ((gxSrc.read(a).x | gxSrc.read(b).x | gxSrc.read(e).x | gxSrc.read(f).x) & 0x8000u) == 0u) { gx.x &= ~0x8000u; gx.y &= ~0x2000u; }
    // A column-wide slot in a wall (both neighbours along an axis higher) shows as a dash down the face: fill it.
    if (c.r > SEA + 2.0) c.r = max(c.r, max(min(h0, h1), min(h2, h3)));
    // A notch in an island's rim raised to its neighbours would be a pillar from the ground to the island's top.
    if (c.r > r0 + 8.0 && ((gapFlags(gxSrc.read(a)) | gapFlags(gxSrc.read(b)) | gapFlags(gxSrc.read(e)) | gapFlags(gxSrc.read(f))) & GAP_ISLAND) != 0u)
        c.r = r0;
    // A fall's run stays as it hangs: lowered to its neighbours, a short one would lose its gap and stand on the ground.
    if ((gapFlags(gx) & GAP_ISLAND) != 0u && c.a <= -4.0) c.r = r0;
    if ((gapFlags(gx) & GAP_CARVED) == 0u) {
        if (c.b > c.r - 3.0) { c.g = 0; c.b = 0; }
        // A roof needs walls: keep a gap only where every neighbour reaches above it.
        if (min(min(h0, h1), min(h2, h3)) < c.b + 1.0) { c.g = 0; c.b = 0; }
        dst.write(c, gid);
        gxDst.write(gx, gid);
        return;
    }
    // Carved rock: its roofs are slabs the carving keeps whole (a test for a supporting neighbour cut a pillar wherever
    // one ceiling stood a voxel above both its neighbours'); a gap a single column wide along either axis (rock on both
    // sides at its middle) reads as a dash and is filled, and every roof keeps three voxels of rock over it.
    float4 x0 = extraGaps(gxSrc.read(a)), x1 = extraGaps(gxSrc.read(b)), x2 = extraGaps(gxSrc.read(e)), x3 = extraGaps(gxSrc.read(f));
    float4 ex = extraGaps(gx);
    float3 gaps[3] = { float3(c.g, c.b, 0), float3(ex.x, ex.y, 0), float3(ex.z, ex.w, 0) };
    float top = c.r;
    for (int i = 0; i < 3; i++) {
        float lo = gaps[i].x, hi = gaps[i].y;
        if (hi <= lo) continue;
        float mid = floor((lo + hi) * 0.5);
        bool keep = hi <= top - 3.0 && hi - lo >= 3.0;
        if (i == 0 && (gapFlags(gx) & GAP_ISLAND) != 0u) { if (!keep) gaps[i] = float3(0); else top = lo; continue; }
        keep = keep && !(solidIn(c0.rgb, x0, mid) && solidIn(c1.rgb, x1, mid)) && !(solidIn(c2.rgb, x2, mid) && solidIn(c3.rgb, x3, mid));
        if (!keep) gaps[i] = float3(0);
        else top = lo;
    }
    // The highest surviving gap is the principal one.
    int pi = gaps[0].y > gaps[0].x ? 0 : gaps[1].y > gaps[1].x ? 1 : 2;
    Col o;
    o.h = c.r;
    o.gl = gaps[pi].x; o.gh = gaps[pi].y;
    o.lo1 = pi == 0 ? gaps[1].x : pi == 1 ? gaps[2].x : 0.0;
    o.hi1 = pi == 0 ? gaps[1].y : pi == 1 ? gaps[2].y : 0.0;
    o.lo2 = pi == 0 ? gaps[2].x : 0.0;
    o.hi2 = pi == 0 ? gaps[2].y : 0.0;
    if (o.hi1 <= o.lo1) { o.lo1 = o.lo2; o.hi1 = o.hi2; o.lo2 = 0; o.hi2 = 0; }
    o.carved = o.gh > o.gl;
    o.island = (gapFlags(gx) & GAP_ISLAND) != 0u && pi == 0;
    o.leaf = 0.0;
    o.isk = 0u;
    dst.write(float4(o.h, o.gl, o.gh, c.a), gid);
    gxDst.write(packGaps(o), gid);
}
