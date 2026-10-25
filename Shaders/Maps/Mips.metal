// Height a column is solid from the ground up: under an arch's gap, a canopy, a building's openings or a cave only the
// part below them counts. Extra gaps lie below the principal one.
static inline float solidFloor(float4 c, uint4 gx) {
    if (c.a > 0.0) return c.g;
    float4 e = extraGaps(gx);
    // Structures too: a lintel, eave, deck, stilt house or hull is open in [g, b) whatever its base says.
    float f = c.b > c.g ? c.g : c.r;
    if (e.y > e.x) f = min(f, e.x);
    if (e.w > e.z) f = min(f, e.z);
    return f;
}

// Mip 0 of the max-mip texture: the column tops alone, 2 bytes a column for sky visibility (the DDA reads mips from 1).
// Under an island its ground: the sky over the land beside one is open but for the island's own occlusion.
kernel void copyTop(texture2d<float, access::read> src [[texture(0)]],
                    texture2d<float, access::write> dst [[texture(1)]],
                    texture2d<uint, access::read> gx [[texture(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float4 c = src.read(gid);
    bool isl = c.a == 0.0 && c.b > c.g && (gapFlags(gx.read(gid)) & GAP_ISLAND) != 0u;
    dst.write(float4(isl ? c.g : c.r, 0, 0, 0), gid);
}

// Band mips: per cell the height band [x, y) every column in it leaves open, so a ray under a giant's crown
// skips cells as one over the tops does. A giant column offers its tallest gap, any other column the air above it.
// The open band of one column: a canopy column's tallest gap (at least 4 tall), any other column the air above it.
static inline float2 columnBand(texture2d<float, access::read> src, texture2d<uint, access::read> gx, uint2 c) {
    float4 col = src.read(c);
    float2 b = float2(col.r, 4096.0);
    if (col.a == GIANT_A) {
        float4 e = extraGaps(gx.read(c));
        float2 g = col.b > col.g ? float2(col.g, col.b) : float2(0);
        if (e.y - e.x > g.y - g.x) g = e.xy;
        if (e.w - e.z > g.y - g.x) g = e.zw;
        if (g.y - g.x >= 4.0) b = g;
    } else if (col.a > 0.0 && col.b - col.g >= 4.0) {
        // Under a floating canopy (the jungle's roof).
        b = col.gb;
    }
    return b;
}

// A band texel's floor and whether it holds for every morph toward level 1 (stored as -1 - floor), so the trace may use
// it inside the morph band too.
static inline float2 bandLo(float x) { return x < 0.0 ? float2(-1.0 - x, 1.0) : float2(x, 0.0); }

// Band mips from mip 1 (the trace never reads mip 0): per 2x2 columns the band [x, y) all of them leave open. Under an
// island the band holds through the morph toward level 1 (columnMorphW): its columns' air meets the level-1 column's;
// elsewhere the floor is mip 1's max, which bounds that column's top as well.
kernel void bandTop(texture2d<float, access::read> src [[texture(0)]],
                    texture2d<uint, access::read> gx [[texture(1)]],
                    texture2d<float, access::write> dst [[texture(2)]],
                    texture2d<float, access::read> mx1 [[texture(3)]],
                    constant int4& info [[buffer(0)]],
                    device const GiantRec* giants [[buffer(1)]],
                    device const IsleRec* isles [[buffer(2)]],
                    constant uint& row0 [[buffer(3)]],
                    uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    uint2 cs[4] = { s, s + uint2(1, 0), s + uint2(0, 1), s + uint2(1, 1) };
    bool ig[4];
    bool any = false;
    for (int i = 0; i < 4; i++) {
        float4 col = src.read(cs[i]);
        ig[i] = col.a == 0.0 && col.b - col.g >= ISLE_AIR;
        any = any || ig[i];
    }
    if (any) {
        float2 g2;
        float dp;
        float3 pc = coarseColumn((int2(s) + info.xy) >> 1, 1, uint(info.w), giants, isles, g2, dp);
        bool pIsl = pc.y < 0.0;
        float2 pb = pc.z > abs(pc.y) ? float2(abs(pc.y), pc.z) : float2(pc.x, 4096.0);
        float2 b = float2(-1e4, 4096.0);
        for (int i = 0; i < 4; i++) {
            float4 col = src.read(cs[i]);
            float2 cb = float2(max(col.r, pc.x), 4096.0);
            if (ig[i] || pIsl) {
                float2 o = ig[i] ? col.gb : float2(col.r, 4096.0);
                o = float2(max(o.x, pb.x), min(o.y, pb.y));
                if (o.y - o.x >= 4.0) cb = o;
            }
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
        dst.write(float4(-1.0 - b.x, b.y, 0, 0), gid);
        return;
    }
    float2 a = columnBand(src, gx, s), b = columnBand(src, gx, s + uint2(1, 0));
    float2 c = columnBand(src, gx, s + uint2(0, 1)), d = columnBand(src, gx, s + uint2(1, 1));
    float2 r = float2(max(max(a.x, b.x), max(c.x, d.x)), min(min(a.y, b.y), min(c.y, d.y)));
    if (r.y >= 4000.0) r = float2(-1.0 - max(r.x, abs(mx1.read(gid).r)), 4096.0);
    dst.write(float4(r, 0, 0), gid);
}

kernel void bandReduce(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float2 a = src.read(s).rg, b = src.read(s + uint2(1, 0)).rg, c = src.read(s + uint2(0, 1)).rg, d = src.read(s + uint2(1, 1)).rg;
    float2 la = bandLo(a.x), lb = bandLo(b.x), lc = bandLo(c.x), ld = bandLo(d.x);
    float lo = max(max(la.x, lb.x), max(lc.x, ld.x));
    bool safe = min(min(la.y, lb.y), min(lc.y, ld.y)) > 0.0;
    dst.write(float4(safe ? -1.0 - lo : lo, min(min(a.y, b.y), min(c.y, d.y)), 0, 0), gid);
}

// Negates a max mip's cells whose band is a real one (under a giant's crown), after every mip is built.
kernel void bandFlag(texture2d<float, access::read_write> mx [[texture(0)]],
                     texture2d<float, access::read> band [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= mx.get_width() || gid.y >= mx.get_height()) return;
    float2 b = band.read(gid).rg;
    b.x = bandLo(b.x).x;
    if (b.y < 4000.0 && b.y - b.x >= 4.0) mx.write(float4(-abs(mx.read(gid).r)), gid);
}

kernel void reduceMax(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float m = max(max(src.read(s).r, src.read(s + uint2(1, 0)).r), max(src.read(s + uint2(0, 1)).r, src.read(s + uint2(1, 1)).r));
    dst.write(float4(m, 0, 0, 0), gid);
}

// (ground, lowest underside) of a column: an island's ground and the underside over it, else its top and none.
static inline float2 isleGroundUnder(float4 c, uint4 gx) {
    return c.a == 0.0 && c.b > c.g && (gapFlags(gx) & GAP_ISLAND) != 0u ? c.gb : float2(c.r, 4096.0);
}

// The CPU readback's mips. r: the highest column below (the camera steers by it); g: the lowest solid floor below
// (the birds' sightline test); b: the highest ground below, islands aside; a: the lowest island underside (4096: none).
kernel void reduceCpu(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      texture2d<uint, access::read> gx [[texture(2)]],
                      constant int& fromColumns [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float4 a = src.read(s), b = src.read(s + uint2(1, 0)), c = src.read(s + uint2(0, 1)), d = src.read(s + uint2(1, 1));
    float m = max(max(a.r, b.r), max(c.r, d.r));
    if (fromColumns == 0) {
        dst.write(float4(m, min(min(a.g, b.g), min(c.g, d.g)), max(max(a.b, b.b), max(c.b, d.b)), min(min(a.a, b.a), min(c.a, d.a))), gid);
        return;
    }
    uint4 ga = gx.read(s), gb = gx.read(s + uint2(1, 0)), gc = gx.read(s + uint2(0, 1)), gd = gx.read(s + uint2(1, 1));
    float lo = min(min(solidFloor(a, ga), solidFloor(b, gb)), min(solidFloor(c, gc), solidFloor(d, gd)));
    float2 ia = isleGroundUnder(a, ga), ib = isleGroundUnder(b, gb), ic = isleGroundUnder(c, gc), id = isleGroundUnder(d, gd);
    dst.write(float4(m, lo, max(max(ia.x, ib.x), max(ic.x, id.x)), min(min(ia.y, ib.y), min(ic.y, id.y))), gid);
}
