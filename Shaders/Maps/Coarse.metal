// Coarse level k (1...4) has columns s = 2^k voxels wide, built from the same terrainColumn field: the mean of
// four point samples, one per child column, so no coarse level carries trees or arches.
constant int COARSE_N = 2048;
// Seen at a grazing angle, trees standing several voxels tall hide far more ground than their footprint covers.
constant float COVER_GAIN = 2.5;

// Levels 1 to GAP_LEVELS carry one gap per column, the carved principal gaps of their four samples: the rounded means of
// those there are, where they overlap, under at least two voxels of roof (a coarse column hollow wherever any sample is,
// or a recess's ragged back wall stands in it as a row of pillars). Their outline (H) ignores gaps.
constant int GAP_LEVELS = 4;
// Of those, levels 1...GAP2_LEVELS also carry a second gap below it (a giant's crown: the air under it, with the gap
// between its tiers as the principal one); further out a giant keeps only the air under its crown. Must match
// Clipmap.gap2Levels in Sources/Strata/World/Clipmap.swift.
constant int GAP2_LEVELS = 1;

static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles, thread float2& g2, thread float& depth) {
    g2 = 0.0;
    depth = 0.0;
    int s = 1 << k, off = s >> 2, hs = s >> 1;
    int2 b = cw * s + off;
    device const CaveRec* caves = caveTable(giants);
    Col c0 = terrainColumnC(b, seed, s, caves, isles), c1 = terrainColumnC(b + int2(hs, 0), seed, s, caves, isles);
    Col c2 = terrainColumnC(b + int2(0, hs), seed, s, caves, isles), c3 = terrainColumnC(b + int2(hs, hs), seed, s, caves, isles);
    float H = coarseCombine(c0.h, c1.h, c2.h, c3.h, float(s));
    // A giant: where at least two samples stand in it, the mean of their tops, and under it air where most of them have
    // air at that height (a mean of their gaps paired a gap under a limb with the crown's air beside it and hung a slab
    // between them). Air must be a hanging root's too, so each root keeps its length and stands in the column; a buttress
    // sample raises the air's floor. Of the gaps that leaves, the principal is the highest and the second the one below it;
    // past GAP2_LEVELS the one gap is the lowest, the crown's air.
    Giant gi;
    if (giantAt(giants, float2(b + hs / 2) + 0.5, float(s), gi)) {
        float hs4[4] = { c0.h, c1.h, c2.h, c3.h };
        int2 o4[4] = { int2(0), int2(hs, 0), int2(0, hs), int2(hs, hs) };
        float nT = 0.0, sH = 0.0, nTop = 0.0, sLowH = 0.0, floorH = -1e9;
        float2 sg[4][3];
        float sTop[4];
        bool sThin[4];
        float bnd[30];
        int nb = 0;
        for (int i = 0; i < 4; i++) {
            GCol gc;
            GRuns rr;
            rr.n = 0;
            rr.thin = false;
            if (!giantColumn(gi, float2(b + o4[i]) + 0.5, hs4[i], rr, gc, k)) continue;
            int v = int(nT);
            nT += 1.0;
            sg[v][0] = gc.e.zw; sg[v][1] = gc.e.xy; sg[v][2] = float2(gc.gl, gc.gh);
            sTop[v] = gc.H;
            sThin[v] = rr.thin;
            bool gap = false;
            for (int j = 0; j < 3; j++) {
                if (sg[v][j].y <= sg[v][j].x) continue;
                gap = true;
                bnd[nb++] = sg[v][j].x; bnd[nb++] = sg[v][j].y;
            }
            bnd[nb++] = gc.H;
            // A buttress or an arching root (solid, low, no crown over it) raises the floor of the crown's air.
            if (!gap && !rr.thin && gc.H < gi.G + 120.0) { floorH = max(floorH, gc.H); sLowH += gc.H; continue; }
            sH += gc.H;
            nTop += 1.0;
        }
        if (nT >= 2.0) {
            // Marks the column as a giant's for shading (coverDepthL).
            depth = -1.0;
            // On levels 1...GAP2_LEVELS a column with no second gap holds an empty one at the top of its wood (roots,
            // buttresses) for giantSurface; an empty gap is no gap to the trace or the morph.
            if (nTop < 1.0) { float t = max(floor(sLowH / nT + 0.5), H); g2 = float2(t); return float3(t, 0, 0); }
            float top = floor(sH / nTop + 0.5);
            float fl = max(floorH, H);
            g2 = float2(max(floorH, 0.0));
            if (k > GAP_LEVELS) return float3(max(top, H), 0, 0);
            bnd[nb++] = fl;
            bnd[nb++] = top;
            for (int i = 1; i < nb; i++) {
                float a = bnd[i];
                int j = i - 1;
                while (j >= 0 && bnd[j] > a) { bnd[j + 1] = bnd[j]; j--; }
                bnd[j + 1] = a;
            }
            // The air runs between fl and top, bottom up.
            float2 air[6];
            int na = 0;
            bool open = false;
            for (int i = 0; i + 1 < nb; i++) {
                float a = max(bnd[i], fl), c = min(bnd[i + 1], top);
                if (c <= a) continue;
                float y = 0.5 * (a + c);
                float votes = 0.0;
                bool rootSolid = false;
                for (int v = 0; v < int(nT); v++) {
                    bool inAir = y >= sTop[v];
                    for (int j = 0; j < 3; j++) inAir = inAir || (y >= sg[v][j].x && y < sg[v][j].y);
                    votes += inAir ? 1.0 : 0.0;
                    rootSolid = rootSolid || (sThin[v] && !inAir);
                }
                bool isAir = 2.0 * votes > nT && !rootSolid;
                if (isAir && open) air[na - 1].y = c;
                else if (isAir && na < 6) { air[na++] = float2(a, c); open = true; }
                else open = false;
            }
            for (int i = 0; i < na; i++) air[i] = floor(air[i] + 0.5);
            // Air up to the top, or under less than a voxel of roof, leaves none: the column ends under it (dropped,
            // it stood a pillar from the ground to the crown).
            if (na > 0 && top - air[na - 1].y < 1.0) { top = max(air[na - 1].x, H); na--; }
            int m = 0;
            for (int i = 0; i < na; i++) if (air[i].y - air[i].x >= 2.0) air[m++] = air[i];
            if (m == 0) return float3(max(top, H), 0, 0);
            // More gaps than the level keeps: two over a run as thin as a limb's tip or an arching root join, so no run
            // is left standing in the air; past that the shortest gap over the crown's air fills, joining the runs
            // round it (joined across a tier of the crown, they left a hole through it).
            for (int keep = k > GAP2_LEVELS ? 1 : 2; m > keep; m--) {
                int j = 0;
                for (int i = 1; i < m - 1; i++) if (air[i + 1].x - air[i].y < air[j + 1].x - air[j].y) j = i;
                if (air[j + 1].x - air[j].y < 8.0) {
                    air[j].y = air[j + 1].y;
                    j++;
                } else {
                    j = 1;
                    for (int i = 2; i < m; i++) if (air[i].y - air[i].x < air[j].y - air[j].x) j = i;
                }
                for (int i = j; i < m - 1; i++) air[i] = air[i + 1];
            }
            g2 = air[0];
            return float3(top, air[m - 1]);
        }
    }
    // An island's canopy is in its samples' tops already (isleSpan): the depth of leaves is down to their ground's
    // combined top. A column most of whose island samples are water (a stream, a fall) is marked by ISLE_WATER_DEPTH.
    bool isl = c0.island || c1.island || c2.island || c3.island;
    float2 vg;
    if (isl) {
        float g0 = c0.isk == ISK_ROCK && c0.leaf > 0.0 ? c0.leaf : c0.h, g1 = c1.isk == ISK_ROCK && c1.leaf > 0.0 ? c1.leaf : c1.h;
        float g2 = c2.isk == ISK_ROCK && c2.leaf > 0.0 ? c2.leaf : c2.h, g3 = c3.isk == ISK_ROCK && c3.leaf > 0.0 ? c3.leaf : c3.h;
        float dv = H - coarseCombine(g0, g1, g2, g3, float(s));
        float nw = float(c0.isk == ISK_WATER) + float(c1.isk == ISK_WATER) + float(c2.isk == ISK_WATER) + float(c3.isk == ISK_WATER);
        float ni = float(c0.island) + float(c1.island) + float(c2.island) + float(c3.island);
        float nf = float(c0.isk == ISK_WATER && c0.leaf > 0.0) + float(c1.isk == ISK_WATER && c1.leaf > 0.0)
                 + float(c2.isk == ISK_WATER && c2.leaf > 0.0) + float(c3.isk == ISK_WATER && c3.leaf > 0.0);
        vg = float2(H, nw * 2.0 > ni ? (nf * 2.0 >= nw ? ISLE_WATER_DEPTH : ISLE_STREAM_DEPTH) : clamp(dv, 0.0, 40.0));
    } else {
        // No trees take root in or beside a cave on the 1-voxel map (plantTrees keeps off carved rock): nor here.
        TerrainGround tg { seed };
        vg = c0.cave || c1.cave || c2.cave || c3.cave ? float2(H, 0) : coarseVeg(cw, k, float4(c0.h, c1.h, c2.h, c3.h), H, seed, tg);
    }
    depth = vg.y;
    float Hv = vg.x;
    // An island keeps its air out to the last ring, where other carved gaps are dropped.
    if (k > GAP_LEVELS || (k > 3 && !isl)) return float3(Hv, 0, 0);
    float n = 0, lo = 0, hi = 0, maxLo = 0, minHi = 1e9;
    if (c0.carved && c0.gh > c0.gl) { n += 1; lo += c0.gl; hi += c0.gh; maxLo = max(maxLo, c0.gl); minHi = min(minHi, c0.gh); }
    if (c1.carved && c1.gh > c1.gl) { n += 1; lo += c1.gl; hi += c1.gh; maxLo = max(maxLo, c1.gl); minHi = min(minHi, c1.gh); }
    if (c2.carved && c2.gh > c2.gl) { n += 1; lo += c2.gl; hi += c2.gh; maxLo = max(maxLo, c2.gl); minHi = min(minHi, c2.gh); }
    if (c3.carved && c3.gh > c3.gl) { n += 1; lo += c3.gl; hi += c3.gh; maxLo = max(maxLo, c3.gl); minHi = min(minHi, c3.gh); }
    if (n < 1.0 || maxLo >= minHi) return float3(Hv, 0, 0);
    lo = floor(lo / n + 0.5);
    hi = floor(hi / n + 0.5);
    if (hi - lo < 2.0 || H - hi < 2.0) return float3(Hv, 0, 0);
    // An island's air is flagged by its floor's sign (columnL and parentGap read the magnitude).
    return float3(Hv, isl ? -lo : lo, hi);
}
static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles) { float2 g2; float dp; return coarseColumn(cw, k, seed, giants, isles, g2, dp); }

// Writes the world cells region.xy + [0, region.zw) of level info.x into its toroidal slice (texel = cell & 2047):
// height and canopy cover, and every second cell also the half-resolution climate texel it starts.
kernel void generateCoarse(texture2d<float, access::write> hOut [[texture(0)]],
                           texture2d<float, access::write> cOut [[texture(1)]],
                           texture2d<float, access::write> vOut [[texture(2)]],
                           texture2d<float, access::write> gOut [[texture(3)]],
                           texture2d<float, access::write> g2Out [[texture(4)]],
                           constant int4& region [[buffer(0)]],
                           constant int4& info [[buffer(1)]],
                           device const GiantRec* giants [[buffer(2)]],
                           device const IsleRec* isles [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int k = info.x, s = 1 << k;
    uint seed = uint(info.y);
    int2 cw = region.xy + int2(gid);
    float2 cc2;
    float depth;
    float3 cc = coarseColumn(cw, k, seed, giants, isles, cc2, depth);
    float H = cc.x;
    if (k <= GAP_LEVELS) gOut.write(float4(cc.y, cc.z, 0, 0), uint2(cw & (COARSE_N - 1)));
    if (k <= GAP2_LEVELS) g2Out.write(float4(cc2, 0, 0), uint2(cw & (COARSE_N - 1)));
    float2 centre = float2(cw * s) + 0.5 * float(s);
    // Expected share of the column under leaves: planting chance per tree cell times a mean canopy of ~13 columns. An
    // island carries its own (its depth of leaves).
    float cover = cc.y < 0.0 ? 0.0 : saturate(groveDensity(centre, H, seed) * 13.0 / float(TREE_CELL * TREE_CELL));
    hOut.write(float4(H), uint2(cw & (COARSE_N - 1)));
    // g: the depth of leaves under the top (coarseVeg), in units of 64 voxels.
    vOut.write(float4(cover, depth < 0.0 ? 1.0 : min(depth, 63.0) / 64.0, 0, 0), uint2(cw & (COARSE_N - 1)));
    if (all((cw & 1) == 0)) {
        int2 cj = cw >> 1;
        Climate c = climate((float2(cj) + 0.5) * float(2 * s), seed);
        cOut.write(float4(c.temp, c.moist, c.fall, c.basin), uint2(cj & (COARSE_N / 2 - 1)));
    }
}

// Mip 1 of level info.x over world mip-1 cells region.xy + [0, region.zw): each is one column of the next level,
// whose height the morph band slides toward, so it bounds that too. band: mip 0 of the level's band pyramid, the air
// those four columns leave open through every morph toward the parent: under an island (theirs or the parent's) both
// columns' air (columnMorphW keeps the morphed air inside both), else the air above the higher of the two tops.
kernel void reduceMaxParent(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            texture2d<float, access::read> cg [[texture(2)]],
                            texture2d<float, access::write> band [[texture(3)]],
                            constant int4& region [[buffer(0)]],
                            constant int4& info [[buffer(1)]],
                            device const GiantRec* giants [[buffer(2)]],
                            device const IsleRec* isles [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int2 c = region.xy + int2(gid);
    int2 s = c * 2;
    float m = max(max(src.read(uint2(s & (COARSE_N - 1))).r, src.read(uint2((s + int2(1, 0)) & (COARSE_N - 1))).r),
                  max(src.read(uint2((s + int2(0, 1)) & (COARSE_N - 1))).r, src.read(uint2((s + int2(1, 1)) & (COARSE_N - 1))).r));
    float3 pc = coarseColumn(c, info.x + 1, uint(info.y), giants, isles);
    dst.write(float4(max(m, pc.x)), uint2(c & (COARSE_N / 2 - 1)));
    bool pIsl = pc.y < 0.0;
    float2 pb = pc.z > abs(pc.y) ? float2(abs(pc.y), pc.z) : float2(pc.x, 4096.0);
    float2 b = float2(-1e4, 4096.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint2 t = uint2((s + int2(i, j)) & (COARSE_N - 1));
            float H = src.read(t).r;
            float2 g = cg.read(t).rg;
            bool isl = g.x < 0.0;
            g.x = abs(g.x);
            float2 cb = float2(max(H, pc.x), 4096.0);
            if (isl || pIsl) {
                float2 o = g.y > g.x ? g : float2(H, 4096.0);
                o = float2(max(o.x, pb.x), min(o.y, pb.y));
                if (o.y - o.x >= 4.0) cb = o;
            }
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
    band.write(float4(b, 0, 0), uint2(c & (COARSE_N / 2 - 1)));
}

// The last level's band mip 0 (it morphs toward nothing): per cell of its mip 1 the air its four columns' islands leave.
kernel void coarseBandTop(texture2d<float, access::read> ch [[texture(0)]],
                          texture2d<float, access::read> cg [[texture(1)]],
                          texture2d<float, access::write> dst [[texture(2)]],
                          constant int4& region [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int2 c = region.xy + int2(gid);
    float2 b = float2(-1e4, 4096.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint2 t = uint2((c * 2 + int2(i, j)) & (COARSE_N - 1));
            float H = ch.read(t).r;
            float2 g = cg.read(t).rg;
            float2 cb = g.x < 0.0 && g.y + g.x >= 4.0 ? float2(-g.x, g.y) : float2(H, 4096.0);
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
    dst.write(float4(b, 0, 0), uint2(c & (COARSE_N / 2 - 1)));
}

kernel void coarseBandReduce(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             constant int4& region [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    uint2 s = g * 2;
    float2 a = src.read(s).rg, b = src.read(s + uint2(1, 0)).rg, c = src.read(s + uint2(0, 1)).rg, d = src.read(s + uint2(1, 1)).rg;
    dst.write(float4(max(max(a.x, b.x), max(c.x, d.x)), min(min(a.y, b.y), min(c.y, d.y)), 0, 0), g);
}

// Negates a coarse max mip's cells (region) whose band is a real one, as bandFlag does on the 1-voxel map: the trace
// reads the band only there.
kernel void coarseBandFlag(texture2d<float, access::read_write> mx [[texture(0)]],
                           texture2d<float, access::read> band [[texture(1)]],
                           constant int4& region [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    float2 b = band.read(g).rg;
    float m = abs(mx.read(g).r);
    mx.write(float4(b.y < 4000.0 && b.y - b.x >= 4.0 ? -m : m), g);
}

// Max mips are read through abs: a negated cell holds a real band (coarseBandFlag).
kernel void reduceMaxRegion(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            constant int4& region [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    uint2 s = g * 2;
    float m = max(max(abs(src.read(s).r), abs(src.read(s + uint2(1, 0)).r)), max(abs(src.read(s + uint2(0, 1)).r), abs(src.read(s + uint2(1, 1)).r)));
    dst.write(float4(m, 0, 0, 0), g);
}
