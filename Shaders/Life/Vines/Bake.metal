// A giant's vines are no part of the column maps, whose rings and morph bands shortened and dropped them: each giant's
// strands are baked once from its own rule (bakeVines) and traced analytically by a pass of their own over the scene
// (traceStrands, vineFragment), so a strand's length and presence are the same at every distance. Near, a strand is
// the 2x2 voxel column it always was; narrower than a pixel, the same set is drawn as its expected cover along the
// ray. The maps and the trace know only strands over a window, so an island's roots could share them.

// A window of VINE_W voxels a side round a giant, 8-aligned on the world grid: strand cells 2 voxels square (the fine
// map: bottom, top; top <= bottom holds none), cells of 8 (coarse) and of 32 (super) summing them.
constant int VINE_W = 512;
constant int VINE_FINE = VINE_W / 2, VINE_COARSE = VINE_W / 8, VINE_SUPER = VINE_W / 32;
// Must match Vines.slots in Sources/Strata/World/Vines.swift.
constant int VINE_SLOTS = 6;

// Must match the records Vines writes (Vines.recordSize, Sources/Strata/World/Vines.swift). a: axis xz (map-local), G, yT; b: the strands' box (map-local) in xz
// (lo) and the strands' lowest bottom and highest top; c: hash, kind, slice; d: window origin xz (world), the box's
// hi xz (map-local).
struct VineRec { float4 a; float4 b; uint4 c; int4 d; };
struct VineTable { int4 n; VineRec r[VINE_SLOTS]; };

// Must match VineBake in Sources/Strata/GPU/VineBake.swift. a: axis xz (world), G, yT; b: window origin xz (world); c: hash, kind,
// seed, slice.
struct VineBake { float4 a; int4 b; uint4 c; };

// Strand heights are summed in bands of VINE_BAND voxels up from the giant's ground: a cell's band bits say where
// along a column its strands lie, so a ray crossing it between two tiers' curtains passes.
constant float VINE_BAND = 20.0;

// A coarse or super cell: bits (coarse: which of its 4x4 strand cells hold a strand, bit y * 4 + x; super: which of its
// 4x4 coarse cells hold any), their count, the ranges of their bottoms and of their tops, and their band bits. A super
// cell also has, per band, which of its coarse cells hold a strand in it: two texels of the bands map, eight bands
// each, two to a word (the lower band in the low half).
static inline uint4 packStrands(uint bits, uint count, float minBot, float maxBot, float minTop, float maxTop, uint bands) {
    return uint4(bits | (count << 16), uint(minBot) | (uint(maxBot) << 16), uint(minTop) | (uint(maxTop) << 16), bands);
}

// The band bits of heights [lo, hi) over ground G.
static inline uint strandBands(float lo, float hi, float G) {
    int a = clamp(int(floor((lo - G) / VINE_BAND)), 0, 15), b = clamp(int(floor((hi - 1.0 - G) / VINE_BAND)), 0, 15);
    return ((2u << uint(b)) - 1u) & ~((1u << uint(a)) - 1u);
}

// The fine map of one giant's window: giantIntervals' strand rule, over the crown's underside.
kernel void bakeVines(texture2d_array<float, access::write> fine [[texture(0)]],
                      constant VineBake& b [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(VINE_FINE) || gid.y >= uint(VINE_FINE)) return;
    Giant g;
    g.c = b.a.xy; g.G = b.a.z; g.yT = b.a.w; g.h = b.c.x; g.kind = int(b.c.y);
    uint seed = b.c.z;
    // The strand cell on the world's even grid.
    int2 vb = (b.b.xy >> 1) + int2(gid);
    float2 wq = float2(vb * 2) + 1.0;
    float2 out = 0.0;
    uint hv = hash2i(vb, g.h + 30u);
    float mask = smoothstep(0.0, 0.35, perlin2((float2(vb) * 2.0 + 1.0) / 19.0, g.h + 31u));
    // A weeping giant trails long curtains; a kapok's crown hardly any.
    if (length(wq - g.c) < GIANT_REACH - 1.0 && hashf(hv) < (g.kind == 2 ? 0.18 : g.kind == 1 ? 0.016 : 0.07) * mask) {
        // The crown's underside over the strand: the highest of its four columns', so it reaches into the leaves of
        // each.
        GRuns r;
        r.n = 0;
        float hi = -1e9;
        for (int i = 0; i < 4; i++) {
            float pb = giantPads(g, float2(vb * 2) + float2(i & 1, i >> 1) + 0.5, r, false);
            if (pb < 1e8) hi = max(hi, pb);
        }
        if (hi > -1e8) {
            float len = 4.0 + (g.kind == 2 ? 55.0 : 30.0) * pow(hashf(pcg(hv)), g.kind == 2 ? 1.3 : 2.0);
            float gl = terrainColumnBare(vb * 2, seed).h;
            out = float2(max(hi - len, gl + 3.0), hi + 1.0);
        }
    }
    fine.write(float4(out, 0, 0), gid, b.c.w);
}

kernel void vineCoarse(texture2d_array<float, access::read> fine [[texture(0)]],
                       texture2d_array<uint, access::write> coarse [[texture(1)]],
                       constant VineBake& b [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(VINE_COARSE) || gid.y >= uint(VINE_COARSE)) return;
    uint slice = b.c.w;
    uint bits = 0u, n = 0u, bands = 0u;
    float minBot = 1e9, maxBot = -1e9, minTop = 1e9, maxTop = -1e9;
    for (uint j = 0; j < 4; j++)
        for (uint i = 0; i < 4; i++) {
            float2 s = fine.read(gid * 4 + uint2(i, j), slice).rg;
            if (s.y <= s.x) continue;
            bits |= 1u << (j * 4 + i);
            n++;
            bands |= strandBands(s.x, s.y, b.a.z);
            minBot = min(minBot, s.x); maxBot = max(maxBot, s.x);
            minTop = min(minTop, s.y); maxTop = max(maxTop, s.y);
        }
    coarse.write(n > 0u ? packStrands(bits, n, minBot, maxBot, minTop, maxTop, bands) : uint4(0), gid, slice);
}

kernel void vineSuper(texture2d_array<uint, access::read> coarse [[texture(0)]],
                      texture2d_array<uint, access::write> sup [[texture(1)]],
                      texture2d_array<uint, access::write> bandMap [[texture(2)]],
                      constant VineBake& b [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(VINE_SUPER) || gid.y >= uint(VINE_SUPER)) return;
    uint slice = b.c.w;
    uint bits = 0u, n = 0u, bands = 0u;
    uint occ[16];
    for (int k = 0; k < 16; k++) occ[k] = 0u;
    float minBot = 1e9, maxBot = -1e9, minTop = 1e9, maxTop = -1e9;
    for (uint j = 0; j < 4; j++)
        for (uint i = 0; i < 4; i++) {
            uint4 c = coarse.read(gid * 4 + uint2(i, j), slice);
            if ((c.x & 0xffffu) == 0u) continue;
            bits |= 1u << (j * 4 + i);
            n += c.x >> 16;
            bands |= c.w;
            for (uint m = c.w; m != 0u; m &= m - 1u) occ[ctz(m)] |= 1u << (j * 4 + i);
            minBot = min(minBot, float(c.y & 0xffffu)); maxBot = max(maxBot, float(c.y >> 16));
            minTop = min(minTop, float(c.z & 0xffffu)); maxTop = max(maxTop, float(c.z >> 16));
        }
    sup.write(n > 0u ? packStrands(bits, n, minBot, maxBot, minTop, maxTop, bands) : uint4(0), gid, slice);
    for (uint h = 0; h < 2; h++) {
        uint4 wd;
        for (uint c = 0; c < 4; c++) wd[c] = occ[h * 8 + c * 2] | (occ[h * 8 + c * 2 + 1] << 16);
        bandMap.write(wd, uint2(gid.x * 2 + h, gid.y), slice);
    }
}

// The window's strands' bounds (Vines reads them back to bound the trace): their lowest bottom and highest top, and the
// super cells they lie in (lo xy, hi xy, in cells).
kernel void vineRange(texture2d_array<uint, access::read> sup [[texture(0)]],
                      device float4* range [[buffer(0)]],
                      constant VineBake& b [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float lo = 1e9, hi = -1e9;
    int2 cLo = int2(VINE_SUPER), cHi = int2(-1);
    for (int j = 0; j < VINE_SUPER; j++)
        for (int i = 0; i < VINE_SUPER; i++) {
            uint4 c = sup.read(uint2(i, j), b.c.w);
            if ((c.x & 0xffffu) == 0u) continue;
            lo = min(lo, float(c.y & 0xffffu));
            hi = max(hi, float(c.z >> 16));
            cLo = min(cLo, int2(i, j));
            cHi = max(cHi, int2(i, j));
        }
    range[b.c.w * 2u] = float4(lo, hi, float(cLo.x), float(cLo.y));
    range[b.c.w * 2u + 1u] = float4(float(cHi.x), float(cHi.y), 0, 0);
}
