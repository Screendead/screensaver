struct Hit {
    bool hit;
    float t;
    int3 voxel;     // xz in frame-local cells of `level`, y in voxels
    int3 normal;
    int iters;
    int level;
};

static inline float3 columnAt(texture2d<float> hm, int2 c, int size) {
    c = clamp(c, int2(0), int2(size - 1));
    return hm.read(uint2(c), 0).rgb;
}

// The principal gap only: the extra gaps below it (notches, tubes) read as rock to corner AO, which only darkens their
// already dark corners a little more; decoding them in every AO tap cost a tenth of a millisecond.
static inline bool solid(texture2d<float> hm, int3 v, int size) {
    float3 col = columnAt(hm, v.xz, size);
    float y = float(v.y);
    return y < col.x && !(y >= min(col.y, col.z) && y < max(col.y, col.z));
}

// The LOD rings. Level 0 is the 1-voxel map (texel = frame-local column); level k >= 1 is slice k - 1 of the coarse
// array, columns 2^k wide, addressed by world cell & 2047. Positions are frame-local (relative to the map origin,
// a multiple of 64), so a level's cells and max mips are world-aligned and the origin in level cells is exact.
struct Terrain {
    texture2d<float> hm;
    texture2d_array<float> ch;   // coarse height, with max mips that also bound the next level (its morph target)
    texture2d_array<float> cv;   // coarse canopy cover
    texture2d_array<float> cc;   // coarse climate, half resolution
    texture2d<float> clim;
    constant float4* lod;        // Frame.lod
    int2 origin;
    int size;
    int levels;
    float2 cam;                  // camera xz, frame-local
    float rmax;                  // the world's edge: a circle around the camera
    float fadeStart;             // where the haze starts closing in on that edge
    float globalMax;
    texture2d<float> hx;         // r16Float: mip 0 the column tops, mips from 1 their maxima (the DDA's skips)
    texture2d<uint> gx;          // the 1-voxel map's extra gaps (packGaps)
    texture2d_array<float> cg;   // levels 1...GAP_LEVELS: each column's gap (lo, hi)
    texture2d_array<float> cg2;  // levels 1...GAP2_LEVELS: a second gap below it (a giant crown's air)
    texture2d<float> hb;         // band mips (bandTop): the air a cell leaves open under canopies
    uint seed;
    device const GiantRec* giants; // the giant table (placeGiants)
    texture2d<float> pv;           // the province fields (bakeProvinces)
    constant TowerTable* towers; // the ruined towers nearest the camera (towerN of them; none outside the scene pass)
    int towerN;
    texture2d_array<float> cb;   // coarse band mips (reduceMaxParent): slice k - 1, mip m - 1 holds the air of max mip m's cells
    float caves;                 // 1 while a cave lies within the world's edge (scene pass only)
    float caveNight;             // how far night has come for the glow deep in caves (bioLandRamp)
    float caveFire;              // a lived-in cave's hearth's level: low by day, banked late at night (lampLevel, updateLamps)
};

static inline int2 levelOrigin(thread const Terrain& tr, int k) { return k == 0 ? int2(0) : tr.origin >> k; }

// Whether column c's principal gap (col: columnL) is an island's air: on the 1-voxel map a rock gap at least ISLE_AIR
// tall (no other gap is), further out its floor's sign.
static inline bool isleGapL(thread const Terrain& tr, int k, int2 c, float4 col) {
    if (k == 0) return col.a == 0.0 && col.z - col.y >= ISLE_AIR;
    if (k > GAP_LEVELS) return false;
    return tr.cg.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).r < 0.0;
}

// Column (H, gapLo, gapHi, first leaf) of frame-local level cell c, unmorphed.
static inline float4 columnL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return tr.hm.read(uint2(clamp(c, int2(0), int2(tr.size - 1))), 0);
    uint2 t = uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1));
    float H = tr.ch.read(t, uint(k - 1), 0).r;
    if (k > GAP_LEVELS) return float4(H, 0, 0, 0);
    float2 g = tr.cg.read(t, uint(k - 1)).rg;
    return float4(H, abs(g.x), g.y, 0);
}



// Terrain height under any tree, as a point at height y sees it: under an island, the island's own ground, not its top.
static inline float groundL(thread const Terrain& tr, int k, int2 c, float y) {
    if (k == 0) {
        float4 col = tr.hm.read(uint2(clamp(c, int2(0), int2(tr.size - 1))), 0);
        // A select, not a branch: this sits in the shading's many-tap slope filters.
        return (col.a > 0.0 || (SCENE_NEAR && y < col.b && col.a == 0.0 && col.b - col.g >= ISLE_AIR)) ? col.g : col.r;
    }
    uint2 t = uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1));
    float H = tr.ch.read(t, uint(k - 1), 0).r;
    if (SCENE_NEAR && k <= GAP_LEVELS && y < H - 3.0) {
        float2 g = tr.cg.read(t, uint(k - 1)).rg;
        if (g.x < 0.0 && y < g.y) return -g.x;
    }
    return H;
}

// Ground bilinearly interpolated between column centres: the continuous slope under the stairs.
static inline float groundSmoothL(thread const Terrain& tr, int k, float2 p, float y) {
    float2 q = p / tr.lod[k].w - 0.5, f = fract(q);
    int2 i = int2(floor(q));
    return mix(mix(groundL(tr, k, i, y), groundL(tr, k, i + int2(1, 0), y), f.x),
               mix(groundL(tr, k, i + int2(0, 1), y), groundL(tr, k, i + int2(1, 1), y), f.x), f.y);
}
// As the sea sees it: under an island, the bed.
static inline float groundSmoothL(thread const Terrain& tr, int k, float2 p) { return groundSmoothL(tr, k, p, SEA); }

// (cover, depth of the leaves under the column's top in voxels: coarseVeg; 64 on a giant's columns) of coarse column c.
static inline float2 coverDepthL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return 0.0;
    float2 v = tr.cv.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1), 0).rg;
    return float2(v.x, floor(v.y * 64.0 + 0.5));
}

// Height of the next level's column over level-k cell c.
static inline float parentH(thread const Terrain& tr, int k, int2 c) {
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.ch.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k), 0).r;
}

// Gap (lo, hi) of the next level's column over level-k cell c.
// The next level's second gap over level-k cell c (a giant crown's air under it; zero elsewhere).
static inline float2 parentGap2(thread const Terrain& tr, int k, int2 c) {
    if (k + 1 > GAP2_LEVELS) return float2(0);
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.cg2.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k)).rg;
}

// Its floor negative where it is an island's air.
static inline float2 parentGapS(thread const Terrain& tr, int k, int2 c) {
    if (k + 1 > GAP_LEVELS) return float2(0);
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.cg.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k)).rg;
}
static inline float2 parentGap(thread const Terrain& tr, int k, int2 c) {
    float2 g = parentGapS(tr, k, c);
    return float2(abs(g.x), g.y);
}


// Morph weight toward the next level at frame-local xz p: 0 inside the band, 1 from 1.5 parent columns before the
// handoff circle on, so every parent cell the circle cuts is already fully morphed on the inner side.
static inline float morphAt(thread const Terrain& tr, int k, float2 p) {
    if (k >= tr.levels - 1) return 0.0;
    return smoothstep(tr.lod[k].x, tr.lod[k].z, length(p - tr.cam));
}

// Whether world xz w lies on a tower's plinth.
static inline bool towerWallAt(constant TowerTable* tt, float2 w) { return towerIndexAt(tt, w) >= 0; }

// Whether level-k cell c (k <= GAP2_LEVELS) holds a giant crown's air as its second gap.
static inline bool giantAirAt(thread const Terrain& tr, int k, int2 c) {
    float2 g = tr.cg2.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).rg;
    return g.y > g.x;
}

// A giant's column across the morph band, on the 1-voxel map (a giant's column, or any under a parent holding a crown's
// air) and on the last level with a second gap (a column holding one): (H, principal gap, the height under which the
// others lie) and the others (the 1-voxel map's extras, or the second gap in xy). A column with at most one gap whose
// parent has as many, its gap and the run over it each overlapping the parent's, slides to the parent: nothing in it
// moves through the air. Any other steps whole, H with its gaps, to its parent at a point of the band of its own.
// Opening or closing a gap there (a gap under a limb beside the crown's air over it, a trunk's edge, bare ground under a
// crown's rim) hung a run in the air that slid across the band; sliding a run between two gaps toward the parent's broke
// a limb into pieces at each column's height; and a jungle crown under a giant's rim rose through the air to its crown.
struct BandCol { float4 col; float4 e; };

static inline bool giantBandAt(thread const Terrain& tr, int k, int2 c, float4 cl) {
    if (k == 0) { float2 p2 = parentGap2(tr, 0, c); return cl.a == GIANT_A || p2.y > p2.x; }
    if (k != GAP2_LEVELS) return false;
    // Or one under a parent of the giant's own (cover depth 64): bare ground under the crown's rim would otherwise rise to
    // the parent's crown with its air opening beneath, a slab crossing the band in mid-air.
    int2 w = (c + (tr.origin >> k)) >> 1;
    return giantAirAt(tr, k, c) || tr.cv.read(uint2(w & (COARSE_N - 1)), uint(k), 0).g > 0.99;
}

static BandCol giantBand(thread const Terrain& tr, int k, int2 c, float4 cl, float w) {
    float4 e = k == 0 ? extraGaps(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1)))))
                      : float4(tr.cg2.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).rg, 0, 0);
    float2 cg[3], pp[2];
    int nc = 0, np = 0;
    // A coarse column files a lone gap in both slots; the 1-voxel map's principal slot holds one inverted.
    float2 cs[3] = { e.zw, e.xy, cl.yz };
    for (int i = 0; i < 3; i++)
        if (cs[i].y > cs[i].x && !(nc > 0 && all(cs[i] == cg[nc - 1]))) cg[nc++] = cs[i];
    float2 ps[2] = { parentGap2(tr, k, c), parentGap(tr, k, c) };
    for (int i = 0; i < 2; i++)
        if (ps[i].y > ps[i].x && !(np > 0 && all(ps[i] == pp[np - 1]))) pp[np++] = ps[i];
    for (int i = 1; i < nc; i++)
        for (int j = i; j > 0 && cg[j].x < cg[j - 1].x; j--) { float2 t = cg[j]; cg[j] = cg[j - 1]; cg[j - 1] = t; }
    if (np == 2 && pp[1].x < pp[0].x) { float2 t = pp[0]; pp[0] = pp[1]; pp[1] = t; }
    float ph = parentH(tr, k, c);
    bool pair = nc == np && nc <= 1 && (nc == 0 || (cg[0].x < pp[0].y && pp[0].x < cg[0].y && cg[0].y < ph && pp[0].y < cl.x));
    float H = cl.x;
    if (pair) {
        H = mix(cl.x, ph, w);
        for (int i = 0; i < nc; i++) cg[i] = mix(cg[i], pp[i], w);
    } else if (w > 0.1 + 0.8 * hashf(hash2i(c + (k == 0 ? tr.origin : tr.origin >> k), 913u))) {
        H = ph;
        nc = np;
        for (int i = 0; i < np; i++) cg[i] = pp[i];
    }
    BandCol r;
    float2 top = nc > 0 ? cg[nc - 1] : float2(0);
    r.col = float4(H, top, top.y);
    r.e = float4(nc > 1 ? cg[nc - 2] : float2(0), nc > 2 ? cg[0] : float2(0));
    return r;
}

// Morphed (H, gapLo, gapHi): across the band a column slides to its parent's height and its gap to the parent's (opening
// from the parent's floor where it has none), or closes from the top where the parent has none, so at the handoff the
// level equals the next one exactly; a giant's column as giantBand has it, which also gives its other gaps (ext, with
// got set). Where either is an island's air, the air at every stage lies inside both columns' instead, so the bands
// under islands hold through the morph: the floor slides from one floor to the other (a top, where a column has no
// gap) and the roof stays the one either has.
// w: the morph weight (columnMorphW).
// w of the result: the height under which the column's extra gaps can lie (the top of its unmorphed principal gap, or of
// its morphed one's floor).
static inline float4 columnMorphE(thread const Terrain& tr, int k, int2 c, float w, thread float4& ext, thread bool& got) {
    float4 cl = columnL(tr, k, c);
    got = false;
    if (w > 0.0 && k <= GAP2_LEVELS && giantBandAt(tr, k, c, cl)) {
        BandCol b = giantBand(tr, k, c, cl, w);
        ext = b.e;
        got = true;
        return b.col;
    }
    float4 col = float4(cl.rgb, max(cl.g, cl.b));
    if (w > 0.0) {
        float ph = parentH(tr, k, c);
        // A column sliding up or down a tower's sheer wall stands part way up it as a sliver, and the band of them rules
        // the wall with stripes: there it steps at the middle of the band instead.
        float wh = w;
        if (tr.towerN > 0 && abs(ph - col.x) > 12.0 && towerWallAt(tr.towers, (float2(c) + 0.5) * tr.lod[k].w + float2(tr.origin)))
            wh = step(0.5, w);
        float2 pg = parentGapS(tr, k, c);
        bool cv = col.z > col.y;
        // Without the islands' surfaces no island shows: its air morphs as a rock gap's.
        bool isl = ISLE_SHAPE && (pg.x < 0.0 || (cv && isleGapL(tr, k, c, cl)));
        pg.x = abs(pg.x);
        bool pv = pg.y > pg.x;
        if (isl) {
            float lo = mix(cv ? col.y : col.x, pv ? pg.x : ph, w);
            float hi = cv && pv ? mix(col.z, pg.y, w) : cv ? col.z : pv ? pg.y : lo;
            col.x = mix(col.x, ph, wh);
            if (hi >= col.x) col.xyz = float3(min(col.x, lo), 0, 0);
            else col.yz = hi > lo ? float2(lo, hi) : float2(0);
        } else {
            col.x = mix(col.x, ph, wh);
            if (pv) {
                float2 g0 = cv ? col.yz : pg.xx;
                col.yz = mix(g0, pg, w);
            } else {
                col.z = mix(col.z, col.y, w);
            }
        }
        col.w = max(col.w, col.y);
    }
    return col;
}

static inline float4 columnMorphW(thread const Terrain& tr, int k, int2 c, float w) {
    float4 ext;
    bool got;
    return columnMorphE(tr, k, c, w, ext, got);
}

// A coarse column's gap below the principal one, morphed by w: on levels 1...GAP2_LEVELS a giant crown's air under it
// (cg2), else none. Only a point under the principal gap can be in it.
static inline float2 secondGapW(thread const Terrain& tr, int k, int2 c, float w) {
    float2 g = k <= GAP2_LEVELS ? tr.cg2.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).rg : float2(0);
    if (w <= 0.0) return g;
    if (k == GAP2_LEVELS && g.y > g.x) return giantBand(tr, k, c, columnL(tr, k, c), w).e.xy;
    return float2(g.x, mix(g.y, g.x, w));
}

static inline float3 columnMorph(thread const Terrain& tr, int k, int2 c) {
    return columnMorphW(tr, k, c, morphAt(tr, k, (float2(c) + 0.5) * tr.lod[k].w)).xyz;
}

// The 1-voxel map's extra gaps (lo1, hi1, lo2, hi2) at frame-local column c, closing from the top across the morph band
// (the coarser levels carry only the principal gap); a giant's as giantBand has them.
static inline float4 extraGapsMorph(thread const Terrain& tr, int2 c) {
    float w = morphAt(tr, 0, float2(c) + 0.5);
    if (w > 0.0) {
        float4 cl = columnL(tr, 0, c);
        if (giantBandAt(tr, 0, c, cl)) return giantBand(tr, 0, c, cl, w).e;
    }
    float4 e = extraGaps(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1)))));
    if (w > 0.0) e.yw = mix(e.yw, e.xz, w);
    return e;
}

static inline Climate climateL(thread const Terrain& tr, int k, float2 p) {
    if (k == 0) return climateAt(tr.clim, p, tr.pv, tr.origin);
    constexpr sampler bl(filter::linear, address::repeat);
    float2 u = (p / (2.0 * tr.lod[k].w) + float2((tr.origin >> (k + 1)) & (COARSE_N / 2 - 1))) / float(COARSE_N / 2);
    float4 v = tr.cc.sample(bl, u, uint(k - 1), level(0));
    Climate c;
    c.temp = v.x; c.moist = v.y; c.fall = v.z; c.basin = v.w;
    float3 pv = provincesBaked(tr.pv, p + float2(tr.origin));
    c.jungle = pv.x; c.old = pv.y; c.isle = pv.z;
    return c;
}

// The level a point at distance r from the camera belongs to (levels: past the world's edge).
static inline int levelAt(thread const Terrain& tr, float r) {
    int k = 0;
    while (k < tr.levels && r >= tr.lod[k].y) k++;
    return k;
}

// Ray parameter at which o + d t leaves the circle of radius R around the camera in xz (0 if never inside).
static inline float circleExit(float2 o, float2 dxz, float2 cam, float R) {
    float2 q = o - cam;
    float a = dot(dxz, dxz), b = dot(q, dxz), c = dot(q, q) - R * R;
    if (a < 1e-12) return c < 0.0 ? 1e30 : 0.0;
    float disc = b * b - a * c;
    return disc <= 0.0 ? 0.0 : (-b + sqrt(disc)) / a;
}
