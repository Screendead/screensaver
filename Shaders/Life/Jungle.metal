// A jungle is a roof, not a crowd of separate trees: rounded crowns packed one to a JUNGLE_CELL cell over a lower, lumpy
// fill, a dozen and more voxels above the ground and a few thick, opened by clearings that widen toward the province's
// edge until the roof breaks into groves; emergents (one to an EMERGENT_CELL cell, about half of them) break through it
// on tall trunks with flat, layered umbrella crowns. Every part is a closed-form function of position and of the ground
// under it, so the coarse rings rebuild the same forest.
constant float JUNGLE_CELL = 8.0;
constant float EMERGENT_CELL = 26.0;
// Ground range over a 10-voxel window (jungleWindow's +-5) from which the roof steps down, and past which none stands.
// Mountain flanks rise 1.3-3 voxels a voxel (range 17-42): the roof clings to them, lower on the steepest; faces of 5
// and more and a massif's walls (range 60 to hundreds) are bare rock. Emergents stand only on gentle ground (range
// under EMERGENT_STEEP), and the clearings' ferns under FERN_STEEP.
constant float JUNGLE_STEEP = 28.0, JUNGLE_CLIFF = 60.0, EMERGENT_STEEP = 7.0, FERN_STEEP = 13.0;
// In wall country the aprons and benches under a massif's walls are no steeper than a mountain flank, but stand within a
// few voxels of a wall, and from below read as one face with it: no forest stands there where the ground's range over a
// 20-voxel window (+-10) reaches WALL_AT. Mountain flanks of 2:1 and more range past it too, so only wall country asks.
// The cut is whole: a roof thinned to bushes on terraced ground leaves one or two on each tread, a speckle.
constant float WALL_AT = 40.0;

// Ground heights at world columns, as each map judges a forest's site: the trees on level 0 and the coarse rings over
// them must read the same ground, or the forest's edge moves with the ring.
// The 1-voxel map (plantTrees), clamped to its edge.
struct MapGround {
    texture2d<float, access::read> map;
    int2 origin;
    float at(int2 w) const {
        int n = int(map.get_width());
        return map.read(uint2(clamp(w - origin, int2(0), int2(n - 1)))).r;
    }
};
// Raw terrain (coarse rings).
struct TerrainGround {
    uint seed;
    float at(int2 w) const { return terrainColumnBare(w, seed).h; }
};
// The 1-voxel map's raw columns where it has them: reduceMaxBase must judge level 1's forest exactly as the coarse
// kernel does, or mip 1 would not bound the level-1 roof.
struct RawGround {
    texture2d<float, access::read> raw;
    int2 origin;
    uint seed;
    float at(int2 w) const {
        int2 q = w - origin;
        int n = int(raw.get_width());
        return all(q >= 0) && all(q < n) ? raw.read(uint2(q)).r : terrainColumnBare(w, seed).h;
    }
};

// The jungle's ground round world column c: (range, mean) over a 3x3 window 5 apart.
template <typename G>
static float2 jungleWindow(int2 c, thread const G& ground) {
    float lo = 1e9, hi = -1e9, sum = 0.0;
    for (int k = 0; k < 9; k++) {
        float h = ground.at(c + int2(k % 3 - 1, k / 3 - 1) * 5);
        lo = min(lo, h); hi = max(hi, h); sum += h;
    }
    return float2(hi - lo, sum / 9.0);
}

// Whether a forest may stand at world column c for its nearness to a massif's wall.
template <typename G>
static bool wallClear(int2 c, thread const G& ground, uint seed) {
    // Under a quarter (the alpine ranges' fringe of karst) the bar, 4 * WALL_AT, is past any apron's range.
    float wall = wallCountry(float2(c) + 0.5, seed);
    if (wall < 0.25) return true;
    float lo = 1e9, hi = -1e9;
    for (int k = 0; k < 9; k++) {
        float h = ground.at(c + int2(k % 3 - 1, k / 3 - 1) * 10);
        lo = min(lo, h); hi = max(hi, h);
    }
    return (hi - lo) * wall < WALL_AT;
}

// The share of the jungle's roof that stands at world column c, whose jungleWindow range is steep (0: none).
template <typename G>
static float jungleKeep(int2 c, float steep, thread const G& ground, uint seed) {
    if (steep < EMERGENT_STEEP) return 1.0;
    if (steep >= JUNGLE_CLIFF || !wallClear(c, ground, seed)) return 0.0;
    return 1.0 - smoothstep(JUNGLE_STEEP, JUNGLE_CLIFF, steep);
}

struct JCan { float lo, hi; bool trunk; };

// The roof over world column centre w, on ground gs (smoothed over a few columns); empty (hi <= lo) where open. Crowns
// are cauliflower domes. Toward a clearing, the water or (keep < 1) steep ground the roof steps down through lower and
// lower crowns to bushes standing on the ground: a margin, never a wall of leaves, and no roof on poles seen from the side.
static JCan jungleCanopy(float2 w, float gs, float J, uint seed, float keep = 1.0) {
    JCan o;
    o.lo = 0.0; o.hi = 0.0; o.trunk = false;
    if (J < 0.15) return o;
    float2 so = seedOffset(seed), p = w + so;
    // Clearings: about one part in eight deep in the province, most of the land at its edge. Crowns stand only where
    // the roof does: one of the forest's reaching over a clearing would stand there on its own.
    float thr = mix(0.45, -0.38, smoothstep(0.15, 0.8, J));
    float openW = perlin2(p / 52.0, seed + 323u) - thr;
    if (openW < 0.0) return o;
    // e: how far in from the margin (over roughly a dozen voxels), here rather than at each crown's centre, so the roof
    // and its crowns step down together as one slope of leaves.
    keep *= smoothstep(SEA + 2.0, SEA + 10.0, gs);
    float e = smoothstep(0.0, 0.3, openW) * keep;
    float rise0 = 14.0 + 5.0 * perlin2(p / 110.0, seed + 321u);
    float lump = 1.2 * perlin2(p / 2.6, seed + 325u) + 1.0 * perlin2(p / 6.0, seed + 327u);
    float hi = -1e9, lo = 1e9;
    int2 c0 = int2(floor(w / JUNGLE_CELL));
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cc = c0 + int2(i, j);
            uint h = hash2i(cc, seed + 322u);
            float2 ctr = (float2(cc) + 0.2 + 0.6 * float2(hashf(h), hashf(pcg(h)))) * JUNGLE_CELL;
            float R = 3.5 + 4.0 * hashf(pcg(h + 1u));
            float2 dq = w - ctr;
            float d2 = dot(dq, dq);
            if (d2 >= R * R) continue;
            float dome = sqrt(1.0 - d2 / (R * R));
            float lift = 5.0 * hashf(pcg(h + 2u)) - 1.0;
            float rise = mix(0.5, rise0 + lift, e * e);
            float bot = rise - 2.0 - 1.5 * dome;
            hi = max(hi, gs + rise + (mix(0.3, 0.85, e) * R + lump) * dome);
            lo = min(lo, bot < 3.0 ? -1e9 : gs + bot);
            if (bot >= 3.0 && d2 < 0.5 && hashf(pcg(h + 3u)) < 0.3) o.trunk = true;
        }
    float rise = mix(0.0, rise0 - 0.5, e * e);
    hi = max(hi, gs + rise + 1.5 * perlin2(p / 5.0, seed + 324u));
    lo = min(lo, rise < 7.5 ? -1e9 : gs + rise - 4.0);
    if (hi < max(lo, gs) + 1.0) return o;
    o.lo = max(floor(lo + 0.5), -1.0);
    o.hi = max(floor(hi + 0.5), o.lo + 1.0);
    return o;
}

// One emergent's crown clump j (of 3 to 5): centre (xz relative to the trunk, y above its ground) and radius in w,
// thickness in T. The clumps sit round the top one at different heights, so the crown is a cluster of rounded heads
// rather than one flat umbrella.
static inline float4 emergentPad(uint h, int j, float R, float y0, thread float& T) {
    T = 8.0 + 4.0 * hashf(pcg(h + 10u));
    if (j == 0) return float4(0.0, y0, 0.0, R);
    uint hj = pcg(h + 20u + uint(j) * 7u);
    float a = (float(j) + 0.6 * hashf(hj)) * 2.3999632 + hashf(h) * TAU;
    float2 o = float2(cos(a), sin(a)) * R * (0.55 + 0.3 * hashf(pcg(hj)));
    T *= 0.85;
    return float4(o.x, y0 - 5.0 + 7.0 * hashf(pcg(hj + 1u)), o.y, R * (0.55 + 0.25 * hashf(pcg(hj + 2u))));
}

// The emergent of cell cc, if it has one: trunk position (world), its crown's height above the ground, crown radius
// and pad count.
static inline bool emergentIn(int2 cc, uint seed, float J, thread float2& ctr, thread float& y0, thread float& R, thread int& pads, thread uint& h) {
    h = hash2i(cc, seed + 330u);
    if (hashf(h) > 0.13 * smoothstep(0.6, 0.85, J)) return false;
    ctr = floor((float2(cc) + 0.25 + 0.5 * float2(hashf(pcg(h + 1u)), hashf(pcg(h + 2u)))) * EMERGENT_CELL) + 0.5;
    // Only out of the roof at its full height: alone in a clearing or over its margin it would be a crown on a pole.
    if (perlin2((ctr + seedOffset(seed)) / 52.0, seed + 323u) < mix(0.45, -0.38, smoothstep(0.15, 0.8, J)) + 0.4) return false;
    y0 = 24.0 + 9.0 * hashf(pcg(h + 3u));
    R = 6.5 + 4.5 * hashf(pcg(h + 4u));
    pads = 3 + int(hashf(pcg(h + 5u)) * 2.99);
    return true;
}

// Pad j of an emergent's crown over q (from its axis): the pad (emergentPad: P, T), and how far out q lies across its
// noise-edged rim, in [0, 1), or -1 off it.
static inline float emergentPadAt(float2 q, uint h, int j, float R, float y0, uint seed, thread float4& P, thread float& T) {
    P = emergentPad(h, j, R, y0, T);
    float2 dq = q - P.xz;
    float dd = length(dq);
    if (dd > P.w * 1.35) return -1.0;
    float2 u2 = dq / max(dd, 1e-3);
    float edge = P.w * (1.0 + 0.30 * perlin2(u2 * 1.7 + float(j) * 4.1 + hashf(h) * 50.0, seed + 331u));
    if (dd >= edge) return -1.0;
    return dd / edge;
}

// The lumps on an emergent's crown pads at seed-offset world position p.
static inline float emergentLumps(float2 p, uint seed) {
    return 1.6 * perlin2(p / 3.2, seed + 332u) + 1.2 * perlin2(p / 7.0, seed + 333u);
}

// The intervals emergent e (trunk ground gt) adds to world column centre w: its trunk, and its crown's noise-edged,
// domed pads. lod >= 1 (coarse rings): the trunk only where it fills the column.
static void emergentIntervals(float2 w, float2 ctr, float gt, float y0, float R, int pads, uint h, uint seed, int lod, thread GRuns& r) {
    float2 q = w - ctr;
    float d = length(q);
    float tr = R > 9.0 ? 1.2 : 0.75;
    if (d < tr + (lod > 0 ? 0.0 : 0.0)) gAdd(r, gt, gt + y0, GM_WOOD);
    float2 p = w + seedOffset(seed);
    float lumps = emergentLumps(p, seed);
    for (int j = 0; j < pads; j++) {
        float T;
        float4 P;
        float u = emergentPadAt(q, h, j, R, y0, seed, P, T);
        if (u < 0.0) continue;
        // Domed and lumpy on top, a sagging, ragged underside: a cloud of leaves, not a plate.
        float top = gt + P.y + T * 0.6 * sqrt(1.0 - u * u) + lumps * sqrt(1.0 - u * u);
        float bot = gt + P.y - T * 0.55 * pow(1.0 - u * u, 0.7) + 1.2 * perlin2(p / 4.5, seed + 334u);
        if (top > bot + 1.0) gAdd(r, floor(bot), floor(top) + 1.0, GM_LEAF);
        // A limb out to each side pad.
        if (j > 0 && lod == 0) {
            float2 a = float2(0), b = P.xz;
            float t = saturate(dot(q - a, b - a) / max(dot(b - a, b - a), 1e-3));
            if (length(q - (a + (b - a) * t)) < 0.7) {
                float yl = gt + mix(y0 - 6.0, P.y, t);
                gAdd(r, yl - 0.5, yl + 1.0, GM_WOOD);
            }
        }
    }
}
