// Old-growth conifers, redwood and Douglas fir: one to an OLD_CELL cell, straight trunks two or three voxels thick
// standing 34-62 tall (a veteran in thirty, 75-95), a clear bole for the lower half, then a narrow crown of drooping
// whorls, each a skirt wider at its foot, ragged by branch, so the outer columns of a crown are a stack of shelves with
// sky between them.
constant float OLD_CELL = 12.0;
constant int OLD_TIERS = 5;

// The tree of cell cc, if it has one: trunk (world xz, column centre), height, crown radius, trunk radius.
static inline bool oldTreeIn(int2 cc, uint seed, float O, thread float2& ctr, thread float& Ht, thread float& R, thread float& tr, thread uint& h) {
    h = hash2i(cc, seed + 350u);
    if (hashf(h) > 0.85 * smoothstep(0.2, 0.55, O)) return false;
    ctr = floor((float2(cc) + 0.2 + 0.6 * float2(hashf(pcg(h + 1u)), hashf(pcg(h + 2u)))) * OLD_CELL) + 0.5;
    bool vet = hashf(pcg(h + 3u)) < 0.035;
    Ht = vet ? 75.0 + 20.0 * hashf(pcg(h + 4u)) : 34.0 + 28.0 * hashf(pcg(h + 4u));
    // The forest rises toward the province's heart, so the flight meets it as a slope, not a wall.
    Ht *= mix(0.4, 1.0, smoothstep(0.2, 0.8, O));
    R = (vet ? 7.5 : 4.5) + 0.03 * Ht * hashf(pcg(h + 5u));
    tr = vet ? 1.9 : Ht > 62.0 ? 1.2 : 0.8;
    return true;
}

// The ground's range over the four columns 3 from a trunk past which no old tree stands: they climb mountain flanks
// (to 3 voxels a voxel), not a massif's face, nor (wallClear) the aprons under one. OLD_GENTLE: ground no wall test asks.
constant float OLD_CLIFF = 18.0, OLD_GENTLE = 6.0;

// Whether an old tree's trunk at world column c has a site (not on a cliff), and its ground gt.
template <typename G>
static bool oldSite(int2 c, thread const G& ground, uint seed, thread float& gt) {
    gt = ground.at(c);
    float s0 = ground.at(c + int2(3, 0)), s1 = ground.at(c - int2(3, 0));
    float s2 = ground.at(c + int2(0, 3)), s3 = ground.at(c - int2(0, 3));
    float r = max(max(s0, s1), max(s2, s3)) - min(min(s0, s1), min(s2, s3));
    return r <= OLD_GENTLE || (r <= OLD_CLIFF && wallClear(c, ground, seed));
}

// The intervals old-growth tree (trunk ground gt) adds to world column centre w; lod >= 1: crowns only, as a top.
static float oldTreeIntervals(float2 w, float2 ctr, float gt, float Ht, float R, float tr, uint h, uint seed, bool add, thread GRuns& r) {
    float2 q = w - ctr;
    float d = length(q);
    float top = -1e9;
    if (d < tr) {
        if (add) gAdd(r, gt, gt + Ht - 2.0, GM_WOOD);
        top = gt + Ht - 2.0;
    }
    if (d >= R * 1.15) return top;
    float th = atan2(q.y, q.x);
    float yb = gt + Ht * (0.42 + 0.12 * hashf(pcg(h + 6u)));
    float L = gt + Ht - yb;
    for (int i = 0; i < OLD_TIERS; i++) {
        float t0 = float(i) / float(OLD_TIERS), t1 = float(i + 1) / float(OLD_TIERS);
        float ri = R * pow(1.0 - t0, 0.8);
        // Branches: the skirt's edge ragged by angle, differently in each whorl.
        ri *= 0.72 + 0.4 * saturate(perlin2(float2(th * 2.2, float(i) * 3.1 + hashf(h) * 40.0), seed + 351u) + 0.5);
        if (d >= ri) continue;
        float y0 = yb + L * t0, hT = L * (t1 - t0) * 1.15;
        float f = min(1.0, (1.0 - d / ri) / 0.6);
        // A whorl droops: its outer columns start lower.
        float droop = 1.5 * (d / ri);
        if (add) gAdd(r, floor(y0 - droop), floor(y0 - droop + max(hT * f, 1.0)) + 1.0, GM_LEAF);
        top = max(top, floor(y0 - droop + max(hT * f, 1.0)) + 1.0);
    }
    // The leader.
    if (d < 0.8) { if (add) gAdd(r, gt + Ht - 3.0, gt + Ht + 2.0, GM_LEAF); top = max(top, gt + Ht + 2.0); }
    return top;
}
