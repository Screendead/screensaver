// Rock and earth islands over island country (Climate.isle). An island is a landform: terrainColumn raises the column to
// the island's top over a gap from its ground to the underside (flagged GAP_ISLAND), so the 1-voxel map, every ring, the
// max mips, the horizon, the shadow maps and the CPU copy all carry it. Islands never overlap in plan, since a coarse
// column holds one gap. The table (placeIslands) holds a candidate per ISLE_CELL cell and a massif per ISLE_MCELL cell, in
// toroidal windows round the map's middle wider than the coarsest ring's window plus a massif's reach.
constant float ISLE_CELL = 640.0;
constant float ISLE_MCELL = 3200.0;
// Must match Islands.win and Islands.mwin in Sources/StrataCore/Islands/Islands.swift.
constant int ISLE_WIN = 60;
constant int ISLE_MWIN = 13;
constant int ISLE_SLOTS = ISLE_WIN * ISLE_WIN + ISLE_MWIN * ISLE_MWIN;

// c: centre (world xz); R: radius (> 0 standing, < 0 dropped for a neighbour it would overlap, 0 none); A: the top at
// the rim's line; D: the keel's depth; B: the highest ground under it; K: a bound under its lowest underside; h: hash.
// A waterfall (f.w > 0): fa its bearing out from the lip and half its width; f the lip column's centre, the water's top
// there and the fall's foot. Must match IsleRec in Sources/StrataCore/Islands/IslandInfo.swift.
struct IsleRec { int2 cell; float2 c; float R; float A; float D; float B; float K; uint h; float2 fa; float4 f; };

static inline int isleSlot(int2 cell, bool massif) {
    int W = massif ? ISLE_MWIN : ISLE_WIN;
    int2 m = ((cell % W) + W) % W;
    return (massif ? ISLE_WIN * ISLE_WIN : 0) + m.y * W + m.x;
}

// Shape choices by hash: one in three of the smaller islands (and of the massifs) drawn out along an axis.
static inline float isleStretch(float R, uint h) { return (R > 40.0 && (R <= 210.0 || R >= 420.0) && hashf(pcg(h + 10u)) < 0.33) ? 1.5 : 1.0; }

// Nothing of an island lies further than this from its centre.
static inline float isleReach(float R, uint h) { return isleStretch(R, h) * (1.45 * R + 8.0); }

// Its form by hash: 0 a lens; 1 a spire (one in five of the smaller ones), sheer flanks over a keel twice as deep and a
// crag on top up to isleCrag; 2 tilted, its top and keel dropping toward one side by up to isleTilt; 3 a prow, its keel's
// deepest point off toward one side. Must match Islands.form in Sources/StrataCore/Islands/Islands.swift.
static inline int isleForm(float R, uint h) {
    float c = hashf(pcg(h + 30u));
    return R >= 30.0 && R <= 130.0 && c < 0.2 ? 1 : c < 0.45 ? 2 : c < 0.7 ? 3 : 0;
}
// The keel's depth before the room under it limits it.
static inline float isleDepth(float R, uint h) {
    float u = hashf(pcg(h + 5u));
    return isleForm(R, h) == 1 ? R * mix(1.8, 2.6, u) + 20.0 : R < 55.0 ? R * mix(0.6, 1.0, u) + 6.0 : R * mix(0.7, 1.4, u) + 10.0;
}
// Must match Islands.crag in Sources/StrataCore/Islands/Islands.swift.
static inline float isleCrag(float R, uint h) { return isleForm(R, h) == 1 ? 0.3 * R : 0.0; }
static inline float isleTilt(float R, uint h) { return isleForm(R, h) == 2 ? min(R * (0.12 + 0.28 * hashf(pcg(h + 31u))), 45.0) : 0.0; }

// A canopy over an island's meadow at world column centre p (d: how far out toward the rim, 0 to 1): its height over
// the ground there, 0 for none. Crowns packed on a 7-voxel grid over a low fill on the lush, groves on meadow on the
// rest, a bare one in six; none on the rim, so the edge is a lip of earth, not a wall of leaves.
static inline float isleCanopy(float R, uint h, float2 p, float d, uint seed) {
    float lush = hashf(pcg(h + 15u));
    if (R < 30.0 || lush < 0.16) return 0.0;
    float sz = clamp(R / 180.0, 0.45, 1.0);
    float open = perlin2(p / 47.0, seed + 720u) + 0.45 * perlin2(p / 17.0, seed + 721u) - mix(0.32, -0.5, lush);
    float dl = 0.86 + 0.1 * perlin2(p / 13.0, seed + 728u);
    float e = smoothstep(0.0, 0.3, open) * (1.0 - smoothstep(dl - 0.16, dl, d));
    if (e <= 0.0) return 0.0;
    float hi = mix(0.0, 6.0 * sz, e * e) * smoothstep(0.55, 0.85, lush) + 0.8 * perlin2(p / 4.0, seed + 723u);
    int2 c0 = int2(floor(p / 7.0));
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cc = c0 + int2(i, j);
            uint hc = hash2i(cc, seed + 722u);
            float2 ctr = (float2(cc) + 0.2 + 0.6 * float2(hashf(hc), hashf(pcg(hc)))) * 7.0;
            float Rc = (2.5 + 3.0 * hashf(pcg(hc + 1u))) * mix(0.75, 1.0, sz);
            float2 dq = p - ctr;
            float d2 = dot(dq, dq);
            if (d2 >= Rc * Rc) continue;
            float dome = sqrt(1.0 - d2 / (Rc * Rc));
            hi = max(hi, (2.0 + 6.0 * sz * hashf(pcg(hc + 2u))) * e + Rc * 0.9 * dome * mix(0.45, 1.0, e));
        }
    hi = floor(hi + 0.5);
    return hi >= 1.0 ? hi : 0.0;
}

// What an island's column holds besides rock: a stream's bed (or a fall), a vine.
constant uint ISK_ROCK = 0u, ISK_WATER = 1u, ISK_VINE = 2u;
// A coarse island column of water carries this leaf depth (generateCoarse; coverDepthL reads it back exactly).
constant float ISLE_WATER_DEPTH = 63.0;
// ...and one most of whose water is a stream's (whose column is rock under its bed), this one.
constant float ISLE_STREAM_DEPTH = 62.0;

// (underside, top, leaf) of island r over world column centre p; empty (top <= underside) outside it; leaf > 0 where a
// canopy stands on the island's ground from leaf to the top. kind: ISK_*. s: the sample spacing (roots and vines hang on
// the 1-voxel map and level 1, whose columns their clumps match, so they go only at level 1's far handoff).
static inline float3 isleSpan(IsleRec r, float2 p, uint seed, int s, thread uint& kind) {
    kind = ISK_ROCK;
    float R = r.R;
    uint h = r.h;
    float2 q = p - r.c;
    float st = isleStretch(R, h);
    float th0 = hashf(pcg(h + 11u)) * 3.14159265;
    float2 ax = float2(cos(th0), sin(th0));
    float2 ql = st > 1.0 ? float2(dot(q, ax) / st, dot(q, float2(-ax.y, ax.x)) / 0.75) : q;
    float rr = length(ql);
    if (rr > 1.45 * R + 8.0) return float3(0);
    float o = float(h & 1023u) * 3.7;
    float2 pq = ql / R;
    float e = R * (1.0 + 0.28 * perlin2(pq * 1.6 + o, seed + 701u) + 0.12 * perlin2(pq * 4.0 + o, seed + 702u))
            + 5.0 * perlin2(p / 11.0, seed + 703u);
    // A bay bitten into half of them.
    if (hashf(pcg(h + 12u)) < 0.5) e *= 1.0 - 0.32 * smoothstep(0.2, 0.55, perlin2(ql / max(rr, 1.0) * 1.3 + o + 7.1, seed + 708u));
    float d = rr / max(e, 4.0);
    if (d >= 1.0) return float3(0);
    // A low dome of meadow, rolling on the broad ones, terraced as the lowland is: wide treads, short risers.
    // Small ones are boulders: a rounder crown, a pointed keel, no beds.
    float small = 1.0 - smoothstep(35.0, 55.0, R);
    float top = r.A + mix(0.06 * R, min(0.3 * R, 0.06 * R + 9.0), small) * (1.0 - d * d) + 1.2 * perlin2(p / 37.0, seed + 704u);
    if (R > 120.0) top += 7.0 * fbm2(p / 90.0, 2, seed + 709u) * smoothstep(120.0, 250.0, R) * (1.0 - d * d);
    int form = isleForm(R, h);
    if (form == 1) top += isleCrag(R, h) * (1.0 - d) * (1.0 - d) * (0.7 + 0.3 * perlin2(p / 15.0, seed + 729u));
    if (form == 2) {
        float th1 = hashf(pcg(h + 32u)) * TAU;
        top -= isleTilt(R, h) * 0.5 * (1.0 + clamp(dot(q, float2(cos(th1), sin(th1))) / isleReach(R, h), -1.0, 1.0));
    }
    float tl = floor(top / 5.0);
    // The rim falls away by a varying step, from nothing to three, so no edge runs round the top at one inset.
    top = mix((tl + smoothstep(0.6, 0.95, top / 5.0 - tl)) * 5.0, top, small)
        - 3.0 * smoothstep(0.8 + 0.12 * perlin2(p / 17.0, seed + 726u), 1.0, d) * saturate(0.5 + 1.2 * perlin2(p / 29.0, seed + 727u));
    // A rim of rock teeth on some.
    if (hashf(pcg(h + 13u)) < 0.3) top += 3.0 * ridged2(p / 7.0, 2, seed + 710u) * smoothstep(0.88, 0.97, d);
    top = floor(top + 0.5);
    // A spire: steep flanks half its depth, then a long taper to a point.
    float t = form == 1 ? 0.5 * pow(1.0 - d, 0.3) + 0.5 * pow(1.0 - d, 1.3) : pow(1.0 - d, mix(mix(0.5, 0.9, hashf(pcg(h + 34u))), 1.0, small));
    if (form == 3) {
        float th2 = hashf(pcg(h + 33u)) * TAU;
        t = pow(1.0 - length(ql - float2(cos(th2), sin(th2)) * 0.35 * R) / (max(e, 4.0) + 0.35 * R), 0.55);
    }
    float rag = 0.6 + 0.4 * ridged2(p / (0.3 * R + 8.0), 2, seed + 705u);
    // Crags a couple of beds deep at a few tens of voxels, so no bed runs on round the keel as a course.
    float th = 5.0 + r.D * t * (rag + 0.14 * (ridged2(p / 23.0, 2, seed + 730u) - 0.5)) + 6.0 * t * perlin2(p / 6.0, seed + 706u);
    // Two or three drips: spires of rock hanging under a big keel.
    if (R > 150.0) {
        int n = 2 + int(pcg(h + 14u) % 2u);
        for (int i = 0; i < n; i++) {
            uint hi = pcg(h + 20u + uint(i));
            float a = hashf(hi) * TAU, rd = R * (0.15 + 0.35 * hashf(pcg(hi)));
            float2 sc = r.c + float2(cos(a), sin(a)) * rd;
            float f = max(0.0, 1.0 - length(p - sc) / (18.0 + 10.0 * hashf(pcg(hi + 1u))));
            th += r.D * 0.3 * f * f;
        }
    }
    // The keel steps in beds 3 to 13 deep, each tread facing down: dark seams, not lit lines.
    float thw = th;
    float b0 = floor(thw / 8.0);
    float Bb = 8.0 * b0 + 5.0 * (hashf(pcg(h + uint(int(b0) + 4096))) - 0.5);
    if (thw < Bb) { b0 -= 1.0; Bb = 8.0 * b0 + 5.0 * (hashf(pcg(h + uint(int(b0) + 4096))) - 0.5); }
    float Bn = 8.0 * b0 + 8.0 + 5.0 * (hashf(pcg(h + uint(int(b0) + 4097))) - 0.5);
    th = mix(Bb + smoothstep(0.6, 1.0, (thw - Bb) / (Bn - Bb)) * (Bn - Bb), th, small);
    float bot = floor(top - max(th, 4.0));
    // The stream that feeds a fall: a channel cut two deep, wandering in from the lip.
    float bank = 1e9;
    if (r.f.w > 0.0 && s <= 4) {
        float2 u = float2(cos(r.fa.x), sin(r.fa.x));
        float2 dq = p - r.f.xy;
        float along = -dot(dq, u), Ls = 0.45 * R;
        if (along > -3.0 && along < Ls) {
            float lat = dot(dq, float2(-u.y, u.x)) - 5.0 * perlin2(float2(along / 29.0, float(h & 255u) + 0.5), seed + 731u) * smoothstep(0.0, 20.0, along);
            bank = abs(lat) - max(1.6, 0.5 * float(s) + 0.3) * mix(1.0, 0.6, smoothstep(0.6 * Ls, Ls, along));
            // Its bed keeps four of rock under it: cleanupColumns fills a thinner island run's gap, a pillar to the ground.
            if (bank < 0.0 && along < Ls - 2.0) { top -= 2.0; bot = min(bot, top - 4.0); kind = ISK_WATER; }
        }
    }
    if (s <= 2 && R >= 50.0) {
        // Roots in clumps under the outer keel and vines in curtains from under the rim: tapering cones of columns on a
        // jittered 5-voxel grid, the centre longest and a short collar round it, some roots forked, the vines finer and
        // longer, their lengths drifting along a curtain.
        float clump = smoothstep(0.05, 0.35, perlin2(p / 19.0, seed + 711u)) * smoothstep(0.45, 0.9, d);
        float vc = smoothstep(0.25, 0.5, perlin2(p / 23.0, seed + 717u)) * smoothstep(0.86, 0.95, d);
        if (clump > 0.0 || vc > 0.0) {
            float vlen = 8.0 + 22.0 * saturate(0.5 + 0.9 * perlin2(p / 9.0, seed + 719u));
            float dr = 0.0, dv = 0.0;
            int2 g0 = int2(floor(p / 5.0));
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++) {
                    int2 gc = g0 + int2(i, j);
                    uint gh = hash2i(gc, seed + 707u);
                    if (hashf(gh) < 0.45 * clump) {
                        float2 c1 = (float2(gc) + 0.15 + 0.7 * float2(hashf(pcg(gh + 1u)), hashf(pcg(gh + 2u)))) * 5.0;
                        float Lr = 5.0 + 20.0 * pow(hashf(pcg(gh + 3u)), 2.0), r0 = 1.2 + 1.4 * hashf(pcg(gh + 4u));
                        dr = max(dr, Lr * pow(saturate(1.0 - length(p - c1) / r0), 1.6));
                        if (hashf(pcg(gh + 5u)) < 0.4) {
                            float an = hashf(pcg(gh + 6u)) * TAU;
                            float2 c2 = c1 + float2(cos(an), sin(an)) * 1.3 * r0;
                            dr = max(dr, 0.55 * Lr * pow(saturate(1.0 - length(p - c2) / (0.7 * r0)), 1.6));
                        }
                    }
                    uint vh = hash2i(gc, seed + 716u);
                    if (hashf(vh) < 0.5 * vc) {
                        float2 c1 = (float2(gc) + 0.15 + 0.7 * float2(hashf(pcg(vh + 1u)), hashf(pcg(vh + 2u)))) * 5.0;
                        float r0 = 0.9 + 0.6 * hashf(pcg(vh + 4u));
                        dv = max(dv, vlen * (0.6 + 0.4 * hashf(pcg(vh + 3u))) * pow(saturate(1.0 - length(p - c1) / r0), 1.3));
                    }
                }
            if (kind == ISK_ROCK && dv >= 2.0 && dv > dr) { bot -= floor(dv); kind = ISK_VINE; }
            else bot -= floor(dr);
        }
    }
    float leaf = 0.0;
    if (kind == ISK_ROCK) {
        float c = isleCanopy(R, h, p, d, seed);
        if (bank < 4.0) c = bank < 1.0 ? 0.0 : min(c, floor(bank - 1.0));
        if (c >= 1.0) { leaf = top; top += c; }
    }
    return float3(bot, top, leaf);
}

// How far out from its lip (lipD from the centre) a fall's foot lands (drop L), its whole sheet inside the island's
// reach. Must match Islands.fallOut in Sources/StrataCore/Islands/Islands.swift.
static inline float isleFallOut(float L, float reach, float lipD) { return clamp(min(0.09 * L, reach - 6.0 - lipD), 2.0, 24.0); }
// The side a fall drifts to and its spray blows, by its lip. Must match Islands.fallSide.
static inline float isleFallSide(float2 lip) { return (int(floor(lip.x + lip.y)) & 1) != 0 ? 1.0 : -1.0; }

// Narrows [lo, hi] to the t where c0 + c1 t > 0.
static inline void isleLin(float c0, float c1, thread float& lo, thread float& hi) {
    if (c1 > 0.0) lo = max(lo, -c0 / c1);
    else if (c1 < 0.0) hi = min(hi, -c0 / c1);
    else if (c0 <= 0.0) hi = -1.0;
}

// The fall off island r over world column centre p, if it is in one: a sheet two voxels thick (as thick as the spacing
// s further out, up to level 2's) arcing out from the lip, out 0.09 of its drop at its foot, narrowing as it falls and
// drifting to one side, its foot frayed and rounded. (underside, top, foot) in sp.
static inline bool isleFall(IsleRec r, float2 p, uint seed, int s, thread float3& sp) {
    if (r.f.w <= 0.0 || s > 4) return false;
    float2 u = float2(cos(r.fa.x), sin(r.fa.x));
    float2 q = p - r.f.xy;
    float L = r.f.z - r.f.w;
    float xo = isleFallOut(L, isleReach(r.R, r.h), length(r.f.xy - r.c));
    float hT = 0.5 * max(2.0, float(s)) + 0.5;
    float a = dot(q, u);
    if (a <= -hT || a >= xo + hT) return false;
    float side = isleFallSide(r.f.xy);
    float b = dot(q, float2(-u.y, u.x)) * side;
    float h0 = max(r.fa.y, 0.5 * float(s) + 0.5);
    float h1 = max(0.0, min(0.45 * h0, h0 - (0.5 * float(s) + 0.5)));
    float D = r.fa.y * mix(0.5, 1.0, hashf(pcg(r.h + 33u)));
    if (abs(b) >= h0 + D) return false;
    // Where the sheet's front, out xo sqrt(t) at drop t L, passes this column.
    float lo = a - hT <= 0.0 ? 0.0 : (a - hT) * (a - hT) / (xo * xo);
    float hi = a + hT >= xo ? 2.0 : (a + hT) * (a + hT) / (xo * xo);
    // Within its half-width there, h0 - h1 t about its drift D t.
    isleLin(h0 - b, D - h1, lo, hi);
    isleLin(h0 + b, -(D + h1), lo, hi);
    float bb = saturate(abs(b - D) / (h0 - h1));
    // Its lower part breaks into ribbons, some ending well above the rest.
    float rib = saturate(perlin2(float2(b / 2.5, float(r.h & 255u) + 0.5), seed + 737u) * 1.6);
    hi = min(hi, 1.0 - 0.4 * bb * bb - 0.1 * (hashf(hash2i(int2(floor(p)), seed + 732u)) - 0.3) - 0.4 * rib);
    if (hi <= lo) return false;
    // Six deep at least: cleanupColumns fills an island's gap under a thinner run, standing a pillar on the ground.
    float top = floor(r.f.z - L * lo + 0.5);
    sp = float3(min(floor(r.f.z - L * hi), top - 6.0), top, r.f.w);
    return true;
}

// The span of the island (or its fall) over world column centre p, if any.
static inline bool isleAt(device const IsleRec* T, float2 p, uint seed, int s, thread float3& sp, thread uint& kind) {
    int2 c0 = int2(floor(p / ISLE_CELL));
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cell = c0 + int2(i, j);
            device const IsleRec& r = T[isleSlot(cell, false)];
            if (r.R <= 0.0 || any(r.cell != cell)) continue;
            float re = isleReach(r.R, r.h);
            if (length_squared(p - r.c) > re * re) continue;
            sp = isleSpan(r, p, seed, s, kind);
            if (sp.y > sp.x) return true;
            if (isleFall(r, p, seed, s, sp)) { kind = ISK_WATER; return true; }
        }
    int2 m0 = int2(floor(p / ISLE_MCELL - 0.5));
    for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++) {
            int2 cell = m0 + int2(i, j);
            device const IsleRec& r = T[isleSlot(cell, true)];
            if (r.R <= 0.0 || any(r.cell != cell)) continue;
            float re = isleReach(r.R, r.h);
            if (length_squared(p - r.c) > re * re) continue;
            sp = isleSpan(r, p, seed, s, kind);
            if (sp.y > sp.x) return true;
            if (isleFall(r, p, seed, s, sp)) { kind = ISK_WATER; return true; }
        }
    return false;
}

static inline Col islandColumn(Col col, int2 w, uint seed, int s, device const IsleRec* T) {
    float3 sp;
    uint kind;
    if (!isleAt(T, float2(w) + 0.5, seed, s, sp, kind)) return col;
    float G = col.h;
    float ground = kind == ISK_ROCK && sp.z > 0.0 ? sp.z : sp.y;
    // Sunk in a hill: nothing; resting on one: solid to its ground.
    if (ground <= G + 3.0) return col;
    if (sp.x <= G + 3.0) { col.h = ground; return col; }
    if (col.gh > col.gl) { col.lo2 = col.lo1; col.hi2 = col.hi1; col.lo1 = col.gl; col.hi1 = col.gh; }
    col.gl = G; col.gh = sp.x; col.h = sp.y;
    col.carved = true;
    col.island = true;
    col.leaf = sp.z;
    col.isk = kind;
    return col;
}

// The map's alpha for an island column from heights to planting, which turns it into its ST_ISLE word: the canopy's
// first leaf; -2 a vine; -4 less a fall's foot (0 a stream's); 0 elsewhere.
static inline float isleAlpha(Col c) {
    return !c.island ? 0.0 : c.isk == ISK_WATER ? -4.0 - c.leaf : c.isk == ISK_VINE ? -2.0 : c.leaf;
}
