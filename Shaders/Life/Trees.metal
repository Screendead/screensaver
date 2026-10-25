// Trees sit on a jittered world-space grid, one candidate per cell; a canopy reaches at most 2 columns
// from its trunk, so a column only needs the trees of its own and the 8 neighbouring cells.
constant int TREE_CELL = 5;

// Grove density: a low-frequency Perlin field gated by climate; zero on sand, snow, desert and water.
static inline float groveDensity(float2 w, float g, uint seed) {
    float2 p = w + seedOffset(seed);
    Climate c = climate(w, seed);
    float grove = fbm2(p / 300.0, 3, seed + 90u) + 0.3 * perlin2(p / 70.0, seed + 91u);
    float painted = paintedAmount(c);
    float autumn = autumnAmount(c);
    float open = (1.0 - smoothstep(0.2, 0.5, desertAmount(c))) * (1.0 - smoothstep(0.1, 0.4, painted))
               * (1.0 - smoothstep(0.15, 0.5, saltAmount(c)));
    float wet = smoothstep(0.25, 0.6, c.moist) * open;
    float snowline = treelineAt(snowlineAt(c.temp));
    // Lagoon islets are only a few voxels high, so their groves start right above the sand.
    float lagoon = lagoonAmount(c);
    float band = smoothstep(mix(SEA + 2.0, SEA + 1.0, lagoon), mix(SEA + 6.0, SEA + 2.5, lagoon), g) * smoothstep(snowline - 4.0, snowline - 30.0, g);
    float dense = smoothstep(0.22, 0.44, grove + 0.35 * (wet - 0.5) + 0.12 * autumn + 0.3 * lagoon);
    // Sage scrub on the painted hills' valley floors.
    float fl = paintedFloor(p, seed);
    float scrub = painted * smoothstep(fl + 3.0, fl + 0.5, g) * (1.0 - smoothstep(0.02, 0.2, saltAmount(c)))
                * smoothstep(0.2, 0.45, perlin2(p / 60.0, seed + 113u));
    // Lone trees dot even dry grassland; groves need moisture.
    float dens = (0.02 * open + mix(0.32, 0.45, autumn) * dense * wet + 0.35 * scrub) * band;
    // Old-growth country is forest throughout; the jungle grows its own roof (jungleCanopy), and groves round its edge.
    dens = max(dens, 0.62 * c.old * band);
    return dens;
}

struct Tree { bool ok; int2 c; float g; int lo; int top; int kind; };

static Tree treeShape(Tree t, uint h, Climate c, float dense, float2 w, uint seed);

static Tree treeInCell(texture2d<float, access::read> src, texture2d<uint, access::read_write> gx, int2 cell, int2 origin, uint seed) {
    Tree t;
    t.ok = false;
    uint h = hash2i(cell, seed + 92u);
    t.c = cell * TREE_CELL + int2(h % uint(TREE_CELL), (h / 7u) % uint(TREE_CELL)) - origin;
    int n = int(src.get_width());
    if (any(t.c < 1) || any(t.c >= n - 1)) return t;
    float4 col = src.read(uint2(t.c));
    t.g = col.r;
    if (col.b > col.g || col.a < 0.0) return t;
    float g0 = src.read(uint2(t.c + int2(1, 0))).r, g1 = src.read(uint2(t.c - int2(1, 0))).r;
    float g2 = src.read(uint2(t.c + int2(0, 1))).r, g3 = src.read(uint2(t.c - int2(0, 1))).r;
    if (max(max(abs(g0 - t.g), abs(g1 - t.g)), max(abs(g2 - t.g), abs(g3 - t.g))) > 1.0) return t;
    // Nothing grows on a ledge against a wall: its trunk and crown would hang down the face as a dash.
    for (int k = 0; k < 8; k++) {
        int2 o2 = int2(round(float2(cos(float(k) * 0.785398), sin(float(k) * 0.785398)) * 2.0));
        int2 q = clamp(t.c + o2, int2(0), int2(n - 1));
        float4 cq = src.read(uint2(q));
        if (cq.r > t.g + 8.0) return t;
        // Nor beside carved rock (a shelter's mouth, a notch): planting skips its columns, which would bite the crown.
        if (cq.b > cq.g && (gapFlags(gx.read(uint2(q))) & GAP_CARVED) != 0u) return t;
    }
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(t.c + int2(k & 1 ? 2 : -2, k & 2 ? 2 : -2), int2(0), int2(n - 1));
        float4 cq = src.read(uint2(q));
        if (cq.b > cq.g && (gapFlags(gx.read(uint2(q))) & GAP_CARVED) != 0u) return t;
    }
    float2 w = float2(t.c + origin);
    float dense = groveDensity(w, t.g, seed);
    if (hashf(pcg(h)) > dense) return t;
    Climate c = climate(w, seed);
    c.temp = altitudeTemp(c.temp, t.g);
    return treeShape(t, h, c, dense, w, seed);
}

// A planted tree's kind and leaf span from its hash h, the climate c and grove density at its trunk w (world); t.g is
// its ground. Also builds the coarse rings' trees (coarseVeg).
static Tree treeShape(Tree t, uint h, Climate c, float dense, float2 w, uint seed) {
    uint r = pcg(h + 5u);
    // 0 shrub, 1 small round, 2 round, 3 conifer, 4 birch.
    if (paintedAmount(c) > 0.3) t.kind = 0;
    else if (autumnAmount(c) > 0.3 && birchAmount(w, seed) > 0.5) t.kind = 4;
    else if (hashf(r) < coniferAmount(c)) t.kind = 3;
    else if (dense < 0.15) t.kind = int(hashf(pcg(r)) * 3.0);
    else t.kind = hashf(pcg(r)) < 0.3 ? 1 : 2;
    // A lone one-voxel shrub on open sand or salt reads as debris.
    if (t.kind == 0 && (saltAmount(c) > 0.02 || lagoonAmount(c) > 0.2 || desertAmount(c) > 0.3)) return t;
    int g = int(t.g);
    if (t.kind == 0) { t.lo = g; t.top = g + 1 + int(r >> 30 & 1u); }
    else if (t.kind == 1) { t.lo = g + 2 + int(r >> 29 & 1u); t.top = t.lo + 3; }
    else if (t.kind == 2) { t.lo = g + 2 + int(r >> 29 & 1u); t.top = t.lo + 4; }
    else if (t.kind == 4) { t.lo = g + 3 + int(r >> 29 & 1u); t.top = t.lo + 5; }
    else { t.lo = g + 1; t.top = g + 6 + int(r >> 28 & 3u); }
    t.ok = true;
    return t;
}

// Leaf interval [lo, hi) a tree contributes to the column at offset o from its trunk; empty if lo >= hi.
static inline int2 canopySpan(Tree t, int2 o) {
    int d2 = o.x * o.x + o.y * o.y;
    if (t.kind == 0) return d2 == 0 ? int2(t.lo, t.top) : d2 == 1 ? int2(t.lo, t.top - 1) : int2(0);
    if (t.kind == 1 || t.kind == 4) return d2 <= 1 ? int2(t.lo, t.top) : d2 == 2 ? int2(t.lo + 1, t.top - 1) : int2(0);
    if (t.kind == 2) return d2 <= 2 ? int2(t.lo, t.top) : d2 <= 5 ? int2(t.lo + 1, t.top - 1) : int2(0);
    // Conifer: a stepped cone, widest a little above the ground.
    int h = t.top - t.lo;
    if (d2 == 0) return int2(t.lo + 1, t.top);
    if (d2 == 1) return int2(t.lo + 1, t.top - 2);
    if (d2 == 2) return int2(t.lo + 1, t.lo + max(h - 4, 1));
    if (d2 == 4 && h >= 7) return int2(t.lo + 1, t.lo + 2);
    return int2(0);
}

// Writes a vegetation column built by runsColumn over ground G: as a tree column (w = first leaf, g the ground, b = g on
// trunks and buried canopies, b = w under a floating canopy) where it is one, else as a multi-run column (GIANT_A, its
// ground and materials in st, extra gaps in gx).
static void writeVeg(GCol gc, float G, uint2 gid, texture2d<float, access::write> dst, texture2d<uint, access::read_write> gx,
                     texture2d<uint, access::write> st, uint pal) {
    bool simple = gc.n == 1 || (gc.n == 2 && gc.gl == G && gc.leafStart == gc.gh);
    if (simple && (gc.mats >> (2u * uint(gc.n - 1)) & 3u) != uint(GM_VINE) && G < 2047.0) {
        dst.write(float4(gc.H, G, gc.n == 2 ? gc.gh : G, max(gc.leafStart, G)), gid);
        return;
    }
    // The highest gap in the principal slot; the taller of any two below it (the air under a crown) in the first extra
    // slot, the other in the second (the morph band takes them all by height: giantBand). With one gap the principal slot
    // holds it inverted (lo > hi): no gap to the trace, but still the bound under which it looks for extra gaps, and air
    // to corner AO.
    float4 e = gc.n == 2 ? float4(gc.gl, gc.gh, 0, 0) : gc.n == 4 && gc.e.w - gc.e.z > gc.e.y - gc.e.x ? gc.e.zwxy : gc.e;
    dst.write(float4(gc.H, gc.n == 2 ? float2(gc.gh, gc.gl) : float2(gc.gl, gc.gh), GIANT_A), gid);
    gx.write(uint4(uint(e.x), uint(e.y), uint(e.z) | (GAP_GIANT << 11), uint(e.w)), gid);
    st.write(uint4(ST_GIANT | uint(G) | (uint(gc.leafStart) << 11) | (gc.mats << 22) | (pal << 28), 0, 0, 0), gid);
}
