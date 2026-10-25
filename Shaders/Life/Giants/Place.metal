// Hometree-scale giants rising out of the jungle: at most one per GIANT_CELL world cell, at a hashed point in the
// cell's middle 0.6 (so two stand at least 0.4 of a cell apart), only on flat, dry lowland deep in a jungle province
// (placeGiants). Nothing of a giant lies further than GIANT_REACH from its axis.
constant float GIANT_CELL = 1536.0;
constant float GIANT_REACH = 210.0;
// The table: a toroidal window of GIANT_WIN cells a side (+-18k) around the map centre, wider than the rings' disc and
// the coarsest ring's window, so the entries a recentring rewrites are never read.
// Must match World.giantWin in Sources/Renderer.swift.
constant int GIANT_WIN = 24;
constant float GIANT_A = -4000.0;
constant uint ST_GIANT = 1u << 31;
// An island column's word in st (plantTrees): its canopy's first leaf in the low 11 bits, or a stream or fall, or a vine.
constant uint ST_ISLE = 1u << 30, ST_ISLE_WATER = 1u << 11, ST_ISLE_VINE = 1u << 12;
constant uint GAP_GIANT = 2u;
// The band test in the DDA: compiled in only while a canopy band is on the map (Renderer picks the pipeline); off in
// every pipeline built without the constant.
constant bool BAND_FC [[function_constant(0)]];
constant bool BAND_SKIP = is_function_constant_defined(BAND_FC) && BAND_FC;
// Renderer picks the pipeline per tile of the frame (Renderer.sceneTile) by what of the islands may show there.
// The islands' shade and sky occlusion: compiled in only where an island within isleShadeReach casts them.
constant bool ISLE_FC [[function_constant(3)]];
constant bool SCENE_ISLES = is_function_constant_defined(ISLE_FC) && ISLE_FC;
// Close by an island within isleShadeReach (its column of air), with its shade: the ground under it, its falls, streams,
// canopies and glow.
constant bool NEAR_FC [[function_constant(6)]];
constant bool SCENE_NEAR = SCENE_ISLES && is_function_constant_defined(NEAR_FC) && NEAR_FC;
// The islands' surfaces (their rock, roots and vines, air that is no shelter's roof): wherever an island at any distance
// could show.
constant bool SHAPE_FC [[function_constant(4)]];
constant bool ISLE_SHAPE = SCENE_NEAR || (is_function_constant_defined(SHAPE_FC) && SHAPE_FC);
// The islands' mist (isleMist), close by them.
constant bool MIST_FC [[function_constant(5)]];
constant bool SCENE_MIST = SCENE_NEAR && is_function_constant_defined(MIST_FC) && MIST_FC;
constant int GIANT_LIMBS = 12;
constant int GIANT_FLANGES = 9;
constant int GM_WOOD = 0, GM_LEAF = 1, GM_VINE = 2;
// The coarsest level whose columns carry a giant's aerial roots (one of four sparse samples catching one stands it 8
// wide there): on level 4 it would hang a pillar 16 wide. (Vines are no part of any level: traceStrands.)
constant int GIANT_THIN_LOD = 3;

// One table entry; G <= 0: no giant in the cell. Must match GiantRec in Sources/Renderer.swift.
struct GiantRec { int2 cell; float2 c; float G; float yT; uint h; uint kind; };
static_assert(sizeof(GiantRec) * GIANT_WIN * GIANT_WIN == CAVE_TAIL, "the cave table follows the giant table");
static_assert(sizeof(CaveRec) == 48, "World.CaveRec");
static inline device const CaveRec* caveTable(device const GiantRec* giants) {
    return (device const CaveRec*)((device const uchar*)giants + CAVE_TAIL);
}

// c: axis (world xz); G: the ground there, which places its parts (roots and flanges meet each column's own ground).
struct Giant { float2 c; float G; uint h; float yT; int kind; };

static inline int giantSlot(int2 cell) {
    int2 m = ((cell % GIANT_WIN) + GIANT_WIN) % GIANT_WIN;
    return m.y * GIANT_WIN + m.x;
}

// A giant's leaf palette: 1 deep green, 2 teal-green, 3 green flushed with muted rust (one in six); 0 is the jungle's
// own (emergents).
static inline uint giantPalette(uint h) {
    float k = hashf(pcg(h + 77u));
    return k < 1.0 / 6.0 ? 3u : k < 0.58 ? 1u : 2u;
}

// The giant whose reach, grown by pad, covers world xz w. Reach and pad stay under half a cell, so at most the 2x2
// cells round w can hold it.
static inline bool giantAt(device const GiantRec* T, float2 w, float pad, thread Giant& g) {
    float R = GIANT_REACH + pad;
    int2 c0 = int2(floor((w - R) / GIANT_CELL)), c1 = int2(floor((w + R) / GIANT_CELL));
    for (int j = c0.y; j <= c1.y; j++)
        for (int i = c0.x; i <= c1.x; i++) {
            GiantRec r = T[giantSlot(int2(i, j))];
            if (r.G <= 0.0 || any(r.cell != int2(i, j)) || length_squared(w - r.c) >= R * R) continue;
            g.c = r.c; g.G = r.G; g.h = r.h; g.yT = r.yT; g.kind = int(r.kind);
            return true;
        }
    return false;
}

// Fills the table's cells info.xy + gid. A cell may hold a giant in three of four, tried at up to eight points, each
// deep in jungle, outside the mega relief, dry and flat for 70 round (a slope or a wall there would have its buttresses
// hang in the air) with no water within 100.
kernel void placeGiants(device GiantRec* out [[buffer(0)]],
                        constant int4& info [[buffer(1)]],
                        uint2 g [[thread_position_in_grid]]) {
    // info.z: the first row of this dispatch.
    uint2 gid = g + uint2(0, uint(info.z));
    if (gid.x >= uint(GIANT_WIN) || gid.y >= uint(GIANT_WIN)) return;
    uint seed = uint(info.w);
    int2 cell = info.xy + int2(gid);
    // A slot already holding this cell (the table moves a cell at a time) keeps it: placing costs a few hundred terrain
    // columns per cell. Slots start out holding no cell (World.MapBuffer).
    if (all(out[giantSlot(cell)].cell == cell)) return;
    GiantRec r;
    r.cell = cell; r.c = 0; r.G = -1.0; r.yT = 0; r.h = 0; r.kind = 0;
    uint hc = hash2i(cell, seed + 700u);
    if (hashf(hc) < 0.75) {
        for (int a = 0; a < 8; a++) {
            uint h = pcg(hc + uint(a) * 977u);
            float2 P = floor((float2(cell) + 0.2 + 0.6 * float2(hashf(pcg(h + 1u)), hashf(pcg(h + 2u)))) * GIANT_CELL) + 0.5;
            Climate c = climate(P, seed);
            if (c.jungle < 0.6) continue;
            // A crown spreading through a tower's masonry would hang leaves inside it.
            if (towerNear(P, GIANT_REACH + 20.0, seed)) continue;
            Col col = terrainColumnBare(int2(floor(P)), seed);
            float G = col.h;
            if (G < SEA + 3.0 || G > 600.0 || col.gh > col.gl) continue;
            if (megaLift(P + seedOffset(seed), lagoonAmount(c), seed).U > 0.5) continue;
            bool ok = true;
            for (int i = 0; i < 36 && ok; i++) {
                float rr = i < 12 ? 30.0 : i < 24 ? 70.0 : 100.0;
                float an = float(i) * 0.5236 + (i < 12 ? 0.0 : i < 24 ? 0.26 : 0.13);
                Col q = terrainColumnBare(int2(floor(P + float2(cos(an), sin(an)) * rr)), seed);
                if (q.h < SEA + 2.0 || q.gh > q.gl) ok = false;
                if (rr < 80.0 && abs(q.h - G) > 9.0) ok = false;
            }
            if (!ok) continue;
            r.c = P; r.G = G; r.h = h;
            r.yT = mix(165.0, 255.0, hashf(pcg(h + 3u)));
            // Banyan, kapok or weeping, two in five, two in five, one in five.
            float kh = hashf(pcg(h + 4u));
            r.kind = kh < 0.4 ? 0u : kh < 0.8 ? 1u : 2u;
            // A kapok stands taller.
            if (r.kind == 1u) r.yT = mix(210.0, 265.0, hashf(pcg(h + 3u)));
            break;
        }
    }
    out[giantSlot(cell)] = r;
}
