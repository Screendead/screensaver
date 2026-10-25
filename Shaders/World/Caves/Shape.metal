// Caves in cliff faces, mostly in the mountains: at most one per CAVE_CELL world cell, its mouth in the cell's middle
// half and nothing of it farther than CAVE_REACH from the mouth, so a column asks its own cell only. placeCaves finds
// each cell's cave on the terrain once and keeps it in a toroidal table of CAVE_WIN cells a side round the map centre,
// wider than level 3's window: the 1-voxel map and coarse levels 1-3 carve the same cave at their sample points, and
// level 4 none (its window is wider than the table).
constant float CAVE_CELL = 256.0;
constant float CAVE_REACH = 64.0;
// Must match MapLayout.caveWin in Sources/StrataCore/World/MapLayout.swift.
constant int CAVE_WIN = 96;
// The table lies in the giant table's buffer after its GIANT_WIN^2 records, so whatever has the giants has the caves.
constant int CAVE_TAIL = 24 * 24 * 32;
// The coarsest sample spacing that carves caves (level 3).
constant int CAVE_MAX_S = 8;

// m: the mouth's centre (world xz), u: into the rock, yF: the floor, W: the mouth's half width, L: depth, Hc: height at
// the mouth's middle. Must match CaveRec in Sources/StrataCore/World/MapRecords.swift.
struct CaveRec { int2 cell; float2 m; float2 u; float yF; float W; float L; float Hc; uint h; uint flags; };
constant uint CV_ON = 1u, CV_HOME = 2u, CV_GROTTO = 4u, CV_LOW = 8u;

static inline int caveSlot(int2 cell) {
    int2 m = ((cell % CAVE_WIN) + CAVE_WIN) % CAVE_WIN;
    return m.y * CAVE_WIN + m.x;
}

static inline bool caveFind(device const CaveRec* caves, float2 w, thread CaveRec& r) {
    int2 cell = int2(floor(w / CAVE_CELL));
    r = caves[caveSlot(cell)];
    return all(r.cell == cell) && (r.flags & CV_ON) != 0u;
}

// A cave's section at world column q: (floor, roof) of its hollow, or lo >= hi outside it, and the arch's height there
// as a share of its crown's. a: depth into the rock from the mouth line, b: across it.
static inline float3 caveSection(CaveRec r, float2 q, uint seed) {
    float2 d = q - r.m;
    float a = dot(d, r.u), b = dot(d, float2(-r.u.y, r.u.x));
    if (a < -4.0 || a > r.L + 2.0 || abs(b) > r.W * 1.5 + 3.0) return float3(0);
    float ta = saturate(a / r.L);
    // Full width at the mouth, swelling into a chamber, pinching out at the back; the edge wanders.
    float wa = r.W * (1.0 + 0.3 * sin(3.14159 * min(ta * 1.6, 1.0))) * sqrt(saturate(1.0 - pow(max(ta - 0.5, 0.0) / 0.5, 2.0)));
    // Each cave's crown sits off its axis by its own amount, wandering with depth, and its arch has its own profile
    // (pointed to round shouldered), so no two mouths share a silhouette.
    // A home's crown stays near its axis: its furnishing stands either side of it.
    float bc = b - wa * ((r.flags & CV_HOME) != 0u ? 0.3 : 1.0) * (0.35 * (hashf(pcg(r.h + 11u)) - 0.5) + 0.2 * perlin2(float2(a / 9.0, 0.5), seed + 915u));
    float x = abs(bc) / max(wa, 0.5) + 0.2 * perlin2(q / 6.0, seed + 910u) + 0.08 * perlin2(q / 2.7, seed + 911u);
    if (x >= 1.0) return float3(0);
    // A rounded arch, its crown ragged.
    float arch = pow(1.0 - x * x, 0.6 + 0.6 * hashf(pcg(r.h + 12u)));
    float hc = r.Hc * (1.0 - 0.45 * smoothstep(0.25, 1.0, ta)) * (0.88 + 0.24 * perlin2(q / 8.0, seed + 912u));
    float lo = r.yF + floor(1.8 * (1.0 - sqrt(1.0 - x * x)) * saturate(0.6 + perlin2(q / 3.3, seed + 913u)));
    float hi = r.yF + floor(hc * arch + 0.5 + 1.3 * perlin2(q / 3.5, seed + 914u) * arch);
    return float3(lo, hi, arch);
}

// A lived-in cave's hearth: its depth along the cave's axis; its gear stands to one side of it.
static inline float caveHearthA(CaveRec r) { return clamp(0.3 * r.L, 4.0, 9.0); }
constant int CP_STONE = 1, CP_FIRE = 2, CP_WOOD = 3, CP_BED = 4;

// What of a home's furnishing stands at cave coordinates (a, b) (a column's centre), and how far it rises over the floor:
// a fire in a ring of stones, a woodpile and a bedroll either side of it.
static inline int caveProp(CaveRec r, float a, float b, thread float& rise) {
    rise = 0.0;
    float da = a - caveHearthA(r), sb = (r.h & 64u) != 0u ? b : -b;
    if (abs(da) < 1.5 && abs(b) < 1.5) {
        if (abs(da) < 0.5 && abs(b) < 0.5) { rise = 0.75; return CP_FIRE; }
        rise = abs(da) > 0.5 && abs(b) > 0.5 ? 0.25 : 0.5;
        return CP_STONE;
    }
    if (da > -1.0 && da < 1.0 && sb > 2.5 && sb < 3.5) { rise = 1.0; return CP_WOOD; }
    if (da > 1.5 && da < 4.5 && sb < -2.0 && sb > -3.0) { rise = 0.25; return CP_BED; }
    return 0;
}

// The column with its cave carved: a hollow under at least three voxels of roof; in the mouth, rock under the arch with
// less roof than that is cut down to the floor, which opens the arch in the cliff.
static inline Col caveColumn(Col c, int2 w, CaveRec r, uint seed, int s) {
    if (c.gh > c.gl) return c;
    float2 q = float2(w) + 0.5;
    float3 g = caveSection(r, q, seed);
    float lo = g.x, hi = g.y;
    float a = dot(q - r.m, r.u);
    // In the mouth the arch's shoulders carry down to two voxels.
    if (hi - lo < (a < 3.0 ? 2.0 : 3.0) || c.h <= lo) return c;
    c.cave = true;
    if (c.h < hi + 3.0) {
        // Rock in the mouth with too little roof is cut away to a ragged sill toward the shoulders.
        if (a < 3.0) { c.h = min(c.h, lo + floor(0.8 + 1.2 * perlin2(q / 2.5, seed + 916u)) * step(g.z, 0.8)); return c; }
        hi = c.h - 3.0;
        if (hi - lo < 3.0) return c;
    }
    // A home's furnishing on its floor, on the 1-voxel map only (a quarter voxel to a voxel: nothing a coarse level shows).
    if (s == 1 && (r.flags & CV_HOME) != 0u && lo == r.yF) {
        float rise;
        if (caveProp(r, dot(q - r.m, r.u), dot(q - r.m, float2(-r.u.y, r.u.x)), rise) != 0) lo += rise;
    }
    c.gl = lo; c.gh = hi; c.carved = true;
    return c;
}
