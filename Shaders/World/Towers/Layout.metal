// Rare, colossal ruined towers. A tower is a landform: terrainColumn adds it to the column, so the 1-voxel map, every
// coarse ring, the max mips, the horizon, the shadow maps and the CPU copy (camera, birds) all carry it. A world cell
// TOWER_CELL wide holds at most one group of 1 to 3 towers, centred in the middle half of the cell; nothing of a group
// lies farther than TOWER_GROUP_R from that centre, so a column asks its own cell only. Whether a group stands is
// decided at its centre alone (climate, provinces), so every map and ring decides it alike.
constant float TOWER_CELL = 9000.0;
// Chance a cell holds a group where the climate weighs 1 (wet, warm country weighs 1.5).
constant float TOWER_P = 0.4;
constant float TOWER_GROUP_R = 2250.0;
constant int TOWER_MEMBERS = 4;
// The widest base apothem any member can draw (drums), for spacing a group.
constant float TOWER_R_MAX = 150.0;
// Records in the shading table (fragment buffer 9); must match Towers.tableSize in Sources/Towers.swift.
constant int TOWER_TABLE = 16;


struct Tower {
    float2 c;              // world xz of the base's axis
    float R0;              // the base shaft's apothem
    float Hb;              // the ground it stands on (the lake's surface where it stands in water)
    float Ht;              // its full height above Hb
    int K;                 // stages; the top two are one hollow shell
    int N;                 // sides of the plan
    float rot, th0, th1;   // plan rotation; the bearing the crown fell toward; the bearing of the wound
    int kb;                // the wounded stage
    float Sy, bw;          // storey height, bay width
    uint h;
    int kind;              // 0 stepped spire, 1 drum, 2 needle
    float reach;           // nothing of it, rubble included, lies farther than this from c
};

struct TowerGroup { float2 c; uint h; int n; };

// A tower's build, by its hash alone (the shading table holds the hash): 0 a slender spire, 1 a broad terraced keep of
// a few deep setbacks, 2 the stump of a great drum, its upper storeys long fallen.
static inline int towerStyle(uint h) {
    float u = hashf(pcg(h + 15u));
    return u < 0.4 ? 0 : u < 0.72 ? 1 : 2;
}

static inline bool towerGroupIn(int2 cell, uint seed, thread TowerGroup& g) {
    uint h = hash2i(cell, seed + 500u);
    if (hashf(h) >= TOWER_P * 1.5) return false;
    g.h = h;
    g.c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h + 1u)), hashf(pcg(h + 2u)))) * TOWER_CELL;
    float u = hashf(pcg(h + 3u));
    // Mostly alone; a pair now and then; three rarely, the third a stump. More read as a skyline.
    g.n = u < 0.65 ? 1 : u < 0.95 ? 2 : 3;
    return true;
}

// Member m of a group, all but its base height: hashes only.
static inline Tower towerMember(TowerGroup g, int m) {
    Tower T;
    uint h0 = pcg(g.h + 11u);
    // The first stands tallest; a few pierce the cloud deck.
    float Ht0 = floor(420.0 + 780.0 * pow(hashf(h0), 1.5));
    uint h = m == 0 ? h0 : pcg(g.h + 101u * uint(m));
    for (int i = 0; i < 8 && m == 2 && towerStyle(h) != 2; i++) h = pcg(h + 7u);
    T.h = h;
    float u = hashf(pcg(h + 1u));
    int st = towerStyle(h);
    // A slender drum reads as a modern stack: the spires are stepped or needles, the drums broad.
    T.kind = st == 0 ? (u < 0.7 ? 0 : 2) : (u < 0.55 ? 0 : 1);
    T.Ht = m == 0 ? Ht0 : floor(Ht0 * (0.38 + 0.4 * hashf(pcg(h + 2u))));
    if (st == 2) T.Ht = floor(T.Ht * (0.42 + 0.18 * hashf(pcg(h + 17u))));
    float u3 = hashf(pcg(h + 3u));
    float girth = st == 0 ? 0.08 + 0.04 * u3 : st == 1 ? 0.13 + 0.05 * u3 : 0.2 + 0.08 * u3;
    T.R0 = floor(clamp(T.Ht * girth, 36.0, 130.0) * (T.kind == 1 ? 1.15 : T.kind == 2 ? 0.85 : 1.0));
    T.K = T.kind == 2 ? 3 + int(pcg(h + 4u) % 2u) : st == 1 ? 4 + int(pcg(h + 4u) % 2u) : st == 2 ? 3 + int(pcg(h + 4u) % 2u)
        : 3 + int(pcg(h + 4u) % 3u);
    T.N = T.kind == 0 ? (hashf(pcg(h + 5u)) < 0.5 ? 6 : 8) : 12;
    T.rot = hashf(pcg(h + 6u)) * 6.2831853 / float(T.N);
    T.th0 = hashf(pcg(h + 7u)) * 6.2831853;
    T.th1 = T.th0 + 2.0 + 2.0 * hashf(pcg(h + 8u));
    T.kb = 1 + int(pcg(h + 9u) % 2u);
    T.Sy = floor(16.0 + 6.0 * hashf(pcg(h + 10u)));
    T.bw = floor(11.0 + 3.0 * hashf(pcg(h + 12u)));
    T.reach = T.R0 * 2.8;
    T.c = floor(g.c) + 0.5;
    if (m > 0) {
        // Round the first, each in its own sector, far enough that no two reaches meet.
        float a = hashf(pcg(g.h + 21u)) * 6.2831853 + float(m - 1) * 6.2831853 / float(max(g.n - 1, 1))
                + 0.6 * (hashf(pcg(h + 13u)) - 0.5);
        float d = 2.8 * (T.R0 + TOWER_R_MAX) + 500.0 + 600.0 * hashf(pcg(h + 14u));
        T.c = floor(g.c + float2(cos(a), sin(a)) * d) + 0.5;
    }
    T.Hb = 0.0;
    return T;
}

// The field megaLift's provinces rise from (they stand where it passes 0.5; keep in step with megaLift), and whether a
// tepui stands within d of p: what a tower group asks instead of megaLift itself, which every column of the group would
// otherwise evaluate many times over.
static inline float towerProvinceField(float2 p, uint seed) {
    float2 q;
    return megaField(p, seed, q);
}

// How far world column centre w lies in wall country: 1 where tableland, karst or a tepui stands at full height, 0 in the
// lowland and the alpine ranges: the forest keeps off the aprons under those walls (wallClear).
static inline float wallCountry(float2 w, uint seed) {
    float2 p = w + seedOffset(seed), q, c;
    float R, wall = 0.0;
    uint hc;
    float M = smoothstep(0.50, 0.80, megaField(p, seed, q));
    if (M > 0.0) wall = smoothstep(0.0, 0.3, M) * (1.0 - megaAlpine(q, seed));
    if (tepuiIn(p, seed, c, R, hc)) wall = max(wall, 1.0 - smoothstep(R + 1200.0, R + 1700.0, length(p - c)));
    return wall;
}

static inline bool towerNearTepui(float2 p, float d, uint seed) {
    const float CELL = 30000.0;
    int2 c0 = int2(floor(p / CELL));
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cell = c0 + int2(i, j);
            uint hc = hash2i(cell, seed + 240u);
            if (hashf(hc) >= 0.35) continue;
            float2 c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(hc)), hashf(pcg(hc + 1u)))) * CELL;
            if (length(p - c) < 1500.0 + 1600.0 + 520.0 + d) return true;
        }
    return false;
}

// Whether the group fits its country: lowland well clear of the provinces (it out-scales the land there; beside a
// province's walls it is dwarfed) and of the tepuis, never on a salt pan, most often in wet, warm country, where the
// jungle has half reclaimed it.
static inline bool towerGroupFits(TowerGroup g, uint seed, thread float& lag) {
    Climate cl = climate(g.c, seed);
    if (saltAmount(cl) > 0.1) return false;
    float wgt = mix(1.0, 1.5, smoothstep(0.45, 0.65, cl.moist) * smoothstep(0.4, 0.55, cl.temp));
    wgt *= mix(1.0, 0.5, smoothstep(0.35, 0.7, desertAmount(cl)));
    wgt *= mix(1.0, 0.3, smoothstep(0.32, 0.2, cl.temp));
    if (hashf(g.h) >= TOWER_P * wgt) return false;
    float2 p = g.c + seedOffset(seed);
    if (towerProvinceField(p, seed) > 0.46) return false;
    for (int i = 0; i < 4; i++) {
        float a = float(i) * 1.5707963 + 0.785398;
        if (towerProvinceField(p + float2(cos(a), sin(a)) * 2500.0, seed) > 0.5) return false;
    }
    if (towerNearTepui(p, 2500.0, seed)) return false;
    lag = lagoonAmount(cl);
    return true;
}

// Whether the group stands: it fits, and most stand on dry land; some in a lake or a river's bed, more often in lagoon
// country.
static inline bool towerGroupStands(TowerGroup g, uint seed) {
    float lag = 0.0;
    if (!towerGroupFits(g, seed, lag)) return false;
    if (terrainColumnBase(int2(floor(g.c)), seed).h >= SEA + 2.0 || lag >= 1.0) return true;
    return hashf(pcg(g.h + 5u)) < mix(0.3, 1.0, lag);
}

// A member whose ground stands higher than this (a group's edge reaching up a province's slopes) is not built.
constant float TOWER_BASE_MAX = 130.0;

// A member's base: the ground at its axis, or the lake's surface where it stands in water.
static inline float towerBase(Tower T, uint seed) {
    return max(terrainColumnBase(int2(floor(T.c)), seed).h, floor(SEA) + 1.0);
}

// The member whose reach (plus margin) holds wf, by hashes only (members never share ground); the caller still asks
// towerGroupStands.
static inline bool towerFind(float2 wf, float margin, uint seed, thread Tower& T, thread TowerGroup& g) {
    if (!towerGroupIn(int2(floor(wf / TOWER_CELL)), seed, g)) return false;
    if (length(wf - g.c) > TOWER_GROUP_R + margin) return false;
    for (int m = 0; m < TOWER_MEMBERS; m++) {
        if (m >= g.n) break;
        Tower t = towerMember(g, m);
        if (length(wf - t.c) < t.reach + margin) { T = t; return true; }
    }
    return false;
}

// Fraction of the height at the top of stage k of a tower of K stages and hash h: a tall base shaft, then setback
// stages of uneven height. (The shading table holds h and K, so the masonry finds the ledges too.)
static inline float tierTop(uint h, int k, int K) {
    float acc = 0.0, tot = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= K) break;
        float u = hashf(pcg(h + 80u + uint(i)));
        float wgt = i == 0 ? (towerStyle(h) == 1 ? 1.0 + 0.6 * u : 1.4 + 1.2 * u) : 0.65 + 0.6 * u;
        tot += wgt;
        if (i <= k) acc += wgt;
    }
    return acc / tot;
}

// Stage k's apothem as a share of the base's: each setback its own depth (the shell's two stages share one).
static inline float towerTierScale(Tower T, int k) {
    float r = 1.0;
    for (int i = 1; i < 6; i++) {
        if (i > min(k, T.K - 2)) break;
        float u = hashf(pcg(T.h + 90u + uint(i)));
        r -= T.kind == 2 ? 0.14 + 0.1 * u : towerStyle(T.h) == 1 ? 0.12 + 0.1 * u : 0.09 + 0.14 * u;
    }
    return r;
}

// Stage k's axis: the upper stages stand off the base's, mostly toward one side, so a setback is a broad terrace on one
// side and narrow on the other (concentric setbacks read as a wedding cake; near flush, as fins down the face).
static inline float2 towerStageAxis(Tower T, int k) {
    int j = min(k, T.K - 2);
    if (j <= 0) return T.c;
    uint h = pcg(T.h + 40u + uint(j));
    float a = hashf(pcg(T.h + 39u)) * 6.2831853 + 1.6 * (hashf(h) - 0.5);
    return T.c + float2(cos(a), sin(a)) * T.R0 * (1.0 - towerTierScale(T, j)) * (0.2 + 0.3 * hashf(pcg(h)));
}

struct TowerPlan { float r; float a; float am; float reff; float t; int face; float2 dir; };

// The plan's coordinates of offset d from an axis: reff, the distance to the face's line (faces are flat walls with
// square corners), t along the face, am the bearing from the face's middle.
static inline TowerPlan towerPlan(Tower T, float2 d) {
    TowerPlan P;
    P.r = length(d);
    P.dir = P.r > 1e-3 ? d / P.r : float2(1, 0);
    P.a = atan2(d.y, d.x);
    float seg = 6.2831853 / float(T.N);
    float fi = floor((P.a - T.rot) / seg);
    P.face = (int(fi) % T.N + T.N) % T.N;
    P.am = P.a - T.rot - seg * fi - 0.5 * seg;
    P.reff = P.r * cos(P.am);
    P.t = P.r * sin(P.am);
    return P;
}

// Stage k's apothem. (Corner piers drew a crisp vertical line down every corner: art deco, not a ruin.)
static inline float tierR(Tower T, TowerPlan P, int k) {
    return T.R0 * towerTierScale(T, k);
}

// Top of stage k (below the shell) at this bearing.
static inline float tierY(Tower T, TowerPlan P, int k, uint seed) {
    float t1 = tierTop(T.h, k, T.K), t0 = k > 0 ? tierTop(T.h, k - 1, T.K) : 0.0;
    float hk = T.Ht * (t1 - t0);
    float y = T.Hb + T.Ht * t1;
    // The rim is worn at a broad angular scale, so a ledge stays level along whole faces.
    y -= floor(6.0 * abs(perlin2(P.dir * 2.2 + float(k) * 5.1, seed + 503u)));
    float fall = 0.5 + 0.5 * cos(P.a - T.th0);
    // The stage under the shell slumped on the side the crown fell to.
    if (k == T.K - 3) y -= floor(hk * 0.55 * smoothstep(0.55, 0.95, fall) * saturate(0.5 + perlin2(P.dir * 3.0, seed + 504u)) / 5.0) * 5.0;
    return floor(y);
}

// Top of the shell's wall: broken away down to stumps on the side the crown fell toward, rising round the shell to a
// ragged peak opposite, the break stepping down in whole courses a few columns at a time.
static inline float crownY(Tower T, TowerPlan P, float y0, float y1, uint seed) {
    float up = 0.5 - 0.5 * cos(P.a - T.th0);
    float nz = perlin2(P.dir * 2.1 + float(T.h & 63u), seed + 544u);
    // Each crown broke its own way: a ragged peak opposite the fall, a low ring worn all round (every stump), or one
    // tall shard.
    float cr = hashf(pcg(T.h + 16u));
    float g = 0.9 * up * up * (3.0 - 2.0 * up) - 0.08 + 0.38 * nz;
    if (towerStyle(T.h) == 2 || cr < 0.35) g = 0.15 + 0.3 * up + 0.5 * perlin2(P.dir * 3.3 + float(T.h & 63u), seed + 548u);
    else if (cr > 0.75) g = 1.15 * pow(up, 6.0) - 0.05 + 0.2 * nz;
    // Masonry breaks in blocks: one or two courses up or down per 16th of the round.
    uint sh = hash2i(int2(int(floor((P.a + 3.14159265) / 6.2831853 * 24.0)), int(T.h & 1023u)), seed + 545u);
    float y = y0 + (y1 - y0) * saturate(g) + 5.0 * floor(3.0 * hashf(sh) - 1.0);
    return clamp(floor(y / 5.0) * 5.0, y0 + 3.0, y1);
}

// Stage k's bay width on plan face `face`: each face of each stage laid out on its own.
static inline float towerBayW(uint h, int k, int face, float bw) {
    return floor(bw * (0.8 + 0.7 * hashf(hash3i(int3(face, k, 7), h + 122u))));
}

// Whether stage k's loggia opens bay `bay` of plan face `face` (tower hash h): about half the stages have none, some an
// arcade all round, some arches in pairs; whole runs of bays are walled up or gone, and here and there one bay. Rows of
// identical arches on every stage read as a block of flats.
static inline bool towerLoggiaBay(uint h, int k, int face, float bay) {
    float u = hashf(pcg(h + 120u + uint(k)));
    if (u < 0.45) return false;
    int b = int(bay);
    if (u > 0.78 && ((b % 3) + 3) % 3 == 0) return false;
    if (hashf(hash3i(int3(face, int(floor(bay / 4.0)), k), h + 124u)) < 0.35) return false;
    return hashf(hash3i(int3(face, b, k), h + 121u)) > 0.12;
}

// Whether an open bay's arch and the rim over it have fallen, leaving a notch down to the loggia's floor.
static inline bool towerBayFallen(uint h, int k, int face, float bay) {
    return hashf(hash3i(int3(face, int(bay), k), h + 123u)) < 0.22;
}

// How deep the masonry is weathered back from stage face P (apothem R) at height y, in whole courses, deepest
// at the corners, some faces worn hard and others nearly sound. Isotropic in the face's plane, so it never streaks.
static inline float towerErosion(Tower T, TowerPlan P, float R, float y, uint seed) {
    float3 q = float3(P.dir.x * R, y * 0.7, P.dir.y * R);
    float n = perlin3(q / 34.0, seed + 541u) + 0.4 * perlin3(q / 12.0, seed + 542u);
    float corner = smoothstep(0.3, 0.5, abs(P.am) * float(T.N) / 6.2831853);
    float zone = saturate(perlin3(float3(P.dir * 1.6, y / 170.0), seed + 543u) * 2.0 + 0.2);
    float e = 16.0 * saturate((n + 0.3 * corner) * 2.0 - 0.5) * zone;
    // At least 6 deep wherever the face is worn at all: a depth from zero would follow the jogs of a diagonal face and
    // carve it in slivers a column wide.
    return e < 0.5 ? 0.0 : max(e, 6.0);
}

// The roots of the trees that took the tower, poured down its base onto the plinth: a few fans round the base, each
// some ridges that leave the wall together high up and part as they run out over the plinth, thinning. The root mass's
// height over the plinth's top at plan P (from the base's axis), 0 where there is none. towerSurface takes a column
// outside the base and above the plinth's top for root.
// The talus of fallen masonry heaped against the base at plan P (from the base's axis), world wf, deepest and widest on
// the side the crown fell to, lumpy, gone by 0.4 R0 out: x its height over the slope from the outer plinth's top at the
// wall down to the ground y out, y its share of the way to the wall.
static inline float2 towerTalus(Tower T, TowerPlan P, float2 wf, int s, uint seed) {
    float e = P.reff - T.R0;
    if (e < 0.0) return 0.0;
    float fall = 0.5 + 0.5 * cos(P.a - T.th0);
    float L = T.R0 * (0.24 + 0.14 * fall) * (0.85 + 0.3 * saturate(0.5 + perlin2(P.dir * 4.1, seed + 518u)));
    float f = saturate(1.0 - e / L);
    if (f <= 0.0) return 0.0;
    float amp = T.R0 * (0.15 + 0.15 * fall) * (0.6 + 0.4 * saturate(0.5 + 0.9 * perlin2(P.dir * 2.3, seed + 517u)));
    // Octaves finer than a coarse ring's columns would alias there, and its talus differ from the finer ring's.
    float lump = (s < 8 ? 4.0 * perlin2(wf / 8.0, seed + 519u) : 0.0) + (s < 4 ? 1.5 * perlin2(wf / 3.5, seed + 524u) : 0.0);
    return float2(max(amp * f * sqrt(f) + lump * saturate(f * 3.0), 0.0), f);
}

static inline float towerRootH(Tower T, TowerPlan P, uint seed) {
    float e = P.reff - T.R0;
    if (e < 0.0 || e > 0.45 * T.R0) return 0.0;
    float best = 0.0;
    int nf = 2 + int(pcg(T.h + 150u) % 3u);
    for (int i = 0; i < 4; i++) {
        if (i >= nf) break;
        uint h = pcg(T.h + 151u + uint(i) * 7u);
        float da = P.a - hashf(h) * 6.2831853;
        da -= 6.2831853 * floor(da / 6.2831853 + 0.5);
        float t = da * T.R0;
        // A fan widening as it runs out, its edge wandering.
        float hw = 9.0 + 7.0 * hashf(pcg(h + 3u)) + 0.5 * e + 4.0 * perlin2(float2(e / 14.0, float(i) * 3.3), seed + 571u);
        if (abs(t) > hw) continue;
        float Hr = 14.0 + 22.0 * hashf(pcg(h + 1u));
        float len = T.R0 * (0.25 + 0.2 * hashf(pcg(h + 2u)));
        float f = saturate(1.0 - e / len);
        // Separate roots part out of the mass: channels between them deepen outward.
        float ch = abs(perlin2(float2(t / 5.0, e / 22.0), seed + 572u));
        float y = (Hr * f * f + 2.0 * f) * smoothstep(hw, hw * 0.6, abs(t)) * (1.0 - smoothstep(0.08, 0.02, ch) * (1.0 - f * f));
        best = max(best, floor(y));
    }
    return best;
}
