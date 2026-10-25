// Reefs grow in the warm lagoons only. Where, from cheap fields of seed-offset position p alone, so the shading can ask
// again what the terrain decided: broad reefy reaches, cut by meandering sandy channels.
static inline float reefRegion(float2 p, float lagoonRaw, uint seed) {
    float r = smoothstep(-0.06, 0.06, seaNoise(p / 520.0, seed + 140u) + 0.14) * smoothstep(0.45, 0.65, lagoonRaw);
    if (r <= 0.0) return 0.0;
    float ch = abs(seaNoise(p / 75.0, seed + 141u));
    return r * smoothstep(0.03, 0.06, ch);
}

constant float REEF_CELL = 5.0;
constant int CORAL_BOULDER = 0, CORAL_STAG = 1, CORAL_TABLE = 2, CORAL_FAN = 3;
// Colonies grow on the reef's shallows, not on the deeper sand: scattered there they would leave the trace no open water
// to skip.
constant float REEF_FLOOR = 15.0;

// A coral colony stands on a reef: rise, its height in voxels over the reef at this column; tip, 1 on its upper
// voxels (staghorn branch ends, a table's rim, a boulder's crown, a fan's edge); h, its hash.
struct Coral { float rise; int kind; float tip; uint h; };

static inline float byteOf(uint h, uint k) { return float((h >> (8u * k)) & 255u) / 255.0; }

// One pcg per colony and per column: the shading asks again for every reef pixel. reefMix spreads a hash's bits
// into a second word.
static inline uint reefHash(int2 c, uint s) { return pcg(uint(c.x) * 1597334677u ^ uint(c.y) * 3812015801u ^ s); }
static inline uint reefMix(uint h) { h *= 0x9e3779b1u; return h ^ (h >> 15); }

// The colony of REEF_CELL cell c, if any: its centre cc (world xz), kind, size 0..1 and hash h2.
static inline bool coralColony(int2 c, uint seed, thread float2& cc, thread int& kind, thread float& sz, thread uint& h2) {
    uint h = reefHash(c, seed + 150u);
    // Colonies clump, leaving sandy openings between thickets.
    if (hashf(h) > 0.82 * smoothstep(-0.3, 0.1, seaNoise(float2(c) * (REEF_CELL / 40.0), seed + 152u))) return false;
    h2 = reefMix(h);
    cc = (float2(c) + 0.2 + 0.6 * float2(byteOf(h2, 0), byteOf(h2, 1))) * REEF_CELL;
    float k01 = byteOf(h2, 2);
    kind = k01 < 0.34 ? CORAL_BOULDER : k01 < 0.70 ? CORAL_STAG : k01 < 0.86 ? CORAL_TABLE : CORAL_FAN;
    sz = byteOf(h, 3);
    return true;
}

// The glow palette of land, sea and reef: deep blue, turquoise and pink (#2a6bff, #2fe3d0, #ff5fc8), linear.
constant float3 BIO_LAND[3] = { float3(0.023, 0.147, 1.0), float3(0.028, 0.768, 0.630), float3(1.0, 0.114, 0.578) };

static inline float3 coralGlowHue(uint g) {
    float k = byteOf(g, 1);
    return k < 0.45 ? BIO_LAND[1] : k < 0.7 ? mix(BIO_LAND[1], BIO_LAND[0], 0.5) : BIO_LAND[2] * 0.8;
}

// The colony at world column w (h: 7 bits of its hash). Colonies lie within REEF_CELL / 2 of their centres, so a column
// asks the 2x2 cells nearest it.
static inline Coral coralAt(int2 w, uint seed) {
    Coral best;
    best.rise = 0; best.kind = 0; best.tip = 0; best.h = 0;
    float2 q = float2(w) + 0.5;
    int2 c0 = int2(floor(q / REEF_CELL - 0.5));
    // Staghorn branches by column; every kind's shape is worked out and one picked, as branching by kind would run
    // them all one after another wherever neighbouring pixels differ.
    uint hb = reefHash(w, seed + 151u);
    float stagB = byteOf(hb, 0), stagH = byteOf(hb, 1);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            // A colony reaches 3 from its centre, which keeps to the middle 0.6 of its cell.
            float2 lo = (float2(c0 + int2(i, j)) + 0.2) * REEF_CELL;
            if (length_squared(q - clamp(q, lo, lo + 0.6 * REEF_CELL)) > 3.3 * 3.3) continue;
            float2 cc;
            int kind;
            float sz;
            uint h2;
            if (!coralColony(c0 + int2(i, j), seed, cc, kind, sz, h2)) continue;
            float2 v = q - cc;
            float dd = length(v);
            float R = kind == CORAL_BOULDER ? 1.4 + 1.4 * sz : kind == CORAL_STAG ? 1.8 + 1.2 * sz : 1.6 + 1.3 * sz;
            float inR = step(dd, R);
            float Hc = 1.0 + floor(2.2 * sz + 0.5);
            float boulder = floor(Hc * sqrt(saturate(1.0 - dd * dd / (R * R))) + 0.65) * inR;
            float Hb = 1.0 + floor(2.99 * stagH * (1.0 - 0.5 * dd / R));
            float stag = (stagB < 0.55 ? Hb : stagB < 0.8 ? 1.0 : 0.0) * inR;
            float table = (2.0 + floor(1.4 * sz)) * inR;
            float2 ax = normalize(float2(byteOf(h2, 3), byteOf(h2, 0)) - 0.5 + 1e-3);
            float L = 1.5 + 1.3 * sz;
            float u = clamp(dot(v, ax), -L, L);
            float fan = length(v - ax * u) < 0.62 ? floor(3.0 - 1.6 * abs(u) / L + 0.5) : 0.0;
            float rise = kind == CORAL_BOULDER ? boulder : kind == CORAL_STAG ? stag : kind == CORAL_TABLE ? table : fan;
            float tip = kind == CORAL_BOULDER ? step(Hc - 0.5, rise) : kind == CORAL_STAG ? step(1.5, rise)
                      : kind == CORAL_TABLE ? smoothstep(R - 1.2, R - 0.4, dd) : 1.0;
            if (rise > best.rise) { best.rise = rise; best.kind = kind; best.tip = tip; best.h = h2 & 127u; }
        }
    return best;
}

// Day colours of a colony, muted so the water's own tint leads: dark olive boulders, mauve and brown staghorn, sage and
// tan tables, rose and violet fans, one in five warmer; barely paler at the tips.
static inline float3 coralColour(Coral cr) {
    uint g = pcg(cr.h + 3u);
    float k = byteOf(g, 0);
    bool warm = byteOf(g, 1) < 0.2;
    float3 c = cr.kind == CORAL_BOULDER ? (warm ? float3(0.36, 0.27, 0.12) : mix(float3(0.22, 0.21, 0.11), float3(0.18, 0.22, 0.12), k))
             : cr.kind == CORAL_STAG ? (warm ? float3(0.34, 0.20, 0.13) : mix(float3(0.27, 0.18, 0.19), float3(0.28, 0.17, 0.12), k))
             : cr.kind == CORAL_TABLE ? mix(float3(0.15, 0.22, 0.17), float3(0.26, 0.23, 0.15), k)
             : (warm ? float3(0.38, 0.15, 0.22) : mix(float3(0.31, 0.14, 0.17), float3(0.27, 0.15, 0.24), k));
    c = mix(float3(dot(c, float3(0.3, 0.59, 0.11))), c, 1.1);
    return mix(c, c * 1.08, cr.tip);
}

// The 1-voxel map keeps each reef column's make in the spare high bits of its extra-gap word (bits 11-15 of x, y and w;
// extraGaps reads the low 11): bits 0-3 the reef's share (reefRegion, in 15ths), 4 a colony, 5-6 its kind, 7 a tip,
// 8-14 its hash. The shading reads it rather than working the colonies out again.
static inline uint4 reefPack(uint4 g, uint code) {
    return uint4(g.x | ((code & 31u) << 11), g.y | (((code >> 5) & 31u) << 11), g.z, g.w | (((code >> 10) & 31u) << 11));
}
static inline uint reefCode(uint4 g) { return (g.x >> 11) | ((g.y >> 11) << 5) | ((g.w >> 11) << 10); }

// Night: about one colony in four glows, soft and dim, at its tips most.
static inline float3 coralGlowCode(uint code, float time) {
    if ((code & 16u) == 0u) return 0;
    uint gh = reefMix((code >> 8) ^ 0x5bd1e995u);
    if (byteOf(gh, 0) >= 0.24) return 0;
    return coralGlowHue(gh) * (0.7 + 0.3 * sin(time * (0.4 + 0.3 * byteOf(gh, 2)) + 6.2831853 * byteOf(gh, 3))) * ((code & 128u) != 0u ? 1.0 : 0.4);
}

// The reef floor between colonies: sand under algal turf and rubble.
constant float3 REEF_TURF = float3(0.27, 0.27, 0.19);
// The mean of a reef's colonies and floor, for coarse columns.
constant float3 REEF_MEAN = float3(0.33, 0.28, 0.21);
// Reef colours and glow fade out by here: past it the haze and the surface's mirror hide the bed.
constant float REEF_FAR = 1100.0;
