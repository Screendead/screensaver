// The map's alpha for an island column from heights to planting, which turns it into its ST_ISLE word: the canopy's
// first leaf; -2 a vine; -4 less a fall's foot (0 a stream's); 0 elsewhere.
static inline float isleAlpha(Col c) {
    return !c.island ? 0.0 : c.isk == ISK_WATER ? -4.0 - c.leaf : c.isk == ISK_VINE ? -2.0 : c.leaf;
}

// An island's rock: dark basalt and umber in soft courses that drift sideways, so no bed reads as a line across a keel.
static inline float3 isleRock(float y, float2 w, uint seed, float fp) {
    float yd = y + 4.0 * perlin2(w / 70.0, seed + 713u);
    float k = saturate(0.5 + 0.7 * perlin2(float2(yd / 11.0, 0.37) + w / 500.0, seed + 714u));
    float3 c = mix(float3(0.12, 0.112, 0.125), float3(0.19, 0.14, 0.10), smoothstep(0.25, 0.75, k));
    // From one island to the next: warmer ochre on some, cooler grey on others.
    float hue = perlin2(w / 900.0, seed + 725u);
    c = mix(c, c * float3(1.3, 1.12, 0.86), saturate(2.5 * hue));
    c = mix(c, dot(c, float3(0.33)) * float3(0.95, 1.0, 1.1), saturate(-2.5 * hue));
    if (fp < 3.0) c *= 1.0 + 0.14 * perlin2(float2(w.x + w.y, yd) / 19.0, seed + 715u) * (1.0 - smoothstep(1.0, 3.0, fp));
    return c;
}

// A soft glowing spot in cell ci of face coordinates q (cell wide), present with chance pr, kept off the cell's edges;
// once its core nears a pixel (fp) it widens and dims, keeping its light, so it never shrinks to a sparkling speck.
static inline float isleSpot(float2 q, int plane, float cell, float pr, float r, float fp, uint salt, thread uint& hh) {
    float2 ci = floor(q / cell);
    hh = hash3i(int3(int2(ci), plane), salt);
    if (hashf(hh) >= pr) return 0.0;
    float m = 0.35 * cell;
    float2 c = ci * cell + m + (cell - 2.0 * m) * float2(hashf(pcg(hh)), hashf(pcg(hh + 1u)));
    float dd = length(q - c);
    if (dd >= m) return 0.0;
    float re = max(r, 0.8 * fp);
    float e2 = dd * dd / (re * re);
    return (2.0 * exp(-e2) + 0.5 * exp(-e2 / 9.0)) * (r * r) / (re * re) * (1.0 - smoothstep(0.6 * m, m, dd));
}

// An island's glow at night: sparse spots on the rock of its flanks and keel, blue and turquoise; the tips of its
// roots and vines, pulsing slowly; faint turquoise in a fall's spray. Pink on about one island in six (by place).
static float3 isleGlow(int3 v, int3 n, float2 wq, float wy, float gapTop, bool root, bool water, bool rock, float fp, uint seed, float time) {
    float far = 1.0 - smoothstep(0.5, 1.0, fp);
    bool pinkHere = hashf(hash2i(int2(floor(wq / 700.0)), seed + 750u)) < 0.17;
    if (root) {
        float tip = 1.0 - smoothstep(1.0, 3.5, wy - gapTop);
        if (tip <= 0.0) return 0.0;
        uint hr = hash2i(v.xz, seed + 751u);
        if (hashf(hr) > 0.55) return 0.0;
        float pulse = 0.7 + 0.3 * sin(time * 0.314 + 6.283 * hashf(pcg(hr)));
        float3 hue = pinkHere && hashf(pcg(hr + 1u)) < 0.5 ? BIO_LAND[2] : hashf(pcg(hr + 2u)) < 0.4 ? BIO_LAND[0] : BIO_LAND[1];
        return hue * 0.05 * tip * pulse * far;
    }
    if (water) {
        float foot = 1.0 - saturate((wy - gapTop) / 30.0);
        return n.y == 0 ? BIO_LAND[1] * 0.02 * foot * far : 0.0;
    }
    if (!rock || n.y > 0) return 0.0;
    float2 q = n.y != 0 ? wq : float2(n.x != 0 ? wq.y : wq.x, wy);
    int plane = n.x != 0 ? v.x * 2 + n.x : n.z != 0 ? v.z * 2 + n.z : -7;
    uint hh;
    float g = isleSpot(q, plane, 4.0, 0.12, 0.6, fp, seed + 752u, hh);
    if (g <= 0.0) return 0.0;
    float hp = hashf(pcg(hh + 3u));
    float3 hue = pinkHere && hp < 0.35 ? BIO_LAND[2] : hp < 0.6 ? BIO_LAND[0] : BIO_LAND[1];
    return hue * g * 0.02 * (0.75 + 0.25 * sin(time * 0.5 + 6.283 * hp)) * far;
}

// The shading table (fragment buffer 10; towers hold 9, giants 8): the islands nearest the camera (Sources/Islands.swift).
// a: centre xz, reach, the underside's mean height; b: radius, top, keel, its mist's fade; c: its fall's lip xz, top and
// foot (zero: none); mb: its mist bank's bounding sphere (radius 0: none), mp its puffs (centre, radius); o.x the square of
// how far its sky occlusion reaches (Islands.occlusionReach). World coordinates.
constant int ISLE_TABLE = 16;
constant int ISLE_MIST_PUFFS = 6;
// The mist's depth where its bodies begin (isleMist): banks and plumes that cannot reach most of it alone are not worked out.
constant float ISLE_MIST_TAU0 = 0.5;
struct IsleShade { float4 a; float4 b; float4 c; float4 mb; float4 mp[ISLE_MIST_PUFFS]; float4 o; };
struct IsleTable { int4 n; IsleShade r[ISLE_TABLE]; };
