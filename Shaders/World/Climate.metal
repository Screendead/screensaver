// jungle, old, isle: the province fields (provinces, isleProvince), 0 outside them.
struct Climate { float temp; float moist; float fall; float basin; float jungle; float old; float isle; };

static inline float2 seedOffset(uint seed) {
    return float2(hashf(pcg(seed)), hashf(pcg(seed ^ 0x9e3779b9u))) * 50000.0;
}

// Jungle (x) and old-growth (y) provinces: the climate fields swing across a few hundred voxels, too fast for a forest
// that holds giants hundreds of voxels across, so these regions come from their own fields, 2 km and more across
// (jungle about a tenth of the land, old growth about a twentieth). p is seed-offset world position. The textures
// bake only the climate they push; the fields are evaluated wherever they are read.
// A biome chosen from the menu rides in the seed: top byte 0xFF, then the biome (1-9, Biome in Sources/Menu.swift); 0 is
// the whole world. The world is then that biome everywhere, its climate fields squeezed into the biome's ranges.
static inline uint pinnedBiome(uint seed) { return seed >> 24 == 0xFFu ? (seed >> 20) & 15u : 0u; }

static inline float2 provinces(float2 p, uint seed) {
    uint b = pinnedBiome(seed);
    if (b != 0) return float2(b == 8 ? 1.0 : 0.0, b == 9 ? 1.0 : 0.0);
    float J = smoothstep(0.24, 0.32, fbm2(p / 6000.0 + 300.7, 2, seed + 300u));
    float O = smoothstep(0.30, 0.38, fbm2(p / 6000.0 + 411.3, 2, seed + 301u)) * (1.0 - J);
    return float2(J, O);
}

static inline float towerProvinceField(float2 p, uint seed);

// Floating-island country, about a twentieth of the world: a third province, outside the forest provinces (pv: their
// fields) and the mega relief (its field stays under the threshold megaLift rises from). p is seed-offset world position.
static inline float isleProvince(float2 p, float2 pv, uint seed) {
    float f = smoothstep(0.20, 0.28, fbm2(p / 7000.0 + 523.1, 2, seed + 302u)) * (1.0 - pv.x) * (1.0 - pv.y);
    return f > 0.0 ? f * smoothstep(0.50, 0.42, towerProvinceField(p, seed)) : 0.0;
}

static inline Climate climate(float2 w, uint seed) {
    float2 p = w + seedOffset(seed);
    Climate c;
    c.temp = saturate(fbm2(p / 900.0, 2, seed + 11) * 1.6 + 0.5);
    c.moist = saturate(fbm2(p / 800.0 + 91.0, 2, seed + 12) * 1.6 + 0.5);
    c.fall = saturate(perlin2(p / 1000.0 + 213.0, seed + 15) * 2.0 + 0.5);
    c.basin = saturate(fbm2(p / 1100.0 + 53.0, 2, seed + 15) * 1.6 + 0.5);
    uint b = pinnedBiome(seed);
    if (b >= 1 && b <= 7) {
        // (temp, moist, fall, basin) low and high, inside each biome's thresholds so it holds everywhere.
        const float4 lo[7] = { float4(0.38, 0.35, 0.00, 0.00), float4(0.36, 0.50, 0.62, 0.00), float4(0.10, 0.35, 0.00, 0.00),
                               float4(0.70, 0.00, 0.00, 0.00), float4(0.62, 0.45, 0.00, 0.00), float4(0.30, 0.10, 0.00, 0.72),
                               float4(0.68, 0.68, 0.00, 0.00) };
        const float4 hi[7] = { float4(0.46, 0.62, 0.30, 0.55), float4(0.44, 0.70, 0.95, 0.55), float4(0.32, 0.70, 0.30, 0.55),
                               float4(0.95, 0.26, 0.30, 0.55), float4(0.90, 0.52, 0.30, 0.55), float4(0.44, 0.35, 0.30, 1.00),
                               float4(0.90, 0.92, 0.30, 0.55) };
        float4 v = mix(lo[b - 1], hi[b - 1], float4(c.temp, c.moist, c.fall, c.basin));
        c.temp = v.x; c.moist = v.y; c.fall = v.z; c.basin = v.w;
    }
    float2 pv = provinces(p, seed);
    c.jungle = pv.x;
    c.old = pv.y;
    c.isle = isleProvince(p, pv, seed);
    // Jungle is warm and wet, old growth cool and wet; the biomes those climates would otherwise make there (lagoon,
    // desert, painted hills, salt, autumn) are gated off by the province, so no ring of them forms at its edge.
    c.temp = mix(c.temp, max(c.temp, 0.68), pv.x);
    c.temp = mix(c.temp, clamp(c.temp, 0.26, 0.40), pv.y);
    c.moist = mix(c.moist, max(c.moist, 0.75), max(pv.x, pv.y));
    // Island country is mild and green: meadow, lakes and the jungle's edge under the islands.
    float ic = smoothstep(0.0, 0.5, c.isle);
    c.temp = mix(c.temp, clamp(c.temp, 0.48, 0.6), ic);
    c.moist = mix(c.moist, max(c.moist, 0.66), ic);
    return c;
}

// Painted hills: the hot, semi-dry belt between grassland and desert.
// Climate is smooth over hundreds of voxels but costs seven noise octaves, so each map bakes it per column.
kernel void bakeClimate(texture2d<float, access::write> out [[texture(0)]],
                        constant int4& info [[buffer(0)]],
                        constant uint& row0 [[buffer(1)]],
                        uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    Climate c = climate(float2(int2(gid) + info.xy), uint(info.w));
    out.write(float4(c.temp, c.moist, c.fall, c.basin), gid);
}

// The province fields baked around the camera (bakeProvinces): PROV_N texels a side, PROV_CELL voxels each, held
// toroidally by world texel. Must match Renderer.provN and provCell in Sources/Renderer.swift.
constant int PROV_N = 512;
constant float PROV_CELL = 96.0;

static inline float3 provincesBaked(texture2d<float> pv, float2 w) {
    constexpr sampler br(filter::linear, address::repeat);
    return pv.sample(br, w / (PROV_CELL * float(PROV_N)), level(0)).rgb;
}

kernel void bakeProvinces(texture2d<float, access::write> out [[texture(0)]],
                          constant int4& info [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(PROV_N) || gid.y >= uint(PROV_N)) return;
    // info.xy: the window's first texel (world texel index); w: seed.
    int2 c = info.xy + int2(gid);
    uint seed = uint(info.w);
    float2 p = (float2(c) + 0.5) * PROV_CELL + seedOffset(seed);
    float2 pv = provinces(p, seed);
    out.write(float4(pv, isleProvince(p, pv, seed), 0), uint2(c & (PROV_N - 1)));
}

// Baked climate at map-local position p (column centres at integer + 0.5); pv, origin: the province texture and the
// map's origin.
static inline Climate climateAt(texture2d<float> t, float2 p, texture2d<float> pv, int2 origin) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    float4 v = t.sample(bl, p / float2(t.get_width(), t.get_height()), level(0));
    Climate c;
    c.temp = v.x; c.moist = v.y; c.fall = v.z; c.basin = v.w;
    float3 q = provincesBaked(pv, p + float2(origin));
    c.jungle = q.x; c.old = q.y; c.isle = q.z;
    return c;
}

// The share of a column in either forest province, where the other biomes give way.
static inline float provinceAmount(Climate c) { return max(c.jungle, c.old); }

static inline float paintedAmount(Climate c) {
    return smoothstep(0.45, 0.6, c.temp) * smoothstep(0.30, 0.42, c.moist) * smoothstep(0.66, 0.54, c.moist) * (1.0 - provinceAmount(c));
}

static inline float desertAmount(Climate c) {
    return smoothstep(0.45, 0.68, c.temp) * smoothstep(0.7, 0.4, c.moist) * (1.0 - paintedAmount(c)) * (1.0 - provinceAmount(c));
}

// Level of the flat alluvial floors between painted hills.
static inline float paintedFloor(float2 p, uint seed) {
    return SEA + 3.0 + 8.0 * saturate(perlin2(p / 700.0, seed + 110u) * 0.7 + 0.5);
}

// Autumn hardwoods: a third climate field picks temperate, moist country out of the grassland.
static inline float autumnAmount(Climate c) {
    return smoothstep(0.38, 0.60, c.fall) * smoothstep(0.25, 0.50, c.moist) * smoothstep(0.20, 0.36, c.temp)
         * (1.0 - smoothstep(0.1, 0.4, desertAmount(c))) * (1.0 - provinceAmount(c));
}

static inline float lagoonAmount(Climate c) {
    return smoothstep(0.47, 0.65, c.temp) * smoothstep(0.47, 0.65, c.moist) * (1.0 - provinceAmount(c));
}

// Salt pans: dry basins of any temperature. 1 on the open pan, falling through the terraced rims.
static inline float saltAmount(Climate c) {
    return smoothstep(0.6, 0.75, c.basin) * smoothstep(0.62, 0.42, c.moist) * (1.0 - provinceAmount(c));
}

// Dry crust stands just above the water plane; pools are one voxel lower, so they hold a skin of water.
constant float PAN = 20.0;

// Pools fill where this exceeds poolLevel; p is seed-offset world position.
static inline float poolNoise(float2 p, uint seed) { return fbm2(p / 140.0, 3, seed + 16) + 0.25 * perlin2(p / 35.0, seed + 17); }
static inline float poolLevel(float salt) { return 0.08 + 0.5 * (1.0 - smoothstep(0.8, 0.95, salt)); }
