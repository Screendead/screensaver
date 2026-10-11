#include <metal_stdlib>
using namespace metal;

// Must match `Frame` in Sources/Renderer.swift.
struct Frame {
    float4 camPos;     // xyz map-local position, w tan(fovY/2)
    float4 camFwd;     // w aspect
    float4 camRight;   // w wrapped time (s)
    float4 camUp;      // w quality (0 battery, 1 full)
    float4 sunDir;     // xyz direction to sun, w day phase
    float4 resolution; // xy pixels, zw inverse
    int4   mapInfo;    // xy world xz of texel 0, z size, w seed
    float4 misc;       // x exposure, y global max height, z wind angle, w cloud cover
    float4 misc2;      // x frame index, y fog density scale, z debug mode, w EDR headroom (0 = SDR)
    float4 prevPos;    // previous frame's camera, in this frame's map-local coords; w history valid
    float4 prevFwd;
    float4 prevRight;
    float4 prevUp;
    float4 jitter;     // xy subpixel jitter in NDC
    float4 cloud;      // xy cloud field origin in drifted world xz, z field texel size (0: no field)
    float4 lights;     // x sun elevation, y night progress (0 dusk .. 1 dawn), z day index, w lamp light on the land (0 by day)
    int4   refs;       // x lamps in the haze list, y smoke plumes
    float4 birds;      // x bird flock count
    float4 lod[5];     // per level: x morph band start, y handoff radius (the last level's is the world's edge), z morph end, w column width
    float4 fade;       // x fog fade start, y world radius, z level count, w ring centre's offset ahead of the camera (tests)
    float4 shadow;     // x 1 when the shadow-height maps hold the current key light (3 no shadows), y profiling flags, z 1 when
                       // the horizon buffer is valid, w 1 when an island is in view
    float4 slowCloud;  // xy slow cloud field origin in world xz, z texel size (0: no field), w the time it was baked for
};

constant float SEA = 19.85;
constant int MAX_LEVEL = 8;

// MARK: - Hash & noise

static inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline uint hash2i(int2 p, uint s) { return pcg(uint(p.x) * 1597334677u ^ pcg(uint(p.y) * 3812015801u ^ s)); }
static inline uint hash3i(int3 p, uint s) { return pcg(uint(p.x) * 1597334677u ^ pcg(uint(p.y) * 3812015801u ^ pcg(uint(p.z) * 2798796415u ^ s))); }
static inline float hashf(uint h) { return float(h & 0xffffffu) / 16777215.0; }

static inline float2 grad2(int2 c, uint s) {
    float a = hashf(hash2i(c, s)) * 6.2831853;
    return float2(cos(a), sin(a));
}

static inline float perlin2(float2 p, uint s) {
    float2 i = floor(p), f = p - i;
    int2 c = int2(i);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = dot(grad2(c, s), f);
    float b = dot(grad2(c + int2(1, 0), s), f - float2(1, 0));
    float cc = dot(grad2(c + int2(0, 1), s), f - float2(0, 1));
    float d = dot(grad2(c + int2(1, 1), s), f - float2(1, 1));
    return mix(mix(a, b, u.x), mix(cc, d, u.x), u.y) * 1.414;
}

static inline float3 grad3(int3 c, uint s) {
    uint h = hash3i(c, s);
    float z = hashf(h) * 2.0 - 1.0;
    float a = hashf(pcg(h)) * 6.2831853;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(a), r * sin(a), z);
}

static inline float perlin3(float3 p, uint s) {
    float3 i = floor(p), f = p - i;
    int3 c = int3(i);
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = dot(grad3(c, s), f);
    float n100 = dot(grad3(c + int3(1, 0, 0), s), f - float3(1, 0, 0));
    float n010 = dot(grad3(c + int3(0, 1, 0), s), f - float3(0, 1, 0));
    float n110 = dot(grad3(c + int3(1, 1, 0), s), f - float3(1, 1, 0));
    float n001 = dot(grad3(c + int3(0, 0, 1), s), f - float3(0, 0, 1));
    float n101 = dot(grad3(c + int3(1, 0, 1), s), f - float3(1, 0, 1));
    float n011 = dot(grad3(c + int3(0, 1, 1), s), f - float3(0, 1, 1));
    float n111 = dot(grad3(c + int3(1, 1, 1), s), f - float3(1, 1, 1));
    float x0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    float x1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(x0, x1, u.z) * 1.15;
}

constant float2x2 OCT_ROT = float2x2(float2(0.8, 0.6), float2(-0.6, 0.8));

static inline float fbm2(float2 p, int oct, uint s) {
    float sum = 0, amp = 0.5, norm = 0;
    for (int i = 0; i < oct; i++) {
        sum += amp * perlin2(p, s + uint(i) * 1013u);
        norm += amp;
        p = OCT_ROT * p * 2.0 + 17.3;
        amp *= 0.5;
    }
    return sum / norm;
}

static inline float ridged2(float2 p, int oct, uint s) {
    float sum = 0, amp = 0.5, norm = 0, w = 1;
    for (int i = 0; i < oct; i++) {
        float n = 1.0 - abs(perlin2(p, s + uint(i) * 7919u));
        n *= n;
        sum += amp * n * w;
        w = saturate(n * 1.5);
        norm += amp;
        p = OCT_ROT * p * 2.0 + 31.7;
        amp *= 0.5;
    }
    return sum / norm;
}

// MARK: - World

// jungle, old, isle: the province fields (provinces, isleProvince), 0 outside them.
struct Climate { float temp; float moist; float fall; float basin; float jungle; float old; float isle; };

static inline float2 seedOffset(uint seed) {
    return float2(hashf(pcg(seed)), hashf(pcg(seed ^ 0x9e3779b9u))) * 50000.0;
}

// Jungle (x) and old-growth (y) provinces: the climate fields swing across a few hundred voxels, too fast for a forest
// that holds giants hundreds of voxels across, so these regions come from their own fields, 2 km and more across
// (jungle about a tenth of the land, old growth about a twentieth). p is seed-offset world position. The textures
// bake only the climate they push; the fields are evaluated wherever they are read.
static inline float2 provinces(float2 p, uint seed) {
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

// MARK: - Mega relief

// Heights are rgba16Float/r16Float everywhere (map, rings, gaps, first leaf, the CPU copy): integers are exact only
// up to 2048, so land stops at 2000 and trees on the highest summit still fit.
constant float TERRAIN_CEIL = 2000.0;

// Lift over the lowland, and the signals a later wall/overhang pass needs. Provinces about 30% of the land, each
// one landform across its extent, separated by lowland 6-12k wide: alpine ranges, tablelands with sheer
// escarpments, karst pillar forests; a low-frequency river net cuts gorges through them, and a rare tepui stands
// on its own. U is 0 outside them, so the lowland there is exactly as before. p is seed-offset world position.
// lipU: the lift the column would have on the cap rock behind an escarpment's edge (0: no lip here); lipR: how far the
// cap reaches out over the soft beds, lipU01: the column's place under it (0 at the wall line, 1 at the lip's edge).
// notchFoot, notchH: a notch cut round a karst tower's foot, its floor that far above the lowland ground (-1: at the
// water line) and its height at this column (0: none).
struct Mega { float U; float wK; float lipU; float lipR; float lipU01; float notchFoot; float notchH; };

// The tableland field, also tapped beside a column to measure its distance to an escarpment in voxels.
static inline float tableField(float2 p, float M, uint seed) {
    float2 tw = float2(fbm2(p / 1500.0, 3, seed + 220u), fbm2(p / 1500.0 + 5.3, 3, seed + 221u)) * 700.0
              + float2(perlin2(p / 260.0, seed + 222u), perlin2(p / 260.0 + 3.1, seed + 223u)) * 90.0;
    return fbm2((p + tw) / 6000.0, 3, seed + 224u) * 1.4 + 0.5 + 0.5 * (M - 0.5);
}

static inline Mega megaLift(float2 p, float lagoon, uint seed) {
    Mega g;
    g.U = 0.0;
    g.wK = 0.0;
    g.lipU = 0.0; g.lipR = 0.0; g.lipU01 = 0.0; g.notchFoot = 0.0; g.notchH = 0.0;
    float lipU = 0.0, uT = 0.0;
    float2 cw = float2(fbm2(p / 7000.0, 2, seed + 200u), fbm2(p / 7000.0 + 71.3, 2, seed + 201u)) * 3000.0;
    float2 q = p + cw;
    float m = fbm2(q / 16000.0, 3, seed + 202u) * 1.6 + 0.5;
    float M = smoothstep(0.50, 0.80, m);
    if (M > 0.0) {
        float a = perlin2(q / 22000.0, seed + 203u), b = perlin2(q / 22000.0 + 9.7, seed + 204u);
        float wA = smoothstep(-0.12, 0.12, a);
        float wT = (1.0 - wA) * smoothstep(-0.12, 0.12, b);
        float wK = (1.0 - wA) * (1.0 - wT);
        float core = smoothstep(0.55, 0.95, m);
        float U = 0.0;
        if (wA > 0.0) {
            // Ridged multifractal, domain-warped into ranges; its finest octave is 200 wide, so the 16-voxel ring
            // carries every octave. The pow keeps the valleys between ridges low.
            float2 aw = float2(fbm2(p / 1800.0, 3, seed + 210u), fbm2(p / 1800.0 + 13.1, 3, seed + 211u)) * 500.0;
            float r = ridged2((p + aw) / 3200.0, 5, seed + 212u);
            U += wA * M * (120.0 + 1450.0 * pow(saturate(r), 1.8)) * mix(0.55, 1.0, core);
        }
        if (wT > 0.0) {
            // Two tiers, each a sloping talus apron (a quarter of its height) under a sheer cap rock; outliers of the
            // field near a threshold stand as mesas, then buttes.
            float sv = tableField(p, M, seed);
            float s1 = 0.25 * smoothstep(0.38, 0.45, sv) + 0.75 * smoothstep(0.45, 0.456, sv);
            float s2 = 0.25 * smoothstep(0.55, 0.62, sv) + 0.75 * smoothstep(0.62, 0.626, sv);
            float roll = 18.0 * fbm2(p / 400.0, 2, seed + 225u);
            float fT = wT * smoothstep(0.0, 0.3, M);
            uT = fT * (430.0 * s1 + 380.0 * s2 + roll * s1);
            U += uT;
            // Caprock lips: where the cap rock is hard it reaches out over the soft beds of the wall below, by a distance in
            // voxels (the field's offset from the top of the wall over its gradient), so the overhang is a slab of even
            // reach rather than one that thins to nothing where the field is flat. The column takes the cap's own height.
            float reachK = smoothstep(0.30, 0.75, perlin2(p / 520.0, seed + 262u) * 0.9 + 0.5);
            float thr = sv < 0.54 ? 0.456 : 0.626;
            if (fT > 0.3 && reachK > 0.0 && sv < thr && sv > thr - 0.03) {
                float e = 3.0;
                float gx = tableField(p + float2(e, 0), M, seed) - sv, gz = tableField(p + float2(0, e), M, seed) - sv;
                float grad = max(length(float2(gx, gz)) / e, 1e-6);
                float delta = (thr - sv) / grad;
                float R = 40.0 * reachK;
                if (delta < R && R > 4.0) {
                    float c1 = thr < 0.5 ? 1.0 : s1, c2 = thr < 0.5 ? s2 : 1.0;
                    lipU = fT * (430.0 * c1 + 380.0 * c2 + roll * c1);
                    g.lipR = R;
                    g.lipU01 = delta / R;
                }
            }
        }
        if (wK > 0.0) {
            // Tower karst. A slow field gathers the towers into clusters (fengcong: peaks crowding on a shared rocky
            // pedestal) with open plains between, where towers stand alone (fenglin). Two populations: slender
            // towers on a ~50 lattice and massive ones on a ~130 lattice, so girth and spacing vary; each tower's
            // height comes from noise on its own scale, so neighbours differ. Each stands on a talus apron, its sheer
            // shaft ends in a rounded, craggy crown, and thin needles rise among the fringe towers.
            float P = 280.0 + 260.0 * saturate(fbm2(q / 5000.0, 2, seed + 230u) * 1.5 + 0.5);
            float2 kw = float2(perlin2(p / 60.0, seed + 231u), perlin2(p / 60.0 + 7.7, seed + 232u)) * 14.0;
            float clus = saturate(fbm2(q / 750.0, 2, seed + 239u) * 3.0 + 0.3 + 0.2 * (core - 0.5));
            float dens = clus * clus * (3.0 - 2.0 * clus);
            float c1 = perlin2((p + kw) / 52.0, seed + 233u);
            float c2 = perlin2((p + kw) / 23.0, seed + 235u);
            float cB = perlin2((p + kw * 2.5) / 130.0, seed + 245u);
            float thr = mix(0.5, 0.22, dens) + 0.16 * perlin2(p / 170.0, seed + 248u);
            float skirt = smoothstep(thr - 0.14, thr, c1);
            float big = smoothstep(thr, thr + 0.02, c1);
            float crown = smoothstep(thr + 0.01, thr + 0.18, c1);
            float thrB = mix(0.42, 0.27, dens);
            float skirtB = smoothstep(thrB - 0.16, thrB, cB);
            float bigB = smoothstep(thrB, thrB + 0.015, cB);
            float crownB = smoothstep(thrB + 0.01, thrB + 0.16, cB);
            float thin = smoothstep(0.52, 0.54, c2) * (1.0 - dens) * smoothstep(0.1, 0.4, perlin2(p / 900.0, seed + 236u) + 0.3);
            float hv = 0.25 + 0.75 * saturate(perlin2(p / 61.0, seed + 234u) * 1.3 + 0.5);
            float hB = 0.45 + 0.5 * saturate(perlin2(p / 150.0, seed + 246u) * 1.2 + 0.5);
            float crag = perlin2(p / 11.0, seed + 237u) + 0.5 * perlin2(p / 4.7, seed + 238u);
            float shaft = P * M * hv * (0.10 * skirt + 0.62 * big + 0.28 * crown * (2.0 - crown)) * 0.5;
            float shaftB = P * M * hB * (0.12 * skirtB + 0.58 * bigB + 0.30 * crownB * (2.0 - crownB));
            // The cluster's pedestal: a rolling rock base the crowded towers share.
            float ped = P * M * 0.22 * dens * smoothstep(-0.15, 0.35, perlin2((p + kw * 3.0) / 210.0, seed + 247u));
            float kh = max(max(shaft, shaftB) + 9.0 * crag * max(big, bigB), P * M * hv * 0.8 * thin);
            kh = max(kh, ped + 0.5 * kh);
            U += wK * max(kh, 0.0);
            // Notches: the rock just inside a big tower's foot is undercut, a rounded nip at the water line or above the
            // talus, its depth a distance in voxels inside the tower's edge (the field over its gradient), so a needle
            // thinner than two notches is pierced right through.
            bool useB = shaftB > shaft;
            float baseT = P * M * (useB ? hB : hv);
            float F = useB ? cB : c1, tF = (useB ? thrB : thr) + 0.01;
            if (wK > 0.5 && baseT > 110.0 && F > tF && F < tF + 0.4 && thin < 0.5) {
                float2 e1 = float2(1, 0), e2 = float2(0, 1);
                float2 k1 = float2(perlin2((p + e1) / 60.0, seed + 231u), perlin2((p + e1) / 60.0 + 7.7, seed + 232u)) * 14.0;
                float2 k2 = float2(perlin2((p + e2) / 60.0, seed + 231u), perlin2((p + e2) / 60.0 + 7.7, seed + 232u)) * 14.0;
                float Fx = useB ? perlin2((p + e1 + k1 * 2.5) / 130.0, seed + 245u) : perlin2((p + e1 + k1) / 52.0, seed + 233u);
                float Fz = useB ? perlin2((p + e2 + k2 * 2.5) / 130.0, seed + 245u) : perlin2((p + e2 + k2) / 52.0, seed + 233u);
                float grad = max(length(float2(Fx - F, Fz - F)), 1e-6);
                float delta = (F - tF) / grad;
                float D = 4.0 + 8.0 * saturate(perlin2(p / 150.0, seed + 265u) + 0.5);
                if (delta < D) {
                    float v = delta / D;
                    float hn = 8.0 + 12.0 * saturate(perlin2(p / 120.0, seed + 266u) + 0.5);
                    g.notchH = hn * sqrt(1.0 - v * v);
                    g.notchFoot = wK * max(baseT * (useB ? 0.12 : 0.05), ped + baseT * (useB ? 0.06 : 0.025));
                }
            }
        }
        // Lagoon country keeps only its pillars standing in the water.
        float keepU = mix(1.0, wK, lagoon);
        // Rivers cut gorges: slots through tableland and karst, broader valleys through ranges.
        float gorge = abs(fbm2(q / 5000.0, 3, seed + 250u));
        keepU *= smoothstep(0.008, mix(0.035, 0.12, wA), gorge);
        // No lips over a gorge's slot, where the cap would roof the river.
        if (lipU > 0.0) lipU = keepU > 0.98 ? (U - uT + lipU) * keepU : 0.0;
        U *= keepU;
        g.notchFoot *= keepU;
        if (keepU < 0.98) g.notchH = 0.0;
        g.U = U;
        g.wK = wK;
        g.lipU = lipU;
    }
    // Tepui: in about one 30k cell in three, a sheer-walled table mountain over a talus skirt.
    const float CELL = 30000.0;
    int2 cell = int2(floor(p / CELL));
    uint hc = hash2i(cell, seed + 240u);
    if (hashf(hc) < 0.35) {
        float2 c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(hc)), hashf(pcg(hc + 1u)))) * CELL;
        float R = 900.0 + 600.0 * hashf(pcg(hc + 2u));
        float dd = length(p - c);
        if (dd < R + 1300.0) {
            // Lobes and buttresses, so the rim reads as eroded rock, not a drum.
            float d = dd + 520.0 * fbm2(p / 1100.0, 3, seed + 241u) + 140.0 * perlin2(p / 330.0, seed + 244u)
                    + 50.0 * perlin2(p / 140.0, seed + 242u);
            float Hl = 1500.0 + 350.0 * hashf(pcg(hc + 3u));
            float t = 0.22 * smoothstep(R + 420.0, R + 10.0, d) + 0.78 * smoothstep(R + 10.0, R - 6.0, d);
            float top = 30.0 * fbm2(p / 350.0, 2, seed + 243u) * smoothstep(R, R - 80.0, d);
            float T = Hl * t + top;
            if (T > g.U) { g.U = T; g.wK = 0.0; g.lipU = 0.0; g.notchH = 0.0; }
        }
    }
    return g;
}

// A generated column: solid where y < h, except inside its gaps. [gl, gh) is the principal (highest) gap, [lo1, hi1) and
// [lo2, hi2) up to two more below it, highest first (empty when hi <= lo). carved: the gaps are rock carved by the mega
// relief (overhangs, caves, bridges), whose roofs rest on any neighbour rather than all four; canyon arches are not.
struct Col { float h, gl, gh, lo1, hi1, lo2, hi2; bool carved; bool island; };

// Extra gaps (World gx, rgba16Uint): x lo1, y hi1, z lo2, w hi2 in the low 11 bits (heights stay under 2048); the top 5
// bits of z hold flags (bit 0: carved). An empty gap is lo = hi = 0.
constant uint GAP_BITS = 0x7ffu;
constant uint GAP_CARVED = 1u;
constant uint GAP_ISLAND = 4u;

static inline uint4 packGaps(Col c) {
    uint flags = (c.carved ? GAP_CARVED : 0u) | (c.island ? GAP_ISLAND : 0u);
    bool g1 = c.hi1 > c.lo1, g2 = c.hi2 > c.lo2;
    return uint4(g1 ? uint(c.lo1) : 0u, g1 ? uint(c.hi1) : 0u, (g2 ? uint(c.lo2) : 0u) | (flags << 11), g2 ? uint(c.hi2) : 0u);
}

// (lo1, hi1, lo2, hi2) of packed extra gaps.
static inline float4 extraGaps(uint4 g) { return float4(g & GAP_BITS); }
static inline uint gapFlags(uint4 g) { return g.z >> 11; }

// Lifted land is re-terraced into cliff bands, so mountain faces read as contours, not a stair lattice. Karst keeps its
// sheer pillar walls.
static inline float megaTerrace(float h, float U, float wK, float td, uint seed) {
    float hm = h + U;
    float tdm = mix(td, 14.0, smoothstep(40.0, 400.0, U));
    // Beds differ in thickness from one formation to the next, so a tall wall is not one even ruled stripe.
    tdm *= mix(1.0, 0.6 + 0.8 * hashf(hash2i(int2(int(floor(hm / 70.0)), 7), seed + 260u)), smoothstep(60.0, 200.0, U));
    float hn3 = hm / tdm, lv3 = floor(hn3);
    // Tall relief gets short risers between wide treads: a long ramp of single steps reads as a stair lattice.
    float e3 = mix(0.6, 0.85, smoothstep(100.0, 400.0, U));
    return min(mix(hm, (lv3 + smoothstep(e3, 1.0, hn3 - lv3)) * tdm, smoothstep(0.5, 30.0, U) * (1.0 - wK)), TERRAIN_CEIL);
}

static inline Col terrainColumnBase(int2 w, uint seed) {
    float2 p = float2(w) + seedOffset(seed);
    float2 warp = float2(fbm2(p / 900.0, 2, seed + 1), fbm2(p / 900.0 + 37.1, 2, seed + 2)) * 300.0;
    float c = saturate(fbm2((p + warp) / 2000.0, 3, seed + 3) * 1.4 + 0.5);
    float hi = smoothstep(0.35, 0.75, c);
    float rough = fbm2((p + warp * 0.3) / 350.0, 3, seed + 4) * 0.5 + 0.5;
    float ridge = ridged2((p + warp * 0.2) / 520.0, 3, seed + 5);
    float r = mix(rough, ridge, hi);
    float h = mix(16.0, 62.0, c) + mix(26.0, 115.0, hi) * r;
    // Crenellated ridgelines instead of soft domes.
    h += (ridged2((p + warp * 0.4) / 110.0, 3, seed + 14) - 0.45) * mix(0.0, 8.0, smoothstep(0.3, 0.7, hi));
    Climate cl = climate(float2(w), seed);
    // Island country lies low: hills are pressed down toward meadow, so islands ride high over it with room beneath.
    if (cl.isle > 0.0) {
        float L = 62.0 + 20.0 * fbm2(p / 650.0, 2, seed + 303u);
        h = mix(h, h > L ? L + (h - L) * 0.15 : h, cl.isle);
    }
    // Soft terracing everywhere: risers cluster into short cliff bands between wide treads,
    // so slopes read as contour lines instead of a 1:1 stair lattice.
    float td = mix(4.0, 8.0, saturate(perlin2(p / 900.0, seed + 13) * 0.8 + 0.5));
    float hn2 = h / td;
    float lv = floor(hn2);
    h = (lv + smoothstep(mix(0.55, 0.9, hi), 1.0, hn2 - lv)) * td;

    // Canyon country: in hot, dry regions the land steps into flat-topped mesas with sheer risers.
    float salt = saltAmount(cl);
    float canyon = smoothstep(0.35, 0.7, desertAmount(cl)) * (1.0 - salt);
    float tier = 13.0 + 4.0 * perlin2(p / 700.0, seed + 7);
    float hn = h / tier;
    float level = floor(hn);
    float mesa = (level + smoothstep(0.62, 0.9, hn - level)) * tier;
    h = mix(h, mesa, canyon);

    // Painted hills: rounded mounds from billowed Perlin noise, untreaded so every voxel step shows a bed,
    // rising from flat floors where the billow's creases are filled in.
    float painted = paintedAmount(cl);
    // Low islands in lagoons, salt basins and painted country: a short-wavelength warp and roughening, or smooth
    // domes floor into concentric rings that look built.
    float lagoon = lagoonAmount(cl);
    float eroded = max(max(lagoon, smoothstep(0.0, 0.3, salt)), smoothstep(0.0, 0.6, painted));
    float2 ew = eroded > 0.0 ? float2(perlin2(p / 38.0, seed + 120u), perlin2(p / 38.0 + 19.1, seed + 121u)) * 14.0 * eroded : 0.0;
    if (painted > 0.0) {
        float b = abs(fbm2((p + warp * 0.3 + ew) / 160.0, 3, seed + 111u));
        float amp = mix(22.0, 52.0, saturate(perlin2(p / 600.0, seed + 112u) * 0.8 + 0.5));
        float mound = smoothstep(0.07, 0.5, b);
        h = mix(h, paintedFloor(p, seed) + amp * mound * (2.0 - mound) * 0.75, smoothstep(0.0, 0.6, painted));
    }

    // Long meandering valleys with flat water floors: the camera follows these.
    float river = abs(fbm2((p + warp * 0.6) / 1400.0, 3, seed + 6));
    float valley = smoothstep(0.02, 0.30 + 0.10 * hi, river);
    h = mix(SEA - 3.0, h, valley);
    // Lagoon shallows: the land sinks to a sandy bed rolling a few voxels either side of the surface,
    // so sand bars and islets break it and channels run deeper between them.
    if (lagoon > 0.0) {
        float bed = fbm2((p + warp * 0.3 + ew) / 210.0, 3, seed + 15u) + 0.25 * perlin2(p / 600.0, seed + 16u);
        h = mix(h, SEA - 2.5 + bed * 14.0, lagoon);
    }
    Mega mg = megaLift(p, lagoon, seed);
    float U = mg.U;
    float h0 = h;
    if (U > 0.5) h = megaTerrace(h0, U, mg.wK, td, seed);
    // The cap rock's height over a lip column: the plateau behind it by the same formula, so the lip meets it level.
    float hCap = mg.lipU > U + 6.0 ? floor(megaTerrace(h0, mg.lipU, mg.wK, td, seed)) : 0.0;
    // A pan's rim would otherwise chop a massif flat.
    salt *= 1.0 - smoothstep(10.0, 80.0, U);
    if (eroded > 0.0) h += eroded * (1.6 * perlin2(p / 21.0 + ew / 14.0, seed + 122u) + 0.9 * perlin2(p / 9.0, seed + 125u)) * smoothstep(SEA + 40.0, SEA + 2.0, h);
    h = floor(max(h, 2.0));

    // Salt pans: the land is cut down to old shorelines, low benches climbing out of a dead-flat floor,
    // and rivers that reach the pan end at its edge.
    bool pool = false;
    if (salt > 0.02) {
        float bench = 3.0;
        float rs = salt + 0.05 * perlin2(p / 45.0, seed + 123u) + 0.025 * perlin2(p / 16.0, seed + 124u);
        float rim = PAN + floor(70.0 * pow(1.0 - smoothstep(0.1, 0.85, rs), 1.5) / bench) * bench;
        h = min(h, rim);
        h = max(h, PAN - floor(6.0 * (1.0 - smoothstep(0.6, 0.85, salt))));
        if (h == PAN && poolNoise(p, seed) > poolLevel(salt)) { h = PAN - 1.0; pool = true; }
    }
    // Lake beds are one flat shelf: submerged terraces read as stepping stones under the surface.
    if (h < SEA && !pool) h = floor(mix(floor(SEA) - 2.0, h, lagoon) + 0.5);

    // Arches and undercuts: noise blobs hollow out a band low in each mesa tier.
    float gl = 0, gh = 0;
    if (canyon > 0.2 && U <= 0.5) {
        float blob = fbm2(p / 34.0, 3, seed + 8);
        float g = (blob - 0.2) * canyon;
        if (g > 0.0) {
            float base = max(level * tier - tier, SEA + 1.0);
            gl = floor(base + 2.0 + 2.0 * perlin2(p / 400.0, seed + 10));
            gh = gl + floor(min(g * 40.0, tier * 0.8));
            if (gh > h - 3.0 || gh - gl < 3.0 || gl < SEA + 1.0) { gl = 0; gh = 0; }
        }
    }
    Col col;
    col.lo1 = 0; col.hi1 = 0; col.lo2 = 0; col.hi2 = 0;
    col.carved = false;
    col.island = false;
    // A caprock lip: the column rises to the cap, hollow beneath it down to its own ground. The underside is the cap
    // rock's bedding plane, level: a sloping one would be an upturned staircase, every step a lit line. The wall below
    // runs up under it into a rock shelter.
    if (hCap > h + 6.0) {
        float t0 = 12.0 + 10.0 * saturate(perlin2(p / 400.0, seed + 263u) + 0.5);
        float ceil0 = floor(hCap - t0);
        // The shelter is a recess of bounded height under the slab: below it the band is the escarpment's sheer wall.
        float V = floor(18.0 + 40.0 * saturate(perlin2(p / 300.0, seed + 264u) * 0.9 + 0.5));
        float floor0 = max(h, ceil0 - V);
        if (ceil0 - floor0 >= 3.0) { gl = floor0; gh = ceil0; col.carved = true; }
        h = hCap;
    }
    // A notch round a karst tower's foot: at the water line for a tower standing in water, else on its talus.
    if (mg.notchH >= 3.0 && gh <= gl) {
        float lo = h0 < SEA ? floor(SEA) - 1.0 : floor(h0 + mg.notchFoot) + 1.0;
        float hi = lo + floor(mg.notchH);
        if (hi < h - 8.0 && hi - lo >= 3.0) { gl = lo; gh = hi; col.carved = true; }
    }
    col.h = h; col.gl = gl; col.gh = gh;
    return col;
}

// MARK: - Ruined towers

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
// Function constant 1: false compiles the towers out of sceneFragment, for frames with none in view. Compiled in, even
// unused, they slow every pixel's shading.
constant bool sceneTowers [[function_constant(1)]];

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
    float2 cw = float2(fbm2(p / 7000.0, 2, seed + 200u), fbm2(p / 7000.0 + 71.3, 2, seed + 201u)) * 3000.0;
    return fbm2((p + cw) / 16000.0, 3, seed + 202u) * 1.6 + 0.5;
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
            if (length(p - c) < 1500.0 + 1300.0 + 520.0 + d) return true;
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

// The tower's share of column wf: its plinth, stages, shell and openings over the terrain column col. s: the column
// width this is sampled for (coarse level k: 2^k); from 4 on, no hollows or openings, which the samples would alias.
static inline Col towerShape(Col col, Tower T, float2 wf, int s, uint seed) {
    float2 d0 = wf - T.c;
    TowerPlan P0 = towerPlan(T, d0);
    // Rubble of the fallen crown in a lobe toward th0: mounds only (loose blocks read as toys).
    float2 f0 = float2(cos(T.th0), sin(T.th0));
    float along = dot(d0, f0), across = abs(dot(d0, float2(-f0.y, f0.x)));
    float lobe = smoothstep(T.R0 * 0.9, T.R0 * 1.4, along) * smoothstep(T.R0 * 2.7, T.R0 * 1.7, along)
               * smoothstep(T.R0 * (0.8 + 0.003 * along), T.R0 * 0.3, across);
    if (lobe > 0.0) {
        float mnd = perlin2(wf / 23.0, seed + 510u) * 0.7 + perlin2(wf / 9.0, seed + 511u) * 0.3;
        col.h = max(col.h, floor(col.h + 16.0 * lobe * saturate(mnd + 0.4)));
    }
    // The plinth: two broad platforms, cut and filled level, the outer one broken up and gone in places.
    if (P0.reff >= T.R0 * 1.42) return col;
    float inner = T.R0 * (1.2 + 0.08 * perlin2(P0.dir * 3.7, seed + 520u));
    // Where the ground falls away there is no plinth: built out over it, it stood on a sheer pedestal of ground.
    float ground = col.h;
    if (ground > T.Hb - 12.0) col.h = P0.reff < inner ? T.Hb + 6.0 : perlin2(wf / 37.0, seed + 521u) > -0.15 ? T.Hb + 3.0 : col.h;
    col.gl = 0; col.gh = 0; col.lo1 = 0; col.hi1 = 0; col.lo2 = 0; col.hi2 = 0; col.carved = false;
    // The highest stage whose plan covers the column, and how far outside each stage's plan it lies.
    int m = -1;
    float Rm = 0.0;
    TowerPlan Pm = P0;
    float out[6] = { 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < 6; k++) {
        if (k >= T.K) break;
        TowerPlan Pk = towerPlan(T, wf - towerStageAxis(T, k));
        float Rk = tierR(T, Pk, k);
        out[k] = Pk.reff - Rk;
        if (Pk.reff < Rk) { m = k; Rm = Rk; Pm = Pk; }
    }
    if (m < 0) {
        float root = s < 4 ? towerRootH(T, P0, seed) : 0.0;
        float2 tal = towerTalus(T, P0, wf, s, seed);
        col.h = max(col.h + root, floor(mix(min(ground, T.Hb + 3.0), T.Hb + 3.0, tal.y) + tal.x));
        return col;
    }
    bool shell = m >= T.K - 2;
    float yS = tierY(T, Pm, T.K - 3, seed);                       // the shell stands on this
    float y0 = m == 0 ? T.Hb + 6.0 : shell ? yS : tierY(T, Pm, m - 1, seed);
    float H = shell ? crownY(T, Pm, yS, T.Hb + T.Ht, seed) : tierY(T, Pm, m, seed);
    float wallIn = Rm - Pm.reff;
    // Whole sections of a stage's outer wall have fallen from the top, in broad bites, so no two stages share an outline;
    // each bite a few dozen courses at most (cut down a whole stage, a bite's sides ran down the face as crisp lines).
    float bite = 30.0 * saturate(perlin2(Pm.dir * 2.6 + float(m) * 2.3, seed + 506u) * 1.7 - 0.2) * (m == 0 ? 0.6 : 1.0);
    if (!shell && bite > 4.0 && wallIn < bite * (0.7 + 0.5 * perlin2(wf / 9.0, seed + 533u)))
        H = max(y0 + 3.0, H - floor(min((H - y0) * (0.15 + 0.01 * bite), 25.0 + 2.0 * bite) / 5.0) * 5.0);
    // Chunks bitten out of the corners, part of a stage's height, their edges ragged. (Stepped down toward the corner,
    // they drew more setbacks.)
    float seg = 6.2831853 / float(T.N);
    uint ch = hash3i(int3((Pm.face + (Pm.am > 0.0 ? 1 : 0)) % T.N, m, int(T.h & 1023u)), seed + 515u);
    if (hashf(ch) < 0.65) {
        float hf = Rm * tan(0.5 * seg);
        float drop = min((H - y0) * (0.25 + 0.5 * hashf(pcg(ch + 1u))), 20.0 + 50.0 * hashf(pcg(ch + 2u)));
        float dc = hf - abs(Pm.t) + (s < 8 ? 7.0 * perlin2(wf / 7.0, seed + 523u) : 0.0) + 0.3 * drop * perlin2(wf / 13.0, seed + 530u);
        if (dc < hf * (0.25 + 0.45 * hashf(pcg(ch))))
            H = max(y0, H - floor(drop / 5.0) * 5.0);
    }
    // The parapet along each rim broken down a course or three, in runs of a few columns.
    if (!shell && wallIn < 8.0 + 0.05 * T.R0)
        H = max(y0, H - 5.0 * floor(3.5 * saturate(perlin2(wf / 11.0, seed + 531u) + (s < 4 ? 0.6 * perlin2(wf / 4.0, seed + 532u) : 0.0) + 0.15)));
    // A great collapse on most towers: a wedge of the upper storeys fallen on the crown's side, its floor broken in
    // courses, the outer skin gone deeper than the core.
    uint hc = pcg(T.h + 30u);
    if (hashf(hc) < 0.75) {
        float thc = T.th0 + 0.9 * (hashf(pcg(hc)) - 0.5);
        float da = P0.a - thc;
        da -= 6.2831853 * floor(da / 6.2831853 + 0.5);
        float u = abs(da) / (0.3 + 0.4 * hashf(pcg(hc + 1u))) + 0.25 * perlin2(wf / 23.0, seed + 525u);
        if (u < 1.0) {
            float yF = T.Hb + T.Ht * (0.3 + 0.35 * hashf(pcg(hc + 2u)));
            float yc = yF + (T.Hb + T.Ht - yF) * u * u + 1.2 * wallIn;
            uint bh = hash2i(int2(floor(wf / 6.0)), seed + 526u);
            yc = floor(yc / 5.0) * 5.0 + 5.0 * floor(3.0 * hashf(bh) - 1.0);
            H = min(H, max(yc, T.Hb + 6.0));
        }
    }
    // Weathering: the top crumbles back course by course where the worn skin has gone, a few courses at most: run down a
    // whole stage, the worn skin cut grooves down the face, a crisp line each. (Pockets cut in the face read as dark
    // bricks or windows punched in it, so none are.)
    if (wallIn < 16.0 && s < 8) {
        float yl = max(y0, H - 25.0);
        for (float yc = y0 + 5.0 * (ceil((H - y0) / 5.0) - 1.0); yc > yl; yc -= 5.0) {
            if (wallIn >= towerErosion(T, Pm, Rm, yc + 2.5, seed)) break;
            H = yc;
        }
    }
    // Rubble heaped in mounds on the tread at the foot of the stage above.
    if (!shell && H > y0 + 3.0) {
        float L = max(8.0, 0.45 * T.R0 * (towerTierScale(T, m) - towerTierScale(T, m + 1)));
        float f = saturate(1.0 - out[m + 1] / L);
        if (f > 0.0)
            H += floor((3.0 + 0.1 * T.R0) * f * sqrt(f) * saturate(0.35 + perlin2(wf / 19.0, seed + 527u) + (s < 8 ? 0.4 * perlin2(wf / 7.0, seed + 528u) : 0.0)));
    }
    float gaps[6] = { 0, 0, 0, 0, 0, 0 };
    int ng = 0;
    // A stage fallen away beneath the one above (the wound): a sector of stage kb's outer mass is gone, the stages over
    // it overhang.
    if (T.kb <= m && T.kb < T.K - 2) {
        TowerPlan Pb = towerPlan(T, wf - towerStageAxis(T, T.kb));
        float Rk = tierR(T, Pb, T.kb);
        float lobeW = 0.5 + 0.5 * cos(Pb.a - T.th1);
        float depth = T.R0 * 0.35 * smoothstep(0.82, 0.97, lobeW) * saturate(0.7 + perlin2(Pb.dir * 8.0, seed + 513u));
        if (Pb.reff > Rk - depth) {
            float yb0 = tierY(T, Pb, T.kb - 1, seed) + 8.0 + floor(10.0 * abs(perlin2(Pb.dir * 6.0, seed + 514u)));
            float yb1 = tierY(T, Pb, T.kb, seed) - 4.0;
            if (m == T.kb) H = min(H, yb0);
            else if (s < 4 && yb1 - yb0 > 6.0) { gaps[0] = yb0; gaps[1] = yb1; ng = 1; }
        }
    }
    if (shell && Pm.reff < Rm - max(10.0, 1.5 * float(s))) {
        // The shell is open to the sky, its floor a court where a grove can root. Every ring keeps the court: filled, its
        // top would follow the crown's broken rim and show a radial pattern from above. The wall thickens with the
        // samples' spacing, or whole runs of the rim would fall between them and the crown lose its silhouette.
        H = min(H, yS + 3.0);
    }
    if (s < 4) {
        // A passage through the base under arches, in some towers.
        if ((T.h & 128u) != 0u && ng < 3) {
            float2 pd = float2(cos(T.th1 + 1.5708), sin(T.th1 + 1.5708));
            float off = abs(d0.x * pd.y - d0.y * pd.x);
            if (off < 7.0 && P0.reff < T.R0) {
                float hi = T.Hb + 36.0 + floor(0.2 * T.R0) - floor(6.0 * (off / 7.0) * (off / 7.0));
                gaps[2 * ng] = T.Hb + 6.0; gaps[2 * ng + 1] = min(hi, H - 4.0); ng++;
            }
        }
        // Great portals in the base's faces.
        if (m == 0 && P0.face % (T.N > 8 ? 3 : 2) == 0 && abs(Pm.t) < 7.0 && wallIn < 14.0 && ng < 3) {
            float hi = T.Hb + 40.0 - floor(4.0 * (Pm.t / 7.0) * (Pm.t / 7.0));
            gaps[2 * ng] = T.Hb + 6.0; gaps[2 * ng + 1] = min(hi, H - 4.0); ng++;
        }
        bool through = shell && Pm.reff >= Rm - 10.0;
        float bwF = towerBayW(T.h, m, Pm.face, T.bw);
        float bay = floor(Pm.t / bwF + 0.5);
        float dt = abs(Pm.t - bay * bwF);
        int ns = int((H - y0) / T.Sy);
        // A loggia under each stage's rim: a storey of deep arches round the stage, wherever its top still stands.
        if (!shell && wallIn < 6.0 && dt < 0.32 * bwF && ng < 3 && towerLoggiaBay(T.h, m, Pm.face, bay)) {
            float top = tierY(T, Pm, m, seed);
            float hi = top - 4.0 - (dt > 0.25 * bwF ? 2.0 : dt > 0.18 * bwF ? 1.0 : 0.0);
            float lo = top - 4.0 - floor(T.Sy * 1.1);
            if (hi <= H - 3.0 && lo > y0 + 4.0) {
                if (towerBayFallen(T.h, m, Pm.face, bay)) H = lo;
                else { gaps[2 * ng] = lo; gaps[2 * ng + 1] = hi; ng++; }
            }
        }
        // Arcades through the shell's wall: on each face a few of every other storey, bays missing from them. (Single
        // windows in the stage walls read as slots punched in a slab, a grid of them as a block of flats.)
        if (through && dt < 3.0) {
            for (int st = 40; st >= 0; st--) {
                if (ng >= 3) break;
                if (st >= ns || (st & 1) != 0 || hashf(hash3i(int3(Pm.face, st, 9), T.h + 125u)) > 0.4
                    || hashf(hash3i(int3(Pm.face, int(bay), st + 16), T.h + 121u)) < 0.3) continue;
                float lo = y0 + float(st) * T.Sy + 3.0;
                float hi = lo + floor(T.Sy * 0.75) - (dt > 2.0 ? 2.0 : dt > 1.0 ? 1.0 : 0.0);
                if (hi > H - 3.0) continue;
                gaps[2 * ng] = lo; gaps[2 * ng + 1] = hi; ng++;
            }
        }
    }
    col.h = min(max(col.h, floor(H)), TERRAIN_CEIL);
    // Highest gap first.
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++)
            if (gaps[2 * j + 1] > gaps[2 * i + 1]) {
                float a0 = gaps[2 * i], a1 = gaps[2 * i + 1];
                gaps[2 * i] = gaps[2 * j]; gaps[2 * i + 1] = gaps[2 * j + 1];
                gaps[2 * j] = a0; gaps[2 * j + 1] = a1;
            }
    col.gl = floor(gaps[0]); col.gh = floor(gaps[1]);
    col.lo1 = floor(gaps[2]); col.hi1 = floor(gaps[3]);
    col.lo2 = floor(gaps[4]); col.hi2 = floor(gaps[5]);
    col.carved = col.gh > col.gl;
    return col;
}

static inline Col towerColumn(Col col, int2 w, uint seed, int s) {
    float2 wf = float2(w) + 0.5;
    Tower T;
    TowerGroup g;
    if (!towerFind(wf, 0.0, seed, T, g)) return col;
    // Most of a member's reach is open ground, off its plinth and its rubble: those columns are let go before asking
    // whether the group stands, which costs two terrain columns and the province fields.
    float2 d0 = wf - T.c, f0 = float2(cos(T.th0), sin(T.th0));
    float along = dot(d0, f0), across = abs(dot(d0, float2(-f0.y, f0.x)));
    bool rubble = along > T.R0 * 0.9 && along < T.R0 * 2.7 && across < T.R0 * (0.8 + 0.003 * along);
    if (!rubble && towerPlan(T, d0).reff >= T.R0 * 1.42) return col;
    if (!towerGroupStands(g, seed)) return col;
    T.Hb = towerBase(T, seed);
    if (T.Hb > TOWER_BASE_MAX) return col;
    return towerShape(col, T, wf, s, seed);
}

// Within margin of a standing tower's reach (sites keep off them).
static inline bool towerNear(float2 wf, float margin, uint seed) {
    Tower T;
    TowerGroup g;
    return towerFind(wf, margin, seed, T, g) && towerGroupStands(g, seed) && towerBase(T, seed) <= TOWER_BASE_MAX;
}

// How wet a tower's country is, for its greenery.
static inline float towerWet(Climate c) {
    return smoothstep(0.25, 0.65, c.moist) * (1.0 - smoothstep(0.35, 0.7, desertAmount(c)));
}

// How far the ivy hangs below a rim at height rim, at most room, at world xz w on a face of the tower whose axis is c
// (plan rotation rot, N sides, hash th): mats several columns wide with bare stone between, their lower edges drifting
// smoothly along the face, a whole column at a time on the 1-voxel map (q quantised by the caller).
static inline float towerIvyDrop(float2 w, float2 c, float rot, int N, uint th, float rim, float room, float wet, uint seed) {
    float2 d = w - c;
    float seg = 6.2831853 / float(N);
    float a = atan2(d.y, d.x) - rot;
    float fi = floor(a / seg);
    float t = length(d) * sin(a - seg * fi - 0.5 * seg) + 97.0 * fi;
    float2 q = float2(t, rim * 0.4 + float(th & 255u) * 7.3);
    float mat = perlin2(q / 34.0, seed + 560u) + 0.35 * perlin2(q / 14.0, seed + 561u);
    return min(room, 20.0 + 90.0 * wet) * smoothstep(-0.25, 0.2, mat + 0.4 * wet - 0.1) * (0.75 + 0.25 * perlin2(q / 9.0, seed + 562u));
}

// MARK: - Floating islands

// Rock and earth islands over island country (Climate.isle). An island is a landform: terrainColumn raises the column to
// the island's top over a gap from its ground to the underside (flagged GAP_ISLAND), so the 1-voxel map, every ring, the
// max mips, the horizon, the shadow maps and the CPU copy all carry it. Islands never overlap in plan, since a coarse
// column holds one gap. The table (placeIslands) holds a candidate per ISLE_CELL cell and a massif per ISLE_MCELL cell, in
// toroidal windows round the map's middle wider than the coarsest ring's window plus a massif's reach.
constant float ISLE_CELL = 640.0;
constant float ISLE_MCELL = 3200.0;
// Must match Islands.win and Islands.mwin in Sources/Islands.swift.
constant int ISLE_WIN = 60;
constant int ISLE_MWIN = 13;
constant int ISLE_SLOTS = ISLE_WIN * ISLE_WIN + ISLE_MWIN * ISLE_MWIN;

// c: centre (world xz); R: radius (> 0 standing, < 0 dropped for a neighbour it would overlap, 0 none); A: the top at
// the rim's line; D: the keel's depth; B: the highest ground under it; K: a bound under its lowest underside; h: hash.
// Must match IsleRec in Sources/Islands.swift.
struct IsleRec { int2 cell; float2 c; float R; float A; float D; float B; float K; uint h; float2 pad; };

static inline int isleSlot(int2 cell, bool massif) {
    int W = massif ? ISLE_MWIN : ISLE_WIN;
    int2 m = ((cell % W) + W) % W;
    return (massif ? ISLE_WIN * ISLE_WIN : 0) + m.y * W + m.x;
}

// Shape choices by hash: one in three of the smaller islands (and of the massifs) drawn out along an axis.
static inline float isleStretch(float R, uint h) { return (R > 40.0 && (R <= 210.0 || R >= 420.0) && hashf(pcg(h + 10u)) < 0.33) ? 1.5 : 1.0; }

// Nothing of an island lies further than this from its centre.
static inline float isleReach(float R, uint h) { return isleStretch(R, h) * (1.45 * R + 8.0); }

// (underside, top) of island r over world column centre p; empty outside it. s: the sample spacing (roots hang on the
// 1-voxel map and level 1, whose columns their clumps match, so they go only at level 1's far handoff).
static inline float2 isleSpan(IsleRec r, float2 p, uint seed, int s) {
    float R = r.R;
    uint h = r.h;
    float2 q = p - r.c;
    float st = isleStretch(R, h);
    float th0 = hashf(pcg(h + 11u)) * 3.14159265;
    float2 ax = float2(cos(th0), sin(th0));
    float2 ql = st > 1.0 ? float2(dot(q, ax) / st, dot(q, float2(-ax.y, ax.x)) / 0.75) : q;
    float rr = length(ql);
    if (rr > 1.45 * R + 8.0) return float2(0);
    float o = float(h & 1023u) * 3.7;
    float2 pq = ql / R;
    float e = R * (1.0 + 0.28 * perlin2(pq * 1.6 + o, seed + 701u) + 0.12 * perlin2(pq * 4.0 + o, seed + 702u))
            + 5.0 * perlin2(p / 11.0, seed + 703u);
    // A bay bitten into half of them.
    if (hashf(pcg(h + 12u)) < 0.5) e *= 1.0 - 0.32 * smoothstep(0.2, 0.55, perlin2(ql / max(rr, 1.0) * 1.3 + o + 7.1, seed + 708u));
    float d = rr / max(e, 4.0);
    if (d >= 1.0) return float2(0);
    // A low dome of meadow, rolling on the broad ones, terraced as the lowland is: wide treads, short risers.
    float top = r.A + 0.06 * R * (1.0 - d * d) + 1.2 * perlin2(p / 37.0, seed + 704u);
    if (R > 120.0) top += 7.0 * fbm2(p / 90.0, 2, seed + 709u) * smoothstep(120.0, 250.0, R) * (1.0 - d * d);
    float tl = floor(top / 5.0);
    top = (tl + smoothstep(0.6, 0.95, top / 5.0 - tl)) * 5.0 - 3.0 * smoothstep(0.85, 1.0, d);
    // A rim of rock teeth on some.
    if (hashf(pcg(h + 13u)) < 0.3) top += 3.0 * ridged2(p / 7.0, 2, seed + 710u) * smoothstep(0.88, 0.97, d);
    top = floor(top + 0.5);
    float t = pow(1.0 - d, 0.55);
    float rag = 0.6 + 0.4 * ridged2(p / (0.3 * R + 8.0), 2, seed + 705u);
    float th = 5.0 + r.D * t * rag + 6.0 * t * perlin2(p / 6.0, seed + 706u);
    // Two or three drips: spires of rock hanging under a big keel.
    if (R > 150.0) {
        int n = 2 + int(pcg(h + 14u) % 2u);
        for (int i = 0; i < n; i++) {
            uint hi = pcg(h + 20u + uint(i));
            float a = hashf(hi) * 6.2831853, rd = R * (0.15 + 0.35 * hashf(pcg(hi)));
            float2 sc = r.c + float2(cos(a), sin(a)) * rd;
            float f = max(0.0, 1.0 - length(p - sc) / (18.0 + 10.0 * hashf(pcg(hi + 1u))));
            th += r.D * 0.3 * f * f;
        }
    }
    // The keel steps in beds of seven voxels, each bed's tread facing down: dark seams, not lit lines.
    float hb = th / 7.0;
    th = (floor(hb) + smoothstep(0.6, 1.0, fract(hb))) * 7.0;
    float bot = floor(top - max(th, 4.0));
    if (s <= 2) {
        // Roots and vines hang in clumps under the outer keel, two columns wide.
        uint rh = hash2i(int2(floor(p / 2.0)), seed + 707u);
        float clump = smoothstep(0.05, 0.35, perlin2(p / 19.0, seed + 711u)) * smoothstep(0.45, 0.9, d);
        if (hashf(rh) < 0.07 * clump) bot -= floor(4.0 + 26.0 * pow(hashf(pcg(rh)), 2.0));
    }
    return float2(bot, top);
}

// The span of the island over world column centre p, if any.
static inline bool isleAt(device const IsleRec* T, float2 p, uint seed, int s, thread float2& sp) {
    int2 c0 = int2(floor(p / ISLE_CELL));
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cell = c0 + int2(i, j);
            device const IsleRec& r = T[isleSlot(cell, false)];
            if (r.R <= 0.0 || any(r.cell != cell)) continue;
            float re = isleReach(r.R, r.h);
            if (length_squared(p - r.c) > re * re) continue;
            sp = isleSpan(r, p, seed, s);
            if (sp.y > sp.x) return true;
        }
    int2 m0 = int2(floor(p / ISLE_MCELL - 0.5));
    for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++) {
            int2 cell = m0 + int2(i, j);
            device const IsleRec& r = T[isleSlot(cell, true)];
            if (r.R <= 0.0 || any(r.cell != cell)) continue;
            float re = isleReach(r.R, r.h);
            if (length_squared(p - r.c) > re * re) continue;
            sp = isleSpan(r, p, seed, s);
            if (sp.y > sp.x) return true;
        }
    return false;
}

static inline Col islandColumn(Col col, int2 w, uint seed, int s, device const IsleRec* T) {
    float2 sp;
    if (!isleAt(T, float2(w) + 0.5, seed, s, sp)) return col;
    float G = col.h;
    // Sunk in a hill: nothing; resting on one: solid to its top.
    if (sp.y <= G + 3.0) return col;
    if (sp.x <= G + 3.0) { col.h = sp.y; return col; }
    if (col.gh > col.gl) { col.lo2 = col.lo1; col.hi2 = col.hi1; col.lo1 = col.gl; col.hi1 = col.gh; }
    col.gl = G; col.gh = sp.x; col.h = sp.y;
    col.carved = true;
    col.island = true;
    return col;
}

// The shading table (fragment buffer 10; towers hold 9, giants 8): the islands nearest the camera (Sources/Islands.swift).
// a: centre xz, reach, the underside's mean height; b: radius, top, keel.
constant int ISLE_TABLE = 16;
struct IsleShade { float4 a; float4 b; };
struct IsleTable { int4 n; IsleShade r[ISLE_TABLE]; };

static inline Col terrainColumnS(int2 w, uint seed, int s, device const IsleRec* T) {
    return islandColumn(towerColumn(terrainColumnBase(w, seed), w, seed, s), w, seed, s, T);
}
static inline Col terrainColumn(int2 w, uint seed, device const IsleRec* T) { return terrainColumnS(w, seed, 1, T); }
// The land without its islands (placing the giants, which the islands then keep clear of).
static inline Col terrainColumnBare(int2 w, uint seed) { return towerColumn(terrainColumnBase(w, seed), w, seed, 1); }

// The shading table (fragment buffer 9; stage 3's giants hold 8): the towers standing nearest the camera (Sources/Towers.swift fills it from placeTowers).
// a: axis xz, R0, Hb; b: Ht, K, N, rot; c: th0, th1, Sy, bw; d: hash, kind, lit at night, member index | moisture << 8 |
// temperature << 16 (bytes; for listings).
struct TowerRec { float4 a; float4 b; float4 c; uint4 d; };
struct TowerTable { int4 n; TowerRec r[TOWER_TABLE]; };

// The table record of the standing tower whose plinth holds world point w, or -1.
static inline int towerIndexAt(constant TowerTable* tt, float2 w) {
    int n = min(tt->n.x, TOWER_TABLE);
    for (int i = 0; i < n; i++) {
        float4 a = tt->r[i].a;
        float2 d = w - a.xy;
        float R = a.z * 1.45;
        if (dot(d, d) < R * R) return i;
    }
    return -1;
}


// One thread per world cell info.xy + gid of an info.z-square grid: its group's members, TOWER_MEMBERS records a cell
// (Ht 0: none).
kernel void placeTowers(device TowerRec* out [[buffer(0)]],
                        constant int4& info [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    uint seed = uint(info.w);
    uint base = (gid.y * uint(info.z) + gid.x) * uint(TOWER_MEMBERS);
    TowerRec none;
    none.a = 0; none.b = 0; none.c = 0; none.d = 0;
    for (int m = 0; m < TOWER_MEMBERS; m++) out[base + uint(m)] = none;
    TowerGroup g;
    if (!towerGroupIn(info.xy + int2(gid), seed, g) || !towerGroupStands(g, seed)) return;
    for (int m = 0; m < TOWER_MEMBERS; m++) {
        if (m >= g.n) break;
        Tower T = towerMember(g, m);
        T.Hb = towerBase(T, seed);
        if (T.Hb > TOWER_BASE_MAX) continue;
        TowerRec r;
        r.a = float4(T.c, T.R0, T.Hb);
        r.b = float4(T.Ht, float(T.K), float(T.N), T.rot);
        r.c = float4(T.th0, T.th1, T.Sy, T.bw);
        Climate cl = climate(T.c, seed);
        r.d = uint4(T.h, uint(T.kind), hashf(pcg(T.h + 60u)) < 0.35 ? 1u : 0u,
                    uint(m) | (uint(cl.moist * 255.0) << 8) | (uint(cl.temp * 255.0) << 16));
        out[base + uint(m)] = r;
    }
}


kernel void generateHeight(texture2d<float, access::write> out [[texture(0)]],
                           texture2d<uint, access::write> gxOut [[texture(1)]],
                           constant int4& info [[buffer(0)]],
                           constant uint& row0 [[buffer(1)]],
                           device const IsleRec* isles [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    int2 w = int2(gid) + info.xy;
    Col c = terrainColumn(w, uint(info.w), isles);
    out.write(float4(c.h, c.gl, c.gh, 0), gid);
    gxOut.write(packGaps(c), gid);
}

// Solid at height y in a column (H, gapLo, gapHi) with extra gaps e (lo1, hi1, lo2, hi2).
static inline bool solidIn(float3 c, float4 e, float y) {
    return y < c.x && !(y >= c.y && y < c.z) && !(y >= e.x && y < e.y) && !(y >= e.z && y < e.w);
}

// Clamps each column into its 4-neighbour height range: removes single-voxel spikes, pits and slots.
kernel void cleanupColumns(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           texture2d<uint, access::read> gxSrc [[texture(2)]],
                           texture2d<uint, access::write> gxDst [[texture(3)]],
                           constant uint& row0 [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 c = src.read(gid);
    uint4 gx = gxSrc.read(gid);
    uint2 a = uint2(max(int(gid.x) - 1, 0), gid.y), b = uint2(min(gid.x + 1, n - 1), gid.y);
    uint2 e = uint2(gid.x, max(int(gid.y) - 1, 0)), f = uint2(gid.x, min(gid.y + 1, n - 1));
    float4 c0 = src.read(a), c1 = src.read(b), c2 = src.read(e), c3 = src.read(f);
    float h0 = c0.r, h1 = c1.r, h2 = c2.r, h3 = c3.r;
    float r0 = c.r;
    c.r = clamp(c.r, min(min(h0, h1), min(h2, h3)), max(max(h0, h1), max(h2, h3)));
    // A column-wide slot in a wall (both neighbours along an axis higher) shows as a dash down the face: fill it.
    if (c.r > SEA + 2.0) c.r = max(c.r, max(min(h0, h1), min(h2, h3)));
    // A notch in an island's rim raised to its neighbours would be a pillar from the ground to the island's top.
    if (c.r > r0 + 8.0 && ((gapFlags(gxSrc.read(a)) | gapFlags(gxSrc.read(b)) | gapFlags(gxSrc.read(e)) | gapFlags(gxSrc.read(f))) & GAP_ISLAND) != 0u)
        c.r = r0;
    if ((gapFlags(gx) & GAP_CARVED) == 0u) {
        if (c.b > c.r - 3.0) { c.g = 0; c.b = 0; }
        // A roof needs walls: keep a gap only where every neighbour reaches above it.
        if (min(min(h0, h1), min(h2, h3)) < c.b + 1.0) { c.g = 0; c.b = 0; }
        dst.write(c, gid);
        gxDst.write(gx, gid);
        return;
    }
    // Carved rock: its roofs are slabs the carving keeps whole (a test for a supporting neighbour cut a pillar wherever
    // one ceiling stood a voxel above both its neighbours'); a gap a single column wide along either axis (rock on both
    // sides at its middle) reads as a dash and is filled, and every roof keeps three voxels of rock over it.
    float4 x0 = extraGaps(gxSrc.read(a)), x1 = extraGaps(gxSrc.read(b)), x2 = extraGaps(gxSrc.read(e)), x3 = extraGaps(gxSrc.read(f));
    float4 ex = extraGaps(gx);
    float3 gaps[3] = { float3(c.g, c.b, 0), float3(ex.x, ex.y, 0), float3(ex.z, ex.w, 0) };
    float top = c.r;
    for (int i = 0; i < 3; i++) {
        float lo = gaps[i].x, hi = gaps[i].y;
        if (hi <= lo) continue;
        float mid = floor((lo + hi) * 0.5);
        bool keep = hi <= top - 3.0 && hi - lo >= 3.0;
        if (i == 0 && (gapFlags(gx) & GAP_ISLAND) != 0u) { if (!keep) gaps[i] = float3(0); else top = lo; continue; }
        keep = keep && !(solidIn(c0.rgb, x0, mid) && solidIn(c1.rgb, x1, mid)) && !(solidIn(c2.rgb, x2, mid) && solidIn(c3.rgb, x3, mid));
        if (!keep) gaps[i] = float3(0);
        else top = lo;
    }
    // The highest surviving gap is the principal one.
    int pi = gaps[0].y > gaps[0].x ? 0 : gaps[1].y > gaps[1].x ? 1 : 2;
    Col o;
    o.h = c.r;
    o.gl = gaps[pi].x; o.gh = gaps[pi].y;
    o.lo1 = pi == 0 ? gaps[1].x : pi == 1 ? gaps[2].x : 0.0;
    o.hi1 = pi == 0 ? gaps[1].y : pi == 1 ? gaps[2].y : 0.0;
    o.lo2 = pi == 0 ? gaps[2].x : 0.0;
    o.hi2 = pi == 0 ? gaps[2].y : 0.0;
    if (o.hi1 <= o.lo1) { o.lo1 = o.lo2; o.hi1 = o.hi2; o.lo2 = 0; o.hi2 = 0; }
    o.carved = o.gh > o.gl;
    o.island = (gapFlags(gx) & GAP_ISLAND) != 0u && pi == 0;
    dst.write(float4(o.h, o.gl, o.gh, c.a), gid);
    gxDst.write(packGaps(o), gid);
}

// MARK: - Vegetation

// Trees sit on a jittered world-space grid, one candidate per cell; a canopy reaches at most 2 columns
// from its trunk, so a column only needs the trees of its own and the 8 neighbouring cells.
constant int TREE_CELL = 5;

// Snow above this height: cold country as ever, while warm provinces lift it far up their ranges (hot ones never
// snow), so massifs are rock and forest under a cap rather than white sheets.
static inline float snowlineAt(float temp) {
    return mix(160.0, 92.0, smoothstep(0.42, 0.18, temp)) + 1400.0 * smoothstep(0.42, 0.75, temp);
}

// Forest gives out well under a high snowline, leaving a band of bare rock.
static inline float treelineAt(float snowline) { return snowline - 150.0 * smoothstep(200.0, 700.0, snowline); }

// Air cools up a range: deciduous, then conifer, then bare rock and snow.
static inline float altitudeTemp(float temp, float H) { return temp - 0.22 * smoothstep(150.0, 1400.0, H); }

static inline float coniferAmount(Climate c) { return smoothstep(0.50, 0.32, c.temp) * (1.0 - 0.8 * autumnAmount(c)); }

// Whole groves of birch within the autumn woods.
static inline float birchAmount(float2 w, uint seed) {
    return smoothstep(0.22, 0.38, perlin2((w + seedOffset(seed)) / 110.0, seed + 95u));
}

// MARK: - Giant trees

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
constant uint GAP_GIANT = 2u;
// The band test in the DDA: compiled in only while a canopy band is on the map (Renderer picks the pipeline); off in
// every pipeline built without the constant.
constant bool BAND_FC [[function_constant(0)]];
constant bool BAND_SKIP = is_function_constant_defined(BAND_FC) && BAND_FC;
constant int GIANT_LIMBS = 12;
constant int GIANT_FLANGES = 9;
constant int GM_WOOD = 0, GM_LEAF = 1, GM_VINE = 2;

// One table entry; G <= 0: no giant in the cell. Must match GiantRec in Sources/Renderer.swift.
struct GiantRec { int2 cell; float2 c; float G; float yT; uint h; uint kind; };

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

// MARK: - Placing islands

// Whether a giant (the map's table) stands within d of world xz c.
static inline bool isleNearGiant(device const GiantRec* T, float2 c, float d) {
    int2 c0 = int2(floor((c - d) / GIANT_CELL)), c1 = int2(floor((c + d) / GIANT_CELL));
    for (int j = c0.y; j <= c1.y; j++)
        for (int i = c0.x; i <= c1.x; i++) {
            GiantRec r = T[giantSlot(int2(i, j))];
            if (r.G > 0.0 && all(r.cell == int2(i, j)) && length_squared(c - r.c) < d * d) return true;
        }
    return false;
}

// The highest ground under an island of the given reach round c: its centre and three rings.
static inline float isleGround(float2 c, float reach, uint seed) {
    float B = terrainColumnBase(int2(floor(c)), seed).h;
    for (int i = 0; i < 24; i++) {
        float rr = reach * (i < 8 ? 0.35 : i < 16 ? 0.7 : 1.0);
        float an = float(i) * 0.785398 + (i < 8 ? 0.0 : i < 16 ? 0.39 : 0.2);
        B = max(B, terrainColumnBase(int2(floor(c + float2(cos(an), sin(an)) * rr)), seed).h);
    }
    return B;
}

// The test field (VR_ISLE_FIELD): islands in a disc round (-3500, -2500) whatever the country, one massif among them.
constant float2 ISLE_FIELD_C = float2(-3500.0, -2500.0);
constant float ISLE_FIELD_R = 5200.0;

// One thread per slot: the candidate of its cell in the window round the map's middle (info.xy, world), unless the slot
// holds that cell already. info.z bit 0: the test field; bit 1: none (island country without its islands). A candidate stands where island country is deep enough at its
// centre (a massif deeper still), clear of the mega relief's provinces, the tepuis, the towers and the giants, over
// ground low across its whole reach. Big islands ride low and small ones high; a big keel clears the ground by 100, a
// small one by 60.
kernel void placeIslands(device IsleRec* out [[buffer(0)]],
                         constant int4& info [[buffer(1)]],
                         device const GiantRec* giants [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(ISLE_SLOTS)) return;
    uint seed = uint(info.w);
    bool massif = gid >= uint(ISLE_WIN * ISLE_WIN);
    int W = massif ? ISLE_MWIN : ISLE_WIN;
    int i = int(gid) - (massif ? ISLE_WIN * ISLE_WIN : 0);
    float cw = massif ? ISLE_MCELL : ISLE_CELL;
    int2 cell = int2(floor(float2(info.xy) / cw)) - W / 2 + int2(i % W, i / W);
    int slot = isleSlot(cell, massif);
    if (all(out[slot].cell == cell)) return;
    IsleRec r;
    r.cell = cell; r.c = 0; r.R = 0; r.A = 0; r.D = 0; r.B = 0; r.K = 0; r.h = 0; r.pad = 0;
    if ((info.z & 2) != 0) { out[slot] = r; return; }
    if ((info.z & 1) != 0) {
        if (massif) {
            float2 bc = ISLE_FIELD_C + float2(600.0, 900.0);
            if (all(int2(floor(bc / ISLE_MCELL)) == cell)) {
                r.c = bc; r.R = 560.0; r.A = 520.0; r.D = 380.0; r.h = 0x5bd1e995u;
            }
        } else {
            uint h = hash2i(cell, seed + 700u);
            float2 c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h)), hashf(pcg(h + 1u)))) * ISLE_CELL;
            if (hashf(pcg(h + 2u)) <= 0.55 && length(c - ISLE_FIELD_C) <= ISLE_FIELD_R) {
                float u = hashf(pcg(h + 3u));
                r.c = c; r.h = h;
                r.R = 18.0 + 300.0 * u * u * u;
                r.A = floor(190.0 + 420.0 * hashf(pcg(h + 4u)));
                r.D = min(r.R * mix(0.7, 1.4, hashf(pcg(h + 5u))) + 10.0, r.A - 120.0);
            }
        }
        r.K = r.A + 0.06 * r.R - 8.5 - (r.R > 150.0 ? 1.3 : 1.0) * r.D - 18.0;
        out[slot] = r;
        return;
    }
    uint h = hash2i(cell, seed + (massif ? 721u : 711u));
    float2 c = floor((float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h)), hashf(pcg(h + 1u)))) * cw) + 0.5;
    Climate cl = climate(c, seed);
    float chance = massif ? 0.4 * smoothstep(0.3, 0.7, cl.isle) : 0.7 * smoothstep(0.15, 0.5, cl.isle);
    bool ok = hashf(pcg(h + 2u)) < chance && desertAmount(cl) < 0.1 && saltAmount(cl) < 0.1;
    float u = hashf(pcg(h + 3u));
    float R = massif ? 420.0 + 230.0 * u : 18.0 + 300.0 * u * u * u;
    float reach = isleReach(R, h);
    float2 p = c + seedOffset(seed);
    for (int k = 0; k < 4 && ok; k++) {
        float a = float(k) * 1.5707963 + 0.785398;
        ok = towerProvinceField(p + float2(cos(a), sin(a)) * 2500.0, seed) <= 0.5;
    }
    ok = ok && !towerNearTepui(p, reach + 600.0, seed) && !towerNear(c, reach + 300.0, seed)
            && !isleNearGiant(giants, c, reach + 300.0 + GIANT_REACH);
    float B = ok ? max(isleGround(c, reach, seed), SEA + 1.0) : 0.0;
    ok = ok && B <= (massif ? 130.0 : 150.0);
    if (ok) {
        float v = hashf(pcg(h + 4u));
        float rise = massif ? 260.0 + 160.0 * v : R > 150.0 ? 230.0 + 170.0 * v : R > 60.0 ? 260.0 + 240.0 * v : 320.0 + 330.0 * v;
        float drip = R > 150.0 ? 1.3 : 1.0;
        float A = floor(B + rise);
        float topC = A + 0.06 * R - 8.5;
        float D = min(R * mix(0.7, 1.4, hashf(pcg(h + 5u))) + 10.0, (topC - 18.0 - B - (R > 150.0 ? 100.0 : 60.0)) / drip);
        r.c = c; r.R = R; r.A = A; r.D = D; r.B = B; r.h = h;
        r.K = topC - drip * D - 18.0;
    }
    out[slot] = r;
}

// Drops every candidate that would overlap a stronger one in plan (a massif, else the larger), judged against the raw
// candidates round it, so no chain of drops depends on the order slots are visited. Reads |R| alone, which it never
// changes.
kernel void resolveIslands(device IsleRec* T [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(ISLE_SLOTS)) return;
    IsleRec r = T[gid];
    float R = abs(r.R);
    if (R <= 0.0) return;
    bool massif = gid >= uint(ISLE_WIN * ISLE_WIN);
    float reach = isleReach(R, r.h);
    float pr = (massif ? 1e5 : 0.0) + R + 0.001 * hashf(r.h);
    bool drop = false;
    int2 n0 = int2(floor(r.c / ISLE_CELL));
    for (int j = -2; j <= 2 && !drop && !massif; j++)
        for (int i = -2; i <= 2 && !drop; i++) {
            int2 cell = n0 + int2(i, j);
            int sl = isleSlot(cell, false);
            if (sl == int(gid)) continue;
            IsleRec q = T[sl];
            float Rq = abs(q.R);
            if (Rq <= 0.0 || any(q.cell != cell)) continue;
            float pq = Rq + 0.001 * hashf(q.h);
            if (pq < pr || (pq == pr && sl > int(gid))) continue;
            drop = length(q.c - r.c) < reach + isleReach(Rq, q.h) + 30.0;
        }
    int2 m0 = int2(floor(r.c / ISLE_MCELL));
    for (int j = -1; j <= 1 && !drop; j++)
        for (int i = -1; i <= 1 && !drop; i++) {
            int2 cell = m0 + int2(i, j);
            int sl = isleSlot(cell, true);
            if (sl == int(gid)) continue;
            IsleRec q = T[sl];
            float Rq = abs(q.R);
            if (Rq <= 0.0 || any(q.cell != cell)) continue;
            float pq = 1e5 + Rq + 0.001 * hashf(q.h);
            if (pq < pr || (pq == pr && sl > int(gid))) continue;
            drop = length(q.c - r.c) < reach + isleReach(Rq, q.h) + 30.0;
        }
    T[gid].R = drop ? -R : R;
}

struct GLimb { float3 p0, p1, p2; float r0, r1; float L; };

static inline GLimb giantLimb(thread const Giant& g, int i) {
    uint hi = pcg(g.h + 1000u + uint(i) * 131u);
    float f = (float(i) + 0.5) / float(GIANT_LIMBS);
    // Kapok (kind 1): limbs only near the top, long and level, a flat crown on a clear column.
    bool kapok = g.kind == 1;
    float y0 = g.yT * mix(kapok ? 0.58 : 0.42, 0.98, f) + 12.0 * (hashf(hi) - 0.5);
    float az = float(i) * 2.39996 + 0.5 * hashf(pcg(hi));
    float L = mix(kapok ? 112.0 : 125.0, kapok ? 80.0 : 62.0, f) * (0.85 + 0.3 * hashf(pcg(hi + 1u)));
    float2 dir = float2(cos(az), sin(az)), perp = float2(-dir.y, dir.x);
    float bend = (hashf(pcg(hi + 2u)) - 0.5) * 0.5;
    GLimb l;
    l.p0 = float3(dir * 8.0, y0).xzy;
    // A weeping giant (kind 2) arches its limbs out and down.
    float rise = kapok ? 0.16 : g.kind == 2 ? 0.30 : 0.34, end = kapok ? 0.10 : g.kind == 2 ? 0.02 : 0.30;
    l.p1 = float3(dir * L * 0.45 + perp * L * bend * 0.4, y0 + L * rise).xzy;
    l.p2 = float3(dir * L + perp * L * bend, y0 + L * end + 6.0 * hashf(pcg(hi + 3u))).xzy;
    l.r0 = mix(10.0, 5.5, f);
    l.r1 = mix(2.6, 1.6, f);
    l.L = L;
    return l;
}

static inline float3 bez(GLimb l, float t) { return mix(mix(l.p0, l.p1, t), mix(l.p1, l.p2, t), t); }

// Pad j: centre (xz relative to the axis, y absolute), radius, thickness. Each limb carries a clump at its tip, a
// second beside it and, on the longer lower limbs, a third part way out; the last few are the crown's dome.
constant int GIANT_PADS = GIANT_LIMBS * 3 + 4;

static inline float4 giantPad(thread const Giant& g, int j, thread float& thick) {
    if (j >= GIANT_LIMBS * 3) {
        int d = j - GIANT_LIMBS * 3;
        uint hd = pcg(g.h + 900u + uint(d));
        float a = float(d) * 2.1 + hashf(hd);
        float2 o = d == 0 ? float2(0) : float2(cos(a), sin(a)) * (26.0 + 14.0 * hashf(pcg(hd)));
        thick = (d == 0 ? 56.0 : 36.0) * (g.kind == 1 ? 0.75 : 1.0);
        return float4(o.x, g.G + g.yT + (d == 0 ? (g.kind == 1 ? 18.0 : 36.0) : 14.0 + 10.0 * hashf(pcg(hd + 1u))), o.y, d == 0 ? 40.0 : 28.0);
    }
    GLimb l = giantLimb(g, j / 3);
    int kind = j % 3;
    uint hp = pcg(g.h + 1500u + uint(j) * 29u);
    float3 p = kind == 1 ? bez(l, 0.62) : l.p2;
    float R = kind == 0 ? 0.27 * l.L + 10.0 : kind == 1 ? 0.22 * l.L + 6.0 : 0.2 * l.L + 7.0;
    thick = (kind == 0 ? 26.0 + 0.14 * l.L : 20.0 + 0.1 * l.L) * (g.kind == 1 ? 0.85 : 1.0);
    if (kind == 1 && l.L < 95.0) R = 0.0;
    if (kind == 2) {
        float2 dir = normalize(l.p2.xz - l.p0.xz);
        float side = hashf(hp) < 0.5 ? -1.0 : 1.0;
        p.xz += float2(-dir.y, dir.x) * side * (0.30 * l.L) - dir * 0.12 * l.L;
        p.y += 4.0 + 10.0 * hashf(pcg(hp));
    }
    return float4(p.x, g.G + p.y + (kind == 1 ? 3.0 : 6.0), p.z, R);
}

// Solid runs of a column at offset q from the axis (ground gl): appended as intervals, then sorted and merged.
struct GRuns { float lo[24]; float hi[24]; int m[24]; int n; };

static inline void gAdd(thread GRuns& r, float lo, float hi, int m) {
    if (hi - lo < 0.5 || r.n >= 24) return;
    r.lo[r.n] = lo; r.hi[r.n] = hi; r.m[r.n] = m; r.n++;
}

static inline float giantTrunkR(thread const Giant& g, float2 q, float yr) {
    float t = saturate(yr / (g.yT + 30.0));
    float R = mix(25.0, 8.0, t * t * (3.0 - 2.0 * t)) + 26.0 * exp(-max(yr, 0.0) / 18.0);
    // Braided like a strangler fig's mass of roots: deep flutes that twist slowly up the trunk; a kapok's column is
    // smooth and straight, its foot carried by the buttresses.
    if (g.kind == 1) R = mix(19.0, 10.0, t) + 10.0 * exp(-max(yr, 0.0) / 12.0);
    float th = atan2(q.y, q.x) + yr / 150.0;
    float f = perlin3(float3(cos(th) * 2.6, sin(th) * 2.6, yr / 34.0), g.h + 5u);
    return R * (1.0 + (g.kind == 1 ? 0.10 : 0.34) * f);
}

static void giantIntervals(thread const Giant& g, float2 wq, float gl, int lod, thread GRuns& r) {
    float2 q = wq - g.c;
    float d = length(q);
    float G = g.G;
    // Everything that reads the table relies on this bound.
    if (d > GIANT_REACH - 1.0) return;
    // Trunk: marched up from the ground (its radius is not monotone in height once fluted).
    if (d < 80.0) {
        float lo = -1.0;
        float yTop = g.yT + 34.0;
        for (float yr = gl - G; yr < yTop; yr += 1.0) {
            bool in = d < giantTrunkR(g, q, yr + 0.5);
            if (in && lo < 0.0) lo = G + yr;
            if (!in && lo >= 0.0) { gAdd(r, lo, G + yr, GM_WOOD); lo = -1.0; }
        }
        if (lo >= 0.0) gAdd(r, lo, G + yTop, GM_WOOD);
    }
    // Buttress flanges, sinuous plates falling away from the trunk, each running on as a root that arches clear of
    // the ground and dives back in.
    for (int i = 0; i < GIANT_FLANGES; i++) {
        uint hi = pcg(g.h + uint(i) * 7919u);
        float phi = (float(i) + 0.6 * hashf(hi)) * 6.2831853 / float(GIANT_FLANGES);
        float2 dir = float2(cos(phi), sin(phi));
        float along = dot(q, dir);
        if (along < 8.0) continue;
        float Lb = 40.0 + 36.0 * hashf(pcg(hi));
        float La = 45.0 + 70.0 * hashf(pcg(hi + 1u));
        if (along > Lb + La) continue;
        float side = dot(q, float2(-dir.y, dir.x)) + 9.0 * perlin2(float2(along / 38.0, float(i) * 3.7), g.h + 11u) * smoothstep(10.0, 50.0, along);
        // A kapok's buttresses are tall plates, flaring far up its column.
        float Hb = (g.kind == 1 ? 55.0 : 30.0) + 42.0 * hashf(pcg(hi + 2u));
        if (along < Lb) {
            float u = saturate((along - 14.0) / (Lb - 14.0));
            float w = mix(7.0, 2.5, u);
            if (abs(side) < 0.5 * w) {
                float top = G + Hb * pow(1.0 - u, 1.7) * (1.0 + 0.12 * perlin2(float2(along / 9.0, float(i)), g.h + 12u)) + 2.0;
                gAdd(r, gl - 2.0, max(top, gl + 1.0), GM_WOOD);
            }
        }
        float s = (along - Lb * 0.75) / (Lb * 0.25 + La);
        if (s > 0.0 && s < 1.0) {
            float rr = mix(2.8, 1.3, s);
            if (abs(side) < rr) {
                float arch = max(0.0, sin(s * 3.14159 * 2.0 + hashf(pcg(hi + 3u)))) * (6.0 + 10.0 * hashf(pcg(hi + 4u))) * (1.0 - s);
                float yc = gl + arch;
                float hh = sqrt(rr * rr - side * side);
                gAdd(r, yc - hh, yc + hh + 0.5, GM_WOOD);
            }
        }
    }
    // Limbs, and props dropped from them to the ground like a banyan's.
    for (int i = 0; i < GIANT_LIMBS; i++) {
        GLimb l = giantLimb(g, i);
        float2 a = l.p0.xz, b = l.p2.xz;
        float2 ab = b - a;
        float tt = saturate(dot(q - a, ab) / dot(ab, ab));
        if (length(q - (a + ab * tt)) > l.L * 0.3 + 10.0) continue;
        float lo = 1e9, hi = -1e9;
        for (int k = 0; k <= 28; k++) {
            float t = float(k) / 28.0;
            float3 p = bez(l, t);
            float rad = mix(l.r0, l.r1, t);
            float dh = length(q - p.xz);
            if (dh < rad) {
                float hh = sqrt(rad * rad - dh * dh);
                lo = min(lo, p.y - hh);
                hi = max(hi, p.y + hh);
            }
        }
        if (hi > lo) gAdd(r, G + lo, G + hi, GM_WOOD);
        // Aerial roots: dropped from the limb, leaning out as they fall, thickening where they root.
        for (int k = 0; k < 3; k++) {
            uint hp = pcg(g.h + uint(i) * 97u + uint(k) * 13u + 400u);
            if (hashf(hp) > (g.kind == 1 ? 0.0 : g.kind == 2 ? 0.25 : 0.45) || lod >= 1) continue;
            float t = 0.35 + 0.2 * float(k) + 0.1 * hashf(pcg(hp));
            float3 top = bez(l, t);
            float2 out = normalize(top.xz + 1e-3) * (4.0 + 10.0 * hashf(pcg(hp + 2u)));
            float3 foot = float3(top.x + out.x, gl - G - 1.0, top.z + out.y);
            float rad = 0.8 + 2.8 * pow(hashf(pcg(hp + 1u)), 2.0);
            float2 ab2 = foot.xz - top.xz;
            float tq = saturate(dot(q - top.xz, ab2) / max(dot(ab2, ab2), 1e-3));
            if (length(q - (top.xz + ab2 * tq)) > rad + 6.0) continue;
            float lo2 = 1e9, hi2 = -1e9;
            for (int m = 0; m <= 24; m++) {
                float u = float(m) / 24.0;
                // Straight down at first, swinging out low: a hanging root that found the ground.
                float3 pp = float3(mix(top.x, foot.x, u * u), mix(top.y, foot.y, u), mix(top.z, foot.z, u * u));
                pp.xz += float2(perlin2(float2(u * 2.2, hashf(hp) * 40.0), g.h + 42u), perlin2(float2(u * 2.2 + 7.7, hashf(hp) * 40.0), g.h + 43u))
                       * 4.0 * sin(u * 3.14159);
                float rr = rad * (1.0 + 0.8 * smoothstep(0.75, 1.0, u));
                float dh = length(q - pp.xz);
                if (dh < rr) {
                    float hh = max(sqrt(rr * rr - dh * dh), 0.5 * abs(top.y - foot.y) / 24.0 + 0.5);
                    lo2 = min(lo2, pp.y - hh);
                    hi2 = max(hi2, pp.y + hh);
                }
            }
            if (hi2 > lo2) gAdd(r, G + lo2, G + hi2, GM_WOOD);
        }
    }
    // Canopy pads: broad, flat-bottomed clouds of leaves with lumpy tops, the tiers of the crown; vines hang from them.
    // Fine terms break the clumps' smooth profiles, which column by column read as ribs.
    float lump = 0.6 * perlin2(wq / 13.0, g.h + 20u) + 0.4 * perlin2(wq / 5.5, g.h + 21u) + 0.35 * perlin2(wq / 2.3, g.h + 25u);
    float sag = perlin2(wq / 7.0, g.h + 22u) + 0.6 * perlin2(wq / 2.7 + 3.1, g.h + 26u);
    float padBot = 1e9;
    for (int j = 0; j < GIANT_PADS; j++) {
        float T;
        float4 P = giantPad(g, j, T);
        if (P.w <= 0.0) continue;
        float2 dq = q - P.xz;
        float dd = length(dq);
        if (dd > P.w * 1.3) continue;
        float2 u2 = dq / max(dd, 1e-3);
        float edge = P.w * (1.0 + 0.32 * perlin2(u2 * 1.8 + float(j) * 5.3, g.h + 23u) + 0.12 * perlin2(u2 * 4.5 + float(j), g.h + 24u));
        if (dd >= edge) continue;
        float u = dd / edge;
        float top = P.y + T * 0.55 * sqrt(1.0 - u * u) + 7.0 * lump;
        float bot = P.y - T * 0.45 * pow(1.0 - u * u, 0.6) + 3.0 * sag;
        if (top > bot + 1.0) { gAdd(r, floor(bot), floor(top) + 1.0, GM_LEAF); padBot = min(padBot, floor(bot)); }
    }
    if (padBot < 1e8 && lod == 0) {
        uint hv = hash2i(int2(floor(wq)), g.h + 30u);
        float mask = smoothstep(0.0, 0.35, perlin2(wq / 19.0, g.h + 31u));
        // A weeping giant trails long curtains; a kapok's crown hardly any.
        if (hashf(hv) < (g.kind == 2 ? 0.09 : g.kind == 1 ? 0.008 : 0.035) * mask) {
            float len = 4.0 + (g.kind == 2 ? 55.0 : 30.0) * pow(hashf(pcg(hv)), g.kind == 2 ? 1.3 : 2.0);
            gAdd(r, max(padBot - len, gl + 3.0), padBot + 1.0, GM_VINE);
        }
    }
}

// A vegetation column from its solid intervals over ground gl: up to four solid runs (the lowest from the ground), as
// (top, principal gap) plus two extra gaps, the run count n, the top run's first leaf, and per run (the first three) a
// material: leaf, vine or wood by covered height; for the top run, that of its part below the first leaf. False if
// nothing stands over the ground.
struct GCol { float H, gl, gh; float4 e; float leafStart; uint mats; int n; };

static bool runsColumn(thread GRuns& r, float gl, thread GCol& o) {
    if (r.n == 0) return false;
    // Sort by start.
    for (int i = 1; i < r.n; i++) {
        float a = r.lo[i], b = r.hi[i];
        int m = r.m[i];
        int j = i - 1;
        while (j >= 0 && r.lo[j] > a) { r.lo[j + 1] = r.lo[j]; r.hi[j + 1] = r.hi[j]; r.m[j + 1] = r.m[j]; j--; }
        r.lo[j + 1] = a; r.hi[j + 1] = b; r.m[j + 1] = m;
    }
    // Merge into runs, the ground run first; gaps under two voxels close.
    float RL[8], RH[8], leafLo[8], leafH[8], vineH[8];
    int n = 1;
    RL[0] = 0.0; RH[0] = gl; leafLo[0] = 1e9; leafH[0] = 0; vineH[0] = 0;
    for (int i = 0; i < r.n; i++) {
        float a = max(r.lo[i], 0.0), b = r.hi[i];
        if (b <= a) continue;
        if (a <= RH[n - 1] + 2.0) RH[n - 1] = max(RH[n - 1], b);
        else if (n < 8) { RL[n] = a; RH[n] = b; leafLo[n] = 1e9; leafH[n] = 0; vineH[n] = 0; n++; }
        else RH[n - 1] = max(RH[n - 1], b);
        if (r.m[i] == GM_LEAF) { leafLo[n - 1] = min(leafLo[n - 1], a); leafH[n - 1] += b - a; }
        if (r.m[i] == GM_VINE) vineH[n - 1] += b - a;
    }
    if (n == 1 && RH[0] <= gl) return false;
    // At most four runs: close the narrowest gaps.
    while (n > 4) {
        int best = 1;
        float bw = 1e9;
        for (int i = 1; i < n; i++) if (RL[i] - RH[i - 1] < bw) { bw = RL[i] - RH[i - 1]; best = i; }
        RH[best - 1] = RH[best];
        leafLo[best - 1] = min(leafLo[best - 1], leafLo[best]);
        leafH[best - 1] += leafH[best];
        vineH[best - 1] += vineH[best];
        for (int i = best; i < n - 1; i++) { RL[i] = RL[i + 1]; RH[i] = RH[i + 1]; leafLo[i] = leafLo[i + 1]; leafH[i] = leafH[i + 1]; vineH[i] = vineH[i + 1]; }
        n--;
    }
    o.n = n;
    o.H = RH[n - 1];
    o.gl = n >= 2 ? RH[n - 2] : 0.0;
    o.gh = n >= 2 ? RL[n - 1] : 0.0;
    o.e = float4(n >= 3 ? RH[n - 3] : 0.0, n >= 3 ? RL[n - 2] : 0.0, n >= 4 ? RH[n - 4] : 0.0, n >= 4 ? RL[n - 3] : 0.0);
    o.leafStart = leafLo[n - 1] < 1e8 ? max(leafLo[n - 1], RL[n - 1]) : o.H;
    o.mats = 0u;
    for (int i = 0; i < min(n, 3); i++) {
        float lo = max(RL[i], i == 0 ? gl : RL[i]);
        uint m;
        if (i == n - 1) m = vineH[i] > 0.4 * (o.leafStart - lo) ? uint(GM_VINE) : uint(GM_WOOD);
        else m = leafH[i] > 0.4 * (RH[i] - lo) ? uint(GM_LEAF) : vineH[i] > 0.4 * (RH[i] - lo) ? uint(GM_VINE) : uint(GM_WOOD);
        o.mats |= m << (2u * uint(i));
    }
    return true;
}

// The column a giant makes at world column wq over ground gl, with any other intervals already in r (the jungle's).
static bool giantColumn(thread const Giant& g, float2 wq, float gl, thread GRuns& r, thread GCol& o, int lod = 0) {
    giantIntervals(g, wq, gl, lod, r);
    return runsColumn(r, gl, o);
}

// MARK: - Jungle

// A jungle is a roof, not a crowd of separate trees: rounded crowns packed one to a JUNGLE_CELL cell over a lower, lumpy
// fill, a dozen and more voxels above the ground and a few thick, opened by clearings that widen toward the province's
// edge until the roof breaks into groves; emergents (one to an EMERGENT_CELL cell, about half of them) break through it
// on tall trunks with flat, layered umbrella crowns. Every part is a closed-form function of position and of the ground
// under it, so the coarse rings rebuild the same forest.
constant float JUNGLE_CELL = 8.0;
constant float EMERGENT_CELL = 26.0;
// Ground range over a 10-voxel window (plantTrees' +-5) from which the roof steps down, and past which none stands.
constant float JUNGLE_STEEP = 7.0, JUNGLE_CLIFF = 13.0;

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
    float a = (float(j) + 0.6 * hashf(hj)) * 2.3999632 + hashf(h) * 6.2831853;
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

// The intervals emergent e (trunk ground gt) adds to world column centre w: its trunk, and its crown's noise-edged,
// domed pads. lod >= 1 (coarse rings): the trunk only where it fills the column.
static void emergentIntervals(float2 w, float2 ctr, float gt, float y0, float R, int pads, uint h, uint seed, int lod, thread GRuns& r) {
    float2 q = w - ctr;
    float d = length(q);
    float tr = R > 9.0 ? 1.2 : 0.75;
    if (d < tr + (lod > 0 ? 0.0 : 0.0)) gAdd(r, gt, gt + y0, GM_WOOD);
    float2 p = w + seedOffset(seed);
    float lumps = 1.6 * perlin2(p / 3.2, seed + 332u) + 1.2 * perlin2(p / 7.0, seed + 333u);
    for (int j = 0; j < pads; j++) {
        float T;
        float4 P = emergentPad(h, j, R, y0, T);
        float2 dq = q - P.xz;
        float dd = length(dq);
        if (dd > P.w * 1.35) continue;
        float2 u2 = dq / max(dd, 1e-3);
        float edge = P.w * (1.0 + 0.30 * perlin2(u2 * 1.7 + float(j) * 4.1 + hashf(h) * 50.0, seed + 331u));
        if (dd >= edge) continue;
        float u = dd / edge;
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

// MARK: - Old growth

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
    dst.write(float4(gc.H, gc.gl, gc.gh, GIANT_A), gid);
    gx.write(uint4(uint(gc.e.x), uint(gc.e.y), uint(gc.e.z) | (GAP_GIANT << 11), uint(gc.e.w)), gid);
    st.write(uint4(ST_GIANT | uint(G) | (uint(gc.leafStart) << 11) | (gc.mats << 22) | (pal << 28), 0, 0, 0), gid);
}

// MARK: - Tower overgrowth

// The towers' own greenery, the hook stage-3 vegetation replaces: the leaf span [x, y) it adds to map column p (world
// wc, top G), empty off the towers: low crowns standing well back on the broad treads, the court and the plinth. A crown
// must have tread under the most of it: one hanging off a narrow ledge or a pier's top reads, from below, as a ball stuck
// to a blank wall.
// (The ivy on the faces is painted by towerSurface: hung as leaves in the column in front of a face, any mat broke up
// into vertical bars on the faces' jogs.)
static int2 towerOvergrowth(texture2d<float, access::read> src, int2 p, int2 wc, float G, uint seed) {
    float2 wf = float2(wc) + 0.5;
    Tower T;
    TowerGroup g;
    if (!towerFind(wf, 0.0, seed, T, g)) return int2(0);
    float2 d0 = wf - T.c;
    float Rg = T.R0 * 1.42 + 6.0;
    if (dot(d0, d0) > Rg * Rg || !towerGroupStands(g, seed)) return int2(0);
    T.Hb = towerBase(T, seed);
    if (T.Hb > TOWER_BASE_MAX) return int2(0);
    Climate cl = climate(T.c, seed);
    float wet = towerWet(cl);
    float snow = treelineAt(snowlineAt(cl.temp));
    int n = int(src.get_width());
    int2 span = int2(0);
    // Crowns on a jittered 9-voxel grid: a site holds one where it stands on a flat tread of the tower or its plinth.
    const int C = 9;
    int2 home = int2(floor(wf / float(C)));
    int2 origin = wc - p;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cell = home + int2(i, j);
            uint h = hash2i(cell, seed + 563u);
            int2 cw = cell * C + int2(h % uint(C), (h / 11u) % uint(C));
            float r = 2.5 + 4.0 * hashf(pcg(h + 1u));
            float2 o = float2(wc - cw);
            if (dot(o, o) > r * r) continue;
            int2 pc = cw - origin;
            if (any(pc < 1) || any(pc >= n - 1)) continue;
            float4 cc = src.read(uint2(pc));
            float Gc = cc.r;
            if (cc.b > cc.g || Gc < SEA + 2.0 || Gc > snow) continue;
            float2 dc = float2(cw) + 0.5 - T.c;
            if (dot(dc, dc) > T.R0 * T.R0 * 1.9) continue;
            bool flat = true;
            for (int k = 0; k < 4; k++) {
                int2 q = pc + int2(k == 0 ? 1 : k == 1 ? -1 : 0, k == 2 ? 1 : k == 3 ? -1 : 0);
                if (abs(src.read(uint2(q)).r - Gc) > 1.0) flat = false;
            }
            // Tread (or the wall behind it) under the crown out to 0.85 of its radius all round.
            for (int k = 0; k < 8 && flat; k++) {
                float a = float(k) * 0.785398;
                int2 q = pc + int2(round(float2(cos(a), sin(a)) * (r * 0.85 + 0.5)));
                if (any(q < 0) || any(q >= n) || src.read(uint2(q)).r < Gc - 1.0) flat = false;
            }
            if (!flat) continue;
            bool ledge = Gc > T.Hb + 8.0;
            if (hashf(pcg(h + 2u)) > (ledge ? 0.1 : 0.15) + 0.45 * wet) continue;
            float trunk = floor(2.0 * hashf(pcg(h + 3u)));
            float hv = 0.6 * r * sqrt(saturate(1.0 - dot(o, o) / (r * r)));
            float yc = Gc + trunk + 0.6 * r;
            int2 s = int2(int(floor(yc - hv)), int(ceil(yc + hv)));
            // The trunk's column reaches the ground.
            if (all(wc == cw)) s.x = int(Gc);
            if (s.y <= s.x) continue;
            if (span.y <= span.x || (s.x <= span.y && s.y >= span.x)) span = span.y > span.x ? int2(min(s.x, span.x), max(s.y, span.y)) : s;
            else if (s.y > span.y) span = s;
        }
    // Scrub and fern in a low, patchy mat over the tower's level treads (not the plinth, which the land's own trees
    // and the crowns hold): a green fringe along the rims rather than a row of beads.
    int2 pp = wc - origin;
    if (G > T.Hb + 8.0 && G < snow && all(pp >= 1) && all(pp < n - 1)) {
        float mn = perlin2(wf / 7.0, seed + 564u) + 0.5 * perlin2(wf / 2.7, seed + 565u) + 0.7 * wet - 0.35;
        if (mn > 0.0) {
            int2 m2 = int2(int(G), int(G) + 1 + int(floor(3.0 * saturate(mn * 1.5))));
            if (span.y <= span.x) span = m2;
            else span = int2(min(span.x, m2.x), max(span.y, m2.y));
        }
    }
    return span;
}

// Runs after cleanupColumns, which would clamp one-column trunks away. Tree columns set w = first leaf
// voxel; g then holds the ground height (b = g on trunks and buried canopies, b = w under a floating canopy).
kernel void plantTrees(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       texture2d<uint, access::read_write> gx [[texture(2)]],
                       texture2d<uint, access::write> st [[texture(3)]],
                       constant int4& info [[buffer(0)]],
                       constant uint& row0 [[buffer(1)]],
                       device const GiantRec* giants [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 col = src.read(gid);
    // Structures pass through; a clearing keeps its mark unless a canopy spreads over it.
    bool cleared = col.a < 0.0;
    col.a = min(col.a, 0.0);
    bool built = col.a < 0.0 && -col.a < col.r;
    if (built || col.b > col.g) { dst.write(col, gid); return; }
    int2 p = int2(gid);
    int2 wc = p + info.xy;
    uint seed = uint(info.w);
    float2 w = float2(wc) + 0.5;
    Giant gi;
    bool isGiant = giantAt(giants, w, 0.0, gi);
    // A crown reaches out over water.
    if (col.r < SEA + 2.0 && !isGiant) { dst.write(col, gid); return; }
    float G = col.r;
    GRuns rr;
    rr.n = 0;
    int2 tg = towerOvergrowth(src, p, wc, G, seed);
    float J = col.r >= SEA + 2.0 ? provinces(w + seedOffset(seed), seed).x : 0.0;
    if (tg.y > tg.x && (J >= 0.15 || (col.r >= SEA + 2.0 && provinces(w + seedOffset(seed), seed).y >= 0.2)))
        gAdd(rr, float(tg.x), float(tg.y), GM_LEAF);
    if (J >= 0.15) {
        // The roof follows the ground smoothed over a few columns, and keeps off cliffs.
        float gs = 0.0, gmin = 1e9, gmax = -1e9;
        for (int k = 0; k < 9; k++) {
            int2 q = clamp(p + int2(k % 3 - 1, k / 3 - 1) * 5, int2(0), int2(int(n) - 1));
            float h = src.read(uint2(q)).r;
            gs += h; gmin = min(gmin, h); gmax = max(gmax, h);
        }
        gs /= 9.0;
        uint hc = hash2i(wc, seed + 326u);
        int2 e0 = int2(floor(w / EMERGENT_CELL));
        bool eTrunk = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                float2 ec;
                float y0, R;
                int pads;
                uint eh;
                if (!emergentIn(e0 + int2(i, j), seed, J, ec, y0, R, pads, eh)) continue;
                if (length_squared(w - ec) > (1.35 * R + 0.75 * R) * (1.35 * R + 0.75 * R)) continue;
                int2 tq = int2(floor(ec)) - info.xy;
                if (any(tq < 5) || any(tq >= int(n) - 5)) continue;
                float4 tc = src.read(uint2(tq));
                // On level ground only: not on a ledge, in water, or among buildings.
                float t0 = src.read(uint2(tq + int2(5, 0))).r, t1 = src.read(uint2(tq - int2(5, 0))).r;
                float t2 = src.read(uint2(tq + int2(0, 5))).r, t3 = src.read(uint2(tq - int2(0, 5))).r;
                float tr = max(max(max(t0, t1), max(t2, t3)), tc.r) - min(min(min(t0, t1), min(t2, t3)), tc.r);
                if (tc.r < SEA + 10.0 || tc.b > tc.g || tc.a < 0.0 || abs(tc.r - gs) > 6.0 || tr >= JUNGLE_STEEP) continue;
                eTrunk = eTrunk || length(w - ec) < (R > 9.0 ? 1.2 : 0.75);
                emergentIntervals(w, ec, tc.r, y0, R, pads, eh, seed, 0, rr);
            }
        // A trunk column is bark through the roof (merged with it, its leaves would start at the roof's underside).
        // Steep ground steps the roof down to bushes on each column's own ground; a cliff carries none.
        float steep = gmax - gmin;
        if (!cleared && !eTrunk && steep < JUNGLE_CLIFF) {
            float keep = 1.0 - smoothstep(JUNGLE_STEEP, JUNGLE_CLIFF, steep);
            JCan cn = jungleCanopy(w, mix(G, gs, keep), J, seed, keep);
            if (cn.hi > max(cn.lo, G)) {
                gAdd(rr, cn.lo, cn.hi, GM_LEAF);
                // Its crowns stand on poles; lianas hang from it.
                if (cn.trunk) gAdd(rr, G, cn.lo, GM_WOOD);
                else if (hashf(hc) < 0.02) gAdd(rr, G, cn.lo, GM_VINE);
            } else if (hashf(hc) < 0.3) {
                // Ferns and saplings in the clearings.
                gAdd(rr, G, G + 1.0 + floor(2.5 * hashf(pcg(hc))), GM_LEAF);
            }
        }
    } else if (col.r >= SEA + 2.0 && provinces(w + seedOffset(seed), seed).y >= 0.2) {
        float O = provinces(w + seedOffset(seed), seed).y;
        int2 c0 = int2(floor(w / OLD_CELL));
        bool covered = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                float2 tc;
                float Ht, R, tr;
                uint th;
                if (!oldTreeIn(c0 + int2(i, j), seed, O, tc, Ht, R, tr, th)) continue;
                if (length_squared(w - tc) > 1.3 * R * R) continue;
                int2 tq = int2(floor(tc)) - info.xy;
                if (any(tq < 3) || any(tq >= int(n) - 3)) continue;
                float4 t0 = src.read(uint2(tq));
                // On level ground, clear of water, rock and buildings.
                float s0 = src.read(uint2(tq + int2(3, 0))).r, s1 = src.read(uint2(tq - int2(3, 0))).r;
                float s2 = src.read(uint2(tq + int2(0, 3))).r, s3 = src.read(uint2(tq - int2(0, 3))).r;
                if (t0.r < SEA + 2.0 || t0.b > t0.g || t0.a < 0.0 ||
                    max(max(s0, s1), max(s2, s3)) - min(min(s0, s1), min(s2, s3)) > 6.0) continue;
                covered = oldTreeIntervals(w, tc, t0.r, Ht, R, tr, th, seed, true, rr) > G || covered;
            }
        // Ferns on the mossy floor.
        uint hc = hash2i(wc, seed + 352u);
        if (!covered && !cleared && hashf(hc) < 0.25) gAdd(rr, G, G + 1.0 + floor(1.8 * hashf(pcg(hc))), GM_LEAF);
    } else if (col.r >= SEA + 2.0) {
        int2 home = int2(floor(float2(wc) / float(TREE_CELL)));
        int lo = 100000, hi = 0;
        bool trunk = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                Tree t = treeInCell(src, gx, home + int2(i, j), info.xy, seed);
                if (!t.ok) continue;
                int2 o = p - t.c;
                if (abs(o.x) > 2 || abs(o.y) > 2) continue;
                int2 s = canopySpan(t, o);
                // Shrubs hug the ground they spill onto rather than hovering over a lower step.
                if (t.kind == 0) s += int(col.r) - t.lo;
                if (s.y <= s.x) continue;
                lo = min(lo, s.x);
                hi = max(hi, s.y);
                if (all(o == 0) && t.kind != 0) trunk = true;
            }
        if (tg.y > tg.x) { lo = min(lo, tg.x); hi = max(hi, tg.y); }
        if (!isGiant) {
            if (float(hi) <= G) { dst.write(col, gid); return; }
            float leaf = max(float(lo), G);
            bool floating = !trunk && leaf > G;
            dst.write(float4(float(hi), G, floating ? leaf : G, leaf), gid);
            return;
        }
        if (float(hi) > G) gAdd(rr, trunk ? G : float(lo), float(hi), GM_LEAF);
    }
    GCol gc;
    if (isGiant ? giantColumn(gi, w, G, rr, gc) : runsColumn(rr, G, gc)) writeVeg(gc, G, gid, dst, gx, st, isGiant ? giantPalette(gi.h) : 0u);
    else dst.write(col, gid);
}

// Height a column is solid from the ground up: under an arch's gap, a canopy, a building's openings or a cave only the
// part below them counts. Extra gaps lie below the principal one.
static inline float solidFloor(float4 c, uint4 gx) {
    if (c.a > 0.0) return c.g;
    float4 e = extraGaps(gx);
    if (e.w > e.z) return e.z;
    if (e.y > e.x) return e.x;
    // Structures too: a lintel, eave, deck, stilt house or hull is open in [g, b) whatever its base says.
    return c.b > c.g ? c.g : c.r;
}

// Mip 0 of the max-mip texture: the column tops alone, 2 bytes a column for sky visibility (the DDA reads mips from 1).
// Under an island its ground: the sky over the land beside one is open but for the island's own occlusion.
kernel void copyTop(texture2d<float, access::read> src [[texture(0)]],
                    texture2d<float, access::write> dst [[texture(1)]],
                    texture2d<uint, access::read> gx [[texture(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float4 c = src.read(gid);
    bool isl = c.a == 0.0 && c.b > c.g && (gapFlags(gx.read(gid)) & GAP_ISLAND) != 0u;
    dst.write(float4(isl ? c.g : c.r, 0, 0, 0), gid);
}

// Band mips: per cell the height band [x, y) every column in it leaves open, so a ray under a giant's crown
// skips cells as one over the tops does. A giant column offers its tallest gap, any other column the air above it.
// The open band of one column: a canopy column's tallest gap (at least 4 tall), any other column the air above it.
static inline float2 columnBand(texture2d<float, access::read> src, texture2d<uint, access::read> gx, uint2 c) {
    float4 col = src.read(c);
    float2 b = float2(col.r, 4096.0);
    if (col.a == GIANT_A) {
        float4 e = extraGaps(gx.read(c));
        float2 g = col.b > col.g ? float2(col.g, col.b) : float2(0);
        if (e.y - e.x > g.y - g.x) g = e.xy;
        if (e.w - e.z > g.y - g.x) g = e.zw;
        if (g.y - g.x >= 4.0) b = g;
    } else if (col.a > 0.0 && col.b - col.g >= 4.0) {
        // Under a floating canopy (the jungle's roof).
        b = col.gb;
    }
    return b;
}

static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles, thread float2& g2, thread float& depth);

// A band texel's floor and whether it holds for every morph toward level 1 (stored as -1 - floor), so the trace may use
// it inside the morph band too.
static inline float2 bandLo(float x) { return x < 0.0 ? float2(-1.0 - x, 1.0) : float2(x, 0.0); }

// Band mips from mip 1 (the trace never reads mip 0): per 2x2 columns the band [x, y) all of them leave open. Under an
// island the band holds through the morph toward level 1 (columnMorphW): its columns' air meets the level-1 column's;
// elsewhere the floor is mip 1's max, which bounds that column's top as well.
kernel void bandTop(texture2d<float, access::read> src [[texture(0)]],
                    texture2d<uint, access::read> gx [[texture(1)]],
                    texture2d<float, access::write> dst [[texture(2)]],
                    texture2d<float, access::read> mx1 [[texture(3)]],
                    constant int4& info [[buffer(0)]],
                    device const GiantRec* giants [[buffer(1)]],
                    device const IsleRec* isles [[buffer(2)]],
                    constant uint& row0 [[buffer(3)]],
                    uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    uint2 cs[4] = { s, s + uint2(1, 0), s + uint2(0, 1), s + uint2(1, 1) };
    bool ig[4];
    bool any = false;
    for (int i = 0; i < 4; i++) {
        float4 col = src.read(cs[i]);
        ig[i] = col.a == 0.0 && col.b > col.g && (gapFlags(gx.read(cs[i])) & GAP_ISLAND) != 0u;
        any = any || ig[i];
    }
    if (any) {
        float2 g2;
        float dp;
        float3 pc = coarseColumn((int2(s) + info.xy) >> 1, 1, uint(info.w), giants, isles, g2, dp);
        bool pIsl = pc.y < 0.0;
        float2 pb = pc.z > abs(pc.y) ? float2(abs(pc.y), pc.z) : float2(pc.x, 4096.0);
        float2 b = float2(-1e4, 4096.0);
        for (int i = 0; i < 4; i++) {
            float4 col = src.read(cs[i]);
            float2 cb = float2(max(col.r, pc.x), 4096.0);
            if (ig[i] || pIsl) {
                float2 o = ig[i] ? col.gb : float2(col.r, 4096.0);
                o = float2(max(o.x, pb.x), min(o.y, pb.y));
                if (o.y - o.x >= 4.0) cb = o;
            }
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
        dst.write(float4(-1.0 - b.x, b.y, 0, 0), gid);
        return;
    }
    float2 a = columnBand(src, gx, s), b = columnBand(src, gx, s + uint2(1, 0));
    float2 c = columnBand(src, gx, s + uint2(0, 1)), d = columnBand(src, gx, s + uint2(1, 1));
    float2 r = float2(max(max(a.x, b.x), max(c.x, d.x)), min(min(a.y, b.y), min(c.y, d.y)));
    if (r.y >= 4000.0) r = float2(-1.0 - max(r.x, abs(mx1.read(gid).r)), 4096.0);
    dst.write(float4(r, 0, 0), gid);
}

kernel void bandReduce(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float2 a = src.read(s).rg, b = src.read(s + uint2(1, 0)).rg, c = src.read(s + uint2(0, 1)).rg, d = src.read(s + uint2(1, 1)).rg;
    float2 la = bandLo(a.x), lb = bandLo(b.x), lc = bandLo(c.x), ld = bandLo(d.x);
    float lo = max(max(la.x, lb.x), max(lc.x, ld.x));
    bool safe = min(min(la.y, lb.y), min(lc.y, ld.y)) > 0.0;
    dst.write(float4(safe ? -1.0 - lo : lo, min(min(a.y, b.y), min(c.y, d.y)), 0, 0), gid);
}

// Negates a max mip's cells whose band is a real one (under a giant's crown), after every mip is built.
kernel void bandFlag(texture2d<float, access::read_write> mx [[texture(0)]],
                     texture2d<float, access::read> band [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= mx.get_width() || gid.y >= mx.get_height()) return;
    float2 b = band.read(gid).rg;
    b.x = bandLo(b.x).x;
    if (b.y < 4000.0 && b.y - b.x >= 4.0) mx.write(float4(-abs(mx.read(gid).r)), gid);
}

kernel void reduceMax(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float m = max(max(src.read(s).r, src.read(s + uint2(1, 0)).r), max(src.read(s + uint2(0, 1)).r, src.read(s + uint2(1, 1)).r));
    dst.write(float4(m, 0, 0, 0), gid);
}

// (ground, lowest underside) of a column: an island's ground and the underside over it, else its top and none.
static inline float2 isleGroundUnder(float4 c, uint4 gx) {
    return c.a == 0.0 && c.b > c.g && (gapFlags(gx) & GAP_ISLAND) != 0u ? c.gb : float2(c.r, 4096.0);
}

// The CPU readback's mips. r: the highest column below (the camera steers by it); g: the lowest solid floor below
// (the birds' sightline test); b: the highest ground below, islands aside; a: the lowest island underside (4096: none).
kernel void reduceCpu(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      texture2d<uint, access::read> gx [[texture(2)]],
                      constant int& fromColumns [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float4 a = src.read(s), b = src.read(s + uint2(1, 0)), c = src.read(s + uint2(0, 1)), d = src.read(s + uint2(1, 1));
    float m = max(max(a.r, b.r), max(c.r, d.r));
    if (fromColumns == 0) {
        dst.write(float4(m, min(min(a.g, b.g), min(c.g, d.g)), max(max(a.b, b.b), max(c.b, d.b)), min(min(a.a, b.a), min(c.a, d.a))), gid);
        return;
    }
    uint4 ga = gx.read(s), gb = gx.read(s + uint2(1, 0)), gc = gx.read(s + uint2(0, 1)), gd = gx.read(s + uint2(1, 1));
    float lo = min(min(solidFloor(a, ga), solidFloor(b, gb)), min(solidFloor(c, gc), solidFloor(d, gd)));
    float2 ia = isleGroundUnder(a, ga), ib = isleGroundUnder(b, gb), ic = isleGroundUnder(c, gc), id = isleGroundUnder(d, gd);
    dst.write(float4(m, lo, max(max(ia.x, ib.x), max(ic.x, id.x)), min(min(ia.y, ib.y), min(ic.y, id.y))), gid);
}

// MARK: - Structures

// Sites: one candidate per 128-voxel world cell, so the same places recur when the map recentres. Every part
// and lamp of a site lies within 128 of its centre, so a column only needs its own and the 8 neighbouring cells.
constant int SITE_CELL = 128;
constant int SITE_GRID = 18;
constant int MAX_PARTS = 32;
constant int MAX_LAMPS = 8;

// Must match the layout read in Sources/Renderer.swift (World.SiteLayout).
struct Site {
    int4 head;               // xy world centre, z type, w hash
    float4 info;             // x bounding radius, y age, z part count, w lamp count
    float4 lamps[MAX_LAMPS]; // xyz offset from the centre (y absolute), w kind + 16 * smoke rise + 1024 * smoke offset code
    int4 parts[MAX_PARTS];
};

constant int T_HENGE = 1, T_CAIRN = 2, T_RUINS = 3, T_HAMLET = 4, T_JETTY = 5, T_CAMP = 6;
constant int P_BOX = 1, P_HOUSE = 2, P_DECK = 3, P_BOAT = 4, P_HENGE = 5, P_WALL = 6, P_RING = 7, P_FIRE = 8, P_CLEAR = 9, P_ROUND = 10;
// Yard and camp props (y1 holds the type, y0 the ground they stand on).
constant int P_PROP = 11;
constant int PR_WOODPILE = 1, PR_HAY = 2, PR_GARDEN = 3, PR_WASH = 4, PR_OVEN = 5, PR_RACK = 6, PR_PEN = 7, PR_LOGSEAT = 8,
             PR_BEDROLL = 9, PR_RUG = 10, PR_CANOE = 11, PR_STUMP = 12, PR_PEAT = 13, PR_BALES = 14, PR_BARREL = 15;
// Part flags; bits 8-11 hold a lamp index + 1, bits 12-15 the dwelling style.
constant int F_REL = 1, F_AXISZ = 2, F_FLAT = 4, F_STILT = 8, F_TENT = 16, F_CHIMNEY = 32, F_WATER = 64;
constant int M_STONE = 4, M_TIMBER = 5, M_ROOF = 6, M_PLASTER = 7, M_PLANK = 8, M_CLOTH = 9, M_FIRE = 10;
constant int M_THATCH = 11, M_SHINGLE = 12, M_LOG = 13, M_FELT = 14, M_ADOBE = 15;
// Column features (structTex bits 24-27): pieces whose material differs from their part's.
constant int FT_VIGA = 1, FT_POST = 2, FT_DECK = 3, FT_CROWN = 4, FT_POLE = 5, FT_FENCE = 6, FT_CROP = 7;
// Dwelling styles by biome: hamlets 0-7, camps 8-12.
constant int ST_MEADOW = 0, ST_TIMBER = 1, ST_CABIN = 2, ST_CROFT = 3, ST_ADOBE = 4, ST_PUEBLO = 5, ST_STILT = 6, ST_YURT = 7;
constant int ST_AFRAME = 8, ST_LEANTO = 9, ST_TIPI = 10, ST_CARAVAN = 11, ST_PALM = 12;
constant int L_FIRE = 1, L_HEARTH = 2, L_WINDOW = 3, L_LANTERN = 4, L_BEACON = 5, L_RITUAL = 6, L_TENT = 7;

struct Part { int2 o; int2 sz; int kind; int mat; int y0; int y1; int y2; int flags; };

static inline int4 packPart(Part p) {
    return int4((p.o.x & 0xffff) | (p.o.y << 16),
                (p.sz.x & 0xff) | ((p.sz.y & 0xff) << 8) | ((p.kind & 0xff) << 16) | ((p.mat & 0xff) << 24),
                (p.y0 & 0xffff) | (p.y1 << 16), (p.y2 & 0xffff) | (p.flags << 16));
}

static inline Part unpackPart(int4 v) {
    Part p;
    p.o = int2((v.x << 16) >> 16, v.x >> 16);
    p.sz = int2(v.y & 0xff, (v.y >> 8) & 0xff);
    p.kind = (v.y >> 16) & 0xff;
    p.mat = (v.y >> 24) & 0xff;
    p.y0 = (v.z << 16) >> 16;
    p.y1 = v.z >> 16;
    p.y2 = (v.w << 16) >> 16;
    p.flags = (v.w >> 16) & 0xffff;
    return p;
}

static inline Part makePart(int kind, int mat, int2 o, int2 sz, int y0, int y1, int y2, int flags) {
    Part p;
    p.o = o; p.sz = sz; p.kind = kind; p.mat = mat; p.y0 = y0; p.y1 = y1; p.y2 = y2; p.flags = flags;
    return p;
}

// World cell of site-grid index 0 for a map whose texel 0 sits at world `origin`.
static inline int2 siteCell0(int2 origin) { return int2(floor(float2(origin) / float(SITE_CELL))); }

static inline float colH(texture2d<float, access::read> hm, int2 q) {
    int n = int(hm.get_width());
    return hm.read(uint2(clamp(q, int2(0), int2(n - 1)))).r;
}

static inline bool colArch(texture2d<float, access::read> hm, int2 q) {
    int n = int(hm.get_width());
    float4 c = hm.read(uint2(clamp(q, int2(0), int2(n - 1))));
    return c.b > c.g;
}

struct SiteGen { Site s; int np; int nl; float r; };

static inline void addPart(thread SiteGen& G, Part p, float reach) {
    if (G.np >= MAX_PARTS) return;
    G.s.parts[G.np++] = packPart(p);
    G.r = max(G.r, reach);
}

static inline int partReach(Part p) {
    return int(ceil(length(float2(max(abs(p.o), abs(p.o + p.sz))))));
}

// A window lamp stands at its door; its smoke rises from the chimney, `smoke` columns away (each within 15).
static inline int addLamp(thread SiteGen& G, float3 off, int kind, float rise, int2 smoke = int2(0)) {
    if (G.nl >= MAX_LAMPS) return -1;
    float code = rise > 0.0 && any(smoke != 0) ? float((clamp(smoke.x, -15, 15) + 16) + 32 * (clamp(smoke.y, -15, 15) + 16)) : 0.0;
    G.s.lamps[G.nl] = float4(off, float(kind) + 16.0 * floor(clamp(rise, 0.0, 63.0)) + 1024.0 * code);
    G.r = max(G.r, length(off.xz) + 2.0);
    return G.nl++;
}

static inline float lampRise(float w) { return fmod(floor(w / 16.0), 64.0); }
static inline float2 lampSmokeOff(float w) {
    float code = floor(w / 1024.0);
    return code > 0.0 ? float2(fmod(code, 32.0) - 16.0, floor(code / 32.0) - 16.0) : float2(0);
}

static inline bool flatPatch(texture2d<float, access::read> hm, int2 q, int rad, float tol, thread float& lo, thread float& hi) {
    lo = 1e9; hi = -1e9;
    for (int j = -rad; j <= rad; j++)
        for (int i = -rad; i <= rad; i++) {
            if (colArch(hm, q + int2(i, j))) return false;
            float h = colH(hm, q + int2(i, j));
            lo = min(lo, h); hi = max(hi, h);
        }
    return hi - lo <= tol;
}

// A jetty from the shore along `dir` (a unit grid step) into the water, with a lantern post and moored boats.
static bool buildJetty(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, uint h, int2 dir, int maxReach) {
    int k = 0;
    for (; k < maxReach; k++) if (colH(hm, c + dir * k) < SEA) break;
    if (k == maxReach || k == 0) return false;
    int2 shore = c + dir * (k - 1);
    if (colH(hm, shore) > SEA + 3.0 || colArch(hm, shore)) return false;
    int run = 0;
    for (int m = 1; m <= 16; m++) {
        int2 q = shore + dir * m;
        if (colH(hm, q) >= SEA || colArch(hm, q)) break;
        run++;
    }
    if (run < 5 || G.nl >= MAX_LAMPS || G.np > MAX_PARTS - 3) return false;
    int len = min(run - 1, 6 + int(h % 9u));
    int2 perp = int2(abs(dir.y), abs(dir.x));
    int w = (h >> 5 & 3u) == 0u ? 2 : 1;
    int2 head = shore + dir * len;
    int2 r0 = min(shore, head), r1 = max(shore, head) + 1 + perp * (w - 1);
    int axis = dir.y != 0 ? F_AXISZ : 0;
    int lamp = addLamp(G, float3(float(head.x - c.x) + 0.5, 22.6, float(head.y - c.y) + 0.5), L_LANTERN, 0);
    Part post = makePart(P_BOX, M_TIMBER, head - c, int2(1), 22, 0, 0, F_WATER | ((lamp + 1) << 8));
    addPart(G, post, float(partReach(post)));
    Part deck = makePart(P_DECK, M_PLANK, r0 - c, r1 - r0, 21, 0, int(h & 0x7fffu), axis | F_WATER);
    addPart(G, deck, float(partReach(deck)));
    int boats = int((h >> 9) % 3u);
    for (int b = 0; b < boats && G.np < MAX_PARTS; b++) {
        int side = ((h >> (12 + b)) & 1u) != 0u ? -2 : 1 + w;
        int2 bs = shore + dir * (2 + 3 * b) + perp * side;
        int2 be = bs + dir * 2;
        bool wet = true;
        for (int m = 0; m < 3; m++) wet = wet && colH(hm, bs + dir * m) < SEA - 0.5 && !colArch(hm, bs + dir * m);
        if (!wet) continue;
        int2 b0 = min(bs, be);
        Part boat = makePart(P_BOAT, M_TIMBER, b0 - c, max(bs, be) + 1 - b0, 0, 0, 0, axis | F_WATER);
        addPart(G, boat, float(partReach(boat)));
    }
    return true;
}

// A style's house: wall height over the floor, roof pitch (rise per column), the eave lip's depth, and the overhangs
// past the walls across the ridge (lo, hi side) and beyond each gable. The door side's overhang is a porch or deck:
// a longhouse's deck runs along it, a cabin's porch is under its front gable.
struct HouseForm { int ovLo, ovHi, ovGLo, ovGHi; float wall, pitch, lip; bool hip, flat; };

static inline HouseForm houseForm(int style, uint hh) {
    HouseForm f;
    f.ovLo = 1; f.ovHi = 1; f.ovGLo = 1; f.ovGHi = 1; f.wall = 2.5; f.pitch = 1.0; f.lip = 1.0; f.hip = false; f.flat = false;
    bool doorHi = (hh & 4u) != 0u;
    if (style == ST_MEADOW) { f.hip = true; f.pitch = 1.0; }
    else if (style == ST_TIMBER) { f.wall = 3.0; f.pitch = 1.25; f.lip = 0.75; }
    else if (style == ST_CABIN || style == ST_STILT) {
        f.pitch = style == ST_CABIN ? 1.25 : 1.5; f.lip = 1.0;
        if (style == ST_CABIN) { f.wall = 2.0; if (doorHi) f.ovGHi = 2; else f.ovGLo = 2; }
        else if (doorHi) f.ovHi = 2; else f.ovLo = 2;
    }
    else if (style == ST_CROFT) { f.ovLo = 0; f.ovHi = 0; f.ovGLo = 0; f.ovGHi = 0; f.wall = 2.0; f.pitch = 0.75; f.hip = true; }
    else if (style == ST_ADOBE || style == ST_PUEBLO) { f.flat = true; f.ovLo = 1; f.ovHi = 0; f.ovGLo = 0; f.ovGHi = 0; f.wall = 3.0; }
    return f;
}

// A house column in wall coordinates: iw along the ridge, jw across it (0 at the first wall), and its distance in
// from the nearest long wall (dA) and gable wall (dG); negative over the overhangs.
struct HouseAt { int iw, jw, Lw, Dw, dA, dG; bool axisZ; HouseForm f; };

static inline int4 partIJ(Part P, int2 q, int2 c) {
    int2 r0 = c + P.o;
    bool axisZ = (P.flags & F_AXISZ) != 0;
    return axisZ ? int4(q.y - r0.y, q.x - r0.x, P.sz.y, P.sz.x) : int4(q.x - r0.x, q.y - r0.y, P.sz.x, P.sz.y);
}

static inline void houseAt(Part P, int2 q, int2 c, thread HouseAt& h) {
    int4 ij = partIJ(P, q, c);
    h.axisZ = (P.flags & F_AXISZ) != 0;
    h.f = houseForm((P.flags >> 12) & 15, uint(P.y2));
    h.Lw = ij.z - h.f.ovGLo - h.f.ovGHi; h.Dw = ij.w - h.f.ovLo - h.f.ovHi;
    h.iw = ij.x - h.f.ovGLo; h.jw = ij.y - h.f.ovLo;
    h.dA = min(h.jw, h.Dw - 1 - h.jw); h.dG = min(h.iw, h.Lw - 1 - h.iw);
}

// Roof top over the floor (before quantizing) and its rise per column there, from the eave line half a voxel over
// the wall top; hipped roofs slope toward the gables too.
static inline float2 roofProfile(thread const HouseAt& h) {
    int ridge = (h.Dw - 1) / 2;
    float d = float(h.f.hip ? min(h.dA, h.dG) : h.dA);
    return float2(h.f.wall + 0.5 + h.f.pitch * max(d + 1.0, 0.0), d >= float(ridge) ? 0.0 : h.f.pitch);
}

// The tallest step between neighbouring roof columns: the roof course is at least this deep.
static inline float roofStep(thread const HouseAt& h) { return h.f.pitch; }

static inline float ridgeTop(thread const HouseAt& h) {
    return round((h.f.wall + 0.5 + h.f.pitch * float((h.Dw - 1) / 2 + 1)) * 4.0) * 0.25;
}

// A pitched house's top over the floor at a column, in quarter voxels.
static inline float houseTop(thread const HouseAt& h) {
    if (h.dA < -1) return h.f.wall;
    return round(roofProfile(h).x * 4.0) * 0.25;
}

// The along-ridge column of a house's chimney: in the gable wall, outside it (a cabin's stack) or through the hip.
static inline int chimneyAt(int style, int Lw, uint hh) {
    if (style == ST_CABIN) return (hh & 4u) != 0u ? -1 : Lw;
    int i = style == ST_MEADOW ? 1 : style == ST_TIMBER ? 0 : -100;
    return ((hh >> 12) & 1u) != 0u ? Lw - 1 - i : i;
}

// Height over the floor of an adobe house's mass (wings round a courtyard, U or L) or a pueblo's (a block with a
// second tier set back on its back half); a low wall closes the courtyard, with a gate. 0 outside the mass.
static inline float flatMass(int style, uint hh, int iw, int jw, int Lw, int Dw) {
    if (iw < 0 || jw < 0 || iw >= Lw || jw >= Dw) return 0.0;
    if (style == ST_PUEBLO) return jw < Dw / 2 && iw >= 1 && iw < Lw - 1 - int((hh >> 7) & 1u) ? 5.5 : 3.0;
    bool uPlan = ((hh >> 7) & 1u) != 0u;
    bool court = jw >= 3 && iw >= 3 && (!uPlan || iw <= Lw - 4);
    if (court) {
        int gate = uPlan ? Lw / 2 : (Lw + 2) / 2;
        return jw == Dw - 1 && iw != gate ? 1.5 : 0.0;
    }
    return jw < 3 && ((hh >> 8) & 1u) != 0u ? 4.0 : 3.0;
}

// The wall column a ladder leans on, in wall coordinates (x < 0: none), and the height it starts from: the face of a
// pueblo's second tier toward +j, or an adobe side wing's face toward +i, onto the courtyard.
static inline int3 flatLadder(int style, uint hh, int Lw, int Dw) {
    if (style == ST_PUEBLO) return int3(((hh >> 10) & 1u) != 0u ? 2 : Lw - 4, Dw / 2 - 1, 3);
    if (((hh >> 10) & 1u) == 0u) return int3(-1);
    return int3(2, Dw - 2, 0);
}

// The mass with a parapet round its roof edges, lower at the corners.
static inline float flatTop(int style, uint hh, int iw, int jw, int Lw, int Dw) {
    float m = flatMass(style, hh, iw, jw, Lw, Dw);
    if (m < 2.0) return m;
    bool ex = flatMass(style, hh, iw - 1, jw, Lw, Dw) < m || flatMass(style, hh, iw + 1, jw, Lw, Dw) < m;
    bool ez = flatMass(style, hh, iw, jw - 1, Lw, Dw) < m || flatMass(style, hh, iw, jw + 1, Lw, Dw) < m;
    return m + (ex && ez ? 0.25 : ex || ez ? 0.5 : 0.0);
}

// What a house wall column shows on its face toward `side` (0 -x, 1 +x, 2 -z, 3 +z): 0 plain wall, 1 a window, 2 the
// door, 3 a window on a pueblo's upper tier.
static inline int houseOpening(Part P, thread const HouseAt& h, uint side) {
    uint hh = uint(P.y2);
    int style = (P.flags >> 12) & 15;
    uint sLo = h.axisZ ? 0u : 2u, sHi = h.axisZ ? 1u : 3u, gLo = h.axisZ ? 2u : 0u, gHi = h.axisZ ? 3u : 1u;
    int di = side == gLo ? -1 : side == gHi ? 1 : 0, dj = side == sLo ? -1 : side == sHi ? 1 : 0;
    float k = hashf(pcg(hh * 131u + uint(h.iw + 8) * 17u + uint(h.jw + 8) * 5u + side));
    if (h.f.flat) {
        float m = flatMass(style, hh, h.iw, h.jw, h.Lw, h.Dw);
        float o = flatMass(style, hh, h.iw + di, h.jw + dj, h.Lw, h.Dw);
        if (m < 2.0 || o >= m) return 0;
        if (o >= 2.0) return k < 0.5 ? 3 : 0;
        bool doorCol = style == ST_PUEBLO ? h.iw == h.Lw / 2 && h.jw == h.Dw - 1 : h.iw == h.Lw / 2 && h.jw == 2;
        if (doorCol && dj == 1) return 2;
        // Corners stay plain.
        if (flatMass(style, hh, h.iw + dj, h.jw + di, h.Lw, h.Dw) < 2.0 || flatMass(style, hh, h.iw - dj, h.jw - di, h.Lw, h.Dw) < 2.0) return 0;
        return k < 0.3 ? 1 : 0;
    }
    if (h.iw < 0 || h.jw < 0 || h.iw >= h.Lw || h.jw >= h.Dw) return 0;
    if (dj != 0) {
        if (!(dj < 0 ? h.jw == 0 : h.jw == h.Dw - 1) || h.iw == 0 || h.iw == h.Lw - 1) return 0;
        bool doorWall = (dj > 0) == ((hh & 4u) != 0u) && style != ST_CABIN;
        if (doorWall && h.iw == h.Lw / 2) return 2;
        if (((h.iw - h.Lw / 2) & 1) != 0) return 0;
        return k < (style == ST_CABIN ? 0.95 : 0.7) ? 1 : 0;
    }
    if (!(di < 0 ? h.iw == 0 : h.iw == h.Lw - 1)) return 0;
    // A cabin's front gable: the door between two windows.
    if (style == ST_CABIN) {
        if ((di > 0) != ((hh & 4u) != 0u)) return 0;
        int dr = abs(h.jw - (h.Dw - 1) / 2);
        return dr == 0 ? 2 : dr == 1 ? 1 : 0;
    }
    return h.jw == (h.Dw - 1) / 2 && k < 0.6 ? 1 : 0;
}

// Wall coordinates (iw along the ridge, jw across) to a column offset from the site centre.
static inline int2 wallCol(Part P, thread const HouseForm& f, int iw, int jw) {
    bool axisZ = (P.flags & F_AXISZ) != 0;
    int2 w0 = P.o + (axisZ ? int2(f.ovLo, f.ovGLo) : int2(f.ovGLo, f.ovLo));
    return w0 + (axisZ ? int2(jw, iw) : int2(iw, jw));
}

// The point just outside a dwelling's door (offset from the site centre, z 1 if it has one): where its lamp spills
// light, or with `path` where the path from the hearth arrives (an adobe's courtyard gate, past a porch).
static inline float3 houseDoor(Part P, uint sh, int k, bool path) {
    int style = (P.flags >> 12) & 15;
    if (P.kind == P_ROUND) {
        if (style != ST_YURT) return 0;
        float da = hashf(pcg(sh + uint(k))) * 6.2831853;
        return float3(float2(P.o) + 0.5 + float2(cos(da), sin(da)) * (float(P.sz.x) * 0.5 + 1.0), 1.0);
    }
    if (P.kind != P_HOUSE || (P.flags & F_TENT) != 0) return 0;
    uint hh = uint(P.y2);
    HouseForm f = houseForm(style, hh);
    bool axisZ = (P.flags & F_AXISZ) != 0;
    int4 ij = axisZ ? int4(0, 0, P.sz.y, P.sz.x) : int4(0, 0, P.sz.x, P.sz.y);
    int Lw = ij.z - f.ovGLo - f.ovGHi, Dw = ij.w - f.ovLo - f.ovHi;
    int iw = Lw / 2, jw, out;
    if (f.flat) {
        uint uPlan = (hh >> 7) & 1u;
        if (style == ST_ADOBE && !path) { jw = 2; out = 1; }
        else { jw = Dw - 1; out = 1; if (style == ST_ADOBE) iw = uPlan != 0u ? Lw / 2 : (Lw + 2) / 2; }
    } else if (style == ST_CABIN) {
        bool hi = (hh & 4u) != 0u;
        jw = (Dw - 1) / 2;
        int oi = (hi ? 1 : -1) * (path ? (hi ? f.ovGHi : f.ovGLo) + 1 : 1);
        return float3(float2(wallCol(P, f, (hi ? Lw - 1 : 0) + oi, jw)) + 0.5, 1.0);
    } else {
        bool hi = (hh & 4u) != 0u;
        jw = hi ? Dw - 1 : 0;
        int ov = hi ? f.ovHi : f.ovLo;
        out = (hi ? 1 : -1) * (path ? ov + 1 : 1);
        if (style == ST_STILT && path) return 0;
    }
    int2 cc = wallCol(P, f, iw, jw + out);
    return float3(float2(cc) + 0.5, 1.0);
}

static inline bool rectFree(thread const int4* rects, int nr, int2 r0, int2 sz, int gap) {
    for (int m = 0; m < nr; m++)
        if (all(r0 < rects[m].zw + gap) && all(r0 + sz + gap > rects[m].xy)) return false;
    return true;
}

// Lowest and highest ground over a rect; false where an arch stands in it.
static inline bool rectGround(texture2d<float, access::read> hm, int2 r0, int2 sz, thread float& lo, thread float& hi, thread int& wet) {
    lo = 1e9; hi = -1e9; wet = 0;
    for (int j = 0; j < sz.y; j++)
        for (int i = 0; i < sz.x; i++) {
            float4 col = hm.read(uint2(r0 + int2(i, j)));
            if (col.b > col.g) return false;
            lo = min(lo, col.r); hi = max(hi, col.r);
            if (col.r < SEA) wet++;
        }
    return true;
}

// Houses on flat treads around a central hearth, in the local style: cottages, timber frames, cabins, crofts,
// adobe courtyards, pueblos, stilt longhouses or yurts.
static bool buildHamlet(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, uint h, int style) {
    int n = int(hm.get_width());
    int4 rects[8];
    int nr = 0;
    rects[nr++] = int4(c - 2, c + 3);
    int houses = 0, want = 3 + int(h % 3u);
    float a0 = hashf(pcg(h + 1u)) * 6.2831853;
    bool stilt = style == ST_STILT, pueblo = style == ST_PUEBLO, yurt = style == ST_YURT;
    float rIn = stilt || pueblo ? 5.0 : 8.0, rSpan = pueblo ? 12.0 : 26.0;
    for (int k = 0; k < 64 && houses < want && G.np < MAX_PARTS - 8 && G.nl < MAX_LAMPS - 3; k++) {
        float r = rIn + rSpan * sqrt(float(k) / 64.0);
        float a = a0 + float(k) * 2.39996;
        int2 pos = c + int2(round(r * float2(cos(a), sin(a))));
        uint hh = pcg(h + 17u * uint(k) + 3u) & 0x7fffu;
        if (style == ST_CABIN) {
            // A cabin turns its porch gable to the hearth.
            int2 dh = c - pos;
            bool az = abs(dh.y) > abs(dh.x);
            hh = (hh & ~5u) | (az ? 1u : 0u) | ((az ? dh.y : dh.x) > 0 ? 4u : 0u);
        }
        bool axisZ = (hh & 1u) != 0u;
        int b9 = int((hh >> 9) % 3u), b11 = int((hh >> 11) & 1u);
        if (yurt) {
            int R2 = 5 + b9, rad = (R2 + 1) / 2;
            int2 r0 = pos - rad, sz = int2(2 * rad + 1);
            if (any(r0 < 2) || any(r0 + sz >= n - 2) || !rectFree(rects, nr, r0, sz, 2)) continue;
            float lo, hi;
            int wet;
            if (!rectGround(hm, r0, sz, lo, hi, wet) || hi - lo > 1.0 || lo < SEA + 2.2) continue;
            int floorY = int(hi);
            Part yp = makePart(P_ROUND, M_FELT, pos - c, int2(R2, 0), floorY, 8, 14, F_CHIMNEY | (style << 12));
            float2 lxz = floor(houseDoor(yp, uint(G.s.head.w), G.np, false).xy) + 0.5;
            float rBefore = G.r;
            int lamp = addLamp(G, float3(lxz.x, float(floorY) + 1.2, lxz.y), L_WINDOW, 2.4, pos - c - int2(floor(lxz)));
            G.r = max(rBefore, length(float2(pos - c) + 0.5) + 2.0);
            yp.flags |= (lamp + 1) << 8;
            addPart(G, yp, length(float2(pos - c)) + float(rad) + 2.0);
            rects[nr++] = int4(r0, r0 + sz);
            houses++;
            continue;
        }
        HouseForm f = houseForm(style, hh);
        int Lw = 6 + (b9 & 1), Dw = 4 + b11;
        if (style == ST_TIMBER) { Lw = 7 + (b9 & 1); Dw = 5; }
        else if (style == ST_CABIN) { Lw = 5 + (b9 & 1); Dw = 5; }
        else if (style == ST_CROFT) Lw = 7 + b9;
        else if (stilt) Lw = 8 + b9;
        else if (style == ST_ADOBE) { Lw = 7 + b9; Dw = 7 + b11; }
        else if (pueblo) { Lw = 6 + b9; Dw = 5 + b11; }
        int L = Lw + f.ovGLo + f.ovGHi, Dt = Dw + f.ovLo + f.ovHi;
        int2 sz = axisZ ? int2(Dt, L) : int2(L, Dt);
        int2 r0 = pos - sz / 2;
        if (any(r0 < 2) || any(r0 + sz >= n - 2) || !rectFree(rects, nr, r0, sz, pueblo ? 1 : 2)) continue;
        int2 w0 = r0 + (axisZ ? int2(f.ovLo, f.ovGLo) : int2(f.ovGLo, f.ovLo));
        int2 wsz = axisZ ? int2(Dw, Lw) : int2(Lw, Dw);
        float lo, hi, alo, ahi;
        int wet, awet;
        if (!rectGround(hm, r0, sz, alo, ahi, awet) || !rectGround(hm, w0, wsz, lo, hi, wet)) continue;
        if (stilt) { if (ahi > SEA + 1.2 || alo < SEA - 6.0 || awet * 3 < sz.x * sz.y) continue; }
        else if (hi - lo > (f.flat ? 1.5 : 2.0) || ahi > hi + 1.0 || lo < SEA + 2.2) continue;
        int floorY = stilt ? 23 : int(hi);
        HouseAt ha;
        ha.f = f; ha.Lw = Lw; ha.Dw = Dw;
        bool chim = !f.flat && !stilt && (style == ST_CABIN || style == ST_CROFT || ((hh >> 3) & 1u) != 0u);
        float top = f.flat ? (pueblo ? 6.0 : 4.5) : ridgeTop(ha);
        int flags = (axisZ ? F_AXISZ : 0) | (stilt ? F_STILT | F_WATER : 0) | (chim ? F_CHIMNEY : 0) | (style << 12);
        int wall = style == ST_CROFT ? M_STONE : f.flat ? M_ADOBE : style == ST_CABIN || stilt ? M_LOG : M_PLASTER;
        Part hp = makePart(P_HOUSE, wall, r0 - c, sz, floorY, 0, int(hh), flags);
        // The lamp at the door; smoke from the chimney stack, or a croft's roof hole over the house centre.
        float2 lxz = floor(houseDoor(hp, 0u, 0, false).xy) + 0.5;
        int ci = chimneyAt(style, Lw, hh);
        int2 smokeCol = ci > -50 ? wallCol(hp, f, ci, (Dw - 1) / 2) : pos - c;
        float rBefore = G.r;
        int lamp = addLamp(G, float3(lxz.x, float(floorY) + 1.2, lxz.y), L_WINDOW, chim ? top + 1.0 - 1.2 : 0.0,
                           smokeCol - int2(floor(lxz)));
        G.r = max(rBefore, length(float2(pos - c) + 0.5) + 2.0);
        hp.flags |= (lamp + 1) << 8;
        addPart(G, hp, float(partReach(hp)));
        rects[nr++] = int4(r0, r0 + sz);
        houses++;
    }
    return houses >= 3;
}

// The ground a part takes, offset from the site centre (min, max exclusive); empty for parts that take none.
static inline int4 partRect(Part P) {
    if (P.kind == P_ROUND) { int rad = (P.sz.x + 1) / 2; return int4(P.o - rad, P.o + rad + 1); }
    if (P.kind == P_HOUSE || P.kind == P_DECK || P.kind == P_BOAT || P.kind == P_PROP || P.kind == P_BOX || P.kind == P_FIRE)
        return int4(P.o, P.o + P.sz);
    return int4(0);
}

// A prop at offset o if its ground is free (`gap` columns from every other part), dry and level, and within 1.5 of
// `ref`; wet props stand in the water off a stilt house, on floor `ref`.
static bool tryProp(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, int2 o, int2 sz, int type, int mat,
                    int style, float ref, bool wet, int gap, uint ph) {
    if (G.np >= MAX_PARTS) return false;
    int n = int(hm.get_width());
    int2 r0 = c + o;
    if (any(r0 < 2) || any(r0 + sz >= n - 2)) return false;
    for (int k = 0; k < G.np; k++) {
        int4 r = partRect(unpackPart(G.s.parts[k]));
        if (r.z > r.x && all(o < r.zw + gap) && all(o + sz + gap > r.xy)) return false;
    }
    float lo, hi;
    int nw;
    if (!rectGround(hm, r0, sz, lo, hi, nw)) return false;
    int fl;
    if (wet) {
        if (nw != sz.x * sz.y || lo < SEA - 6.0) return false;
        fl = int(ref);
    } else {
        if (nw > 0 || hi - lo > 1.0 || lo < SEA + 1.2 || abs(hi - ref) > 1.5) return false;
        fl = int(ceil(hi - 0.01));
    }
    Part p = makePart(P_PROP, mat, o, sz, fl, type, int(ph & 0x7fffu), (style << 12) | (wet ? F_WATER : 0));
    addPart(G, p, float(partReach(p)) + 1.0);
    return true;
}

// A prop somewhere on a ring round the site centre, lying along the ring.
static bool ringProp(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, int type, int mat, int style, float ref,
                     int Lp, int Dp, float r0, uint ph) {
    float a0 = hashf(ph) * 6.2831853;
    for (int t = 0; t < 12; t++) {
        float a = a0 + float(t) * 2.39996, r = r0 + float(t % 3);
        float2 dir = float2(cos(a), sin(a));
        int2 sz = abs(dir.x) > abs(dir.y) ? int2(Dp, Lp) : int2(Lp, Dp);
        int2 o = int2(round(dir * r)) - sz / 2;
        if (tryProp(hm, G, c, o, sz, type, mat, style, ref, false, 1, pcg(ph + uint(t)))) return true;
    }
    return false;
}

// Seats round a fire's ring of stones, a column clear of it.
static void fireSeats(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int style, uint h, int want) {
    const int2 so[4] = { int2(3, -1), int2(-3, -1), int2(-1, 3), int2(-1, -3) };
    int got = 0;
    for (int t = 0; t < 4 && got < want; t++) {
        int m = int((h + uint(t)) & 3u);
        int2 sz = m < 2 ? int2(1, 3) : int2(3, 1);
        if (tryProp(hm, G, c, so[m], sz, PR_LOGSEAT, M_LOG, style, g, false, 1, pcg(h + uint(t) * 5u))) got++;
    }
}

// What a dwelling keeps round it, by style: (type, length, width, material); a wet prop stands in the water.
struct PropKind { int type, Lp, Dp, mat; bool wet; };

static inline PropKind propFor(int style, uint pick) {
    PropKind k;
    k.wet = false;
    int t;
    if (style == ST_MEADOW) { const int m[4] = { PR_GARDEN, PR_WASH, PR_HAY, PR_BARREL }; t = m[pick % 4u]; }
    else if (style == ST_TIMBER) { const int m[4] = { PR_WOODPILE, PR_GARDEN, PR_WASH, PR_BARREL }; t = m[pick % 4u]; }
    else if (style == ST_CABIN) { const int m[3] = { PR_WOODPILE, PR_STUMP, PR_BARREL }; t = m[pick % 3u]; }
    else if (style == ST_CROFT) { const int m[3] = { PR_PEN, PR_PEAT, PR_HAY }; t = m[pick % 3u]; }
    else if (style == ST_ADOBE || style == ST_PUEBLO) { const int m[2] = { PR_OVEN, PR_RACK }; t = m[pick % 2u]; }
    else if (style == ST_STILT) { const int m[2] = { PR_CANOE, PR_RACK }; t = m[pick % 2u]; k.wet = true; }
    else { const int m[3] = { PR_PEN, PR_BALES, PR_RUG }; t = m[pick % 3u]; }
    k.type = t;
    k.Lp = t == PR_GARDEN ? 5 : t == PR_PEN ? (style == ST_YURT ? 7 : 6) : t == PR_WASH ? 4 : t == PR_CANOE ? 5
         : t == PR_HAY || t == PR_OVEN || t == PR_WOODPILE || t == PR_RACK || t == PR_RUG ? 3 : t == PR_STUMP || t == PR_BARREL ? 1 : 2;
    k.Dp = t == PR_GARDEN ? 4 : t == PR_PEN ? 5 : t == PR_HAY || t == PR_OVEN ? 3 : t == PR_RUG ? 2 : 1;
    k.mat = t == PR_WOODPILE || t == PR_STUMP ? M_LOG : t == PR_HAY || t == PR_BALES ? M_THATCH : t == PR_OVEN ? M_ADOBE
          : t == PR_PEN ? (style == ST_CROFT ? M_STONE : M_TIMBER) : t == PR_WASH || t == PR_RACK || t == PR_RUG ? M_CLOTH
          : t == PR_GARDEN ? M_PLANK : M_TIMBER;
    return k;
}

// Each dwelling's yard: up to two props by its gables, back or front corners (never before the door), then seats by
// the hearth. Added after every other part, so nothing the hamlet had before moves.
static void hamletProps(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int style, uint hb, bool hearth) {
    int nh = G.np;
    for (int k = 0; k < nh; k++) {
        Part H = unpackPart(G.s.parts[k]);
        bool round = H.kind == P_ROUND;
        if (!(H.kind == P_HOUSE && (H.flags & F_TENT) == 0) && !round) continue;
        uint hh = pcg(hb + uint(k) * 101u);
        int fl = H.y0;
        float3 door = houseDoor(H, uint(G.s.head.w), k, true);
        int2 dc = int2(floor(door.xy));
        float da = hashf(pcg(uint(G.s.head.w) + uint(k))) * 6.2831853;
        for (int pi = 0; pi < 2; pi++) {
            PropKind pk = propFor(style, hh >> (pi * 5));
            if (pi == 1 && pk.type == propFor(style, hh).type) continue;
            float ref = pk.wet ? (pk.type == PR_CANOE ? SEA : float(fl)) : float(fl);
            int gap = pk.type == PR_WOODPILE || pk.type == PR_PEAT || pk.type == PR_BARREL ? 0 : 1;
            for (int t = 0; t < 4; t++) {
                int pos = int((hh >> 11) + uint(t) + uint(pi) * 2u) & 3;
                int2 o, sz;
                if (round) {
                    int rad = (H.sz.x + 1) / 2;
                    int2 d = pos == 0 ? int2(1, 0) : pos == 1 ? int2(-1, 0) : pos == 2 ? int2(0, 1) : int2(0, -1);
                    if (dot(float2(d), float2(cos(da), sin(da))) > 0.7) continue;
                    sz = d.x != 0 ? int2(pk.Dp, pk.Lp) : int2(pk.Lp, pk.Dp);
                    int2 cc = H.o + d * (rad + 2 + (d.x != 0 ? sz.x : sz.y) / 2);
                    o = cc - sz / 2;
                } else {
                    bool axisZ = (H.flags & F_AXISZ) != 0;
                    int L = axisZ ? H.sz.y : H.sz.x, Dt = axisZ ? H.sz.x : H.sz.y;
                    bool doorHi = (uint(H.y2) & 4u) != 0u || houseForm((H.flags >> 12) & 15, uint(H.y2)).flat;
                    int li, lj, i0, j0;
                    if (pos < 2) { li = pk.Dp; lj = pk.Lp; i0 = pos == 0 ? -li - gap : L + gap; j0 = (Dt - lj) / 2; }
                    else {
                        li = pk.Lp; lj = pk.Dp;
                        bool back = pos == 2;
                        j0 = back == doorHi ? -lj - gap : Dt + gap;
                        i0 = back ? (L - li) / 2 : ((hh >> 14) & 1u) != 0u ? 0 : L - li;
                    }
                    o = H.o + (axisZ ? int2(j0, i0) : int2(i0, j0));
                    sz = axisZ ? int2(lj, li) : int2(li, lj);
                }
                // Nothing stands at or beside the spot outside the door.
                if (door.z > 0.0 && all(dc >= o - 1) && all(dc <= o + sz)) continue;
                if (tryProp(hm, G, c, o, sz, pk.type, pk.mat, style, ref, pk.wet, pk.wet ? 1 : gap, pcg(hh + uint(t) * 3u + uint(pi)))) break;
            }
        }
    }
    if (hearth && style != ST_STILT && style != ST_ADOBE && style != ST_PUEBLO) fireSeats(hm, G, c, g, style, hb >> 3, 2);
}

// A camp's seats round its fire, a woodpile, and its style's gear.
static void campProps(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int cst, uint hb) {
    fireSeats(hm, G, c, g, cst, hb >> 6, 2);
    ringProp(hm, G, c, PR_WOODPILE, M_LOG, cst, g, 2, 1, 5.0, pcg(hb + 61u));
    if (cst == ST_AFRAME || cst == ST_LEANTO) ringProp(hm, G, c, PR_BEDROLL, M_CLOTH, cst, g, 2, 1, 4.0, pcg(hb + 62u));
    else if (cst == ST_TIPI) ringProp(hm, G, c, PR_RACK, M_CLOTH, cst, g, 3, 1, 5.0, pcg(hb + 63u));
    else if (cst == ST_CARAVAN) {
        ringProp(hm, G, c, PR_RUG, M_CLOTH, cst, g, 3, 2, 4.0, pcg(hb + 64u));
        ringProp(hm, G, c, PR_BALES, M_THATCH, cst, g, 2, 1, 5.0, pcg(hb + 65u));
    } else if (cst == ST_PALM) {
        ringProp(hm, G, c, PR_CANOE, M_TIMBER, cst, g, 4, 1, 5.0, pcg(hb + 66u));
        ringProp(hm, G, c, PR_RACK, M_CLOTH, cst, g, 3, 1, 5.0, pcg(hb + 67u));
    }
}

// The lie of the land at a candidate site centre, and how well each type suits it.
struct Lie {
    float g, desert, painted, lagoon, salt, cold, snowline, waterR, prom, autumn, conifer, moist, grove;
    int2 waterDir;
    bool flat5;
    float score[7];
};

static Lie survey(texture2d<float, access::read> hm, int2 c, int2 cw, uint seed) {
    Lie L;
    for (int t = 0; t < 7; t++) L.score[t] = 0;
    L.grove = 0;
    L.g = colH(hm, c);
    float g = L.g;
    float2 w = float2(cw);
    float2 p = w + seedOffset(seed);
    Climate cl = climate(w, seed);
    L.desert = desertAmount(cl); L.painted = paintedAmount(cl); L.lagoon = lagoonAmount(cl); L.salt = saltAmount(cl);
    L.cold = smoothstep(0.42, 0.18, cl.temp);
    L.autumn = autumnAmount(cl); L.conifer = coniferAmount(cl); L.moist = cl.moist;
    L.snowline = snowlineAt(cl.temp);
    float settled = fbm2(p / 1500.0, 2, seed + 140u) + 0.5;
    float ancient = fbm2(p / 1800.0 + 77.0, 2, seed + 141u) + 0.5;
    float lo5, hi5, lo9, hi9;
    L.flat5 = flatPatch(hm, c, 2, 1.0, lo5, hi5);
    bool flat9 = flatPatch(hm, c, 4, 3.0, lo9, hi9);
    // Ring scan: nearest water and its bearing, prominence over the surroundings, how much of a bay this is.
    L.waterR = 1e9;
    L.waterDir = int2(0);
    float ring = 0, ringN = 0, wet16 = 0, max6 = -1e9;
    const float rads[4] = { 8.0, 16.0, 24.0, 40.0 };
    for (int ri = 0; ri < 4; ri++)
        for (int k = 0; k < 16; k++) {
            float a = float(k) * 0.3926991;
            float2 dv = float2(cos(a), sin(a));
            float hq = colH(hm, c + int2(round(dv * rads[ri])));
            if (hq < SEA && rads[ri] < L.waterR) {
                L.waterR = rads[ri];
                L.waterDir = abs(dv.x) > abs(dv.y) ? int2(dv.x > 0 ? 1 : -1, 0) : int2(0, dv.y > 0 ? 1 : -1);
            }
            if (ri >= 2) { ring += hq; ringN += 1; }
            if (ri == 1 && hq < SEA) wet16 += 1.0 / 16.0;
        }
    float min6 = 1e9;
    for (int k = 0; k < 8; k++) {
        float a = float(k) * 0.785398;
        float h6 = colH(hm, c + int2(round(float2(cos(a), sin(a)) * 6.0)));
        max6 = max(max6, h6);
        min6 = min(min6, h6);
    }
    // Nothing is built within 6 of a sheer drop of more than 40 (a pillar top's or an escarpment's edge).
    if (g - min6 > 40.0) return L;
    L.prom = g - ring / ringN;
    // A tread with a riser close by: a terrace.
    float riser = 0;
    for (int k = 0; k < 8; k++) {
        float a = float(k) * 0.785398;
        for (int r = 3; r <= 9; r += 3) riser = max(riser, colH(hm, c + int2(round(float2(cos(a), sin(a)) * float(r)))) - g);
    }
    float land = step(SEA + 2.0, g);
    float notDry = (1.0 - smoothstep(0.3, 0.5, L.desert)) * (1.0 - smoothstep(0.2, 0.4, L.salt)) * (1.0 - smoothstep(0.3, 0.5, L.lagoon));
    float anc = smoothstep(0.35, 0.6, ancient), set = smoothstep(0.35, 0.6, settled);
    float prom = L.prom, salt = L.salt;
    L.score[T_HENGE] = anc * smoothstep(1.0, 5.0, prom) * smoothstep(45.0, 30.0, prom) * float(flat9) * notDry
                     * step(g, L.snowline + 10.0) * land;
    L.score[T_CAIRN] = anc * step(max6, g) * max(smoothstep(3.0, 9.0, prom), smoothstep(0.3, 0.6, salt)) * land * 0.85;
    L.score[T_RUINS] = anc * float(L.flat5) * land * max(smoothstep(3.0, 6.0, riser) * (1.0 - smoothstep(0.3, 0.5, salt)),
                                                         smoothstep(0.35, 0.6, L.desert) * smoothstep(3.0, 8.0, prom));
    L.score[T_HAMLET] = set * step(L.waterR, 40.0) * float(L.flat5) * step(SEA + 2.0, g) * step(g, SEA + 30.0)
                      * (1.0 - smoothstep(0.2, 0.4, salt)) * step(g, L.snowline - 10.0);
    L.score[T_JETTY] = set * step(L.waterR, 16.0) * smoothstep(0.1, 0.25, wet16) * smoothstep(0.75, 0.55, wet16)
                     * step(g, SEA + 6.0) * (1.0 - smoothstep(0.2, 0.4, salt)) * 0.9;
    float grove = groveDensity(w, g, seed);
    L.grove = grove;
    L.score[T_CAMP] = (0.4 + 0.6 * set) * float(L.flat5) * land * step(g, L.snowline - 5.0)
                    * max(max(smoothstep(0.08, 0.2, grove), smoothstep(0.15, 0.3, salt) * smoothstep(0.7, 0.45, salt)),
                          smoothstep(0.3, 0.6, L.painted)) * 0.8;
    return L;
}

// Open dry grass: no trees, not desert, painted hills or salt.
static inline bool steppe(thread const Lie& L) {
    return L.moist < 0.45 && L.grove < 0.1 && L.desert < 0.2 && L.painted < 0.2 && L.salt < 0.15 && L.lagoon < 0.2;
}

static inline int hamletStyle(thread const Lie& L) {
    if (L.lagoon > 0.4) return ST_STILT;
    if (L.desert > 0.3) return ST_ADOBE;
    if (L.painted > 0.35) return ST_PUEBLO;
    if (L.cold > 0.5 || L.g > L.snowline - 30.0) return L.grove > 0.15 ? ST_CABIN : ST_CROFT;
    if (L.conifer > 0.5) return ST_CABIN;
    if (L.autumn > 0.3) return ST_TIMBER;
    if (steppe(L)) return ST_YURT;
    return ST_MEADOW;
}

static inline int campStyle(thread const Lie& L) {
    if (L.lagoon > 0.3 && L.autumn < 0.3) return ST_PALM;
    if (L.painted > 0.35) return ST_TIPI;
    if ((L.salt > 0.15 && L.salt < 0.7) || steppe(L) || L.desert > 0.3) return ST_CARAVAN;
    if (L.conifer > 0.5) return ST_LEANTO;
    return ST_AFRAME;
}

// Site at one world cell: a type chosen by two broad Perlin fields (where people live now, and where an older
// culture built) and by the lie of the land at the best of nine candidate centres, then its parts.
kernel void placeSites(texture2d<float, access::read> hm [[texture(0)]],
                       texture2d<uint, access::read> gx [[texture(1)]],
                       device Site* sites [[buffer(0)]],
                       constant int4& info [[buffer(1)]],
                       device const GiantRec* giants [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(SITE_GRID) || gid.y >= uint(SITE_GRID)) return;
    int n = info.z;
    uint seed = uint(info.w);
    int2 cell = siteCell0(info.xy) + int2(gid);
    uint h = hash2i(cell, seed + 200u);
    SiteGen G;
    G.np = 0; G.nl = 0; G.r = 0;
    G.s.head = int4(cell * SITE_CELL + 64, 0, int(h));
    G.s.info = float4(0);
    for (int i = 0; i < MAX_LAMPS; i++) G.s.lamps[i] = float4(0);
    for (int i = 0; i < MAX_PARTS; i++) G.s.parts[i] = int4(0);
    uint idx = gid.y * uint(SITE_GRID) + gid.x;

    const float thresh[7] = { 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 };
    // Chance a type is built where it is the best fit; these set the density of each type.
    const float accept[7] = { 0, 0.25, 0.7, 0.55, 0.22, 0.14, 0.16 };
    float best[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int2 at[7];
    // Only cells whose every candidate, survey and part (64 out) reads lie inside the map, so a cell decides the
    // same way on every map that holds it.
    int2 cl = cell * SITE_CELL - info.xy;
    if (any(cl < 48) || any(cl > n - 176)) { sites[idx] = G.s; return; }
    if (towerNear(float2(cell * SITE_CELL + 64), 128.0, seed)) { sites[idx] = G.s; return; }
    for (int k = 0; k < 9; k++) {
        uint hk = pcg(h + uint(k) * 7919u);
        int2 ck = cell * SITE_CELL + 16 + int2(k % 3, k / 3) * 32 + int2(int(hk % 32u), int((hk >> 5) % 32u));
        int2 lc = ck - info.xy;
        if (colArch(hm, lc)) continue;
        // Giants are placed first: a site keeps clear of a giant's reach by more than its own.
        Giant gk;
        if (giantAt(giants, float2(ck) + 0.5, 60.0, gk)) continue;
        Lie L = survey(hm, lc, ck, seed);
        for (int t = 1; t <= 6; t++)
            if (L.score[t] >= thresh[t] && L.score[t] > best[t]) { best[t] = L.score[t]; at[t] = ck; }
    }
    // Types in order of fit, each with its own chance.
    int type = 0;
    int2 cw = int2(0);
    for (int pass = 0; pass < 6 && type == 0; pass++) {
        int bt = 0;
        for (int t = 1; t <= 6; t++) if (best[t] > 0.0 && (bt == 0 || best[t] > best[bt])) bt = t;
        if (bt == 0) break;
        if (hashf(pcg(h + 99u + uint(bt) * 13u)) < accept[bt]) { type = bt; cw = at[bt]; }
        best[bt] = 0;
    }
    if (type == 0) { sites[idx] = G.s; return; }
    int2 c = cw - info.xy;
    G.s.head.xy = cw;
    Lie L = survey(hm, c, cw, seed);
    float g = L.g, snowline = L.snowline, waterR = L.waterR;
    int2 waterDir = L.waterDir;
    bool flat5 = L.flat5;

    uint hb = pcg(h + 7u);
    bool ok = false, hearth = false;
    float age = 0.5;
    if (type == T_HENGE) {
        int R = 5 + int(hb % 4u);
        int N = min(7 + int((hb >> 4) % 7u), int(float(R) * 1.6));
        Part hp = makePart(P_HENGE, M_STONE, int2(0), int2(0), R * 4, N, int(hb & 0x7fffu), 0);
        if (((hb >> 12) & 7u) < 3u) {
            int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_RITUAL, 1.0);
            addPart(G, makePart(P_FIRE, M_FIRE, int2(0), int2(1), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 2.0);
        }
        addPart(G, hp, float(R) + 3.0);
        // A heel stone outside the ring.
        if (((hb >> 16) & 1u) != 0u) {
            float a = hashf(pcg(hb)) * 6.2831853;
            int2 o = int2(round(float2(cos(a), sin(a)) * float(R + 7)));
            if (!colArch(hm, c + o) && colH(hm, c + o) > SEA + 2.0)
                addPart(G, makePart(P_BOX, M_STONE, o, int2(1), 4, 0, 0, F_REL), float(R + 9));
        }
        age = 0.85 + 0.15 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_CAIRN) {
        addPart(G, makePart(P_BOX, M_STONE, int2(0), int2(1), 2, 0, 0, F_REL), 2.0);
        addPart(G, makePart(P_BOX, M_STONE, int2(-1), int2(3), 1, 0, 0, F_REL), 3.0);
        age = 0.8 + 0.2 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_RUINS) {
        float2 gr = float2(colH(hm, c + int2(6, 0)) - colH(hm, c - int2(6, 0)), colH(hm, c + int2(0, 6)) - colH(hm, c - int2(0, 6)));
        float ga = length(gr) > 0.5 ? atan2(gr.y, gr.x) : hashf(hb) * 6.2831853;
        float2 up = float2(cos(ga), sin(ga)), along = float2(-up.y, up.x);
        int runs = 2 + int(hb % 3u);
        for (int i = 0; i < runs; i++) {
            uint rh = pcg(hb + uint(i) * 31u);
            float off = (float(i) - float(runs - 1) * 0.5) * 5.0 + (hashf(rh) - 0.5) * 3.0;
            float len = 6.0 + hashf(rh >> 4) * 10.0;
            float shift = (hashf(rh >> 9) - 0.5) * 8.0;
            int2 a = int2(round(up * off + along * (shift - len * 0.5)));
            int2 b = int2(round(up * off + along * (shift + len * 0.5)));
            addPart(G, makePart(P_WALL, M_STONE, a, int2(0), b.x, b.y, 4, int(rh & 0xfu) << 12), length(float2(max(abs(a), abs(b)))) + 2.0);
        }
        if (((hb >> 8) & 1u) != 0u) {
            int2 tc = int2(round(along * (12.0 + hashf(hb >> 10) * 6.0) * (((hb >> 9) & 1u) != 0u ? 1.0 : -1.0)));
            float tl, th;
            if (flatPatch(hm, c + tc, 3, 3.0, tl, th) && tl > SEA + 2.0) {
                addPart(G, makePart(P_RING, M_STONE, tc, int2(2, 3), 0, int(hb >> 14 & 0x7fffu), 6, 0), length(float2(tc)) + 5.0);
                int2 rb = tc + int2(round(-up * 4.0));
                int2 re = tc + int2(round(-up * (8.0 + hashf(hb >> 20) * 4.0)));
                addPart(G, makePart(P_WALL, M_STONE, rb, int2(0), re.x, re.y, 1, 0), length(float2(max(abs(rb), abs(re)))) + 2.0);
            }
        }
        age = 0.6 + 0.3 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_HAMLET) {
        int style = hamletStyle(L);
        ok = buildHamlet(hm, G, c, hb, style);
        hearth = ok && float(flat5) > 0.0 && g < snowline;
        if (hearth) {
            int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_HEARTH, 1.0);
            addPart(G, makePart(P_FIRE, M_FIRE, int2(-1), int2(3), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 3.0);
        }
        if (ok && waterR <= 40.0 && style != ST_STILT) buildJetty(hm, G, c, pcg(hb + 5u), waterDir, 48);
        if (ok && ((hb >> 20) & 3u) == 0u && G.nl < MAX_LAMPS) {
            // A beacon on the highest flat ground nearby.
            float bestH = g + 8.0;
            int2 bo = int2(0);
            for (int ri = 1; ri <= 3; ri++)
                for (int k = 0; k < 16; k++) {
                    float a = float(k) * 0.3926991;
                    int2 o = int2(round(float2(cos(a), sin(a)) * float(ri * 18)));
                    float bl, bh;
                    float hq = colH(hm, c + o);
                    if (hq > bestH && flatPatch(hm, c + o, 1, 0.0, bl, bh)) { bestH = hq; bo = o; }
                }
            if (any(bo != 0)) {
                int lamp = addLamp(G, float3(float(bo.x) + 0.5, bestH + 3.5, float(bo.y) + 0.5), L_BEACON, 1.0);
                addPart(G, makePart(P_FIRE, M_FIRE, bo, int2(1), 2, 0, 0, F_REL | ((lamp + 1) << 8)), length(float2(bo)) + 2.0);
                addPart(G, makePart(P_BOX, M_STONE, bo - 1, int2(3), 1, 0, 0, F_REL), length(float2(bo)) + 3.0);
            }
        }
        age = 0.2 + 0.3 * hashf(hb >> 3);
    } else if (type == T_JETTY) {
        ok = buildJetty(hm, G, c, hb, waterDir, 24);
        age = 0.3;
    } else if (type == T_CAMP) {
        int cst = campStyle(L);
        int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_FIRE, 1.0);
        addPart(G, makePart(P_FIRE, M_FIRE, int2(-1), int2(3), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 3.0);
        int tents = 1 + int(hb % 2u);
        float a0 = hashf(hb >> 4) * 6.2831853;
        int4 rects[4];
        int nr = 0;
        rects[nr++] = int4(c - 1, c + 2);
        bool lit = false;
        for (int i = 0; i < 10 && nr <= tents; i++) {
            float a = a0 + float(i) * 2.39996;
            float2 dir = float2(cos(a), sin(a));
            float rad = 5.0 + float(i / 5);
            int2 tc = c + int2(round(dir * rad));
            // The camp's first shelter is its style's; a lean-to camp adds an A-frame, a caravan an awning.
            int ks = cst;
            if (nr > 1 && cst == ST_LEANTO) ks = ST_AFRAME;
            bool roundT = ks == ST_TIPI || (ks == ST_CARAVAN && nr == 1);
            int2 sz;
            bool axisZ = false;
            if (roundT) {
                sz = int2(5);
            } else {
                // A-frames point a gable at the fire; lean-tos and shelters face it.
                bool radialX = abs(dir.x) >= abs(dir.y);
                axisZ = ks == ST_AFRAME ? !radialX : radialX;
                int L = ks == ST_AFRAME ? 6 : ks == ST_LEANTO ? 5 : 4, D = ks == ST_AFRAME ? 5 : ks == ST_PALM || ks == ST_LEANTO ? 4 : 3;
                sz = axisZ ? int2(D, L) : int2(L, D);
            }
            int2 r0 = tc - sz / 2;
            if (!rectFree(rects, nr, r0, sz, 1)) continue;
            float lo, hi;
            int wet;
            if (!rectGround(hm, r0, sz, lo, hi, wet) || hi - lo > 1.0 || lo < SEA + 2.0) continue;
            float reach = length(float2(tc - c)) + 5.0;
            int lt = -1;
            if (!lit && ks != ST_PALM && ks != ST_LEANTO) {
                lt = addLamp(G, float3(float(tc.x - c.x) + 0.5, hi + 1.0, float(tc.y - c.y) + 0.5), L_TENT, 0.0);
                lit = true;
            }
            int lf = (lt + 1) << 8;
            if (roundT) {
                bool tipi = ks == ST_TIPI;
                addPart(G, makePart(P_ROUND, M_FELT, tc - c, int2(tipi ? 4 : 5, 0), int(hi), tipi ? 0 : 6, tipi ? 18 : 10, (ks << 12) | lf), reach);
            } else {
                addPart(G, makePart(P_HOUSE, ks == ST_PALM ? M_THATCH : M_FELT, r0 - c, sz, int(hi), 0,
                                    int(pcg(hb + uint(i)) & 0x7fffu), F_TENT | (axisZ ? F_AXISZ : 0) | (ks << 12) | lf), reach);
            }
            rects[nr++] = int4(r0, r0 + sz);
        }
        addPart(G, makePart(P_CLEAR, 0, int2(0), int2(13, 0), 0, 0, 0, 0), 14.0);
        age = 0.1;
        ok = true;
    }
    // No part may stand on carved rock (a shelter or notch): stamping skips such columns, which would leave the site
    // with pieces missing. Every second column suffices, as cleanup leaves no carved gap a single column wide; the reach
    // is capped where the reads stay inside every map that holds the cell.
    if (ok && G.np > 0) {
        int R = min(int(ceil(G.r)), 60);
        for (int j = -R; j <= R && ok; j += 2)
            for (int i = -R; i <= R; i += 2) {
                if (i * i + j * j > R * R) continue;
                int2 q = c + int2(i, j);
                if (colArch(hm, q) && (gapFlags(gx.read(uint2(q))) & GAP_CARVED) != 0u) { ok = false; break; }
            }
    }
    if (!ok || G.np == 0) {
        G.s.head.z = 0;
        G.s.info = float4(0);
        for (int i = 0; i < MAX_PARTS; i++) G.s.parts[i] = int4(0);
        for (int i = 0; i < MAX_LAMPS; i++) G.s.lamps[i] = float4(0);
        sites[idx] = G.s;
        return;
    }
    // Props come last, so nothing placed before them moves; tryProp keeps them off every arch column.
    if (type == T_HAMLET) hamletProps(hm, G, c, g, hamletStyle(L), hb, hearth);
    else if (type == T_CAMP) campProps(hm, G, c, g, campStyle(L), hb);
    G.s.head.z = type;
    G.s.info = float4(min(G.r, 125.0), age, float(G.np), float(G.nl));
    sites[idx] = G.s;
}

// structTex bits: 0-3 material, 4-6 roof plane (1 -x, 2 +x, 3 -z, 4 +z, 5 level ridge), 7 flat top, 8 clearing,
// 9-13 part, 14-22 site, 23 chimney, 24-27 feature, 28-29 ground mark.
constant uint S_FLAT = 1u << 7, S_CLEAR = 1u << 8, S_CHIMNEY = 1u << 23;
constant uint S_PART = 9u, S_SITE = 14u, S_FEAT = 24u, S_MARK = 28u;
constant uint MK_PATH = 1u, MK_FRINGE = 2u, MK_APRON = 3u;

struct Claim { bool hit; float H, gl, gh, base; uint bits; };

static inline Claim noClaim() { Claim c; c.hit = false; c.H = 0; c.gl = 0; c.gh = 0; c.base = 0; c.bits = 0; return c; }
static inline Claim solidClaim(float H, float base, uint bits) {
    Claim c; c.hit = true; c.H = H; c.gl = 0; c.gh = 0; c.base = base; c.bits = bits; return c;
}
// Solid below gl and from gh to H: lintels, eaves, decks and hulls over open ground or water.
static inline Claim gapClaim(float gl, float gh, float H, uint bits) {
    Claim c; c.hit = true; c.H = H; c.gl = gl; c.gh = gh; c.base = gh; c.bits = bits; return c;
}

// A round dwelling's top over its floor at distance r from its centre, and the outward fall per voxel there.
static inline float2 roundProfile(Part P, float r) {
    float R = float(P.sz.x) * 0.5, crown = float(P.y2) * 0.25, wall = float(P.y1) * 0.25;
    if (((P.flags >> 12) & 15) == ST_TIPI) {
        float Re = R + 0.5;
        return float2(max(crown * (1.0 - r / Re), 0.5), crown / Re);
    }
    float x = min(r / R, 1.0);
    return float2(r > R - 0.9 ? wall : wall + (crown - wall) * (1.0 - x * x), (crown - wall) * 2.0 * x / R);
}

// One part's column at q (map-local), on ground g; `near` collects the clearing around it.
static Claim stampPart(Part P, int2 q, int2 c, float g, texture2d<float, access::read> hm, uint hash, thread bool& near) {
    int2 r0 = c + P.o, r1 = r0 + P.sz;
    uint mat = uint(P.mat);
    bool inRect = all(q >= r0) && all(q < r1);
    bool wetOK = (P.flags & F_WATER) != 0;
    if (P.kind == P_BOX || P.kind == P_FIRE) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        if (!inRect || (!wetOK && g < SEA + 1.0)) return noClaim();
        // A hearth's ring of stones round its flame, lower at the corners so it reads round.
        if (P.kind == P_FIRE && P.sz.x == 3 && any(q != r0 + 1))
            return solidClaim(g + ((q.x == r0.x + 1) != (q.y == r0.y + 1) ? 0.5 : 0.25), g, uint(M_STONE) | S_FLAT);
        float top = P.kind == P_FIRE ? g + (P.sz.x == 3 ? 0.75 : 1.0) + float(P.y0) : (P.flags & F_REL) != 0 ? g + float(P.y0) : float(P.y0);
        if (top <= g) return noClaim();
        return solidClaim(top, g, mat | S_FLAT);
    }
    if (P.kind == P_HOUSE) {
        if (all(q >= r0 - 3) && all(q < r1 + 3)) near = true;
        if (!inRect) return noClaim();
        int style = (P.flags >> 12) & 15;
        uint post = uint(M_TIMBER) | (uint(FT_POST) << S_FEAT);
        if ((P.flags & F_TENT) != 0) {
            // Camp shelters, in half voxels over each column's own ground.
            if (g < SEA + 1.0) return noClaim();
            int4 ij = partIJ(P, q, c);
            int i = ij.x, j = ij.y, L = ij.z, D = ij.w;
            bool endI = i == 0 || i == L - 1, endJ = j == 0 || j == D - 1;
            if (style == ST_AFRAME) {
                // The ridge sags mid-span.
                int rj = abs(j - D / 2);
                if (endI) return noClaim();
                if (rj > 0) return solidClaim(g + 2.5 - float(rj), g, mat);
                return solidClaim(g + 2.5 - 0.25 * round(2.0 * sin(3.14159265 * (float(i) - 0.5) / float(L - 2))), g, mat);
            }
            if (style == ST_LEANTO) {
                // A tarp from a pole on two posts down to the ground behind.
                float2 mid = float2(r0 - c) + float2(P.sz) * 0.5;
                bool frontLo = ((P.flags & F_AXISZ) != 0 ? mid.x : mid.y) > 0.0;
                int jf = frontLo ? j : D - 1 - j;
                if (jf == 0) return endI ? solidClaim(g + 2.5, g, post) : gapClaim(g, g + 1.75, g + 2.5, mat);
                if (jf == 1) return gapClaim(g, g + 1.0, g + 1.75, mat);
                return solidClaim(g + (jf == 2 ? 1.0 : 0.5), g, mat);
            }
            // Palm shelters and caravan awnings: a roof on four posts.
            if (endI && endJ) return solidClaim(g + 2.5, g, post);
            bool inner = !endI && !endJ;
            return gapClaim(g, g + 2.0, g + (style == ST_PALM && inner ? 3.0 : 2.5), mat);
        }
        HouseAt h;
        houseAt(P, q, c, h);
        uint hh = uint(P.y2);
        float fl = float(P.y0);
        if (h.f.flat) {
            if (g < SEA + 1.0) return noClaim();
            if (h.jw < 0) {
                // Vigas: roof beams poking out of the back wall.
                bool viga = (h.iw & 1) == 1 && h.iw < h.Lw - 1 && flatMass(style, hh, h.iw, 0, h.Lw, h.Dw) >= 2.0;
                if (!viga || fl + 2.25 < g + 1.0) return noClaim();
                return gapClaim(g, fl + 2.25, fl + 2.75, uint(M_TIMBER) | (uint(FT_VIGA) << S_FEAT));
            }
            float top = flatTop(style, hh, h.iw, h.jw, h.Lw, h.Dw);
            if (top <= 0.0 || fl + top <= g) return noClaim();
            return solidClaim(fl + top, g, mat | S_FLAT);
        }
        bool stilt = style == ST_STILT;
        bool inWall = h.iw >= 0 && h.jw >= 0 && h.iw < h.Lw && h.jw < h.Dw;
        int ridge = (h.Dw - 1) / 2;
        if ((P.flags & F_CHIMNEY) != 0 && h.iw == chimneyAt(style, h.Lw, hh)) {
            if (h.jw == ridge) return solidClaim(fl + ridgeTop(h) + 1.0, g, uint(M_STONE) | S_FLAT | S_CHIMNEY);
            // The breast stops under the verge, so the roof edge runs across it rather than a pale post.
            if (style == ST_CABIN && abs(h.jw - ridge) == 1) return solidClaim(fl + houseTop(h), g, uint(M_STONE) | S_CHIMNEY);
        }
        if (h.dA < -1) {
            bool endPost = h.iw == 0 || h.iw == h.Lw - 1;
            if (stilt) {
                // The deck in front of the door, on its piles.
                if (h.iw < 0 || h.iw >= h.Lw || g >= fl) return noClaim();
                if (endPost) return solidClaim(fl, g, post);
                uint deck = uint(M_PLANK) | S_FLAT | (uint(FT_DECK) << S_FEAT);
                return g < fl - 1.0 ? gapClaim(g, fl - 0.5, fl, deck) : solidClaim(fl, g, deck);
            }
            return noClaim();
        }
        float H = fl + houseTop(h);
        if (!inWall) {
            // Eaves and rakes: the roof's edge on air; a longhouse's ridge ends rise in horns. A cabin's porch roof
            // stands on posts at its front corners.
            if (stilt && h.jw == ridge && h.dG < 0) H += 1.0;
            if (h.dG < -1 && (h.jw == 0 || h.jw == h.Dw - 1)) return solidClaim(H, g, post);
            float under = H - (style == ST_CABIN && h.dG < 0 && h.dA >= 0 ? max(h.f.lip, roofStep(h) + 0.25) : h.f.lip);
            if (under > g + 1.0) return gapClaim(g, under, H, mat);
            return g < H ? solidClaim(H, g, mat) : noClaim();
        }
        if (stilt) {
            bool pile = ((h.iw & 1) == 0 || h.iw == h.Lw - 1) && (h.jw == 0 || h.jw == h.Dw - 1);
            if (g < fl - 0.5 && !pile) return gapClaim(g, fl, H, mat);
            return solidClaim(H, g, mat);
        }
        if (g < SEA + 1.0) return noClaim();
        return solidClaim(H, g, mat);
    }
    if (P.kind == P_ROUND) {
        // Yurts (a wall ring under a dome) and tipis (a cone), in half voxels.
        float2 dq = float2(q - c - P.o);
        float r = length(dq), R = float(P.sz.x) * 0.5;
        if (r < R + 3.0) near = true;
        if (r > R + 0.2 || g < SEA + 1.0) return noClaim();
        float2 pr = roundProfile(P, r);
        float H = float(P.y0) + round(pr.x * 2.0) * 0.5;
        uint ft = 0u;
        if (r < 0.5) { H += 0.5; ft = uint(((P.flags >> 12) & 15) == ST_TIPI ? FT_POLE : FT_CROWN); }
        if (H <= g) return noClaim();
        return solidClaim(H, g, mat | (ft << S_FEAT));
    }
    if (P.kind == P_PROP) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        if (!inRect || (g < SEA + 1.0 && !wetOK)) return noClaim();
        bool az = P.sz.y > P.sz.x;
        int i = az ? q.y - r0.y : q.x - r0.x, j = az ? q.x - r0.x : q.y - r0.y;
        int L = max(P.sz.x, P.sz.y), D = min(P.sz.x, P.sz.y);
        bool endI = i == 0 || i == L - 1, endJ = j == 0 || j == D - 1;
        float fl = float(P.y0);
        // Quarter steps below 512, where float16 holds them; half steps above.
        float qs = fl < 500.0 ? 0.25 : 0.5;
        uint post = uint(M_TIMBER) | (uint(FT_POST) << S_FEAT);
        uint ph = pcg(hash + uint(P.y2) * 31u + uint(i) * 7u + uint(j) * 131u);
        int type = P.y1;
        float top = 0.0;
        uint bits = mat;
        if (type == PR_WOODPILE) top = fl + (endI ? 1.0 : 1.5);
        else if (type == PR_HAY) top = fl + (endI && endJ ? 1.0 : endI || endJ ? 1.5 : 2.0);
        else if (type == PR_OVEN) { if (endI && endJ) return noClaim(); top = fl + (endI || endJ ? 1.0 : 1.5); }
        else if (type == PR_GARDEN) {
            // A low rail on posts round the beds.
            if ((endI && endJ) || (endI && (j & 1) == 0) || (endJ && (i & 1) == 0)) { top = fl + 1.25; bits = post; }
            else if (endI || endJ) {
                if (fl + 0.4 <= g) return noClaim();
                return gapClaim(g, fl + 0.4, fl + 0.9, uint(M_PLANK) | (uint(FT_FENCE) << S_FEAT));
            }
            else { top = fl + ((i & 1) != 0 ? 0.5 : qs); bits = mat | (uint(FT_CROP) << S_FEAT); }
        } else if (type == PR_WASH || type == PR_RACK) {
            if (endI) { top = fl + (type == PR_WASH ? 2.5 : 2.0); bits = post; }
            else {
                float line = fl + (type == PR_WASH ? 2.0 : 1.75);
                float hang = type == PR_WASH ? ((ph & 3u) == 0u ? 1.0 : 0.5) + 0.5 : 1.0;
                if (line - hang <= g + 0.5) return line > g ? solidClaim(line, g, mat) : noClaim();
                return gapClaim(g, line - hang, line, mat);
            }
        } else if (type == PR_PEN) {
            if (!endI && !endJ) return noClaim();
            if (endJ && j == D - 1 && i == L / 2) return noClaim();
            if (mat == uint(M_STONE)) top = fl + 1.0 + ((ph & 1u) != 0u ? 0.5 : 0.0) - ((ph & 6u) == 0u ? 0.5 : 0.0);
            else if ((endI && endJ) || (endJ && (i & 1) == 0) || (endI && (j & 1) == 0)) { top = fl + 1.5; bits = post; }
            else { if (fl + 0.5 <= g) return noClaim(); return gapClaim(g, fl + 0.5, fl + 1.0, post); }
        } else if (type == PR_LOGSEAT) top = fl + (qs < 0.5 ? 0.75 : 0.5);
        else if (type == PR_BEDROLL) top = fl + (i == 0 ? 0.5 : qs);
        else if (type == PR_RUG) top = fl + qs;
        else if (type == PR_CANOE) {
            if (g < SEA) return gapClaim(g, SEA - 0.35, SEA + (endI ? 0.65 : 0.4), mat);
            top = fl + (endI ? 0.75 : 0.5);
        }
        else if (type == PR_STUMP) top = fl + (qs < 0.5 ? 0.75 : 1.0);
        else if (type == PR_PEAT || type == PR_BALES || type == PR_BARREL) top = fl + 1.0;
        if (top <= g) return noClaim();
        return solidClaim(top, g, bits);
    }
    if (P.kind == P_DECK) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        float T = float(P.y0);
        if (!inRect || g >= T) return noClaim();
        bool axisZ = (P.flags & F_AXISZ) != 0;
        int i = axisZ ? q.y - r0.y : q.x - r0.x;
        int j = axisZ ? q.x - r0.x : q.y - r0.y;
        int W = max(axisZ ? P.sz.x : P.sz.y, 1);
        // Pilings every third plank, on alternate sides of a wide deck.
        if (g < SEA && i % 3 == 1 && j == (i / 3) % W) return solidClaim(T, g, uint(M_TIMBER) | S_FLAT);
        if (g >= T - 1.0) return solidClaim(T, g, mat | S_FLAT);
        return gapClaim(g, T - 1.0, T, mat | S_FLAT);
    }
    if (P.kind == P_BOAT) {
        if (all(q >= r0 - 1) && all(q < r1 + 1)) near = true;
        if (!inRect) return noClaim();
        if (g < SEA - 1.0) return gapClaim(g, 19.0, 21.0, mat);
        return solidClaim(max(g + 1.0, 21.0), g, mat);
    }
    if (P.kind == P_HENGE) {
        float2 cc = float2(c + P.o) + 0.5;
        float R = float(P.y0) * 0.25;
        int N = max(P.y1, 1);
        float2 dq = float2(q) + 0.5 - cc;
        float r = length(dq);
        if (r > R + 4.0) return noClaim();
        near = true;
        if (r < R - 3.0) return noClaim();
        if (g < SEA + 1.0) return noClaim();
        float stepA = 6.2831853 / float(N);
        int k = int(round(atan2(dq.y, dq.x) / stepA));
        k = (k % N + N) % N;
        float ang = float(k) * stepA;
        float2 dir = float2(cos(ang), sin(ang));
        int2 sc = int2(floor(cc + dir * R));
        // The tangent runs along z where the radius runs along x.
        bool alongZ = abs(dir.x) >= abs(dir.y);
        int2 tv = alongZ ? int2(0, 1) : int2(1, 0);
        int2 rd = alongZ ? int2(dir.x > 0.0 ? 1 : -1, 0) : int2(0, dir.y > 0.0 ? 1 : -1);
        uint kh = pcg(hash + uint(P.y2) * 31u + uint(k) * 977u);
        float u = hashf(kh);
        float T = colH(hm, sc) + 3.0 + float((kh >> 8) % 3u);
        if (u < 0.2) return noClaim();
        if (u < 0.34) {
            // Fallen outward.
            for (int m = 0; m < 3; m++) if (all(q == sc + rd * m)) return solidClaim(g + 1.0, g, mat | S_FLAT);
            return noClaim();
        }
        if (u > 0.8) {
            // Trilithon: two uprights and a lintel over the gap between them.
            if (all(q == sc - tv) || all(q == sc + tv)) return solidClaim(max(T, g + 2.0), g, mat | S_FLAT);
            if (all(q == sc) && T - 1.0 > g + 1.5) return gapClaim(g, T - 1.0, T, mat | S_FLAT);
            return noClaim();
        }
        if (all(q == sc) || (((kh >> 12) & 1u) != 0u && all(q == sc + tv))) return solidClaim(max(T - 1.0, g + 2.0), g, mat | S_FLAT);
        return noClaim();
    }
    if (P.kind == P_WALL) {
        int2 a = c + P.o, b = c + int2(P.y0, P.y1);
        int2 lo = min(a, b), hi = max(a, b);
        if (any(q < lo - 2) || any(q > hi + 2)) return noClaim();
        float2 dv = float2(b - a);
        float s;
        bool on;
        if (abs(dv.x) >= abs(dv.y)) {
            float zl = float(a.y) + float(q.x - a.x) * (dv.x != 0.0 ? dv.y / dv.x : 0.0);
            on = q.x >= lo.x && q.x <= hi.x && int(round(zl)) == q.y;
            if (abs(zl - float(q.y)) < 2.5) near = true;
            s = abs(float(q.x - a.x));
        } else {
            float xl = float(a.x) + float(q.y - a.y) * dv.x / dv.y;
            on = q.y >= lo.y && q.y <= hi.y && int(round(xl)) == q.x;
            if (abs(xl - float(q.x)) < 2.5) near = true;
            s = abs(float(q.y - a.y));
        }
        if (!on || g < SEA + 1.0) return noClaim();
        // Eroded: the run breaks where the noise dips, and its broken top rises and falls.
        float nz = perlin2(float2(s / 5.0, float(P.flags >> 12) * 7.3 + 0.5), hash + 300u);
        if (nz < -0.22) return noClaim();
        float hgt = max(1.0, round(float(P.y2) * (0.6 + nz)));
        return solidClaim(g + hgt, g, mat | S_FLAT);
    }
    if (P.kind == P_CLEAR) {
        // A glade: no trunk within sz.x of o, so canopies keep off the camp.
        float2 dq = float2(q - c - P.o);
        if (dot(dq, dq) < float(P.sz.x * P.sz.x)) near = true;
        return noClaim();
    }
    if (P.kind == P_RING) {
        float2 cc = float2(c + P.o) + 0.5;
        float2 dq = float2(q) + 0.5 - cc;
        float r = length(dq);
        if (r < float(P.sz.y) + 3.0) near = true;
        if (r < float(P.sz.x) - 0.3 || r > float(P.sz.y) + 0.3 || g < SEA + 1.0) return noClaim();
        // A toppled tower: a ring of wall whose broken top falls from one side to the other.
        float th0 = hashf(uint(P.y1)) * 6.2831853;
        float k = 0.5 + 0.5 * cos(atan2(dq.y, dq.x) - th0);
        float hgt = 1.0 + round(float(P.y2) * k * k + perlin2(dq * 0.7, hash + 301u));
        return solidClaim(g + max(hgt, 1.0), g, mat | S_FLAT);
    }
    return noClaim();
}

// Worn ground at a column no part claims: the apron round a hearth or camp fire, and a hamlet's paths from its hearth
// to each door (and the jetty), wandering a little.
static uint groundMark(device const Site& s, int2 q, int2 c) {
    int np = int(s.info.z);
    bool hearth = false;
    for (int k = 0; k < np && !hearth; k++) {
        int4 v = s.parts[k];
        hearth = ((v.y >> 16) & 0xff) == P_FIRE && (v.y & 0xff) == 3 && (v.x & 0xffff) == 0xffff;
    }
    if (!hearth) return 0u;
    uint sh = uint(s.head.w);
    float2 p = float2(q - c);
    float2 wob = float2(perlin2(p / 6.0, sh + 340u), perlin2(p / 6.0 + 31.7, sh + 341u));
    float r = length(p);
    if (r < 2.7 + 1.2 * wob.x) return MK_APRON;
    uint fringe = r < 4.3 + 1.6 * wob.y ? MK_FRINGE : 0u;
    if (s.head.z != T_HAMLET) return fringe;
    float2 pw = p + 1.1 * wob;
    for (int k = 0; k < np; k++) {
        Part P = unpackPart(s.parts[k]);
        float2 d;
        if (P.kind == P_DECK) d = clamp(float2(0.0), float2(P.o), float2(P.o + P.sz - 1));
        else {
            float3 dr = houseDoor(P, sh, k, true);
            if (dr.z == 0.0) continue;
            d = dr.xy - 0.5;
        }
        float t = saturate(dot(pw, d) / max(dot(d, d), 1e-3));
        if (length(pw - d * t) < 0.75 || length(p - d) < 1.2) return MK_PATH;
    }
    return fringe;
}

// Raises each site's parts into the cleaned columns, and marks the clearing around them, where no tree takes a
// trunk. A structure column holds -base in alpha (base: its first structure voxel) and its detail in `st`; a
// clearing column holds -H, which reads as no structure.
kernel void stampStructures(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            texture2d<uint, access::write> st [[texture(2)]],
                            device const Site* sites [[buffer(0)]],
                            constant int4& info [[buffer(1)]],
                            constant uint& row0 [[buffer(2)]],
                            uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 col = src.read(gid);
    int2 q = int2(gid);
    // Arches keep their single gap.
    if (col.b <= col.g) {
        int2 home = int2(floor(float2(q + info.xy) / float(SITE_CELL))) - siteCell0(info.xy);
        bool near = false;
        int nearSite = 0;
        uint mark = 0u;
        for (int jj = -1; jj <= 1; jj++)
            for (int ii = -1; ii <= 1; ii++) {
                int2 gi = home + int2(ii, jj);
                if (any(gi < 0) || any(gi >= SITE_GRID)) continue;
                int si = gi.y * SITE_GRID + gi.x;
                device const Site& s = sites[si];
                if (s.head.z == 0) continue;
                int2 c = s.head.xy - info.xy;
                float2 dd = float2(q - c);
                if (dot(dd, dd) > (s.info.x + 4.0) * (s.info.x + 4.0)) continue;
                int np = int(s.info.z);
                for (int k = 0; k < np; k++) {
                    bool nk = false;
                    Claim cl = stampPart(unpackPart(s.parts[k]), q, c, col.r, src, uint(s.head.w), nk);
                    if (cl.hit) {
                        dst.write(float4(cl.H, cl.gl, cl.gh, -cl.base), gid);
                        st.write(uint4(cl.bits | (uint(k) << S_PART) | (uint(si) << S_SITE), 0, 0, 0), gid);
                        return;
                    }
                    if (nk && !near) { near = true; nearSite = si; }
                }
                if (mark == 0u && (s.head.z == T_HAMLET || s.head.z == T_CAMP)) {
                    mark = groundMark(s, q, c);
                    if (mark != 0u && !near) { near = true; nearSite = si; }
                }
            }
        if (near) {
            dst.write(float4(col.rgb, -col.r), gid);
            st.write(uint4(S_CLEAR | (uint(nearSite) << S_SITE) | (mark << S_MARK), 0, 0, 0), gid);
            return;
        }
    }
    dst.write(float4(col.rgb, 0), gid);
    st.write(uint4(0), gid);
}

// MARK: - Lamps

// Per kind (index = lamp kind): colour, light on the land (intensity at the source, reach in voxels), and the
// emission of the lamp's own faces. Scene units, before exposure (night exposure is 1.6-2.4).
// A window lamp's light on the land is the spill from its door.
constant float3 LAMP_COL[8] = { float3(0), float3(1.0, 0.42, 0.12), float3(1.0, 0.45, 0.14), float3(1.0, 0.60, 0.26),
                                float3(1.0, 0.66, 0.32), float3(1.0, 0.50, 0.16), float3(1.0, 0.40, 0.11), float3(1.0, 0.55, 0.22) };
constant float LAMP_I[8] = { 0, 3.0, 2.2, 0.45, 1.4, 4.0, 3.0, 0.3 };
constant float LAMP_R[8] = { 1, 20, 16, 7, 10, 24, 22, 6 };
constant float LAMP_EMIT[8] = { 0, 7.0, 6.0, 1.4, 2.6, 11.0, 8.0, 1.0 };

// Slow flicker that survives the temporal resolve: under ~2 Hz, with a rare gust.
static inline float flicker(float t, float i, uint seed) {
    float gust = smoothstep(0.55, 0.8, perlin2(float2(t * 0.15, i), seed + 401u)) * 0.3 * perlin2(float2(t * 4.0, i + 3.1), seed + 402u);
    return 1.0 + 0.12 * perlin2(float2(t * 0.9, i), seed + 403u) + 0.08 * perlin2(float2(t * 2.3, 2.0 * i), seed + 404u) + gust;
}

// How far a lamp is lit: lights come on through golden and blue hour, and some go out late in the night.
// lights: x sun elevation, y night progress (0 dusk .. 1 dawn), z day index.
static float lampLevel(int kind, uint h, float4 lights) {
    float sunY = lights.x, prog = lights.y;
    float on = kind == L_WINDOW || kind == L_TENT ? mix(0.05, -0.03, hashf(h)) : mix(0.03, -0.10, hashf(h));
    float dark = smoothstep(on + 0.015, on - 0.015, sunY);
    if (kind == L_WINDOW || kind == L_TENT) {
        float off = hashf(pcg(h + 1u)) < 0.55 ? smoothstep(0.0, 0.03, prog - 0.45 - 0.4 * hashf(pcg(h + 2u))) : 0.0;
        float relight = hashf(pcg(h + 3u)) < 0.15 ? smoothstep(0.9, 0.93, prog) : 0.0;
        return dark * max(1.0 - off, relight);
    }
    // Fires burn by day too (they smoke); their light only shows as the sun sinks.
    if (kind == L_FIRE) return 1.0 - (hashf(pcg(h + 1u)) < 0.3 ? 0.7 * smoothstep(0.7, 0.74, prog) : 0.0);
    if (kind == L_HEARTH) return 1.0 - 0.75 * smoothstep(0.6, 0.66, prog);
    if (kind == L_LANTERN) return dark;
    uint nightH = hash2i(int2(int(lights.z), 7), h);
    if (kind == L_BEACON) return dark * step(hashf(nightH), 0.35);
    if (kind == L_RITUAL) return dark * step(hashf(nightH), 0.3);
    return 0.0;
}

// One thread per lamp: its map-local position and kind (pos), and its colour times level and flicker plus its
// reach (state). Fires burn low by day.
kernel void updateLamps(device const Site* sites [[buffer(0)]],
                        device float4* pos [[buffer(1)]],
                        device float4* state [[buffer(2)]],
                        constant Frame& f [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= uint(SITE_GRID * SITE_GRID * MAX_LAMPS)) return;
    device const Site& s = sites[id / uint(MAX_LAMPS)];
    int li = int(id % uint(MAX_LAMPS));
    if (s.head.z == 0 || float(li) >= s.info.w) { pos[id] = float4(0); state[id] = float4(0); return; }
    float4 L = s.lamps[li];
    int kind = int(L.w) % 16;
    float3 p = float3(float(s.head.x - f.mapInfo.x) + L.x, L.y, float(s.head.y - f.mapInfo.y) + L.z);
    uint h = pcg(uint(s.head.w) + uint(li) * 7717u);
    float lvl = lampLevel(kind, h, f.lights);
    float t = f.camRight.w;
    float fl = flicker(t, float(li) * 3.7 + hashf(h) * 40.0, uint(f.mapInfo.w));
    float k = kind == L_WINDOW || kind == L_TENT ? 1.0 + (fl - 1.0) * 0.3 : kind == L_LANTERN ? 1.0 + (fl - 1.0) * 0.25 : fl;
    float sunAmt = smoothstep(-0.02, 0.05, f.lights.x);
    float day = kind == L_WINDOW || kind == L_LANTERN || kind == L_TENT ? 1.0 : mix(1.0, 0.02, sunAmt);
    // Past the 1-voxel map's morph band the land is coarse and carries no structures: lamps fade out with them.
    float2 ringCam = f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w;
    float far = 1.0 - smoothstep(f.lod[0].x, f.lod[0].y, length(p.xz - ringCam));
    // A plaza's cooking fire among pale walls a few voxels off: a full hearth floods them.
    int st0 = (unpackPart(s.parts[0]).flags >> 12) & 15;
    float dim = kind == L_HEARTH && (st0 == ST_ADOBE || st0 == ST_PUEBLO) ? 0.6 : 1.0;
    pos[id] = float4(p, float(kind));
    state[id] = float4(LAMP_COL[kind] * lvl * k * day * far * dim, LAMP_R[kind] * mix(1.0, dim, 0.8));
}

// Light from every lamp on a 4-voxel grid over the map: rgb irradiance, and the lamp position weighted by it
// (xyz) with the weight (w), so shading can face it and fall off with height.
kernel void bakeLights(texture2d<float, access::write> lm [[texture(0)]],
                       texture2d<float, access::write> ld [[texture(1)]],
                       device const Site* sites [[buffer(0)]],
                       device const float4* pos [[buffer(1)]],
                       device const float4* state [[buffer(2)]],
                       constant Frame& f [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= lm.get_width() || gid.y >= lm.get_height()) return;
    float texel = float(f.mapInfo.z) / float(lm.get_width());
    float2 p = (float2(gid) + 0.5) * texel;
    int2 home = int2(floor((p + float2(f.mapInfo.xy)) / float(SITE_CELL))) - siteCell0(f.mapInfo.xy);
    float3 E = 0;
    float4 P = 0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 gi = home + int2(i, j);
            if (any(gi < 0) || any(gi >= SITE_GRID)) continue;
            int si = gi.y * SITE_GRID + gi.x;
            device const Site& s = sites[si];
            if (s.head.z == 0 || s.info.w < 1.0) continue;
            float2 dc = p - float2(s.head.xy - f.mapInfo.xy);
            if (dot(dc, dc) > (s.info.x + 30.0) * (s.info.x + 30.0)) continue;
            for (int l = 0; l < int(s.info.w); l++) {
                float4 lp = pos[si * MAX_LAMPS + l], ls = state[si * MAX_LAMPS + l];
                float R = ls.w;
                float dd = length(p - lp.xz);
                if (dd >= R || all(ls.rgb == 0.0)) continue;
                float3 e = ls.rgb * LAMP_I[int(lp.w)] / (1.0 + (dd / (0.25 * R)) * (dd / (0.25 * R))) * smoothstep(R, 0.6 * R, dd);
                float wgt = dot(e, float3(0.3, 0.59, 0.11));
                E += e;
                P += float4(lp.xyz, 1.0) * wgt;
            }
        }
    lm.write(float4(E, 0), gid);
    ld.write(P, gid);
}

// A coarse column from its four child samples: their rounded mean, or toward their max where they straddle a sheer
// wall far taller than the column is wide, so walls fatten by half a column instead of each handoff growing a
// half-height ledge. Lowland relief never comes near the threshold.
static inline float coarseCombine(float a, float b, float c, float d, float s) {
    float mean = (a + b + c + d) * 0.25;
    float hi = max(max(a, b), max(c, d)), range = hi - min(min(a, b), min(c, d));
    float lo = max(3.0 * s, 40.0);
    if (range > lo) mean = mix(mean, hi, smoothstep(lo, 2.0 * lo, range));
    return floor(mean + 0.5);
}

// The top of an emergent's crown pads over world column centre w (trunk ground gt), as emergentIntervals builds them;
// -1e9 off the crown.
static float emergentTop(float2 w, float2 ctr, float gt, float y0, float R, int pads, uint h, uint seed) {
    float2 q = w - ctr, p = w + seedOffset(seed);
    float top = -1e9;
    float lumps = 1.6 * perlin2(p / 3.2, seed + 332u) + 1.2 * perlin2(p / 7.0, seed + 333u);
    for (int j = 0; j < pads; j++) {
        float T;
        float4 P = emergentPad(h, j, R, y0, T);
        float2 dq = q - P.xz;
        float dd = length(dq);
        if (dd > P.w * 1.35) continue;
        float2 u2 = dq / max(dd, 1e-3);
        float edge = P.w * (1.0 + 0.30 * perlin2(u2 * 1.7 + float(j) * 4.1 + hashf(h) * 50.0, seed + 331u));
        if (dd >= edge) continue;
        float u = dd / edge;
        top = max(top, floor(gt + P.y + T * 0.6 * sqrt(1.0 - u * u) + lumps * sqrt(1.0 - u * u)) + 1.0);
    }
    return top;
}

// The forest in the coarse rings, so it runs on to the horizon rather than melting into the ground past the 1-voxel
// map: per sample (the coarse column's four, at heights hs4) the top of the leaves over it, the jungle's roof and
// emergents or the planted trees as plantTrees makes them (without the checks that need the map: ledges, carved
// rock), and the column's top the mean of the samples' (leaves, else ground) rounded, as its ground is. Returns
// (top, depth of leaves under it); H is the bare column's height. Grove density and climate are taken once per column.
static float2 coarseVeg(int2 cw, int k, float4 hs4, float H, uint seed) {
    int s = 1 << k, off = s >> 2, hs = s >> 1;
    int2 b = cw * s + off;
    float2 centre = float2(cw * s) + 0.5 * float(s);
    float2 so = seedOffset(seed);
    float J = provinces(centre + so, seed).x;
    float sum = 0.0, leaves = 0.0;
    int2 o4[4] = { int2(0), int2(hs, 0), int2(0, hs), int2(hs, hs) };
    // The roof steps down on steep ground and stands on no cliff, as plantTrees grows it.
    float spread = max(max(hs4.x, hs4.y), max(hs4.z, hs4.w)) - min(min(hs4.x, hs4.y), min(hs4.z, hs4.w));
    // The samples' slope over plantTrees' 10-voxel window (they are hs apart; neighbours a voxel apart differ by a
    // whole step on gentle slopes, so the run counts as no shorter than 3).
    float steep = spread * 10.0 / max(1.41 * float(hs), 3.0);
    if (J >= 0.15) {
        for (int i = 0; i < 4; i++) {
            float2 wi = float2(b + o4[i]) + 0.5;
            float g = hs4[i], t = g;
            if (g >= SEA + 2.0 && steep < JUNGLE_CLIFF) {
                JCan cn = jungleCanopy(wi, g, J, seed, 1.0 - smoothstep(JUNGLE_STEEP, JUNGLE_CLIFF, steep));
                if (cn.hi > cn.lo) t = max(t, cn.hi);
                int2 e0 = int2(floor(wi / EMERGENT_CELL));
                for (int j = -1; j <= 1; j++)
                    for (int ii = -1; ii <= 1; ii++) {
                        float2 ec;
                        float y0, R;
                        int pads;
                        uint eh;
                        if (steep >= JUNGLE_STEEP || g < SEA + 10.0 || !emergentIn(e0 + int2(ii, j), seed, J, ec, y0, R, pads, eh)) continue;
                        if (length_squared(wi - ec) > 4.4 * R * R) continue;
                        t = max(t, emergentTop(wi, ec, g, y0, R, pads, eh, seed));
                    }
            }
            sum += t;
            leaves += t > g ? 1.0 : 0.0;
        }
    } else if (J < 0.15 && provinces(centre + so, seed).y >= 0.2) {
        float O = provinces(centre + so, seed).y;
        if (spread >= max(10.0, 0.6 * float(s))) return float2(H, 0);
        GRuns none;
        none.n = 0;
        for (int i = 0; i < 4; i++) {
            float2 wi = float2(b + o4[i]) + 0.5;
            float g = hs4[i], t = g;
            if (g >= SEA + 2.0) {
                int2 c0 = int2(floor(wi / OLD_CELL));
                for (int j = -1; j <= 1; j++)
                    for (int ii = -1; ii <= 1; ii++) {
                        float2 tc;
                        float Ht, R, tr;
                        uint th;
                        if (!oldTreeIn(c0 + int2(ii, j), seed, O, tc, Ht, R, tr, th)) continue;
                        if (length_squared(wi - tc) > 1.3 * R * R) continue;
                        t = max(t, oldTreeIntervals(wi, tc, g, Ht, R, tr, th, seed, false, none));
                    }
            }
            sum += t;
            leaves += t > g ? 1.0 : 0.0;
        }
    } else if (J < 0.15) {
        float dens = groveDensity(centre, H, seed);
        if (dens <= 0.0) return float2(H, 0);
        Climate c = climate(centre, seed);
        c.temp = altitudeTemp(c.temp, H);
        for (int i = 0; i < 4; i++) {
            int2 wi = b + o4[i];
            float g = hs4[i], t = g;
            if (g >= SEA + 2.0) {
                int2 home = int2(floor(float2(wi) / float(TREE_CELL)));
                for (int j = -1; j <= 1; j++)
                    for (int ii = -1; ii <= 1; ii++) {
                        int2 cell = home + int2(ii, j);
                        uint h = hash2i(cell, seed + 92u);
                        int2 tc = cell * TREE_CELL + int2(h % uint(TREE_CELL), (h / 7u) % uint(TREE_CELL));
                        int2 o = wi - tc;
                        if (abs(o.x) > 2 || abs(o.y) > 2 || hashf(pcg(h)) > dens) continue;
                        Tree tr;
                        tr.ok = false; tr.c = tc; tr.g = g; tr.lo = 0; tr.top = 0; tr.kind = 0;
                        tr = treeShape(tr, h, c, dens, float2(tc), seed);
                        if (!tr.ok || tr.kind == 0) continue;
                        int2 sp = canopySpan(tr, o);
                        if (sp.y > sp.x) t = max(t, float(sp.y));
                    }
            }
            sum += t;
            leaves += t > g ? 1.0 : 0.0;
        }
    }
    if (leaves == 0.0) return float2(H, 0);
    float top = max(floor(sum * 0.25 + 0.5), H);
    return float2(top, top - H);
}

// Mip 1 of the 1-voxel map also bounds the level-1 column over it, which the morph band slides toward: the
// four raw columns combined as coarseHeight combines them.
static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles, thread float2& g2, thread float& depth);
static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles) { float2 g2; float dp; return coarseColumn(cw, k, seed, giants, isles, g2, dp); }

kernel void reduceMaxBase(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::read> raw [[texture(1)]],
                          texture2d<float, access::write> dst [[texture(2)]],
                          constant int4& info [[buffer(0)]],
                          device const GiantRec* giants [[buffer(1)]],
                          constant uint& row0 [[buffer(2)]],
                          uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float m = max(max(src.read(s).r, src.read(s + uint2(1, 0)).r), max(src.read(s + uint2(0, 1)).r, src.read(s + uint2(1, 1)).r));
    float4 r4 = float4(raw.read(s).r, raw.read(s + uint2(1, 0)).r, raw.read(s + uint2(0, 1)).r, raw.read(s + uint2(1, 1)).r);
    float parent = coarseCombine(r4.x, r4.y, r4.z, r4.w, 2.0);
    int2 cw = (int2(s) + info.xy) >> 1;
    // The level-1 column's forest (coarseVeg from the same raw heights) as well.
    parent = max(parent, coarseVeg(cw, 1, r4, parent, uint(info.w)).x);
    // A giant's level-1 column needs nothing more: its samples are these four columns, and its top their tops' mean
    // (coarseColumn), which m bounds (working it out cost several ms a band).
    dst.write(float4(max(m, parent), 0, 0, 0), gid);
}

// MARK: - Coarse levels

// Coarse level k (1...4) has columns s = 2^k voxels wide, built from the same terrainColumn field: the mean of
// four point samples, one per child column, so no coarse level carries trees or arches.
constant int COARSE_N = 2048;
// Seen at a grazing angle, trees standing several voxels tall hide far more ground than their footprint covers.
constant float COVER_GAIN = 2.5;

// Levels 1 to GAP_LEVELS carry one gap per column, the carved principal gaps of their four samples: the rounded means of
// those there are, where they overlap, under at least two voxels of roof (a coarse column hollow wherever any sample is,
// or a recess's ragged back wall stands in it as a row of pillars). Their outline (H) ignores gaps.
constant int GAP_LEVELS = 4;
// Of those, levels 1...GAP2_LEVELS also carry a second gap below it (a giant's crown: the air under it, with the gap
// between its tiers as the principal one); further out a giant keeps only the air under its crown. Must match
// Clipmap.gap2Levels in Sources/Renderer.swift.
constant int GAP2_LEVELS = 1;

static inline float3 coarseColumn(int2 cw, int k, uint seed, device const GiantRec* giants, device const IsleRec* isles, thread float2& g2, thread float& depth) {
    g2 = 0.0;
    depth = 0.0;
    int s = 1 << k, off = s >> 2, hs = s >> 1;
    int2 b = cw * s + off;
    Col c0 = terrainColumnS(b, seed, s, isles), c1 = terrainColumnS(b + int2(hs, 0), seed, s, isles);
    Col c2 = terrainColumnS(b + int2(0, hs), seed, s, isles), c3 = terrainColumnS(b + int2(hs, hs), seed, s, isles);
    float H = coarseCombine(c0.h, c1.h, c2.h, c3.h, float(s));
    // A giant: where at least two samples stand in it, the mean of their tops, of their lowest gaps (the air under the
    // crown, the second gap) and of the tallest gap above that (between tiers, the principal one); solid where two are
    // trunk.
    Giant gi;
    if (giantAt(giants, float2(b + hs / 2) + 0.5, float(s), gi)) {
        float hs4[4] = { c0.h, c1.h, c2.h, c3.h };
        int2 o4[4] = { int2(0), int2(hs, 0), int2(0, hs), int2(hs, hs) };
        float nT = 0.0, sH = 0.0, nS = 0.0;
        float4 sLow = 0.0, sUp = 0.0;   // (sum lo, sum hi, count, -)
        for (int i = 0; i < 4; i++) {
            GCol gc;
            GRuns rr;
            rr.n = 0;
            if (giantColumn(gi, float2(b + o4[i]) + 0.5, hs4[i], rr, gc, k)) {
                nT += 1.0;
                sH += gc.H;
                float2 gs[3] = { gc.e.zw, gc.e.xy, float2(gc.gl, gc.gh) };
                int lowI = -1;
                for (int j = 0; j < 3; j++) if (gs[j].y > gs[j].x) { lowI = j; break; }
                if (lowI < 0) { nS += 1.0; continue; }
                sLow += float4(gs[lowI], 1.0, 0.0);
                float2 up = float2(0);
                for (int j = lowI + 1; j < 3; j++) if (gs[j].y - gs[j].x > up.y - up.x) up = gs[j];
                if (up.y > up.x) sUp += float4(up, 1.0, 0.0);
            }
        }
        if (nT >= 2.0) {
            // Marks the column as a giant's for shading (coverDepthL).
            depth = -1.0;
            float top = floor(sH / nT + 0.5);
            if (k > GAP_LEVELS || nS >= 2.0 || sLow.z < 1.0) return float3(max(top, H), 0, 0);
            float2 lowG = floor(sLow.xy / sLow.z + 0.5);
            lowG.x = max(lowG.x, H);
            if (lowG.y - lowG.x < 2.0 || top - lowG.y < 1.0) {
                // The lowest gaps disagree (a drooping skirt): the gap between tiers alone, if the samples share one.
                float2 upA = sUp.z >= 2.0 ? floor(sUp.xy / sUp.z + 0.5) : float2(0);
                if (upA.y - upA.x >= 2.0 && top - upA.y >= 1.0 && upA.x > H) return float3(top, upA);
                return float3(max(top, H), 0, 0);
            }
            float2 upG = sUp.z >= max(1.0, 0.5 * sLow.z) ? floor(sUp.xy / sUp.z + 0.5) : float2(0);
            if (k <= GAP2_LEVELS && upG.y - upG.x >= 2.0 && upG.x >= lowG.y + 2.0 && top - upG.y >= 1.0) { g2 = lowG; return float3(top, upG); }
            return float3(top, lowG);
        }
    }
    // No forest on an island: the 1-voxel map plants none there.
    bool isl = c0.island || c1.island || c2.island || c3.island;
    float2 vg = isl ? float2(H, 0.0) : coarseVeg(cw, k, float4(c0.h, c1.h, c2.h, c3.h), H, seed);
    depth = vg.y;
    float Hv = vg.x;
    // An island keeps its air out to the last ring, where other carved gaps are dropped.
    if (k > GAP_LEVELS || (k > 3 && !isl)) return float3(Hv, 0, 0);
    float n = 0, lo = 0, hi = 0, maxLo = 0, minHi = 1e9;
    if (c0.carved && c0.gh > c0.gl) { n += 1; lo += c0.gl; hi += c0.gh; maxLo = max(maxLo, c0.gl); minHi = min(minHi, c0.gh); }
    if (c1.carved && c1.gh > c1.gl) { n += 1; lo += c1.gl; hi += c1.gh; maxLo = max(maxLo, c1.gl); minHi = min(minHi, c1.gh); }
    if (c2.carved && c2.gh > c2.gl) { n += 1; lo += c2.gl; hi += c2.gh; maxLo = max(maxLo, c2.gl); minHi = min(minHi, c2.gh); }
    if (c3.carved && c3.gh > c3.gl) { n += 1; lo += c3.gl; hi += c3.gh; maxLo = max(maxLo, c3.gl); minHi = min(minHi, c3.gh); }
    if (n < 1.0 || maxLo >= minHi) return float3(Hv, 0, 0);
    lo = floor(lo / n + 0.5);
    hi = floor(hi / n + 0.5);
    if (hi - lo < 2.0 || H - hi < 2.0) return float3(Hv, 0, 0);
    // An island's air is flagged by its floor's sign (columnL and parentGap read the magnitude).
    return float3(Hv, isl ? -lo : lo, hi);
}

// Writes the world cells region.xy + [0, region.zw) of level info.x into its toroidal slice (texel = cell & 2047):
// height and canopy cover, and every second cell also the half-resolution climate texel it starts.
kernel void generateCoarse(texture2d<float, access::write> hOut [[texture(0)]],
                           texture2d<float, access::write> cOut [[texture(1)]],
                           texture2d<float, access::write> vOut [[texture(2)]],
                           texture2d<float, access::write> gOut [[texture(3)]],
                           texture2d<float, access::write> g2Out [[texture(4)]],
                           constant int4& region [[buffer(0)]],
                           constant int4& info [[buffer(1)]],
                           device const GiantRec* giants [[buffer(2)]],
                           device const IsleRec* isles [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int k = info.x, s = 1 << k;
    uint seed = uint(info.y);
    int2 cw = region.xy + int2(gid);
    float2 cc2;
    float depth;
    float3 cc = coarseColumn(cw, k, seed, giants, isles, cc2, depth);
    float H = cc.x;
    if (k <= GAP_LEVELS) gOut.write(float4(cc.y, cc.z, 0, 0), uint2(cw & (COARSE_N - 1)));
    if (k <= GAP2_LEVELS) g2Out.write(float4(cc2, 0, 0), uint2(cw & (COARSE_N - 1)));
    float2 centre = float2(cw * s) + 0.5 * float(s);
    // Expected share of the column under leaves: planting chance per tree cell times a mean canopy of ~13 columns.
    float cover = saturate(groveDensity(centre, H, seed) * 13.0 / float(TREE_CELL * TREE_CELL));
    hOut.write(float4(H), uint2(cw & (COARSE_N - 1)));
    // g: the depth of leaves under the top (coarseVeg), in units of 64 voxels.
    vOut.write(float4(cover, depth < 0.0 ? 1.0 : min(depth, 63.0) / 64.0, 0, 0), uint2(cw & (COARSE_N - 1)));
    if (all((cw & 1) == 0)) {
        int2 cj = cw >> 1;
        Climate c = climate((float2(cj) + 0.5) * float(2 * s), seed);
        cOut.write(float4(c.temp, c.moist, c.fall, c.basin), uint2(cj & (COARSE_N / 2 - 1)));
    }
}

// Mip 1 of level info.x over world mip-1 cells region.xy + [0, region.zw): each is one column of the next level,
// whose height the morph band slides toward, so it bounds that too. band: mip 0 of the level's band pyramid, the air
// those four columns leave open through every morph toward the parent: under an island (theirs or the parent's) both
// columns' air (columnMorphW keeps the morphed air inside both), else the air above the higher of the two tops.
kernel void reduceMaxParent(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            texture2d<float, access::read> cg [[texture(2)]],
                            texture2d<float, access::write> band [[texture(3)]],
                            constant int4& region [[buffer(0)]],
                            constant int4& info [[buffer(1)]],
                            device const GiantRec* giants [[buffer(2)]],
                            device const IsleRec* isles [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int2 c = region.xy + int2(gid);
    int2 s = c * 2;
    float m = max(max(src.read(uint2(s & (COARSE_N - 1))).r, src.read(uint2((s + int2(1, 0)) & (COARSE_N - 1))).r),
                  max(src.read(uint2((s + int2(0, 1)) & (COARSE_N - 1))).r, src.read(uint2((s + int2(1, 1)) & (COARSE_N - 1))).r));
    float3 pc = coarseColumn(c, info.x + 1, uint(info.y), giants, isles);
    dst.write(float4(max(m, pc.x)), uint2(c & (COARSE_N / 2 - 1)));
    bool pIsl = pc.y < 0.0;
    float2 pb = pc.z > abs(pc.y) ? float2(abs(pc.y), pc.z) : float2(pc.x, 4096.0);
    float2 b = float2(-1e4, 4096.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint2 t = uint2((s + int2(i, j)) & (COARSE_N - 1));
            float H = src.read(t).r;
            float2 g = cg.read(t).rg;
            bool isl = g.x < 0.0;
            g.x = abs(g.x);
            float2 cb = float2(max(H, pc.x), 4096.0);
            if (isl || pIsl) {
                float2 o = g.y > g.x ? g : float2(H, 4096.0);
                o = float2(max(o.x, pb.x), min(o.y, pb.y));
                if (o.y - o.x >= 4.0) cb = o;
            }
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
    band.write(float4(b, 0, 0), uint2(c & (COARSE_N / 2 - 1)));
}

// The last level's band mip 0 (it morphs toward nothing): per cell of its mip 1 the air its four columns' islands leave.
kernel void coarseBandTop(texture2d<float, access::read> ch [[texture(0)]],
                          texture2d<float, access::read> cg [[texture(1)]],
                          texture2d<float, access::write> dst [[texture(2)]],
                          constant int4& region [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    int2 c = region.xy + int2(gid);
    float2 b = float2(-1e4, 4096.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint2 t = uint2((c * 2 + int2(i, j)) & (COARSE_N - 1));
            float H = ch.read(t).r;
            float2 g = cg.read(t).rg;
            float2 cb = g.x < 0.0 && g.y + g.x >= 4.0 ? float2(-g.x, g.y) : float2(H, 4096.0);
            b = float2(max(b.x, cb.x), min(b.y, cb.y));
        }
    dst.write(float4(b, 0, 0), uint2(c & (COARSE_N / 2 - 1)));
}

kernel void coarseBandReduce(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             constant int4& region [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    uint2 s = g * 2;
    float2 a = src.read(s).rg, b = src.read(s + uint2(1, 0)).rg, c = src.read(s + uint2(0, 1)).rg, d = src.read(s + uint2(1, 1)).rg;
    dst.write(float4(max(max(a.x, b.x), max(c.x, d.x)), min(min(a.y, b.y), min(c.y, d.y)), 0, 0), g);
}

// Negates a coarse max mip's cells (region) whose band is a real one, as bandFlag does on the 1-voxel map: the trace
// reads the band only there.
kernel void coarseBandFlag(texture2d<float, access::read_write> mx [[texture(0)]],
                           texture2d<float, access::read> band [[texture(1)]],
                           constant int4& region [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    float2 b = band.read(g).rg;
    float m = abs(mx.read(g).r);
    mx.write(float4(b.y < 4000.0 && b.y - b.x >= 4.0 ? -m : m), g);
}

// Max mips are read through abs: a negated cell holds a real band (coarseBandFlag).
kernel void reduceMaxRegion(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            constant int4& region [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(region.z) || gid.y >= uint(region.w)) return;
    uint2 g = gid + uint2(region.xy);
    uint2 s = g * 2;
    float m = max(max(abs(src.read(s).r), abs(src.read(s + uint2(1, 0)).r)), max(abs(src.read(s + uint2(0, 1)).r), abs(src.read(s + uint2(1, 1)).r)));
    dst.write(float4(m, 0, 0, 0), g);
}

// MARK: - Traversal

struct Hit {
    bool hit;
    float t;
    int3 voxel;     // xz in frame-local cells of `level`, y in voxels
    int3 normal;
    int iters;
    int level;
};

static inline float3 columnAt(texture2d<float> hm, int2 c, int size) {
    c = clamp(c, int2(0), int2(size - 1));
    return hm.read(uint2(c), 0).rgb;
}

// The principal gap only: the extra gaps below it (notches, tubes) read as rock to corner AO, which only darkens their
// already dark corners a little more; decoding them in every AO tap cost a tenth of a millisecond.
static inline bool solid(texture2d<float> hm, int3 v, int size) {
    float3 col = columnAt(hm, v.xz, size);
    float y = float(v.y);
    return y < col.x && !(y >= col.y && y < col.z);
}

// The LOD rings. Level 0 is the 1-voxel map (texel = frame-local column); level k >= 1 is slice k - 1 of the coarse
// array, columns 2^k wide, addressed by world cell & 2047. Positions are frame-local (relative to the map origin,
// a multiple of 64), so a level's cells and max mips are world-aligned and the origin in level cells is exact.
struct Terrain {
    texture2d<float> hm;
    texture2d_array<float> ch;   // coarse height, with max mips that also bound the next level (its morph target)
    texture2d_array<float> cv;   // coarse canopy cover
    texture2d_array<float> cc;   // coarse climate, half resolution
    texture2d<float> clim;
    constant float4* lod;        // Frame.lod
    int2 origin;
    int size;
    int levels;
    float2 cam;                  // camera xz, frame-local
    float rmax;                  // the world's edge: a circle around the camera
    float fadeStart;             // where the haze starts closing in on that edge
    float globalMax;
    texture2d<float> hx;         // r16Float: mip 0 the column tops, mips from 1 their maxima (the DDA's skips)
    texture2d<uint> gx;          // the 1-voxel map's extra gaps (packGaps)
    texture2d_array<float> cg;   // levels 1...GAP_LEVELS: each column's gap (lo, hi)
    texture2d_array<float> cg2;  // levels 1...GAP2_LEVELS: a second gap below it (a giant crown's air)
    texture2d<float> hb;         // band mips (bandTop): the air a cell leaves open under canopies
    uint seed;
    device const GiantRec* giants; // the giant table (placeGiants)
    texture2d<float> pv;           // the province fields (bakeProvinces)
    constant TowerTable* towers; // the ruined towers nearest the camera (towerN of them; none outside the scene pass)
    int towerN;
    texture2d_array<float> cb;   // coarse band mips (reduceMaxParent): slice k - 1, mip m - 1 holds the air of max mip m's cells
    bool isles;                  // an island stands in view
};

static inline int2 levelOrigin(thread const Terrain& tr, int k) { return k == 0 ? int2(0) : tr.origin >> k; }

// Max height of mip-m cell mc (in the level's own aligned cell index).
static inline float maxAt(thread const Terrain& tr, int k, int2 mc, int m) {
    if (k == 0) return abs(tr.hx.read(uint2(clamp(mc, int2(0), int2((tr.size >> m) - 1))), m).r);
    return abs(tr.ch.read(uint2(mc & ((COARSE_N >> m) - 1)), uint(k - 1), uint(m)).r);
}

// Whether a column's principal gap is an island's air: on the 1-voxel map its flag, further out its floor's sign.
static inline bool isleGapL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return (gapFlags(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1))))) & GAP_ISLAND) != 0u;
    if (k > GAP_LEVELS) return false;
    return tr.cg.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).r < 0.0;
}

// Column (H, gapLo, gapHi, first leaf) of frame-local level cell c, unmorphed.
static inline float4 columnL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return tr.hm.read(uint2(clamp(c, int2(0), int2(tr.size - 1))), 0);
    uint2 t = uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1));
    float H = tr.ch.read(t, uint(k - 1), 0).r;
    if (k > GAP_LEVELS) return float4(H, 0, 0, 0);
    float2 g = tr.cg.read(t, uint(k - 1)).rg;
    return float4(H, abs(g.x), g.y, 0);
}



// Terrain height under any tree, as a point at height y sees it: under an island, the island's own ground, not its top.
static inline float groundL(thread const Terrain& tr, int k, int2 c, float y) {
    if (k == 0) {
        float4 col = tr.hm.read(uint2(clamp(c, int2(0), int2(tr.size - 1))), 0);
        if (col.a == 0.0 && col.b > col.g && y < col.b && isleGapL(tr, 0, c)) return col.g;
        return col.a > 0.0 ? col.g : col.r;
    }
    uint2 t = uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1));
    float H = tr.ch.read(t, uint(k - 1), 0).r;
    if (tr.isles && k <= GAP_LEVELS && y < H - 3.0) {
        float2 g = tr.cg.read(t, uint(k - 1)).rg;
        if (g.x < 0.0 && y < g.y) return -g.x;
    }
    return H;
}

// Ground bilinearly interpolated between column centres: the continuous slope under the stairs.
static inline float groundSmoothL(thread const Terrain& tr, int k, float2 p, float y) {
    float2 q = p / tr.lod[k].w - 0.5, f = fract(q);
    int2 i = int2(floor(q));
    return mix(mix(groundL(tr, k, i, y), groundL(tr, k, i + int2(1, 0), y), f.x),
               mix(groundL(tr, k, i + int2(0, 1), y), groundL(tr, k, i + int2(1, 1), y), f.x), f.y);
}

static inline float coverL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return 0.0;
    return tr.cv.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1), 0).r;
}

// (cover, depth of the leaves under the column's top in voxels: coarseVeg; 64 on a giant's columns) of coarse column c.
static inline float2 coverDepthL(thread const Terrain& tr, int k, int2 c) {
    if (k == 0) return 0.0;
    float2 v = tr.cv.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1), 0).rg;
    return float2(v.x, floor(v.y * 64.0 + 0.5));
}

// Height of the next level's column over level-k cell c.
static inline float parentH(thread const Terrain& tr, int k, int2 c) {
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.ch.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k), 0).r;
}

// Gap (lo, hi) of the next level's column over level-k cell c.
// The next level's second gap over level-k cell c (a giant crown's air under it; zero elsewhere).
static inline float2 parentGap2(thread const Terrain& tr, int k, int2 c) {
    if (k + 1 > GAP2_LEVELS) return float2(0);
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.cg2.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k)).rg;
}

// A lower gap across the morph band: toward the parent's second gap where it has one (opening from its floor where
// this column has none), else closing from the top. A crown's air would otherwise fill in across the band.
static inline float2 morphGap2(float2 g, float2 p2, float w) {
    if (p2.y > p2.x) return mix(g.y > g.x ? g : p2.xx, p2, w);
    return float2(g.x, mix(g.y, g.x, w));
}

// Its floor negative where it is an island's air.
static inline float2 parentGapS(thread const Terrain& tr, int k, int2 c) {
    if (k + 1 > GAP_LEVELS) return float2(0);
    int2 w = c + (k == 0 ? tr.origin : tr.origin >> k);
    return tr.cg.read(uint2((w >> 1) & (COARSE_N - 1)), uint(k)).rg;
}


// Morph weight toward the next level at frame-local xz p: 0 inside the band, 1 from 1.5 parent columns before the
// handoff circle on, so every parent cell the circle cuts is already fully morphed on the inner side.
static inline float morphAt(thread const Terrain& tr, int k, float2 p) {
    if (k >= tr.levels - 1) return 0.0;
    return smoothstep(tr.lod[k].x, tr.lod[k].z, length(p - tr.cam));
}

// Whether world xz w lies on a tower's plinth.
static inline bool towerWallAt(constant TowerTable* tt, float2 w) { return towerIndexAt(tt, w) >= 0; }

// Morphed (H, gapLo, gapHi): across the band a column slides to its parent's height and its gap to the parent's (opening
// from the parent's floor where it has none), or closes from the top where the parent has none, so at the handoff the
// level equals the next one exactly. Where either is an island's air, the air at every stage lies inside both columns'
// instead, so the bands under islands hold through the morph: the floor slides from one floor to the other (a top,
// where a column has no gap) and the roof stays the one either has.
// w: the morph weight (columnMorphW).
static inline float3 columnMorphW(thread const Terrain& tr, int k, int2 c, float w) {
    float4 c4 = columnL(tr, k, c);
    float3 col = c4.rgb;
    if (w > 0.0) {
        float ph = parentH(tr, k, c);
        // A column sliding up or down a tower's sheer wall stands part way up it as a sliver, and the band of them rules
        // the wall with stripes: there it steps at the middle of the band instead.
        float wh = w;
        if (tr.towerN > 0 && abs(ph - col.x) > 12.0 && towerWallAt(tr.towers, (float2(c) + 0.5) * tr.lod[k].w + float2(tr.origin)))
            wh = step(0.5, w);
        float2 pg = parentGapS(tr, k, c);
        bool cv = col.z > col.y;
        bool isl = pg.x < 0.0 || (cv && c4.a == 0.0 && isleGapL(tr, k, c));
        pg.x = abs(pg.x);
        bool pv = pg.y > pg.x;
        if (isl) {
            float lo = mix(cv ? col.y : col.x, pv ? pg.x : ph, w);
            float hi = cv && pv ? mix(col.z, pg.y, w) : cv ? col.z : pv ? pg.y : lo;
            col.x = mix(col.x, ph, wh);
            if (hi >= col.x) col = float3(min(col.x, lo), 0, 0);
            else col.yz = hi > lo ? float2(lo, hi) : float2(0);
        } else {
            col.x = mix(col.x, ph, wh);
            if (pv) {
                float2 g0 = cv ? col.yz : pg.xx;
                col.yz = mix(g0, pg, w);
            } else {
                col.z = mix(col.z, col.y, w);
            }
        }
    }
    return col;
}

// A coarse column's gap below the principal one, morphed by w: on levels 1...GAP2_LEVELS a giant crown's air under it
// (cg2), else none. Only a point under the principal gap can be in it.
static inline float2 secondGapW(thread const Terrain& tr, int k, int2 c, float w) {
    float2 g = k <= GAP2_LEVELS ? tr.cg2.read(uint2((c + levelOrigin(tr, k)) & (COARSE_N - 1)), uint(k - 1)).rg : float2(0);
    return w > 0.0 ? morphGap2(g, parentGap2(tr, k, c), w) : g;
}

static inline float3 columnMorph(thread const Terrain& tr, int k, int2 c) {
    return columnMorphW(tr, k, c, morphAt(tr, k, (float2(c) + 0.5) * tr.lod[k].w));
}

// The 1-voxel map's extra gaps (lo1, hi1, lo2, hi2) at frame-local column c, closing from the top across the morph band
// (the coarser levels carry only the principal gap).
static inline float4 extraGapsMorph(thread const Terrain& tr, int2 c) {
    float4 e = extraGaps(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1)))));
    float w = morphAt(tr, 0, float2(c) + 0.5);
    if (w > 0.0) { e.xy = morphGap2(e.xy, parentGap2(tr, 0, c), w); e.w = mix(e.w, e.z, w); }
    return e;
}

static inline Climate climateL(thread const Terrain& tr, int k, float2 p) {
    if (k == 0) return climateAt(tr.clim, p, tr.pv, tr.origin);
    constexpr sampler bl(filter::linear, address::repeat);
    float2 u = (p / (2.0 * tr.lod[k].w) + float2((tr.origin >> (k + 1)) & (COARSE_N / 2 - 1))) / float(COARSE_N / 2);
    float4 v = tr.cc.sample(bl, u, uint(k - 1), level(0));
    Climate c;
    c.temp = v.x; c.moist = v.y; c.fall = v.z; c.basin = v.w;
    float3 pv = provincesBaked(tr.pv, p + float2(tr.origin));
    c.jungle = pv.x; c.old = pv.y; c.isle = pv.z;
    return c;
}

// The level a point at distance r from the camera belongs to (levels: past the world's edge).
static inline int levelAt(thread const Terrain& tr, float r) {
    int k = 0;
    while (k < tr.levels && r >= tr.lod[k].y) k++;
    return k;
}

// Ray parameter at which o + d t leaves the circle of radius R around the camera in xz (0 if never inside).
static inline float circleExit(float2 o, float2 dxz, float2 cam, float R) {
    float2 q = o - cam;
    float a = dot(dxz, dxz), b = dot(q, dxz), c = dot(q, q) - R * R;
    if (a < 1e-12) return c < 0.0 ? 1e30 : 0.0;
    float disc = b * b - a * c;
    return disc <= 0.0 ? 0.0 : (-b + sqrt(disc)) / a;
}

// Max-mip quadtree walk over the column heightfields; columns are stacks of unit voxels. A ray never moves to a
// finer level: it starts at the coarser of startLevel and its origin's ring, and steps out a level exactly where
// it leaves each handoff circle. Every level covers the whole disc inside its circle, so a ray turning back toward
// the camera still has data under it.
static Hit trace(thread const Terrain& tr, float3 o, float3 d, float tMax, int maxIter, int startLevel) {
    Hit h;
    h.hit = false;
    h.t = tMax;
    h.iters = -1;
    h.level = 0;
    float3 inv = 1.0 / select(d, copysign(float3(1e-6), d), abs(d) < 1e-6);

    // Clip to [-1, globalMax] in y and to the world's circle.
    float ty0 = (-1.0 - o.y) * inv.y, ty1 = (tr.globalMax - o.y) * inv.y;
    float tEnter = max(min(ty0, ty1), 0.0);
    float tExit = min(min(max(ty0, ty1), tMax), circleExit(o.xz, d.xz, tr.cam, tr.rmax));
    if (tEnter >= tExit) return h;

    float t = tEnter;
    int k = max(startLevel, levelAt(tr, length(o.xz + d.xz * t - tr.cam)));
    if (k >= tr.levels) return h;
    float tSwitch = k < tr.levels - 1 ? circleExit(o.xz, d.xz, tr.cam, tr.lod[k].y) : 1e30;
    float s = tr.lod[k].w, invS = 1.0 / s;
    int2 fo = levelOrigin(tr, k);
    int level = MAX_LEVEL;
    // A ray that starts inside a column hits the side it last crossed into.
    float2 tbox = min((0.0 - o.xz) * inv.xz, (float(tr.size) - o.xz) * inv.xz);
    int lastAxis = (tbox.x > tbox.y) ? 0 : 2;
    // Spatial nudge across the boundary just crossed; a t-epsilon vanishes at large map coords.
    float2 nudge = float2(0);
    for (int i = 0; i < maxIter; i++) {
        h.iters = i;
        if (t >= tExit) return h;
        if (t >= tSwitch) {
            k++;
            tSwitch = k < tr.levels - 1 ? circleExit(o.xz, d.xz, tr.cam, tr.lod[k].y) : 1e30;
            s = tr.lod[k].w;
            invS = 1.0 / s;
            fo = levelOrigin(tr, k);
            level = MAX_LEVEL;
            nudge = float2(0);
            continue;
        }
        float3 p = o + d * t;
        int2 mc = (int2(floor((p.xz + nudge) * invS)) + fo) >> level;
        int2 lc = (mc << level) - fo;
        float2 cLo = float2(lc) * s, cHi = cLo + s * float(1 << level);
        float2 tb = (select(cLo, cHi, d.xz > 0.0) - o.xz) * inv.xz;
        float tCell = min(min(tb.x, tb.y), min(tExit, tSwitch));
        float yA = o.y + d.y * t, yB = o.y + d.y * tCell;
        if (level > 0) {
            // Mips from 1 up also bound the next level's columns, which morph-band columns rise or sink toward.
            // A negative max flags a cell with an open band: under a crown or an island (bandFlag, coarseBandFlag).
            float mx = k == 0 ? tr.hx.read(uint2(clamp(mc, int2(0), int2((tr.size >> level) - 1))), level).r
                              : tr.ch.read(uint2(mc & ((COARSE_N >> level) - 1)), uint(k - 1), uint(level)).r;
            bool clear = min(yA, yB) >= abs(mx);
            // The 1-voxel map's canopy bands are the columns' unmorphed gaps, which close across the morph band: only a
            // cell wholly inside its start may skip through one. Island bands, and every coarse level's, hold through it.
            if (!clear && BAND_SKIP && mx < 0.0) {
                float2 bd = k == 0 ? tr.hb.read(uint2(clamp(mc, int2(0), int2((tr.size >> level) - 1))), level - 1).rg
                                   : tr.cb.read(uint2(mc & ((COARSE_N >> level) - 1)), uint(k - 1), uint(level - 1)).rg;
                float2 lo = bandLo(bd.x);
                if (k > 0 || lo.y > 0.0 || length(0.5 * (cLo + cHi) - tr.cam) + 0.71 * (cHi.x - cLo.x) < tr.lod[0].x)
                    clear = min(yA, yB) >= lo.x && max(yA, yB) < bd.y;
            }
            if (clear) {
                lastAxis = tb.x < tb.y ? 0 : 2;
                nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
                t = tCell;
                level = min(level + 1, MAX_LEVEL);
                continue;
            }
            level--;
            continue;
        }

        // One column, solid below H except in its gaps (the principal one, and on the 1-voxel map up to two more below
        // it); a ray passing over it moves on as a skip would. A ray above the column never needs its gaps, which lie
        // at least two voxels under its top: coarse levels read them only for a ray inside.
        float wm = morphAt(tr, k, (float2(lc) + 0.5) * s);
        float3 col = columnMorphW(tr, k, lc, wm);
        float H = col.x, gl = col.y, gh = col.z;
        bool inGap = yA >= gl && yA < gh;
        // Under the principal gap: the 1-voxel map's extra gaps, a coarse giant column's air under its crown.
        if (!inGap && k <= GAP2_LEVELS && gh > gl && yA < gl) {
            float4 e;
            if (k == 0) {
                e = extraGaps(tr.gx.read(uint2(lc)));
                if (wm > 0.0) { e.xy = morphGap2(e.xy, parentGap2(tr, 0, lc), wm); e.w = mix(e.w, e.z, wm); }
            } else e = float4(secondGapW(tr, k, lc, wm), 0, 0);
            bool g1 = yA >= e.x && yA < e.y, g2 = yA >= e.z && yA < e.w;
            inGap = g1 || g2;
            gl = g1 ? e.x : e.z;
            gh = g1 ? e.y : e.w;
        }
        h.level = k;
        if (yA < H && !inGap) {
            h.hit = true;
            h.t = t;
            h.normal = lastAxis == 0 ? int3(d.x > 0 ? -1 : 1, 0, 0) : int3(0, 0, d.z > 0 ? -1 : 1);
            h.voxel = int3(lc.x, int(floor(yA)), lc.y);
            return h;
        }
        float floorY = inGap ? gl : H;
        if (d.y < 0 && yB < floorY) {
            h.hit = true;
            h.t = (floorY - o.y) * inv.y;
            h.normal = int3(0, 1, 0);
            h.voxel = int3(lc.x, int(ceil(floorY)) - 1, lc.y);
            return h;
        }
        if (inGap && d.y > 0 && yB > gh) {
            h.hit = true;
            h.t = (gh - o.y) * inv.y;
            h.normal = int3(0, -1, 0);
            h.voxel = int3(lc.x, int(gh), lc.y);
            return h;
        }
        lastAxis = tb.x < tb.y ? 0 : 2;
        nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
        t = tCell;
        level = 1;
    }
    // Out of steps, not out of the world.
    h.iters = maxIter;
    return h;
}

// MARK: - Horizon

// The terrain horizon around the camera: per azimuth bin, the steepest tangent of elevation any terrain reaches,
// from the max mips (cells a few percent of the distance wide, so a bin also bounds its neighbours' gaps). A
// primary ray above its bin's horizon (and the neighbours') can hit nothing and skips the trace.
constant int HORIZON_BINS = 1024;
constant int HORIZON_SEGS = 16;

kernel void bakeHorizon(device float* out [[buffer(0)]],
                        constant Frame& f [[buffer(1)]],
                        texture2d<float> hm [[texture(0)]],
                        texture2d_array<float> coarse [[texture(1)]],
                        texture2d<float> hx [[texture(2)]],
                        texture2d<uint> gx [[texture(3)]],
                        texture2d_array<float> cg [[texture(4)]], texture2d_array<float> cg2 [[texture(5)]],
                        device const GiantRec* giants [[buffer(2)]],
                        uint2 tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    // HORIZON_SEGS threads per bin walk geometric slices of the distance and reduce, so the bake is not one long chain.
    threadgroup float part[HORIZON_SEGS];
    uint gid = tid.y;
    int levels = int(f.fade.z);
    Terrain tr { hm, coarse, coarse, coarse, hm, f.lod, f.mapInfo.xy, f.mapInfo.z, levels,
                 f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w, f.fade.y, f.fade.x, 0.0, hx, gx, cg, cg2, hx, uint(f.mapInfo.w), giants, hx, nullptr, 0, cg, false };
    float a = (float(gid) + 0.5) * 6.2831853 / float(HORIZON_BINS);
    float2 dir = float2(sin(a), cos(a));
    float3 o = f.camPos.xyz;
    float best = -1e9;
    float d = lane == 0 ? 1.0 : pow(tr.rmax, float(lane) / float(HORIZON_SEGS));
    float dEnd = pow(tr.rmax, float(lane + 1) / float(HORIZON_SEGS));
    while (d < dEnd) {
        float2 X = o.xz + dir * d;
        int k = levelAt(tr, length(X - tr.cam));
        if (k >= levels) break;
        float cs = tr.lod[k].w;
        // Cell about 3% of the distance (and at least two columns) wide, from mip 1 so morph targets are bounded.
        int m = clamp(int(ceil(log2(max(0.03 * d, 2.0 * cs) / cs))), 1, MAX_LEVEL);
        float w = cs * float(1 << m);
        int2 mc = (int2(floor(X / cs)) + levelOrigin(tr, k)) >> m;
        float hmax = maxAt(tr, k, mc, m);
        best = max(best, (hmax - o.y) / max(d - 0.75 * w, 0.5));
        d += 0.5 * w;
    }
    part[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (int i = 1; i < HORIZON_SEGS; i++) best = max(best, part[i]);
        out[gid] = best;
    }
}

static inline bool aboveHorizon(device const float* hz, float3 d) {
    float l = length(d.xz);
    if (l < 1e-4) return d.y > 0.0;
    float a = atan2(d.x, d.z);
    int i = int(floor((a < 0.0 ? a + 6.2831853 : a) / 6.2831853 * float(HORIZON_BINS)));
    float t = max(max(hz[(i + HORIZON_BINS - 1) % HORIZON_BINS], hz[i % HORIZON_BINS]), hz[(i + 1) % HORIZON_BINS]);
    return d.y / l > t + 0.002;
}

// MARK: - Fullscreen

struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

// MARK: - Sky & light

struct Light {
    float3 key;     // direction to the current key light (sun, afterglow or moon)
    float3 keyCol;
    bool shadows;
    float3 zenith;   // terrain/cloud fill
    float3 horizon;
    float3 skyZen;   // sky radiance, kept darker and more saturated than the fill so twilight skies hold colour
    float3 skyHor;
    float3 sun;
    float3 moon;
    float sunAmt;
    float night;
    float gold;     // golden and blue hour; 0 at noon and deep night
};

// Keyframes by sun elevation: night, blue hour, sunrise, golden, midday.
static inline float3 key5(float e, float3 a, float3 b, float3 c, float3 d, float3 f) {
    float k0 = -0.30, k1 = -0.10, k2 = 0.0, k3 = 0.18, k4 = 0.32;
    if (e < k1) return mix(a, b, smoothstep(k0, k1, e));
    if (e < k2) return mix(b, c, smoothstep(k1, k2, e));
    if (e < k3) return mix(c, d, smoothstep(k2, k3, e));
    return mix(d, f, smoothstep(k3, k4, e));
}

static inline float goldAmount(float e) { return smoothstep(0.30, 0.12, e) * smoothstep(-0.16, -0.04, e); }

static Light makeLight(float3 sunDir) {
    Light L;
    float e = sunDir.y;
    L.sun = sunDir;
    // The moon rides lower than the anti-sun (at most ~20 degrees), so a frame that holds it still has room
    // for land below the horizon; a high moon left land a strip along the bottom.
    L.moon = normalize(float3(-sunDir.x, max(-sunDir.y * 0.7, 0.08), -sunDir.z));
    L.zenith = key5(e, float3(0.018, 0.022, 0.045), float3(0.10, 0.10, 0.22), float3(0.13, 0.16, 0.32),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    L.horizon = key5(e, float3(0.022, 0.026, 0.045), float3(0.30, 0.24, 0.36), float3(0.80, 0.45, 0.28),
                     float3(0.95, 0.60, 0.32), float3(0.46, 0.62, 0.86));
    L.skyHor = key5(e, float3(0.022, 0.026, 0.045), float3(0.16, 0.10, 0.22), float3(0.50, 0.22, 0.09),
                    float3(0.80, 0.48, 0.22), float3(0.50, 0.66, 0.90));
    L.skyZen = key5(e, float3(0.010, 0.013, 0.028), float3(0.03, 0.04, 0.12), float3(0.07, 0.09, 0.24),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    float3 sunCol = key5(e, float3(0), float3(0), float3(1.8, 0.6, 0.22), float3(3.0, 1.45, 0.55), float3(3.0, 2.7, 2.2));
    L.sunAmt = smoothstep(-0.02, 0.05, e);
    L.night = smoothstep(-0.04, -0.20, e);
    L.gold = goldAmount(e);
    if (e > -0.02) {
        L.key = sunDir;
        L.keyCol = sunCol * L.sunAmt + float3(1.3, 0.5, 0.3) * 0.4 * (1.0 - L.sunAmt);
        L.shadows = true;
    } else {
        // Below the horizon an afterglow from the sun's azimuth hands over to the moon.
        float3 glowDir = normalize(float3(sunDir.x, 0.05, sunDir.z));
        float3 glowCol = float3(1.3, 0.5, 0.3) * 0.4 * smoothstep(-0.20, -0.02, e);
        float m = smoothstep(-0.09, -0.16, e);
        L.key = normalize(mix(glowDir, L.moon, m));
        L.keyCol = mix(glowCol, float3(0.60, 0.66, 0.80) * L.night, m);
        L.shadows = m > 0.5;
    }
    return L;
}

static inline float hg(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (12.566 * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

static float3 skyColor(float3 d, Light L, bool withDiscs, uint seed, float time) {
    float up = saturate(d.y);
    float lowSun = 1.0 - smoothstep(0.0, 0.35, abs(L.sun.y));
    float3 hor = L.skyHor;
    // Low sun: gold toward the sun, a cooler pink-lavender band opposite it.
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(L.sun.xz + 1e-5);
    float toward = 0.5 + 0.5 * dot(dh, sh2);
    hor *= mix(float3(1.0), mix(float3(0.75, 0.6, 0.85), float3(1.0), toward), lowSun);
    // Golden hour: the horizon band burns amber under the sun and cools to dusty rose and slate opposite it.
    hor *= mix(float3(1.0), mix(float3(0.70, 0.70, 1.0), float3(0.92, 0.68, 0.42), toward * toward), L.gold * mix(0.5, 1.0, L.sunAmt));
    float3 c = mix(hor, L.skyZen, 1.0 - exp(-up * mix(mix(5.0, 7.0, saturate(L.sun.y / 0.4)), 8.0, L.gold)));
    if (d.y < 0) c = hor * mix(1.0, 0.75, saturate(-d.y * 4));
    float s = max(dot(d, L.sun), 0.0);
    float3 glowCol = key5(L.sun.y, float3(0), float3(0.45, 0.18, 0.12), float3(1.2, 0.40, 0.12), float3(1.0, 0.65, 0.35), float3(0.5, 0.45, 0.4));
    glowCol = mix(glowCol, glowCol * float3(1.0, 0.78, 0.5), L.gold);
    c += glowCol * (pow(s, 6.0) * mix(0.35, 0.22, L.gold) + pow(s, 64.0) * 0.6) * (1.0 - 0.5 * up);
    // A tighter aureole and a dome that deepens away from the sun, so the sun sits in its own pool of light.
    c *= mix(float3(1.0), float3(0.70, 0.70, 0.95), L.gold * mix(0.5, 1.0, L.sunAmt) * (1.0 - toward) * smoothstep(-0.05, 0.25, d.y));
    c += glowCol * pow(s, 24.0) * 0.5 * L.gold * (1.0 - 0.5 * up);
    if (!withDiscs) return c;
    // Disc takes the key light's hue (amber near the horizon) and blooms into the haze.
    float3 tint = L.keyCol / max(max(L.keyCol.r, max(L.keyCol.g, L.keyCol.b)), 1e-3);
    c += tint * 12.0 * smoothstep(0.99955, 0.9998, s) * L.sunAmt;
    c += L.keyCol * (pow(s, 1200.0) * mix(0.8, 0.35, L.gold) + pow(s, 180.0) * 0.25) * L.sunAmt;

    if (L.night > 0.0) {
        float md = max(dot(d, L.moon), 0.0);
        c += float3(0.10, 0.13, 0.22) * (pow(md, 24.0) * 0.6 + pow(md, 400.0) * 1.5) * L.night;
        // Stars: one hashed point per cell, thinned by low-frequency noise into clusters and voids and
        // denser along a band (a galactic plane at a seed-dependent tilt).
        float3 q = d * 260.0;
        int3 cell = int3(floor(q));
        uint h = hash3i(cell, seed + 77u);
        float3 gp = normalize(float3(0.6, 0.5, 0.3 + hashf(pcg(seed + 3u)) * 0.6));
        float bd = dot(d, gp);
        float band = exp(-pow(bd / 0.18, 2.0));
        // Milky Way: a wide glow and a brighter core built from layered noise, a fine grain of unresolved
        // stars, a meandering dark rift and ragged dust clouds.
        float n1 = perlin3(d * 3.0, seed + 80u), n2 = perlin3(d * 9.0, seed + 81u);
        float n3 = perlin3(d * 24.0, seed + 82u), n4 = perlin3(d * 60.0, seed + 85u);
        float wob = bd + 0.05 * n1;
        float core = exp(-pow(wob / 0.08, 2.0));
        float glow = (exp(-pow(wob / 0.22, 2.0)) * 0.45 + core) * saturate(0.7 + 0.5 * n1 + 0.4 * n2 + 0.3 * n3 + 0.2 * n4);
        float rift = exp(-pow((wob - 0.015 - 0.025 * n2 - 0.01 * n3) / 0.028, 2.0)) * smoothstep(-0.3, 0.2, n1 + 0.5 * n2);
        float murk = smoothstep(0.05, 0.35, n2 + 0.6 * n3 + 0.3 * n4) * exp(-pow(wob / 0.14, 2.0));
        float dust = saturate(rift * 0.6 + murk * 0.4);
        float3 mwCol = mix(float3(0.55, 0.62, 0.90), float3(0.95, 0.88, 0.78), saturate(core * (0.5 + 0.8 * n2)));
        float horizonFade = smoothstep(0.0, 0.12, d.y);
        float mw = glow * (1.0 - dust) * L.night * horizonFade;
        c += mwCol * mw * 0.07;
        if (mw > 0.02) {
            float3 fq = d * 500.0;
            uint fh = hash3i(int3(floor(fq)), seed + 86u);
            float3 fsp = floor(fq) + float3(hashf(pcg(fh)), hashf(pcg(fh + 1)), hashf(pcg(fh + 2)));
            c += mwCol * 0.25 * mw * step(0.75, hashf(pcg(fh + 3))) * smoothstep(0.75, 0.15, length(fq - fsp));
        }
        float clump = saturate(perlin3(d * 3.0, seed + 78u) * 0.9 + 0.5);
        float density = mix(0.004, 0.035, saturate(clump * 0.8 + band * 0.7 + glow * 0.6));
        if (hashf(h) > 1.0 - density) {
            float3 sp = float3(cell) + float3(hashf(pcg(h)), hashf(pcg(h + 1)), hashf(pcg(h + 2)));
            float dd = length(q - sp);
            float b = pow(hashf(pcg(h + 3)), 9.0) * 7.0 + 0.12;
            float twinkle = 0.7 + 0.3 * sin(time * 1.3 + hashf(pcg(h + 4)) * 40.0);
            c += float3(0.9, 0.95, 1.1) * b * twinkle * smoothstep(0.55, 0.0, dd) * L.night * smoothstep(0.0, 0.05, d.y) * (1.0 - 0.8 * dust);
        }
        // Moon: a clean disc whose maria are Perlin noise.
        if (md > 0.99) {
            float3 mu = normalize(cross(L.moon, float3(0, 1, 0)));
            float3 mv = cross(mu, L.moon);
            float2 xy = float2(dot(d, mu), dot(d, mv)) / 0.03;
            float disc = 1.0 - smoothstep(0.96, 1.0, length(xy));
            float lum = 0.92 - 0.3 * smoothstep(0.15, 0.55, fbm2(xy * 1.3 + 3.0, 3, 9u) + 0.25);
            c = mix(c, float3(0.75, 0.80, 0.92) * 2.4 * lum, L.night * disc);
        }
    }
    return c;
}

// MARK: - Surface

struct Surface {
    float3 albedo;
    float3 tread;   // the column's top colour, used to prefilter risers at distance
    float3 face;    // the colour a steep face of this column shows from afar: rock, snow-flecked over the snowline
    float4 nh;      // neighbour heights: -x, +x, -z, +z
    float H;
    int mat;        // 0 terrain, 1 leaf, 2 wood, 3 pale bark
    float canopy;   // ambient left under a canopy overhead
    float2 slope;   // heightfield gradient of fine relief on top faces
    float riser;    // drop below this face, for side faces
    float autumn;
    float level;    // 1 shades the face as dead-level ground: up normal, no corner AO
    bool gapNear;   // this column or the one in front of the face has a rock gap (on the 1-voxel map: any 4-neighbour)
    float2 gapSpan; // the lowest floor and highest roof of those gaps (extra gaps lie lower still: floor 0 then)
    float pale;     // sand, salt crust or snow
    float wall;     // megaWall: among the great walls
    float3 nrm;     // the plane a structure's roof voxel or a tower's wall shades as (zero: the face normal)
    float3 emit;    // fire, lanterns and lit windows
    bool isleUnder; // under an island: the sky it hides is isleSkyOcc's
    float gy;       // the height the ground round it is judged from (groundL): above any island for an island's own rock
};

// Autumn foliage: a slow Perlin hue field runs lingering green, gold, orange, scarlet to crimson;
// birch groves lean gold. Kept dark and earthy so the colour reads as paint, not candy.
static inline float3 autumnLeaf(float2 w, float birch, uint seed) {
    float k = saturate(perlin2(w / 75.0, seed + 96u) * 1.1 + 0.55 + 0.12 * perlin2(w / 19.0, seed + 97u));
    k = mix(k, 0.28, 0.75 * birch);
    float3 green = float3(0.055, 0.075, 0.03), gold = float3(0.22, 0.15, 0.035), orange = float3(0.21, 0.09, 0.035);
    float3 scarlet = float3(0.18, 0.04, 0.025), crimson = float3(0.12, 0.018, 0.018);
    if (k < 0.2) return mix(green, gold, smoothstep(0.0, 0.2, k));
    if (k < 0.45) return mix(gold, orange, smoothstep(0.2, 0.45, k));
    if (k < 0.7) return mix(orange, scarlet, smoothstep(0.45, 0.7, k));
    return mix(scarlet, crimson, smoothstep(0.7, 1.0, k));
}

// Sedimentary bedding by world height: broad formations, beds of uneven tone and fine laminae. Beds dip gently
// with a slow horizontal warp, so a layer runs continuously across neighbouring columns and around canyon walls.
// Bed edges widen and fine terms fade with the pixel footprint fp (voxels per pixel).
// The bands' phase also wanders over ~55 voxels (tilting beds ~10 degrees typically, ~14 at the 90th percentile), so
// rock bodies more than ~55 voxels apart carry different bands: with one phase for all, the bands on a near tower meet
// those on the rock behind at the same screen rows near eye level and it reads as see-through. Bodies closer than that
// still share bands; parting them needs a steeper tilt. Pure world space: a phase that moved with fp would slide the
// bands as the camera approaches.
// `wall` (0..1) marks the great walls of mega relief, whose rock reads as formations from foot to rim.
// Formation tones (cap rock, wall rock, shale, buff) by index. Selects, not a local array: indexing one per pixel puts
// it in scratch memory, which costs more than all the noise around it.
static inline float3 formationWarm(int i) {
    return i == 0 ? float3(0.62, 0.56, 0.47) : i == 1 ? float3(0.55, 0.31, 0.21) : i == 2 ? float3(0.34, 0.31, 0.30) : float3(0.60, 0.47, 0.33);
}
static inline float3 formationCool(int i) {
    return i == 0 ? float3(0.48, 0.47, 0.44) : i == 1 ? float3(0.27, 0.28, 0.31) : i == 2 ? float3(0.42, 0.36, 0.29) : float3(0.20, 0.21, 0.24);
}

// lite (reflections): rippled water blurs a 30-voxel dip wobble of a fraction of a voxel away.
static inline float3 strataTone(float y, float2 w, uint seed, float desertAmt, float fp, float wall = 0.0, bool lite = false) {
    float yd = y + 1.2 * perlin2(w / 140.0, seed + 24u) + (lite ? 0.0 : 0.3 * perlin2(w / 30.0, seed + 25u));
    // Beds of uneven thickness (the phase runs faster and slower up the face) with soft contacts: an even, hard-edged
    // 16-voxel zebra reads as posterised banding on a tall face.
    float k = 0.5 + 0.5 * sin(yd * 0.38 + 5.0 * perlin2(float2(yd / 41.0, 0.27), seed + 39u)
                             + 2.2 * perlin2(w * 0.01, seed + 22u) + 1.3 * perlin2(w * 0.003, seed + 23u)
                             + 3.0 * perlin2(w / 55.0, seed + 45u));
    k = smoothstep(0.1, 0.9, k);
    // On the great walls the regular 16-voxel bands would read as a zebra stripe; the formations carry the tone there.
    float fz = max(smoothstep(120.0, 300.0, y), wall);
    k = mix(k, 0.5, 0.9 * fz);
    float3 grey = mix(float3(0.22, 0.23, 0.25), float3(0.35, 0.34, 0.33), k);
    float3 sand = mix(float3(0.56, 0.36, 0.24), float3(0.70, 0.52, 0.37), k);
    float3 base = mix(grey, sand, desertAmt);
    // Formations: on the great walls, broad units ~200 tall of pale cap rock, red or grey wall rock, shale and buff, so
    // an 800-voxel face reads as geology rather than one repeated stripe. Absent from the lowland.
    if (fz > 0.0) {
        float f = saturate(perlin2(float2(yd / 380.0 + 0.6 * perlin2(w / 2600.0, seed + 29u), 0.13), seed + 30u) * 0.8 + 0.5) * 3.0;
        float fi = min(floor(f), 2.0), ff = smoothstep(0.35, 0.65, f - fi);
        float wm = saturate(desertAmt + 0.5);
        float3 fa = mix(formationCool(int(fi)), formationWarm(int(fi)), wm), fb = mix(formationCool(int(fi) + 1), formationWarm(int(fi) + 1), wm);
        base = mix(base, mix(fa, fb, ff), 0.55 * fz);
        // Weathering: stains and lichen in broad drifts, so a great face never reads as one flat grey plane. Taken in
        // the face's own plane (across it and up it), never by ground position alone, which would streak a wall
        // from foot to rim.
        float along = w.x + w.y;
        float st = perlin2(float2(along / 80.0, yd / 45.0), seed + 34u)
                 + (fp < 4.0 ? 0.5 * perlin2(float2(along / 26.0, yd / 17.0), seed + 35u) * (1.0 - smoothstep(1.0, 4.0, fp)) : 0.0);
        base *= 1.0 + 0.22 * fz * st;
    }
    // Past here every term has faded to its mean.
    if (fp >= 1.8) return base;

    float yb = yd / 2.4;
    float bi = floor(yb), bf = yb - bi;
    float e = clamp(fp * 0.8, 0.04, 0.45);
    uint sb = seed + 26u;
    float t0 = hashf(hash2i(int2(int(bi), 0), sb)), t1 = hashf(hash2i(int2(int(bi) + 1, 0), sb));
    float bed = mix(t0, t1, smoothstep(1.0 - e, 1.0, bf)) - 0.5;
    float bedFade = (1.0 - smoothstep(0.6, 1.8, fp)) * (1.0 - 0.5 * wall);
    float lam = fp < 0.05 ? sin(yd * 21.0 + 3.0 * perlin2(w / 11.0, seed + 27u)) * (1.0 - smoothstep(0.015, 0.05, fp)) : 0.0;
    float parting = (1.0 - smoothstep(0.0, 0.03, min(bf, 1.0 - bf))) * (1.0 - smoothstep(0.02, 0.06, fp));
    // Desert beds swing between rust and pale cream; grey rock between warm umber and cool slate.
    float3 lo = mix(float3(1.04, 0.99, 0.94), float3(1.04, 0.88, 0.70), desertAmt);
    float3 hi = mix(float3(0.95, 0.98, 1.04), float3(1.10, 1.07, 1.00), desertAmt);
    float3 tint = mix(lo, hi, saturate(0.5 + bed * 1.8));
    float lum = 1.0 + bedFade * 0.22 * bed + 0.05 * lam - 0.10 * parting;
    return base * mix(float3(1.0), tint, bedFade) * lum;
}

// Painted-hill beds: 1D Perlin noise in a gently dipping height picks each voxel's place on a ramp of
// charcoal, rust, ochre, cream and lavender-grey; bed edges soften with the pixel footprint fp.
static inline float2 paintedLateral(float2 w, uint seed) {
    return float2(14.0 * perlin2(w / 520.0, seed + 114u) + 1.5 * perlin2(w / 90.0, seed + 115u), 0.35 * perlin2(w / 900.0, seed + 118u));
}

static inline float3 paintedPal(int i) {
    return i == 0 ? float3(0.21, 0.19, 0.18) : i == 1 ? float3(0.43, 0.24, 0.17) : i == 2 ? float3(0.53, 0.41, 0.26)
         : i == 3 ? float3(0.53, 0.47, 0.39) : float3(0.45, 0.40, 0.40);
}

// `lat` is the column's (dip offset, lateral facies shift) from paintedLateral.
static inline float3 paintedTone(float y, float2 lat, uint seed, float fp) {
    float yd = y + lat.x;
    float v = perlin2(float2(yd / 11.0, 0.37), seed + 116u) + (fp < 1.8 ? 0.3 * perlin2(float2(yd / 4.0, 0.71), seed + 117u) * (1.0 - smoothstep(0.6, 1.8, fp)) : 0.0) + lat.y;
    float t = saturate(v * 0.75 + 0.5) * 4.0;
    float i = min(floor(t), 3.0), f = t - i;
    float e = clamp(fp * 0.5, 0.12, 0.5);
    return mix(paintedPal(int(i)), paintedPal(int(i) + 1), smoothstep(0.5 - e, 0.5 + e, f));
}

// Wind ripples on sand. Crests are level sets of a Perlin-warped wind field, so they bend and pinch across a dune,
// and creep slowly downwind. Returns (albedo offset, slope along the wind) with amplitude already applied.
static inline float3 sandRipples(int2 v, float2 uv, float2 w, float fp, float time, uint seed) {
    if (fp >= 2.0) return float3(0);
    const float2 wind = float2(0.6, 0.8);
    const float lambda = 0.8;
    // Integer part of the phase in exact arithmetic: float world coords lose the ripple far from the origin.
    int ip = (3 * v.x + 4 * v.y) % 20;
    float2 q = w + uv;
    float ipm = float(ip + (ip < 0 ? 20 : 0));
    float psi = dot(uv, wind) + 7.0 * perlin2(q / 26.0, seed + 60u) + 1.6 * perlin2(q / 7.0, seed + 61u);
    float phase = ipm / (5.0 * lambda) + psi / lambda - time * 0.02;
    // Low dune swells, five ripples apart, carry the pattern into the middle distance.
    float swell = sin((ipm / 20.0 + psi / 4.0 - time * 0.004) * 6.2831853) * (1.0 - smoothstep(0.4, 1.4, fp));
    float x = phase * 6.2831853;
    // Long gentle stoss, short steep lee.
    float r = sin(x) + 0.35 * sin(2.0 * x - 0.9);
    float dr = (cos(x) + 0.7 * cos(2.0 * x - 0.9)) * 6.2831853 / lambda;
    float calm = fp < 0.24 ? smoothstep(-0.35, 0.35, perlin2(q / 48.0, seed + 62u)) : 0.0;
    float amp = mix(0.25, 1.0, calm) * (1.0 - smoothstep(0.08, 0.24, fp));
    // Streaks drawn out along the wind give broad plains a calm grain at any distance.
    float along = dot(q, wind), across = dot(q, float2(-wind.y, wind.x));
    float streak = perlin2(float2(along / 70.0, across / 7.0), seed + 63u) * (1.0 - smoothstep(0.5, 2.0, fp));
    return float3(0.045 * amp * r + 0.035 * swell + 0.05 * streak, 0.018 * amp * dr * wind);
}

// 0..1 where world xz w lies among mega relief, from the coarsest ring's max over ~128-voxel cells (lowland rarely rises
// past ~190). Interpolated between cell corners, each the max of the four cells meeting there, so every column rates at
// least its own cell's relief and the field stays continuous: interpolated between cell centres, a wall's foot columns
// past a cell edge rated partly as lowland, and a jogging face crossfaded column by column between the formations and the
// lowland's 16-voxel beds, which near eye level line up with the beds on every rock behind and read as see-through rock.
// Wider cells would carry the walls' rock a whole cell out over the lowland at their feet.
static inline float megaWall(thread const Terrain& tr, float2 w) {
    int kc = tr.levels - 1;
    if (kc < 1) return 0.0;
    int m = max(7 - kc, 0), mask = (COARSE_N >> m) - 1;
    float2 q = w / float(1 << (kc + m));
    int2 c = int2(floor(q));
    float2 f = q - float2(c);
    uint sl = uint(kc - 1), mp = uint(m);
    #define MW(i, j) tr.ch.read(uint2((c + int2(i, j)) & mask), sl, mp).r
    float a0 = MW(-1, -1), a1 = MW(0, -1), a2 = MW(1, -1);
    float b0 = MW(-1, 0), b1 = MW(0, 0), b2 = MW(1, 0);
    float c0 = MW(-1, 1), c1 = MW(0, 1), c2 = MW(1, 1);
    #undef MW
    float v00 = max(max(a0, a1), max(b0, b1)), v10 = max(max(a1, a2), max(b1, b2));
    float v01 = max(max(b0, b1), max(c0, c1)), v11 = max(max(b1, b2), max(c1, c2));
    return smoothstep(190.0, 270.0, mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y));
}

// A giant's bark, leaves or vines (surfaceAt), returned before the ground's shading: none of its tones apply.
static Surface giantSurface(thread const Terrain& tr, Surface s, int k, int3 v, int3 n, float2 uv, float fp, float2 wq,
                                   float wy, float3 col, uint gsb, float gLoc, Giant gi, float cs, int2 mapOrigin, uint seed) {
    float y = float(v.y);
    s.autumn = 0.0;
    float runLo = gLoc, runHi = col.x;
    int gm = GM_WOOD;
    if (k == 0) {
        float4 e = extraGaps(tr.gx.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))));
        float3 g0 = float3(e.z, e.x, col.y), g1 = float3(e.w, e.y, col.z);
        int ri = 0, nG = 0;
        bool found = false;
        for (int i = 0; i < 3; i++) {
            if (g1[i] <= g0[i]) continue;
            nG++;
            if (found) continue;
            if (y >= g1[i]) { ri++; runLo = g1[i]; }
            else { runHi = g0[i]; found = true; }
        }
        float leafStart = float((gsb >> 11) & 2047u);
        int rm = ri < 3 ? int((gsb >> (22u + 2u * uint(ri))) & 3u) : GM_WOOD;
        gm = ri == nG && y >= leafStart ? GM_LEAF : rm;
        if (ri == nG && gm == GM_LEAF) runLo = leafStart;
    } else {
        float2 q = float2(v.xz * int(cs) + mapOrigin) + 0.5 * cs - gi.c;
        bool above = col.z > col.y && y >= col.z;
        // Bark only on the trunk; elsewhere a solid coarse column is the crown's skirt of leaves and vines.
        gm = above || length(q) > 40.0 || (y - gi.G > gi.yT * 0.55 && length(q) > 24.0) ? GM_LEAF : GM_WOOD;
        runLo = above ? col.z : gLoc;
    }
    float3 a;
    if (gm == GM_LEAF) {
        float kq = saturate(perlin2(wq / 31.0, seed + 601u) * 0.9 + 0.5);
        uint pal = k == 0 ? (gsb >> 28) & 7u : giantPalette(gi.h);
        a = pal == 2u ? mix(float3(0.024, 0.070, 0.060), float3(0.050, 0.098, 0.066), kq)
                      : mix(float3(0.034, 0.078, 0.050), float3(0.075, 0.112, 0.042), kq);
        // A rust flush drifts over part of the crown, strongest on its upper leaves.
        if (pal == 3u) a = mix(a, float3(0.115, 0.062, 0.030), 0.7 * smoothstep(0.05, 0.45, perlin2(wq / 23.0, seed + 604u) + 0.25 * saturate((y - runLo) / max(runHi - runLo, 1.0))));
        int2 tx = int2(floor(uv * 16.0));
        int3 tk = n.y != 0 ? int3(v.x * 16 + tx.x, v.y * 2 + max(n.y, 0), v.z * 16 + tx.y)
                : n.x != 0 ? int3(v.x * 2 + max(n.x, 0), v.y * 16 + tx.y, v.z * 16 + tx.x)
                           : int3(v.x * 16 + tx.x, v.y * 16 + tx.y, v.z * 2 + max(n.z, 0));
        uint th = hash3i(tk, seed + 94u);
        float tex = hashf(th) < 0.35 ? 0.72 : 1.0 + 0.12 * hashf(pcg(th));
        a *= mix(tex, 0.92, smoothstep(0.015, 0.04, fp));
        // Darker with depth under the run's top: a clump's side darkens down from its crest the way the dome turns
        // from the sky, rather than by each column's own span (which striped the clumps column by column).
        a *= mix(1.15, 0.62, saturate((runHi - 1.0 - y) / 14.0));
        s.mat = 1;
        s.canopy = 1.0;
    } else if (gm == GM_VINE) {
        a = float3(0.036, 0.050, 0.028) * (0.85 + 0.3 * hashf(hash2i(v.xz, seed + 602u)));
        s.mat = 1;
        s.canopy = mix(1.0, 0.75, smoothstep(6.0, 80.0, col.x - y));
    } else {
        // Bark in vertical ridges, mossed where it faces up and low down.
        a = float3(0.115, 0.098, 0.082);
        float fade = 1.0 - smoothstep(0.1, 0.5, fp);
        if (fade > 0.0) a *= 1.0 + 0.40 * fade * perlin3(float3(wq.x / 1.6, wy / 12.0, wq.y / 1.6), seed + 600u);
        float moss = saturate((n.y > 0 ? 0.75 : 0.05) + 0.6 * perlin3(float3(wq.x, wy * 0.7, wq.y) / 6.0, seed + 603u)
                              - 0.5 * smoothstep(gLoc + 15.0, gLoc + 140.0, wy));
        a = mix(a, float3(0.062, 0.092, 0.040), moss);
        s.mat = 2;
        s.canopy = mix(1.0, 0.75, smoothstep(6.0, 80.0, col.x - y));
    }
    s.albedo = a;
    s.tread = a;
    s.face = a;
    s.H = runHi;
    s.pale = 0;
    s.slope = 0;
    s.level = 0;
    return s;
}

constant float3 PROP_DYES[5] = { float3(0.36, 0.35, 0.32), float3(0.235, 0.097, 0.085), float3(0.08, 0.092, 0.158),
                                 float3(0.319, 0.247, 0.145), float3(0.116, 0.158, 0.11) };
constant float3 WOOL[3] = { float3(0.17, 0.08, 0.055), float3(0.12, 0.12, 0.08), float3(0.19, 0.14, 0.075) };

static float3 propShade(Part P, float3 a, int feat, int3 v, int3 n, float2 uv, float2 wq, float wy, float hs, float Ht,
                        float fade, int2 sc, int2 mapOrigin, uint sh, uint seed, float autumn) {
    float floorY = float(P.y0), fy = wy - floorY;
    int style = (P.flags >> 12) & 15;
    if (feat != FT_POST) {
        int type = P.y1;
        bool az = P.sz.y > P.sz.x;
        int2 pr0 = sc + P.o;
        int pi = az ? v.z - pr0.y : v.x - pr0.x, pj = az ? v.x - pr0.x : v.z - pr0.y;
        int PL = max(P.sz.x, P.sz.y);
        bool endFace = n.y == 0 && (az ? n.z != 0 : n.x != 0);
        uint ch = pcg(sh + uint(P.y2) * 13u + uint(pi) * 7u + uint(pj));
        if (type == PR_WOODPILE || type == PR_LOGSEAT || type == PR_STUMP) {
            // Sawn ends: heartwood rings in bark; a woodpile's logs lie across it, so its long faces show their ends.
            bool ends = type == PR_WOODPILE ? n.y == 0 && !endFace : type == PR_LOGSEAT ? endFace : n.y > 0;
            if (ends) {
                float2 e = type == PR_WOODPILE ? float2(fract(hs * 2.0 + 0.5 * fmod(floor(wy * 2.0), 2.0)) - 0.5, fract(wy * 2.0) - 0.5) * 2.0
                         : type == PR_LOGSEAT ? float2(fract(hs) - 0.5, wy - floorY - 0.35) * 2.4 : (fract(wq) - 0.5) * 2.2;
                float r = length(e);
                float3 wood = float3(0.31, 0.25, 0.17) * (0.92 + 0.08 * sin(r * 22.0));
                float3 endc = r < 0.62 ? wood : r < 0.8 ? float3(0.15, 0.11, 0.07) : float3(0.05, 0.04, 0.03);
                a = mix(type == PR_WOODPILE ? float3(0.19, 0.13, 0.08) : float3(0.24, 0.17, 0.10), endc, fade);
            }
        } else if (type == PR_BALES) {
            a *= 1.0 - 0.5 * fade * (n.y == 0 && !endFace ? smoothstep(0.08, 0.02, abs(fract(hs * 1.5) - 0.5) - 0.2) : 0.0);
        } else if (type == PR_WASH) {
            a = PROP_DYES[ch % 5u] * (1.0 - 0.12 * fade * step(0.5, fract(hs * 3.0)));
        } else if (type == PR_RACK) {
            float3 item = style == ST_ADOBE || style == ST_PUEBLO ? float3(0.26, 0.08, 0.05)
                        : style == ST_TIPI ? float3(0.32, 0.24, 0.15) : float3(0.24, 0.24, 0.23);
            float strand = step(fract(hs * 3.0), 0.6) * (n.y == 0 ? 1.0 : 0.5);
            a = mix(mix(float3(0.04, 0.035, 0.03), item, 0.6), mix(float3(0.04, 0.035, 0.03), item, strand), fade);
        } else if (type == PR_BEDROLL) {
            float3 wool = WOOL[uint(P.y2) % 3u];
            a = pi == 0 ? wool * 0.85 : wool;
            if (pi != 0 && n.y > 0) a = mix(a, wool * 0.7, 0.6 * fade * step(abs(fract(az ? wq.y : wq.x) - 0.3), 0.08));
        } else if (type == PR_RUG) {
            float2 lq = wq - float2(mapOrigin) - float2(pr0);
            float2 hsz = float2(P.sz) * 0.5;
            float edge = min(hsz.x - abs(lq.x - hsz.x), hsz.y - abs(lq.y - hsz.y));
            float dm = abs(lq.x - hsz.x) + abs(lq.y - hsz.y);
            float3 field = PROP_DYES[1u + 2u * (uint(P.y2) % 2u)], border = PROP_DYES[(uint(P.y2) % 2u) != 0u ? 1 : 3], motif = PROP_DYES[0];
            float3 pat = edge < 0.3 ? border : dm < 0.7 ? motif : field;
            a = mix(field * 0.8 + border * 0.2, pat, fade);
        } else if (type == PR_CANOE) {
            bool inside = n.y > 0 && pi > 0 && pi < PL - 1;
            a = inside ? float3(0.04, 0.03, 0.02) : mix(float3(0.13, 0.09, 0.06), float3(0.24, 0.17, 0.10), step(Ht - 0.15, wy) * fade);
        } else if (type == PR_BARREL) {
            a = float3(0.16, 0.10, 0.055);
            float hoop = n.y == 0 ? max(step(abs(wy - floorY - 0.2), 0.06), step(abs(wy - floorY - 0.8), 0.06)) : step(0.35, length(fract(wq) - 0.5)) * step(length(fract(wq) - 0.5), 0.42);
            float stave = n.y == 0 ? step(0.44, abs(fract(hs * 4.0) - 0.5)) : 0.0;
            a *= 1.0 - fade * max(0.65 * hoop, 0.35 * stave);
        } else if (type == PR_PEAT) {
            a = float3(0.075, 0.055, 0.04) * (1.0 - 0.3 * fade * step(0.42, abs(fract(wy * 3.0) - 0.5)));
        } else if (type == PR_GARDEN) {
            if (feat == FT_FENCE) {
                a = float3(0.16, 0.12, 0.08);
            } else if (Ht - floorY < 0.4) {
                a = float3(0.12, 0.085, 0.055) * (0.9 + 0.2 * hashf(ch));
            } else {
                uint ck = hash2i(int2(floor(wq * 2.0)), seed + 350u);
                float3 leafc = float3(0.08, 0.15, 0.05) * (0.8 + 0.4 * hashf(ck));
                if (autumn > 0.2 && hashf(pcg(ck)) < 0.18) leafc = float3(0.40, 0.17, 0.03);
                a = mix(float3(0.09, 0.14, 0.05), leafc, fade);
            }
        }
    }
    if (P.y1 == PR_OVEN && n.y == 0) {
        // The oven's mouth, a dark arch low on one side.
        int m = int(uint(P.y2) & 3u);
        int2 md = m == 0 ? int2(1, 0) : m == 1 ? int2(-1, 0) : m == 2 ? int2(0, 1) : int2(0, -1);
        int2 rc = v.xz - (sc + P.o + 1);
        if (all(n.xz == md) && all(rc == md) && fy < 0.8 && abs(uv.x - 0.5) < 0.3)
            a = mix(a, float3(0.03, 0.02, 0.015), fade + (1.0 - fade) * 0.5);
    }
    return a;
}

// Surface of level k's column v (xz in level cells) on face n; uv is the face position in [0, 1) of the column.
// Coarse columns are s voxels wide but still unit voxels tall, and carry no structures.
// lite: seen in rippled water, so detail finer than the ripples' blur is shaded at its mean.
static Surface surfaceAt(thread const Terrain& tr, texture2d<uint> st, device const Site* sites, device const float4* lampState,
                         int k, int3 v, int3 n, float2 uv, float fp, float fpA, float3 p, uint seed, float time, bool lite = false) {
    if (lite) { fp = max(fp, 4.0); fpA = max(fpA, 4.0); }
    Surface s;
    s.slope = 0;
    s.mat = 0;
    s.level = 0;
    s.pale = 0;
    s.wall = 0;
    s.face = 0;
    s.nrm = 0;
    s.emit = 0;
    s.isleUnder = false;
    s.gy = p.y;
    int2 mapOrigin = tr.origin;
    float cs = tr.lod[k].w;
    float4 col4 = columnL(tr, k, v.xz);
    float3 col = col4.rgb;
    bool tree = col4.a > 0.0;
    // A structure stands on this column from voxel `base` up; below it (or below its gap) is the terrain.
    float base = -col4.a;
    bool built = col4.a < 0.0 && base < col.x;
    // A giant: on the 1-voxel map its columns carry GIANT_A, with the ground and materials in st; further out its
    // columns are those within its reach, wood and leaf from a little above the ground at its axis.
    Giant gi;
    float2 cdK = coverDepthL(tr, k, v.xz);
    bool giant = k == 0 ? col4.a == GIANT_A : cdK.y >= 64.0 && giantAt(tr.giants, float2(v.xz * int(cs) + mapOrigin) + 0.5 * cs, cs, gi);
    uint gsb = giant && k == 0 ? st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r : 0u;
    float gLoc = k == 0 ? float(gsb & 2047u) : 0.0;
    if (giant && k > 0) gLoc = gi.G + 6.0;
    bool gWood = giant && float(v.y) >= gLoc;
    // A coarse forested column's top is its leaves (coarseVeg).
    float vegLo = col.x - cdK.y;
    bool vegLeaf = k > 0 && !giant && cdK.y > 0.0 && float(v.y) >= vegLo;
    // Bark and leaves: the ground's own tones (rock strata, painted beds) would be worked out only to be replaced.
    bool vegVox = gWood || vegLeaf || (tree && float(v.y) >= col.y);
    float H = giant ? (gWood ? col.x : gLoc) : tree ? col.y : built ? (col.z > col.y ? col.y : base) : col.x;
    // Just under a rock gap the voxel belongs to the floor below it: its height is the gap's floor. Further down it is
    // part of the wall the column makes as a whole, which shades as one with its neighbours'.
    if (!tree && !built && !giant && col.z > col.y && float(v.y) < col.y) {
        float seg = col.y;
        if (k == 0) {
            float4 e = extraGaps(tr.gx.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))));
            if (e.y > e.x && float(v.y) < e.x) seg = e.x;
            if (e.w > e.z && float(v.y) < e.z) seg = e.z;
        }
        if (float(v.y) >= seg - 6.0) H = seg;
    }
    s.canopy = tree ? 0.55 : 1.0;
    s.H = H;
    // An island's body: its top is meadow, coloured as the lowland under it is (its height would read as upland rock).
    bool isleCol = !tree && !built && !giant && col.z > col.y && isleGapL(tr, k, v.xz);
    float Hc = isleCol && float(v.y) >= col.z ? col.y + 4.0 : H;
    bool isleRoot = false;
    s.isleUnder = isleCol && float(v.y) < col.z;
    // An island's own rock sees its neighbours whole: the ground under them is the ground's business.
    if (isleCol && float(v.y) >= col.z) s.gy = 1e9;
    float hx0, hx1, hz0, hz1;
    float4 frontCol = 0.0;
    if (k == 0) {
        // The neighbours' whole columns: their ground, and whether any holds a gap, from the same reads.
        float4 a0 = columnL(tr, 0, v.xz + int2(-1, 0)), a1 = columnL(tr, 0, v.xz + int2(1, 0));
        float4 a2 = columnL(tr, 0, v.xz + int2(0, -1)), a3 = columnL(tr, 0, v.xz + int2(0, 1));
        hx0 = a0.a > 0.0 ? a0.g : a0.r; hx1 = a1.a > 0.0 ? a1.g : a1.r;
        hz0 = a2.a > 0.0 ? a2.g : a2.r; hz1 = a3.a > 0.0 ? a3.g : a3.r;
        // Under an island's edge the ground sees the island's own ground beside it, not a cliff to its top.
        float vy = min(float(v.y), s.gy);
        if (a0.a == 0.0 && a0.b > a0.g && vy < a0.b && isleGapL(tr, 0, v.xz + int2(-1, 0))) hx0 = a0.g;
        if (a1.a == 0.0 && a1.b > a1.g && vy < a1.b && isleGapL(tr, 0, v.xz + int2(1, 0))) hx1 = a1.g;
        if (a2.a == 0.0 && a2.b > a2.g && vy < a2.b && isleGapL(tr, 0, v.xz + int2(0, -1))) hz0 = a2.g;
        if (a3.a == 0.0 && a3.b > a3.g && vy < a3.b && isleGapL(tr, 0, v.xz + int2(0, 1))) hz1 = a3.g;
        // A root hangs well below a neighbour's underside (a keel's beds step by seven at most).
        if (isleCol && vy >= col.z) {
            float hang = max(max(a0.a == 0.0 && a0.b > a0.g ? a0.b : 0.0, a1.a == 0.0 && a1.b > a1.g ? a1.b : 0.0),
                             max(a2.a == 0.0 && a2.b > a2.g ? a2.b : 0.0, a3.a == 0.0 && a3.b > a3.g ? a3.b : 0.0));
            isleRoot = vy < hang - 8.0;
        }
        // Beside a multi-run column (a giant's, an emergent's) the ground does not see its top as a wall: it would
        // light and paint a ring round every crown's footprint as a cliff foot.
        if (!giant) {
            if (a0.a == GIANT_A) hx0 = H;
            if (a1.a == GIANT_A) hx1 = H;
            if (a2.a == GIANT_A) hz0 = H;
            if (a3.a == GIANT_A) hz1 = H;
        }
        // Rock gaps only: canopies and buildings hold theirs with a non-zero alpha, and an island's air is no roof.
        s.gapSpan = float2(1e9, -1e9);
        float4 cs5[5] = { col4, a0, a1, a2, a3 };
        int2 o5[5] = { int2(0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
        for (int i = 0; i < 5; i++)
            if (cs5[i].b > cs5[i].g && cs5[i].a == 0.0 && (i == 0 ? !isleCol : !isleGapL(tr, 0, v.xz + o5[i])))
                s.gapSpan = float2(min(s.gapSpan.x, cs5[i].g), max(s.gapSpan.y, cs5[i].b));
        s.gapNear = s.gapSpan.y > s.gapSpan.x;
        frontCol = n.x < 0 ? a0 : n.x > 0 ? a1 : n.z < 0 ? a2 : a3;
    } else {
        float vy = min(float(v.y), s.gy);
        hx0 = groundL(tr, k, v.xz + int2(-1, 0), vy); hx1 = groundL(tr, k, v.xz + int2(1, 0), vy);
        hz0 = groundL(tr, k, v.xz + int2(0, -1), vy); hz1 = groundL(tr, k, v.xz + int2(0, 1), vy);
        float4 fr = n.y == 0 ? columnL(tr, k, v.xz + n.xz) : float4(0);
        frontCol = fr;
        bool cg0 = col.z > col.y && !isleCol, fg = fr.z > fr.y && !isleGapL(tr, k, v.xz + n.xz);
        s.gapNear = cg0 || fg;
        s.gapSpan = float2(min(cg0 ? col.y : 1e9, fg ? fr.y : 1e9), max(cg0 ? col.z : -1e9, fg ? fr.z : -1e9));
    }
    s.nh = float4(hx0, hx1, hz0, hz1);
    float slope = max(max(H - hx0, H - hx1), max(H - hz0, H - hz1));
    float2 w = float2(v.xz * int(cs) + mapOrigin);
    // Continuous world position on the face, so tonal detail flows across block edges.
    float2 wq = w + (n.y != 0 ? uv : n.x != 0 ? float2(n.x > 0 ? 1.0 : 0.0, uv.x) : float2(uv.x, n.z > 0 ? 1.0 : 0.0)) * cs;
    float wy = n.y != 0 ? float(v.y + max(n.y, 0)) : float(v.y) + uv.y;
    float nb = n.x < 0 ? hx0 : n.x > 0 ? hx1 : n.z < 0 ? hz0 : hz1;
    s.riser = n.y == 0 ? H - nb : 0.0;
    if (gWood) return giantSurface(tr, s, k, v, n, uv, fp, wq, wy, col, gsb, gLoc, gi, cs, mapOrigin, seed);
    Climate c = climateL(tr, k, (float2(v.xz) + 0.5) * cs);
    // An island is green to its rim, wherever its edge reaches out over drier land.
    if (Hc != H) { c.temp = clamp(c.temp, 0.48, 0.6); c.moist = max(c.moist, 0.66); }
    float desert = desertAmount(c);
    // Painted-hill beds belong to low mounds; on the great walls their 11-voxel bands would read as stripes.
    float wallM = megaWall(tr, w);
    s.wall = wallM;
    float painted = smoothstep(0.0, 0.6, paintedAmount(c)) * (1.0 - wallM);
    float snowline = snowlineAt(c.temp) + 5.0 * perlin2(w / 160.0, seed + 21u);
    // On the ranges rock breaks through the lower snowfields in drifts, so a massif is not one white sheet.
    if (Hc > 150.0) snowline += 60.0 * perlin2(w / 700.0, seed + 28u) * smoothstep(150.0, 400.0, Hc)
                              + 260.0 * saturate(perlin2(w / 90.0, seed + 33u) * 0.9 + 0.45) * smoothstep(150.0, 450.0, Hc);
    c.temp = altitudeTemp(c.temp, Hc);
    float cold = smoothstep(0.42, 0.18, c.temp);
    float autumn = autumnAmount(c);
    s.autumn = autumn;
    float birch = 0.0;
    float3 fallLeaf = 0.0;
    if (autumn > 0.0) {
        birch = birchAmount(w, seed);
        fallLeaf = autumnLeaf(w, birch, seed);
    }
    // A side face looking into a rock gap (a shelter's back wall) is rock under a roof too: by its neighbours' tops it
    // would read as a low riser and take the tread colour, column by column.
    // An island's air is no roof.
    bool underGap = !tree && !built && !giant && ((col.z > col.y && float(v.y) < col.z && !isleCol)
                    || (n.y == 0 && frontCol.a == 0.0 && frontCol.b > frontCol.g && float(v.y) >= frontCol.g && float(v.y) < frontCol.b
                        && !isleGapL(tr, k, v.xz + n.xz)));

    float salt = saltAmount(c);
    float rimW = smoothstep(0.05, 0.4, salt);

    // Beaches and beds take sand or crust, faces included, so they need none of the ground and rock tones.
    bool beach = H <= SEA + 1.2;
    // Rock tone shows on side and under faces, and on high treads; a low tread never reads it.
    bool needStone = !beach && !vegVox && (!(n.y > 0 && !underGap) || Hc - SEA > 0.55 * 110.0);
    float3 stone = needStone ? strataTone(wy, wq, seed, max(desert, rimW), fpA, wallM, lite) : float3(0);

    float lagoon = smoothstep(0.0, 0.35, lagoonAmount(c));
    // Pale ground (sand, salt, snow) sits well below the clouds, and drifts in tone so open flats never read as one sheet.
    float drift = 1.0 + 0.12 * perlin2(wq / 110.0, seed + 32u);
    float3 sand = mix(float3(0.33, 0.29, 0.21), float3(0.37, 0.33, 0.245), lagoon) * drift;
    float3 snow = float3(0.31, 0.325, 0.35) * drift;
    float3 tread = sand;
    if (!beach) {
        float3 grassWet = float3(0.12, 0.20, 0.13), grassDry = float3(0.40, 0.34, 0.16), grassCold = float3(0.20, 0.23, 0.17);
        float3 grass = mix(mix(grassDry, grassWet, smoothstep(0.3, 0.7, c.moist)), grassCold, cold);
        // Autumn ground: tawny grass, with fallen leaves drifting out from the groves in the canopy's own hues.
        if (autumn > 0.0) {
            float litter = smoothstep(0.08, 0.36, perlin2((w + seedOffset(seed)) / 300.0, seed + 90u));
            float3 fallen = mix(fallLeaf * 1.5, float3(0.30, 0.15, 0.07), 0.45);
            grass = mix(grass, mix(float3(0.22, 0.18, 0.09), fallen, litter), autumn * 0.8);
        }
        grass = mix(grass, float3(0.14, 0.20, 0.12), lagoon);
        float3 desertTop = float3(0.54, 0.34, 0.19);
        tread = mix(grass, desertTop, desert) * (0.95 + 0.1 * perlin2(wq / 40.0, seed + 30u));
        // Shoreline benches alternate rust and ochre with height, drifting slowly along the rim.
        float benchK = saturate(0.5 + 0.9 * perlin2(float2(H * 0.23 + w.x / 400.0, w.y / 400.0), seed + 18u));
        tread = mix(tread, mix(float3(0.48, 0.25, 0.15), float3(0.62, 0.46, 0.25), benchK), rimW);
        // Broad, soft mottling: a little warmer and drier in one drift, cooler and lusher in the next.
        float mott = fpA < 2.0 ? perlin2(wq / 13.0, seed + 31u) * (1.0 - smoothstep(0.6, 2.0, fpA)) : 0.0;
        tread *= 1.0 + mott * float3(0.05, 0.035, -0.01);
        // Gentle hypsometric tint, like a printed topo map: sage valleys, paler ochre-grey uplands.
        float alt = saturate((Hc - SEA) / 110.0);
        tread *= mix(float3(0.93, 1.03, 0.90), float3(1.10, 1.04, 0.98), alt);
        tread = mix(tread, stone * 1.35, 0.30 * smoothstep(0.55, 1.0, alt));
        // Hypsometric banding, as on a printed topo map: each 6-voxel elevation band steps its tint slightly,
        // so broad hillsides read as layered contour bands rather than one repeated stair.
        float bandK = fract(floor(H / 6.0) * 0.618034);
        tread *= mix(float3(1.12, 1.04, 0.90), float3(0.90, 0.98, 1.10), bandK);
        if (painted > 0.0 && !vegVox) {
            float2 p = w + seedOffset(seed);
            // Alluvium on the valley floors: pale, dusty sage.
            float fl = paintedFloor(p, seed);
            float floorK = smoothstep(fl + 2.0, fl + 0.5, H);
            float2 lat = paintedLateral(w, seed);
            float3 bed = mix(paintedTone(H - 1.0, lat, seed, fpA), float3(0.44, 0.40, 0.28), floorK);
            tread = mix(tread, bed, painted);
            if (needStone) stone = mix(stone, paintedTone(float(v.y), lat, seed, fpA), painted);
        }
    }
    // Salt crust: cream with slow Perlin drifts, and a faint polygon net from two crossing sets of
    // Perlin zero-lines that fades long before it could alias.
    float crustW = beach ? smoothstep(0.3, 0.6, salt) : 0.0;
    if (crustW > 0.0) {
        float3 crust = mix(float3(0.32, 0.305, 0.27), float3(0.33, 0.29, 0.27), saturate(perlin2(wq / 210.0, seed + 23u) + 0.5))
                     * (drift + 0.07 * perlin2(wq / 60.0, seed + 19u) + 0.04 * perlin2(wq / 17.0, seed + 20u));
        if (fpA < 0.2) {
            float lw = 0.06 + fpA * 0.6;
            float net = max(1.0 - smoothstep(0.0, lw, abs(perlin2(wq / 2.4, seed + 21u))),
                            1.0 - smoothstep(0.0, lw, abs(perlin2(OCT_ROT * wq / 2.4 + 13.7, seed + 22u))));
            crust *= 1.0 - 0.045 * net * (1.0 - smoothstep(0.05, 0.2, fpA));
        }
        // Broader pressure ridges carry the net into the middle distance.
        if (fpA < 0.8) {
            float ridge = 1.0 - smoothstep(0.0, 0.04 + fpA * 0.08, abs(perlin2(wq / 15.0, seed + 24u)));
            crust *= 1.0 - 0.035 * ridge * (1.0 - smoothstep(0.3, 0.8, fpA));
        }
        // Wet salt is greyer: pool floors, and a damp margin that shades into each pool.
        float lvl = poolLevel(salt);
        float damp = H < PAN ? 1.0 : smoothstep(lvl - 0.14, lvl, poolNoise(wq + seedOffset(seed), seed));
        crust *= mix(float3(1.0), float3(0.80, 0.82, 0.85), damp);
        tread = mix(tread, crust, crustW);
        // The pan is dead level: lit by the smoothed heightfield, or with the lip and pool floor lit as
        // walls and corners, every pool would be outlined.
        s.level = crustW;
    }
    // Coarse levels carry no trees: the share of a column under canopy tints its ground with their leaves.
    if (k > 0) {
        float2 cd = cdK;
        float cov = saturate(cd.x * COVER_GAIN);
        if (cov > 0.0) {
            float conifer = smoothstep(0.50, 0.32, c.temp);
            float3 leaf = mix(float3(0.05, 0.08, 0.0375), float3(0.025, 0.055, 0.045), conifer);
            leaf = mix(leaf, float3(0.11, 0.125, 0.09), painted);
            leaf = mix(leaf, fallLeaf, autumn * (1.0 - conifer * 0.5)) * 0.92;
            tread = mix(tread, leaf, cov);
            s.canopy = mix(1.0, 0.7, cov);
        }
    }
    // Worn ground round a dwelling: paths between doors and hearth, and the apron round a fire.
    // A giant's column (GIANT_A) holds its own word in st, not a clearing's.
    if (k == 0 && col4.a < 0.0 && !built && !giant) {
        uint mk = (st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r >> S_MARK) & 3u;
        if (mk != 0u) {
            float3 earth = mix(mix(float3(0.20, 0.135, 0.08), float3(0.16, 0.13, 0.10), cold), float3(0.50, 0.36, 0.22), desert);
            earth = mix(earth, sand * 0.88, beach ? 1.0 : lagoon * 0.6);
            float br = fpA < 0.3 ? perlin2(wq / 1.3, seed + 342u) * (1.0 - smoothstep(0.05, 0.3, fpA)) : 0.0;
            // Worn patchily, column by column, thinning out to the fringe.
            float wear = mix(0.55 + 0.6 * hashf(hash2i(v.xz + mapOrigin, seed + 343u)), 0.85, smoothstep(0.1, 0.4, fpA));
            float w = (mk == MK_APRON ? 0.62 : mk == MK_FRINGE ? 0.26 : 0.75) * wear;
            tread = mix(tread, earth * 0.9, saturate(w + 0.2 * br));
        }
    }
    s.tread = tread;
    bool snowy = float(int(Hc)) > snowline;
    // Snow risers prefilter to snow, except on the great walls, whose faces stay rock under snowy ledges.
    if (snowy && wallM < 0.5) s.riser = 0.0;
    // Snow risers prefilter to snow; eased across the snowline, which far off is one straight contour.
    s.tread = mix(tread, snow, smoothstep(snowline - 6.0, snowline + 6.0, Hc));
    s.face = needStone ? (snowy ? mix(stone, snow, 0.45 * (1.0 - smoothstep(6.0, 30.0, slope))) : stone * 0.95) : s.tread;

    if (n.y > 0 && !underGap) {
        if (snowy) s.albedo = snow;
        else s.albedo = tread;
        float sandy = snowy ? 0.0 : beach ? 1.0 - crustW : smoothstep(0.35, 0.7, desert);
        s.pale = snowy || beach ? 1.0 : sandy;
        if (sandy > 0.0) {
            float3 rp = (k == 0 ? sandRipples(v.xz + mapOrigin, uv, w, fpA, time, seed)
                                : sandRipples(int2(floor(p.xz)) + mapOrigin, fract(p.xz), float2(int2(floor(p.xz)) + mapOrigin), fpA, time, seed)) * sandy;
            s.albedo *= 1.0 + rp.x;
            s.slope = rp.yz;
        }
    } else if (n.y == 0 && !underGap) {
        s.pale = beach || snowy ? 1.0 : 0.0;
        if (beach) s.albedo = tread * 0.85;
        else {
            // Low risers half take the tread; a column's deepest drop alone would make a staircase wall of mixed step
            // heights alternate per column (dashes down the face), so on steep relief the slope across the
            // neighbours decides too.
            float steepC = 0.5 * length(float2(hx0 - hx1, hz0 - hz1));
            float rockW = snowy ? 1.0 : max(step(4.0, slope), max(smoothstep(1.0, 2.5, steepC), wallM));
            // Snow lies on the treads; a tall wall stays rock under it.
            float3 rock = snowy ? mix(stone, snow, 0.45 * (1.0 - smoothstep(6.0, 30.0, slope))) : stone;
            s.albedo = mix(mix(mix(tread, stone, 0.5) * 0.9, stone * 0.95, painted), rock, rockW);
        }
    } else {
        s.albedo = stone * 0.9;
    }
    // Roots and vines under an island: umber and olive, a clump at a time.
    if (isleRoot) {
        s.albedo = mix(float3(0.075, 0.058, 0.04), float3(0.058, 0.07, 0.035), hashf(hash2i(v.xz >> 1, seed + 712u)));
        s.tread = s.albedo;
        s.face = s.albedo;
    }
    if ((tree && float(v.y) >= col4.a) || vegLeaf) {
        float leafLo = vegLeaf ? vegLo : col4.a;
        // Leaves: deep, low-chroma greens that drift slowly across a grove; conifers cooler and darker.
        float conifer = smoothstep(0.50, 0.32, c.temp);
        float k = saturate(perlin2(w / 23.0, seed + 93u) * 0.9 + 0.5);
        float3 leaf = mix(float3(0.035, 0.07, 0.035), float3(0.065, 0.09, 0.04), k);
        leaf = mix(leaf, float3(0.025, 0.055, 0.045), conifer);
        leaf = mix(leaf, float3(0.11, 0.125, 0.09), painted);
        leaf = mix(leaf, fallLeaf, autumn * (1.0 - conifer * 0.5));
        if (c.jungle > 0.0) {
            // The jungle's crowns differ one from the next: deep green, olive, now and then a cool teal.
            float jk = perlin2(w / 9.0, seed + 340u) + 0.6 * perlin2(w / 31.0, seed + 341u);
            float3 jl = mix(float3(0.028, 0.062, 0.034), float3(0.072, 0.094, 0.032), smoothstep(-0.3, 0.5, jk));
            jl = mix(jl, float3(0.024, 0.068, 0.058), smoothstep(0.25, 0.6, perlin2(w / 17.0 + 7.3, seed + 342u)));
            leaf = mix(leaf, jl, smoothstep(0.1, 0.4, c.jungle));
        }
        // A 16x16 leaf texture: two-tone texels, faded out once a texel nears a pixel.
        if (fp < 0.04) {
        int2 tx = int2(floor(uv * 16.0));
        int3 tk = n.y != 0 ? int3(v.x * 16 + tx.x, v.y * 2 + max(n.y, 0), v.z * 16 + tx.y)
                : n.x != 0 ? int3(v.x * 2 + max(n.x, 0), v.y * 16 + tx.y, v.z * 16 + tx.x)
                           : int3(v.x * 16 + tx.x, v.y * 16 + tx.y, v.z * 2 + max(n.z, 0));
        uint th = hash3i(tk, seed + 94u);
        float tex = hashf(th) < 0.35 ? 0.72 : 1.0 + 0.12 * hashf(pcg(th));
        leaf *= mix(tex, 0.92, smoothstep(0.015, 0.04, fp));
        } else leaf *= 0.92;
        // Upper leaves catch more sky.
        leaf *= mix(0.85, 1.15, saturate((float(v.y) - leafLo) / max(col.x - leafLo - 1.0, 1.0)));
        s.albedo = leaf;
        s.tread = leaf;
        s.H = col.x;
        s.mat = 1;
        s.canopy = 1.0;
    } else if (tree && float(v.y) >= col.y) {
        s.albedo = float3(0.06, 0.058, 0.052);
        s.mat = 2;
        if (autumn > 0.3 && birch > 0.5) {
            // Birch bark: chalk white, broken by dark lenticel bands.
            uint bh = hash3i(int3(v.x, int(floor(wy * 3.0)), v.z), seed + 98u);
            s.albedo = mix(hashf(bh) < 0.22 ? float3(0.06, 0.055, 0.05) : float3(0.30, 0.29, 0.27), float3(0.24, 0.23, 0.215), smoothstep(0.04, 0.12, fp));
            s.mat = 3;
        }
        s.tread = s.albedo;
        s.canopy = 1.0;
    }
    if (giant && !gWood && col.x > gLoc + 4.0) s.canopy = 0.7;
    // Over a gap the structure starts part way up its lowest voxel.
    if (built && float(v.y) >= (col.z > col.y ? floor(base) : base)) {
        uint sb = st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r;
        int mat = int(sb & 15u);
        uint pk = (sb >> S_PART) & 31u;
        device const Site& site = sites[(sb >> S_SITE) & 511u];
        Part P = unpackPart(site.parts[pk]);
        uint sh = uint(site.head.w);
        float age = site.info.y;
        float Ht = col.x;
        int style = (P.flags >> 12) & 15;
        int feat = int((sb >> S_FEAT) & 15u);
        bool chim = (sb & S_CHIMNEY) != 0u;
        bool tent = P.kind == P_HOUSE && (P.flags & F_TENT) != 0;
        bool house = P.kind == P_HOUSE && !tent;
        int2 sc = site.head.xy - mapOrigin;
        float floorY = float(P.y0);
        float k = hashf(pcg(sh + pk * 7919u + uint(P.y2)));
        // Fine pattern fades to its mean before it nears a pixel.
        float fade = 1.0 - smoothstep(0.02, 0.07, fp);
        // This voxel's material, and the plane a roof or a round wall shades as (zero: its own face).
        int vm = mat;
        bool roof = false;
        float3 rn = 0;
        // Roof coordinates: along the fall line, and along the eave.
        float2 ra = wq;
        HouseAt ha;
        if (house) {
            houseAt(P, v.xz, sc, ha);
            if (chim && (ha.jw == (ha.Dw - 1) / 2 || wy < Ht - ha.f.lip)) vm = M_STONE;
            else if (feat == FT_VIGA || (feat == FT_POST && (style != ST_CABIN || wy < Ht - roofStep(ha) - 0.25))) vm = M_TIMBER;
            else if (feat == FT_DECK) vm = M_PLANK;
            else if (float(v.y) < floorY) vm = style == ST_STILT ? M_LOG : M_STONE;
            else if (!ha.f.flat) {
                bool inWall = ha.iw >= 0 && ha.jw >= 0 && ha.iw < ha.Lw && ha.jw < ha.Dw;
                // The roof course is as deep as the steepest step, so every riser of the roof is roof.
                float thick = max(1.0, roofStep(ha)) + 0.3;
                bool gableWall = !ha.f.hip && ha.dG == 0;
                float under = gableWall ? Ht - thick : min(floorY + ha.f.wall, Ht - thick);
                if (!inWall || wy >= under) {
                    roof = true;
                    vm = style == ST_TIMBER || style == ST_CABIN ? M_SHINGLE : M_THATCH;
                    bool along = ha.f.hip && ha.dG < ha.dA;
                    float side = along ? (ha.iw * 2 < ha.Lw - 1 ? -1.0 : 1.0)
                                       : (ha.jw * 2 < ha.Dw - 1 ? -1.0 : ha.jw * 2 > ha.Dw - 1 ? 1.0 : 0.0);
                    float slope = ha.dA < -1 ? 0.5 : roofProfile(ha).y;
                    bool useX = ha.axisZ != along;
                    rn = normalize(float3(useX ? side * slope : 0.0, 1.0, useX ? 0.0 : side * slope));
                    ra = useX ? wq : wq.yx;
                }
            }
        } else if (tent) {
            int4 ij = partIJ(P, v.xz, sc);
            // Posts show as timber under the roof they carry.
            if (feat == FT_POST && (style == ST_AFRAME || wy < Ht - 0.5)) vm = M_TIMBER;
            else {
                if (feat == FT_POST) vm = style == ST_PALM ? M_THATCH : M_FELT;
                roof = true;
                bool axisZ = (P.flags & F_AXISZ) != 0;
                float side = 0.0, slope = 0.0;
                if (style == ST_AFRAME) { side = sign(float(ij.y - ij.w / 2)); slope = 1.0; }
                else if (style == ST_LEANTO) {
                    float2 mid = float2(P.o) + float2(P.sz) * 0.5;
                    side = (axisZ ? mid.x : mid.y) > 0.0 ? 1.0 : -1.0;
                    slope = 0.65;
                }
                rn = normalize(float3(axisZ ? side * slope : 0.0, 1.0, axisZ ? 0.0 : side * slope));
                ra = axisZ ? wq : wq.yx;
            }
        } else if (P.kind == P_ROUND) {
            float2 dq = wq - float2(mapOrigin) - float2(sc + P.o) - 0.5;
            float r = length(dq);
            float2 rd = dq / max(r, 1e-3);
            float2 pr = roundProfile(P, r);
            if (feat == FT_POLE && wy >= Ht - 0.75) vm = M_TIMBER;
            // The wall ring shades as a drum, the roof as a dome or cone.
            bool wallFace = n.y == 0 && wy < floorY + float(P.y1) * 0.25;
            rn = wallFace ? float3(rd.x, 0.0, rd.y) : normalize(float3(rd.x * pr.y, 1.0, rd.y * pr.y));
            roof = !wallFace;
        }
        if (P.kind == P_FIRE && (float(v.y) < Ht - 1.0 || mat == M_STONE)) vm = M_STONE;
        if (P.kind == P_PROP && feat == FT_POST) vm = M_TIMBER;
        float fy = wy - floorY;
        // Along a wall face, and the glow of an open door (lit from within at night).
        float hs = n.x != 0 ? wq.y : wq.x;
        float doorM = 0.0;
        float3 a;
        if (vm == M_STONE) {
            // The hill's own rock, lichened with age.
            a = strataTone(wy, wq, seed, desert, fpA) * 0.8;
            if (painted > 0.0) a = mix(a, paintedTone(wy, paintedLateral(w, seed), seed, fpA) * 0.85, painted);
            float3 lichen = mix(mix(float3(0.17, 0.19, 0.14), float3(0.26, 0.15, 0.07), autumn), float3(0.26, 0.27, 0.27), cold);
            float ln = fp < 0.15 ? perlin3(float3(wq.x, wy * 1.3, wq.y) / 1.4, seed + 310u) : 0.0;
            float cov = smoothstep(0.0, 0.3, ln + 0.45 * age - 0.3 + (n.y > 0 ? 0.15 : 0.0));
            cov = mix(cov, 0.35 * age, smoothstep(0.04, 0.15, fp));
            a = mix(a, lichen, cov * (1.0 - desert) * 0.55);
            // Built stone (walls and houses, not the standing stones) is laid in courses, darker at the joints.
            if (P.kind != P_HENGE && P.kind != P_BOX && fp < 0.08) {
                float jy = abs(fract(wy * 2.0) - 0.5), jx = abs(fract((n.x != 0 ? wq.y : wq.x) * 1.5 + floor(wy * 2.0) * 0.5) - 0.5);
                float joint = max(smoothstep(0.42, 0.48, jy), n.y == 0 ? smoothstep(0.44, 0.49, jx) : 0.0);
                a *= 1.0 - 0.25 * joint * (1.0 - smoothstep(0.03, 0.08, fp));
            }
            if (P.kind == P_FIRE) a = mix(a, float3(0.16, 0.135, 0.10), 0.75) * (0.8 + 0.35 * hashf(hash2i(v.xz + mapOrigin, seed + 344u)));
            else if (chim) a = mix(a, float3(dot(a, float3(0.3, 0.59, 0.11))), 0.5) * (wy > Ht - 0.4 ? 0.3 : 0.6);
            else if (P.kind != P_HENGE && P.kind != P_BOX) a *= house && ha.f.flat ? 0.6 : 0.85;
        } else if (vm == M_TIMBER) {
            a = (P.kind == P_PROP ? float3(0.17, 0.125, 0.08) : float3(0.10, 0.075, 0.055)) * (1.0 + 0.15 * sin(wy * 9.0 + perlin2(wq * 3.0, seed + 311u) * 3.0) * (1.0 - smoothstep(0.02, 0.06, fp)));
        } else if (vm == M_THATCH) {
            if (P.kind == P_PROP) a = float3(0.232, 0.202, 0.148) * (0.9 + 0.2 * k);
            else if (style == ST_STILT || style == ST_PALM) a = mix(float3(0.27, 0.23, 0.14), float3(0.19, 0.18, 0.15), 0.5 * age + 0.4 * k);
            else if (style == ST_CROFT) a = mix(float3(0.12, 0.15, 0.08), float3(0.14, 0.12, 0.08), saturate(0.4 * k + 0.25 * cold));
            else a = mix(mix(float3(0.24, 0.18, 0.09), float3(0.16, 0.15, 0.11), 0.7 * k + 0.4 * age), float3(0.15, 0.12, 0.07), 0.4 * autumn);
            // Bundles laid in courses up the slope, combed down it.
            float course = hashf(hash2i(int2(int(floor((n.y != 0 ? ra.x : wy) * 2.0)), int(floor(ra.y * 0.8))), seed + 330u));
            a *= 1.0 + fade * (0.14 * (course - 0.5) + 0.10 * perlin2(float2(ra.y * 4.0, wy * 0.7), seed + 312u));
        } else if (vm == M_SHINGLE) {
            a = (style == ST_CABIN ? float3(0.25, 0.165, 0.10) : float3(0.20, 0.12, 0.075)) * (0.9 + 0.2 * k);
            float moss = saturate(perlin2(wq / 2.5, seed + 331u) + 0.5) * age;
            a = mix(a, float3(0.09, 0.11, 0.05), (style == ST_CABIN ? 0.25 : 0.5) * mix(moss, 0.5 * age, 1.0 - fade));
            // Rows of shingles, each row's joints offset half a shingle.
            float rw = (n.y != 0 ? ra.x : wy) * 2.0;
            float u = ra.y * 1.5 + 0.5 * float(int(floor(rw)) & 1);
            float seam = max(smoothstep(0.40, 0.48, abs(fract(rw) - 0.5)), smoothstep(0.44, 0.49, abs(fract(u) - 0.5)));
            a *= 1.0 - 0.3 * seam * (1.0 - smoothstep(0.03, 0.08, fp));
        } else if (vm == M_LOG) {
            float along = n.x != 0 ? wq.y : wq.x;
            if (style == ST_STILT) {
                // Bamboo: pale canes standing side by side.
                a = float3(0.26, 0.21, 0.13) * (0.9 + 0.15 * k);
                a *= 1.0 - 0.18 * fade * smoothstep(0.36, 0.48, abs(fract(along * 3.0) - 0.5));
            } else {
                // Log courses with dark chinking between them.
                float lc = fract(wy * 2.0);
                a = (P.kind == P_PROP ? float3(0.14, 0.105, 0.075) : float3(0.21, 0.14, 0.085)) * (0.9 + 0.2 * k);
                a *= mix(1.0, 0.8 + 0.25 * sin(lc * 3.14159265) - 0.3 * smoothstep(0.38, 0.5, abs(lc - 0.5)), fade);
            }
        } else if (vm == M_PLASTER) {
            a = float3(0.24, 0.22, 0.19) * (0.95 + 0.08 * perlin2(wq / 3.0 + wy, seed + 313u));
            float3 beam = float3(0.08, 0.055, 0.04);
            // A dark timber sill; timber-framed houses show their posts and rails over the infill.
            float frame = smoothstep(0.25, 0.2, fy);
            if (style == ST_TIMBER) {
                float along = n.x != 0 ? wq.y : wq.x;
                frame = max(frame, max(smoothstep(0.42, 0.36, abs(fract(along * 0.5) - 0.5)) , smoothstep(2.8, 2.85, fy)));
                frame = max(frame, smoothstep(0.08, 0.03, abs(fy - 1.6)));
            }
            a = mix(a, beam, n.y == 0 ? mix(frame, 0.2, 1.0 - fade) : 0.0);
        } else if (vm == M_ADOBE) {
            a = mix(float3(0.44, 0.28, 0.16), float3(0.50, 0.36, 0.22), smoothstep(0.5, 2.5, fy));
            if (style == ST_PUEBLO) a = mix(a, paintedTone(wy, paintedLateral(w, seed), seed, fpA), 0.35);
            a *= (0.94 + 0.08 * perlin2(wq / 2.0 + wy * 0.5, seed + 313u)) * mix(0.82, 1.0, smoothstep(0.2, 0.5, fy));
        } else if (vm == M_FELT) {
            float3 band = 0;
            bool banded = false;
            if (P.kind == P_ROUND && style == ST_TIPI) {
                a = float3(0.32, 0.25, 0.17) * (0.9 + 0.15 * k);
                band = k < 0.5 ? float3(0.22, 0.06, 0.03) : float3(0.05, 0.07, 0.12);
                banded = fy > 0.75 && fy < 1.25;
            } else if (P.kind == P_ROUND) {
                a = float3(0.28, 0.26, 0.22) * (0.92 + 0.1 * k);
                band = k < 0.5 ? float3(0.24, 0.05, 0.04) : float3(0.06, 0.07, 0.18);
                banded = n.y == 0 && fy > float(P.y1) * 0.25 - 0.5 && fy < float(P.y1) * 0.25;
                if (feat == FT_CROWN && wy >= Ht - 0.5) a *= 0.5;
            } else if (style == ST_CARAVAN) {
                a = k < 0.5 ? float3(0.22, 0.09, 0.06) : float3(0.09, 0.09, 0.15);
            } else {
                a = k < 0.4 ? float3(0.15, 0.13, 0.09) : k < 0.75 ? float3(0.11, 0.12, 0.07) : float3(0.18, 0.09, 0.05);
            }
            if (banded) a = mix(a, band, fade * 0.85 + 0.15);
            // A painted door in a yurt's wall.
            if (P.kind == P_ROUND && style != ST_TIPI && n.y == 0) {
                float2 dq = wq - float2(mapOrigin) - float2(sc + P.o) - 0.5;
                float da = hashf(pcg(sh + pk)) * 6.2831853;
                if (dot(normalize(dq), float2(cos(da), sin(da))) > 0.96 && fy < 1.6) { a = float3(0.30, 0.12, 0.03); doorM = 0.6; }
            }
            // An A-frame's flap stands open on the gable facing the fire.
            if (style == ST_AFRAME && n.y == 0) {
                int4 ij = partIJ(P, v.xz, sc);
                bool az = (P.flags & F_AXISZ) != 0;
                float2 mid = float2(P.o) + float2(P.sz) * 0.5;
                bool gable = az ? float(n.z) * mid.y < 0.0 : float(n.x) * mid.x < 0.0;
                int rj = abs(ij.y - ij.w / 2);
                if (gable && rj < 2 && wy - base < (rj == 0 ? 1.75 : 0.75)) { a = float3(0.03, 0.025, 0.02); doorM = 1.0; }
            }
        } else if (vm == M_PLANK) {
            float seam = n.y > 0 ? 1.0 - smoothstep(0.0, 0.08, abs(fract(((P.flags & F_AXISZ) != 0 ? wq.y : wq.x) * 3.0) - 0.5) - 0.42) : 0.0;
            a = float3(0.12, 0.10, 0.08) * (1.0 - 0.3 * seam * (1.0 - smoothstep(0.02, 0.06, fp)));
        } else if (vm == M_CLOTH) {
            uint ch = pcg(sh + uint(P.y2)) % 3u;
            a = ch == 0u ? float3(0.22, 0.20, 0.17) : ch == 1u ? float3(0.24, 0.16, 0.08) : float3(0.10, 0.11, 0.15);
        } else {
            // A fire's bed: charred ends in grey ash.
            a = n.y > 0 && length(uv - 0.5) < 0.3 ? float3(0.045, 0.035, 0.025) : float3(0.12, 0.105, 0.09);
        }
        if (P.kind == P_PROP) {
            a = propShade(P, a, feat, v, n, uv, wq, wy, hs, Ht, fade, sc, mapOrigin, sh, seed, autumn);
            // Yard clutter settles toward the ground's tone once a voxel nears a pixel.
            a = mix(a, tread, 0.6 * smoothstep(0.2, 0.8, fpA));
        }
        // Roofs snow where the ground they stand on does: a few voxels up they would wear a white lid over green land.
        if (roof && (vm == M_THATCH || vm == M_SHINGLE || vm == M_FELT) && snowy) a = float3(0.29, 0.30, 0.33);
        if (P.kind == P_BOAT && n.y > 0) a *= 0.55;
        // Windows and doors on the wall faces of houses.
        float winM = 0.0;
        if (house && n.y == 0 && !roof && !chim && float(v.y) >= floorY && feat == 0) {
            uint side = n.x < 0 ? 0u : n.x > 0 ? 1u : n.z < 0 ? 2u : 3u;
            int op = houseOpening(P, ha, side);
            float wfy = fy - (op == 3 ? 3.0 : 0.0) - (ha.f.wall >= 3.0 ? 0.2 : 0.0);
            float e = clamp(fpA * 0.6, 0.03, 0.25);
            if (op == 1 || op == 3) {
                float mx = smoothstep(0.28 - e, 0.28 + e, uv.x) * smoothstep(0.72 + e, 0.72 - e, uv.x);
                // The height band is not foreshortened when the wall is seen edge-on, so the glow keeps it and
                // only the band across the wall fades to its mean.
                float ey = clamp(fp * 0.6, 0.03, 0.25);
                float my = smoothstep(0.55 - ey, 0.55 + ey, wfy) * smoothstep(1.45 + ey, 1.45 - ey, wfy);
                float m = mix(mx * my, 0.2, smoothstep(0.05, 0.3, fpA));
                a = mix(a, float3(0.025, 0.025, 0.03), m);
                winM = mix(mx, 0.44, smoothstep(0.05, 0.3, fpA)) * mix(my, 0.3, smoothstep(0.3, 1.0, fp));
            }
            if (op == 2) {
                float m = smoothstep(0.25 - e, 0.25 + e, uv.x) * smoothstep(0.75 + e, 0.75 - e, uv.x) * smoothstep(1.8 + e, 1.8 - e, wfy);
                a = mix(a, float3(0.045, 0.035, 0.028), mix(m, 0.4, smoothstep(0.05, 0.3, fpA)));
                doorM = mix(m, 0.4, smoothstep(0.05, 0.3, fpA)) * 0.6;
            }
        }
        int3 ld = house && ha.f.flat && n.y == 0 ? flatLadder(style, uint(P.y2), ha.Lw, ha.Dw) : int3(-1);
        if (ld.x >= 0 && ha.iw == ld.x && ha.jw == ld.y && fy >= float(ld.z)) {
            bool want = style == ST_PUEBLO ? (ha.axisZ ? n.x > 0 : n.z > 0) : (ha.axisZ ? n.z > 0 : n.x > 0);
            if (want) {
                float rail = step(abs(abs(uv.x - 0.5) - 0.28), 0.07);
                float rung = step(abs(uv.x - 0.5), 0.28) * step(fract(fy * 2.5), 0.2);
                float shadow = step(abs(abs(uv.x - 0.5) - 0.38), 0.05);
                float3 wood = float3(0.20, 0.14, 0.085);
                a = mix(a * mix(1.0, 0.92, 1.0 - fade), mix(a * (1.0 - 0.35 * shadow), wood, max(rail, rung)), fade);
                winM = 0.0;
            }
        }
        // Lamps: flame on a fire's top course, a lantern atop its post, light in the windows.
        int li = ((P.flags >> 8) & 15) - 1;
        if (li >= 0) {
            int lk = int(site.lamps[li].w) % 16;
            float3 E = lampState[((sb >> S_SITE) & 511u) * uint(MAX_LAMPS) + uint(li)].rgb * LAMP_EMIT[lk];
            bool topCourse = float(v.y) >= Ht - 1.0;
            if (P.kind == P_FIRE && topCourse && mat == M_FIRE) {
                float flame = perlin3(float3(wq * 2.5, wy * 1.6 - time * 1.4), seed + 320u) * 0.5 + 0.5;
                flame = mix(flame, 0.5, smoothstep(0.05, 0.3, fpA));
                // Per-channel tonemapping bleaches a hot core to white; held this low, the core stays orange and
                // the flame noise carries it.
                s.emit = E * float3(1.0, 0.7, 0.5) * (n.y > 0 ? mix(0.05, 0.24, flame) : mix(0.02, 0.17, flame) * (0.45 + 0.55 * fract(wy)));
            } else if (P.kind == P_BOX && topCourse) {
                s.emit = E * float3(1.0, 0.8, 0.6) * (n.y > 0 ? 0.16 : 0.3 * smoothstep(0.3, 0.8, fract(wy)));
            } else if (house || P.kind == P_ROUND || tent) {
                float canvas = lk == L_TENT && (vm == M_FELT || vm == M_THATCH) ? 0.12 : 0.0;
                s.emit = E * (winM + canvas) + E * doorM * float3(1.0, 0.72, 0.45);
            }
        }
        s.albedo = a;
        s.tread = a;
        s.mat = max(vm, 4);
        s.canopy = 1.0;
        s.H = Ht;
        s.pale = 0;
        s.slope = 0;
        s.riser = n.y == 0 ? Ht - nb : 0.0;
        s.level = (sb & S_FLAT) != 0u && n.y > 0 ? 1.0 : 0.0;
        if (any(rn != 0.0)) s.nrm = rn;
    }
    return s;
}

static inline float vao(float s1, float s2, float c) { return (s1 > 0.5 && s2 > 0.5) ? 0.0 : 3.0 - (s1 + s2 + c); }

// Minecraft-style per-corner AO, bilinearly interpolated across the face.
static float cornerAO(texture2d<float> hm, int size, int3 v, int3 n, float2 uv) {
    int3 a = v + n;
    int3 tu = n.x != 0 ? int3(0, 0, 1) : int3(1, 0, 0);
    int3 tv = n.y != 0 ? int3(0, 0, 1) : int3(0, 1, 0);
    float su0 = solid(hm, a - tu, size), su1 = solid(hm, a + tu, size);
    float sv0 = solid(hm, a - tv, size), sv1 = solid(hm, a + tv, size);
    float c00 = solid(hm, a - tu - tv, size), c10 = solid(hm, a + tu - tv, size);
    float c01 = solid(hm, a - tu + tv, size), c11 = solid(hm, a + tu + tv, size);
    float a00 = vao(su0, sv0, c00), a10 = vao(su1, sv0, c10), a01 = vao(su0, sv1, c01), a11 = vao(su1, sv1, c11);
    float ao = mix(mix(a00, a10, uv.x), mix(a01, a11, uv.x), uv.y) / 3.0;
    return mix(0.6, 1.0, pow(ao, 0.8));
}

// Fraction of open sky above a point, from how far the surrounding columns rise above y within 6 voxels.
// Heights are filtered, so a c on a column centre reads exact columns.
// Coarse levels sample their own slice at the same world radii.
// mode 0: 8 directions at 2, 4 and 6; 1 (far: a pixel spans columns): 8 directions at 4; 2 (seen in water): 4 at 4.
static float skyVisibility(thread const Terrain& tr, int k, float2 c, float y, int mode = 0) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    constexpr sampler blr(filter::linear, address::repeat);
    float2 inv = 1.0 / float2(tr.hm.get_width(), tr.hm.get_height());
    float cs = tr.lod[k].w;
    float2 fo = float2(levelOrigin(tr, k) & (COARSE_N - 1));
    float occ = 0.0;
    for (int i = 0; i < 8; i += mode == 2 ? 2 : 1) {
        // (1,0) (-1,0) (0,1) (0,-1) (1,1) (-1,1) (1,-1) (-1,-1), computed: a local table indexed in a loop of dynamic
        // stride lives in scratch memory.
        int2 dir = i < 2 ? int2(i == 0 ? 1 : -1, 0) : i < 4 ? int2(0, i == 2 ? 1 : -1) : int2((i & 1) == 0 ? 1 : -1, i < 6 ? 1 : -1);
        float m = 0.0;
        for (int r = mode > 0 ? 2 : 1; r <= (mode > 0 ? 2 : 3); r++) {
            float dist = float(r * 2) * (i < 4 ? 1.0 : 1.414);
            float2 q = c + float2(dir * (r * 2));
            float rise = (k == 0 ? tr.hx.sample(bl, q * inv, level(0)).r
                                 : tr.ch.sample(blr, (q / cs + fo) / float(COARSE_N), uint(k - 1), level(0)).r) - y;
            m = max(m, rise / sqrt(rise * rise + dist * dist));
        }
        occ += m;
    }
    return 1.0 - occ / (mode == 2 ? 4.0 : 8.0);
}

// Position on the face in [0, 1) of a column s voxels wide (y stays in unit voxels).
static inline float2 faceUV(float3 p, int3 n, float s) {
    float3 f = fract(p / float3(s, 1.0, s));
    if (n.y != 0) return f.xz;
    if (n.x != 0) return float2(f.z, f.y);
    return float2(f.x, f.y);
}

// MARK: - Clouds

// Cumulus base about 1100, over all but the alpine summits and the tepuis, which pierce it. The whole cloud field
// is scaled by CLOUD_S from its original 240-high design, so from cruise height the sky looks as before.
constant float CLOUD_Y = 1100.0;
constant float CLOUD_S = 3.5;
constant float CLOUD_Y0 = CLOUD_Y - 10.0 * CLOUD_S, CLOUD_Y1 = CLOUD_Y + 34.0 * CLOUD_S;

// Coverage = a fine fbm part, which only drifts with the wind, and a slow part (one broad, slowly evolving
// octave plus the cover offset).
constant float2 CLOUD_DRIFT = float2(0.825336, 0.564642) * 4.8; // (cos 0.6, sin 0.6) units per second

static inline float cloudFineAt(float2 qd, uint seed) { return fbm2(qd / (300.0 * CLOUD_S), 3, seed + 40u); }

static inline float cloudFine(float2 q, uint seed, float time) { return cloudFineAt(q - CLOUD_DRIFT * time, seed); }

// The fine field in drifted coordinates, baked once into a texture around the camera.
kernel void cloudField(texture2d<float, access::write> out [[texture(0)]],
                       constant float4& info [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    out.write(float4(cloudFineAt(info.xy + (float2(gid) + 0.5) * info.z, as_type<uint>(info.w))), gid);
}

// The slow part without the cover offset, baked around the camera every frame (cloudSlowBaked reads it).
static inline float cloudSlowRaw(float2 q, uint seed, float time) {
    q -= float2(cos(0.6), sin(0.6)) * time * 4.8;
    return perlin3(float3(q / (520.0 * CLOUD_S), time * 0.004), seed + 41u) * 0.35;
}

kernel void cloudSlowField(texture2d<float, access::write> out [[texture(0)]],
                           constant float4& info [[buffer(0)]],
                           constant uint& seed [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    out.write(float4(cloudSlowRaw(info.xy + (float2(gid) + 0.5) * info.z, seed, info.w)), gid);
}

static inline float cloudSlow(float2 q, uint seed, float time, float cover) {
    q -= float2(cos(0.6), sin(0.6)) * time * 4.8;
    return perlin3(float3(q / (520.0 * CLOUD_S), time * 0.004), seed + 41u) * 0.35 + cover - 0.5;
}

static inline float cloudDensity(float2 q, uint seed, float time, float cover) {
    return cloudFine(q, seed, time) + cloudSlow(q, seed, time, cover);
}


struct Ctx {
    Terrain tr;
    int2 origin;
    uint seed;
    float time;
    float quality;
    float pixelAngle;
    float cover;
    Light L;
    float dither; // per-pixel, per-frame in [0, 1); TAA resolves it
    texture2d<float> cloudTex;
    float3 cloudInfo; // as Frame.cloud
    texture2d<uint> st;
    device const Site* sites;
    device const float4* lampPos;
    device const float4* lampState;
    texture2d<float> lightMap;
    texture2d<float> lightDir;
    float4 lights;
    texture2d<float> shadowNear;  // shadow-height maps, SHADOW_CELL and 4x that, toroidal
    texture2d<float> shadowFar;
    float shadowMode;             // 0 none, 1 maps, 2 brute-force rays (debug)
    bool prof1;
    int prof;
    texture2d<float> slowTex;
    float4 slowInfo;
    uint tiles;                   // haze lamps (bits 0-15) and smoke plumes (16-27) that may touch this pixel's screen tile
    constant TowerTable* towers;  // the ruined towers nearest the camera
    bool isleShadows;             // an island stands in view: the shadow maps hold island shade
    constant IsleTable* isles;    // the islands nearest the camera
};

// Baked fine cloud field where it covers q, else evaluated.
static inline float cloudFineBaked(thread const Ctx& x, float2 q) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    float2 qd = q - CLOUD_DRIFT * x.time;
    float2 u = (qd - x.cloudInfo.xy) / (x.cloudInfo.z * float(x.cloudTex.get_width()));
    if (x.cloudInfo.z > 0.0 && all(u > 0.002) && all(u < 0.998)) return x.cloudTex.sample(bl, u, level(0)).r;
    return cloudFineAt(qd, x.seed);
}

// The slow part from its baked field where that covers q (texels a twentieth of its features), else evaluated.
static inline float cloudSlowBaked(thread const Ctx& x, float2 q) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    float2 u = (q - x.slowInfo.xy) / (x.slowInfo.z * float(x.slowTex.get_width()));
    if (x.slowInfo.z > 0.0 && all(u > 0.002) && all(u < 0.998)) return x.slowTex.sample(bl, u, level(0)).r + x.cover - 0.5;
    return cloudSlow(q, x.seed, x.time, x.cover);
}

static inline float cloudShadow(thread const Ctx& x, float3 p, float3 L) {
    if (L.y <= 0.02 || p.y >= CLOUD_Y1) return 1.0;
    float2 q = p.xz + L.xz / L.y * (CLOUD_Y - p.y) + float2(x.origin);
    float dens = cloudFineBaked(x, q) + cloudSlowBaked(x, q);
    // Land inside the slab is in its cloud; land above it is in sunlight.
    return mix(1.0, mix(0.6, 0.32, smoothstep(0.3, 0.5, L.y)), smoothstep(0.0, 0.14, dens) * smoothstep(CLOUD_Y1, CLOUD_Y0, p.y));
}

// MARK: - Shadow-height maps

// Two world-anchored toroidal grids around the camera (SHADOW_N texels a side; SHADOW_CELL voxels and 4x that):
// each texel holds the height below which its column is in the shadow of terrain at least SHADOW_NEAR away toward
// the key light, from the coarse rings' max mips, and the heights [z, w] the islands toward the light shade (empty:
// z > w), so at a low sun an island's shadow falls clear of the ground beneath it. Rebaked a band per frame; shade()
// traces only the near ray.
constant float SHADOW_CELL = 8.0;
constant int SHADOW_N = 1024;
constant float SHADOW_NEAR = 96.0;

struct ShadowBake {
    float4 key;     // xyz key light direction, w horizontal start distance
    float4 cam;     // xy camera world xz, z cell size, w levels built
    int4 rows;      // x first row of this band, y camera cell x, z camera cell z
};

kernel void bakeShadowHeight(texture2d<float, access::write> out [[texture(0)]],
                             texture2d_array<float> ch [[texture(1)]],
                             texture2d_array<float> cb [[texture(2)]],
                             texture2d_array<float> cg [[texture(3)]],
                             constant ShadowBake& b [[buffer(0)]],
                             uint2 g [[thread_position_in_grid]]) {
    int2 t = int2(g) + int2(0, b.rows.x);
    if (t.x >= SHADOW_N || t.y >= SHADOW_N) return;
    // The window's cell for this texel: the one within half a window of the camera's cell.
    int2 lo = b.rows.yz - SHADOW_N / 2;
    int2 c = lo + ((t - lo) & (SHADOW_N - 1));
    float cell = b.cam.z;
    float2 P = (float2(c) + 0.5) * cell;
    float hl = length(b.key.xz);
    if (hl < 1e-3 || b.key.y <= 0.0) { out.write(float4(-1e4, -1e4, 1e4, -1e4), uint2(t)); return; }
    float2 dir = b.key.xz / hl;
    float tanE = b.key.y / hl;
    int levels = int(b.cam.w);
    // x: occluders from the near ray's reach on; y: from two texels out, for pixels that trace no near ray.
    float S = -1e4, Sa = -1e4, Ilo = 1e4, Ihi = -1e4;
    float d = 2.0 * cell;
    float step = cell;
    float iA = 1e9, iB = -1e9;
    for (int i = 0; i < 96; i++) {
        float2 X = P + dir * d;
        float r = length(X - b.cam.xy);
        // Finest ring that holds X, at the max mip whose cells are as wide as the step.
        int k = 1;
        while (k <= levels && r > 900.0 * float(1 << k)) k++;
        if (k > levels) break;
        int m = max(int(round(log2(step))) - k, 0);
        int2 mc = int2(floor(X / float(1 << k))) >> m;
        float hm = ch.read(uint2(mc & ((2048 >> m) - 1)), uint(k - 1), uint(m)).r;
        float w = float(1 << (k + m));
        // Under an island the ground stops at the band's floor, and the island fills the band's roof to the top.
        float2 isl = float2(1e4, -1e4);
        if (hm < 0.0) {
            hm = -hm;
            float2 bd = cb.read(uint2(mc & ((1024 >> (m - 1)) - 1)), uint(k - 1), uint(m - 1)).rg;
            isl = float2(bd.y, hm);
            hm = bd.x;
        } else if (m == 0 && k <= GAP_LEVELS) {
            float2 gp = cg.read(uint2(mc & 2047), uint(k - 1)).rg;
            if (gp.x < 0.0) { isl = float2(gp.y, hm); hm = -gp.x; }
        }
        // The cell's nearest part lies up to ~0.7 of its width closer than X.
        float hs = hm - max(d - 0.7 * w, cell) * tanE;
        Sa = max(Sa, hs);
        if (d >= b.key.w) S = max(S, hm - max(d - 0.7 * w, b.key.w) * tanE);
        if (isl.y > isl.x) { iA = min(iA, d - w); iB = max(iB, d + w); }
        if ((d - w) * tanE > 2100.0) break;
        d += step;
        if (d > 5.0 * step) step *= 2.0;
        if (d > 7000.0) break;
    }
    // The islands' shade, marched again over their stretch of the ray in cells at most 16 voxels wide (the coarse
    // steps' cells would cut its edge into blocks), from the near ray's reach on, as S: nearer, a cell's max over an
    // island's rolling top would shade the top itself.
    float st = max(cell, 16.0);
    d = max(iA, b.key.w);
    for (int i = 0; i < 160 && d <= iB; i++) {
        float2 X = P + dir * d;
        float r = length(X - b.cam.xy);
        int k = 1;
        while (k <= levels && r > 900.0 * float(1 << k)) k++;
        if (k > levels) break;
        int m = max(int(round(log2(st))) - k, 0);
        int2 mc = int2(floor(X / float(1 << k))) >> m;
        float hm = ch.read(uint2(mc & ((2048 >> m) - 1)), uint(k - 1), uint(m)).r;
        float w = float(1 << (k + m));
        float2 isl = float2(1e4, -1e4);
        if (hm < 0.0) isl = float2(cb.read(uint2(mc & ((1024 >> (m - 1)) - 1)), uint(k - 1), uint(m - 1)).g, -hm);
        else if (m == 0 && k <= GAP_LEVELS) {
            float2 gp = cg.read(uint2(mc & 2047), uint(k - 1)).rg;
            if (gp.x < 0.0) isl = float2(gp.y, hm);
        }
        if (isl.y > isl.x) {
            Ilo = min(Ilo, isl.x - (d + 0.7 * w) * tanE);
            Ihi = max(Ihi, isl.y - max(d - 0.7 * w, b.key.w) * tanE);
        }
        d += st;
    }
    out.write(float4(S, Sa, Ilo, Ihi), uint2(t));
}

// The light an island's shade (the maps' [z, w]) leaves at height y over world xz w, bilinear over the four texels
// round it, each lit or shaded on its own: a filtered interval would blend an empty one into a wrong one.
static float isleShadeLit(texture2d<float> sm, float2 w, float cell, float y, float pen) {
    float2 q = w / cell - 0.5, f = fract(q);
    int2 i0 = int2(floor(q));
    float lit = 0.0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            float2 I = sm.read(uint2((i0 + int2(i, j)) & (SHADOW_N - 1))).zw;
            float in = I.y > I.x ? smoothstep(I.x - pen, I.x + pen, y) * (1.0 - smoothstep(I.y - pen, I.y + pen, y)) : 0.0;
            lit += (1.0 - in) * (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
        }
    return lit;
}

// Fraction of the key light the maps let through to p (map-local); never lighter than a lit column.
static float shadowMapLit(thread const Ctx& x, float3 p, bool all) {
    constexpr sampler rs(filter::linear, address::repeat);
    float2 w = p.xz + float2(x.origin);
    float2 cw = x.tr.cam + float2(x.origin);
    float2 dv = abs(w - cw);
    float rn = max(dv.x, dv.y) / (SHADOW_CELL * float(SHADOW_N / 2));
    float bn = smoothstep(0.80, 0.92, rn);
    float2 Sn = x.shadowNear.sample(rs, w / (SHADOW_CELL * float(SHADOW_N)), level(0)).rg;
    if (bn > 0.0) Sn = mix(Sn, x.shadowFar.sample(rs, w / (4.0 * SHADOW_CELL * float(SHADOW_N)), level(0)).rg, bn);
    float S = all ? Sn.y : Sn.x;
    float pen = mix(6.0, 14.0, bn);
    float lit = smoothstep(S - pen, S + pen, p.y);
    if (x.isleShadows) {
        float li = isleShadeLit(x.shadowNear, w, SHADOW_CELL, p.y, pen);
        if (bn > 0.0) li = mix(li, isleShadeLit(x.shadowFar, w, 4.0 * SHADOW_CELL, p.y, pen), bn);
        lit = min(lit, li);
    }
    return mix(lit, 1.0, smoothstep(0.85, 0.95, rn * 0.25));
}

// Mean light reaching the stairs around p, as seen along d, averaged bilinearly over the four nearest
// columns: a higher neighbour toward the light shades a strip of a column's tread (to contact level ct), and
// risers sit at contact level cr, as the traced shadow had them: its origin, kept inside the hit column, starts
// a side face's ray inside the solid.
// far: a pixel spans several columns, so the nearest column alone stands for them (TAA averages the rest).
static float stairShadow(thread const Terrain& tr, int k, float2 p, float y, float3 L, float3 d, float ct, float cr, bool far = false) {
    float cs = tr.lod[k].w;
    float2 q = p / cs - 0.5, f = fract(q);
    int2 c0 = int2(floor(q));
    int2 sx = int2(L.x > 0.0 ? 1 : -1, 0), sz = int2(0, L.z > 0.0 ? 1 : -1);
    int2 vx = int2(d.x > 0.0 ? 1 : -1, 0), vz = int2(0, d.z > 0.0 ? 1 : -1);
    // Horizontal reach of a unit step's shadow along each axis, in columns.
    float2 reach = abs(L.xz) / max(L.y, 0.02) / cs;
    float lit = 0.0, area = 0.0;
    int n = far ? 0 : 1;
    if (far) { c0 = int2(floor(p / cs)); f = 0.0; }
    for (int j = 0; j <= n; j++)
        for (int i = 0; i <= n; i++) {
            int2 c = c0 + int2(i, j);
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            float h = groundL(tr, k, c, y);
            // A step j columns toward the light covers what its shadow has left after crossing j - 1 columns.
            float fx = 0.0, fz = 0.0;
            for (int j = 1; j <= 2; j++) {
                fx = max(fx, (groundL(tr, k, c + sx * j, y) - h) * reach.x - float(j - 1));
                fz = max(fz, (groundL(tr, k, c + sz * j, y) - h) * reach.y - float(j - 1));
            }
            fx = saturate(fx);
            fz = saturate(fz);
            float tread = 1.0 - (1.0 - (1.0 - fx) * (1.0 - fz)) * (1.0 - ct);
            // Risers on the camera's side of the column, by visible area.
            float rx = max(h - groundL(tr, k, c - vx, y), 0.0) * abs(d.x);
            float rz = max(h - groundL(tr, k, c - vz, y), 0.0) * abs(d.z);
            lit += w * (abs(d.y) * tread + (rx + rz) * cr);
            area += w * (abs(d.y) + rx + rz);
        }
    return lit / area;
}

// Whether column c of level k is hollow at height y (inside one of its rock gaps).
// An island's air is open sky, not a shelter.
static inline bool hollowAt(thread const Terrain& tr, int k, int2 c, float y) {
    float4 col = columnL(tr, k, c);
    if (col.z <= col.y || col.a != 0.0) return false;
    if (y >= col.y && y < col.z) return !isleGapL(tr, k, c);
    if (k > 0 || y >= col.y) return false;
    float4 e = extraGaps(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1)))));
    return (y >= e.x && y < e.y) || (y >= e.z && y < e.w);
}

// The share of open shade's sky light reaching under a roof.
constant float ROOF_SKY = 0.55;

// hollowAt as the trace sees it: the column morphed toward the next level's.
static inline bool hollowAtMorph(thread const Terrain& tr, int k, int2 c, float y) {
    if (columnL(tr, k, c).a != 0.0) return false;
    float3 col = columnMorph(tr, k, c);
    if (col.z <= col.y) return false;
    if (y >= col.y && y < col.z) return !isleGapL(tr, k, c) && parentGapS(tr, k, c).x >= 0.0;
    if (k > 0 || y >= col.y) return false;
    float4 e = extraGapsMorph(tr, c);
    return (y >= e.x && y < e.y) || (y >= e.z && y < e.w);
}

// Near p at height y: x, the share of the ground under a roof (per column, whether it or a 4-neighbour is hollow at y,
// bilinear over the four columns nearest p); yz, which way the hollow lies (its share's gradient over those four). A
// function of position alone, never of which way a face turns, so the jogging faces of a wall under a roof all agree,
// and it fades over a column or two at a roof's edge. Twelve columns, each read once.
static float3 roofField(thread const Terrain& tr, int k, float2 p, float y) {
    float cs = tr.lod[k].w;
    float2 q = p / cs - 0.5, f = fract(q);
    int2 i0 = int2(floor(q));
    // The 4x4 block from i0 - 1, less its corners; bit (j * 4 + i).
    uint hb = 0u;
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            if ((i == 0 || i == 3) && (j == 0 || j == 3)) continue;
            if (hollowAt(tr, k, i0 + int2(i - 1, j - 1), y)) hb |= 1u << uint(j * 4 + i);
        }
    if (hb == 0u) return float3(0);
    float share = 0.0;
    float hv[4];
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint b = uint((j + 1) * 4 + i + 1);
            uint near = (1u << b) | (1u << (b - 1u)) | (1u << (b + 1u)) | (1u << (b - 4u)) | (1u << (b + 4u));
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            if ((hb & near) != 0u) share += w;
            hv[j * 2 + i] = float((hb >> b) & 1u);
        }
    float2 g = float2(mix(hv[1] - hv[0], hv[3] - hv[2], f.y), mix(hv[2] - hv[0], hv[3] - hv[1], f.x));
    return float3(share, g);
}

// MARK: - Tower masonry

// The flicker key of the lit room at w, stage j's loggia (j = 7: the portals), or 0 where the room is dark. A tower that
// lights any lights one or two, a near the middle of a face, so the open rooms near by and the painted bays far off light
// the same ones. (A share of every bay lit read as an inhabited block.)
static inline uint towerLitRoom(Tower T, float2 w, int j) {
    TowerPlan P = towerPlan(T, w - (j < 7 ? towerStageAxis(T, j) : T.c));
    int bay = j < 7 ? int(floor(P.t / towerBayW(T.h, j, P.face, T.bw) + 0.5)) : 0;
    int n = hashf(pcg(T.h + 61u)) < 0.4 ? 2 : 1;
    int step = T.N > 8 ? 3 : 2;
    // Candidates in turn, skipping bays walled up or fallen, until n are chosen.
    for (int i = 0; i < 8; i++) {
        if (n <= 0) break;
        uint r = pcg(T.h + 62u + 5u * uint(i));
        int jj = hashf(r) < 0.3 ? 7 : int(pcg(r + 1u) % uint(max(T.K - 2, 1)));
        int ff = jj == 7 ? int(pcg(r + 2u) % uint(T.N / step)) * step : int(pcg(r + 2u) % uint(T.N));
        int bb = jj == 7 ? 0 : int(pcg(r + 3u) % 3u) - 1;
        if (jj != 7 && (!towerLoggiaBay(T.h, jj, ff, float(bb)) || towerBayFallen(T.h, jj, ff, float(bb)))) continue;
        if (j == jj && P.face == ff && bay == bb) return r | 1u;
        n--;
    }
    return 0u;
}

static inline Tower towerFromRec(TowerRec R) {
    Tower T;
    T.c = R.a.xy; T.R0 = R.a.z; T.Hb = R.a.w;
    T.Ht = R.b.x; T.K = int(R.b.y); T.N = int(R.b.z); T.rot = R.b.w;
    T.th0 = R.c.x; T.th1 = R.c.y; T.Sy = R.c.z; T.bw = R.c.w;
    T.h = R.d.x; T.kind = int(R.d.y);
    T.kb = 1 + int(pcg(T.h + 9u) % 2u);
    T.reach = T.R0 * 2.8;
    return T;
}

// The plane a side face of tower column wc (normal n) shades as, or 0 where it keeps its own (broken masonry). Every
// plane off the grid is a run of voxel jogs; each jog's faces take the plane they make up: the polygon face's (out of
// the wall, or into the court). A jog's face nearly across the plane would otherwise turn to the light on its own and
// rule the wall with a line per jog.
static float3 towerWallPlane(Tower T, float2 wc, float3 n) {
    float seg = 6.2831853 / float(T.N);
    float2 d0 = wc - T.c;
    // The highest stage whose plan holds the column (none: the plinth, the base's polygon).
    int m = -1;
    float Rm = 0.0;
    TowerPlan Pm = towerPlan(T, d0);
    float2 ax = T.c;
    for (int k = 0; k < 6; k++) {
        if (k >= T.K) break;
        float2 a = towerStageAxis(T, k);
        TowerPlan Pk = towerPlan(T, wc - a);
        float Rk = tierR(T, Pk, k);
        if (Pk.reff < Rk + 0.75) { m = k; Rm = Rk; Pm = Pk; ax = a; }
    }
    // Off the stages (plinth, talus, roots) and deep in a stage (the breaks and the rubble on its treads) the masonry is
    // broken, its faces their own.
    if (m < 0 || (m < T.K - 2 && Rm - Pm.reff > 3.0)) return 0.0;
    float fa = T.rot + seg * (float(Pm.face) + 0.5);
    float2 pn = float2(cos(fa), sin(fa));
    // At a corner a jog's face takes the plane its own centre lies nearer: by the column's centre, or by the plane it is
    // nearer square to, a line of the corner's faces lit as the other face ran down it.
    float fb = fa + (Pm.am > 0.0 ? seg : -seg);
    float2 pb = float2(cos(fb), sin(fb)), q = wc + 0.5 * n.xz - ax;
    if (dot(q, pb) > dot(q, pn)) pn = pb;
    float cp = dot(n.xz, pn);
    if (cp > 0.1) return float3(pn.x, 0.0, pn.y);
    if (cp < -0.1) return float3(-pn.x, 0.0, -pn.y);
    // Nearly across the plane: a jog of the outer face, or of the court's wall within it.
    bool inner = m >= T.K - 2 && Pm.reff < Rm - 4.0;
    return inner ? float3(-pn.x, 0.0, -pn.y) : float3(pn.x, 0.0, pn.y);
}

// A lit room's light: faint, a lamp deep inside.
constant float3 TOWER_ROOM = float3(1.0, 0.55, 0.22) * 0.5;

// The light of a lit room on the faces of an opening of tower R at voxel v of level k (0 where none): only the great
// portals and the loggias' bays, a storey and a tenth tall just under a rim; weathered pockets stay dark. Out of line:
// it runs only at night, and inlined it slows every pixel's shading by day.
static __attribute__((noinline)) float3 towerRoomLight(thread const Ctx& x, TowerRec R, int k, int3 v, int3 n, float2 w, float night) {
    float cs = x.tr.lod[k].w;
    float Hb = R.a.w, Sy = R.c.z;
    int K = int(R.b.y);
    uint seed = x.seed;
    float yv = float(v.y);
    float4 here = columnL(x.tr, k, v.xz);
    float4 e = k == 0 ? extraGaps(x.tr.gx.read(uint2(clamp(v.xz, int2(0), int2(x.tr.size - 1))))) : float4(0);
    float2 g = float2(-1.0);
    if (n.y > 0) {
        if (here.b > here.g && here.g == yv + 1.0) g = here.gb;
        else if (e.y > e.x && e.x == yv + 1.0) g = e.xy;
        else if (e.w > e.z && e.z == yv + 1.0) g = e.zw;
    } else if (n.y < 0) {
        if (here.b > here.g && here.b == yv) g = here.gb;
        else if (e.y > e.x && e.y == yv) g = e.xy;
        else if (e.w > e.z && e.w == yv) g = e.zw;
    } else {
        int2 q = v.xz + n.xz;
        float4 f = columnL(x.tr, k, q);
        float4 fe = k == 0 ? extraGaps(x.tr.gx.read(uint2(clamp(q, int2(0), int2(x.tr.size - 1))))) : float4(0);
        if (f.b > f.g && yv >= f.g && yv < f.b) g = f.gb;
        else if (fe.y > fe.x && yv >= fe.x && yv < fe.y) g = fe.xy;
        else if (fe.w > fe.z && yv >= fe.z && yv < fe.w) g = fe.zw;
    }
    int jr = -1;
    Tower T = towerFromRec(R);
    float2 wc = w + 0.5 * cs;
    if (g.y > g.x) {
        for (int j = 0; j < 6; j++) {
            if (j >= K - 2) break;
            float top = tierY(T, towerPlan(T, wc - towerStageAxis(T, j)), j, seed);
            if (g.y <= top - 3.0 && g.y >= top - 7.0 && abs(g.y - g.x - floor(Sy * 1.1)) <= 2.0 + float(k)) jr = j;
        }
        if (abs(g.x - (Hb + 6.0)) <= float(k) && g.y - g.x >= 26.0) jr = 7;
    }
    if (jr >= 0) {
        uint lh = towerLitRoom(T, wc, jr);
        if (lh != 0u) return TOWER_ROOM * night * (0.8 + 0.2 * hashf(pcg(lh))) * flicker(x.time, float(lh & 255u), seed);
    }
    return 0.0;
}

// A tower's masonry in place of the terrain's surface s: each tower its own old stone laid in storey courses, stained
// under its ledges, lichened, darkened by age at the foot; greened on its treads, hung with ivy under its rims (painted
// where the coarse rings carry no leaves), roots over its plinth, and at night a few rooms lit. Every term is taken in
// the face's own plane and fades to its mean with the footprint.
static void towerSurface(thread const Ctx& x, thread Surface& s, int k, int3 v, int3 n, float2 uv, float fp, float fpA, float3 d) {
    if (s.mat != 0) return;
    float cs = x.tr.lod[k].w;
    float2 w = float2(v.xz * int(cs) + x.tr.origin);
    int i = towerIndexAt(x.towers, w + 0.5 * cs);
    if (i < 0) return;
    TowerRec R = x.towers->r[i];
    float Hb = R.a.w, Ht = R.b.x, Sy = R.c.z, bw = R.c.w, R0 = R.a.z;
    int K = int(R.b.y);
    uint th = R.d.x;
    uint seed = x.seed;
    float2 wq = w + (n.y != 0 ? uv : n.x != 0 ? float2(n.x > 0 ? 1.0 : 0.0, uv.x) : float2(uv.x, n.z > 0 ? 1.0 : 0.0)) * cs;
    float wy = n.y != 0 ? float(v.y + max(n.y, 0)) : float(v.y) + uv.y;
    // Along a side face, the distance along the plan's face it lies on: by the face's own axis, the faces of each jog of
    // a diagonal face would take different stone.
    float2 dq = wq - R.a.xy;
    float aq = atan2(dq.y, dq.x) - R.b.w, sq = 6.2831853 / R.b.z;
    float fq = floor(aq / sq);
    float along = n.y != 0 ? wq.x : length(dq) * sin(aq - sq * fq - 0.5 * sq) + 97.0 * fq;
    Climate c = climateL(x.tr, k, (float2(v.xz) + 0.5) * cs);
    float wet = towerWet(c);
    float dry = smoothstep(0.35, 0.7, desertAmount(c));
    // The plan at the column: out from the base's face, and whether the column stands outside the base (roots).
    float2 dc0 = w + 0.5 * cs - R.a.xy;
    float seg = 6.2831853 / R.b.z;
    float ang = atan2(dc0.y, dc0.x) - R.b.w;
    float am = ang - seg * floor(ang / seg) - 0.5 * seg;
    float reff = length(dc0) * cos(am);
    bool off = reff > R0 + 0.5 * cs;
    // Below the plinth's foot it is the ground it was cut from; the base's walls go down to the ground wherever it falls
    // away.
    if (float(v.y) < Hb - 1.0 && off) return;
    bool root = false;
    if (off && k <= 1) {
        // Roots stand over the plinth and over the talus heaped on it.
        Tower T = towerFromRec(R);
        TowerPlan P0 = towerPlan(T, dc0);
        float inner = R0 * (1.2 + 0.08 * perlin2(P0.dir * 3.7, seed + 520u));
        root = wy > Hb + max(reff < inner ? 6.0 : 3.0, 3.0 + towerTalus(T, P0, w + 0.5 * cs, 1, seed).x) + 0.5;
    }
    // A face of the plan's polygon is a plane: its voxel jogs shade as that plane, and every detail fades with the
    // footprint across the plane, not across the jog's own face, or the jogs' two orientations fade apart in stripes.
    float3 pl = n.y == 0 && !root ? towerWallPlane(towerFromRec(R), w + 0.5 * cs, float3(n)) : float3(0);
    if (any(pl != 0.0)) fpA = fp / max(abs(dot(pl, d)), 0.15);
    // Each tower its own stone: pale limestone, warm sandstone, grey granite or rose.
    float u = hashf(pcg(th + 70u));
    float3 stone = u < 0.4 ? float3(0.27, 0.25, 0.215) : u < 0.7 ? float3(0.29, 0.24, 0.185) : u < 0.9 ? float3(0.24, 0.235, 0.225)
                 : float3(0.265, 0.235, 0.215);
    // Broad drifts of tone over the tower and up it.
    float3 m = stone * mix(1.06, 0.9, saturate(perlin2(wq / 90.0 + float(th & 255u), seed + 530u) + 0.5));
    m = mix(m, float3(0.42, 0.33, 0.24), 0.5 * dry);
    m *= (1.0 + 0.12 * perlin2(float2(along / 37.0, wy / 29.0), seed + 531u)) * (1.0 + 0.2 * perlin2(float2(along / 130.0, wy / 95.0), seed + 546u));
    // Age darkens the foot and the wet country's stone, and grime lies over it in broad blotches.
    m *= mix(1.0, 0.72, smoothstep(Hb + 120.0, Hb, wy) * (0.4 + 0.6 * wet));
    m *= mix(0.78, 1.0, smoothstep(-0.35, 0.25, perlin2(float2(along / 30.0, wy / 70.0), seed + 529u) + 0.4 * perlin2(float2(along / 10.0, wy / 25.0), seed + 538u)));
    // Courses: each storey a different tone, a voxel block's own tone close up; the fallen masonry off the stages more so.
    // (Far off, storey bands read as strata.)
    if (fpA < 0.6) {
        float course = hashf(hash2i(int2(int(floor((wy - Hb) / Sy)), int(th & 1023u)), seed + 532u)) - 0.5;
        m *= 1.0 + 0.12 * course * (1.0 - smoothstep(0.15, 0.6, fpA));
    }
    if (fp < 0.3) {
        uint bh = hash3i(v, seed + 533u);
        m *= 1.0 + (off ? 0.3 : 0.16) * (hashf(bh) - 0.5) * (1.0 - smoothstep(0.05, 0.3, fp));
    }
    // Ashlar on the walls: courses a third of a storey high, blocks of 6 to 13 laid in bond, each its own tone; and the
    // weathered crust spalled off in broad patches of paler, fresher stone. No joints are drawn (a grid of lines reads as
    // toy blocks); both fade to their mean with the footprint.
    if (n.y == 0) {
        float fade = 1.0 - smoothstep(0.3, 1.1, fpA);
        if (fade > 0.0) {
            float ch = max(4.0, floor(Sy / 3.0));
            float row = floor((wy - Hb) / ch);
            float rh = hashf(hash2i(int2(int(row), int(th & 1023u)), seed + 554u));
            float len = 6.0 + floor(8.0 * rh);
            float blk = floor((along + row * 3.7) / len);
            m *= (1.0 + 0.12 * (rh - 0.5) * fade) * (1.0 + 0.26 * (hashf(hash3i(int3(int(blk), int(row), int(th & 1023u)), seed + 555u)) - 0.5) * fade);
        }
        float sp = perlin2(float2(along, wy) / 23.0, seed + 556u) + 0.45 * perlin2(float2(along, wy) / 8.0, seed + 557u);
        float spall = mix(smoothstep(0.15, 0.5, sp), 0.15, smoothstep(0.6, 4.0, fpA));
        m *= mix(0.96, 1.07, spall);
    }
    // Lichen in pale crusts on the drier stone, faded to its mean with the footprint.
    float lich = smoothstep(0.25, 0.45, perlin2(float2(along, wy) / 7.0 + (n.y != 0 ? 3.1 : 0.0), seed + 547u) + 0.25 * perlin2(float2(along, wy) / 31.0, seed + 548u));
    lich = mix(lich, 0.2, smoothstep(0.15, 0.6, fpA)) * (1.0 - 0.6 * wet);
    m *= 1.0 + 0.07 * lich;
    float3 moss = mix(float3(0.06, 0.085, 0.04), float3(0.09, 0.11, 0.055), saturate(perlin2(wq / 13.0, seed + 534u) + 0.5));
    float conifer = smoothstep(0.50, 0.32, c.temp);
    float3 leaf = mix(mix(float3(0.035, 0.07, 0.035), float3(0.065, 0.09, 0.04), saturate(perlin2(wq / 23.0, seed + 93u) * 0.9 + 0.5)),
                      float3(0.025, 0.055, 0.045), conifer);
    float snowline = snowlineAt(c.temp);
    // The ledge (stage rim) above this height, by the stages' nominal tops.
    float ledge = Hb + Ht;
    int jl = -1;
    for (int j = 0; j < 6; j++) {
        if (j >= K - 2) break;
        float yk = Hb + Ht * tierTop(th, j, K);
        if (yk >= wy - (n.y > 0 ? 1.0 : 0.0)) { ledge = yk; jl = j; break; }
    }
    if (root) {
        // Roots: grey bark under moss, mossed over their tops.
        float3 bark = mix(float3(0.17, 0.145, 0.115), float3(0.12, 0.11, 0.09), saturate(perlin2(wq / 5.0, seed + 549u) + 0.5));
        m = n.y > 0 ? mix(bark, moss, 0.6 + 0.3 * wet) : mix(bark, moss, 0.25 * wet);
    } else if (n.y > 0) {
        if (wy > snowline) m = float3(0.31, 0.325, 0.35);
        else {
            // Moss and turf over the treads, more in wet country.
            float cov = smoothstep(-0.25, 0.25, perlin2(wq / 6.0, seed + 535u) + 0.5 * perlin2(wq / 23.0, seed + 536u) + 0.6 * wet - 0.3);
            cov = mix(cov, saturate(0.2 + 0.6 * wet), smoothstep(0.1, 0.6, fpA));
            m = mix(m, moss, cov * (0.25 + 0.75 * wet));
            // Past the 1-voxel map the treads carry no crowns: their share of the tread takes the leaves' colour.
            if (k > 0 && wy < treelineAt(snowline)) m = mix(m, leaf, 0.45 * (0.25 + 0.6 * wet));
        }
    } else if (n.y == 0) {
        float drop = max(ledge - wy, 0.0);
        // Rain off each rim stains the stone below it in soft, broad runs.
        float run = saturate(0.55 + 0.6 * perlin2(float2(along / 13.0, wy / 70.0), seed + 550u));
        run = mix(run, 0.55, smoothstep(0.3, 1.5, fpA));
        m *= 1.0 - 0.42 * run * exp(-drop / 80.0) * step(wy, ledge);
        // The ivy below the rim: hung as leaves on the 1-voxel map, painted past it, over a halo of moss on the stone.
        // Taken in the column in front of the face, as the leaves are hung.
        float2 wf = wq + float2(n.xz) * 0.5;
        if (k == 0) wf = floor(wf) + 0.5;
        float L = towerIvyDrop(wf, R.a.xy, R.b.w, int(R.b.z), th, ledge, drop + 200.0, wet, seed);
        if (k == 0) L = floor(L);
        float e = 0.5 * max(fpA, cs);
        // The mat's edge is ragged in leafy clumps, with stone showing through toward it; both fade to its mean with the
        // footprint (whole voxels on the 1-voxel map).
        float2 lq = k == 0 ? floor(float2(along, wy)) + 0.5 : float2(along, wy);
        float rag = (0.6 * perlin2(lq / 5.0, seed + 552u) + 0.4 * perlin2(lq / 2.2, seed + 553u)) * (1.0 - smoothstep(0.3, 2.0, fpA));
        float cover = (L - drop) / max(L, 1.0) + 0.45 * rag + 0.05;
        float ec = 0.5 * max(fpA, cs) / max(L, 4.0) + 0.04;
        float vine = smoothstep(-ec, ec, cover) * step(1.0, L) * step(drop, L * 1.5 + 2.0) * mix(1.0, 0.8, smoothstep(0.3, 2.0, fpA));
        float halo = smoothstep(L * 1.3 + 8.0 + e, L * 1.3 + 8.0 - e, drop) * step(1.0, L);
        m = mix(m, moss * 1.1, 0.4 * halo);
        if (wy < treelineAt(snowline)) {
            // Leaves in three tones voxel by voxel close up (sunlit tips, mid, the dark between), their mean past that.
            float lt = hashf(hash3i(int3(int(floor(wq.x)), int(floor(wy)), int(floor(wq.y))), seed + 551u));
            float tex = lt < 0.25 ? 0.65 : lt < 0.75 ? 1.0 : 1.25;
            float3 ivy = mix(leaf, float3(0.06, 0.075, 0.04), 0.5);
            m = mix(m, ivy * mix(tex, 1.0, smoothstep(0.05, 0.3, fp)), vine);
        }
        // A damp band of moss along the foot.
        m = mix(m, moss * 1.1, smoothstep(Hb + 30.0, Hb + 6.0, wy) * 0.6 * wet);
        // Past level 1 the coarse columns hold no loggias: their dark bays are painted under each rim, eased in across
        // level 1's morph band as its gaps close.
        float paint = k >= 2 ? 1.0 : k == 1 ? morphAt(x.tr, 1, (float2(v.xz) + 0.5) * cs) : 0.0;
        if (paint > 0.0) {
            Tower T = towerFromRec(R);
            float2 wc = w + 0.5 * cs;
            float hole = 0.0;
            int jr = -1;
            if (jl >= 0) {
                TowerPlan Pj = towerPlan(T, wc - towerStageAxis(T, jl));
                float bwF = towerBayW(th, jl, Pj.face, bw);
                float bayI = floor(Pj.t / bwF + 0.5);
                if (towerLoggiaBay(th, jl, Pj.face, bayI) && !towerBayFallen(th, jl, Pj.face, bayI)) {
                    float dt = abs(Pj.t - bwF * bayI);
                    float hi = tierY(T, Pj, jl, seed) - 4.0, lo = hi - floor(Sy * 1.1);
                    float band = smoothstep(lo - e, lo + e, wy) * smoothstep(hi + e, hi - e, wy);
                    float bay = mix(smoothstep(0.32 * bwF + e, 0.32 * bwF - e, dt), 0.64, smoothstep(0.15 * bwF, 0.4 * bwF, max(fpA, cs)));
                    hole = band * bay;
                    jr = jl;
                }
            }
            // The great portals in the base's faces, and the passage under arches.
            TowerPlan P0 = towerPlan(T, wc - T.c);
            if (wy < Hb + 40.0 + e && P0.reff < R0 + 6.0 + cs) {
                float tw = 7.0 + 0.5 * cs;
                float hp = smoothstep(Hb + 40.0 + e, Hb + 40.0 - e, wy + 4.0 * (P0.t / 7.0) * (P0.t / 7.0));
                if (P0.face % (T.N > 8 ? 3 : 2) == 0) {
                    float por = smoothstep(tw + e, tw - e, abs(P0.t)) * hp;
                    if (por > hole) { hole = por; jr = 7; }
                }
                if ((th & 128u) != 0u) {
                    float2 pd = float2(cos(T.th1 + 1.5708), sin(T.th1 + 1.5708));
                    float off = abs((wc - T.c).x * pd.y - (wc - T.c).y * pd.x);
                    float hpass = Hb + 36.0 + floor(0.2 * R0);
                    float pas = smoothstep(tw + e, tw - e, off) * smoothstep(hpass + e, hpass - e, wy + 6.0 * (off / 7.0) * (off / 7.0));
                    hole = max(hole, pas);
                }
            }
            m = mix(m, m * 0.22, hole * paint);
            // Its lit rooms, as the open bays near by light them, glowing on far into the night.
            float night = saturate(x.lights.w * 4.0) * paint * (1.0 - smoothstep(2500.0, 4500.0, length((float2(v.xz) + 0.5) * cs - x.tr.cam)));
            if (night > 0.0 && R.d.z != 0u && jr >= 0) {
                uint lh = towerLitRoom(T, wc, jr);
                if (lh != 0u)
                    s.emit = TOWER_ROOM * night * hole * (0.8 + 0.2 * hashf(pcg(lh))) * flicker(x.time, float(lh & 255u), seed);
            }
        }
    } else {
        m *= 0.85;
    }
    s.albedo = m;
    s.tread = m;
    s.face = m;
    s.pale = n.y > 0 && wy > snowline && !root ? 1.0 : 0.0;
    s.slope = 0.0;
    s.wall = 1.0;
    if (any(pl != 0.0)) s.nrm = pl;
    s.level = n.y > 0 ? 1.0 : 0.0;
    // A few rooms lit at night on some towers: the faces of an opening (back wall, reveals, sill and lintel) on the
    // 1-voxel map, faded out across its morph band, where the painted bays take over.
    float night = saturate(x.lights.w * 4.0) * (k == 0 ? 1.0 : k == 1 ? 1.0 - morphAt(x.tr, 1, (float2(v.xz) + 0.5) * cs) : 0.0);
    if (night > 0.0 && R.d.z != 0u) {
        float3 e = towerRoomLight(x, R, k, v, n, w, night);
        if (any(e > 0.0)) s.emit = e;
    }
}

// shadows: 0 none, 1 traced near ray and maps, 2 maps only (reflections: ripples hide contact shadows).
// L: the light (the bed under water takes it refracted); inWater: seen in rippled water.
// The share of the sky the islands overhead hide from a point at map-local p: each a disc of its radius at its
// underside's mean height (the view factor of a disc to a level patch), out to 2-3k from the camera.
static float isleSkyOcc(thread const Ctx& x, float3 p) {
    int n = min(x.isles->n.x, ISLE_TABLE);
    float2 w = p.xz + float2(x.origin), cam = x.tr.cam + float2(x.origin);
    float occ = 0.0;
    for (int i = 0; i < n; i++) {
        IsleShade s = x.isles->r[i];
        float hh = s.a.w - p.y;
        float2 dv = w - s.a.xy;
        float r2 = dot(dv, dv);
        if (hh < 4.0 || r2 > 9.0 * s.a.z * s.a.z) continue;
        float R2 = s.b.x * s.b.x, h2 = hh * hh;
        float q = h2 + r2 + R2;
        float F = 0.5 * (1.0 - (h2 + r2 - R2) / sqrt(max(q * q - 4.0 * r2 * R2, 1e-3)));
        occ += F * (1.0 - smoothstep(2000.0, 3000.0, length(s.a.xy - cam)));
    }
    return saturate(occ);
}

static float3 shade(thread const Ctx& x, thread const Light& L, float3 o, float3 d, Hit h, int shadows, bool inWater = false) {
    float3 p = o + d * h.t;
    float3 n = float3(h.normal);
    int k = h.level;
    float cs = x.tr.lod[k].w;
    float2 uv = faceUV(p, h.normal, cs);
    float fp = h.t * x.pixelAngle; // voxels per pixel
    // Footprint stretched across a grazing face, for detail that must fade before it aliases along the view.
    float fpA = fp / max(abs(dot(n, d)), 0.15);
    Surface s = surfaceAt(x.tr, x.st, x.sites, x.lampState, k, h.voxel, h.normal, uv, fp, fpA, p, x.seed, x.time, inWater);
    if (sceneTowers && x.towers->n.x > 0) towerSurface(x, s, k, h.voxel, h.normal, uv, inWater ? max(fp, 4.0) : fp, inWater ? max(fpA, 4.0) : fpA, d);
    // A tower wall shading as its plane (s.nrm; terrain has none of its own): no corner AO, which would line every jog of
    // a diagonal face.
    bool planar = sceneTowers && s.mat == 0 && any(s.nrm != 0.0);
    // Prefilter: from ~1/10 voxel per pixel, relax AO and shadows and give risers their tread colour, so
    // distant stairs read as contour bands, not moire.
    float wf = smoothstep(0.025, 0.20, fp);
    float3 nS = normalize(float3(s.nh.x - s.nh.y, 2.0 * cs, s.nh.z - s.nh.w));
    // Below ~2 pixels per voxel step shadows and sky visibility alias into a hatch that crawls under motion;
    // shade the continuous slope under the stairs instead.
    // Taken just over half a column into the air in front of the face: on the face, or exactly half a column out, the
    // footprint's split between rock and air turns on rounding.
    // Floors are asked just above, undersides just below.
    float roofW = 0.0;
    // Only near the gaps' heights: a lip's top or the sheer wall under a shelter lies outside every gap around it.
    // Extra gaps lie under the principal ones, so below those only the floor of the map bounds them.
    if (s.mat == 0 && s.gapNear && p.y < s.gapSpan.y + 1.5 && (p.y > s.gapSpan.x - 1.5 || k == 0)) {
        // A wall is under a roof when the column in front of it is hollow at its height: every face of a jogging back
        // wall then agrees, where a share sampled at the face fell part way at each jog and striped the wall.
        if (h.normal.y == 0) roofW = hollowAtMorph(x.tr, k, h.voxel.xz + h.normal.xz, float(h.voxel.y) + 0.5) ? 1.0 : 0.0;
        else if (h.normal.y > 0) roofW = roofField(x.tr, k, p.xz, p.y + 0.5).x;
    }
    // An underside is its own roof: a share fading toward the lip's edge would light the edge's undersides as a pale line.
    if (s.mat == 0 && h.normal.y < 0) roofW = 1.0;
    // Faces under a roof, and undersides, are not stairs on a heightfield: no heightfield prefilter applies to them. A
    // lip's front and top are ordinary wall and ground.
    float ov = s.mat == 0 ? max(roofW, h.normal.y < 0 ? 1.0 : 0.0) : 0.0;
    float wf2 = s.mat == 0 ? smoothstep(0.2, 0.5, fp) * (1.0 - ov) : 0.0;
    bool built = s.mat >= 4;
    // Tops a few pixels wide keep their exact normal: a column's central difference tilts only the tops at a
    // ledge, so they would shade unlike the rest of their plane. Toward 2 pixels per voxel the slope under the
    // stairs takes over, smooth across ledges. Side faces blend with their tread colour (wf), toward the column
    // normal until that slope takes over: a riser in tread colour lit as a wall reads as a veil over the hill
    // behind it. Canopies keep their column normal blend.
    float wt = smoothstep(0.33, 0.6, fp);
    // Structures keep their own planes: a column-difference normal would tilt a house's edge columns.
    float wn = (built ? 0.0 : s.mat == 0 && h.normal.y != 0 ? wt : wf) * (1.0 - ov);
    if (s.mat == 0 && wt > 0.0 && s.level < 1.0 && shadows != 2) {
        // Side faces take the wall's direction over several columns: a single column's jog in a wall would otherwise
        // turn a narrow face to the light and draw a pale dash down the wall.
        float R = max(max(h.normal.y != 0 ? 2.0 : 6.0, 1.5 * fp), cs);
        float2 grad = float2(groundSmoothL(x.tr, k, p.xz + float2(R, 0), s.gy) - groundSmoothL(x.tr, k, p.xz - float2(R, 0), s.gy),
                             groundSmoothL(x.tr, k, p.xz + float2(0, R), s.gy) - groundSmoothL(x.tr, k, p.xz - float2(0, R), s.gy)) / (2.0 * R);
        float3 g = normalize(float3(-grad.x, 1.0, -grad.y));
        nS = h.normal.y != 0 ? g : normalize(mix(nS, g, wt));
    }
    float3 nl = normalize(mix(normalize(n - float3(s.slope.x, 0.0, s.slope.y)), nS, wn));
    if (any(s.nrm != 0.0)) nl = s.nrm;
    nl = normalize(mix(nl, float3(0, 1, 0), s.level));
    // Coarse levels are always past the footprint where corner AO is fully replaced.
    // Seen in water (reflections, the bed) the surface is rippled out of focus: no corner AO, a coarse sky visibility.
    bool lite = shadows != 1;
    // A wall under a roof skips corner AO too: every jog of a back wall is a concave corner, a dark line per column.
    float ao = (s.level < 1.0 ? mix(k == 0 && !lite && !(h.normal.y == 0 && roofW > 0.0) && !planar ? mix(cornerAO(x.tr.hm, x.tr.size, h.voxel, h.normal, uv), 0.85, wf) : 0.85, 1.0, s.level) : 1.0) * s.canopy;
    // A pitched roof's treads and risers are one plane: no corner AO lines along its steps, and (below) one shadow.
    bool roofPlane = built && s.nrm.y > 0.1;
    if (roofPlane) ao = 0.9;
    float drop = s.H - min(min(s.nh.x, s.nh.y), min(s.nh.z, s.nh.w));
    // Cliffs several pixels tall keep their strata; only sub-pixel stairs take the tread colour. Keyed to the
    // column's deepest drop, not this face's: the faces of a wall running diagonal to the grid alternate
    // between shallow and deep neighbours, which would stripe it strata, tread, strata. A wall built of steps a
    // few voxels each is as much a cliff as a sheer one; by drop alone it turns a pale tread-coloured band.
    float steep = 0.5 * length(float2(s.nh.x - s.nh.y, s.nh.z - s.nh.w));
    float cliff = s.riser > 0.0 ? max(drop, 2.0 * steep) : 0.0;
    float keep = smoothstep(6.0, 14.0, cliff / fp) * smoothstep(3.0, 5.0, cliff);
    // Among the great walls every face is rock: column by column, some faces of a wall would otherwise take the
    // tread colour and stripe it.
    // So is every riser of a tall, steep staircase wall: kept by its own drop alone, its deeper columns would show as
    // strata dashes among tread-coloured ones.
    keep = max(keep, max(s.wall, smoothstep(1.0, 2.5, steep) * smoothstep(100.0, 160.0, s.H)) * smoothstep(1.0, 3.0, cliff));
    // Overhangs keep their strata and their own faces' normals.
    keep = max(keep, ov);
    // A slope steep over a wide span (+-16 voxels and more) is seen almost edge-on to its treads: from afar it is a
    // rock face. Its risers painted in the tread colour read as a veil of the ground above (grass, snow, hypsometric
    // bands) hung across the rock, and its treads as pale lines ruled across it, so both take the face's rock.
    float sus = 0.0;
    // A planar wall is its own plane already: the slope under the stairs is the ground's, and column by column it would
    // turn each face of a jogging wall its own way.
    if (s.mat == 0 && wf > 0.0 && s.H > SEA + 20.0 && shadows != 2 && !planar) {
        float R2 = max(16.0, 2.0 * cs);
        float2 g2 = float2(groundSmoothL(x.tr, k, p.xz + float2(R2, 0), s.gy) - groundSmoothL(x.tr, k, p.xz - float2(R2, 0), s.gy),
                           groundSmoothL(x.tr, k, p.xz + float2(0, R2), s.gy) - groundSmoothL(x.tr, k, p.xz - float2(0, R2), s.gy)) / (2.0 * R2);
        sus = smoothstep(0.6, 1.1, length(g2)) * (1.0 - ov);
        if (h.normal.y == 0) keep = max(keep, sus);
        // Treads and risers alike light as the face they make up, or every tread is a lit line across it.
        nl = normalize(mix(nl, normalize(float3(-g2.x, 1.0, -g2.y)), wf * sus));
    }
    // A riser face only a column wide (a jog in a step edge) is a lone cube face: lit, it is a bright cube; in shade,
    // stone against sand, a dark notch. With the footprint it fades to the tread it stands for, colour and light.
    float lone = 0.0;
    // A planar wall's jogs are its plane's.
    if (h.normal.y == 0 && s.mat == 0 && wn < 1.0 && shadows != 2 && !planar) {
        int2 t = h.normal.x != 0 ? int2(0, 1) : int2(1, 0), o = h.normal.xz;
        float vy = min(float(h.voxel.y), s.gy);
        float run = float(groundL(x.tr, k, h.voxel.xz + t, vy) > groundL(x.tr, k, h.voxel.xz + t + o, vy) + 0.5)
                  + float(groundL(x.tr, k, h.voxel.xz - t, vy) > groundL(x.tr, k, h.voxel.xz - t + o, vy) + 0.5);
        lone = (1.0 - 0.5 * run) * smoothstep(0.01, 0.05, fp) * (1.0 - wn) * (1.0 - keep) * (1.0 - ov);
    }
    float3 alb = h.normal.y == 0 ? mix(s.albedo, s.tread, max(wf * (1.0 - keep), lone)) : mix(s.albedo, s.face, wf * sus);
    float deep = L.night * smoothstep(0.32, 0.45, -L.sun.y);
    // Moonlight drains colour.
    alb = mix(alb, float3(dot(alb, float3(0.3, 0.59, 0.11))), 0.4 * L.night + 0.1 * deep);
    // Canopies would otherwise drop to cut-out silhouettes under the moon: lifted toward the luminance of a lifted
    // green crown, never past it, or fall foliage, already several times brighter, glows like candy.
    if (s.mat == 1 || s.mat == 2) {
        float l = dot(alb, float3(0.3, 0.59, 0.11));
        alb = mix(alb, float3(l), 0.4 * L.night * s.autumn);
        alb *= 1.0 + L.night * (clamp(0.17 / max(l, 1e-3), 1.0, 2.4) - 1.0);
    }
    // Pale sand, salt and snow would glow under the moon brighter than the sky above them. A knee on albedo
    // above mid-tone, not a cut keyed to the material, so tones keep their order: a stone riser never outshines
    // its sand tread.
    float kn = 4.5 * L.night;
    alb *= (1.0 + 0.1 * kn) / (1.0 + kn * dot(alb, float3(0.3, 0.59, 0.11)));
    float ndl = mix(max(dot(nl, L.key), 0.0), max(L.key.y, 0.0), lone);
    // A low moon lights a step's riser several times brighter than its tread, drawing a highlight line along
    // every step; under the moon only cliffs outshine the ground they stand in.
    if (h.normal.y == 0 && s.mat == 0) ndl = mix(ndl, min(ndl, max(L.key.y, 0.0)), L.night * (1.0 - keep));
    // Low structure faces (cairns, boats, deck edges, fallen stones) are steps too; uprights and house walls stay walls.
    if (h.normal.y == 0 && built && drop <= 2.0 && !any(s.nrm != 0.0)) ndl = mix(ndl, min(ndl, max(L.key.y, 0.0)), L.night);
    float hiSun = smoothstep(0.15, 0.45, L.key.y) * (1.0 - L.night);
    float sh = 1.0;
    if (shadows > 0 && ndl > 0.0 && L.shadows && x.shadowMode != 3.0) {
        // Keep the origin inside the hit voxel's footprint so boundary columns can't self-shadow. A planar wall's ray
        // starts out in front of its plane: from a jog's own face it would cross the next column and stripe the wall.
        float3 so = p + (planar ? s.nrm * 1.5 : float3(n) * 0.02);
        if (!planar) so.xz = clamp(so.xz, float2(h.voxel.xz) * cs + 0.002, float2(h.voxel.xz) * cs + (cs - 0.002));
        if (roofPlane && h.normal.y == 0) so.y = max(so.y, s.H + 0.02);
        // Prefiltered, the ray starts a step above the column, and single steps shade by their expected cover.
        so.y = mix(so.y, max(so.y, s.H) + 1.0, wf2);
        // With the shadow-height maps the ray only reaches SHADOW_NEAR (contact and cliff shadows); the maps hold every
        // occluder beyond.
        bool maps = x.shadowMode == 1.0, brute = x.shadowMode == 2.0;
        int shadowIters = brute ? 4000 : maps ? (x.quality > 0.5 ? 48 : 32) : (x.quality > 0.5 ? 120 : 48);
        Hit sHit;
        sHit.hit = false; sHit.t = 1e9; sHit.iters = 0;
        // Past the 1-voxel map a pixel spans columns: the maps alone, with occluders from two texels out.
        // Under the moon, whose shadows are soft and faint, only near ground keeps the traced ray.
        bool mapOnly = maps && (shadows == 2 || k >= 1 || h.t > (L.night > 0.5 ? 120.0 : 500.0));
        if (!mapOnly) sHit = trace(x.tr, so, L.key, brute ? 6000.0 : maps ? SHADOW_NEAR : 700.0, shadowIters, k);
        // A ray that runs out of steps is still threading terrain toward the sun: count it as blocked there, so a
        // shaded spot never comes out lit.
        sHit.hit = sHit.hit || sHit.iters >= shadowIters;
        // Contact shadows from single-voxel steps draw sawtooth noise; keep them faint, cliffs full.
        float contact = 0.35 * (1.0 - smoothstep(2.0, 8.0, 0.5 / L.key.y));
        // The lift moves the ray's start about 1 / L.y along it; contact reach moves with it.
        float reach = wf2 / max(L.key.y, 0.05);
        // A roof overhead is never a contact shadow.
        sh = sHit.hit ? (sHit.normal.y < 0 && sHit.iters < shadowIters ? 0.0 : 0.35 * (1.0 - smoothstep(2.0 + reach, 8.0 + reach, sHit.t))) : 1.0;
        if (maps) sh = min(sh, shadowMapLit(x, so, mapOnly));
        // Under a roof the roof stands in for the ray: the maps see an overhang only from two texels out, and a traced ray
        // that slips out under the edge on one row and not the next stripes the back wall with sun.
        if (wf2 > 0.0 && !planar) sh = mix(sh, min(sh, stairShadow(x.tr, k, p.xz, s.gy, L.key, d, contact, 0.35, fp > 3.0 || shadows == 2)), wf2);
        float floorSh = mix(mix(0.5, 0.25, hiSun), 0.32, L.gold);
        sh = mix(sh, floorSh + (1.0 - floorSh) * sh, wf);
        sh = mix(sh, 1.0, 0.3 * L.night);
        // After the lifts, which stand for light slipping between distant steps: none slips under a roof.
        sh = min(sh, 1.0 - roofW);
    }
    if (!(x.prof & 32) && L.sunAmt > 0.0) sh *= mix(1.0, cloudShadow(x, p, L.key), L.sunAmt);
    float lowFill = mix(0.9, 1.0, smoothstep(0.1, 0.4, L.key.y));
    // Low light: shadows take the cool zenith, against warm key light.
    // High sun: fill is the deep zenith, so shade reads blue against warm key light.
    float zw = mix(mix(0.55, 0.8, hiSun), 0.85, L.gold);
    float3 skyFill = mix(L.horizon, L.zenith, saturate(zw + 0.15 * nl.y)) * 0.9 * lowFill * mix(1.0, 0.85, hiSun);
    // Golden hour: shade is the violet dome opposite the sun, well below the key, so hillsides keep dark values.
    skyFill *= mix(float3(1.0), float3(0.62, 0.58, 0.85), L.gold * L.sunAmt);
    // Under a roof the visibility is taken where the point is, not at its column's centre: column by column it would
    // jump between reaching past the roof's edge and not, striping the wall.
    float2 svAt = planar ? p.xz + s.nrm.xz * 1.5 : mix((float2(h.voxel.xz + h.normal.xz) + 0.5) * cs, p.xz, max(wf2, ov));
    float svY = wf2 > 0.0 ? mix(p.y, groundSmoothL(x.tr, k, p.xz, s.gy), wf2) : p.y;
    float sv = x.prof & 16 || roofW >= 1.0 || s.isleUnder ? 1.0 : skyVisibility(x.tr, k, svAt, svY, lite || fp > 5.0 ? 2 : fp > 2.0 ? 1 : 0);
    // Under a roof the sky comes only through the mouth, an even share well under open shade's: the visibility samples
    // only six voxels out, so in a shelter it reads the roof all round and varies column by column with the back wall's
    // jogs, striping it.
    skyFill *= mix(1.0 - mix(0.55, 0.85, hiSun) * (1.0 - sv), ROOF_SKY, roofW);
    if (x.isles->n.x > 0) skyFill *= 1.0 - 0.85 * isleSkyOcc(x, p);
    // A wall sees half the dome, and the half it sees is the dimmer horizon side; without this a riser in shade
    // outshines the sunlit tread above it under a low sun.
    skyFill *= built ? 0.7 + 0.3 * saturate(nl.y) : 0.5 + 0.5 * saturate(nl.y);
    // Light bounced into a hollow comes in through its mouth, off the lit ground outside, which a low sun still lights
    // face on where it reaches the shelter's own floor: dim, never black; an underside stays well under the lit top of
    // its slab.
    float keyUp = mix(max(L.key.y, 0.0), max(L.key.y, 0.3) * saturate(L.key.y * 10.0), ov);
    float3 bounce = (1.0 - saturate(nl.y)) * (built ? 0.6 : 0.35) * alb * L.keyCol * keyUp;
    // High sun: risers sit in cooler shade so terraces keep their modelling at noon.
    float3 riserShade = mix(float3(1.0), float3(0.82, 0.86, 0.95),
                            (h.normal.y == 0 ? 1.0 - saturate(nl.y) : 0.0) * smoothstep(0.3, 0.6, L.key.y) * smoothstep(1.5, 4.0, drop) * (1.0 - ov));
    // Night: no pure-black riser feet; a faint indigo ambient keeps the steps readable as stone, not holes.
    ao = mix(ao, 1.0, 0.25 * L.night);
    // Night: a cave mouth stays a dark hole.
    skyFill += float3(0.09, 0.12, 0.25) * L.night * mix(1.0, ROOF_SKY, roofW);
    // Deep in the night, land sits below the sky so night reads as night (won night A/B): moonlight is graded
    // to a cool silver and the fill to a deep blue well under it, so slopes facing the moon still model. In
    // early night it is already dark and keeps its contours (darkening lost the dusk A/B).
    float3 col = alb * (L.keyCol * ndl * sh * mix(1.0, 1.1, L.night) * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep)
                        + (skyFill * ao * riserShade + bounce) * mix(float3(1.0), float3(0.19, 0.23, 0.38), deep));
    // Low sun: up-facing treads take a warm sky-dome tint instead of going to charcoal.
    col += alb * L.keyCol * 0.07 * (1.0 - smoothstep(0.05, 0.35, L.key.y)) * saturate(nl.y) * (1.0 - L.night);
    // Backlit leaves: a low sun glows through the canopy edges facing away from it.
    if (s.mat == 1) {
        float back = pow(saturate(dot(d, L.key)), 4.0) * (1.0 - smoothstep(0.08, 0.45, L.key.y)) * (1.0 - L.night);
        float graze = 1.0 - abs(dot(n, d));
        col += alb * mix(float3(0.9, 1.3, 0.3), float3(1.3, 0.9, 0.4), s.autumn) * L.keyCol * back * (1.0 - ndl) * (1.0 - abs(n.y)) * graze * graze * sh;
    }
    // Lamplight is added in absolute terms, outside the moonlight grade, so firelit ground stays warm in a blue night.
    if (x.lights.w > 0.0) {
        constexpr sampler bl(filter::linear, address::clamp_to_edge);
        float2 mu = p.xz / float(x.tr.size);
        // The light map spans the 1-voxel map; coarse ground past it would take its clamped edge.
        float3 Lm = all(mu > 0.0) && all(mu < 1.0) ? x.lightMap.sample(bl, mu, level(0)).rgb : float3(0);
        if (any(Lm > 1e-4)) {
            float4 Ld = x.lightDir.sample(bl, mu, level(0));
            float3 lp = Ld.xyz / max(Ld.w, 1e-6);
            // Hills above a fire and valleys below it stay dark.
            float dy = (p.y - lp.y) / 5.0;
            float face = 0.3 + 0.7 * saturate(dot(nl, normalize(lp - p + float3(0, 0.3, 0))));
            float glow = s.mat == M_CLOTH || s.mat == M_FELT ? 0.2 : 0.0;
            col += s.albedo * Lm * ((face * mix(1.0, ao, 0.5)) / (1.0 + dy * dy) + glow) * x.lights.w;
        }
    }
    return col + s.emit;
}

// In a morph band the shading also slides to the next level's, reaching it where the geometry does, so the switch
// at the handoff circle is seamless in colour and light as well as in shape.
static float3 shadeL(thread const Ctx& x, thread const Light& L, float3 o, float3 d, Hit h, int shadows, bool inWater = false) {
    float3 p = o + d * h.t;
    float w = morphAt(x.tr, h.level, p.xz);
    // One shading per pixel: the next level's with probability w, varying per pixel and frame for TAA to blend.
    // Each shading traces its own level's shadow: the outer one must match the next level's hits past the circle.
    if (w > 0.0 && x.dither < w) {
        int2 c = int2(floor((p.xz - float2(h.normal.xz) * 0.01) / x.tr.lod[h.level + 1].w));
        h.level += 1;
        h.voxel = int3(c.x, h.voxel.y, c.y);
    }
    return shade(x, L, o, d, h, shadows, inWater);
}

// Converges on the actual sky colour as fog thickens (T -> 0), so fogged land meets the sky with no seam.
static inline float3 fogColor(thread const Ctx& x, float3 d, float T) {
    // At night the sky's horizon gradient is mirrored below it, so far land and sea dim smoothly downward.
    float y = max(d.y, -d.y * x.L.night);
    float3 c = skyColor(float3(d.x, mix(y, y * 0.5 + 0.12, T), d.z), x.L, false, x.seed, x.time);
    // Aerial colour cools away from the sun so far ridges separate from the sky.
    float away = saturate(dot(normalize(d.xz + 1e-5), normalize(x.L.sun.xz + 1e-5)) * 0.5 + 0.5);
    float3 cool = mix(float3(0.80, 0.82, 0.95), float3(0.52, 0.55, 0.88), x.L.gold);
    return c * mix(float3(1.0), mix(cool, float3(1.0), away * away), T);
}

// Share of the lowland air haze left at height y (Sources/Sight.swift ports it).
static inline float airAloft(float y) { return exp(-max(y - 120.0, 0.0) / 400.0); }

// Exponential height fog plus a slow-drifting noise modulation that pools mist in valleys.
// mistSamples < 0 takes the quality's count; 0 leaves out the valley mist (the closed-form terms only).
static float fogTransmittance(thread const Ctx& x, float3 o, float3 d, float t, int mistSamples = -1) {
    float hf = 30.0;
    float a = 0.0009, c2 = 1.3e-6;
    // Air: the near haze saturates into a thin linear tail, so far ranges read as layered haze rather than vanish.
    const float K = 1.6, lambda = 6e-5;
    float k = exp(-(o.y - SEA) / hf);
    float dy = abs(d.y) < 1e-4 ? 1e-4 : d.y;
    float heightTerm = a * hf * k * (1.0 - exp(-dy * t / hf)) / dy;
    float mist = 0;
    // Two samples at a per-pixel, per-frame offset along the ray: the temporal resolve integrates them into the four
    // evenly spaced ones this replaced (the mist is smooth over tens of voxels).
    int samples = mistSamples >= 0 ? mistSamples : x.quality > 0.5 ? 2 : 1;
    float jit = mistSamples >= 0 ? 0.5 : x.dither;
    for (int i = 0; i < samples; i++) {
        float st = min(t, 400.0) * (float(i) + jit) / float(samples);
        float3 sp = o + d * st;
        float3 wp = float3(sp.x + float(x.origin.x), sp.y, sp.z + float(x.origin.y));
        // Mist lies in the low ground: well above the water a sample adds nothing worth its noise.
        float low = exp(-max(sp.y - SEA, 0.0) / 10.0);
        if (low > 0.002)
            mist += saturate(perlin3(wp * float3(0.012, 0.03, 0.012) + float3(x.time * 0.01, 0, x.time * 0.006), x.seed + 50u) + 0.2) * low;
    }
    mist = mist / float(max(samples, 1)) * min(t, 400.0) * 0.0014 * mix(0.7, 0.3, smoothstep(0.15, 0.45, x.L.sun.y));
    // The world ends in a circle around the camera (not around o: reflections start elsewhere), fully hazed at its edge.
    float r = length((o + d * t).xz - x.tr.cam);
    if (r >= x.tr.rmax) return 0.0;
    // Clear air under a high sun; haze belongs to the low light.
    float clear = mix(1.0, 0.15, smoothstep(0.2, 0.45, x.L.sun.y));
    // Night air is crisp: moonlit ridges stay legible to the horizon.
    clear *= mix(1.0, 0.35, x.L.night);
    // The tail eases in, leaving the haze within a few percent of the old quadratic out to ~600.
    float air = K * (1.0 - exp(-c2 * t * t / K)) + lambda * t * t / (t + 1500.0);
    // The haze thins aloft: a ray climbing to a summit or a tepui's rim leaves most of it below, so tall relief far
    // off stands solid over a hazy foot instead of hazing to a flat, cloudless cut-out of the sky. Taken at the
    // height of a point most of the haze lies before, its climb capped, so farther along a ray the haze never comes
    // out more than ~5% thinner than nearer.
    air *= airAloft(o.y + min(d.y, 0.2) * 0.75 * min(t, 2000.0));
    return exp(-((heightTerm + air + mist) * clear + 6.0 * smoothstep(x.tr.fadeStart, x.tr.rmax, r)));
}

static float3 applyFogT(thread const Ctx& x, float3 col, float3 o, float3 d, float T) {
    // Thin haze in-scatters less than the full horizon radiance; only thick fog converges on the sky.
    // Forward scatter fades out above the horizon as the sky's does, so fully hazed land matches the sky behind it.
    float scatter = min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night) * mix(1.0, smoothstep(0.1, 0.0, d.y), smoothstep(0.5, 0.0, T));
    float3 f = (fogColor(x, d, T) + x.L.keyCol * scatter) * mix(mix(0.3, 0.12, x.L.gold), 1.0, smoothstep(0.1, 0.9, 1.0 - T));
    return mix(f, col, T);
}

static float3 applyFog(thread const Ctx& x, float3 col, float3 o, float3 d, float t) {
    return applyFogT(x, col, o, d, fogTransmittance(x, o, d, t));
}

// In-scatter of the nearest lamps' light along a ray to tEnd: the closed-form integral of 1/(r^2 + 4) - 1/(Rh^2 + 4)
// over the ray's chord through a sphere of radius Rh = 3 reach, so a hamlet or a fire hangs in a soft glow of its
// own haze that ends at zero a few reaches out, and the glow stops at whatever hides it. Without the sphere the
// integral falls off only as 1/distance and distant lamps tint the whole night.
// Each lamp's glow is seen through the air between, as the land around it is.
static float3 lampHaze(thread const Ctx& x, device const float4* pos, device const float4* state, constant int4* refs, int count,
                       float3 o, float3 d, float tEnd) {
    float3 sum = 0;
    for (uint m = x.tiles & ((1u << uint(count)) - 1u); m != 0u; m &= m - 1u) {
        int i = int(ctz(m));
        int li = refs[i / 4][i % 4];
        float4 ls = state[li];
        if (all(ls.rgb == 0.0)) continue;
        float4 lp = pos[li];
        float3 v = lp.xyz - o;
        float tc = dot(v, d);
        float p2 = max(dot(v, v) - tc * tc, 0.0);
        float Rh = 3.0 * ls.w;
        if (p2 >= Rh * Rh) continue;
        float c = sqrt(Rh * Rh - p2);
        float t0 = max(tc - c, 0.0), t1 = min(tc + c, tEnd);
        if (t1 <= t0) continue;
        float hh = sqrt(p2 + 4.0);
        float I = (atan((t1 - tc) / hh) - atan((t0 - tc) / hh)) / hh - (t1 - t0) / (Rh * Rh + 4.0);
        if (I <= 0.0) continue;
        sum += ls.rgb * LAMP_I[int(lp.w)] * I * fogTransmittance(x, o, d, clamp(tc, t0, t1), 0);
    }
    return sum;
}

struct CloudSample { float3 col; float a; float t; };

// Clouds: a volumetric slab above the peaks, marched in a few steps. The same noise field sets the
// coverage; tops tower where it is dense, and each sample is self-shadowed toward the sun, so clouds have
// lit crowns and shaded bases instead of reading as flat cut-outs.
// tEnd clips the march at terrain that rises into the slab, so a peak disappears into the cloud base.
// steps > 0 overrides the march's step count (a veil in front of land needs fewer).
static CloudSample cloudSheet(thread const Ctx& x, float3 o, float3 d, float tEnd = 1e9, int stepsOverride = 0) {
    CloudSample c;
    c.col = 0; c.a = 0; c.t = 6e4;
    if (d.y < 0.01) return c;
    float y0 = CLOUD_Y0, y1 = CLOUD_Y1;
    float ta = (y0 - o.y) / d.y, tb = (y1 - o.y) / d.y;
    float t0 = max(min(ta, tb), 0.0), t1 = min(max(ta, tb), tEnd);
    if (t0 > 7000.0 * CLOUD_S || t1 <= t0) return c;
    // Night steps are dithered, so fewer suffice.
    int steps = stepsOverride > 0 ? stepsOverride : x.quality > 0.5 ? (x.L.night > 0.99 ? 8 : 12) : 8;
    float dt = (t1 - t0) / float(steps);
    float lowSun = 1.0 - smoothstep(0.02, 0.28, abs(x.L.sun.y));
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(x.L.sun.xz + 1e-5);
    float toSun = 0.5 + 0.5 * dot(dh, sh2);
    float2 sunStep = x.L.key.xz / max(x.L.key.y, 0.2) * 14.0 * CLOUD_S;
    float3 ambient = x.L.zenith * 1.0 + x.L.horizon * 0.35;
    float3 moonGlow = float3(0.10, 0.13, 0.22) * pow(saturate(dot(d, x.L.moon)), 16.0) * 0.9;
    float glowAmt = lowSun * smoothstep(-0.14, -0.02, x.L.sun.y);
    float T = 1.0;
    float3 acc = 0;
    float silver = 0;
    float jit = 0.5 + x.L.night * (x.dither - 0.5);
    // The slow part varies over hundreds of units: sample it at knots ~40 units apart along the ray (or one
    // per step, if those are further apart) and interpolate; only the fine part is evaluated per step.
    float kt = max(40.0 * CLOUD_S / max(length(d.xz), 1e-4), dt);
    float ka = t0, kb = t0 + kt;
    float2 qa = (o.xz + d.xz * ka) + float2(x.origin), qb = (o.xz + d.xz * kb) + float2(x.origin);
    float sa = cloudSlowBaked(x, qa), sb = cloudSlowBaked(x, qb);
    float wa = -1e9, wb = -1e9;
    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + jit) * dt;
        float3 p = o + d * t;
        float2 q = p.xz + float2(x.origin);
        while (t > kb) {
            ka = kb; qa = qb; sa = sb; wa = wb;
            kb += kt;
            qb = (o.xz + d.xz * kb) + float2(x.origin);
            sb = cloudSlowBaked(x, qb);
            wb = -1e9;
        }
        float kf = saturate((t - ka) / kt);
        // Night skies stay mostly clear: fewer, thinner banks.
        float base = cloudFineBaked(x, q) + mix(sa, sb, kf) - 0.07 * x.L.night;
        // Night: a finer octave frays the edges into wisps for the moon to light through. Its octaves are turned off
        // the grid axes, or thresholded lattice cells read as boxes, and it fades in, or its gate draws an edge.
        if (x.L.night > 0.0 && base > -0.15) {
            float2 qd = (q - float2(x.time * 1.3 * 3.0, 0)) / CLOUD_S;
            float fray = 0.16 * perlin2(OCT_ROT * qd / 90.0, x.seed + 42u) + 0.07 * perlin2(transpose(OCT_ROT) * qd / 33.0, x.seed + 43u)
                       + 0.035 * perlin2(OCT_ROT * OCT_ROT * qd / 13.0, x.seed + 44u);
            base += x.L.night * fray * smoothstep(-0.15, -0.05, base);
        }
        float h01 = (p.y - y0) / (y1 - y0);
        float top = 0.35 + 0.65 * saturate(base * 2.5);
        float dens = saturate((base - 0.04) * mix(3.0, 9.0, x.L.night)) * smoothstep(0.0, 0.2, h01) * (1.0 - smoothstep(top * 0.7, top, h01));
        if (dens < 0.01) continue;
        // Under the moon the sample's own density stands in for the occlusion toward the light.
        float toward = base;
        if (x.L.night <= 0.99) {
            if (wa < -1e8) wa = cloudSlowBaked(x, qa + sunStep);
            if (wb < -1e8) wb = cloudSlowBaked(x, qb + sunStep);
            toward = cloudFineBaked(x, q + sunStep) + mix(wa, wb, kf);
        }
        float shadow = exp(-saturate((toward - 0.04) * 3.0) * 2.2) * (0.55 + 0.45 * h01);
        float3 lit = x.L.keyCol * (0.15 + 0.85 * shadow) * (0.6 + 0.4 * toSun) + ambient * (0.6 + 0.4 * h01);
        // Golden hour: bodies sit in violet shade and only sun-facing crowns take the gold.
        float3 litG = x.L.keyCol * (0.04 + 0.7 * shadow * shadow) * (0.2 + 0.8 * toSun * toSun) + ambient * float3(0.62, 0.55, 0.85) * (0.5 + 0.5 * h01);
        // Forward scatter: cloud in front of a low sun glows instead of hiding it as a grey patch.
        litG += x.L.keyCol * min(hg(dot(d, x.L.sun), 0.75), 1.5) * 0.9 * shadow;
        lit = mix(lit, litG, x.L.gold);
        lit += float3(1.6, 0.55, 0.20) * glowAmt * mix(toSun, toSun * toSun * 1.2, x.L.gold) * (1.0 - h01) * 0.9;
        // Moonlit cloud is a dim blue-grey, only a little above the sky behind it.
        lit = mix(lit, float3(0.022, 0.027, 0.042) * (0.5 + 0.7 * shadow) + x.L.skyHor * 0.8 + moonGlow, x.L.night);
        float a = 1.0 - exp(-dens * dt * mix(0.05, 0.045, x.L.night) / CLOUD_S);
        acc += T * a * lit;
        silver += T * a * shadow;
        T *= 1.0 - a;
        if (c.t > 5e4 && T < 0.6) c.t = t;
        if (T < 0.03) break;
    }
    float cover = 1.0 - T;
    c.col = acc / max(cover, 1e-3);
    // Moonlit night: forward scatter through the thin, moon-facing fringes draws a silver lining around
    // the moon.
    float cm = dot(d, x.L.moon);
    float fringe = silver / max(cover, 1e-3) * (1.0 - cover);
    c.col += float3(0.70, 0.80, 1.0) * x.L.night * min(hg(cm, 0.92) * 0.25, 1.5) * fringe;
    c.col *= mix(1.0, 0.9, smoothstep(0.3, 0.5, x.L.sun.y));
    c.a = cover * mix(1.0, 0.6, x.L.night) * exp(-t0 / (3000.0 * CLOUD_S)) * smoothstep(7000.0 * CLOUD_S, 2500.0 * CLOUD_S, t0) * smoothstep(0.02, 0.1, d.y);
    return c;
}

static float4 skyWithClouds(thread const Ctx& x, float3 o, float3 d) {
    float3 raw = skyColor(d, x.L, false, x.seed, x.time);
    float3 bare = raw;
    // Forward-scatter glow near the horizon, matching what applyFog adds to fogged land.
    bare += x.L.keyCol * min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night) * smoothstep(0.1, 0.0, d.y);
    float3 discs = skyColor(d, x.L, true, x.seed, x.time) - raw;
    CloudSample c = cloudSheet(x, o, d, x.prof1 ? 7000.0 : 1e9);
    // Clouds hide the stars, moon and sun behind them.
    // At night thin cloud veils the moon instead of hiding it.
    float3 sky = bare + discs * mix(1.0 - saturate(c.a * 3.0), (1.0 - c.a) * (1.0 - c.a), x.L.night);
    return float4(mix(sky, c.col, c.a), c.a > 0.5 ? c.t : 6e4);
}

// Reflected sky: clouds softened (rippled water blurs them), no stars or discs; glitter handles the lights.
static float3 softCloudSky(thread const Ctx& x, float3 o, float3 r) {
    float3 sky = skyColor(r, x.L, false, x.seed, x.time);
    if (r.y < 0.02) return sky;
    float t = (CLOUD_Y - o.y) / r.y;
    float2 q = o.xz + r.xz * t + float2(x.origin);
    // Soft coverage, blurred as if by ripples, matching the volumetric sky.
    float dens = smoothstep(0.0, 0.3, cloudDensity(q, x.seed, x.time, x.cover) - 0.07 * x.L.night);
    float3 lit = mix(x.L.keyCol * 0.4 + x.L.zenith * 1.2 + x.L.horizon * 0.4, float3(0.04, 0.047, 0.07), x.L.night);
    return mix(sky, lit, dens * 0.4 * exp(-t / (2600.0 * CLOUD_S)));
}

// Smoke from the nearest fires and chimneys: each plume bends downwind as it rises (two straight pieces) and
// thins out by 28 voxels up. Rays take the plume's density at their closest approach to its axis.
static float4 smoke(thread const Ctx& x, device const Site* sites, device const float4* pos, device const float4* state,
                    constant int4* refs, int count, float3 o, float3 d, float tSurf) {
    const float2 wind = float2(0.825336, 0.564642);
    float3 acc = 0;
    float a = 0;
    for (uint m = (x.tiles >> 16) & ((1u << uint(count)) - 1u); m != 0u; m &= m - 1u) {
        int i = int(ctz(m));
        int li = refs[4 + i / 4][i % 4];
        float4 lp = pos[li];
        float lw = sites[li / MAX_LAMPS].lamps[li % MAX_LAMPS].w;
        float2 so = lampSmokeOff(lw);
        float3 b0 = lp.xyz + float3(so.x, lampRise(lw), so.y);
        int kind = int(lp.w);
        float strength = kind == L_WINDOW ? 0.4 : 0.8;
        // Beacons and ritual fires smoke only on the nights they burn.
        if (kind == L_BEACON || kind == L_RITUAL) strength *= saturate(dot(state[li].rgb, float3(0.3, 0.59, 0.11)) * 4.0);
        if (strength <= 0.0) continue;
        // The whole plume lies within 16 of a point 14 up and 6 downwind.
        float3 mid = b0 + float3(wind.x * 6.0, 14.0, wind.y * 6.0) - o;
        float tm = dot(mid, d);
        if (dot(mid, mid) - tm * tm > 256.0 && dot(mid, mid) > 256.0) continue;
        for (int k = 0; k < 2; k++) {
            float h0 = k == 0 ? 0.0 : 10.0, h1 = k == 0 ? 10.0 : 28.0;
            float3 A = b0 + float3(wind.x, 0, wind.y) * (0.15 * h0 + 0.012 * h0 * h0) + float3(0, h0, 0);
            float3 B = b0 + float3(wind.x, 0, wind.y) * (0.15 * h1 + 0.012 * h1 * h1) + float3(0, h1, 0);
            // Closest points between the ray and the segment AB.
            float3 u = B - A, w0 = o - A;
            float bb = dot(d, u), cc = dot(u, u), dd = dot(d, w0), ee = dot(u, w0);
            float den = cc - bb * bb;
            float sP = den > 1e-4 ? saturate((ee - bb * dd) / den) : 0.0;
            float tR = dot(A + u * sP - o, d);
            if (tR <= 0.0 || tR >= tSurf) continue;
            float dist = length(o + d * tR - (A + u * sP));
            float hgt = mix(h0, h1, sP);
            float r = 0.6 + 0.12 * hgt;
            // Thin where it leaves the fire, so a plume seen from above does not veil its hearth.
            float dens = exp(-(dist / r) * (dist / r)) * (1.0 - hgt / 28.0) * smoothstep(0.0, 3.0, hgt) * strength;
            if (dens < 0.01) continue;
            float3 wp = o + d * tR + float3(float(x.origin.x), 0, float(x.origin.y));
            dens *= saturate(0.55 + 0.8 * perlin3(wp / 2.5 - float3(0, x.time * 1.2, 0), x.seed + 330u));
            float al = min(0.28, dens * 0.35) * (1.0 - a);
            // Grey by day; fire-lit from below at night, over a faint moonlit grey.
            float3 day = (x.L.zenith * 0.6 + x.L.horizon * 0.5 + x.L.keyCol * 0.25 * max(x.L.key.y, 0.0)) * 0.9;
            float3 night = state[li].rgb * LAMP_I[int(lp.w)] * 0.12 * (1.0 - smoothstep(0.0, 12.0, hgt)) + float3(0.012, 0.014, 0.02);
            acc += al * mix(day, night, x.L.night);
            a += al;
        }
    }
    return float4(acc, a);
}

// MARK: - Wildlife

// Bird buffer (Sources/Wildlife.swift): MAX_FLOCKS pairs (bounding sphere; first cluster, cluster count), then
// MAX_CLUSTERS pairs (bounding sphere; first bird, count), then four float4s per bird: (map-local position, wingspan),
// (unit heading, arm angle up from level), (bank, kind, tint, wing extension), (hand angle, hand sweep, -, -).
// Screen tiles for the lamp haze and smoke lists (Renderer.lampTiles): a LAMP_TILES x LAMP_TILES grid over the frame.
constant int LAMP_TILES = 16;

constant int MAX_FLOCKS = 8;
constant int MAX_CLUSTERS = 48;
constant int BIRD_STRIDE = 4;

struct BirdHit { float t; float3 n; float3 albedo; };

// Ray against a box with centre c, orthonormal axes a0..a2 and half extents he; returns entry t (1e9 on a miss).
static inline float boxHit(float3 ro, float3 rd, float3 c, float3 a0, float3 a1, float3 a2, float3 he, thread float3& n) {
    float3 p = ro - c;
    float3 lo = float3(dot(p, a0), dot(p, a1), dot(p, a2));
    float3 ld = float3(dot(rd, a0), dot(rd, a1), dot(rd, a2));
    float3 inv = 1.0 / select(ld, copysign(float3(1e-6), ld), abs(ld) < 1e-6);
    float3 t0 = (-he - lo) * inv, t1 = (he - lo) * inv;
    float3 tn = min(t0, t1), tf = max(t0, t1);
    float tN = max(max(tn.x, tn.y), tn.z), tF = min(min(tf.x, tf.y), tf.z);
    if (tN > tF || tF < 0.0) return 1e9;
    float3 sg = -sign(ld);
    n = tn.x >= tN ? a0 * sg.x : (tn.y >= tN ? a1 * sg.y : a2 * sg.z);
    return max(tN, 0.0);
}

// Plumage by kind (0 songbird, 1 raptor, 2 gull, 3 starling, 4 bat, 5 goose) and part (0 body, 1 arm, 2 hand,
// 3 head, 4 tail); `under` for faces turned down, `tip` for the outer hand.
static float3 plumage(int kind, int part, bool under, bool tip) {
    if (kind == 2) {
        if (part == 0 || part >= 3) return float3(0.80, 0.80, 0.78);
        if (tip) return under ? float3(0.12) : float3(0.05);
        return under ? float3(0.74, 0.74, 0.73) : float3(0.40, 0.42, 0.46);
    }
    if (kind == 1) {
        if (tip && part == 2) return float3(0.06, 0.05, 0.04);
        if (part == 3) return float3(0.22, 0.17, 0.11);
        return under ? float3(0.30, 0.25, 0.19) : float3(0.14, 0.10, 0.065);
    }
    if (kind == 5) {
        if (part >= 3) return float3(0.03, 0.03, 0.03);
        if (part == 0) return under ? float3(0.36, 0.33, 0.28) : float3(0.24, 0.21, 0.17);
        return under ? float3(0.30, 0.27, 0.22) : float3(0.17, 0.15, 0.12);
    }
    if (kind == 3) return float3(0.035, 0.035, 0.04);
    if (kind == 4) return part == 1 || part == 2 ? float3(0.05, 0.04, 0.035) : float3(0.03, 0.025, 0.025);
    if (part == 3) return float3(0.07, 0.06, 0.05);
    return under ? float3(0.17, 0.14, 0.11) : float3(0.09, 0.075, 0.06);
}

// A bird is a body and two wings of an arm and a swept hand, plus a head and tail when near. No box is thinner
// than half a cell's footprint, so a far bird stays a speck of a cell or two under the subpixel jitter instead of
// blinking in and out. tOff is path length already travelled (reflections), for the footprint.
static BirdHit traceBirds(device const float4* B, int nFlocks, float3 o, float3 d, float tMax, float pixelAngle, float tOff) {
    BirdHit r;
    r.t = tMax;
    r.n = float3(0, 1, 0);
    r.albedo = float3(0);
    for (int gi = 0; gi < min(nFlocks, MAX_FLOCKS); gi++) {
        float4 g = B[gi * 2];
        float3 og = g.xyz - o;
        float tg = dot(og, d);
        float Rg = g.w + 0.6 * max(tg + tOff, 0.0) * pixelAngle;
        if (tg < -Rg || tg - Rg > r.t || dot(og, og) - tg * tg > Rg * Rg) continue;
        float4 gr = B[gi * 2 + 1];
        int c0 = int(gr.x), c1 = min(int(gr.x) + int(gr.y), MAX_CLUSTERS);
        for (int fi = c0; fi < c1; fi++) {
            float4 s = B[(MAX_FLOCKS + fi) * 2];
            float3 oc = s.xyz - o;
            float tc = dot(oc, d);
            float R = s.w + 0.6 * max(tc + tOff, 0.0) * pixelAngle;
            if (tc < -R || tc - R > r.t || dot(oc, oc) - tc * tc > R * R) continue;
            int2 range = int2(B[(MAX_FLOCKS + fi) * 2 + 1].xy);
            for (int bi = range.x; bi < range.x + range.y; bi++) {
                device const float4* q = B + (MAX_FLOCKS + MAX_CLUSTERS) * 2 + bi * BIRD_STRIDE;
                float4 p0 = q[0];
                float S = p0.w;
                if (S <= 0.0) continue;
                float3 bc = p0.xyz - o;
                float bt = dot(bc, d);
                float m = 0.5 * max(bt + tOff, 0.0) * pixelAngle;
                float Rb = 0.62 * S + 1.8 * m;
                if (bt < -Rb || bt - Rb > r.t || dot(bc, bc) - bt * bt > Rb * Rb) continue;
                float4 p1 = q[1], p2 = q[2], p3 = q[3];
                float3 fw = p1.xyz;
                float3 r0 = normalize(cross(float3(0, 1, 0), fw) + float3(1e-5, 0, 0));
                float3 u0 = cross(fw, r0);
                float3 rr = r0 * cos(p2.x) + u0 * sin(p2.x);
                float3 uu = cross(fw, rr);
                int kind = int(p2.y + 0.5);
                float bodyL = kind == 1 || kind == 5 ? 0.22 : (kind == 4 ? 0.12 : 0.18);
                bool nearBird = S > 10.0 * m;
                float3 n;
                float bestT = r.t;
                int part = -1;
                bool tip = false;
                float3 bn = float3(0);
                float th = boxHit(o, d, p0.xyz, fw, rr, uu, max(float3(bodyL, 0.055, 0.05) * S, m), n);
                if (th < bestT) { bestT = th; part = 0; bn = n; }
                if (nearBird) {
                    float hz = kind == 5 ? bodyL + 0.15 : bodyL + 0.045, hl = kind == 5 ? 0.11 : 0.045;
                    th = boxHit(o, d, p0.xyz + (fw * hz + uu * 0.015) * S, fw, rr, uu, float3(hl, 0.032, 0.032) * S, n);
                    if (th < bestT) { bestT = th; part = 3; bn = n; }
                    if (kind != 4) {
                        th = boxHit(o, d, p0.xyz - fw * (bodyL + 0.06) * S, fw, rr, uu, float3(0.07, 0.055, 0.01) * S, n);
                        if (th < bestT) { bestT = th; part = 4; bn = n; }
                    }
                }
                float L1 = 0.2 * S * p2.w, L2 = 0.28 * S * p2.w;
                float c1 = cos(p1.w), s1 = sin(p1.w), c2 = cos(p3.x), s2 = sin(p3.x);
                float csw = cos(p3.y), ssw = sin(p3.y);
                for (int side = -1; side <= 1; side += 2) {
                    float sd = float(side);
                    float3 d1 = rr * (c1 * sd) + uu * s1;
                    float3 sh = p0.xyz + rr * (0.045 * S * sd) + fw * (0.03 * S);
                    th = boxHit(o, d, sh + d1 * (0.5 * L1), d1, fw, cross(d1, fw), max(float3(0.5 * L1, 0.09 * S, 0.014 * S), m), n);
                    if (th < bestT) { bestT = th; part = 1; bn = n; tip = false; }
                    float3 d2 = (rr * (c2 * sd) + uu * s2) * csw - fw * ssw;
                    float3 n2 = normalize(cross(d2, fw)), ch = cross(n2, d2);
                    float3 hc = sh + d1 * L1 + d2 * (0.5 * L2) + ch * (0.025 * S);
                    th = boxHit(o, d, hc, d2, ch, n2, max(float3(0.5 * L2, 0.062 * S, 0.012 * S), m), n);
                    if (th < bestT) {
                        bestT = th; part = 2; bn = n;
                        tip = dot(o + d * th - hc, d2) > 0.1 * L2;
                    }
                }
                if (part < 0) continue;
                r.t = bestT;
                r.n = bn;
                r.albedo = plumage(kind, part, dot(bn, uu) < -0.3, tip) * (0.85 + 0.3 * p2.z);
            }
        }
    }
    return r;
}

// Birds are lit like small matte objects: key light (with cloud shadow), sky fill from above, ground bounce from below.
static float3 shadeBird(thread const Ctx& x, float3 p, BirdHit b) {
    float3 alb = mix(b.albedo, float3(dot(b.albedo, float3(0.3, 0.59, 0.11))), 0.4 * x.L.night);
    float ndl = max(dot(b.n, x.L.key), 0.0);
    float sh = mix(1.0, cloudShadow(x, p, x.L.key), x.L.sunAmt);
    float deep = x.L.night * smoothstep(0.32, 0.45, -x.L.sun.y);
    float3 sky = mix(x.L.horizon, x.L.zenith, 0.7) * 0.9 * (0.55 + 0.45 * saturate(b.n.y * 0.5 + 0.5));
    sky *= mix(float3(1.0), float3(0.62, 0.58, 0.85), x.L.gold * x.L.sunAmt);
    sky += float3(0.09, 0.12, 0.25) * x.L.night;
    float3 bounce = saturate(-b.n.y) * 0.25 * x.L.keyCol * max(x.L.key.y, 0.0) * float3(0.8, 0.75, 0.6);
    return alb * (x.L.keyCol * ndl * sh * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep)
                  + (sky + bounce) * mix(float3(1.0), float3(0.19, 0.23, 0.38), deep));
}

// Lamp haze and smoke between the eye and a fogged surface at tSurf.
static float3 hazeAndSmoke(thread const Ctx& x, float3 col, device const Site* sites, device const float4* lampPos,
                           device const float4* lampState, constant int4* refs, int4 counts, float3 o, float3 d, float tSurf) {
    if (x.lights.w > 0.0 && counts.x > 0) col += lampHaze(x, lampPos, lampState, refs, counts.x, o, d, tSurf) * 0.05 * x.lights.w;
    if (counts.y > 0) {
        float4 sm = smoke(x, sites, lampPos, lampState, refs, counts.y, o, d, tSurf);
        if (sm.a > 0.0) {
            float T = fogTransmittance(x, o, d, min(tSurf, 600.0));
            col = col * (1.0 - sm.a * T) + sm.rgb * T;
        }
    }
    return col;
}

fragment float4 sceneFragment(VOut in [[stage_in]], constant Frame& f [[buffer(0)]],
                              texture2d<float> hm [[texture(0)]], texture2d<float> cloudTex [[texture(1)]],
                              texture2d<float> clim [[texture(2)]], texture2d<uint> st [[texture(3)]],
                              texture2d<float> lightMap [[texture(4)]], texture2d<float> lightDir [[texture(5)]],
                              texture2d_array<float> coarse [[texture(6)]], texture2d_array<float> coarseClim [[texture(7)]],
                              texture2d_array<float> coarseCover [[texture(8)]],
                              texture2d<float> shadowNear [[texture(9)]], texture2d<float> shadowFar [[texture(10)]],
                              texture2d<float> slowTex [[texture(11)]], texture2d<float> hx [[texture(12)]],
                              texture2d<uint> gx [[texture(13)]], texture2d_array<float> cg [[texture(14)]],
                              texture2d_array<float> cg2 [[texture(17)]],
                              texture2d<float> hb [[texture(15)]], texture2d<float> pvTex [[texture(16)]],
                              texture2d_array<float> cbTex [[texture(18)]],
                              device const Site* sites [[buffer(1)]], device const float4* lampPos [[buffer(2)]],
                              device const float4* lampState [[buffer(3)]], constant int4* refs [[buffer(4)]],
                              device const float4* birds [[buffer(5)]], device const float* horizon [[buffer(6)]],
                              constant uint* tiles [[buffer(7)]], device const GiantRec* giants [[buffer(8)]], constant TowerTable* towers [[buffer(9)]],
                              constant IsleTable* isleTab [[buffer(10)]]) {
    float2 ndc = in.uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    ndc += f.jitter.xy;
    float tanH = f.camPos.w;
    float3 d = normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
    float3 o = f.camPos.xyz;

    int levels = int(f.fade.z);
    float gmax = f.misc.y;
    for (int k = 1; k < levels; k++) gmax = max(gmax, abs(coarse.read(uint2(0), uint(k - 1), coarse.get_num_mip_levels() - 1).r) + 1.0);
    if (int(f.shadow.y) & 2) gmax = min(gmax, 260.0);
    Terrain tr { hm, coarse, coarseCover, coarseClim, clim, f.lod, f.mapInfo.xy, f.mapInfo.z, levels, f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w, f.fade.y, f.fade.x, gmax, hx, gx, cg, cg2, hb, uint(f.mapInfo.w), giants, pvTex, towers, sceneTowers ? towers->n.x : 0, cbTex, f.shadow.w > 0.0 };
    Ctx x { tr, f.mapInfo.xy, uint(f.mapInfo.w), f.camRight.w, f.camUp.w,
            2.0 * tanH * f.resolution.w, f.misc.w, makeLight(f.sunDir.xyz),
            fract(52.9829189 * fract(dot(in.pos.xy + f.misc2.x * float2(5.588, 2.31), float2(0.06711056, 0.00583715)))),
            cloudTex, f.cloud.xyz, st, sites, lampPos, lampState, lightMap, lightDir, f.lights, shadowNear, shadowFar,
            f.misc2.z == 14 ? 2.0 : f.shadow.x, (int(f.shadow.y) & 1) != 0, int(f.shadow.y), slowTex,
            f.slowCloud.w == f.camRight.w ? f.slowCloud : float4(0), 0u };
    {
        int2 ti = clamp(int2(in.uv * float(LAMP_TILES)), int2(0), int2(LAMP_TILES - 1));
        x.tiles = tiles[ti.y * LAMP_TILES + ti.x];
        x.towers = towers;
        x.isles = isleTab;
        x.isleShadows = f.shadow.w > 0.0;
    }

    // Debug map view: 11 level 4 (+-10k tall, +-16k wide), 12 level 2, 13 level 1; hillshade, height ramp, 100-voxel
    // contours, 2k grid, yellow 10k lines, magenta world origin, red camera.
    if (f.misc2.z >= 11 && f.misc2.z <= 13) {
        int k = f.misc2.z == 11 ? 4 : f.misc2.z == 12 ? 2 : 1;
        float cs = float(1 << k);
        float2 uv = in.uv - 0.5;
        uv.x *= f.resolution.x / f.resolution.y;
        float2 cell = f.camPos.xz / cs + float2(f.mapInfo.xy >> k) + uv * 1300.0;
        int2 ci = int2(floor(cell));
        float h0 = coarse.read(uint2(ci & (COARSE_N - 1)), uint(k - 1), 0).r;
        float hx = coarse.read(uint2((ci + int2(1, 0)) & (COARSE_N - 1)), uint(k - 1), 0).r;
        float hz = coarse.read(uint2((ci + int2(0, 1)) & (COARSE_N - 1)), uint(k - 1), 0).r;
        float3 n = normalize(float3(-(hx - h0) / cs, 1.0, -(hz - h0) / cs));
        float lit = saturate(dot(n, normalize(float3(-1, 1.2, 1))));
        float3 c = h0 <= SEA ? float3(0.15, 0.3, 0.55)
                 : h0 < 120 ? mix(float3(0.3, 0.5, 0.25), float3(0.55, 0.5, 0.3), h0 / 120.0)
                 : h0 < 600 ? mix(float3(0.55, 0.5, 0.3), float3(0.5, 0.35, 0.3), (h0 - 120.0) / 480.0)
                 : h0 < 1200 ? mix(float3(0.5, 0.35, 0.3), float3(0.6, 0.6, 0.65), (h0 - 600.0) / 600.0) : float3(0.95);
        c *= 0.35 + 0.9 * lit;
        if (floor(h0 / 100.0) != floor(hx / 100.0) || floor(h0 / 100.0) != floor(hz / 100.0)) c *= 0.6;
        // Gaps (overhangs, caves, bridges) in magenta.
        if (k <= GAP_LEVELS) { float2 gp = cg.read(uint2(ci & (COARSE_N - 1)), uint(k - 1)).rg; if (gp.y > gp.x) c = float3(1, 0, 1); }
        float2 wc = cell * cs;
        // Forest provinces: jungle warm, old growth cool; giants white.
        float3 pv = provincesBaked(pvTex, wc);
        c = mix(c, float3(0.9, 0.5, 0.1), 0.45 * smoothstep(0.15, 0.6, pv.x));
        c = mix(c, float3(1.0, 0.25, 0.55), 0.45 * smoothstep(0.2, 0.6, pv.z));
        c = mix(c, float3(0.2, 0.4, 1.0), 0.45 * smoothstep(0.2, 0.6, pv.y));
        { Giant gm; if (giantAt(giants, wc, 0.0, gm)) c = float3(1); }
        float px = 1300.0 * cs / f.resolution.y;
        float2 g1 = abs(wc - 2000.0 * floor(wc / 2000.0 + 0.5)), g2 = abs(wc - 10000.0 * floor(wc / 10000.0 + 0.5));
        if (any(g1 < px * 0.6)) c = mix(c, float3(0), 0.35);
        if (any(g2 < px * 1.2)) c = float3(1, 1, 0);
        if (all(abs(wc) < px * 4.0)) c = float3(1, 0, 1);
        if (length(uv) < 0.004) c = float3(1, 0, 0);
        return float4(c * c, 1);
    }
    const int primaryIters = 520;
    Hit h;
    h.hit = false; h.t = x.tr.rmax; h.iters = 0; h.level = 0;
    if (f.shadow.z == 0.0 || !aboveHorizon(horizon, d)) h = trace(x.tr, o, d, x.tr.rmax, primaryIters, 0);
    if (f.misc2.z == 1) return float4(float(h.iters) / float(primaryIters), h.hit ? 0.3 : 0.0, h.iters >= primaryIters - 1 ? 1 : 0, 1);
    if (f.misc2.z == 5) return float4(skyColor(d, x.L, false, x.seed, x.time), 1);

    float tw = d.y < 0 ? (SEA - o.y) / d.y : 1e9;
    bool water = d.y < 0 && tw > 0 && (!h.hit || tw < h.t);
    if (f.misc2.z == 9 && (water || !h.hit)) return float4(0, 0, 0, water ? tw : 6e4);
    int nFlocks = int(f.birds.x);
    // Birds in front of land, water and sky. Their pixels carry negative depth, which tells the temporal resolve
    // that they move.
    if (nFlocks > 0) {
        // A trace out of steps has not shown the way clear: a bird past where it stopped could be behind a hill.
        float tFront = !h.hit && h.iters >= primaryIters ? 0.0 : water ? tw : (h.hit ? h.t : 6000.0);
        BirdHit bh = traceBirds(birds, nFlocks, o, d, tFront, x.pixelAngle, 0.0);
        if (bh.t < tFront) {
            float3 bc = shadeBird(x, o + d * bh.t, bh);
            if (f.misc2.z != 2) bc = applyFog(x, bc, o, d, bh.t);
            bc = hazeAndSmoke(x, bc, sites, lampPos, lampState, refs, f.refs, o, d, bh.t);
            return float4(bc, -bh.t);
        }
    }
    // Sea past the world's edge is fully hazed: it takes the fog colour alone, with no ring of ocean around the land.
    if (water && length((o + d * tw).xz - x.tr.cam) >= x.tr.rmax) {
        if (f.misc2.z == 3) return float4(0, 0, 0, 1);
        float3 fogged = hazeAndSmoke(x, applyFog(x, float3(0), o, d, tw), sites, lampPos, lampState, refs, f.refs, o, d, tw);
        return float4(fogged, min(tw, 6e4));
    }
    bool birdRefl = false;
    float3 col;
    float tSurf;
    float Tfog = -1.0;
    // The lagoon bed's shading and the land's share one call: each call site inlines all of shade.
    float3 wp = 0, n = 0, r = 0;
    float2 ww = 0;
    float e0 = 0, lagoon = 0, salt = 0, fres = 0, bedShows = 0;
    int wl = 0;
    Light Ls = x.L;
    bool shadeHit = false;
    if (water) {
        wp = o + d * tw;
        ww = wp.xz + float2(x.origin);
        e0 = perlin3(float3(ww * 0.3, x.time * 0.25), x.seed + 60u);
        float ex = perlin3(float3((ww + float2(0.05, 0)) * 0.3, x.time * 0.25), x.seed + 60u);
        float ez = perlin3(float3((ww + float2(0, 0.05)) * 0.3, x.time * 0.25), x.seed + 60u);
        wl = levelAt(x.tr, length(wp.xz - x.tr.cam));
        Climate wc = climateL(x.tr, wl, wp.xz);
        lagoon = smoothstep(0.0, 0.35, lagoonAmount(wc));
        // Pools on a salt pan are a still skin over pale crust: a near-perfect mirror.
        salt = saltAmount(wc);
        float rs = max(0.35 * saturate(1.0 - tw * x.pixelAngle * 4.0) * (1.0 - 0.6 * lagoon), 0.08 * x.L.night) * (1.0 - 0.7 * salt);
        n = normalize(float3(-(ex - e0) * rs, 1.0, -(ez - e0) * rs));
        r = reflect(d, n);
        r.y = abs(r.y);
        float cosi = saturate(-dot(d, n));
        fres = 0.02 + 0.98 * pow(1.0 - cosi, 5.0);
        float bedPath = h.t - tw;
        // How much of the lit bed survives the water and the surface reflection; below a few thousandths it
        // can't show, and its shade() is the costliest part of a water pixel.
        bedShows = (1.0 - fres) * max(exp(-bedPath * mix(0.35, 0.08, salt)) * (1.0 - lagoon),
                                      lagoon * exp(-(bedPath + SEA - (o.y + d.y * h.t)) * 0.065));
        // Lagoon beds are lit by sunlight refracted steeply down through the surface. Flux per horizontal
        // area is kept (less what the surface reflects), so the light is redistributed from risers to treads.
        float2 kh = x.L.key.xz / 1.33;
        float3 kr = float3(kh.x, sqrt(max(1.0 - dot(kh, kh), 0.0)), kh.y);
        float ky = max(x.L.key.y, 0.0);
        Ls.key = normalize(mix(x.L.key, kr, lagoon));
        Ls.keyCol *= mix(1.0, ky / kr.y * (0.98 - 0.98 * pow(1.0 - ky, 5.0)), lagoon);
        shadeHit = h.hit && bedShows > 0.003;
    } else if (h.hit) {
        if (f.misc2.z != 2) Tfog = fogTransmittance(x, o, d, h.t);
        shadeHit = !((x.prof & 64) || (Tfog >= 0.0 && Tfog < 0.01));
    }
    float3 lit = shadeHit ? shadeL(x, Ls, o, d, h, water ? 0 : 1) : float3(0);
    if (water) {
        float3 refl;
        // Land in the reflected ray hides the sun or moon its glitter would mirror.
        float open = 1.0;
        if (x.quality > 0.5 && (int(f.shadow.y) & 4) == 0) {
            // Rippled reflections of water past 300 trace from the 2-voxel ring: its far shore reads the same, at half
            // the steps.
            Hit rh = trace(x.tr, wp + float3(0, 0.02, 0), r, x.tr.rmax, 260, max(wl, tw > 600.0 ? 2 : tw > 300.0 ? 1 : 0));
            // Shadowed: unshadowed, a sun-facing wall in the shade of the far rim mirrors as a pale streak.
            if (rh.hit) {
                open = 0.0;
                // Land the haze all but hides needs no shading.
                float T = fogTransmittance(x, wp, r, rh.t, 2);
                refl = applyFogT(x, T < 0.01 || (x.prof & 8) ? float3(0.0) : shadeL(x, x.L, wp, r, rh, x.shadowMode == 1.0 ? 2 : 1, true), wp, r, T);
            } else {
                refl = softCloudSky(x, wp, r);
            }
            if (nFlocks > 0) {
                float tB = rh.hit ? rh.t : (rh.iters >= 260 ? 0.0 : 3000.0);
                BirdHit bh = traceBirds(birds, nFlocks, wp, r, tB, x.pixelAngle, tw);
                if (bh.t < tB) { refl = applyFog(x, shadeBird(x, wp + r * bh.t, bh), wp, r, bh.t); birdRefl = true; }
            }
        } else {
            refl = softCloudSky(x, wp, r);
        }
        float3 deep = x.L.zenith * 0.18 + x.L.horizon * 0.06 + float3(0.004, 0.014, 0.018);
        float3 under = deep;
        float bedPath = h.t - tw;
        if (h.hit) {
            float path = bedPath;
            under = mix(deep, lit * mix(float3(0.55, 0.8, 0.85), float3(0.9, 0.95, 0.97), salt), exp(-path * mix(0.35, 0.08, salt)));
            if (lagoon > 0.0) {
                // Clear water over sand: red is absorbed first, so the bed turns aquamarine, then teal, then blue.
                // Light crosses the water twice: down to the bed, then up along the view path.
                float3 T = exp(-(path + SEA - (o.y + d.y * h.t)) * float3(0.30, 0.09, 0.065));
                float3 glow = (x.L.zenith * 0.6 + x.L.keyCol * max(x.L.key.y, 0.0) * 0.12) * float3(0.06, 0.30, 0.50);
                float3 lag = lit * T + glow * (1.0 - T.g);
                // Keep the shallows soft rather than swimming-pool cyan; moonlit, only a hint of colour remains.
                lag = mix(lag, float3(dot(lag, float3(0.3, 0.59, 0.11))), 0.35 + 0.25 * x.L.night) * 0.9;
                under = mix(under, lag, lagoon);
            }
        }
        under *= 1.0 - 0.4 * x.L.night;
        col = mix(under, refl, fres);
        // Glitter path toward the sun or moon: the main EDR highlight.
        col += x.L.keyCol * 6.0 * pow(max(dot(r, x.L.key), 0.0), mix(mix(900.0, 4000.0, x.L.night), 350.0, x.L.gold)) * fres * (x.L.shadows ? 1.0 : 0.3) * (1.0 + 0.5 * x.L.night) * open;
        // Golden hour: a broad sheen of the aureole on the water under the sun.
        col += x.L.keyCol * 0.5 * pow(max(dot(r, x.L.key), 0.0), 40.0) * fres * x.L.gold * open;
        if (x.L.night > 0.0 && open > 0.0) {
            // Moon path: the chance that some ripple facet mirrors the moon (a slope distribution, so it is
            // smooth and TAA-stable). Slopes along the view spread wider than across it, so the path runs out
            // toward the horizon; a broad skirt under the core lets it fade out rather than stop.
            float3 hv = normalize(x.L.moon - d);
            float2 slope = hv.xz / max(hv.y, 0.05);
            float2 vh = normalize(d.xz + 1e-5);
            float sa = dot(slope, vh), sc = slope.x * vh.y - slope.y * vh.x;
            float q = sa * sa / 0.05 + sc * sc / 0.012;
            float cover = 0.55 * exp(-q) + 0.25 * exp(-0.2 * q);
            // Glitter: a facet lights where the ripple field clears a threshold set by the cover, so the core is a
            // broken sheen and the skirt thins to scattered sparks. Each octave hands over to the next coarser as
            // its features shrink toward a pixel, measured along the foreshortened view; far out, swells long
            // across the moon's bearing break the path into bands, and past them the cover itself remains.
            float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
            float fine = perlin3(float3(ww * 1.6, x.time * 0.7), x.seed + 61u);
            float2 mz = normalize(x.L.moon.xz);
            float swell = perlin3(float3(dot(ww, mz) * 0.07, (ww.x * mz.y - ww.y * mz.x) * 0.02, x.time * 0.1), x.seed + 62u);
            float rip = mix(swell, mix(e0, fine, saturate(1.0 - fpv * 2.0)), saturate(1.0 - fpv * 0.25));
            // The ripple noise is near-Gaussian (sd ~0.22); a logistic quantile turns cover into its threshold.
            float thr = 0.13 * log((1.0 - cover) / max(cover, 1e-4));
            float glint = mix(cover, smoothstep(thr - 0.07, thr + 0.07, rip), saturate(1.0 - fpv * 0.04));
            col += float3(0.75, 0.82, 1.0) * x.L.night * (0.05 * cover + 0.55 * glint);
        }
        // Water past the 1-voxel map lies beyond the light map, which would clamp its edge onto it.
        if (x.lights.w > 0.0 && all(wp.xz > 0.0) && all(wp.xz < float(x.tr.size))) {
            // Lamplight on the water: a soft pool, and a broken streak drawn toward the viewer as ripples
            // stretch the moon's path.
            constexpr sampler bl(filter::linear, address::clamp_to_edge);
            float2 vh = normalize(d.xz + 1e-5);
            float inv = 1.0 / float(x.tr.size);
            float3 E = x.lightMap.sample(bl, wp.xz * inv, level(0)).rgb * 0.5 + x.lightMap.sample(bl, (wp.xz + vh * 4.0) * inv, level(0)).rgb * 0.3
                     + x.lightMap.sample(bl, (wp.xz + vh * 10.0) * inv, level(0)).rgb * 0.2;
            if (any(E > 1e-4)) {
                // Ripple facets on fixed world axes; the streak toward the viewer comes from the offset samples
                // above. A basis turned to each pixel's view direction would multiply the jitter by the world
                // coordinates and crawl. Facets hand over to their mean before they shrink to a pixel along the
                // foreshortened view.
                float ripple = perlin3(float3(ww * 1.6, x.time * 0.7), x.seed + 63u) + 0.3 * e0;
                float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
                float spark = mix(smoothstep(0.08, 0.38, ripple), 0.22, saturate(fpv * 2.0));
                col += E * (0.06 + 0.5 * spark) * x.lights.w;
            }
        }
        tSurf = tw;
    } else if (h.hit) {
        col = lit;
        tSurf = h.t;
        // Debug: flat level colours, morph bands white.
        if (f.misc2.z == 9) {
            const float3 flat[5] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0), float3(1, 0, 1) };
            return float4(morphAt(x.tr, h.level, (o + d * h.t).xz) > 0.0 ? float3(1) : flat[h.level], h.t);
        }
        // Debug: levels tinted, morph bands striped toward the next level's tint.
        if (f.misc2.z == 8) {
            const float3 tint[5] = { float3(1, 0.3, 0.3), float3(0.3, 1, 0.3), float3(0.3, 0.5, 1), float3(1, 1, 0.3), float3(1, 0.3, 1) };
            float w = morphAt(x.tr, h.level, (o + d * h.t).xz);
            float l = dot(col, float3(0.3, 0.59, 0.11));
            return float4(mix(tint[h.level], tint[min(h.level + 1, 4)], w) * (0.15 + l), h.t);
        }
    } else {
        float4 sky = skyWithClouds(x, o, d);
        if (x.lights.w > 0.0 && f.refs.x > 0) sky.rgb += lampHaze(x, lampPos, lampState, refs, f.refs.x, o, d, 2500.0) * 0.05 * x.lights.w;
        if (f.refs.y > 0) {
            float4 sm = smoke(x, sites, lampPos, lampState, refs, f.refs.y, o, d, 1e9);
            sky.rgb = sky.rgb * (1.0 - sm.a) + sm.rgb;
        }
        return sky;
    }
    if (f.misc2.z == 10) {
        // Site footprints tinted by type.
        float3 sp = o + d * tSurf;
        int2 q = int2(floor(sp.xz));
        int2 home = int2(floor(float2(q + x.origin) / float(SITE_CELL))) - siteCell0(x.origin);
        const float3 tint[7] = { float3(0), float3(1, 0.2, 1), float3(0.6, 0.6, 0.6), float3(1, 0.5, 0), float3(1, 1, 0.2),
                                 float3(0.2, 0.6, 1), float3(1, 0.2, 0.2) };
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                int2 gi = home + int2(i, j);
                if (any(gi < 0) || any(gi >= SITE_GRID)) continue;
                Site st = sites[gi.y * SITE_GRID + gi.x];
                if (st.head.z == 0) continue;
                float dd = length(float2(q + x.origin - st.head.xy));
                if (dd < max(st.info.x, 6.0)) col = mix(col, tint[st.head.z] * 0.6, dd < 3.0 ? 0.9 : 0.45);
            }
    }
    if (f.misc2.z == 3) return float4(float3(fogTransmittance(x, o, d, tSurf)), 1);
    if (f.misc2.z != 2) col = Tfog >= 0.0 ? applyFogT(x, col, o, d, Tfog) : applyFog(x, col, o, d, tSurf);
    // A summit above the cloud base is seen through the slab in front of it.
    if (!(x.prof & 32) && !water && h.hit && o.y < CLOUD_Y0 && o.y + d.y * h.t > CLOUD_Y0) {
        CloudSample cl = cloudSheet(x, o, d, h.t, 6);
        // A flat base would cut a level line across every wall that rises into it: the veil thickens up the wall.
        // Thin cover would wash the whole wall evenly, which reads as terrain gone translucent rather than as cloud:
        // the veil is drawn only where the cloud in front is dense, as an opaque body with soft edges.
        col = mix(col, cl.col, smoothstep(0.35, 0.8, cl.a) * smoothstep(CLOUD_Y0, CLOUD_Y0 + 110.0, o.y + d.y * h.t));
    }
    col = hazeAndSmoke(x, col, sites, lampPos, lampState, refs, f.refs, o, d, tSurf);
    return float4(col, birdRefl ? -tSurf : tSurf);
}


static inline float3 tmap(float3 v) { return v / (1.0 + max(max(v.r, v.g), v.b)); }
static inline float3 itmap(float3 v) { return v / max(1.0 - max(max(v.r, v.g), v.b), 1e-4); }

// 9-tap Catmull-Rom via bilinear taps: bilinear history fetches blur edges a little more every frame the
// camera moves, which TAA then accumulates.
static float4 sampleCatmullRom(texture2d<float> t, sampler s, float2 uv) {
    float2 size = float2(t.get_width(), t.get_height());
    float2 sp = uv * size;
    float2 tc = floor(sp - 0.5) + 0.5;
    float2 f = sp - tc;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t0 = (tc - 1.0) / size, t3 = (tc + 2.0) / size, t12 = (tc + w2 / w12) / size;
    float4 c = 0;
    c += t.sample(s, float2(t0.x, t0.y)) * w0.x * w0.y;
    c += t.sample(s, float2(t12.x, t0.y)) * w12.x * w0.y;
    c += t.sample(s, float2(t3.x, t0.y)) * w3.x * w0.y;
    c += t.sample(s, float2(t0.x, t12.y)) * w0.x * w12.y;
    c += t.sample(s, float2(t12.x, t12.y)) * w12.x * w12.y;
    c += t.sample(s, float2(t3.x, t12.y)) * w3.x * w12.y;
    c += t.sample(s, float2(t0.x, t3.y)) * w0.x * w3.y;
    c += t.sample(s, float2(t12.x, t3.y)) * w12.x * w3.y;
    c += t.sample(s, float2(t3.x, t3.y)) * w3.x * w3.y;
    return max(c, 0.0);
}

// Temporal resolve, upsampling: runs at output resolution over a jittered scene that may be smaller. Each
// scene sample lands at its jittered position and is splatted by distance in output pixels, so detail finer
// than a scene pixel builds up over frames. History is reprojected through the previous camera (the world
// is static) and clamped to the scene's 3x3 neighbourhood. Scene alpha carries hit distance; history alpha
// carries the accumulated sample weight.
fragment float4 taaFragment(VOut in [[stage_in]], texture2d<float> cur [[texture(0)]],
                            texture2d<float> hist [[texture(1)]], constant Frame& f [[buffer(0)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    float2 outSize = float2(hist.get_width(), hist.get_height());
    float2 inSize = float2(cur.get_width(), cur.get_height());
    float2 toOut = outSize / inSize;
    // Past 2x the splat widens with the sample spacing, or pixels between samples would hardly ever update.
    float spread = max(1.0, toOut.x * toOut.y / 4.0);
    float2 jit = float2(f.jitter.x, -f.jitter.y) * 0.5 * inSize;
    float2 sp = in.pos.xy / toOut - jit;
    int2 base = int2(floor(sp));
    int2 lim = int2(inSize) - 1;
    float3 sum = 0, mn = 1e9, mx = -1e9;
    float wsum = 0, depth = 1e9;
    bool moving = false;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 k = clamp(base + int2(i, j), int2(0), lim);
            float4 v = cur.read(uint2(k));
            float2 dpx = (float2(base + int2(i, j)) + 0.5 - sp) * toOut;
            float w = exp(-2.29 * dot(dpx, dpx) / spread);
            float3 t = tmap(v.rgb);
            sum += t * w;
            wsum += w;
            mn = min(mn, t);
            mx = max(mx, t);
            depth = min(depth, abs(v.a));
            moving = moving || v.a < 0.0;
        }
    float3 c = sum / wsum;
    float4 fresh = float4(itmap(c), wsum);
    if (f.prevPos.w < 0.5 || f.misc2.z == 7) {
        if (all(toOut == 1.0)) return float4(cur.read(uint2(clamp(int2(in.pos.xy), int2(0), lim))).rgb, wsum);
        return fresh;
    }

    float2 ndc = in.uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float tanH = f.camPos.w;
    float3 d = normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
    float3 wp = f.camPos.xyz + d * min(depth, 1e5);
    float3 v = wp - f.prevPos.xyz;
    float z = dot(v, f.prevFwd.xyz);
    if (z <= 0.01) return fresh;
    float2 pn = float2(dot(v, f.prevRight.xyz) / (z * tanH * f.camFwd.w), dot(v, f.prevUp.xyz) / (z * tanH));
    float2 puv = float2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
    if (any(puv < 0.0) || any(puv > 1.0)) return fresh;
    // Clamp and blend in a compressed space so sub-pixel EDR spikes can't smear or flicker.
    float4 hs = sampleCatmullRom(hist, lin, puv);
    float3 h = tmap(hs.rgb);
    float3 hc = clamp(h, mn, mx);
    // Capped so the blend never falls below 0.12 of a full-resolution sample's weight; a history the clamp
    // had to move is partly stale, so it counts for less.
    float hw = min(hs.a, 10.0) * (1.0 - smoothstep(0.02, 0.15, length(hc - h)) * 0.75);
    // Birds move against the static world the reprojection assumes: keep only a short history near them, a slight
    // motion blur rather than a trail. Relative to this pixel's own sample weight, which is smaller when upsampling,
    // so history never outweighs it.
    if (moving) hw = min(hw, min(0.8, 0.65 * wsum));
    float a = wsum / (wsum + hw);
    return float4(itmap(mix(hc, c, a)), hw + wsum);
}

static inline float3 acesFit(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

static inline float3 linearToSRGB(float3 c) {
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

static inline float3 srgbToLinear(float3 c) {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

// misc2.w is the display's EDR headroom; 0 means SDR output (sRGB-encoded 8-bit).
fragment float4 postFragment(VOut in [[stage_in]], texture2d<float> scene [[texture(0)]],
                             constant Frame& f [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    // Golden hour sits lower on the curve, below the shoulder where ACES bleaches warm skies to cream.
    float sunUp = smoothstep(-0.04, 0.04, f.sunDir.y);
    // Retro pixels: the history is the cell grid; each output pixel takes its cell's flat colour.
    float2 grid = float2(scene.get_width(), scene.get_height());
    float2 cellId = min(floor(in.uv * grid), grid - 1.0);
    float2 uv = (cellId + 0.5) / grid;
    float3 x = scene.sample(s, uv).rgb * f.misc.x * mix(1.0, 0.75, goldAmount(f.sunDir.y) * sunUp);
    float3 c = acesFit(x);
    // Noon light is the flattest; restore a little chroma the tonemap shoulder takes away.
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = max(mix(float3(l), c, 1.0 + 0.10 * smoothstep(0.2, 0.45, f.sunDir.y)), 0.0);
    // 8x8 Bayer, fixed to the screen so it never flickers; 8 levels per channel in sRGB.
    uint2 b = uint2(cellId) & 7u;
    uint bx = b.x ^ b.y;
    uint bayer = ((bx & 1u) << 5) | ((b.y & 1u) << 4) | ((bx & 2u) << 2) | ((b.y & 2u) << 1) | ((bx & 4u) >> 1) | ((b.y & 4u) >> 2);
    const float levels = 7.0;
    float3 enc = floor(linearToSRGB(saturate(c)) * levels + (float(bayer) + 0.5) / 64.0) / levels;
    float headroom = f.misc2.w;
    if (headroom <= 1.0) return float4(enc, 1);
    // EDR: identical to SDR below scene value 1; energy above it rolls per channel into the headroom.
    // EDR: a hue-preserving luminance knee above diffuse white, so only glitter, sun, moon and
    // bright stars rise into the headroom.
    float3 lin = srgbToLinear(saturate(enc));
    float Y = dot(x, float3(0.2126, 0.7152, 0.0722));
    float e = (headroom - 1.0) * (1.0 - exp(-max(Y - 1.6, 0.0) / 2.5));
    lin *= 1.0 + e / max(dot(lin, float3(0.2126, 0.7152, 0.0722)), 1e-3);
    if (f.misc2.z == 4) return float4(e > 0.05 ? float3(1, 0, 0) : lin, 1);
    return float4(lin, 1);
}
