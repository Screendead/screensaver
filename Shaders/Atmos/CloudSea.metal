// Sea of cloud: around sunrise the night's cold air lies pooled in the low ground under an inversion, a flat layer
// that hills and trees stand out of. It gathers in the blue hour and burns off as the sun climbs. Exactly 0 at
// dusk (night progress stays under 0.2 until the sun is well down), by day and at night, so those frames are untouched.
// Compiled in only while the sea of cloud is out (SEA_FC, Renderer picks the pipeline): held off in the scene and vine
// passes it costs every pixel about 0.3 ms (pin606n, dusk).
constant bool SEA_FC [[function_constant(8)]];
constant bool SCENE_SEA = is_function_constant_defined(SEA_FC) && SEA_FC;

static inline float cloudSeaAmount(thread const Ctx& x) {
    if (!SCENE_SEA) return 0.0;
    float e = x.L.sun.y;
    return smoothstep(0.5, 0.75, x.lights.y) * smoothstep(-0.24, -0.08, e) * (1.0 - smoothstep(0.08, 0.30, e));
}

struct SeaFog { float T; float3 col; };  // col is premultiplied: seen = behind * T + col

// Antiderivative in u = top - y of the density profile min(u / SEA_SOFT, 1) above the top.
constant float SEA_SOFT = 5.0;
constant float SEA_LID = SEA + 14.0;
static inline float seaG(float u) { return u <= 0.0 ? 0.0 : u < SEA_SOFT ? u * u / (2.0 * SEA_SOFT) : u - 0.5 * SEA_SOFT; }

// Closed form along the ray, no march; one noise sample where the ray meets the top sets the billows and the gaps.
// The low ground round frame-local xz p: the least of nine taps of the coarsest level's mean heights, 80 apart, so a
// hill narrower than that stands out of its valley's level. Means over water are the seabed.
static inline float seaFloorAt(thread const Ctx& x, float2 p) {
    constexpr sampler br(filter::linear, address::repeat);
    int k = x.tr.levels - 1;
    float w = x.tr.lod[k].w;
    float2 base = float2(levelOrigin(x.tr, k) & (COARSE_N - 1));
    float lo = 1e9;
    for (int i = 0; i < 9; i++) {
        float2 q = p + float2(i % 3 - 1, i / 3 - 1) * 80.0;
        lo = min(lo, x.tr.ch.sample(br, (q / w + base) / float(COARSE_N), uint(k - 1), level(0)).r);
    }
    return max(lo, SEA);
}

static SeaFog cloudSea(thread const Ctx& x, float amt, float3 o, float3 d, float t) {
    SeaFog s = { 1.0, float3(0) };
    float burn = smoothstep(0.08, 0.30, x.L.sun.y);
    t = min(t, 3000.0);
    float3 ph = o + d * t;
    // Burning off, the layer sinks as well as thins. The closed form below needs one top per ray: the hit's.
    // The inversion's lid: no top rises over SEA_LID, so the dawn flight (at least 38 x the amount over the sea,
    // Camera.swift) always looks down on the layer; a ray starting under its hit's top would be fogged all its length.
    // Basins well above the sea, and hillsides, stand clear of it.
    float top0 = min(seaFloorAt(x, ph.xz) + 8.0 * amt, SEA_LID);
    float y0 = o.y, y1 = ph.y;
    if (min(y0, y1) > top0 + 3.0) return s;
    float te = d.y < 0.0 && y0 > top0 ? min((y0 - top0) / -d.y, t) : 0.0;
    float2 q = o.xz + d.xz * te + float2(x.origin) - float2(0.35, 0.12) * x.time;
    float n = perlin2(q / 110.0, x.seed + 71u) + 0.3 * perlin2(OCT_ROT * q / 41.0, x.seed + 72u);
    float top = min(top0 + 3.0 * n * amt, SEA_LID);
    // Gaps open where the noise is low, wider as the sun climbs.
    float dens = 0.16 * amt * saturate((n + 0.55 - 1.6 * burn) * 2.5);
    if (dens <= 0.0) return s;
    float depth = abs(d.y) > 1e-3 ? abs(seaG(top - y1) - seaG(top - y0)) / abs(d.y) : saturate((top - 0.5 * (y0 + y1)) / SEA_SOFT) * t;
    s.T = exp(-dens * depth);
    if (s.T > 0.999) { s.T = 1.0; return s; }
    float3 pe = float3(o.x + d.x * te, min(top, y0), o.z + d.z * te);
    // Lit as a volume, not a face: a grazing sun still lights the top, brightest looking toward it. Only the sky's
    // clouds shade it: hills' shadows would need the shadow maps, and their windows' edges show on a flat sheet.
    float vis = cloudShadow(x, pe, x.L.key);
    float3 ambient = x.L.zenith * 0.8 + x.L.horizon * 0.3;
    float3 sun = x.L.keyCol * (0.3 + min(hg(dot(d, x.L.key), 0.55), 1.2)) * vis;
    // Billow crowns catch the sun; the troughs between them sit in their own violet shade. A finer octave and the
    // relief's fall toward the sun (one step sunward) shade the billows' sides, so from above the top reads as a
    // surface rather than a flat sheet.
    float2 sd = 8.0 * normalize(x.L.key.xz + float2(1e-4, 0));
    float rel = n + 0.12 * perlin2(q / 23.0, x.seed + 73u);
    float2 qs = q + sd;
    float relS = perlin2(qs / 110.0, x.seed + 71u) + 0.3 * perlin2(OCT_ROT * qs / 41.0, x.seed + 72u) + 0.12 * perlin2(qs / 23.0, x.seed + 73u);
    float facing = saturate(0.5 + 4.5 * (rel - relS));
    float crown = saturate(0.6 + 0.7 * rel);
    s.col = (ambient * mix(float3(0.75, 0.72, 0.95), float3(1.0), crown) * (0.8 + 0.4 * facing)
             + sun * crown * crown * (0.3 + 1.4 * facing)) * 0.62 * (1.0 - s.T);
    return s;
}
