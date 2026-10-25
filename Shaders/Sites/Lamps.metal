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
