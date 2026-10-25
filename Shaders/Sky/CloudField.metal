// Cumulus base about 1100, over all but the alpine summits and the tepuis, which pierce it. The whole cloud field
// is scaled by CLOUD_S from its original 240-high design, so from cruise height the sky looks as before.
constant float CLOUD_Y = 1100.0;
constant float CLOUD_S = 3.5;
// Scene depth marking the evening star's core pixels, past any real depth (sky is 6e4).
constant float STAR_DEPTH = 65000.0;
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
