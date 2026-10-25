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
    // Land inside the slab is in its cloud; land above it is in sunlight. The shade fades out as the sun nears the
    // horizon: at L.y 0.02 it cut off at 40% and the whole land flashed as the sun crossed it.
    return mix(1.0, 0.32, smoothstep(0.02, 0.5, L.y) * smoothstep(0.0, 0.14, dens) * smoothstep(CLOUD_Y1, CLOUD_Y0, p.y));
}
