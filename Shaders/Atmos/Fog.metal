// Converges on the actual sky colour as fog thickens (T -> 0), so fogged land meets the sky with no seam (less the
// land's solid floor, which fades out toward the world's edge).
static inline float3 fogColor(thread const Ctx& x, float3 d, float T, float solid = 0.0) {
    // At night the sky's horizon gradient is mirrored below it, so far land and sea dim smoothly downward.
    float y = max(d.y, -d.y * x.L.night);
    float3 c = skyColor(float3(d.x, mix(y, y * 0.5 + 0.12, T), d.z), x.L);
    // Aerial colour cools away from the sun so far ridges separate from the sky.
    float away = saturate(dot(normalize(d.xz + 1e-5), normalize(x.L.sun.xz + 1e-5)) * 0.5 + 0.5);
    float3 cool = mix(float3(0.80, 0.82, 0.95), float3(0.52, 0.55, 0.88), x.L.gold);
    return c * mix(float3(1.0), mix(cool, float3(1.0), away * away), T) * mix(float3(1.0), cool * 0.9, solid * (1.0 - T));
}

constant float LAND_SOLID = 0.5;

// Share of the lowland air haze left at height y (Sources/Sight.swift ports it).
static inline float airAloft(float y) { return exp(-max(y - 120.0, 0.0) / 400.0); }

// Exponential height fog plus a slow-drifting noise modulation that pools mist in valleys.
// mistSamples < 0 takes the quality's count; 0 leaves out the valley mist (the closed-form terms only).
static float fogTransmittance(thread const Ctx& x, float3 o, float3 d, float t, int mistSamples) {
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

// solid: land, which even fully hazed stays a shade darker and cooler than the sky behind it, so a far range or a
// tepui's upper wall reads as a solid layer and not a cut-out of the sky; 0 at the world's edge, which goes to sky.
static float3 applyFogT(thread const Ctx& x, float3 col, float3 o, float3 d, float T, float solid = 0.0) {
    // Thin haze in-scatters less than the full horizon radiance; only thick fog converges on the sky.
    // Forward scatter fades out above the horizon as the sky's does, so fully hazed land matches the sky behind it.
    float scatter = min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night) * mix(1.0, smoothstep(0.1, 0.0, d.y), smoothstep(0.5, 0.0, T));
    float3 f = (fogColor(x, d, T, solid) + x.L.keyCol * scatter) * mix(mix(0.3, 0.12, x.L.gold), 1.0, smoothstep(0.1, 0.9, 1.0 - T));
    return mix(f, col, T);
}

static float3 applyFog(thread const Ctx& x, float3 col, float3 o, float3 d, float t) {
    return applyFogT(x, col, o, d, fogTransmittance(x, o, d, t));
}
