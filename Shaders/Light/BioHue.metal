// The glow's hue at seed-free world position w: regions hundreds of voxels across lean teal, cyan or blue, with rare
// violet pockets warming to magenta at their hearts. One hue per region: random hues dot to dot read as confetti.
static inline float3 bioHue(float2 w, uint seed) {
    // Sines at ~650 voxels: at that scale their regularity never shows, and a noise would cost four hashes.
    float sp = float(seed & 1023u) * 0.37;
    float a = 0.22 * (sin(dot(w, float2(0.0083, 0.0049)) + sp) + sin(dot(w, float2(-0.0041, 0.0089)) + 1.7 * sp)
                    + sin(dot(w, float2(0.0047, -0.0043)) + 2.3 * sp));
    float3 teal = BIO_LAND[1], cyan = float3(0.04, 0.58, 1.0), blue = mix(BIO_LAND[0], cyan, 0.35);
    float t = smoothstep(-0.35, 0.2, a);
    float3 h = t < 0.5 ? mix(teal, cyan, t * 2.0) : mix(cyan, blue, t * 2.0 - 1.0);
    // Wide ramps: a narrow one rings each pocket with a hard violet edge.
    float v = bioNoise(w / 620.0 + 31.7, seed + 722u);
    h = mix(h, float3(0.42, 0.10, 1.0), smoothstep(0.16, 0.36, v));
    h = mix(h, float3(0.75, 0.10, 0.75), 0.5 * smoothstep(0.28, 0.48, v));
    // Blue and violet lifted a little toward teal's brightness (x1.2 at most): more reads as a saturated floodlight.
    return h * mix(1.0, 0.5 / max(dot(h, float3(0.3, 0.59, 0.11)), 0.05), 0.3);
}

// Slow swells of light that travel across a grove (a front ~70 voxels apart, bent by warp, a noise in about
// [-0.75, 0.75] over ~150 voxels, every ~8 s) over a slower breathing per region: from about half to full, never a flicker.
static inline float bioPulse(float2 w, float t, uint seed, float warp) {
    float a0 = float(seed & 1023u) * 0.00614;
    float ph = dot(w, float2(cos(a0), sin(a0))) * (6.2831853 / 70.0) + 4.0 * warp - t * 0.75;
    float sw = 0.5 + 0.5 * sin(ph);
    return (0.5 + 0.5 * sw * sw) * (0.85 + 0.15 * sin(t * 0.45 + dot(w, float2(0.013, 0.01))));
}
