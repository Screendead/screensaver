// met: a shooting star in front of the stars. Where what it adds through the clouds reaches `seen`, a is negative,
// which tells the temporal resolve the pixel moves.
static float4 skyWithClouds(thread const Ctx& x, constant Frame& f, float3 o, float3 d, float3 met, float seen, float2 px) {
    float3 raw = skyColor(d, x.L);
    float3 bare = raw;
    // Forward-scatter glow near the horizon, matching what applyFog adds to fogged land.
    bare += x.L.keyCol * min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night) * smoothstep(0.1, 0.0, d.y);
    Planet P = planetOf(f);
    float3 discs = skyDiscs(raw, d, x.L, x.seed, x.time, x.pixelAngle, px, P, moonPhase(x.seed, x.lights.z)) - raw + aurora(x, d) + met;
    bool star = P.amt > 0.0 && all(abs(px - P.px) < P.reach);
    CloudSample c = cloudSheet(x, o, d, x.prof1 ? 7000.0 : 1e9);
    // Clouds hide the stars, moon and sun behind them.
    // At night thin cloud veils the moon instead of hiding it.
    float veil = mix(1.0 - saturate(c.a * 3.0), (1.0 - c.a) * (1.0 - c.a), x.L.night);
    float3 sky = bare + discs * veil;
    float depth = c.a > 0.5 ? c.t : 6e4;
    return float4(mix(sky, c.col, c.a), star ? STAR_DEPTH : max(met.r, max(met.g, met.b)) * veil * (1.0 - c.a) >= seen ? -depth : depth);
}

// Reflected sky: clouds softened (rippled water blurs them), no stars or discs; glitter handles the lights.
static float3 softCloudSky(thread const Ctx& x, float3 o, float3 r) {
    float3 sky = skyColor(r, x.L);
    if (r.y < 0.02) return sky;
    sky += 0.6 * aurora(x, r);
    float t = (CLOUD_Y - o.y) / r.y;
    float2 q = o.xz + r.xz * t + float2(x.origin);
    // Soft coverage, blurred as if by ripples, matching the volumetric sky.
    float dens = smoothstep(0.0, 0.3, cloudDensity(q, x.seed, x.time, x.cover) - 0.07 * x.L.night);
    float3 lit = mix(x.L.keyCol * 0.4 + x.L.zenith * 1.2 + x.L.horizon * 0.4, float3(0.04, 0.047, 0.07) * x.L.moonAmt, x.L.night);
    return mix(sky, lit, dens * 0.4 * exp(-t / (2600.0 * CLOUD_S)));
}
