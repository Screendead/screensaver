// The swash on a beach tread just above the sea: x wetness out to the wash's highest reach, y the film of a wash that
// has just run up, z its foam front. Zero beyond the 1-voxel map's near ring, where the band is under a pixel.
static float3 shoreWash(thread const Ctx& x, float3 p, float H, float fp) {
    float fade = 1.0 - smoothstep(0.35, 0.7, fp);
    if (fade <= 0.0) return 0;
    Climate c = climateL(x.tr, 0, p.xz);
    float dry = smoothstep(0.3, 0.6, saltAmount(c));
    if (dry >= 1.0 || snowlineAt(c.temp) < H + 6.0) return 0;
    // Distance to the sea from the share of sea columns in a box 2R wide, exact over the columns' squares: across a
    // straight shore it falls linearly, so its level sets run parallel to the waterline, smoothed over its voxel jogs.
    constexpr float R = 3.0;
    float2 lo = p.xz - R;
    int2 i0 = int2(floor(lo));
    // The same box gives the shares of ground a step up the beach and of ground no longer beach, so the band fades
    // across tread edges rather than stepping with them.
    float3 sum = 0.0;
    float ds = R * 2.0;
    for (int j = 0; j < 7; j++) {
        float wz = saturate(min(float(i0.y + j + 1), p.z + R) - max(float(i0.y + j), lo.y));
        float3 row = 0.0;
        for (int i = 0; i < 7; i++) {
            int2 cc = clamp(i0 + int2(i, j), int2(0), int2(x.tr.size - 1));
            float top = abs(x.tr.hx.read(uint2(cc), 0).r);
            float wx = saturate(min(float(i0.x + i + 1), p.x + R) - max(float(i0.x + i), lo.x));
            row += wx * float3(top < SEA, top > SEA + 0.5, top > SEA + 1.2);
            if (top < SEA) ds = min(ds, length(max(abs(p.xz - float2(i0 + int2(i, j)) - 0.5) - 0.5, 0.0)));
        }
        sum += row * wz;
    }
    sum /= 4.0 * R * R;
    if (sum.x <= 0.0 || sum.z >= 0.5) return 0;
    float m = R * (1.0 - 2.0 * sum.x) + 1.2 * sum.y;
    float2 ww = p.xz + float2(x.tr.origin);
    float t = x.time;
    // Each stretch of shore its own phase and reach, drifting slowly, so the line never runs up as one rule.
    float ph = 0.9 * perlin2(ww / 70.0, x.seed + 171u) + 0.3 * perlin2(ww / 23.0, x.seed + 172u);
    float amp = 1.05 + 0.35 * perlin2(ww / 45.0 + float2(t * 0.01, 0.0), x.seed + 173u);
    float u = fract(t / 8.5 + ph);
    // Runs up over 40% of the period, easing to a stop, then slides back down more slowly.
    float rise = u < 0.4 ? 1.0 - (1.0 - u / 0.4) * (1.0 - u / 0.4) : 1.0 - smoothstep(0.4, 1.0, u);
    float reach = 0.15 + amp * rise;
    float wet = (1.0 - smoothstep(1.0 + amp, 2.4 + amp, m)) * smoothstep(0.0, 0.1, sum.x) * (1.0 - smoothstep(0.0, 0.5, sum.z));
    // The film thins as the wash drains back.
    float film = (1.0 - smoothstep(reach - 0.5, reach, m)) * (u < 0.4 ? 1.0 : mix(0.35, 1.0, rise));
    // The front a fraction of a voxel wide; as it nears a pixel it widens and dims to the same share of light.
    float fw = max(0.07, 0.8 * fp);
    float e = (m - reach) / fw;
    float brk = smoothstep(-0.05, 0.3, perlin2(ww * 0.6 + float2(0.0, t * 0.06), x.seed + 174u));
    float foam = exp(-e * e) * (0.07 / fw) * mix(0.1, 1.0, brk) * (u < 0.4 ? 1.0 : mix(0.3, 1.0, rise));
    // Round an islet or a pool the front's level sets close into rings. Against a straight shore the sea's share of the
    // box is 1/2 - ds/2R at distance ds; a share well off that marks a shore curling round, and the swash fades there.
    float open = 1.0 - smoothstep(0.06, 0.16, abs(sum.x - max(0.5 - ds / (2.0 * R), 0.0)));
    return float3(wet, film * open, foam * open) * fade * (1.0 - dry);
}
