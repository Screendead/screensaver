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
