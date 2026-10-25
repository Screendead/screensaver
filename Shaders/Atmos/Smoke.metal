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
