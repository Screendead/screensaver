struct CloudSample { float3 col; float a; float t; };

// Clouds: a volumetric slab above the peaks, marched in a few steps. The same noise field sets the
// coverage; tops tower where it is dense, and each sample is self-shadowed toward the sun, so clouds have
// lit crowns and shaded bases instead of reading as flat cut-outs.
// tEnd clips the march at terrain that rises into the slab, so a peak disappears into the cloud base.
// steps > 0 overrides the march's step count (a veil in front of land needs fewer).
static CloudSample cloudSheet(thread const Ctx& x, float3 o, float3 d, float tEnd = 1e9, int stepsOverride = 0) {
    CloudSample c;
    c.col = 0; c.a = 0; c.t = 6e4;
    if (d.y < 0.01) return c;
    float y0 = CLOUD_Y0, y1 = CLOUD_Y1;
    float ta = (y0 - o.y) / d.y, tb = (y1 - o.y) / d.y;
    float t0 = max(min(ta, tb), 0.0), t1 = min(max(ta, tb), tEnd);
    if (t0 > 7000.0 * CLOUD_S || t1 <= t0) return c;
    // Night steps are dithered, so fewer suffice.
    int steps = stepsOverride > 0 ? stepsOverride : x.quality > 0.5 ? (x.L.night > 0.99 ? 8 : 12) : 8;
    float dt = (t1 - t0) / float(steps);
    float lowSun = 1.0 - smoothstep(0.02, 0.28, abs(x.L.sun.y));
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(x.L.sun.xz + 1e-5);
    float toSun = 0.5 + 0.5 * dot(dh, sh2);
    float2 sunStep = x.L.key.xz / max(x.L.key.y, 0.2) * 14.0 * CLOUD_S;
    float3 ambient = x.L.zenith * 1.0 + x.L.horizon * 0.35;
    float3 moonGlow = float3(0.10, 0.13, 0.22) * pow(saturate(dot(d, x.L.moon)), 16.0) * 0.9 * x.L.moonAmt;
    float glowAmt = lowSun * smoothstep(-0.14, -0.02, x.L.sun.y);
    float T = 1.0;
    float3 acc = 0;
    float silver = 0;
    float jit = 0.5 + x.L.night * (x.dither - 0.5);
    // The slow part varies over hundreds of units: sample it at knots ~40 units apart along the ray (or one
    // per step, if those are further apart) and interpolate; only the fine part is evaluated per step.
    float kt = max(40.0 * CLOUD_S / max(length(d.xz), 1e-4), dt);
    float ka = t0, kb = t0 + kt;
    float2 qa = (o.xz + d.xz * ka) + float2(x.origin), qb = (o.xz + d.xz * kb) + float2(x.origin);
    float sa = cloudSlowBaked(x, qa), sb = cloudSlowBaked(x, qb);
    float wa = -1e9, wb = -1e9;
    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + jit) * dt;
        float3 p = o + d * t;
        float2 q = p.xz + float2(x.origin);
        while (t > kb) {
            ka = kb; qa = qb; sa = sb; wa = wb;
            kb += kt;
            qb = (o.xz + d.xz * kb) + float2(x.origin);
            sb = cloudSlowBaked(x, qb);
            wb = -1e9;
        }
        float kf = saturate((t - ka) / kt);
        // Night skies stay mostly clear: fewer, thinner banks.
        float base = cloudFineBaked(x, q) + mix(sa, sb, kf) - 0.07 * x.L.night;
        // Night: a finer octave frays the edges into wisps for the moon to light through. Its octaves are turned off
        // the grid axes, or thresholded lattice cells read as boxes, and it fades in, or its gate draws an edge.
        if (x.L.night > 0.0 && base > -0.15) {
            float2 qd = (q - float2(x.time * 1.3 * 3.0, 0)) / CLOUD_S;
            float fray = 0.16 * perlin2(OCT_ROT * qd / 90.0, x.seed + 42u) + 0.07 * perlin2(transpose(OCT_ROT) * qd / 33.0, x.seed + 43u)
                       + 0.035 * perlin2(OCT_ROT * OCT_ROT * qd / 13.0, x.seed + 44u);
            base += x.L.night * fray * smoothstep(-0.15, -0.05, base);
        }
        float h01 = (p.y - y0) / (y1 - y0);
        float top = 0.35 + 0.65 * saturate(base * 2.5);
        float dens = saturate((base - 0.04) * mix(3.0, 9.0, x.L.night)) * smoothstep(0.0, 0.2, h01) * (1.0 - smoothstep(top * 0.7, top, h01));
        if (dens < 0.01) continue;
        // Under the moon the sample's own density stands in for the occlusion toward the light.
        float toward = base;
        if (x.L.night <= 0.99) {
            if (wa < -1e8) wa = cloudSlowBaked(x, qa + sunStep);
            if (wb < -1e8) wb = cloudSlowBaked(x, qb + sunStep);
            toward = cloudFineBaked(x, q + sunStep) + mix(wa, wb, kf);
        }
        float shadow = exp(-saturate((toward - 0.04) * 3.0) * 2.2) * (0.55 + 0.45 * h01);
        float3 lit = x.L.keyCol * (0.15 + 0.85 * shadow) * (0.6 + 0.4 * toSun) + ambient * (0.6 + 0.4 * h01);
        // Golden hour: bodies sit in violet shade and only sun-facing crowns take the gold.
        float3 litG = x.L.keyCol * (0.04 + 0.7 * shadow * shadow) * (0.2 + 0.8 * toSun * toSun) + ambient * float3(0.62, 0.55, 0.85) * (0.5 + 0.5 * h01);
        // Forward scatter: cloud in front of a low sun glows instead of hiding it as a grey patch.
        litG += x.L.keyCol * min(hg(dot(d, x.L.sun), 0.75), 1.5) * 0.9 * shadow;
        lit = mix(lit, litG, x.L.gold);
        lit += float3(1.6, 0.55, 0.20) * glowAmt * mix(toSun, toSun * toSun * 1.2, x.L.gold) * (1.0 - h01) * 0.9;
        // Moonlit cloud is a dim blue-grey, only a little above the sky behind it.
        lit = mix(lit, float3(0.022, 0.027, 0.042) * (0.5 + 0.7 * shadow) + x.L.skyHor * 0.8 + moonGlow, x.L.night);
        float a = 1.0 - exp(-dens * dt * mix(0.05, 0.045, x.L.night) / CLOUD_S);
        acc += T * a * lit;
        silver += T * a * shadow;
        T *= 1.0 - a;
        if (c.t > 5e4 && T < 0.6) c.t = t;
        if (T < 0.03) break;
    }
    float cover = 1.0 - T;
    c.col = acc / max(cover, 1e-3);
    // Moonlit night: forward scatter through the thin, moon-facing fringes draws a silver lining around
    // the moon.
    float cm = dot(d, x.L.moon);
    float fringe = silver / max(cover, 1e-3) * (1.0 - cover);
    c.col += float3(0.70, 0.80, 1.0) * x.L.night * min(hg(cm, 0.92) * 0.25, 1.5) * fringe * x.L.moonAmt;
    c.col *= mix(1.0, 0.9, smoothstep(0.3, 0.5, x.L.sun.y));
    c.a = cover * mix(1.0, 0.6, x.L.night) * exp(-t0 / (3000.0 * CLOUD_S)) * smoothstep(7000.0 * CLOUD_S, 2500.0 * CLOUD_S, t0) * smoothstep(0.02, 0.1, d.y);
    return c;
}
