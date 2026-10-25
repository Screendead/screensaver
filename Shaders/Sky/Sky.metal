static inline float hg(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (12.566 * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

static float3 skyColor(float3 d, Light L) {
    float up = saturate(d.y);
    float lowSun = 1.0 - smoothstep(0.0, 0.35, abs(L.sun.y));
    float3 hor = L.skyHor;
    // Low sun: gold toward the sun, a cooler pink-lavender band opposite it.
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(L.sun.xz + 1e-5);
    float toward = 0.5 + 0.5 * dot(dh, sh2);
    hor *= mix(float3(1.0), mix(float3(0.75, 0.6, 0.85), float3(1.0), toward), lowSun);
    // Golden hour: the horizon band burns amber under the sun and cools to dusty rose and slate opposite it.
    hor *= mix(float3(1.0), mix(float3(0.70, 0.70, 1.0), float3(0.92, 0.68, 0.42), toward * toward), L.gold * mix(0.5, 1.0, L.sunAmt));
    float3 c = mix(hor, L.skyZen, 1.0 - exp(-up * mix(mix(5.0, 7.0, saturate(L.sun.y / 0.4)), 8.0, L.gold)));
    if (d.y < 0) c = hor * mix(1.0, 0.75, saturate(-d.y * 4));
    float s = max(dot(d, L.sun), 0.0);
    float3 glowCol = key5(L.sun.y, float3(0), float3(0.45, 0.18, 0.12), float3(1.2, 0.40, 0.12), float3(1.0, 0.65, 0.35), float3(0.5, 0.45, 0.4));
    glowCol = mix(glowCol, glowCol * float3(1.0, 0.78, 0.5), L.gold);
    c += glowCol * (pow(s, 6.0) * mix(0.35, 0.22, L.gold) + pow(s, 64.0) * 0.6) * (1.0 - 0.5 * up);
    // A tighter aureole and a dome that deepens away from the sun, so the sun sits in its own pool of light.
    c *= mix(float3(1.0), float3(0.70, 0.70, 0.95), L.gold * mix(0.5, 1.0, L.sunAmt) * (1.0 - toward) * smoothstep(-0.05, 0.25, d.y));
    c += glowCol * pow(s, 24.0) * 0.5 * L.gold * (1.0 - 0.5 * up);
    return c;
}

// The moon's terminator steps in blocks of a 14-cell grid like the voxels: cell i's lit share. The grid is held inside
// the rim, or a full moon's rim cells would turn half dark.
static inline float moonCellLit(float2 i, float3 s) {
    float2 xq = (i + 0.5) / 7.0;
    xq *= min(1.0, 0.98 / length(xq));
    return smoothstep(-0.05, 0.05, dot(float3(xq, sqrt(1.0 - dot(xq, xq))), s));
}

// The sky c (skyColor) with the sun's and moon's discs, and at night the stars and the Milky Way.
static float3 skyDiscs(float3 c, float3 d, Light L, uint seed, float time, float pixelAngle, float2 px, Planet P, float2 phase) {
    float s = max(dot(d, L.sun), 0.0);
    // Disc takes the key light's hue (amber near the horizon) and blooms into the haze.
    float3 tint = L.keyCol / max(max(L.keyCol.r, max(L.keyCol.g, L.keyCol.b)), 1e-3);
    c += tint * 12.0 * smoothstep(0.99955, 0.9998, s) * L.sunAmt;
    c += L.keyCol * (pow(s, 1200.0) * mix(0.8, 0.35, L.gold) + pow(s, 180.0) * 0.25) * L.sunAmt;

    // The planet: a steady 2x2 of scene pixels (it does not twinkle) in a faint halo, out from the moment the sun sets,
    // dimmed and warmed by the thick air near the horizon. The core is the block round a pixel corner, picked unjittered,
    // so it keeps its shape as the sky clock carries it and steps a whole pixel at a time; a disc at the exact
    // direction covers 2 to 5 pixel centres as it drifts.
    if (P.amt > 0.0) {
        float r = length(d - P.dir) / pixelAngle;
        if (r < 12.0) {
            float low = smoothstep(-0.01, 0.08, d.y);
            float3 col = mix(float3(1.0, 0.62, 0.36), float3(1.0, 0.95, 0.86), low);
            float core = all(abs(px - P.px) < P.reach) ? 8.0 : 0.0;
            c += col * (core + exp(-r * 0.55) * 0.3) * P.amt * low;
        }
    }

    if (L.night > 0.0) {
        float md = max(dot(d, L.moon), 0.0);
        float lit = phase.y, th = phase.x;
        c += float3(0.10, 0.13, 0.22) * (pow(md, 24.0) * 0.6 + pow(md, 400.0) * 1.5) * L.night * lit;
        // Stars: one hashed point per cell, thinned by low-frequency noise into clusters and voids and
        // denser along a band (a galactic plane at a seed-dependent tilt).
        float3 q = d * 260.0;
        int3 cell = int3(floor(q));
        uint h = hash3i(cell, seed + 77u);
        float3 gp = normalize(float3(0.6, 0.5, 0.3 + hashf(pcg(seed + 3u)) * 0.6));
        float bd = dot(d, gp);
        float band = exp(-pow(bd / 0.18, 2.0));
        // Milky Way: a wide glow and a brighter core built from layered noise, a fine grain of unresolved
        // stars, a meandering dark rift and ragged dust clouds.
        float n1 = perlin3(d * 3.0, seed + 80u), n2 = perlin3(d * 9.0, seed + 81u);
        float n3 = perlin3(d * 24.0, seed + 82u), n4 = perlin3(d * 60.0, seed + 85u);
        float wob = bd + 0.05 * n1;
        float core = exp(-pow(wob / 0.08, 2.0));
        float glow = (exp(-pow(wob / 0.22, 2.0)) * 0.45 + core) * saturate(0.7 + 0.5 * n1 + 0.4 * n2 + 0.3 * n3 + 0.2 * n4);
        float rift = exp(-pow((wob - 0.015 - 0.025 * n2 - 0.01 * n3) / 0.028, 2.0)) * smoothstep(-0.3, 0.2, n1 + 0.5 * n2);
        float murk = smoothstep(0.05, 0.35, n2 + 0.6 * n3 + 0.3 * n4) * exp(-pow(wob / 0.14, 2.0));
        float dust = saturate(rift * 0.6 + murk * 0.4);
        float3 mwCol = mix(float3(0.55, 0.62, 0.90), float3(0.95, 0.88, 0.78), saturate(core * (0.5 + 0.8 * n2)));
        float horizonFade = smoothstep(0.0, 0.12, d.y);
        // A thin moon leaves the sky darker, so the Milky Way stands out.
        float mw = glow * (1.0 - dust) * L.night * horizonFade * (1.6 - 0.6 * lit);
        c += mwCol * mw * 0.07;
        if (mw > 0.02) {
            float3 fq = d * 500.0;
            uint fh = hash3i(int3(floor(fq)), seed + 86u);
            float3 fsp = floor(fq) + float3(hashf(pcg(fh)), hashf(pcg(fh + 1)), hashf(pcg(fh + 2)));
            c += mwCol * 0.25 * mw * step(0.75, hashf(pcg(fh + 3))) * smoothstep(0.75, 0.15, length(fq - fsp));
        }
        float clump = saturate(perlin3(d * 3.0, seed + 78u) * 0.9 + 0.5);
        float density = mix(0.004, 0.035, saturate(clump * 0.8 + band * 0.7 + glow * 0.6));
        if (hashf(h) > 1.0 - density) {
            float3 sp = float3(cell) + float3(hashf(pcg(h)), hashf(pcg(h + 1)), hashf(pcg(h + 2)));
            float dd = length(q - sp);
            float b = pow(hashf(pcg(h + 3)), 9.0) * 7.0 + 0.12;
            float twinkle = 0.7 + 0.3 * sin(time * 1.3 + hashf(pcg(h + 4)) * 40.0);
            c += float3(0.9, 0.95, 1.1) * b * twinkle * smoothstep(0.55, 0.0, dd) * L.night * smoothstep(0.0, 0.05, d.y) * (1.0 - 0.8 * dust);
        }
        // Moon: a clean disc whose maria are Perlin noise, its terminator stepped in blocks (moonCellLit). The night
        // side keeps a faint ash-blue earthshine, so a crescent still reads as a whole moon.
        if (md > 0.99) {
            float3 mu = normalize(cross(L.moon, float3(0, 1, 0)));
            float3 mv = cross(mu, L.moon);
            float2 xy = float2(dot(d, mu), dot(d, mv)) / 0.03;
            float disc = 1.0 - smoothstep(0.96, 1.0, length(xy));
            float lum = 0.92 - 0.3 * smoothstep(0.15, 0.55, fbm2(xy * 1.3 + 3.0, 3, 9u) + 0.25);
            // The lit limb faces the sun along the great circle between them. The moon rides at the sun's opposite
            // azimuth, lower than the sun is deep once the sun is 4.6 degrees down (L.moon), so that circle runs
            // straight down through the horizon: horns up, waxing or waning. Before that, as the moon fades in, the
            // circle would run over the zenith; the limb stays down rather than flip. A real moon this far from the
            // sun would be full: the phase is licence, the orientation is not.
            float sunlit = 1.0;
            if (lit < 1.0) {
                float3 toSun = float3(0.0, -abs(sin(th)), -cos(th));
                // Each cell edge ramps over 4% of the radius, as the rim does: a hard edge would resnap whole pixels
                // as the moon drifts across them.
                float2 g = xy * 7.0 - 0.5, i = floor(g);
                float2 w = smoothstep(0.36, 0.64, g - i);
                sunlit = mix(mix(moonCellLit(i, toSun), moonCellLit(i + float2(1, 0), toSun), w.x),
                             mix(moonCellLit(i + float2(0, 1), toSun), moonCellLit(i + float2(1, 1), toSun), w.x), w.y);
            }
            float3 moonCol = mix(float3(0.55, 0.65, 0.9) * 0.144, float3(0.75, 0.80, 0.92) * 2.4, sunlit) * lum;
            c = mix(c, moonCol, L.night * disc);
        }
    }
    return c;
}
