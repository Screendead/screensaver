// Sedimentary bedding by world height: broad formations, beds of uneven tone and fine laminae. Beds dip gently
// with a slow horizontal warp, so a layer runs continuously across neighbouring columns and around canyon walls.
// Bed edges widen and fine terms fade with the pixel footprint fp (voxels per pixel).
// The bands' phase also wanders over ~55 voxels (tilting beds ~10 degrees typically, ~14 at the 90th percentile), so
// rock bodies more than ~55 voxels apart carry different bands: with one phase for all, the bands on a near tower meet
// those on the rock behind at the same screen rows near eye level and it reads as see-through. Bodies closer than that
// still share bands; parting them needs a steeper tilt. Pure world space: a phase that moved with fp would slide the
// bands as the camera approaches.
// `wall` (0..1) marks the great walls of mega relief, whose rock reads as formations from foot to rim.
// Formation tones (cap rock, wall rock, shale, buff) by index. Selects, not a local array: indexing one per pixel puts
// it in scratch memory, which costs more than all the noise around it.
static inline float3 formationWarm(int i) {
    return i == 0 ? float3(0.62, 0.56, 0.47) : i == 1 ? float3(0.55, 0.31, 0.21) : i == 2 ? float3(0.34, 0.31, 0.30) : float3(0.60, 0.47, 0.33);
}
static inline float3 formationCool(int i) {
    return i == 0 ? float3(0.48, 0.47, 0.44) : i == 1 ? float3(0.27, 0.28, 0.31) : i == 2 ? float3(0.42, 0.36, 0.29) : float3(0.20, 0.21, 0.24);
}

// lite (reflections): rippled water blurs a 30-voxel dip wobble of a fraction of a voxel away.
static inline float3 strataTone(float y, float2 w, uint seed, float desertAmt, float fp, float wall = 0.0, bool lite = false) {
    float yd = y + 1.2 * perlin2(w / 140.0, seed + 24u) + (lite ? 0.0 : 0.3 * perlin2(w / 30.0, seed + 25u));
    // Beds of uneven thickness (the phase runs faster and slower up the face) with soft contacts: an even, hard-edged
    // 16-voxel zebra reads as posterised banding on a tall face.
    float k = 0.5 + 0.5 * sin(yd * 0.38 + 5.0 * perlin2(float2(yd / 41.0, 0.27), seed + 39u)
                             + 2.2 * perlin2(w * 0.01, seed + 22u) + 1.3 * perlin2(w * 0.003, seed + 23u)
                             + 3.0 * perlin2(w / 55.0, seed + 45u));
    k = smoothstep(0.1, 0.9, k);
    // On the great walls the regular 16-voxel bands would read as a zebra stripe; the formations carry the tone there.
    float fz = max(smoothstep(120.0, 300.0, y), wall);
    k = mix(k, 0.5, 0.9 * fz);
    float3 grey = mix(float3(0.22, 0.23, 0.25), float3(0.35, 0.34, 0.33), k);
    float3 sand = mix(float3(0.56, 0.36, 0.24), float3(0.70, 0.52, 0.37), k);
    float3 base = mix(grey, sand, desertAmt);
    // Formations: on the great walls, broad units ~200 tall of pale cap rock, red or grey wall rock, shale and buff, so
    // an 800-voxel face reads as geology rather than one repeated stripe. Absent from the lowland.
    if (fz > 0.0) {
        float f = saturate(perlin2(float2(yd / 380.0 + 0.6 * perlin2(w / 2600.0, seed + 29u), 0.13), seed + 30u) * 0.8 + 0.5) * 3.0;
        float fi = min(floor(f), 2.0), ff = smoothstep(0.35, 0.65, f - fi);
        float wm = saturate(desertAmt + 0.5);
        float3 fa = mix(formationCool(int(fi)), formationWarm(int(fi)), wm), fb = mix(formationCool(int(fi) + 1), formationWarm(int(fi) + 1), wm);
        base = mix(base, mix(fa, fb, ff), 0.55 * fz);
        // Weathering: stains and lichen in broad drifts, so a great face never reads as one flat grey plane. Taken in
        // the face's own plane (across it and up it), never by ground position alone, which would streak a wall
        // from foot to rim.
        float along = w.x + w.y;
        float st = perlin2(float2(along / 80.0, yd / 45.0), seed + 34u)
                 + (fp < 4.0 ? 0.5 * perlin2(float2(along / 26.0, yd / 17.0), seed + 35u) * (1.0 - smoothstep(1.0, 4.0, fp)) : 0.0);
        base *= 1.0 + 0.22 * fz * st;
    }
    // Past here every term has faded to its mean.
    if (fp >= 1.8) return base;

    float yb = yd / 2.4;
    float bi = floor(yb), bf = yb - bi;
    float e = clamp(fp * 0.8, 0.04, 0.45);
    uint sb = seed + 26u;
    float t0 = hashf(hash2i(int2(int(bi), 0), sb)), t1 = hashf(hash2i(int2(int(bi) + 1, 0), sb));
    float bed = mix(t0, t1, smoothstep(1.0 - e, 1.0, bf)) - 0.5;
    float bedFade = (1.0 - smoothstep(0.6, 1.8, fp)) * (1.0 - 0.5 * wall);
    float lam = fp < 0.05 ? sin(yd * 21.0 + 3.0 * perlin2(w / 11.0, seed + 27u)) * (1.0 - smoothstep(0.015, 0.05, fp)) : 0.0;
    float parting = (1.0 - smoothstep(0.0, 0.03, min(bf, 1.0 - bf))) * (1.0 - smoothstep(0.02, 0.06, fp));
    // Desert beds swing between rust and pale cream; grey rock between warm umber and cool slate.
    float3 lo = mix(float3(1.04, 0.99, 0.94), float3(1.04, 0.88, 0.70), desertAmt);
    float3 hi = mix(float3(0.95, 0.98, 1.04), float3(1.10, 1.07, 1.00), desertAmt);
    float3 tint = mix(lo, hi, saturate(0.5 + bed * 1.8));
    float lum = 1.0 + bedFade * 0.22 * bed + 0.05 * lam - 0.10 * parting;
    return base * mix(float3(1.0), tint, bedFade) * lum;
}

// Painted-hill beds: 1D Perlin noise in a gently dipping height picks each voxel's place on a ramp of
// charcoal, rust, ochre, cream and lavender-grey; bed edges soften with the pixel footprint fp.
static inline float2 paintedLateral(float2 w, uint seed) {
    return float2(14.0 * perlin2(w / 520.0, seed + 114u) + 1.5 * perlin2(w / 90.0, seed + 115u), 0.35 * perlin2(w / 900.0, seed + 118u));
}

static inline float3 paintedPal(int i) {
    return i == 0 ? float3(0.21, 0.19, 0.18) : i == 1 ? float3(0.43, 0.24, 0.17) : i == 2 ? float3(0.53, 0.41, 0.26)
         : i == 3 ? float3(0.53, 0.47, 0.39) : float3(0.45, 0.40, 0.40);
}

// `lat` is the column's (dip offset, lateral facies shift) from paintedLateral.
static inline float3 paintedTone(float y, float2 lat, uint seed, float fp) {
    float yd = y + lat.x;
    float v = perlin2(float2(yd / 11.0, 0.37), seed + 116u) + (fp < 1.8 ? 0.3 * perlin2(float2(yd / 4.0, 0.71), seed + 117u) * (1.0 - smoothstep(0.6, 1.8, fp)) : 0.0) + lat.y;
    float t = saturate(v * 0.75 + 0.5) * 4.0;
    float i = min(floor(t), 3.0), f = t - i;
    float e = clamp(fp * 0.5, 0.12, 0.5);
    return mix(paintedPal(int(i)), paintedPal(int(i) + 1), smoothstep(0.5 - e, 0.5 + e, f));
}

// Wind ripples on sand. Crests are level sets of a Perlin-warped wind field, so they bend and pinch across a dune,
// and creep slowly downwind. Returns (albedo offset, slope along the wind) with amplitude already applied.
static inline float3 sandRipples(int2 v, float2 uv, float2 w, float fp, float time, uint seed) {
    if (fp >= 2.0) return float3(0);
    const float2 wind = float2(0.6, 0.8);
    const float lambda = 0.8;
    // Integer part of the phase in exact arithmetic: float world coords lose the ripple far from the origin.
    int ip = (3 * v.x + 4 * v.y) % 20;
    float2 q = w + uv;
    float ipm = float(ip + (ip < 0 ? 20 : 0));
    float psi = dot(uv, wind) + 7.0 * perlin2(q / 26.0, seed + 60u) + 1.6 * perlin2(q / 7.0, seed + 61u);
    float phase = ipm / (5.0 * lambda) + psi / lambda - time * 0.02;
    // Low dune swells, five ripples apart, carry the pattern into the middle distance.
    float swell = sin((ipm / 20.0 + psi / 4.0 - time * 0.004) * TAU) * (1.0 - smoothstep(0.4, 1.4, fp));
    float x = phase * TAU;
    // Long gentle stoss, short steep lee.
    float r = sin(x) + 0.35 * sin(2.0 * x - 0.9);
    float dr = (cos(x) + 0.7 * cos(2.0 * x - 0.9)) * TAU / lambda;
    float calm = fp < 0.24 ? smoothstep(-0.35, 0.35, perlin2(q / 48.0, seed + 62u)) : 0.0;
    float amp = mix(0.25, 1.0, calm) * (1.0 - smoothstep(0.08, 0.24, fp));
    // Streaks drawn out along the wind give broad plains a calm grain at any distance.
    float along = dot(q, wind), across = dot(q, float2(-wind.y, wind.x));
    float streak = perlin2(float2(along / 70.0, across / 7.0), seed + 63u) * (1.0 - smoothstep(0.5, 2.0, fp));
    return float3(0.045 * amp * r + 0.035 * swell + 0.05 * streak, 0.018 * amp * dr * wind);
}

// 0..1 where world xz w lies among mega relief, from the coarsest ring's max over ~128-voxel cells (lowland rarely rises
// past ~190). Interpolated between cell corners, each the max of the four cells meeting there, so every column rates at
// least its own cell's relief and the field stays continuous: interpolated between cell centres, a wall's foot columns
// past a cell edge rated partly as lowland, and a jogging face crossfaded column by column between the formations and the
// lowland's 16-voxel beds, which near eye level line up with the beds on every rock behind and read as see-through rock.
// Wider cells would carry the walls' rock a whole cell out over the lowland at their feet.
static inline float megaWall(thread const Terrain& tr, float2 w) {
    int kc = tr.levels - 1;
    if (kc < 1) return 0.0;
    int m = max(7 - kc, 0), mask = (COARSE_N >> m) - 1;
    float2 q = w / float(1 << (kc + m));
    int2 c = int2(floor(q));
    float2 f = q - float2(c);
    uint sl = uint(kc - 1), mp = uint(m);
    #define MW(i, j) tr.ch.read(uint2((c + int2(i, j)) & mask), sl, mp).r
    float a0 = MW(-1, -1), a1 = MW(0, -1), a2 = MW(1, -1);
    float b0 = MW(-1, 0), b1 = MW(0, 0), b2 = MW(1, 0);
    float c0 = MW(-1, 1), c1 = MW(0, 1), c2 = MW(1, 1);
    #undef MW
    float v00 = max(max(a0, a1), max(b0, b1)), v10 = max(max(a1, a2), max(b1, b2));
    float v01 = max(max(b0, b1), max(c0, c1)), v11 = max(max(b1, b2), max(c1, c2));
    return smoothstep(190.0, 270.0, mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y));
}
