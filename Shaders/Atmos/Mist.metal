static inline float landSolid(thread const Ctx& x, float3 p) {
    return LAND_SOLID * (1.0 - smoothstep(x.tr.fadeStart, x.tr.rmax, length(p.xz - x.tr.cam)));
}

// Mist an island's puff (centre c, radii rad, density k per voxel at its core) lays along o + d t up to tEnd: the
// closed-form integral of 1 - r^2 over the chord through the ellipsoid, taken about the ray's closest approach (about
// the camera, at a kilometre, float cubes of t would cancel to noise).
static inline float mistPuff(float3 o, float3 d, float tEnd, float3 c, float3 rad, float k) {
    float3 A = (o - c) / rad, B = d / rad;
    float a = dot(B, B);
    float tm = -dot(A, B) / a;
    float3 q = A + B * tm;
    float e = 1.0 - dot(q, q);
    if (e <= 0.0) return 0.0;
    float hl = sqrt(e / a);
    float t0 = max(tm - hl, 0.0), t1 = min(tm + hl, tEnd);
    if (t1 <= t0) return 0.0;
    float s0 = t0 - tm, s1 = t1 - tm;
    return k * (e * (s1 - s0) - a * (s1 * s1 * s1 - s0 * s0 * s0) / 3.0);
}

// The lumps of a mist bank or plume: noise on its billboard (where the ray crosses the plane through its middle square
// to the ray) at a scale fixed per bank (sz), as a factor on its density. Taken along the chords, or per puff, or at a
// scale that followed the puffs' blend, it would streak and ring where the puffs meet.
static inline float mistLumps(thread const Ctx& x, float3 pm, float sz, float t) {
    float3 mw = pm + float3(float(x.origin.x), 0.0, float(x.origin.y));
    // The finer octave fades before its lumps near a pixel.
    float fpm = max(t, 1.0) * x.pixelAngle;
    float s0 = max(0.35 * sz, 24.0), s1 = max(0.1 * sz, 8.0);
    float lump = perlin3(mw / s0 + float3(x.time * 0.02, x.time * 0.015, x.time * 0.013), x.seed + 742u)
               + 0.4 * perlin3(mw / s1 + float3(0.0, x.time * 0.05, 0.0), x.seed + 743u) * (1.0 - smoothstep(0.15, 0.4, fpm / s1));
    return saturate(0.55 + 0.9 * lump);
}

// Banks of mist round the islands' keels (a ring of flattened puffs on most bigger islands, laid out in the table) and the
// plume a fall spills into: in-scatter in front of whatever the ray reaches by tEnd, drawn as opaque bodies with soft,
// lumpy edges, never a thin veil over the land (which would read as terrain gone translucent). Lit as the cloud sheet
// is, in the islands' shade. (colour, opacity); faded out (by the table) before the islands' pipeline variant turns off.
// Whether o + d t meets the ellipsoid (centre c, radii r) before tEnd.
static inline bool isleMistBound(float3 o, float3 d, float tEnd, float3 c, float3 r) {
    float3 A = (o - c) / r, B = d / r;
    float a = dot(B, B), b = dot(A, B), cc = dot(A, A) - 1.0;
    float disc = b * b - a * cc;
    return disc > 0.0 && (cc < 0.0 || (b < 0.0 && (-b - sqrt(disc)) < tEnd * a));
}

static float4 isleMist(thread const Ctx& x, float3 o, float3 d, float tEnd) {
    int n = min(x.isles->n.x, ISLE_TABLE);
    float3 og = float3(float(x.origin.x), 0.0, float(x.origin.y));
    float tau = 0.0, plume = 0.0, hw = 0.0, dist = 0.0;
    for (int i = 0; i < n; i++) {
        constant IsleShade& s = x.isles->r[i];
        float fade = s.b.w;
        if (fade <= 0.0) continue;
        // A bank lies in a flat disc about its ring (its puffs' heights and depths, Islands.table).
        if (s.mb.w > 0.0 && isleMistBound(o, d, tEnd, s.mb.xyz - og,
                                          float3(s.mb.w, 0.08 * (s.b.y - s.b.z) + 0.28 * (0.44 * s.b.x + 15.0) + 2.0, s.mb.w))) {
            float3 oc = o - (s.mb.xyz - og);
            float bb = dot(oc, d);
            {
                float3 pm = o - d * bb;
                float tr = 0.0, hr = 0.0;
                for (int j = 0; j < ISLE_MIST_PUFFS; j++) {
                    float4 m = s.mp[j];
                    float3 pc = m.xyz - og;
                    float t = mistPuff(o, d, tEnd, pc, float3(m.w, 0.28 * m.w, m.w), 1.5 / m.w);
                    if (t <= 0.0) continue;
                    tr += t;
                    hr += saturate((pm.y - pc.y) / (0.28 * m.w) * 0.5 + 0.5) * t;
                }
                if (tr * fade > ISLE_MIST_TAU0 * 0.8) {
                    float k = mistLumps(x, pm, 0.37 * s.b.x + 15.0, -bb) * fade;
                    tau += tr * k;
                    hw += hr * k;
                    dist += -bb * tr * k;
                }
            }
        }
        if (s.c.w > 0.0) {
            // The plume: spray round the fall's foot and breaking away from it out and downwind in a flatter, smaller
            // puff, its lumps cut deep, so it thins into the air instead of standing on the land.
            float2 u = normalize(s.c.xy - s.a.xy), v = float2(-u.y, u.x);
            float L = s.c.z - s.c.w;
            float rw = 5.0 + 0.012 * s.b.x;
            float xo = isleFallOut(L, s.a.z, length(s.c.xy - s.a.xy));
            float3 foot = float3(s.c.x + u.x * xo, s.c.w, s.c.y + u.y * xo) - og;
            float3 pcen = foot + float3(u.x * 0.6 * rw, 0.06 * L, u.y * 0.6 * rw);
            float3 oc = o - pcen;
            float bb = dot(oc, d);
            if (isleMistBound(o, d, tEnd, pcen, float3(2.6 * rw + 2.0, 0.15 * L + rw + 2.0, 2.6 * rw + 2.0))) {
                float side = isleFallSide(s.c.xy);
                float tp = mistPuff(o, d, tEnd, foot + float3(0.4 * rw * u.x, 0.07 * L, 0.4 * rw * u.y), float3(1.5 * rw, max(0.12 * L, rw), 1.5 * rw), 0.9 / rw)
                         + mistPuff(o, d, tEnd, foot + float3((1.6 * u.x + side * v.x) * rw, -0.02 * L, (1.6 * u.y + side * v.y) * rw),
                                    float3(1.1 * rw, 0.07 * L, 1.1 * rw), 1.2 / rw);
                if (tp * fade > ISLE_MIST_TAU0 * 0.8) {
                    float k = saturate(mistLumps(x, o - d * bb, 1.45 * rw, -bb) * 1.6 - 0.45) * fade;
                    tau += tp * k;
                    plume += tp * k;
                    hw += 0.85 * tp * k;
                    dist += -bb * tp * k;
                }
            }
        }
    }
    // Opaque where it is drawn: partial cover lives only in a thin shell at a body's edge.
    float a = smoothstep(ISLE_MIST_TAU0, ISLE_MIST_TAU0 + 0.2, tau);
    if (a <= 0.0) return 0.0;
    float h01 = hw / tau;
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(x.L.sun.xz + 1e-5);
    float toSun = 0.5 + 0.5 * dot(dh, sh2);
    float3 ambient = x.L.zenith * 1.0 + x.L.horizon * 0.35;
    // In the islands' shade: lit mostly from the side and below, by the sky and the bright land.
    float shadow = 0.2 + 0.4 * saturate(h01);
    float3 lit = x.L.keyCol * shadow * (0.6 + 0.4 * toSun) * max(x.L.key.y, 0.0) * 0.9 + ambient * (0.5 + 0.3 * h01);
    float3 litG = x.L.keyCol * (0.04 + 0.7 * shadow * shadow) * (0.2 + 0.8 * toSun * toSun) + ambient * float3(0.62, 0.55, 0.85) * (0.5 + 0.5 * h01);
    litG += x.L.keyCol * min(hg(dot(d, x.L.sun), 0.75), 1.5) * 0.5 * shadow;
    lit = mix(lit, litG, x.L.gold);
    lit = mix(lit, float3(0.020, 0.025, 0.040) * (0.5 + 0.7 * shadow) + x.L.skyHor * 0.7, x.L.night);
    // At night the spray round a fall's foot carries a faint turquoise glow (the shared palette's #2fe3d0).
    lit += BIO_LAND[1] * 0.035 * x.L.night * saturate(plume / tau);
    lit = applyFogT(x, lit, o, d, fogTransmittance(x, o, d, max(dist / tau, 0.0), 0));
    return float4(lit, a);
}
