struct StrandHit {
    bool hit;      // a strand's face in front of tMax, with its distance, face and cell (on the world's even grid)
    float t;
    int3 n;
    int2 cell;
    float top;     // that strand's top
    float w;       // 0: the strands themselves; 1: their cover (by pixel footprint)
    float alpha;   // the cover along the ray, to the hit if any
    float tm;      // the cover's mean distance and top
    float topm;
};

// Share of a uniform spread over [a, b] (integers) below y.
static inline float strandRamp(float y, float a, float b) { return saturate((y - a) / max(b - a, 1.0)); }

// The strands along o + d t, t in [0, tMax): the first one hit, where they still span a pixel, and the cover they
// are expected to give, where they do not; both across the hand-over. The ray walks the cells of 32; in one holding
// strands at its heights, the cells of 8 that hold strands in its bands (vineBands) are tested against its run, and
// the strands of those it crosses. A strand is its own voxel column (a box 2 wide), so near strands look as they did
// in the column maps.
static StrandHit traceStrands(texture2d_array<float> fine, texture2d_array<uint> coarse, texture2d_array<uint> sup,
                              texture2d_array<uint> bands, constant VineTable* vt, int2 origin, float3 o, float3 d,
                              float tMax, float pixelAngle) {
    StrandHit r;
    r.hit = false; r.t = tMax; r.n = int3(0); r.cell = int2(0); r.top = 0.0; r.w = 0.0; r.alpha = 0.0; r.tm = 0.0; r.topm = 0.0;
    float3 inv = 1.0 / select(d, copysign(float3(1e-6), d), abs(d) < 1e-6);
    int2 stp = select(int2(-1), int2(1), d.xz > 0.0);
    float2 fwd = select(float2(0.0), float2(1.0), d.xz > 0.0);
    float lam = 0.0, tSum = 0.0, topSum = 0.0, wSum = 0.0;
    int n = min(vt->n.x, VINE_SLOTS);
    for (int gi = 0; gi < n && !r.hit; gi++) {
        VineRec v = vt->r[gi];
        float2 org = float2(v.d.xy - origin);
        uint slice = v.c.z;
        float G = v.a.z;
        // The ray's run through the strands' box and the giant's reach, in front of tMax.
        float3 bLo = float3(v.b.x, v.b.z, v.b.y), bHi = float3(float(v.d.z), v.b.w, float(v.d.w));
        float3 ta = (bLo - o) * inv, tb3 = (bHi - o) * inv;
        float3 tn = min(ta, tb3), tf = max(ta, tb3);
        float t0 = max(max(max(tn.x, tn.y), tn.z), 0.0), t1 = min(min(min(tf.x, tf.y), tf.z), r.t);
        if (t0 >= t1) continue;
        float2 q = o.xz - v.a.xy;
        float a = dot(d.xz, d.xz), bq = dot(q, d.xz), c = dot(q, q) - GIANT_REACH * GIANT_REACH;
        if (a < 1e-8) {
            if (c > 0.0) continue;
        } else {
            float disc = bq * bq - a * c;
            if (disc <= 0.0) continue;
            float sq = sqrt(disc);
            t0 = max(t0, (-bq - sq) / a);
            t1 = min(t1, (-bq + sq) / a);
        }
        if (t0 >= t1) continue;
        // Strands 2 wide: themselves to about a pixel and a half, their cover from under a pixel.
        float w = smoothstep(1.2, 3.0, t0 * pixelAngle);
        // The cover alone is summed over the cells of 32.
        bool coverOnly = w >= 1.0;
        float t = t0;
        float2 q0 = o.xz + d.xz * (t + 1e-3) - org;
        int2 mc = clamp(int2(floor(q0 / 32.0)), int2(0), int2(VINE_SUPER - 1));
        float2 tDelta = abs(32.0 * inv.xz);
        float2 tMaxA = ((float2(mc) + fwd) * 32.0 + org - o.xz) * inv.xz;
        int lastAxis = tn.x >= tn.z ? 0 : 1;
        for (int i = 0; i < 40; i++) {
            float tCell = min(tMaxA.x, tMaxA.y), tEnd = min(tCell, t1);
            float yA = o.y + d.y * t, yB = o.y + d.y * tEnd;
            float yLo = min(yA, yB), yHi = max(yA, yB);
            uint rb = strandBands(yLo, yHi + 1.0, G);
            uint4 cs = sup.read(uint2(mc), slice);
            if ((cs.w & rb) != 0u && yHi >= float(cs.y & 0xffffu) && yLo < float(cs.z >> 16)) {
                if (coverOnly) {
                    // Strands scattered over the cell: a run of l across it meets each with chance (2 l + 4) / area,
                    // and a strand reaches the ray's heights where its bottom lies below them and its top above.
                    float l = (tEnd - t) * length(d.xz);
                    float pb = strandRamp(yHi, float(cs.y & 0xffffu), float(cs.y >> 16));
                    float pt = 1.0 - strandRamp(yLo, float(cs.z & 0xffffu), float(cs.z >> 16));
                    float lm = float(cs.x >> 16) * (2.0 * l + 4.0) / 1024.0 * pb * pt;
                    lam += lm;
                    tSum += lm * 0.5 * (t + tEnd);
                    topSum += lm * 0.5 * float((cs.z & 0xffffu) + (cs.z >> 16));
                    wSum += lm * w;
                    if (lam > 4.6) break;
                } else {
                    // The cells of 8 holding strands in the ray's bands: each lies in two bands' words of its texel.
                    uint occ = 0u;
                    for (uint m = rb; m != 0u; m &= m - 1u) {
                        uint k = ctz(m);
                        uint word = bands.read(uint2(mc.x * 2 + int(k >> 3), mc.y), slice)[(k >> 1) & 3u];
                        occ |= (k & 1u) ? word >> 16 : word & 0xffffu;
                    }
                    float2 sLo32 = org + float2(mc) * 32.0;
                    for (uint m = occ; m != 0u; m &= m - 1u) {
                        uint bit = ctz(m);
                        int2 cc = int2(bit & 3u, bit >> 2);
                        float2 cLo = sLo32 + float2(cc) * 8.0;
                        float2 tA = (cLo - o.xz) * inv.xz, tB = (cLo + 8.0 - o.xz) * inv.xz;
                        float2 tnn = min(tA, tB), tff = max(tA, tB);
                        float tc0 = max(max(tnn.x, tnn.y), t), tc1 = min(min(tff.x, tff.y), tEnd);
                        if (tc0 >= tc1 || tc0 >= r.t) continue;
                        int2 c8 = mc * 4 + cc;
                        uint4 cq = coarse.read(uint2(c8), slice);
                        float yc0 = o.y + d.y * tc0, yc1 = o.y + d.y * tc1;
                        float ycLo = min(yc0, yc1), ycHi = max(yc0, yc1);
                        if (ycHi < float(cq.y & 0xffffu) || ycLo >= float(cq.z >> 16)) continue;
                        // Each strand of the cell against the ray's run through it: the nearest wins.
                        float tHit = r.t;
                        for (uint ms = cq.x & 0xffffu; ms != 0u; ms &= ms - 1u) {
                            uint sb = ctz(ms);
                            int2 sc = int2(sb & 3u, sb >> 2);
                            float2 sLo = cLo + float2(sc) * 2.0;
                            float2 sA = (sLo - o.xz) * inv.xz, sB = (sLo + 2.0 - o.xz) * inv.xz;
                            float2 snn = min(sA, sB), sff = max(sA, sB);
                            float tN = max(max(snn.x, snn.y), t), tF = min(min(sff.x, sff.y), tEnd);
                            if (tN >= tF || tN >= tHit) continue;
                            float2 st = fine.read(uint2(c8 * 4 + sc), slice).rg;
                            float ty0 = (st.x - o.y) * inv.y, ty1 = (st.y - o.y) * inv.y;
                            float tN2 = max(tN, min(ty0, ty1)), tF2 = min(tF, max(ty0, ty1));
                            if (tN2 >= tF2 || tN2 >= tHit) continue;
                            tHit = tN2;
                            r.hit = true;
                            r.t = tN2;
                            // The face: a cap where the ray came down (or up) onto it, else the side it entered by,
                            // or the cell's own for a strand it started in.
                            int side = tN > t ? (snn.x >= snn.y ? 0 : 1) : lastAxis;
                            r.n = tN2 > tN ? int3(0, d.y > 0.0 ? -1 : 1, 0) : side == 0 ? int3(d.x > 0.0 ? -1 : 1, 0, 0) : int3(0, 0, d.z > 0.0 ? -1 : 1);
                            r.cell = (v.d.xy >> 1) + c8 * 4 + sc;
                            r.top = st.y;
                            r.w = w;
                        }
                        if (w > 0.0) {
                            float l = (min(tc1, r.t) - tc0) * length(d.xz);
                            float pb = strandRamp(ycHi, float(cq.y & 0xffffu), float(cq.y >> 16));
                            float pt = 1.0 - strandRamp(ycLo, float(cq.z & 0xffffu), float(cq.z >> 16));
                            float lm = float(cq.x >> 16) * (2.0 * max(l, 0.0) + 4.0) / 64.0 * pb * pt;
                            lam += lm;
                            tSum += lm * 0.5 * (tc0 + tc1);
                            topSum += lm * 0.5 * float((cq.z & 0xffffu) + (cq.z >> 16));
                            wSum += lm * w;
                        }
                    }
                    if (r.hit) break;
                }
            }
            if (tEnd >= t1) break;
            int axis = tMaxA.x < tMaxA.y ? 0 : 1;
            lastAxis = axis;
            t = tCell;
            if (axis == 0) { mc.x += stp.x; tMaxA.x += tDelta.x; } else { mc.y += stp.y; tMaxA.y += tDelta.y; }
            if (any(mc < 0) || any(mc >= VINE_SUPER)) break;
        }
    }
    r.alpha = 1.0 - exp(-lam);
    if (lam > 0.0) {
        r.tm = tSum / lam;
        r.topm = topSum / lam;
        if (!r.hit) r.w = wSum / lam;
    }
    return r;
}

// A strand's shading: shade's terms for a giant's vine, with no column to read. p = o + d t; n its face (the cover: a
// face turned to the eye); top the strand's top, draw its colour draw.
static float3 shadeVine(thread const Ctx& x, thread const Light& L, float3 o, float3 d, float t, float3 n, float top,
                        float draw, bool cover) {
    float3 p = o + d * t;
    float fp = t * x.pixelAngle;
    float3 alb = float3(0.036, 0.050, 0.028) * (0.85 + 0.3 * draw);
    float deep = L.night * smoothstep(0.32, 0.45, -L.sun.y);
    alb = mix(alb, float3(dot(alb, LUMA)), 0.4 * L.night + 0.1 * deep);
    float l = dot(alb, LUMA);
    alb *= 1.0 + L.night * (clamp(0.17 / max(l, 1e-3), 1.0, 2.4) - 1.0);
    float kn = 4.5 * L.night;
    alb *= (1.0 + 0.1 * kn) / (1.0 + kn * dot(alb, LUMA));
    float dusk = L.gold * (1.0 - L.night) * mix(0.5, 1.0, L.sunAmt);
    float3 albK = alb;
    if (dusk > 0.0) {
        float la = max(dot(alb, LUMA), 1e-4);
        alb *= mix(float3(1.0), clamp(1.5 * sqrt(alb / la), 0.7, 2.0), dusk);
        albK = mix(albK, la * float3(1.5, 0.95, 0.55), 0.5 * dusk) * (1.0 + 0.3 * dusk);
        albK = mix(alb, albK, smoothstep(-0.04, 0.0, L.sun.y) * (1.0 - L.sunAmt));
    }
    // As a column's strand shaded (shade): with the footprint its faces turn up to the sky (the risers' prefilter, from
    // a strand's level neighbours) and its shadow lifts to the floor the far stairs keep; the cover is all footprint.
    float wf = cover ? 1.0 : smoothstep(0.025, 0.20, fp);
    float3 nl = normalize(mix(n, float3(0, 1, 0), wf));
    float ndl = max(dot(nl, L.key), 0.0);
    float hiSun = smoothstep(0.15, 0.45, L.key.y) * (1.0 - L.night);
    float sh = 1.0;
    if (ndl > 0.0 && L.shadows && x.shadowMode != 3.0) {
        // Near strands trace the short ray that finds the crown over them; the rest, and the cover, take the maps.
        bool maps = x.shadowMode == 1.0;
        bool mapOnly = !maps || cover || t > 500.0;
        float3 so = p + n * 0.02;
        float mapLit = maps ? shadowMapLit(x, so, mapOnly) : 1.0;
        if (!mapOnly && mapLit > 0.0) {
            // Without the band skip (the strands' pass compiles it out) the ray needs the steps to clear the crown's air.
            Hit sHit = trace(x.tr, so, L.key, SHADOW_NEAR, 64, 0);
            // A column's strand found its own column first, a contact shadow: the light the user saw on them.
            if (sHit.hit || sHit.iters >= 64) sh = 0.35;
        }
        sh = min(sh, mapLit);
        float floorSh = mix(mix(0.5, 0.25, hiSun), 0.32, L.gold);
        sh = mix(sh, floorSh + (1.0 - floorSh) * sh, wf);
        if (L.shade < 1.0) sh = mix(1.0, sh, L.shade);
        sh = mix(sh, 1.0, 0.3 * L.night);
    }
    if (!(x.prof & 32) && L.sunAmt > 0.0) sh *= mix(1.0, cloudShadow(x, p, L.key), L.sunAmt);
    float lowFill = mix(0.9, 1.0, smoothstep(0.1, 0.4, L.key.y));
    float zw = mix(mix(0.55, 0.8, hiSun), 0.85, L.gold);
    float3 skyFill = mix(L.horizon, L.zenith, saturate(zw + 0.15 * nl.y)) * 0.9 * lowFill * mix(1.0, 0.85, hiSun);
    skyFill *= mix(float3(1.0), float3(0.62, 0.58, 0.85), L.gold * L.sunAmt);
    int k = levelAt(x.tr, length(p.xz - x.tr.cam));
    float sv = x.prof & 16 ? 1.0 : skyVisibility(x.tr, min(k, x.tr.levels - 1), p.xz, p.y, fp > 5.0 ? 2 : fp > 2.0 ? 1 : 0);
    skyFill *= 1.0 - mix(0.55, 0.85, hiSun) * (1.0 - sv);
    if (dusk > 0.0) skyFill *= mix(float3(1.0), mix(float3(0.52, 0.58, 0.8), float3(1.12, 1.08, 1.0), saturate(sv * sv * 0.4)), dusk);
    skyFill *= 0.5 + 0.5 * saturate(nl.y);
    // Under the crown: the ambient a column's vine kept by its depth under the crown's top, which stands some 30 over
    // the strand's.
    float ao = mix(1.0, 0.75, smoothstep(6.0, 80.0, top + 30.0 - p.y));
    ao = mix(ao, 1.0, 0.25 * L.night);
    skyFill += float3(0.09, 0.12, 0.25) * L.night;
    float3 bounce = (1.0 - saturate(nl.y)) * 0.35 * alb * L.keyCol * max(L.key.y, 0.0);
    float3 col = albK * L.keyCol * ndl * sh * mix(1.0, 1.1, L.night) * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep)
               + alb * (skyFill * ao + bounce) * mix(float3(1.0), float3(0.19, 0.23, 0.38), deep);
    col += albK * L.keyCol * 0.07 * (1.0 - smoothstep(0.05, 0.35, L.key.y)) * saturate(nl.y) * (1.0 - L.night);
    float back = pow(saturate(dot(d, L.key)), 4.0) * (1.0 - smoothstep(0.08, 0.45, L.key.y)) * (1.0 - L.night);
    float graze = 1.0 - abs(dot(n, d));
    col += alb * float3(0.9, 1.3, 0.3) * L.keyCol * back * (1.0 - ndl) * (1.0 - abs(n.y)) * graze * graze * sh;
    if (dusk > 0.0) {
        float face = saturate(dot(nl, normalize(float3(L.sun.x, 0.25, L.sun.z)))) * (1.0 - 0.6 * saturate(nl.y));
        col += albK * L.skyHor * float3(1.0, 0.85, 0.7) * face * 0.1 * sv * 0.8 * L.gold * (1.0 - L.night) * (1.0 - 0.5 * L.sunAmt);
    }
    // A giant's vines glow softly, more toward their hanging ends; less deep in the jungle, where the skirts mass close.
    float bio = bioLandRamp(L.sun.y);
    if (bio > 0.0) {
        float2 wp = p.xz + float2(x.origin);
        float jw = smoothstep(0.65, 0.9, climateL(x.tr, min(k, x.tr.levels - 1), p.xz).jungle);
        float e = 0.035 * (0.4 + 0.6 * draw) * smoothstep(10.0, 40.0, top - p.y);
        col += bioHue(wp, x.seed) * bioPulse(wp, x.time, x.seed, bioNoise(wp / 150.0, x.seed + 724u)) * (e + alb * 0.15) * bio
             * (1.0 - jw) * bioNear(t);
    }
    return col;
}
