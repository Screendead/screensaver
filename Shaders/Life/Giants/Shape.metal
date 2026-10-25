struct GLimb { float3 p0, p1, p2; float r0, r1; float L; };

static inline GLimb giantLimb(thread const Giant& g, int i) {
    uint hi = pcg(g.h + 1000u + uint(i) * 131u);
    float f = (float(i) + 0.5) / float(GIANT_LIMBS);
    // Kapok (kind 1): limbs only near the top, long and level, a flat crown on a clear column.
    bool kapok = g.kind == 1;
    float y0 = g.yT * mix(kapok ? 0.58 : 0.42, 0.98, f) + 12.0 * (hashf(hi) - 0.5);
    float az = float(i) * 2.39996 + 0.5 * hashf(pcg(hi));
    float L = mix(kapok ? 112.0 : 125.0, kapok ? 80.0 : 62.0, f) * (0.85 + 0.3 * hashf(pcg(hi + 1u)));
    float2 dir = float2(cos(az), sin(az)), perp = float2(-dir.y, dir.x);
    float bend = (hashf(pcg(hi + 2u)) - 0.5) * 0.5;
    GLimb l;
    l.p0 = float3(dir * 8.0, y0).xzy;
    // A weeping giant (kind 2) arches its limbs out and down.
    float rise = kapok ? 0.16 : g.kind == 2 ? 0.30 : 0.34, end = kapok ? 0.10 : g.kind == 2 ? 0.02 : 0.30;
    l.p1 = float3(dir * L * 0.45 + perp * L * bend * 0.4, y0 + L * rise).xzy;
    l.p2 = float3(dir * L + perp * L * bend, y0 + L * end + 6.0 * hashf(pcg(hi + 3u))).xzy;
    l.r0 = mix(10.0, 5.5, f);
    l.r1 = mix(2.6, 1.6, f);
    l.L = L;
    return l;
}

static inline float3 bez(GLimb l, float t) { return mix(mix(l.p0, l.p1, t), mix(l.p1, l.p2, t), t); }

// Pad j: centre (xz relative to the axis, y absolute), radius, thickness. Each limb carries a clump at its tip, a
// second beside it and, on the longer lower limbs, a third part way out; the last few are the crown's dome.
constant int GIANT_PADS = GIANT_LIMBS * 3 + 4;

static inline float4 giantPad(thread const Giant& g, int j, thread float& thick) {
    if (j >= GIANT_LIMBS * 3) {
        int d = j - GIANT_LIMBS * 3;
        uint hd = pcg(g.h + 900u + uint(d));
        float a = float(d) * 2.1 + hashf(hd);
        float2 o = d == 0 ? float2(0) : float2(cos(a), sin(a)) * (26.0 + 14.0 * hashf(pcg(hd)));
        thick = (d == 0 ? 56.0 : 36.0) * (g.kind == 1 ? 0.75 : 1.0);
        return float4(o.x, g.G + g.yT + (d == 0 ? (g.kind == 1 ? 18.0 : 36.0) : 14.0 + 10.0 * hashf(pcg(hd + 1u))), o.y, d == 0 ? 40.0 : 28.0);
    }
    GLimb l = giantLimb(g, j / 3);
    int kind = j % 3;
    uint hp = pcg(g.h + 1500u + uint(j) * 29u);
    float3 p = kind == 1 ? bez(l, 0.62) : l.p2;
    float R = kind == 0 ? 0.27 * l.L + 10.0 : kind == 1 ? 0.22 * l.L + 6.0 : 0.2 * l.L + 7.0;
    thick = (kind == 0 ? 26.0 + 0.14 * l.L : 20.0 + 0.1 * l.L) * (g.kind == 1 ? 0.85 : 1.0);
    if (kind == 1 && l.L < 95.0) R = 0.0;
    if (kind == 2) {
        float2 dir = normalize(l.p2.xz - l.p0.xz);
        float side = hashf(hp) < 0.5 ? -1.0 : 1.0;
        p.xz += float2(-dir.y, dir.x) * side * (0.30 * l.L) - dir * 0.12 * l.L;
        p.y += 4.0 + 10.0 * hashf(pcg(hp));
    }
    return float4(p.x, g.G + p.y + (kind == 1 ? 3.0 : 6.0), p.z, R);
}

// Solid runs of a column at offset q from the axis (ground gl): appended as intervals, then sorted and merged.
// thin: an aerial root added (coarseColumn keeps those as a union, not a mean).
struct GRuns { float lo[24]; float hi[24]; int m[24]; int n; bool thin; };

static inline void gAdd(thread GRuns& r, float lo, float hi, int m) {
    if (hi - lo < 0.5 || r.n >= 24) return;
    r.lo[r.n] = lo; r.hi[r.n] = hi; r.m[r.n] = m; r.n++;
}

static inline float giantTrunkR(thread const Giant& g, float2 q, float yr) {
    float t = saturate(yr / (g.yT + 30.0));
    float R = mix(25.0, 8.0, t * t * (3.0 - 2.0 * t)) + 26.0 * exp(-max(yr, 0.0) / 18.0);
    // Braided like a strangler fig's mass of roots: deep flutes that twist slowly up the trunk; a kapok's column is
    // smooth and straight, its foot carried by the buttresses.
    if (g.kind == 1) R = mix(19.0, 10.0, t) + 10.0 * exp(-max(yr, 0.0) / 12.0);
    float th = atan2(q.y, q.x) + yr / 150.0;
    float f = perlin3(float3(cos(th) * 2.6, sin(th) * 2.6, yr / 34.0), g.h + 5u);
    return R * (1.0 + (g.kind == 1 ? 0.10 : 0.34) * f);
}

// The crown's pads over world column wq: broad, flat-bottomed clouds of leaves with lumpy tops. Returns the crown's
// underside there, the lowest pad bottom (1e9 under none); add: their leaf runs go into r as well.
// Fine terms break the clumps' smooth profiles, which column by column read as ribs.
static float giantPads(thread const Giant& g, float2 wq, thread GRuns& r, bool add) {
    float2 q = wq - g.c;
    float lump = 0.6 * perlin2(wq / 13.0, g.h + 20u) + 0.4 * perlin2(wq / 5.5, g.h + 21u) + 0.35 * perlin2(wq / 2.3, g.h + 25u);
    float sag = perlin2(wq / 7.0, g.h + 22u) + 0.6 * perlin2(wq / 2.7 + 3.1, g.h + 26u);
    float padBot = 1e9;
    for (int j = 0; j < GIANT_PADS; j++) {
        float T;
        float4 P = giantPad(g, j, T);
        if (P.w <= 0.0) continue;
        float2 dq = q - P.xz;
        float dd = length(dq);
        if (dd > P.w * 1.3) continue;
        float2 u2 = dq / max(dd, 1e-3);
        float edge = P.w * (1.0 + 0.32 * perlin2(u2 * 1.8 + float(j) * 5.3, g.h + 23u) + 0.12 * perlin2(u2 * 4.5 + float(j), g.h + 24u));
        if (dd >= edge) continue;
        float u = dd / edge;
        float top = P.y + T * 0.55 * sqrt(1.0 - u * u) + 7.0 * lump;
        float bot = P.y - T * 0.45 * pow(1.0 - u * u, 0.6) + 3.0 * sag;
        if (top > bot + 1.0) {
            if (add) gAdd(r, floor(bot), floor(top) + 1.0, GM_LEAF);
            padBot = min(padBot, floor(bot));
        }
    }
    return padBot;
}

static void giantIntervals(thread const Giant& g, float2 wq, float gl, int lod, thread GRuns& r) {
    float2 q = wq - g.c;
    float d = length(q);
    float G = g.G;
    // Everything that reads the table relies on this bound.
    if (d > GIANT_REACH - 1.0) return;
    // Trunk: marched up from the ground (its radius is not monotone in height once fluted).
    if (d < 80.0) {
        float lo = -1.0;
        float yTop = g.yT + 34.0;
        for (float yr = gl - G; yr < yTop; yr += 1.0) {
            bool in = d < giantTrunkR(g, q, yr + 0.5);
            if (in && lo < 0.0) lo = G + yr;
            if (!in && lo >= 0.0) { gAdd(r, lo, G + yr, GM_WOOD); lo = -1.0; }
        }
        if (lo >= 0.0) gAdd(r, lo, G + yTop, GM_WOOD);
    }
    // Buttress flanges, sinuous plates falling away from the trunk, each running on as a root that arches clear of
    // the ground and dives back in.
    for (int i = 0; i < GIANT_FLANGES; i++) {
        uint hi = pcg(g.h + uint(i) * 7919u);
        float phi = (float(i) + 0.6 * hashf(hi)) * TAU / float(GIANT_FLANGES);
        float2 dir = float2(cos(phi), sin(phi));
        float along = dot(q, dir);
        if (along < 8.0) continue;
        float Lb = 40.0 + 36.0 * hashf(pcg(hi));
        float La = 45.0 + 70.0 * hashf(pcg(hi + 1u));
        if (along > Lb + La) continue;
        float side = dot(q, float2(-dir.y, dir.x)) + 9.0 * perlin2(float2(along / 38.0, float(i) * 3.7), g.h + 11u) * smoothstep(10.0, 50.0, along);
        // A kapok's buttresses are tall plates, flaring far up its column.
        float Hb = (g.kind == 1 ? 55.0 : 30.0) + 42.0 * hashf(pcg(hi + 2u));
        if (along < Lb) {
            float u = saturate((along - 14.0) / (Lb - 14.0));
            float w = mix(7.0, 2.5, u);
            if (abs(side) < 0.5 * w) {
                float top = G + Hb * pow(1.0 - u, 1.7) * (1.0 + 0.12 * perlin2(float2(along / 9.0, float(i)), g.h + 12u)) + 2.0;
                gAdd(r, gl - 2.0, max(top, gl + 1.0), GM_WOOD);
            }
        }
        float s = (along - Lb * 0.75) / (Lb * 0.25 + La);
        if (s > 0.0 && s < 1.0) {
            float rr = mix(2.8, 1.3, s);
            if (abs(side) < rr) {
                float arch = max(0.0, sin(s * 3.14159 * 2.0 + hashf(pcg(hi + 3u)))) * (6.0 + 10.0 * hashf(pcg(hi + 4u))) * (1.0 - s);
                float yc = gl + arch;
                float hh = sqrt(rr * rr - side * side);
                gAdd(r, yc - hh, yc + hh + 0.5, GM_WOOD);
            }
        }
    }
    // Limbs, and props dropped from them to the ground like a banyan's.
    for (int i = 0; i < GIANT_LIMBS; i++) {
        GLimb l = giantLimb(g, i);
        float2 a = l.p0.xz, b = l.p2.xz;
        float2 ab = b - a;
        float tt = saturate(dot(q - a, ab) / dot(ab, ab));
        if (length(q - (a + ab * tt)) > l.L * 0.3 + 10.0) continue;
        float lo = 1e9, hi = -1e9;
        for (int k = 0; k <= 28; k++) {
            float t = float(k) / 28.0;
            float3 p = bez(l, t);
            float rad = mix(l.r0, l.r1, t);
            float dh = length(q - p.xz);
            if (dh < rad) {
                float hh = sqrt(rad * rad - dh * dh);
                lo = min(lo, p.y - hh);
                hi = max(hi, p.y + hh);
            }
        }
        if (hi > lo) gAdd(r, G + lo, G + hi, GM_WOOD);
        // Aerial roots: dropped plumb from the limb, swelling toward the ground where they root. Plumb and never
        // narrower below, each column of one is a single run down to the ground, which the coarse levels keep whole
        // (a leaning root's columns hold short floating runs, which they cannot).
        for (int k = 0; k < 3; k++) {
            uint hp = pcg(g.h + uint(i) * 97u + uint(k) * 13u + 400u);
            if (hashf(hp) > (g.kind == 1 ? 0.0 : g.kind == 2 ? 0.25 : 0.45) || lod > GIANT_THIN_LOD) continue;
            float t = 0.35 + 0.2 * float(k) + 0.1 * hashf(pcg(hp));
            float3 top = bez(l, t);
            float3 foot = float3(top.x, gl - G - 1.0, top.z);
            float rad = 0.8 + 2.8 * pow(hashf(pcg(hp + 1u)), 2.0);
            float dh = length(q - top.xz);
            if (dh > rad * 1.8) continue;
            float lo2 = 1e9, hi2 = -1e9;
            for (int m = 0; m <= 24; m++) {
                float u = float(m) / 24.0;
                float3 pp = float3(top.x, mix(top.y, foot.y, u), top.z);
                float rr = rad * (0.8 + 0.2 * u) * (1.0 + 0.8 * smoothstep(0.75, 1.0, u));
                if (dh < rr) {
                    float hh = max(sqrt(rr * rr - dh * dh), 0.5 * abs(top.y - foot.y) / 24.0 + 0.5);
                    lo2 = min(lo2, pp.y - hh);
                    hi2 = max(hi2, pp.y + hh);
                }
            }
            if (hi2 > lo2) { gAdd(r, G + lo2, G + hi2, GM_WOOD); r.thin = true; }
        }
    }
    // Canopy pads: the tiers of the crown (giantPads). The vines that hang from them are no part of any column: the
    // strand pass draws them (traceStrands) at every distance alike.
    giantPads(g, wq, r, true);
}

// A vegetation column from its solid intervals over ground gl: up to four solid runs (the lowest from the ground), as
// (top, principal gap) plus two extra gaps, the run count n, the top run's first leaf, and per run (the first three) a
// material: leaf, vine or wood by covered height; for the top run, that of its part below the first leaf. False if
// nothing stands over the ground.
struct GCol { float H, gl, gh; float4 e; float leafStart; uint mats; int n; };

static bool runsColumn(thread GRuns& r, float gl, thread GCol& o) {
    if (r.n == 0) return false;
    // Sort by start.
    for (int i = 1; i < r.n; i++) {
        float a = r.lo[i], b = r.hi[i];
        int m = r.m[i];
        int j = i - 1;
        while (j >= 0 && r.lo[j] > a) { r.lo[j + 1] = r.lo[j]; r.hi[j + 1] = r.hi[j]; r.m[j + 1] = r.m[j]; j--; }
        r.lo[j + 1] = a; r.hi[j + 1] = b; r.m[j + 1] = m;
    }
    // Merge into runs, the ground run first; gaps under two voxels close.
    float RL[8], RH[8], leafLo[8], leafH[8], vineH[8];
    int n = 1;
    RL[0] = 0.0; RH[0] = gl; leafLo[0] = 1e9; leafH[0] = 0; vineH[0] = 0;
    for (int i = 0; i < r.n; i++) {
        float a = max(r.lo[i], 0.0), b = r.hi[i];
        if (b <= a) continue;
        if (a <= RH[n - 1] + 2.0) RH[n - 1] = max(RH[n - 1], b);
        else if (n < 8) { RL[n] = a; RH[n] = b; leafLo[n] = 1e9; leafH[n] = 0; vineH[n] = 0; n++; }
        else RH[n - 1] = max(RH[n - 1], b);
        if (r.m[i] == GM_LEAF) { leafLo[n - 1] = min(leafLo[n - 1], a); leafH[n - 1] += b - a; }
        if (r.m[i] == GM_VINE) vineH[n - 1] += b - a;
    }
    if (n == 1 && RH[0] <= gl) return false;
    // At most four runs: close the narrowest gaps.
    while (n > 4) {
        int best = 1;
        float bw = 1e9;
        for (int i = 1; i < n; i++) if (RL[i] - RH[i - 1] < bw) { bw = RL[i] - RH[i - 1]; best = i; }
        RH[best - 1] = RH[best];
        leafLo[best - 1] = min(leafLo[best - 1], leafLo[best]);
        leafH[best - 1] += leafH[best];
        vineH[best - 1] += vineH[best];
        for (int i = best; i < n - 1; i++) { RL[i] = RL[i + 1]; RH[i] = RH[i + 1]; leafLo[i] = leafLo[i + 1]; leafH[i] = leafH[i + 1]; vineH[i] = vineH[i + 1]; }
        n--;
    }
    o.n = n;
    o.H = RH[n - 1];
    o.gl = n >= 2 ? RH[n - 2] : 0.0;
    o.gh = n >= 2 ? RL[n - 1] : 0.0;
    o.e = float4(n >= 3 ? RH[n - 3] : 0.0, n >= 3 ? RL[n - 2] : 0.0, n >= 4 ? RH[n - 4] : 0.0, n >= 4 ? RL[n - 3] : 0.0);
    o.leafStart = leafLo[n - 1] < 1e8 ? max(leafLo[n - 1], RL[n - 1]) : o.H;
    o.mats = 0u;
    for (int i = 0; i < min(n, 3); i++) {
        float lo = max(RL[i], i == 0 ? gl : RL[i]);
        uint m;
        if (i == n - 1) m = vineH[i] > 0.4 * (o.leafStart - lo) ? uint(GM_VINE) : uint(GM_WOOD);
        else m = leafH[i] > 0.4 * (RH[i] - lo) ? uint(GM_LEAF) : vineH[i] > 0.4 * (RH[i] - lo) ? uint(GM_VINE) : uint(GM_WOOD);
        o.mats |= m << (2u * uint(i));
    }
    return true;
}

// The column a giant makes at world column wq over ground gl, with any other intervals already in r (the jungle's).
static bool giantColumn(thread const Giant& g, float2 wq, float gl, thread GRuns& r, thread GCol& o, int lod = 0) {
    giantIntervals(g, wq, gl, lod, r);
    return runsColumn(r, gl, o);
}
