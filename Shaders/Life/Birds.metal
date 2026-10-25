// Bird buffer (Sources/StrataCore/Wildlife/Wildlife+Upload.swift): MAX_FLOCKS pairs (bounding sphere; first cluster, cluster count), then
// MAX_CLUSTERS pairs (bounding sphere; first bird, count), then four float4s per bird: (map-local position, wingspan),
// (unit heading, arm angle up from level), (bank, kind, tint, wing extension), (hand angle, hand sweep, -, -).

constant int MAX_FLOCKS = 8;
constant int MAX_CLUSTERS = 48;
constant int BIRD_STRIDE = 4;

struct BirdHit { float t; float3 n; float3 albedo; };

// Ray against a box with centre c, orthonormal axes a0..a2 and half extents he; returns entry t (1e9 on a miss).
static inline float boxHit(float3 ro, float3 rd, float3 c, float3 a0, float3 a1, float3 a2, float3 he, thread float3& n) {
    float3 p = ro - c;
    float3 lo = float3(dot(p, a0), dot(p, a1), dot(p, a2));
    float3 ld = float3(dot(rd, a0), dot(rd, a1), dot(rd, a2));
    float3 inv = 1.0 / select(ld, copysign(float3(1e-6), ld), abs(ld) < 1e-6);
    float3 t0 = (-he - lo) * inv, t1 = (he - lo) * inv;
    float3 tn = min(t0, t1), tf = max(t0, t1);
    float tN = max(max(tn.x, tn.y), tn.z), tF = min(min(tf.x, tf.y), tf.z);
    if (tN > tF || tF < 0.0) return 1e9;
    float3 sg = -sign(ld);
    n = tn.x >= tN ? a0 * sg.x : (tn.y >= tN ? a1 * sg.y : a2 * sg.z);
    return max(tN, 0.0);
}

// Plumage by kind (0 songbird, 1 raptor, 2 gull, 3 starling, 4 bat, 5 goose) and part (0 body, 1 arm, 2 hand,
// 3 head, 4 tail); `under` for faces turned down, `tip` for the outer hand.
static float3 plumage(int kind, int part, bool under, bool tip) {
    if (kind == 2) {
        if (part == 0 || part >= 3) return float3(0.80, 0.80, 0.78);
        if (tip) return under ? float3(0.12) : float3(0.05);
        return under ? float3(0.74, 0.74, 0.73) : float3(0.40, 0.42, 0.46);
    }
    if (kind == 1) {
        if (tip && part == 2) return float3(0.06, 0.05, 0.04);
        if (part == 3) return float3(0.22, 0.17, 0.11);
        return under ? float3(0.30, 0.25, 0.19) : float3(0.14, 0.10, 0.065);
    }
    if (kind == 5) {
        if (part >= 3) return float3(0.03, 0.03, 0.03);
        if (part == 0) return under ? float3(0.36, 0.33, 0.28) : float3(0.24, 0.21, 0.17);
        return under ? float3(0.30, 0.27, 0.22) : float3(0.17, 0.15, 0.12);
    }
    if (kind == 3) return float3(0.035, 0.035, 0.04);
    if (kind == 4) return part == 1 || part == 2 ? float3(0.05, 0.04, 0.035) : float3(0.03, 0.025, 0.025);
    if (part == 3) return float3(0.07, 0.06, 0.05);
    return under ? float3(0.17, 0.14, 0.11) : float3(0.09, 0.075, 0.06);
}

// A bird is a body and two wings of an arm and a swept hand, plus a head and tail when near. No box is thinner
// than half a cell's footprint, so a far bird stays a speck of a cell or two under the subpixel jitter instead of
// blinking in and out. tOff is path length already travelled (reflections), for the footprint.
static BirdHit traceBirds(device const float4* B, int nFlocks, float3 o, float3 d, float tMax, float pixelAngle, float tOff) {
    BirdHit r;
    r.t = tMax;
    r.n = float3(0, 1, 0);
    r.albedo = float3(0);
    for (int gi = 0; gi < min(nFlocks, MAX_FLOCKS); gi++) {
        float4 g = B[gi * 2];
        float3 og = g.xyz - o;
        float tg = dot(og, d);
        float Rg = g.w + 0.6 * max(tg + tOff, 0.0) * pixelAngle;
        if (tg < -Rg || tg - Rg > r.t || dot(og, og) - tg * tg > Rg * Rg) continue;
        float4 gr = B[gi * 2 + 1];
        int c0 = int(gr.x), c1 = min(int(gr.x) + int(gr.y), MAX_CLUSTERS);
        for (int fi = c0; fi < c1; fi++) {
            float4 s = B[(MAX_FLOCKS + fi) * 2];
            float3 oc = s.xyz - o;
            float tc = dot(oc, d);
            float R = s.w + 0.6 * max(tc + tOff, 0.0) * pixelAngle;
            if (tc < -R || tc - R > r.t || dot(oc, oc) - tc * tc > R * R) continue;
            int2 range = int2(B[(MAX_FLOCKS + fi) * 2 + 1].xy);
            for (int bi = range.x; bi < range.x + range.y; bi++) {
                device const float4* q = B + (MAX_FLOCKS + MAX_CLUSTERS) * 2 + bi * BIRD_STRIDE;
                float4 p0 = q[0];
                float S = p0.w;
                if (S <= 0.0) continue;
                float3 bc = p0.xyz - o;
                float bt = dot(bc, d);
                float m = 0.5 * max(bt + tOff, 0.0) * pixelAngle;
                float Rb = 0.62 * S + 1.8 * m;
                if (bt < -Rb || bt - Rb > r.t || dot(bc, bc) - bt * bt > Rb * Rb) continue;
                float4 p1 = q[1], p2 = q[2], p3 = q[3];
                float3 fw = p1.xyz;
                float3 r0 = normalize(cross(float3(0, 1, 0), fw) + float3(1e-5, 0, 0));
                float3 u0 = cross(fw, r0);
                float3 rr = r0 * cos(p2.x) + u0 * sin(p2.x);
                float3 uu = cross(fw, rr);
                int kind = int(p2.y + 0.5);
                float bodyL = kind == 1 || kind == 5 ? 0.22 : (kind == 4 ? 0.12 : 0.18);
                bool nearBird = S > 10.0 * m;
                float3 n;
                float bestT = r.t;
                int part = -1;
                bool tip = false;
                float3 bn = float3(0);
                float th = boxHit(o, d, p0.xyz, fw, rr, uu, max(float3(bodyL, 0.055, 0.05) * S, m), n);
                if (th < bestT) { bestT = th; part = 0; bn = n; }
                if (nearBird) {
                    float hz = kind == 5 ? bodyL + 0.15 : bodyL + 0.045, hl = kind == 5 ? 0.11 : 0.045;
                    th = boxHit(o, d, p0.xyz + (fw * hz + uu * 0.015) * S, fw, rr, uu, float3(hl, 0.032, 0.032) * S, n);
                    if (th < bestT) { bestT = th; part = 3; bn = n; }
                    if (kind != 4) {
                        th = boxHit(o, d, p0.xyz - fw * (bodyL + 0.06) * S, fw, rr, uu, float3(0.07, 0.055, 0.01) * S, n);
                        if (th < bestT) { bestT = th; part = 4; bn = n; }
                    }
                }
                float L1 = 0.2 * S * p2.w, L2 = 0.28 * S * p2.w;
                float c1 = cos(p1.w), s1 = sin(p1.w), c2 = cos(p3.x), s2 = sin(p3.x);
                float csw = cos(p3.y), ssw = sin(p3.y);
                for (int side = -1; side <= 1; side += 2) {
                    float sd = float(side);
                    float3 d1 = rr * (c1 * sd) + uu * s1;
                    float3 sh = p0.xyz + rr * (0.045 * S * sd) + fw * (0.03 * S);
                    th = boxHit(o, d, sh + d1 * (0.5 * L1), d1, fw, cross(d1, fw), max(float3(0.5 * L1, 0.09 * S, 0.014 * S), m), n);
                    if (th < bestT) { bestT = th; part = 1; bn = n; tip = false; }
                    float3 d2 = (rr * (c2 * sd) + uu * s2) * csw - fw * ssw;
                    float3 n2 = normalize(cross(d2, fw)), ch = cross(n2, d2);
                    float3 hc = sh + d1 * L1 + d2 * (0.5 * L2) + ch * (0.025 * S);
                    th = boxHit(o, d, hc, d2, ch, n2, max(float3(0.5 * L2, 0.062 * S, 0.012 * S), m), n);
                    if (th < bestT) {
                        bestT = th; part = 2; bn = n;
                        tip = dot(o + d * th - hc, d2) > 0.1 * L2;
                    }
                }
                if (part < 0) continue;
                r.t = bestT;
                r.n = bn;
                r.albedo = plumage(kind, part, dot(bn, uu) < -0.3, tip) * (0.85 + 0.3 * p2.z);
            }
        }
    }
    return r;
}

// Birds are lit like small matte objects: key light (with cloud shadow), sky fill from above, ground bounce from below.
static float3 shadeBird(thread const Ctx& x, float3 p, BirdHit b) {
    float3 alb = mix(b.albedo, float3(dot(b.albedo, LUMA)), 0.4 * x.L.night);
    float ndl = max(dot(b.n, x.L.key), 0.0);
    float sh = mix(1.0, cloudShadow(x, p, x.L.key), x.L.sunAmt);
    float deep = x.L.night * smoothstep(0.32, 0.45, -x.L.sun.y);
    float3 sky = mix(x.L.horizon, x.L.zenith, 0.7) * 0.9 * (0.55 + 0.45 * saturate(b.n.y * 0.5 + 0.5));
    sky *= mix(float3(1.0), float3(0.62, 0.58, 0.85), x.L.gold * x.L.sunAmt);
    sky += float3(0.09, 0.12, 0.25) * x.L.night;
    float3 bounce = saturate(-b.n.y) * 0.25 * x.L.keyCol * max(x.L.key.y, 0.0) * float3(0.8, 0.75, 0.6);
    return alb * (x.L.keyCol * ndl * sh * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep)
                  + (sky + bounce) * mix(float3(1.0), float3(0.19, 0.23, 0.38), deep));
}
