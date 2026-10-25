// A coarse column from its four child samples: their rounded mean, or toward their max where they straddle a sheer
// wall far taller than the column is wide, so walls fatten by half a column instead of each handoff growing a
// half-height ledge. Lowland relief never comes near the threshold.
static inline float coarseCombine(float a, float b, float c, float d, float s) {
    float mean = (a + b + c + d) * 0.25;
    float hi = max(max(a, b), max(c, d)), range = hi - min(min(a, b), min(c, d));
    float lo = max(3.0 * s, 40.0);
    if (range > lo) mean = mix(mean, hi, smoothstep(lo, 2.0 * lo, range));
    return floor(mean + 0.5);
}

// The top of an emergent's crown pads over world column centre w (trunk ground gt), as emergentIntervals builds them;
// -1e9 off the crown.
static float emergentTop(float2 w, float2 ctr, float gt, float y0, float R, int pads, uint h, uint seed) {
    float2 q = w - ctr, p = w + seedOffset(seed);
    float top = -1e9;
    float lumps = 1.6 * perlin2(p / 3.2, seed + 332u) + 1.2 * perlin2(p / 7.0, seed + 333u);
    for (int j = 0; j < pads; j++) {
        float T;
        float4 P = emergentPad(h, j, R, y0, T);
        float2 dq = q - P.xz;
        float dd = length(dq);
        if (dd > P.w * 1.35) continue;
        float2 u2 = dq / max(dd, 1e-3);
        float edge = P.w * (1.0 + 0.30 * perlin2(u2 * 1.7 + float(j) * 4.1 + hashf(h) * 50.0, seed + 331u));
        if (dd >= edge) continue;
        float u = dd / edge;
        top = max(top, floor(gt + P.y + T * 0.6 * sqrt(1.0 - u * u) + lumps * sqrt(1.0 - u * u)) + 1.0);
    }
    return top;
}

// The forest in the coarse rings, so it runs on to the horizon rather than melting into the ground past the 1-voxel
// map: per sample (the coarse column's four, at heights hs4) the top of the leaves over it, the jungle's roof and
// emergents or the planted trees as plantTrees makes them (without the checks that need the map: ledges, carved
// rock), and the column's top the mean of the samples' (leaves, else ground) rounded, as its ground is. Returns
// (top, depth of leaves under it); H is the bare column's height. Grove density and climate are taken once per column.
// ground: heights for the forest's site checks (jungleWindow, oldSite), as plantTrees reads them.
template <typename G>
static float2 coarseVeg(int2 cw, int k, float4 hs4, float H, uint seed, thread const G& ground) {
    int s = 1 << k, off = s >> 2, hs = s >> 1;
    int2 b = cw * s + off;
    float2 centre = float2(cw * s) + 0.5 * float(s);
    float2 so = seedOffset(seed);
    float J = provinces(centre + so, seed).x;
    float sum = 0.0, leaves = 0.0;
    int2 o4[4] = { int2(0), int2(hs, 0), int2(0, hs), int2(hs, hs) };
    if (J >= 0.15) {
        // The roof steps down on steep ground and stands on no cliff, as plantTrees grows it, judged on its own window
        // round the column's centre: samples a few voxels apart on terraced ground often share a tread and read level.
        float steep = jungleWindow(cw * s + s / 2, ground).x;
        float keep = jungleKeep(cw * s + s / 2, steep, ground, seed);
        for (int i = 0; i < 4; i++) {
            float2 wi = float2(b + o4[i]) + 0.5;
            float g = hs4[i], t = g;
            if (g >= SEA + 2.0 && keep > 0.0) {
                JCan cn = jungleCanopy(wi, g, J, seed, keep);
                if (cn.hi > cn.lo) t = max(t, cn.hi);
                int2 e0 = int2(floor(wi / EMERGENT_CELL));
                for (int j = -1; j <= 1; j++)
                    for (int ii = -1; ii <= 1; ii++) {
                        float2 ec;
                        float y0, R;
                        int pads;
                        uint eh;
                        if (steep >= EMERGENT_STEEP || g < SEA + 10.0 || !emergentIn(e0 + int2(ii, j), seed, J, ec, y0, R, pads, eh)) continue;
                        if (length_squared(wi - ec) > 4.4 * R * R) continue;
                        t = max(t, emergentTop(wi, ec, g, y0, R, pads, eh, seed));
                    }
            }
            sum += t;
            leaves += t > g ? 1.0 : 0.0;
        }
    } else if (J < 0.15 && provinces(centre + so, seed).y >= 0.2) {
        float O = provinces(centre + so, seed).y;
        GRuns none;
        none.n = 0;
        float4 t4 = hs4;
        // The trees of the cells round the four samples, each judged once: its site costs five terrain columns.
        int2 cA = int2(floor((float2(b) + 0.5) / OLD_CELL)) - 1, cB = int2(floor((float2(b + hs) + 0.5) / OLD_CELL)) + 1;
        for (int cy = cA.y; cy <= cB.y; cy++)
            for (int cx = cA.x; cx <= cB.x; cx++) {
                float2 tc;
                float Ht, R, tr, gt;
                uint th;
                if (!oldTreeIn(int2(cx, cy), seed, O, tc, Ht, R, tr, th)) continue;
                bool near = false;
                for (int i = 0; i < 4; i++) near = near || (hs4[i] >= SEA + 2.0 && length_squared(float2(b + o4[i]) + 0.5 - tc) <= 1.3 * R * R);
                if (!near || !oldSite(int2(floor(tc)), ground, seed, gt) || gt < SEA + 2.0) continue;
                for (int i = 0; i < 4; i++) {
                    float2 wi = float2(b + o4[i]) + 0.5;
                    if (hs4[i] < SEA + 2.0 || length_squared(wi - tc) > 1.3 * R * R) continue;
                    t4[i] = max(t4[i], oldTreeIntervals(wi, tc, gt, Ht, R, tr, th, seed, false, none));
                }
            }
        for (int i = 0; i < 4; i++) {
            sum += t4[i];
            leaves += t4[i] > hs4[i] ? 1.0 : 0.0;
        }
    } else if (J < 0.15) {
        float dens = groveDensity(centre, H, seed);
        if (dens <= 0.0) return float2(H, 0);
        Climate c = climate(centre, seed);
        c.temp = altitudeTemp(c.temp, H);
        for (int i = 0; i < 4; i++) {
            int2 wi = b + o4[i];
            float g = hs4[i], t = g;
            if (g >= SEA + 2.0) {
                int2 home = int2(floor(float2(wi) / float(TREE_CELL)));
                for (int j = -1; j <= 1; j++)
                    for (int ii = -1; ii <= 1; ii++) {
                        int2 cell = home + int2(ii, j);
                        uint h = hash2i(cell, seed + 92u);
                        int2 tc = cell * TREE_CELL + int2(h % uint(TREE_CELL), (h / 7u) % uint(TREE_CELL));
                        int2 o = wi - tc;
                        if (abs(o.x) > 2 || abs(o.y) > 2 || hashf(pcg(h)) > dens) continue;
                        Tree tr;
                        tr.ok = false; tr.c = tc; tr.g = g; tr.lo = 0; tr.top = 0; tr.kind = 0;
                        tr = treeShape(tr, h, c, dens, float2(tc), seed);
                        if (!tr.ok || tr.kind == 0) continue;
                        int2 sp = canopySpan(tr, o);
                        if (sp.y > sp.x) t = max(t, float(sp.y));
                    }
            }
            sum += t;
            leaves += t > g ? 1.0 : 0.0;
        }
    }
    if (leaves == 0.0) return float2(H, 0);
    float top = max(floor(sum * 0.25 + 0.5), H);
    return float2(top, top - H);
}
