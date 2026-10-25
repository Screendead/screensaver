kernel void plantTrees(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       texture2d<uint, access::read_write> gx [[texture(2)]],
                       texture2d<uint, access::write> st [[texture(3)]],
                       constant int4& info [[buffer(0)]],
                       constant uint& row0 [[buffer(1)]],
                       device const GiantRec* giants [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 col = src.read(gid);
    if (col.b > col.g && (gapFlags(gx.read(gid)) & GAP_ISLAND) != 0u) {
        if (col.a != 0.0) st.write(uint4(ST_ISLE | (col.a <= -4.0 ? ST_ISLE_WATER | uint(-4.0 - col.a) : col.a == -2.0 ? ST_ISLE_VINE : uint(col.a))), gid);
        dst.write(float4(col.rgb, 0.0), gid);
        return;
    }
    // Structures pass through; a clearing keeps its mark unless a canopy spreads over it.
    bool cleared = col.a < 0.0;
    col.a = min(col.a, 0.0);
    bool built = col.a < 0.0 && -col.a < col.r;
    if (built || col.b > col.g) { dst.write(col, gid); return; }
    int2 p = int2(gid);
    int2 wc = p + info.xy;
    uint seed = uint(info.w);
    float2 w = float2(wc) + 0.5;
    Giant gi;
    bool isGiant = giantAt(giants, w, 0.0, gi);
    // A crown reaches out over water.
    if (col.r < SEA + 2.0 && !isGiant) { dst.write(col, gid); return; }
    float G = col.r;
    GRuns rr;
    rr.n = 0;
    int2 tg = towerOvergrowth(src, p, wc, G, seed);
    float J = col.r >= SEA + 2.0 ? provinces(w + seedOffset(seed), seed).x : 0.0;
    if (tg.y > tg.x && (J >= 0.15 || (col.r >= SEA + 2.0 && provinces(w + seedOffset(seed), seed).y >= 0.2)))
        gAdd(rr, float(tg.x), float(tg.y), GM_LEAF);
    MapGround mg { src, info.xy };
    if (J >= 0.15) {
        // The roof follows the ground smoothed over a few columns, and keeps off cliffs.
        float2 jw = jungleWindow(wc, mg);
        float gs = jw.y;
        uint hc = hash2i(wc, seed + 326u);
        int2 e0 = int2(floor(w / EMERGENT_CELL));
        bool eTrunk = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                float2 ec;
                float y0, R;
                int pads;
                uint eh;
                if (!emergentIn(e0 + int2(i, j), seed, J, ec, y0, R, pads, eh)) continue;
                if (length_squared(w - ec) > (1.35 * R + 0.75 * R) * (1.35 * R + 0.75 * R)) continue;
                int2 tq = int2(floor(ec)) - info.xy;
                if (any(tq < 5) || any(tq >= int(n) - 5)) continue;
                float4 tc = src.read(uint2(tq));
                // On level ground only: not on a ledge, in water, or among buildings.
                float t0 = src.read(uint2(tq + int2(5, 0))).r, t1 = src.read(uint2(tq - int2(5, 0))).r;
                float t2 = src.read(uint2(tq + int2(0, 5))).r, t3 = src.read(uint2(tq - int2(0, 5))).r;
                float tr = max(max(max(t0, t1), max(t2, t3)), tc.r) - min(min(min(t0, t1), min(t2, t3)), tc.r);
                if (tc.r < SEA + 10.0 || tc.b > tc.g || tc.a < 0.0 || abs(tc.r - gs) > 6.0 || tr >= EMERGENT_STEEP) continue;
                eTrunk = eTrunk || length(w - ec) < (R > 9.0 ? 1.2 : 0.75);
                emergentIntervals(w, ec, tc.r, y0, R, pads, eh, seed, 0, rr);
            }
        // A trunk column is bark through the roof (merged with it, its leaves would start at the roof's underside).
        // Steep ground steps the roof down to bushes on each column's own ground; a cliff carries none.
        float steep = jw.x;
        float keep = jungleKeep(wc, steep, mg, seed);
        if (!cleared && !eTrunk && keep > 0.0) {
            JCan cn = jungleCanopy(w, mix(G, gs, keep), J, seed, keep);
            if (cn.hi > max(cn.lo, G)) {
                gAdd(rr, cn.lo, cn.hi, GM_LEAF);
                // Its crowns stand on poles; lianas hang from it.
                if (cn.trunk) gAdd(rr, G, cn.lo, GM_WOOD);
                else if (hashf(hc) < 0.02) gAdd(rr, G, cn.lo, GM_VINE);
            } else if (hashf(hc) < 0.3 && steep < FERN_STEEP) {
                // Ferns and saplings in the clearings, on gentle ground: strewn over a slope's treads they read as speckle.
                gAdd(rr, G, G + 1.0 + floor(2.5 * hashf(pcg(hc))), GM_LEAF);
            }
        }
    } else if (col.r >= SEA + 2.0 && provinces(w + seedOffset(seed), seed).y >= 0.2) {
        float O = provinces(w + seedOffset(seed), seed).y;
        int2 c0 = int2(floor(w / OLD_CELL));
        bool covered = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                float2 tc;
                float Ht, R, tr;
                uint th;
                if (!oldTreeIn(c0 + int2(i, j), seed, O, tc, Ht, R, tr, th)) continue;
                if (length_squared(w - tc) > 1.3 * R * R) continue;
                int2 tq = int2(floor(tc)) - info.xy;
                if (any(tq < 3) || any(tq >= int(n) - 3)) continue;
                float4 t0 = src.read(uint2(tq));
                // Off cliffs, clear of water, rock and buildings.
                float gt;
                if (t0.r < SEA + 2.0 || t0.b > t0.g || t0.a < 0.0 || !oldSite(int2(floor(tc)), mg, seed, gt)) continue;
                covered = oldTreeIntervals(w, tc, t0.r, Ht, R, tr, th, seed, true, rr) > G || covered;
            }
        // Ferns on the mossy floor.
        uint hc = hash2i(wc, seed + 352u);
        if (!covered && !cleared && hashf(hc) < 0.25) gAdd(rr, G, G + 1.0 + floor(1.8 * hashf(pcg(hc))), GM_LEAF);
    } else if (col.r >= SEA + 2.0) {
        int2 home = int2(floor(float2(wc) / float(TREE_CELL)));
        int lo = 100000, hi = 0;
        bool trunk = false;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                Tree t = treeInCell(src, gx, home + int2(i, j), info.xy, seed);
                if (!t.ok) continue;
                int2 o = p - t.c;
                if (abs(o.x) > 2 || abs(o.y) > 2) continue;
                int2 s = canopySpan(t, o);
                // Shrubs hug the ground they spill onto rather than hovering over a lower step.
                if (t.kind == 0) s += int(col.r) - t.lo;
                if (s.y <= s.x) continue;
                lo = min(lo, s.x);
                hi = max(hi, s.y);
                if (all(o == 0) && t.kind != 0) trunk = true;
            }
        if (tg.y > tg.x) { lo = min(lo, tg.x); hi = max(hi, tg.y); }
        if (!isGiant) {
            if (float(hi) <= G) { dst.write(col, gid); return; }
            float leaf = max(float(lo), G);
            bool floating = !trunk && leaf > G;
            dst.write(float4(float(hi), G, floating ? leaf : G, leaf), gid);
            return;
        }
        if (float(hi) > G) gAdd(rr, trunk ? G : float(lo), float(hi), GM_LEAF);
    }
    GCol gc;
    if (isGiant ? giantColumn(gi, w, G, rr, gc) : runsColumn(rr, G, gc)) writeVeg(gc, G, gid, dst, gx, st, isGiant ? giantPalette(gi.h) : 0u);
    else dst.write(col, gid);
}
