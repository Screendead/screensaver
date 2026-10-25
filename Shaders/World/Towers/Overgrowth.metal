// The towers' own greenery, the hook stage-3 vegetation replaces: the leaf span [x, y) it adds to map column p (world
// wc, top G), empty off the towers: low crowns standing well back on the broad treads, the court and the plinth. A crown
// must have tread under the most of it: one hanging off a narrow ledge or a pier's top reads, from below, as a ball stuck
// to a blank wall.
// (The ivy on the faces is painted by towerSurface: hung as leaves in the column in front of a face, any mat broke up
// into vertical bars on the faces' jogs.)
static int2 towerOvergrowth(texture2d<float, access::read> src, int2 p, int2 wc, float G, uint seed) {
    float2 wf = float2(wc) + 0.5;
    Tower T;
    TowerGroup g;
    if (!towerFind(wf, 0.0, seed, T, g)) return int2(0);
    float2 d0 = wf - T.c;
    float Rg = T.R0 * 1.42 + 6.0;
    if (dot(d0, d0) > Rg * Rg || !towerGroupStands(g, seed)) return int2(0);
    T.Hb = towerBase(T, seed);
    if (T.Hb > TOWER_BASE_MAX) return int2(0);
    Climate cl = climate(T.c, seed);
    float wet = towerWet(cl);
    float snow = treelineAt(snowlineAt(cl.temp));
    int n = int(src.get_width());
    int2 span = int2(0);
    // Crowns on a jittered 9-voxel grid: a site holds one where it stands on a flat tread of the tower or its plinth.
    const int C = 9;
    int2 home = int2(floor(wf / float(C)));
    int2 origin = wc - p;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int2 cell = home + int2(i, j);
            uint h = hash2i(cell, seed + 563u);
            int2 cw = cell * C + int2(h % uint(C), (h / 11u) % uint(C));
            float r = 2.5 + 4.0 * hashf(pcg(h + 1u));
            float2 o = float2(wc - cw);
            if (dot(o, o) > r * r) continue;
            int2 pc = cw - origin;
            if (any(pc < 1) || any(pc >= n - 1)) continue;
            float4 cc = src.read(uint2(pc));
            float Gc = cc.r;
            if (cc.b > cc.g || Gc < SEA + 2.0 || Gc > snow) continue;
            float2 dc = float2(cw) + 0.5 - T.c;
            if (dot(dc, dc) > T.R0 * T.R0 * 1.9) continue;
            bool flat = true;
            for (int k = 0; k < 4; k++) {
                int2 q = pc + int2(k == 0 ? 1 : k == 1 ? -1 : 0, k == 2 ? 1 : k == 3 ? -1 : 0);
                if (abs(src.read(uint2(q)).r - Gc) > 1.0) flat = false;
            }
            // Tread (or the wall behind it) under the crown out to 0.85 of its radius all round.
            for (int k = 0; k < 8 && flat; k++) {
                float a = float(k) * 0.785398;
                int2 q = pc + int2(round(float2(cos(a), sin(a)) * (r * 0.85 + 0.5)));
                if (any(q < 0) || any(q >= n) || src.read(uint2(q)).r < Gc - 1.0) flat = false;
            }
            if (!flat) continue;
            bool ledge = Gc > T.Hb + 8.0;
            if (hashf(pcg(h + 2u)) > (ledge ? 0.1 : 0.15) + 0.45 * wet) continue;
            float trunk = floor(2.0 * hashf(pcg(h + 3u)));
            float hv = 0.6 * r * sqrt(saturate(1.0 - dot(o, o) / (r * r)));
            float yc = Gc + trunk + 0.6 * r;
            int2 s = int2(int(floor(yc - hv)), int(ceil(yc + hv)));
            // The trunk's column reaches the ground.
            if (all(wc == cw)) s.x = int(Gc);
            if (s.y <= s.x) continue;
            if (span.y <= span.x || (s.x <= span.y && s.y >= span.x)) span = span.y > span.x ? int2(min(s.x, span.x), max(s.y, span.y)) : s;
            else if (s.y > span.y) span = s;
        }
    // Scrub and fern in a low, patchy mat over the tower's level treads (not the plinth, which the land's own trees
    // and the crowns hold): a green fringe along the rims rather than a row of beads.
    int2 pp = wc - origin;
    if (G > T.Hb + 8.0 && G < snow && all(pp >= 1) && all(pp < n - 1)) {
        float mn = perlin2(wf / 7.0, seed + 564u) + 0.5 * perlin2(wf / 2.7, seed + 565u) + 0.7 * wet - 0.35;
        if (mn > 0.0) {
            int2 m2 = int2(int(G), int(G) + 1 + int(floor(3.0 * saturate(mn * 1.5))));
            if (span.y <= span.x) span = m2;
            else span = int2(min(span.x, m2.x), max(span.y, m2.y));
        }
    }
    return span;
}

// Runs after cleanupColumns, which would clamp one-column trunks away. Tree columns set w = first leaf
// voxel; g then holds the ground height (b = g on trunks and buried canopies, b = w under a floating canopy).
