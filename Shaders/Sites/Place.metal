// The lie of the land at a candidate site centre, and how well each type suits it.
struct Lie {
    float g, desert, painted, lagoon, salt, cold, snowline, waterR, prom, autumn, conifer, moist, grove;
    int2 waterDir;
    bool flat5;
    float score[7];
};

static Lie survey(texture2d<float, access::read> hm, int2 c, int2 cw, uint seed) {
    Lie L;
    for (int t = 0; t < 7; t++) L.score[t] = 0;
    L.grove = 0;
    L.g = colH(hm, c);
    float g = L.g;
    float2 w = float2(cw);
    float2 p = w + seedOffset(seed);
    Climate cl = climate(w, seed);
    L.desert = desertAmount(cl); L.painted = paintedAmount(cl); L.lagoon = lagoonAmount(cl); L.salt = saltAmount(cl);
    L.cold = smoothstep(0.42, 0.18, cl.temp);
    L.autumn = autumnAmount(cl); L.conifer = coniferAmount(cl); L.moist = cl.moist;
    L.snowline = snowlineAt(cl.temp);
    float settled = fbm2(p / 1500.0, 2, seed + 140u) + 0.5;
    float ancient = fbm2(p / 1800.0 + 77.0, 2, seed + 141u) + 0.5;
    float lo5, hi5, lo9, hi9;
    L.flat5 = flatPatch(hm, c, 2, 1.0, lo5, hi5);
    bool flat9 = flatPatch(hm, c, 4, 3.0, lo9, hi9);
    // Ring scan: nearest water and its bearing, prominence over the surroundings, how much of a bay this is.
    L.waterR = 1e9;
    L.waterDir = int2(0);
    float ring = 0, ringN = 0, wet16 = 0, max6 = -1e9;
    const float rads[4] = { 8.0, 16.0, 24.0, 40.0 };
    for (int ri = 0; ri < 4; ri++)
        for (int k = 0; k < 16; k++) {
            float a = float(k) * 0.3926991;
            float2 dv = float2(cos(a), sin(a));
            float hq = colH(hm, c + int2(round(dv * rads[ri])));
            if (hq < SEA && rads[ri] < L.waterR) {
                L.waterR = rads[ri];
                L.waterDir = abs(dv.x) > abs(dv.y) ? int2(dv.x > 0 ? 1 : -1, 0) : int2(0, dv.y > 0 ? 1 : -1);
            }
            if (ri >= 2) { ring += hq; ringN += 1; }
            if (ri == 1 && hq < SEA) wet16 += 1.0 / 16.0;
        }
    float min6 = 1e9;
    for (int k = 0; k < 8; k++) {
        float a = float(k) * 0.785398;
        float h6 = colH(hm, c + int2(round(float2(cos(a), sin(a)) * 6.0)));
        max6 = max(max6, h6);
        min6 = min(min6, h6);
    }
    // Nothing is built within 6 of a sheer drop of more than 40 (a pillar top's or an escarpment's edge).
    if (g - min6 > 40.0) return L;
    L.prom = g - ring / ringN;
    // A tread with a riser close by: a terrace.
    float riser = 0;
    for (int k = 0; k < 8; k++) {
        float a = float(k) * 0.785398;
        for (int r = 3; r <= 9; r += 3) riser = max(riser, colH(hm, c + int2(round(float2(cos(a), sin(a)) * float(r)))) - g);
    }
    float land = step(SEA + 2.0, g);
    float notDry = (1.0 - smoothstep(0.3, 0.5, L.desert)) * (1.0 - smoothstep(0.2, 0.4, L.salt)) * (1.0 - smoothstep(0.3, 0.5, L.lagoon));
    float anc = smoothstep(0.35, 0.6, ancient), set = smoothstep(0.35, 0.6, settled);
    float prom = L.prom, salt = L.salt;
    L.score[T_HENGE] = anc * smoothstep(1.0, 5.0, prom) * smoothstep(45.0, 30.0, prom) * float(flat9) * notDry
                     * step(g, L.snowline + 10.0) * land;
    L.score[T_CAIRN] = anc * step(max6, g) * max(smoothstep(3.0, 9.0, prom), smoothstep(0.3, 0.6, salt)) * land * 0.85;
    L.score[T_RUINS] = anc * float(L.flat5) * land * max(smoothstep(3.0, 6.0, riser) * (1.0 - smoothstep(0.3, 0.5, salt)),
                                                         smoothstep(0.35, 0.6, L.desert) * smoothstep(3.0, 8.0, prom));
    L.score[T_HAMLET] = set * step(L.waterR, 40.0) * float(L.flat5) * step(SEA + 2.0, g) * step(g, SEA + 30.0)
                      * (1.0 - smoothstep(0.2, 0.4, salt)) * step(g, L.snowline - 10.0);
    L.score[T_JETTY] = set * step(L.waterR, 16.0) * smoothstep(0.1, 0.25, wet16) * smoothstep(0.75, 0.55, wet16)
                     * step(g, SEA + 6.0) * (1.0 - smoothstep(0.2, 0.4, salt)) * 0.9;
    float grove = groveDensity(w, g, seed);
    L.grove = grove;
    L.score[T_CAMP] = (0.4 + 0.6 * set) * float(L.flat5) * land * step(g, L.snowline - 5.0)
                    * max(max(smoothstep(0.08, 0.2, grove), smoothstep(0.15, 0.3, salt) * smoothstep(0.7, 0.45, salt)),
                          smoothstep(0.3, 0.6, L.painted)) * 0.8;
    return L;
}

// Open dry grass: no trees, not desert, painted hills or salt.
static inline bool steppe(thread const Lie& L) {
    return L.moist < 0.45 && L.grove < 0.1 && L.desert < 0.2 && L.painted < 0.2 && L.salt < 0.15 && L.lagoon < 0.2;
}

static inline int hamletStyle(thread const Lie& L) {
    if (L.lagoon > 0.4) return ST_STILT;
    if (L.desert > 0.3) return ST_ADOBE;
    if (L.painted > 0.35) return ST_PUEBLO;
    if (L.cold > 0.5 || L.g > L.snowline - 30.0) return L.grove > 0.15 ? ST_CABIN : ST_CROFT;
    if (L.conifer > 0.5) return ST_CABIN;
    if (L.autumn > 0.3) return ST_TIMBER;
    if (steppe(L)) return ST_YURT;
    return ST_MEADOW;
}

static inline int campStyle(thread const Lie& L) {
    if (L.lagoon > 0.3 && L.autumn < 0.3) return ST_PALM;
    if (L.painted > 0.35) return ST_TIPI;
    if ((L.salt > 0.15 && L.salt < 0.7) || steppe(L) || L.desert > 0.3) return ST_CARAVAN;
    if (L.conifer > 0.5) return ST_LEANTO;
    return ST_AFRAME;
}

// Site at one world cell: a type chosen by two broad Perlin fields (where people live now, and where an older
// culture built) and by the lie of the land at the best of nine candidate centres, then its parts.
kernel void placeSites(texture2d<float, access::read> hm [[texture(0)]],
                       texture2d<uint, access::read> gx [[texture(1)]],
                       device Site* sites [[buffer(0)]],
                       constant int4& info [[buffer(1)]],
                       device const GiantRec* giants [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(SITE_GRID) || gid.y >= uint(SITE_GRID)) return;
    int n = info.z;
    uint seed = uint(info.w);
    int2 cell = siteCell0(info.xy) + int2(gid);
    uint h = hash2i(cell, seed + 200u);
    SiteGen G;
    G.np = 0; G.nl = 0; G.r = 0;
    G.s.head = int4(cell * SITE_CELL + 64, 0, int(h));
    G.s.info = float4(0);
    for (int i = 0; i < MAX_LAMPS; i++) G.s.lamps[i] = float4(0);
    for (int i = 0; i < MAX_PARTS; i++) G.s.parts[i] = int4(0);
    uint idx = gid.y * uint(SITE_GRID) + gid.x;

    const float thresh[7] = { 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 };
    // Chance a type is built where it is the best fit; these set the density of each type.
    const float accept[7] = { 0, 0.25, 0.7, 0.55, 0.22, 0.14, 0.16 };
    float best[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int2 at[7];
    // Only cells whose every candidate, survey and part (64 out) reads lie inside the map, so a cell decides the
    // same way on every map that holds it.
    int2 cl = cell * SITE_CELL - info.xy;
    if (any(cl < 48) || any(cl > n - 176)) { sites[idx] = G.s; return; }
    if (towerNear(float2(cell * SITE_CELL + 64), 128.0, seed)) { sites[idx] = G.s; return; }
    // A lived-in cave whose hearth lies in this cell: its fire is the site's one lamp (its light, glow and smoke), its
    // furnishing carved in the cave's floor (caveColumn). Smoke leaves by the mouth, over the lip. No other site here.
    {
        CaveRec r;
        float2 cc = float2(cell * SITE_CELL) + 0.5 * float(SITE_CELL);
        if (caveFind(caveTable(giants), cc, r) && (r.flags & CV_HOME) != 0u) {
            float af = caveHearthA(r);
            float2 F = floor(r.m + r.u * af) + 0.5;
            if (all(int2(floor(F / float(SITE_CELL))) == cell)) {
                G.s.head = int4(int2(floor(F)), T_CAVE, int(h));
                int2 so = int2(round(-r.u * (af + 1.5)));
                addLamp(G, float3(0.5, r.yF + 1.5, 0.5), L_HEARTH, max(r.Hc - 1.0, 1.0), so);
                G.s.info = float4(min(G.r, 125.0), 0.1, 0.0, float(G.nl));
                sites[idx] = G.s;
                return;
            }
        }
    }
    for (int k = 0; k < 9; k++) {
        uint hk = pcg(h + uint(k) * 7919u);
        int2 ck = cell * SITE_CELL + 16 + int2(k % 3, k / 3) * 32 + int2(int(hk % 32u), int((hk >> 5) % 32u));
        int2 lc = ck - info.xy;
        if (colArch(hm, lc)) continue;
        // Giants are placed first: a site keeps clear of a giant's reach by more than its own.
        Giant gk;
        if (giantAt(giants, float2(ck) + 0.5, 60.0, gk)) continue;
        Lie L = survey(hm, lc, ck, seed);
        for (int t = 1; t <= 6; t++)
            if (L.score[t] >= thresh[t] && L.score[t] > best[t]) { best[t] = L.score[t]; at[t] = ck; }
    }
    // Types in order of fit, each with its own chance.
    int type = 0;
    int2 cw = int2(0);
    for (int pass = 0; pass < 6 && type == 0; pass++) {
        int bt = 0;
        for (int t = 1; t <= 6; t++) if (best[t] > 0.0 && (bt == 0 || best[t] > best[bt])) bt = t;
        if (bt == 0) break;
        if (hashf(pcg(h + 99u + uint(bt) * 13u)) < accept[bt]) { type = bt; cw = at[bt]; }
        best[bt] = 0;
    }
    if (type == 0) { sites[idx] = G.s; return; }
    int2 c = cw - info.xy;
    G.s.head.xy = cw;
    Lie L = survey(hm, c, cw, seed);
    float g = L.g, snowline = L.snowline, waterR = L.waterR;
    int2 waterDir = L.waterDir;
    bool flat5 = L.flat5;

    uint hb = pcg(h + 7u);
    bool ok = false, hearth = false;
    float age = 0.5;
    if (type == T_HENGE) {
        int R = 5 + int(hb % 4u);
        int N = min(7 + int((hb >> 4) % 7u), int(float(R) * 1.6));
        Part hp = makePart(P_HENGE, M_STONE, int2(0), int2(0), R * 4, N, int(hb & 0x7fffu), 0);
        if (((hb >> 12) & 7u) < 3u) {
            int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_RITUAL, 1.0);
            addPart(G, makePart(P_FIRE, M_FIRE, int2(0), int2(1), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 2.0);
        }
        addPart(G, hp, float(R) + 3.0);
        // A heel stone outside the ring.
        if (((hb >> 16) & 1u) != 0u) {
            float a = hashf(pcg(hb)) * TAU;
            int2 o = int2(round(float2(cos(a), sin(a)) * float(R + 7)));
            if (!colArch(hm, c + o) && colH(hm, c + o) > SEA + 2.0)
                addPart(G, makePart(P_BOX, M_STONE, o, int2(1), 4, 0, 0, F_REL), float(R + 9));
        }
        age = 0.85 + 0.15 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_CAIRN) {
        addPart(G, makePart(P_BOX, M_STONE, int2(0), int2(1), 2, 0, 0, F_REL), 2.0);
        addPart(G, makePart(P_BOX, M_STONE, int2(-1), int2(3), 1, 0, 0, F_REL), 3.0);
        age = 0.8 + 0.2 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_RUINS) {
        float2 gr = float2(colH(hm, c + int2(6, 0)) - colH(hm, c - int2(6, 0)), colH(hm, c + int2(0, 6)) - colH(hm, c - int2(0, 6)));
        float ga = length(gr) > 0.5 ? atan2(gr.y, gr.x) : hashf(hb) * TAU;
        float2 up = float2(cos(ga), sin(ga)), along = float2(-up.y, up.x);
        int runs = 2 + int(hb % 3u);
        for (int i = 0; i < runs; i++) {
            uint rh = pcg(hb + uint(i) * 31u);
            float off = (float(i) - float(runs - 1) * 0.5) * 5.0 + (hashf(rh) - 0.5) * 3.0;
            float len = 6.0 + hashf(rh >> 4) * 10.0;
            float shift = (hashf(rh >> 9) - 0.5) * 8.0;
            int2 a = int2(round(up * off + along * (shift - len * 0.5)));
            int2 b = int2(round(up * off + along * (shift + len * 0.5)));
            addPart(G, makePart(P_WALL, M_STONE, a, int2(0), b.x, b.y, 4, int(rh & 0xfu) << 12), length(float2(max(abs(a), abs(b)))) + 2.0);
        }
        if (((hb >> 8) & 1u) != 0u) {
            int2 tc = int2(round(along * (12.0 + hashf(hb >> 10) * 6.0) * (((hb >> 9) & 1u) != 0u ? 1.0 : -1.0)));
            float tl, th;
            if (flatPatch(hm, c + tc, 3, 3.0, tl, th) && tl > SEA + 2.0) {
                addPart(G, makePart(P_RING, M_STONE, tc, int2(2, 3), 0, int(hb >> 14 & 0x7fffu), 6, 0), length(float2(tc)) + 5.0);
                int2 rb = tc + int2(round(-up * 4.0));
                int2 re = tc + int2(round(-up * (8.0 + hashf(hb >> 20) * 4.0)));
                addPart(G, makePart(P_WALL, M_STONE, rb, int2(0), re.x, re.y, 1, 0), length(float2(max(abs(rb), abs(re)))) + 2.0);
            }
        }
        age = 0.6 + 0.3 * hashf(hb >> 3);
        ok = true;
    } else if (type == T_HAMLET) {
        int style = hamletStyle(L);
        ok = buildHamlet(hm, G, c, hb, style);
        hearth = ok && float(flat5) > 0.0 && g < snowline;
        if (hearth) {
            int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_HEARTH, 1.0);
            addPart(G, makePart(P_FIRE, M_FIRE, int2(-1), int2(3), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 3.0);
        }
        if (ok && waterR <= 40.0 && style != ST_STILT) buildJetty(hm, G, c, pcg(hb + 5u), waterDir, 48);
        if (ok && ((hb >> 20) & 3u) == 0u && G.nl < MAX_LAMPS) {
            // A beacon on the highest flat ground nearby.
            float bestH = g + 8.0;
            int2 bo = int2(0);
            for (int ri = 1; ri <= 3; ri++)
                for (int k = 0; k < 16; k++) {
                    float a = float(k) * 0.3926991;
                    int2 o = int2(round(float2(cos(a), sin(a)) * float(ri * 18)));
                    float bl, bh;
                    float hq = colH(hm, c + o);
                    if (hq > bestH && flatPatch(hm, c + o, 1, 0.0, bl, bh)) { bestH = hq; bo = o; }
                }
            if (any(bo != 0)) {
                int lamp = addLamp(G, float3(float(bo.x) + 0.5, bestH + 3.5, float(bo.y) + 0.5), L_BEACON, 1.0);
                addPart(G, makePart(P_FIRE, M_FIRE, bo, int2(1), 2, 0, 0, F_REL | ((lamp + 1) << 8)), length(float2(bo)) + 2.0);
                addPart(G, makePart(P_BOX, M_STONE, bo - 1, int2(3), 1, 0, 0, F_REL), length(float2(bo)) + 3.0);
            }
        }
        age = 0.2 + 0.3 * hashf(hb >> 3);
    } else if (type == T_JETTY) {
        ok = buildJetty(hm, G, c, hb, waterDir, 24);
        age = 0.3;
    } else if (type == T_CAMP) {
        int cst = campStyle(L);
        int lamp = addLamp(G, float3(0.5, g + 1.5, 0.5), L_FIRE, 1.0);
        addPart(G, makePart(P_FIRE, M_FIRE, int2(-1), int2(3), 0, 0, 0, F_REL | ((lamp + 1) << 8)), 3.0);
        int tents = 1 + int(hb % 2u);
        float a0 = hashf(hb >> 4) * TAU;
        int4 rects[4];
        int nr = 0;
        rects[nr++] = int4(c - 1, c + 2);
        bool lit = false;
        for (int i = 0; i < 10 && nr <= tents; i++) {
            float a = a0 + float(i) * 2.39996;
            float2 dir = float2(cos(a), sin(a));
            float rad = 5.0 + float(i / 5);
            int2 tc = c + int2(round(dir * rad));
            // The camp's first shelter is its style's; a lean-to camp adds an A-frame, a caravan an awning.
            int ks = cst;
            if (nr > 1 && cst == ST_LEANTO) ks = ST_AFRAME;
            bool roundT = ks == ST_TIPI || (ks == ST_CARAVAN && nr == 1);
            int2 sz;
            bool axisZ = false;
            if (roundT) {
                sz = int2(5);
            } else {
                // A-frames point a gable at the fire; lean-tos and shelters face it.
                bool radialX = abs(dir.x) >= abs(dir.y);
                axisZ = ks == ST_AFRAME ? !radialX : radialX;
                int L = ks == ST_AFRAME ? 6 : ks == ST_LEANTO ? 5 : 4, D = ks == ST_AFRAME ? 5 : ks == ST_PALM || ks == ST_LEANTO ? 4 : 3;
                sz = axisZ ? int2(D, L) : int2(L, D);
            }
            int2 r0 = tc - sz / 2;
            if (!rectFree(rects, nr, r0, sz, 1)) continue;
            float lo, hi;
            int wet;
            if (!rectGround(hm, r0, sz, lo, hi, wet) || hi - lo > 1.0 || lo < SEA + 2.0) continue;
            float reach = length(float2(tc - c)) + 5.0;
            int lt = -1;
            if (!lit && ks != ST_PALM && ks != ST_LEANTO) {
                lt = addLamp(G, float3(float(tc.x - c.x) + 0.5, hi + 1.0, float(tc.y - c.y) + 0.5), L_TENT, 0.0);
                lit = true;
            }
            int lf = (lt + 1) << 8;
            if (roundT) {
                bool tipi = ks == ST_TIPI;
                addPart(G, makePart(P_ROUND, M_FELT, tc - c, int2(tipi ? 4 : 5, 0), int(hi), tipi ? 0 : 6, tipi ? 18 : 10, (ks << 12) | lf), reach);
            } else {
                addPart(G, makePart(P_HOUSE, ks == ST_PALM ? M_THATCH : M_FELT, r0 - c, sz, int(hi), 0,
                                    int(pcg(hb + uint(i)) & 0x7fffu), F_TENT | (axisZ ? F_AXISZ : 0) | (ks << 12) | lf), reach);
            }
            rects[nr++] = int4(r0, r0 + sz);
        }
        addPart(G, makePart(P_CLEAR, 0, int2(0), int2(13, 0), 0, 0, 0, 0), 14.0);
        age = 0.1;
        ok = true;
    }
    // No part may stand on carved rock (a shelter or notch): stamping skips such columns, which would leave the site
    // with pieces missing. Every second column suffices, as cleanup leaves no carved gap a single column wide; the reach
    // is capped where the reads stay inside every map that holds the cell.
    if (ok && G.np > 0) {
        int R = min(int(ceil(G.r)), 60);
        for (int j = -R; j <= R && ok; j += 2)
            for (int i = -R; i <= R; i += 2) {
                if (i * i + j * j > R * R) continue;
                int2 q = c + int2(i, j);
                if (colArch(hm, q) && (gapFlags(gx.read(uint2(q))) & GAP_CARVED) != 0u) { ok = false; break; }
            }
    }
    if (!ok || G.np == 0) {
        G.s.head.z = 0;
        G.s.info = float4(0);
        for (int i = 0; i < MAX_PARTS; i++) G.s.parts[i] = int4(0);
        for (int i = 0; i < MAX_LAMPS; i++) G.s.lamps[i] = float4(0);
        sites[idx] = G.s;
        return;
    }
    // Props come last, so nothing placed before them moves; tryProp keeps them off every arch column.
    if (type == T_HAMLET) hamletProps(hm, G, c, g, hamletStyle(L), hb, hearth);
    else if (type == T_CAMP) campProps(hm, G, c, g, campStyle(L), hb);
    G.s.head.z = type;
    G.s.info = float4(min(G.r, 125.0), age, float(G.np), float(G.nl));
    sites[idx] = G.s;
}
