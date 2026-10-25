// Houses on flat treads around a central hearth, in the local style: cottages, timber frames, cabins, crofts,
// adobe courtyards, pueblos, stilt longhouses or yurts.
static bool buildHamlet(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, uint h, int style) {
    int n = int(hm.get_width());
    int4 rects[8];
    int nr = 0;
    rects[nr++] = int4(c - 2, c + 3);
    int houses = 0, want = 3 + int(h % 3u);
    float a0 = hashf(pcg(h + 1u)) * 6.2831853;
    bool stilt = style == ST_STILT, pueblo = style == ST_PUEBLO, yurt = style == ST_YURT;
    float rIn = stilt || pueblo ? 5.0 : 8.0, rSpan = pueblo ? 12.0 : 26.0;
    for (int k = 0; k < 64 && houses < want && G.np < MAX_PARTS - 8 && G.nl < MAX_LAMPS - 3; k++) {
        float r = rIn + rSpan * sqrt(float(k) / 64.0);
        float a = a0 + float(k) * 2.39996;
        int2 pos = c + int2(round(r * float2(cos(a), sin(a))));
        uint hh = pcg(h + 17u * uint(k) + 3u) & 0x7fffu;
        if (style == ST_CABIN) {
            // A cabin turns its porch gable to the hearth.
            int2 dh = c - pos;
            bool az = abs(dh.y) > abs(dh.x);
            hh = (hh & ~5u) | (az ? 1u : 0u) | ((az ? dh.y : dh.x) > 0 ? 4u : 0u);
        }
        bool axisZ = (hh & 1u) != 0u;
        int b9 = int((hh >> 9) % 3u), b11 = int((hh >> 11) & 1u);
        if (yurt) {
            int R2 = 5 + b9, rad = (R2 + 1) / 2;
            int2 r0 = pos - rad, sz = int2(2 * rad + 1);
            if (any(r0 < 2) || any(r0 + sz >= n - 2) || !rectFree(rects, nr, r0, sz, 2)) continue;
            float lo, hi;
            int wet;
            if (!rectGround(hm, r0, sz, lo, hi, wet) || hi - lo > 1.0 || lo < SEA + 2.2) continue;
            int floorY = int(hi);
            Part yp = makePart(P_ROUND, M_FELT, pos - c, int2(R2, 0), floorY, 8, 14, F_CHIMNEY | (style << 12));
            float2 lxz = floor(houseDoor(yp, uint(G.s.head.w), G.np, false).xy) + 0.5;
            float rBefore = G.r;
            int lamp = addLamp(G, float3(lxz.x, float(floorY) + 1.2, lxz.y), L_WINDOW, 2.4, pos - c - int2(floor(lxz)));
            G.r = max(rBefore, length(float2(pos - c) + 0.5) + 2.0);
            yp.flags |= (lamp + 1) << 8;
            addPart(G, yp, length(float2(pos - c)) + float(rad) + 2.0);
            rects[nr++] = int4(r0, r0 + sz);
            houses++;
            continue;
        }
        HouseForm f = houseForm(style, hh);
        int Lw = 6 + (b9 & 1), Dw = 4 + b11;
        if (style == ST_TIMBER) { Lw = 7 + (b9 & 1); Dw = 5; }
        else if (style == ST_CABIN) { Lw = 5 + (b9 & 1); Dw = 5; }
        else if (style == ST_CROFT) Lw = 7 + b9;
        else if (stilt) Lw = 8 + b9;
        else if (style == ST_ADOBE) { Lw = 7 + b9; Dw = 7 + b11; }
        else if (pueblo) { Lw = 6 + b9; Dw = 5 + b11; }
        int L = Lw + f.ovGLo + f.ovGHi, Dt = Dw + f.ovLo + f.ovHi;
        int2 sz = axisZ ? int2(Dt, L) : int2(L, Dt);
        int2 r0 = pos - sz / 2;
        if (any(r0 < 2) || any(r0 + sz >= n - 2) || !rectFree(rects, nr, r0, sz, pueblo ? 1 : 2)) continue;
        int2 w0 = r0 + (axisZ ? int2(f.ovLo, f.ovGLo) : int2(f.ovGLo, f.ovLo));
        int2 wsz = axisZ ? int2(Dw, Lw) : int2(Lw, Dw);
        float lo, hi, alo, ahi;
        int wet, awet;
        if (!rectGround(hm, r0, sz, alo, ahi, awet) || !rectGround(hm, w0, wsz, lo, hi, wet)) continue;
        if (stilt) { if (ahi > SEA + 1.2 || alo < SEA - 6.0 || awet * 3 < sz.x * sz.y) continue; }
        else if (hi - lo > (f.flat ? 1.5 : 2.0) || ahi > hi + 1.0 || lo < SEA + 2.2) continue;
        int floorY = stilt ? 23 : int(hi);
        HouseAt ha;
        ha.f = f; ha.Lw = Lw; ha.Dw = Dw;
        bool chim = !f.flat && !stilt && (style == ST_CABIN || style == ST_CROFT || ((hh >> 3) & 1u) != 0u);
        float top = f.flat ? (pueblo ? 6.0 : 4.5) : ridgeTop(ha);
        int flags = (axisZ ? F_AXISZ : 0) | (stilt ? F_STILT | F_WATER : 0) | (chim ? F_CHIMNEY : 0) | (style << 12);
        int wall = style == ST_CROFT ? M_STONE : f.flat ? M_ADOBE : style == ST_CABIN || stilt ? M_LOG : M_PLASTER;
        Part hp = makePart(P_HOUSE, wall, r0 - c, sz, floorY, 0, int(hh), flags);
        // The lamp at the door; smoke from the chimney stack, or a croft's roof hole over the house centre.
        float2 lxz = floor(houseDoor(hp, 0u, 0, false).xy) + 0.5;
        int ci = chimneyAt(style, Lw, hh);
        int2 smokeCol = ci > -50 ? wallCol(hp, f, ci, (Dw - 1) / 2) : pos - c;
        float rBefore = G.r;
        int lamp = addLamp(G, float3(lxz.x, float(floorY) + 1.2, lxz.y), L_WINDOW, chim ? top + 1.0 - 1.2 : 0.0,
                           smokeCol - int2(floor(lxz)));
        G.r = max(rBefore, length(float2(pos - c) + 0.5) + 2.0);
        hp.flags |= (lamp + 1) << 8;
        addPart(G, hp, float(partReach(hp)));
        rects[nr++] = int4(r0, r0 + sz);
        houses++;
    }
    return houses >= 3;
}

// The ground a part takes, offset from the site centre (min, max exclusive); empty for parts that take none.
static inline int4 partRect(Part P) {
    if (P.kind == P_ROUND) { int rad = (P.sz.x + 1) / 2; return int4(P.o - rad, P.o + rad + 1); }
    if (P.kind == P_HOUSE || P.kind == P_DECK || P.kind == P_BOAT || P.kind == P_PROP || P.kind == P_BOX || P.kind == P_FIRE)
        return int4(P.o, P.o + P.sz);
    return int4(0);
}

// A prop at offset o if its ground is free (`gap` columns from every other part), dry and level, and within 1.5 of
// `ref`; wet props stand in the water off a stilt house, on floor `ref`.
static bool tryProp(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, int2 o, int2 sz, int type, int mat,
                    int style, float ref, bool wet, int gap, uint ph) {
    if (G.np >= MAX_PARTS) return false;
    int n = int(hm.get_width());
    int2 r0 = c + o;
    if (any(r0 < 2) || any(r0 + sz >= n - 2)) return false;
    for (int k = 0; k < G.np; k++) {
        int4 r = partRect(unpackPart(G.s.parts[k]));
        if (r.z > r.x && all(o < r.zw + gap) && all(o + sz + gap > r.xy)) return false;
    }
    float lo, hi;
    int nw;
    if (!rectGround(hm, r0, sz, lo, hi, nw)) return false;
    int fl;
    if (wet) {
        if (nw != sz.x * sz.y || lo < SEA - 6.0) return false;
        fl = int(ref);
    } else {
        if (nw > 0 || hi - lo > 1.0 || lo < SEA + 1.2 || abs(hi - ref) > 1.5) return false;
        fl = int(ceil(hi - 0.01));
    }
    Part p = makePart(P_PROP, mat, o, sz, fl, type, int(ph & 0x7fffu), (style << 12) | (wet ? F_WATER : 0));
    addPart(G, p, float(partReach(p)) + 1.0);
    return true;
}

// A prop somewhere on a ring round the site centre, lying along the ring.
static bool ringProp(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, int type, int mat, int style, float ref,
                     int Lp, int Dp, float r0, uint ph) {
    float a0 = hashf(ph) * 6.2831853;
    for (int t = 0; t < 12; t++) {
        float a = a0 + float(t) * 2.39996, r = r0 + float(t % 3);
        float2 dir = float2(cos(a), sin(a));
        int2 sz = abs(dir.x) > abs(dir.y) ? int2(Dp, Lp) : int2(Lp, Dp);
        int2 o = int2(round(dir * r)) - sz / 2;
        if (tryProp(hm, G, c, o, sz, type, mat, style, ref, false, 1, pcg(ph + uint(t)))) return true;
    }
    return false;
}

// Seats round a fire's ring of stones, a column clear of it.
static void fireSeats(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int style, uint h, int want) {
    const int2 so[4] = { int2(3, -1), int2(-3, -1), int2(-1, 3), int2(-1, -3) };
    int got = 0;
    for (int t = 0; t < 4 && got < want; t++) {
        int m = int((h + uint(t)) & 3u);
        int2 sz = m < 2 ? int2(1, 3) : int2(3, 1);
        if (tryProp(hm, G, c, so[m], sz, PR_LOGSEAT, M_LOG, style, g, false, 1, pcg(h + uint(t) * 5u))) got++;
    }
}

// What a dwelling keeps round it, by style: (type, length, width, material); a wet prop stands in the water.
struct PropKind { int type, Lp, Dp, mat; bool wet; };

static inline PropKind propFor(int style, uint pick) {
    PropKind k;
    k.wet = false;
    int t;
    if (style == ST_MEADOW) { const int m[4] = { PR_GARDEN, PR_WASH, PR_HAY, PR_BARREL }; t = m[pick % 4u]; }
    else if (style == ST_TIMBER) { const int m[4] = { PR_WOODPILE, PR_GARDEN, PR_WASH, PR_BARREL }; t = m[pick % 4u]; }
    else if (style == ST_CABIN) { const int m[3] = { PR_WOODPILE, PR_STUMP, PR_BARREL }; t = m[pick % 3u]; }
    else if (style == ST_CROFT) { const int m[3] = { PR_PEN, PR_PEAT, PR_HAY }; t = m[pick % 3u]; }
    else if (style == ST_ADOBE || style == ST_PUEBLO) { const int m[2] = { PR_OVEN, PR_RACK }; t = m[pick % 2u]; }
    else if (style == ST_STILT) { const int m[2] = { PR_CANOE, PR_RACK }; t = m[pick % 2u]; k.wet = true; }
    else { const int m[3] = { PR_PEN, PR_BALES, PR_RUG }; t = m[pick % 3u]; }
    k.type = t;
    k.Lp = t == PR_GARDEN ? 5 : t == PR_PEN ? (style == ST_YURT ? 7 : 6) : t == PR_WASH ? 4 : t == PR_CANOE ? 5
         : t == PR_HAY || t == PR_OVEN || t == PR_WOODPILE || t == PR_RACK || t == PR_RUG ? 3 : t == PR_STUMP || t == PR_BARREL ? 1 : 2;
    k.Dp = t == PR_GARDEN ? 4 : t == PR_PEN ? 5 : t == PR_HAY || t == PR_OVEN ? 3 : t == PR_RUG ? 2 : 1;
    k.mat = t == PR_WOODPILE || t == PR_STUMP ? M_LOG : t == PR_HAY || t == PR_BALES ? M_THATCH : t == PR_OVEN ? M_ADOBE
          : t == PR_PEN ? (style == ST_CROFT ? M_STONE : M_TIMBER) : t == PR_WASH || t == PR_RACK || t == PR_RUG ? M_CLOTH
          : t == PR_GARDEN ? M_PLANK : M_TIMBER;
    return k;
}

// Each dwelling's yard: up to two props by its gables, back or front corners (never before the door), then seats by
// the hearth. Added after every other part, so nothing the hamlet had before moves.
static void hamletProps(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int style, uint hb, bool hearth) {
    int nh = G.np;
    for (int k = 0; k < nh; k++) {
        Part H = unpackPart(G.s.parts[k]);
        bool round = H.kind == P_ROUND;
        if (!(H.kind == P_HOUSE && (H.flags & F_TENT) == 0) && !round) continue;
        uint hh = pcg(hb + uint(k) * 101u);
        int fl = H.y0;
        float3 door = houseDoor(H, uint(G.s.head.w), k, true);
        int2 dc = int2(floor(door.xy));
        float da = hashf(pcg(uint(G.s.head.w) + uint(k))) * 6.2831853;
        for (int pi = 0; pi < 2; pi++) {
            PropKind pk = propFor(style, hh >> (pi * 5));
            if (pi == 1 && pk.type == propFor(style, hh).type) continue;
            float ref = pk.wet ? (pk.type == PR_CANOE ? SEA : float(fl)) : float(fl);
            int gap = pk.type == PR_WOODPILE || pk.type == PR_PEAT || pk.type == PR_BARREL ? 0 : 1;
            for (int t = 0; t < 4; t++) {
                int pos = int((hh >> 11) + uint(t) + uint(pi) * 2u) & 3;
                int2 o, sz;
                if (round) {
                    int rad = (H.sz.x + 1) / 2;
                    int2 d = pos == 0 ? int2(1, 0) : pos == 1 ? int2(-1, 0) : pos == 2 ? int2(0, 1) : int2(0, -1);
                    if (dot(float2(d), float2(cos(da), sin(da))) > 0.7) continue;
                    sz = d.x != 0 ? int2(pk.Dp, pk.Lp) : int2(pk.Lp, pk.Dp);
                    int2 cc = H.o + d * (rad + 2 + (d.x != 0 ? sz.x : sz.y) / 2);
                    o = cc - sz / 2;
                } else {
                    bool axisZ = (H.flags & F_AXISZ) != 0;
                    int L = axisZ ? H.sz.y : H.sz.x, Dt = axisZ ? H.sz.x : H.sz.y;
                    bool doorHi = (uint(H.y2) & 4u) != 0u || houseForm((H.flags >> 12) & 15, uint(H.y2)).flat;
                    int li, lj, i0, j0;
                    if (pos < 2) { li = pk.Dp; lj = pk.Lp; i0 = pos == 0 ? -li - gap : L + gap; j0 = (Dt - lj) / 2; }
                    else {
                        li = pk.Lp; lj = pk.Dp;
                        bool back = pos == 2;
                        j0 = back == doorHi ? -lj - gap : Dt + gap;
                        i0 = back ? (L - li) / 2 : ((hh >> 14) & 1u) != 0u ? 0 : L - li;
                    }
                    o = H.o + (axisZ ? int2(j0, i0) : int2(i0, j0));
                    sz = axisZ ? int2(lj, li) : int2(li, lj);
                }
                // Nothing stands at or beside the spot outside the door.
                if (door.z > 0.0 && all(dc >= o - 1) && all(dc <= o + sz)) continue;
                if (tryProp(hm, G, c, o, sz, pk.type, pk.mat, style, ref, pk.wet, pk.wet ? 1 : gap, pcg(hh + uint(t) * 3u + uint(pi)))) break;
            }
        }
    }
    if (hearth && style != ST_STILT && style != ST_ADOBE && style != ST_PUEBLO) fireSeats(hm, G, c, g, style, hb >> 3, 2);
}

// A camp's seats round its fire, a woodpile, and its style's gear.
static void campProps(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, float g, int cst, uint hb) {
    fireSeats(hm, G, c, g, cst, hb >> 6, 2);
    ringProp(hm, G, c, PR_WOODPILE, M_LOG, cst, g, 2, 1, 5.0, pcg(hb + 61u));
    if (cst == ST_AFRAME || cst == ST_LEANTO) ringProp(hm, G, c, PR_BEDROLL, M_CLOTH, cst, g, 2, 1, 4.0, pcg(hb + 62u));
    else if (cst == ST_TIPI) ringProp(hm, G, c, PR_RACK, M_CLOTH, cst, g, 3, 1, 5.0, pcg(hb + 63u));
    else if (cst == ST_CARAVAN) {
        ringProp(hm, G, c, PR_RUG, M_CLOTH, cst, g, 3, 2, 4.0, pcg(hb + 64u));
        ringProp(hm, G, c, PR_BALES, M_THATCH, cst, g, 2, 1, 5.0, pcg(hb + 65u));
    } else if (cst == ST_PALM) {
        ringProp(hm, G, c, PR_CANOE, M_TIMBER, cst, g, 4, 1, 5.0, pcg(hb + 66u));
        ringProp(hm, G, c, PR_RACK, M_CLOTH, cst, g, 3, 1, 5.0, pcg(hb + 67u));
    }
}
