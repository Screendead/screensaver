// structTex bits: 0-3 material, 4-6 roof plane (1 -x, 2 +x, 3 -z, 4 +z, 5 level ridge), 7 flat top, 8 clearing,
// 9-13 part, 14-22 site, 23 chimney, 24-27 feature, 28-29 ground mark.
constant uint S_FLAT = 1u << 7, S_CLEAR = 1u << 8, S_CHIMNEY = 1u << 23;
constant uint S_PART = 9u, S_SITE = 14u, S_FEAT = 24u, S_MARK = 28u;
constant uint MK_PATH = 1u, MK_FRINGE = 2u, MK_APRON = 3u;

struct Claim { bool hit; float H, gl, gh, base; uint bits; };

static inline Claim noClaim() { Claim c; c.hit = false; c.H = 0; c.gl = 0; c.gh = 0; c.base = 0; c.bits = 0; return c; }
static inline Claim solidClaim(float H, float base, uint bits) {
    Claim c; c.hit = true; c.H = H; c.gl = 0; c.gh = 0; c.base = base; c.bits = bits; return c;
}
// Solid below gl and from gh to H: lintels, eaves, decks and hulls over open ground or water.
static inline Claim gapClaim(float gl, float gh, float H, uint bits) {
    Claim c; c.hit = true; c.H = H; c.gl = gl; c.gh = gh; c.base = gh; c.bits = bits; return c;
}

// A round dwelling's top over its floor at distance r from its centre, and the outward fall per voxel there.
static inline float2 roundProfile(Part P, float r) {
    float R = float(P.sz.x) * 0.5, crown = float(P.y2) * 0.25, wall = float(P.y1) * 0.25;
    if (((P.flags >> 12) & 15) == ST_TIPI) {
        float Re = R + 0.5;
        return float2(max(crown * (1.0 - r / Re), 0.5), crown / Re);
    }
    float x = min(r / R, 1.0);
    return float2(r > R - 0.9 ? wall : wall + (crown - wall) * (1.0 - x * x), (crown - wall) * 2.0 * x / R);
}

// One part's column at q (map-local), on ground g; `near` collects the clearing around it.
static Claim stampPart(Part P, int2 q, int2 c, float g, texture2d<float, access::read> hm, uint hash, thread bool& near) {
    int2 r0 = c + P.o, r1 = r0 + P.sz;
    uint mat = uint(P.mat);
    bool inRect = all(q >= r0) && all(q < r1);
    bool wetOK = (P.flags & F_WATER) != 0;
    if (P.kind == P_BOX || P.kind == P_FIRE) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        if (!inRect || (!wetOK && g < SEA + 1.0)) return noClaim();
        // A hearth's ring of stones round its flame, lower at the corners so it reads round.
        if (P.kind == P_FIRE && P.sz.x == 3 && any(q != r0 + 1))
            return solidClaim(g + ((q.x == r0.x + 1) != (q.y == r0.y + 1) ? 0.5 : 0.25), g, uint(M_STONE) | S_FLAT);
        float top = P.kind == P_FIRE ? g + (P.sz.x == 3 ? 0.75 : 1.0) + float(P.y0) : (P.flags & F_REL) != 0 ? g + float(P.y0) : float(P.y0);
        if (top <= g) return noClaim();
        return solidClaim(top, g, mat | S_FLAT);
    }
    if (P.kind == P_HOUSE) {
        if (all(q >= r0 - 3) && all(q < r1 + 3)) near = true;
        if (!inRect) return noClaim();
        int style = (P.flags >> 12) & 15;
        uint post = uint(M_TIMBER) | (uint(FT_POST) << S_FEAT);
        if ((P.flags & F_TENT) != 0) {
            // Camp shelters, in half voxels over each column's own ground.
            if (g < SEA + 1.0) return noClaim();
            int4 ij = partIJ(P, q, c);
            int i = ij.x, j = ij.y, L = ij.z, D = ij.w;
            bool endI = i == 0 || i == L - 1, endJ = j == 0 || j == D - 1;
            if (style == ST_AFRAME) {
                // The ridge sags mid-span.
                int rj = abs(j - D / 2);
                if (endI) return noClaim();
                if (rj > 0) return solidClaim(g + 2.5 - float(rj), g, mat);
                return solidClaim(g + 2.5 - 0.25 * round(2.0 * sin(3.14159265 * (float(i) - 0.5) / float(L - 2))), g, mat);
            }
            if (style == ST_LEANTO) {
                // A tarp from a pole on two posts down to the ground behind.
                float2 mid = float2(r0 - c) + float2(P.sz) * 0.5;
                bool frontLo = ((P.flags & F_AXISZ) != 0 ? mid.x : mid.y) > 0.0;
                int jf = frontLo ? j : D - 1 - j;
                if (jf == 0) return endI ? solidClaim(g + 2.5, g, post) : gapClaim(g, g + 1.75, g + 2.5, mat);
                if (jf == 1) return gapClaim(g, g + 1.0, g + 1.75, mat);
                return solidClaim(g + (jf == 2 ? 1.0 : 0.5), g, mat);
            }
            // Palm shelters and caravan awnings: a roof on four posts.
            if (endI && endJ) return solidClaim(g + 2.5, g, post);
            bool inner = !endI && !endJ;
            return gapClaim(g, g + 2.0, g + (style == ST_PALM && inner ? 3.0 : 2.5), mat);
        }
        HouseAt h;
        houseAt(P, q, c, h);
        uint hh = uint(P.y2);
        float fl = float(P.y0);
        if (h.f.flat) {
            if (g < SEA + 1.0) return noClaim();
            if (h.jw < 0) {
                // Vigas: roof beams poking out of the back wall.
                bool viga = (h.iw & 1) == 1 && h.iw < h.Lw - 1 && flatMass(style, hh, h.iw, 0, h.Lw, h.Dw) >= 2.0;
                if (!viga || fl + 2.25 < g + 1.0) return noClaim();
                return gapClaim(g, fl + 2.25, fl + 2.75, uint(M_TIMBER) | (uint(FT_VIGA) << S_FEAT));
            }
            float top = flatTop(style, hh, h.iw, h.jw, h.Lw, h.Dw);
            if (top <= 0.0 || fl + top <= g) return noClaim();
            return solidClaim(fl + top, g, mat | S_FLAT);
        }
        bool stilt = style == ST_STILT;
        bool inWall = h.iw >= 0 && h.jw >= 0 && h.iw < h.Lw && h.jw < h.Dw;
        int ridge = (h.Dw - 1) / 2;
        if ((P.flags & F_CHIMNEY) != 0 && h.iw == chimneyAt(style, h.Lw, hh)) {
            if (h.jw == ridge) return solidClaim(fl + ridgeTop(h) + 1.0, g, uint(M_STONE) | S_FLAT | S_CHIMNEY);
            // The breast stops under the verge, so the roof edge runs across it rather than a pale post.
            if (style == ST_CABIN && abs(h.jw - ridge) == 1) return solidClaim(fl + houseTop(h), g, uint(M_STONE) | S_CHIMNEY);
        }
        if (h.dA < -1) {
            bool endPost = h.iw == 0 || h.iw == h.Lw - 1;
            if (stilt) {
                // The deck in front of the door, on its piles.
                if (h.iw < 0 || h.iw >= h.Lw || g >= fl) return noClaim();
                if (endPost) return solidClaim(fl, g, post);
                uint deck = uint(M_PLANK) | S_FLAT | (uint(FT_DECK) << S_FEAT);
                return g < fl - 1.0 ? gapClaim(g, fl - 0.5, fl, deck) : solidClaim(fl, g, deck);
            }
            return noClaim();
        }
        float H = fl + houseTop(h);
        if (!inWall) {
            // Eaves and rakes: the roof's edge on air; a longhouse's ridge ends rise in horns. A cabin's porch roof
            // stands on posts at its front corners.
            if (stilt && h.jw == ridge && h.dG < 0) H += 1.0;
            if (h.dG < -1 && (h.jw == 0 || h.jw == h.Dw - 1)) return solidClaim(H, g, post);
            float under = H - (style == ST_CABIN && h.dG < 0 && h.dA >= 0 ? max(h.f.lip, roofStep(h) + 0.25) : h.f.lip);
            if (under > g + 1.0) return gapClaim(g, under, H, mat);
            return g < H ? solidClaim(H, g, mat) : noClaim();
        }
        if (stilt) {
            bool pile = ((h.iw & 1) == 0 || h.iw == h.Lw - 1) && (h.jw == 0 || h.jw == h.Dw - 1);
            if (g < fl - 0.5 && !pile) return gapClaim(g, fl, H, mat);
            return solidClaim(H, g, mat);
        }
        if (g < SEA + 1.0) return noClaim();
        return solidClaim(H, g, mat);
    }
    if (P.kind == P_ROUND) {
        // Yurts (a wall ring under a dome) and tipis (a cone), in half voxels.
        float2 dq = float2(q - c - P.o);
        float r = length(dq), R = float(P.sz.x) * 0.5;
        if (r < R + 3.0) near = true;
        if (r > R + 0.2 || g < SEA + 1.0) return noClaim();
        float2 pr = roundProfile(P, r);
        float H = float(P.y0) + round(pr.x * 2.0) * 0.5;
        uint ft = 0u;
        if (r < 0.5) { H += 0.5; ft = uint(((P.flags >> 12) & 15) == ST_TIPI ? FT_POLE : FT_CROWN); }
        if (H <= g) return noClaim();
        return solidClaim(H, g, mat | (ft << S_FEAT));
    }
    if (P.kind == P_PROP) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        if (!inRect || (g < SEA + 1.0 && !wetOK)) return noClaim();
        bool az = P.sz.y > P.sz.x;
        int i = az ? q.y - r0.y : q.x - r0.x, j = az ? q.x - r0.x : q.y - r0.y;
        int L = max(P.sz.x, P.sz.y), D = min(P.sz.x, P.sz.y);
        bool endI = i == 0 || i == L - 1, endJ = j == 0 || j == D - 1;
        float fl = float(P.y0);
        // Quarter steps below 512, where float16 holds them; half steps above.
        float qs = fl < 500.0 ? 0.25 : 0.5;
        uint post = uint(M_TIMBER) | (uint(FT_POST) << S_FEAT);
        uint ph = pcg(hash + uint(P.y2) * 31u + uint(i) * 7u + uint(j) * 131u);
        int type = P.y1;
        float top = 0.0;
        uint bits = mat;
        if (type == PR_WOODPILE) top = fl + (endI ? 1.0 : 1.5);
        else if (type == PR_HAY) top = fl + (endI && endJ ? 1.0 : endI || endJ ? 1.5 : 2.0);
        else if (type == PR_OVEN) { if (endI && endJ) return noClaim(); top = fl + (endI || endJ ? 1.0 : 1.5); }
        else if (type == PR_GARDEN) {
            // A low rail on posts round the beds.
            if ((endI && endJ) || (endI && (j & 1) == 0) || (endJ && (i & 1) == 0)) { top = fl + 1.25; bits = post; }
            else if (endI || endJ) {
                if (fl + 0.4 <= g) return noClaim();
                return gapClaim(g, fl + 0.4, fl + 0.9, uint(M_PLANK) | (uint(FT_FENCE) << S_FEAT));
            }
            else { top = fl + ((i & 1) != 0 ? 0.5 : qs); bits = mat | (uint(FT_CROP) << S_FEAT); }
        } else if (type == PR_WASH || type == PR_RACK) {
            if (endI) { top = fl + (type == PR_WASH ? 2.5 : 2.0); bits = post; }
            else {
                float line = fl + (type == PR_WASH ? 2.0 : 1.75);
                float hang = type == PR_WASH ? ((ph & 3u) == 0u ? 1.0 : 0.5) + 0.5 : 1.0;
                if (line - hang <= g + 0.5) return line > g ? solidClaim(line, g, mat) : noClaim();
                return gapClaim(g, line - hang, line, mat);
            }
        } else if (type == PR_PEN) {
            if (!endI && !endJ) return noClaim();
            if (endJ && j == D - 1 && i == L / 2) return noClaim();
            if (mat == uint(M_STONE)) top = fl + 1.0 + ((ph & 1u) != 0u ? 0.5 : 0.0) - ((ph & 6u) == 0u ? 0.5 : 0.0);
            else if ((endI && endJ) || (endJ && (i & 1) == 0) || (endI && (j & 1) == 0)) { top = fl + 1.5; bits = post; }
            else { if (fl + 0.5 <= g) return noClaim(); return gapClaim(g, fl + 0.5, fl + 1.0, post); }
        } else if (type == PR_LOGSEAT) top = fl + (qs < 0.5 ? 0.75 : 0.5);
        else if (type == PR_BEDROLL) top = fl + (i == 0 ? 0.5 : qs);
        else if (type == PR_RUG) top = fl + qs;
        else if (type == PR_CANOE) {
            if (g < SEA) return gapClaim(g, SEA - 0.35, SEA + (endI ? 0.65 : 0.4), mat);
            top = fl + (endI ? 0.75 : 0.5);
        }
        else if (type == PR_STUMP) top = fl + (qs < 0.5 ? 0.75 : 1.0);
        else if (type == PR_PEAT || type == PR_BALES || type == PR_BARREL) top = fl + 1.0;
        if (top <= g) return noClaim();
        return solidClaim(top, g, bits);
    }
    if (P.kind == P_DECK) {
        if (all(q >= r0 - 2) && all(q < r1 + 2)) near = true;
        float T = float(P.y0);
        if (!inRect || g >= T) return noClaim();
        bool axisZ = (P.flags & F_AXISZ) != 0;
        int i = axisZ ? q.y - r0.y : q.x - r0.x;
        int j = axisZ ? q.x - r0.x : q.y - r0.y;
        int W = max(axisZ ? P.sz.x : P.sz.y, 1);
        // Pilings every third plank, on alternate sides of a wide deck.
        if (g < SEA && i % 3 == 1 && j == (i / 3) % W) return solidClaim(T, g, uint(M_TIMBER) | S_FLAT);
        if (g >= T - 1.0) return solidClaim(T, g, mat | S_FLAT);
        return gapClaim(g, T - 1.0, T, mat | S_FLAT);
    }
    if (P.kind == P_BOAT) {
        if (all(q >= r0 - 1) && all(q < r1 + 1)) near = true;
        if (!inRect) return noClaim();
        if (g < SEA - 1.0) return gapClaim(g, 19.0, 21.0, mat);
        return solidClaim(max(g + 1.0, 21.0), g, mat);
    }
    if (P.kind == P_HENGE) {
        float2 cc = float2(c + P.o) + 0.5;
        float R = float(P.y0) * 0.25;
        int N = max(P.y1, 1);
        float2 dq = float2(q) + 0.5 - cc;
        float r = length(dq);
        if (r > R + 4.0) return noClaim();
        near = true;
        if (r < R - 3.0) return noClaim();
        if (g < SEA + 1.0) return noClaim();
        float stepA = TAU / float(N);
        int k = int(round(atan2(dq.y, dq.x) / stepA));
        k = (k % N + N) % N;
        float ang = float(k) * stepA;
        float2 dir = float2(cos(ang), sin(ang));
        int2 sc = int2(floor(cc + dir * R));
        // The tangent runs along z where the radius runs along x.
        bool alongZ = abs(dir.x) >= abs(dir.y);
        int2 tv = alongZ ? int2(0, 1) : int2(1, 0);
        int2 rd = alongZ ? int2(dir.x > 0.0 ? 1 : -1, 0) : int2(0, dir.y > 0.0 ? 1 : -1);
        uint kh = pcg(hash + uint(P.y2) * 31u + uint(k) * 977u);
        float u = hashf(kh);
        float T = colH(hm, sc) + 3.0 + float((kh >> 8) % 3u);
        if (u < 0.2) return noClaim();
        if (u < 0.34) {
            // Fallen outward.
            for (int m = 0; m < 3; m++) if (all(q == sc + rd * m)) return solidClaim(g + 1.0, g, mat | S_FLAT);
            return noClaim();
        }
        if (u > 0.8) {
            // Trilithon: two uprights and a lintel over the gap between them.
            if (all(q == sc - tv) || all(q == sc + tv)) return solidClaim(max(T, g + 2.0), g, mat | S_FLAT);
            if (all(q == sc) && T - 1.0 > g + 1.5) return gapClaim(g, T - 1.0, T, mat | S_FLAT);
            return noClaim();
        }
        if (all(q == sc) || (((kh >> 12) & 1u) != 0u && all(q == sc + tv))) return solidClaim(max(T - 1.0, g + 2.0), g, mat | S_FLAT);
        return noClaim();
    }
    if (P.kind == P_WALL) {
        int2 a = c + P.o, b = c + int2(P.y0, P.y1);
        int2 lo = min(a, b), hi = max(a, b);
        if (any(q < lo - 2) || any(q > hi + 2)) return noClaim();
        float2 dv = float2(b - a);
        float s;
        bool on;
        if (abs(dv.x) >= abs(dv.y)) {
            float zl = float(a.y) + float(q.x - a.x) * (dv.x != 0.0 ? dv.y / dv.x : 0.0);
            on = q.x >= lo.x && q.x <= hi.x && int(round(zl)) == q.y;
            if (abs(zl - float(q.y)) < 2.5) near = true;
            s = abs(float(q.x - a.x));
        } else {
            float xl = float(a.x) + float(q.y - a.y) * dv.x / dv.y;
            on = q.y >= lo.y && q.y <= hi.y && int(round(xl)) == q.x;
            if (abs(xl - float(q.x)) < 2.5) near = true;
            s = abs(float(q.y - a.y));
        }
        if (!on || g < SEA + 1.0) return noClaim();
        // Eroded: the run breaks where the noise dips, and its broken top rises and falls.
        float nz = perlin2(float2(s / 5.0, float(P.flags >> 12) * 7.3 + 0.5), hash + 300u);
        if (nz < -0.22) return noClaim();
        float hgt = max(1.0, round(float(P.y2) * (0.6 + nz)));
        return solidClaim(g + hgt, g, mat | S_FLAT);
    }
    if (P.kind == P_CLEAR) {
        // A glade: no trunk within sz.x of o, so canopies keep off the camp.
        float2 dq = float2(q - c - P.o);
        if (dot(dq, dq) < float(P.sz.x * P.sz.x)) near = true;
        return noClaim();
    }
    if (P.kind == P_RING) {
        float2 cc = float2(c + P.o) + 0.5;
        float2 dq = float2(q) + 0.5 - cc;
        float r = length(dq);
        if (r < float(P.sz.y) + 3.0) near = true;
        if (r < float(P.sz.x) - 0.3 || r > float(P.sz.y) + 0.3 || g < SEA + 1.0) return noClaim();
        // A toppled tower: a ring of wall whose broken top falls from one side to the other.
        float th0 = hashf(uint(P.y1)) * TAU;
        float k = 0.5 + 0.5 * cos(atan2(dq.y, dq.x) - th0);
        float hgt = 1.0 + round(float(P.y2) * k * k + perlin2(dq * 0.7, hash + 301u));
        return solidClaim(g + max(hgt, 1.0), g, mat | S_FLAT);
    }
    return noClaim();
}

// Worn ground at a column no part claims: the apron round a hearth or camp fire, and a hamlet's paths from its hearth
// to each door (and the jetty), wandering a little.
static uint groundMark(device const Site& s, int2 q, int2 c) {
    int np = int(s.info.z);
    bool hearth = false;
    for (int k = 0; k < np && !hearth; k++) {
        int4 v = s.parts[k];
        hearth = ((v.y >> 16) & 0xff) == P_FIRE && (v.y & 0xff) == 3 && (v.x & 0xffff) == 0xffff;
    }
    if (!hearth) return 0u;
    uint sh = uint(s.head.w);
    float2 p = float2(q - c);
    float2 wob = float2(perlin2(p / 6.0, sh + 340u), perlin2(p / 6.0 + 31.7, sh + 341u));
    float r = length(p);
    if (r < 2.7 + 1.2 * wob.x) return MK_APRON;
    uint fringe = r < 4.3 + 1.6 * wob.y ? MK_FRINGE : 0u;
    if (s.head.z != T_HAMLET) return fringe;
    float2 pw = p + 1.1 * wob;
    for (int k = 0; k < np; k++) {
        Part P = unpackPart(s.parts[k]);
        float2 d;
        if (P.kind == P_DECK) d = clamp(float2(0.0), float2(P.o), float2(P.o + P.sz - 1));
        else {
            float3 dr = houseDoor(P, sh, k, true);
            if (dr.z == 0.0) continue;
            d = dr.xy - 0.5;
        }
        float t = saturate(dot(pw, d) / max(dot(d, d), 1e-3));
        if (length(pw - d * t) < 0.75 || length(p - d) < 1.2) return MK_PATH;
    }
    return fringe;
}

// Raises each site's parts into the cleaned columns, and marks the clearing around them, where no tree takes a
// trunk. A structure column holds -base in alpha (base: its first structure voxel) and its detail in `st`; a
// clearing column holds -H, which reads as no structure.
kernel void stampStructures(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            texture2d<uint, access::write> st [[texture(2)]],
                            device const Site* sites [[buffer(0)]],
                            constant int4& info [[buffer(1)]],
                            constant uint& row0 [[buffer(2)]],
                            uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, row0);
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 col = src.read(gid);
    int2 q = int2(gid);
    // Arches keep their single gap.
    if (col.b <= col.g) {
        int2 home = int2(floor(float2(q + info.xy) / float(SITE_CELL))) - siteCell0(info.xy);
        bool near = false;
        int nearSite = 0;
        uint mark = 0u;
        for (int jj = -1; jj <= 1; jj++)
            for (int ii = -1; ii <= 1; ii++) {
                int2 gi = home + int2(ii, jj);
                if (any(gi < 0) || any(gi >= SITE_GRID)) continue;
                int si = gi.y * SITE_GRID + gi.x;
                device const Site& s = sites[si];
                if (s.head.z == 0) continue;
                int2 c = s.head.xy - info.xy;
                float2 dd = float2(q - c);
                if (dot(dd, dd) > (s.info.x + 4.0) * (s.info.x + 4.0)) continue;
                int np = int(s.info.z);
                for (int k = 0; k < np; k++) {
                    bool nk = false;
                    Claim cl = stampPart(unpackPart(s.parts[k]), q, c, col.r, src, uint(s.head.w), nk);
                    if (cl.hit) {
                        dst.write(float4(cl.H, cl.gl, cl.gh, -cl.base), gid);
                        st.write(uint4(cl.bits | (uint(k) << S_PART) | (uint(si) << S_SITE), 0, 0, 0), gid);
                        return;
                    }
                    if (nk && !near) { near = true; nearSite = si; }
                }
                if (mark == 0u && (s.head.z == T_HAMLET || s.head.z == T_CAMP)) {
                    mark = groundMark(s, q, c);
                    if (mark != 0u && !near) { near = true; nearSite = si; }
                }
            }
        if (near) {
            dst.write(float4(col.rgb, -col.r), gid);
            st.write(uint4(S_CLEAR | (uint(nearSite) << S_SITE) | (mark << S_MARK), 0, 0, 0), gid);
            return;
        }
    }
    // A gap column keeps its alpha: only an island's is non-zero here (isleAlpha), which planting turns into its word.
    dst.write(float4(col.rgb, col.b > col.g ? col.a : 0.0), gid);
    st.write(uint4(0), gid);
}
