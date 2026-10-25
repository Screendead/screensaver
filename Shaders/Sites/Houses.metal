// A style's house: wall height over the floor, roof pitch (rise per column), the eave lip's depth, and the overhangs
// past the walls across the ridge (lo, hi side) and beyond each gable. The door side's overhang is a porch or deck:
// a longhouse's deck runs along it, a cabin's porch is under its front gable.
struct HouseForm { int ovLo, ovHi, ovGLo, ovGHi; float wall, pitch, lip; bool hip, flat; };

static inline HouseForm houseForm(int style, uint hh) {
    HouseForm f;
    f.ovLo = 1; f.ovHi = 1; f.ovGLo = 1; f.ovGHi = 1; f.wall = 2.5; f.pitch = 1.0; f.lip = 1.0; f.hip = false; f.flat = false;
    bool doorHi = (hh & 4u) != 0u;
    if (style == ST_MEADOW) { f.hip = true; f.pitch = 1.0; }
    else if (style == ST_TIMBER) { f.wall = 3.0; f.pitch = 1.25; f.lip = 0.75; }
    else if (style == ST_CABIN || style == ST_STILT) {
        f.pitch = style == ST_CABIN ? 1.25 : 1.5; f.lip = 1.0;
        if (style == ST_CABIN) { f.wall = 2.0; if (doorHi) f.ovGHi = 2; else f.ovGLo = 2; }
        else if (doorHi) f.ovHi = 2; else f.ovLo = 2;
    }
    else if (style == ST_CROFT) { f.ovLo = 0; f.ovHi = 0; f.ovGLo = 0; f.ovGHi = 0; f.wall = 2.0; f.pitch = 0.75; f.hip = true; }
    else if (style == ST_ADOBE || style == ST_PUEBLO) { f.flat = true; f.ovLo = 1; f.ovHi = 0; f.ovGLo = 0; f.ovGHi = 0; f.wall = 3.0; }
    return f;
}

// A house column in wall coordinates: iw along the ridge, jw across it (0 at the first wall), and its distance in
// from the nearest long wall (dA) and gable wall (dG); negative over the overhangs.
struct HouseAt { int iw, jw, Lw, Dw, dA, dG; bool axisZ; HouseForm f; };

static inline int4 partIJ(Part P, int2 q, int2 c) {
    int2 r0 = c + P.o;
    bool axisZ = (P.flags & F_AXISZ) != 0;
    return axisZ ? int4(q.y - r0.y, q.x - r0.x, P.sz.y, P.sz.x) : int4(q.x - r0.x, q.y - r0.y, P.sz.x, P.sz.y);
}

static inline void houseAt(Part P, int2 q, int2 c, thread HouseAt& h) {
    int4 ij = partIJ(P, q, c);
    h.axisZ = (P.flags & F_AXISZ) != 0;
    h.f = houseForm((P.flags >> 12) & 15, uint(P.y2));
    h.Lw = ij.z - h.f.ovGLo - h.f.ovGHi; h.Dw = ij.w - h.f.ovLo - h.f.ovHi;
    h.iw = ij.x - h.f.ovGLo; h.jw = ij.y - h.f.ovLo;
    h.dA = min(h.jw, h.Dw - 1 - h.jw); h.dG = min(h.iw, h.Lw - 1 - h.iw);
}

// Roof top over the floor (before quantizing) and its rise per column there, from the eave line half a voxel over
// the wall top; hipped roofs slope toward the gables too.
static inline float2 roofProfile(thread const HouseAt& h) {
    int ridge = (h.Dw - 1) / 2;
    float d = float(h.f.hip ? min(h.dA, h.dG) : h.dA);
    return float2(h.f.wall + 0.5 + h.f.pitch * max(d + 1.0, 0.0), d >= float(ridge) ? 0.0 : h.f.pitch);
}

// The tallest step between neighbouring roof columns: the roof course is at least this deep.
static inline float roofStep(thread const HouseAt& h) { return h.f.pitch; }

static inline float ridgeTop(thread const HouseAt& h) {
    return round((h.f.wall + 0.5 + h.f.pitch * float((h.Dw - 1) / 2 + 1)) * 4.0) * 0.25;
}

// A pitched house's top over the floor at a column, in quarter voxels.
static inline float houseTop(thread const HouseAt& h) {
    if (h.dA < -1) return h.f.wall;
    return round(roofProfile(h).x * 4.0) * 0.25;
}

// The along-ridge column of a house's chimney: in the gable wall, outside it (a cabin's stack) or through the hip.
static inline int chimneyAt(int style, int Lw, uint hh) {
    if (style == ST_CABIN) return (hh & 4u) != 0u ? -1 : Lw;
    int i = style == ST_MEADOW ? 1 : style == ST_TIMBER ? 0 : -100;
    return ((hh >> 12) & 1u) != 0u ? Lw - 1 - i : i;
}

// Height over the floor of an adobe house's mass (wings round a courtyard, U or L) or a pueblo's (a block with a
// second tier set back on its back half); a low wall closes the courtyard, with a gate. 0 outside the mass.
static inline float flatMass(int style, uint hh, int iw, int jw, int Lw, int Dw) {
    if (iw < 0 || jw < 0 || iw >= Lw || jw >= Dw) return 0.0;
    if (style == ST_PUEBLO) return jw < Dw / 2 && iw >= 1 && iw < Lw - 1 - int((hh >> 7) & 1u) ? 5.5 : 3.0;
    bool uPlan = ((hh >> 7) & 1u) != 0u;
    bool court = jw >= 3 && iw >= 3 && (!uPlan || iw <= Lw - 4);
    if (court) {
        int gate = uPlan ? Lw / 2 : (Lw + 2) / 2;
        return jw == Dw - 1 && iw != gate ? 1.5 : 0.0;
    }
    return jw < 3 && ((hh >> 8) & 1u) != 0u ? 4.0 : 3.0;
}

// The wall column a ladder leans on, in wall coordinates (x < 0: none), and the height it starts from: the face of a
// pueblo's second tier toward +j, or an adobe side wing's face toward +i, onto the courtyard.
static inline int3 flatLadder(int style, uint hh, int Lw, int Dw) {
    if (style == ST_PUEBLO) return int3(((hh >> 10) & 1u) != 0u ? 2 : Lw - 4, Dw / 2 - 1, 3);
    if (((hh >> 10) & 1u) == 0u) return int3(-1);
    return int3(2, Dw - 2, 0);
}

// The mass with a parapet round its roof edges, lower at the corners.
static inline float flatTop(int style, uint hh, int iw, int jw, int Lw, int Dw) {
    float m = flatMass(style, hh, iw, jw, Lw, Dw);
    if (m < 2.0) return m;
    bool ex = flatMass(style, hh, iw - 1, jw, Lw, Dw) < m || flatMass(style, hh, iw + 1, jw, Lw, Dw) < m;
    bool ez = flatMass(style, hh, iw, jw - 1, Lw, Dw) < m || flatMass(style, hh, iw, jw + 1, Lw, Dw) < m;
    return m + (ex && ez ? 0.25 : ex || ez ? 0.5 : 0.0);
}

// What a house wall column shows on its face toward `side` (0 -x, 1 +x, 2 -z, 3 +z): 0 plain wall, 1 a window, 2 the
// door, 3 a window on a pueblo's upper tier.
static inline int houseOpening(Part P, thread const HouseAt& h, uint side) {
    uint hh = uint(P.y2);
    int style = (P.flags >> 12) & 15;
    uint sLo = h.axisZ ? 0u : 2u, sHi = h.axisZ ? 1u : 3u, gLo = h.axisZ ? 2u : 0u, gHi = h.axisZ ? 3u : 1u;
    int di = side == gLo ? -1 : side == gHi ? 1 : 0, dj = side == sLo ? -1 : side == sHi ? 1 : 0;
    float k = hashf(pcg(hh * 131u + uint(h.iw + 8) * 17u + uint(h.jw + 8) * 5u + side));
    if (h.f.flat) {
        float m = flatMass(style, hh, h.iw, h.jw, h.Lw, h.Dw);
        float o = flatMass(style, hh, h.iw + di, h.jw + dj, h.Lw, h.Dw);
        if (m < 2.0 || o >= m) return 0;
        if (o >= 2.0) return k < 0.5 ? 3 : 0;
        bool doorCol = style == ST_PUEBLO ? h.iw == h.Lw / 2 && h.jw == h.Dw - 1 : h.iw == h.Lw / 2 && h.jw == 2;
        if (doorCol && dj == 1) return 2;
        // Corners stay plain.
        if (flatMass(style, hh, h.iw + dj, h.jw + di, h.Lw, h.Dw) < 2.0 || flatMass(style, hh, h.iw - dj, h.jw - di, h.Lw, h.Dw) < 2.0) return 0;
        return k < 0.3 ? 1 : 0;
    }
    if (h.iw < 0 || h.jw < 0 || h.iw >= h.Lw || h.jw >= h.Dw) return 0;
    if (dj != 0) {
        if (!(dj < 0 ? h.jw == 0 : h.jw == h.Dw - 1) || h.iw == 0 || h.iw == h.Lw - 1) return 0;
        bool doorWall = (dj > 0) == ((hh & 4u) != 0u) && style != ST_CABIN;
        if (doorWall && h.iw == h.Lw / 2) return 2;
        if (((h.iw - h.Lw / 2) & 1) != 0) return 0;
        return k < (style == ST_CABIN ? 0.95 : 0.7) ? 1 : 0;
    }
    if (!(di < 0 ? h.iw == 0 : h.iw == h.Lw - 1)) return 0;
    // A cabin's front gable: the door between two windows.
    if (style == ST_CABIN) {
        if ((di > 0) != ((hh & 4u) != 0u)) return 0;
        int dr = abs(h.jw - (h.Dw - 1) / 2);
        return dr == 0 ? 2 : dr == 1 ? 1 : 0;
    }
    return h.jw == (h.Dw - 1) / 2 && k < 0.6 ? 1 : 0;
}

// Wall coordinates (iw along the ridge, jw across) to a column offset from the site centre.
static inline int2 wallCol(Part P, thread const HouseForm& f, int iw, int jw) {
    bool axisZ = (P.flags & F_AXISZ) != 0;
    int2 w0 = P.o + (axisZ ? int2(f.ovLo, f.ovGLo) : int2(f.ovGLo, f.ovLo));
    return w0 + (axisZ ? int2(jw, iw) : int2(iw, jw));
}

// The point just outside a dwelling's door (offset from the site centre, z 1 if it has one): where its lamp spills
// light, or with `path` where the path from the hearth arrives (an adobe's courtyard gate, past a porch).
static inline float3 houseDoor(Part P, uint sh, int k, bool path) {
    int style = (P.flags >> 12) & 15;
    if (P.kind == P_ROUND) {
        if (style != ST_YURT) return 0;
        float da = hashf(pcg(sh + uint(k))) * 6.2831853;
        return float3(float2(P.o) + 0.5 + float2(cos(da), sin(da)) * (float(P.sz.x) * 0.5 + 1.0), 1.0);
    }
    if (P.kind != P_HOUSE || (P.flags & F_TENT) != 0) return 0;
    uint hh = uint(P.y2);
    HouseForm f = houseForm(style, hh);
    bool axisZ = (P.flags & F_AXISZ) != 0;
    int4 ij = axisZ ? int4(0, 0, P.sz.y, P.sz.x) : int4(0, 0, P.sz.x, P.sz.y);
    int Lw = ij.z - f.ovGLo - f.ovGHi, Dw = ij.w - f.ovLo - f.ovHi;
    int iw = Lw / 2, jw, out;
    if (f.flat) {
        uint uPlan = (hh >> 7) & 1u;
        if (style == ST_ADOBE && !path) { jw = 2; out = 1; }
        else { jw = Dw - 1; out = 1; if (style == ST_ADOBE) iw = uPlan != 0u ? Lw / 2 : (Lw + 2) / 2; }
    } else if (style == ST_CABIN) {
        bool hi = (hh & 4u) != 0u;
        jw = (Dw - 1) / 2;
        int oi = (hi ? 1 : -1) * (path ? (hi ? f.ovGHi : f.ovGLo) + 1 : 1);
        return float3(float2(wallCol(P, f, (hi ? Lw - 1 : 0) + oi, jw)) + 0.5, 1.0);
    } else {
        bool hi = (hh & 4u) != 0u;
        jw = hi ? Dw - 1 : 0;
        int ov = hi ? f.ovHi : f.ovLo;
        out = (hi ? 1 : -1) * (path ? ov + 1 : 1);
        if (style == ST_STILT && path) return 0;
    }
    int2 cc = wallCol(P, f, iw, jw + out);
    return float3(float2(cc) + 0.5, 1.0);
}

static inline bool rectFree(thread const int4* rects, int nr, int2 r0, int2 sz, int gap) {
    for (int m = 0; m < nr; m++)
        if (all(r0 < rects[m].zw + gap) && all(r0 + sz + gap > rects[m].xy)) return false;
    return true;
}

// Lowest and highest ground over a rect; false where an arch stands in it.
static inline bool rectGround(texture2d<float, access::read> hm, int2 r0, int2 sz, thread float& lo, thread float& hi, thread int& wet) {
    lo = 1e9; hi = -1e9; wet = 0;
    for (int j = 0; j < sz.y; j++)
        for (int i = 0; i < sz.x; i++) {
            float4 col = hm.read(uint2(r0 + int2(i, j)));
            if (col.b > col.g) return false;
            lo = min(lo, col.r); hi = max(hi, col.r);
            if (col.r < SEA) wet++;
        }
    return true;
}
