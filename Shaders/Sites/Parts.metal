// Sites: one candidate per 128-voxel world cell, so the same places recur when the map recentres. Every part
// and lamp of a site lies within 128 of its centre, so a column only needs its own and the 8 neighbouring cells.
constant int SITE_CELL = 128;
constant int SITE_GRID = 18;
constant int MAX_PARTS = 32;
constant int MAX_LAMPS = 8;

// Must match MapLayout.siteStride in Sources/StrataCore/World/MapLayout.swift.
struct Site {
    int4 head;               // xy world centre, z type, w hash
    float4 info;             // x bounding radius, y age, z part count, w lamp count
    float4 lamps[MAX_LAMPS]; // xyz offset from the centre (y absolute), w kind + 16 * smoke rise + 1024 * smoke offset code
    int4 parts[MAX_PARTS];
};

constant int T_HENGE = 1, T_CAIRN = 2, T_RUINS = 3, T_HAMLET = 4, T_JETTY = 5, T_CAMP = 6, T_CAVE = 7;
constant int P_BOX = 1, P_HOUSE = 2, P_DECK = 3, P_BOAT = 4, P_HENGE = 5, P_WALL = 6, P_RING = 7, P_FIRE = 8, P_CLEAR = 9, P_ROUND = 10;
// Yard and camp props (y1 holds the type, y0 the ground they stand on).
constant int P_PROP = 11;
constant int PR_WOODPILE = 1, PR_HAY = 2, PR_GARDEN = 3, PR_WASH = 4, PR_OVEN = 5, PR_RACK = 6, PR_PEN = 7, PR_LOGSEAT = 8,
             PR_BEDROLL = 9, PR_RUG = 10, PR_CANOE = 11, PR_STUMP = 12, PR_PEAT = 13, PR_BALES = 14, PR_BARREL = 15;
// Part flags; bits 8-11 hold a lamp index + 1, bits 12-15 the dwelling style.
constant int F_REL = 1, F_AXISZ = 2, F_FLAT = 4, F_STILT = 8, F_TENT = 16, F_CHIMNEY = 32, F_WATER = 64;
constant int M_STONE = 4, M_TIMBER = 5, M_ROOF = 6, M_PLASTER = 7, M_PLANK = 8, M_CLOTH = 9, M_FIRE = 10;
constant int M_THATCH = 11, M_SHINGLE = 12, M_LOG = 13, M_FELT = 14, M_ADOBE = 15;
// Column features (structTex bits 24-27): pieces whose material differs from their part's.
constant int FT_VIGA = 1, FT_POST = 2, FT_DECK = 3, FT_CROWN = 4, FT_POLE = 5, FT_FENCE = 6, FT_CROP = 7;
// Dwelling styles by biome: hamlets 0-7, camps 8-12.
constant int ST_MEADOW = 0, ST_TIMBER = 1, ST_CABIN = 2, ST_CROFT = 3, ST_ADOBE = 4, ST_PUEBLO = 5, ST_STILT = 6, ST_YURT = 7;
constant int ST_AFRAME = 8, ST_LEANTO = 9, ST_TIPI = 10, ST_CARAVAN = 11, ST_PALM = 12;
constant int L_FIRE = 1, L_HEARTH = 2, L_WINDOW = 3, L_LANTERN = 4, L_BEACON = 5, L_RITUAL = 6, L_TENT = 7;

struct Part { int2 o; int2 sz; int kind; int mat; int y0; int y1; int y2; int flags; };

static inline int4 packPart(Part p) {
    return int4((p.o.x & 0xffff) | (p.o.y << 16),
                (p.sz.x & 0xff) | ((p.sz.y & 0xff) << 8) | ((p.kind & 0xff) << 16) | ((p.mat & 0xff) << 24),
                (p.y0 & 0xffff) | (p.y1 << 16), (p.y2 & 0xffff) | (p.flags << 16));
}

static inline Part unpackPart(int4 v) {
    Part p;
    p.o = int2((v.x << 16) >> 16, v.x >> 16);
    p.sz = int2(v.y & 0xff, (v.y >> 8) & 0xff);
    p.kind = (v.y >> 16) & 0xff;
    p.mat = (v.y >> 24) & 0xff;
    p.y0 = (v.z << 16) >> 16;
    p.y1 = v.z >> 16;
    p.y2 = (v.w << 16) >> 16;
    p.flags = (v.w >> 16) & 0xffff;
    return p;
}

static inline Part makePart(int kind, int mat, int2 o, int2 sz, int y0, int y1, int y2, int flags) {
    Part p;
    p.o = o; p.sz = sz; p.kind = kind; p.mat = mat; p.y0 = y0; p.y1 = y1; p.y2 = y2; p.flags = flags;
    return p;
}

// World cell of site-grid index 0 for a map whose texel 0 sits at world `origin`.
static inline int2 siteCell0(int2 origin) { return int2(floor(float2(origin) / float(SITE_CELL))); }

static inline float colH(texture2d<float, access::read> hm, int2 q) {
    int n = int(hm.get_width());
    return hm.read(uint2(clamp(q, int2(0), int2(n - 1)))).r;
}

static inline bool colArch(texture2d<float, access::read> hm, int2 q) {
    int n = int(hm.get_width());
    float4 c = hm.read(uint2(clamp(q, int2(0), int2(n - 1))));
    return c.b > c.g;
}

struct SiteGen { Site s; int np; int nl; float r; };

static inline void addPart(thread SiteGen& G, Part p, float reach) {
    if (G.np >= MAX_PARTS) return;
    G.s.parts[G.np++] = packPart(p);
    G.r = max(G.r, reach);
}

static inline int partReach(Part p) {
    return int(ceil(length(float2(max(abs(p.o), abs(p.o + p.sz))))));
}

// A window lamp stands at its door; its smoke rises from the chimney, `smoke` columns away (each within 15).
static inline int addLamp(thread SiteGen& G, float3 off, int kind, float rise, int2 smoke = int2(0)) {
    if (G.nl >= MAX_LAMPS) return -1;
    float code = rise > 0.0 && any(smoke != 0) ? float((clamp(smoke.x, -15, 15) + 16) + 32 * (clamp(smoke.y, -15, 15) + 16)) : 0.0;
    G.s.lamps[G.nl] = float4(off, float(kind) + 16.0 * floor(clamp(rise, 0.0, 63.0)) + 1024.0 * code);
    G.r = max(G.r, length(off.xz) + 2.0);
    return G.nl++;
}

static inline float lampRise(float w) { return fmod(floor(w / 16.0), 64.0); }
static inline float2 lampSmokeOff(float w) {
    float code = floor(w / 1024.0);
    return code > 0.0 ? float2(fmod(code, 32.0) - 16.0, floor(code / 32.0) - 16.0) : float2(0);
}

static inline bool flatPatch(texture2d<float, access::read> hm, int2 q, int rad, float tol, thread float& lo, thread float& hi) {
    lo = 1e9; hi = -1e9;
    for (int j = -rad; j <= rad; j++)
        for (int i = -rad; i <= rad; i++) {
            if (colArch(hm, q + int2(i, j))) return false;
            float h = colH(hm, q + int2(i, j));
            lo = min(lo, h); hi = max(hi, h);
        }
    return hi - lo <= tol;
}
