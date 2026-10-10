#include <metal_stdlib>
using namespace metal;

// Must match `Frame` in Sources/Renderer.swift.
struct Frame {
    float4 camPos;     // xyz map-local position, w tan(fovY/2)
    float4 camFwd;     // w aspect
    float4 camRight;   // w wrapped time (s)
    float4 camUp;      // w quality (0 battery, 1 full)
    float4 sunDir;     // xyz direction to sun, w unused
    float4 resolution; // xy pixels, zw inverse
    int4   mapInfo;    // xy world xz of texel 0, z size, w seed
    float4 misc;       // x exposure, y global max height, z wind angle, w cloud cover
    float4 misc2;      // x frame index, y fog density scale, z debug mode, w EDR headroom (0 = SDR)
    float4 prevPos;    // previous frame's camera, in this frame's map-local coords; w history valid
    float4 prevFwd;
    float4 prevRight;
    float4 prevUp;
    float4 jitter;     // xy subpixel jitter in NDC
};

constant float SEA = 19.85;
constant int MAX_LEVEL = 6;

// MARK: - Hash & noise

static inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline uint hash2i(int2 p, uint s) { return pcg(uint(p.x) * 1597334677u ^ pcg(uint(p.y) * 3812015801u ^ s)); }
static inline uint hash3i(int3 p, uint s) { return pcg(uint(p.x) * 1597334677u ^ pcg(uint(p.y) * 3812015801u ^ pcg(uint(p.z) * 2798796415u ^ s))); }
static inline float hashf(uint h) { return float(h & 0xffffffu) / 16777215.0; }

static inline float2 grad2(int2 c, uint s) {
    float a = hashf(hash2i(c, s)) * 6.2831853;
    return float2(cos(a), sin(a));
}

static inline float perlin2(float2 p, uint s) {
    float2 i = floor(p), f = p - i;
    int2 c = int2(i);
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = dot(grad2(c, s), f);
    float b = dot(grad2(c + int2(1, 0), s), f - float2(1, 0));
    float cc = dot(grad2(c + int2(0, 1), s), f - float2(0, 1));
    float d = dot(grad2(c + int2(1, 1), s), f - float2(1, 1));
    return mix(mix(a, b, u.x), mix(cc, d, u.x), u.y) * 1.414;
}

static inline float3 grad3(int3 c, uint s) {
    uint h = hash3i(c, s);
    float z = hashf(h) * 2.0 - 1.0;
    float a = hashf(pcg(h)) * 6.2831853;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(a), r * sin(a), z);
}

static inline float perlin3(float3 p, uint s) {
    float3 i = floor(p), f = p - i;
    int3 c = int3(i);
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = dot(grad3(c, s), f);
    float n100 = dot(grad3(c + int3(1, 0, 0), s), f - float3(1, 0, 0));
    float n010 = dot(grad3(c + int3(0, 1, 0), s), f - float3(0, 1, 0));
    float n110 = dot(grad3(c + int3(1, 1, 0), s), f - float3(1, 1, 0));
    float n001 = dot(grad3(c + int3(0, 0, 1), s), f - float3(0, 0, 1));
    float n101 = dot(grad3(c + int3(1, 0, 1), s), f - float3(1, 0, 1));
    float n011 = dot(grad3(c + int3(0, 1, 1), s), f - float3(0, 1, 1));
    float n111 = dot(grad3(c + int3(1, 1, 1), s), f - float3(1, 1, 1));
    float x0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    float x1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(x0, x1, u.z) * 1.15;
}

constant float2x2 OCT_ROT = float2x2(float2(0.8, 0.6), float2(-0.6, 0.8));

static inline float fbm2(float2 p, int oct, uint s) {
    float sum = 0, amp = 0.5, norm = 0;
    for (int i = 0; i < oct; i++) {
        sum += amp * perlin2(p, s + uint(i) * 1013u);
        norm += amp;
        p = OCT_ROT * p * 2.0 + 17.3;
        amp *= 0.5;
    }
    return sum / norm;
}

static inline float ridged2(float2 p, int oct, uint s) {
    float sum = 0, amp = 0.5, norm = 0, w = 1;
    for (int i = 0; i < oct; i++) {
        float n = 1.0 - abs(perlin2(p, s + uint(i) * 7919u));
        n *= n;
        sum += amp * n * w;
        w = saturate(n * 1.5);
        norm += amp;
        p = OCT_ROT * p * 2.0 + 31.7;
        amp *= 0.5;
    }
    return sum / norm;
}

// MARK: - World

struct Climate { float temp; float moist; };

static inline float2 seedOffset(uint seed) {
    return float2(hashf(pcg(seed)), hashf(pcg(seed ^ 0x9e3779b9u))) * 50000.0;
}

static inline Climate climate(float2 w, uint seed) {
    float2 p = w + seedOffset(seed);
    Climate c;
    c.temp = saturate(fbm2(p / 2000.0, 2, seed + 11) * 1.6 + 0.5);
    c.moist = saturate(fbm2(p / 1800.0 + 91.0, 2, seed + 12) * 1.6 + 0.5);
    return c;
}

static inline float desertAmount(Climate c) {
    return smoothstep(0.45, 0.68, c.temp) * smoothstep(0.7, 0.4, c.moist);
}

// Column = (height, gapLo, gapHi): solid where y < height, except inside [gapLo, gapHi).
static inline float3 terrainColumn(int2 w, uint seed) {
    float2 p = float2(w) + seedOffset(seed);
    float2 warp = float2(fbm2(p / 900.0, 2, seed + 1), fbm2(p / 900.0 + 37.1, 2, seed + 2)) * 300.0;
    float c = saturate(fbm2((p + warp) / 2000.0, 3, seed + 3) * 1.4 + 0.5);
    float hi = smoothstep(0.35, 0.75, c);
    float rough = fbm2((p + warp * 0.3) / 350.0, 3, seed + 4) * 0.5 + 0.5;
    float ridge = ridged2((p + warp * 0.2) / 520.0, 3, seed + 5);
    float r = mix(rough, ridge, hi);
    float h = mix(16.0, 62.0, c) + mix(26.0, 115.0, hi) * r;
    // Crenellated ridgelines instead of soft domes.
    h += (ridged2((p + warp * 0.4) / 110.0, 3, seed + 14) - 0.45) * mix(0.0, 8.0, smoothstep(0.3, 0.7, hi));
    // Soft terracing everywhere: risers cluster into short cliff bands between wide treads,
    // so slopes read as contour lines instead of a 1:1 stair lattice.
    float td = mix(4.0, 8.0, saturate(perlin2(p / 900.0, seed + 13) * 0.8 + 0.5));
    float hn2 = h / td;
    float lv = floor(hn2);
    h = (lv + smoothstep(mix(0.55, 0.9, hi), 1.0, hn2 - lv)) * td;

    // Canyon country: in hot, dry regions the land steps into flat-topped mesas with sheer risers.
    float canyon = smoothstep(0.35, 0.7, desertAmount(climate(float2(w), seed)));
    float tier = 13.0 + 4.0 * perlin2(p / 700.0, seed + 7);
    float hn = h / tier;
    float level = floor(hn);
    float mesa = (level + smoothstep(0.62, 0.9, hn - level)) * tier;
    h = mix(h, mesa, canyon);

    // Long meandering valleys with flat water floors: the camera follows these.
    float river = abs(fbm2((p + warp * 0.6) / 1400.0, 3, seed + 6));
    float valley = smoothstep(0.02, 0.30 + 0.10 * hi, river);
    h = floor(max(mix(SEA - 3.0, h, valley), 2.0));
    // Lake beds are one flat shelf: submerged terraces read as stepping stones under the surface.
    if (h < SEA) h = floor(SEA) - 2.0;

    // Arches and undercuts: noise blobs hollow out a band low in each mesa tier.
    float gl = 0, gh = 0;
    if (canyon > 0.2) {
        float blob = fbm2(p / 34.0, 3, seed + 8);
        float g = (blob - 0.2) * canyon;
        if (g > 0.0) {
            float base = max(level * tier - tier, SEA + 1.0);
            gl = floor(base + 2.0 + 2.0 * perlin2(p / 400.0, seed + 10));
            gh = gl + floor(min(g * 40.0, tier * 0.8));
            if (gh > h - 3.0 || gh - gl < 3.0 || gl < SEA + 1.0) { gl = 0; gh = 0; }
        }
    }
    return float3(h, gl, gh);
}

kernel void generateHeight(texture2d<float, access::write> out [[texture(0)]],
                           constant int4& info [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    int2 w = int2(gid) + info.xy;
    out.write(float4(terrainColumn(w, uint(info.w)), 0), gid);
}

// Clamps each column into its 4-neighbour height range: removes single-voxel spikes and pits.
kernel void cleanupColumns(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 c = src.read(gid);
    uint2 a = uint2(max(int(gid.x) - 1, 0), gid.y), b = uint2(min(gid.x + 1, n - 1), gid.y);
    uint2 e = uint2(gid.x, max(int(gid.y) - 1, 0)), f = uint2(gid.x, min(gid.y + 1, n - 1));
    float h0 = src.read(a).r, h1 = src.read(b).r, h2 = src.read(e).r, h3 = src.read(f).r;
    c.r = clamp(c.r, min(min(h0, h1), min(h2, h3)), max(max(h0, h1), max(h2, h3)));
    if (c.b > c.r - 3.0) { c.g = 0; c.b = 0; }
    // A roof needs walls: keep a gap only where every neighbour reaches above it.
    if (min(min(h0, h1), min(h2, h3)) < c.b + 1.0) { c.g = 0; c.b = 0; }
    dst.write(c, gid);
}

// MARK: - Vegetation

// Trees sit on a jittered world-space grid, one candidate per cell; a canopy reaches at most 2 columns
// from its trunk, so a column only needs the trees of its own and the 8 neighbouring cells.
constant int TREE_CELL = 5;

static inline float coniferAmount(Climate c) { return smoothstep(0.50, 0.32, c.temp); }

// Grove density: a low-frequency Perlin field gated by climate; zero on sand, snow, desert and water.
static inline float groveDensity(float2 w, float g, uint seed) {
    float2 p = w + seedOffset(seed);
    Climate c = climate(w, seed);
    float grove = fbm2(p / 300.0, 3, seed + 90u) + 0.3 * perlin2(p / 70.0, seed + 91u);
    float open = 1.0 - smoothstep(0.2, 0.5, desertAmount(c));
    float wet = smoothstep(0.25, 0.6, c.moist) * open;
    float snowline = mix(160.0, 92.0, smoothstep(0.42, 0.18, c.temp));
    float band = smoothstep(SEA + 2.0, SEA + 6.0, g) * smoothstep(snowline - 4.0, snowline - 30.0, g);
    float dense = smoothstep(0.12, 0.34, grove + 0.35 * (wet - 0.5));
    // Lone trees dot even dry grassland; groves need moisture.
    return (0.025 * open + 0.55 * dense * wet) * band;
}

struct Tree { bool ok; int2 c; float g; int lo; int top; int kind; };

static Tree treeInCell(texture2d<float, access::read> src, int2 cell, int2 origin, uint seed) {
    Tree t;
    t.ok = false;
    uint h = hash2i(cell, seed + 92u);
    t.c = cell * TREE_CELL + int2(h % uint(TREE_CELL), (h / 7u) % uint(TREE_CELL)) - origin;
    int n = int(src.get_width());
    if (any(t.c < 1) || any(t.c >= n - 1)) return t;
    float4 col = src.read(uint2(t.c));
    t.g = col.r;
    if (col.b > col.g) return t;
    float g0 = src.read(uint2(t.c + int2(1, 0))).r, g1 = src.read(uint2(t.c - int2(1, 0))).r;
    float g2 = src.read(uint2(t.c + int2(0, 1))).r, g3 = src.read(uint2(t.c - int2(0, 1))).r;
    if (max(max(abs(g0 - t.g), abs(g1 - t.g)), max(abs(g2 - t.g), abs(g3 - t.g))) > 1.0) return t;
    float2 w = float2(t.c + origin);
    if (hashf(pcg(h)) > groveDensity(w, t.g, seed)) return t;
    uint r = pcg(h + 5u);
    Climate c = climate(w, seed);
    // 0 shrub, 1 small round, 2 round, 3 conifer.
    float dense = groveDensity(w, t.g, seed);
    if (hashf(r) < coniferAmount(c)) t.kind = 3;
    else if (dense < 0.15) t.kind = int(hashf(pcg(r)) * 3.0);
    else t.kind = hashf(pcg(r)) < 0.3 ? 1 : 2;
    int g = int(t.g);
    if (t.kind == 0) { t.lo = g; t.top = g + 1 + int(r >> 30 & 1u); }
    else if (t.kind == 1) { t.lo = g + 2 + int(r >> 29 & 1u); t.top = t.lo + 3; }
    else if (t.kind == 2) { t.lo = g + 2 + int(r >> 29 & 1u); t.top = t.lo + 4; }
    else { t.lo = g + 1; t.top = g + 6 + int(r >> 28 & 3u); }
    t.ok = true;
    return t;
}

// Leaf interval [lo, hi) a tree contributes to the column at offset o from its trunk; empty if lo >= hi.
static inline int2 canopySpan(Tree t, int2 o) {
    int d2 = o.x * o.x + o.y * o.y;
    if (t.kind == 0) return d2 == 0 ? int2(t.lo, t.top) : d2 == 1 ? int2(t.lo, t.top - 1) : int2(0);
    if (t.kind == 1) return d2 <= 1 ? int2(t.lo, t.top) : d2 == 2 ? int2(t.lo + 1, t.top - 1) : int2(0);
    if (t.kind == 2) return d2 <= 2 ? int2(t.lo, t.top) : d2 <= 5 ? int2(t.lo + 1, t.top - 1) : int2(0);
    // Conifer: a stepped cone, widest a little above the ground.
    int h = t.top - t.lo;
    if (d2 == 0) return int2(t.lo + 1, t.top);
    if (d2 == 1) return int2(t.lo + 1, t.top - 2);
    if (d2 == 2) return int2(t.lo + 1, t.lo + max(h - 4, 1));
    if (d2 == 4 && h >= 7) return int2(t.lo + 1, t.lo + 2);
    return int2(0);
}

// Runs after cleanupColumns, which would clamp one-column trunks away. Tree columns set w = first leaf
// voxel; g then holds the ground height (b = g on trunks and buried canopies, b = w under a floating canopy).
kernel void plantTrees(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       constant int4& info [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint n = src.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float4 col = src.read(gid);
    col.a = 0;
    if (col.b > col.g || col.r < SEA + 2.0) { dst.write(col, gid); return; }
    int2 p = int2(gid);
    int2 wc = p + info.xy;
    int2 home = int2(floor(float2(wc) / float(TREE_CELL)));
    int lo = 100000, hi = 0;
    bool trunk = false;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            Tree t = treeInCell(src, home + int2(i, j), info.xy, uint(info.w));
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
    float G = col.r;
    if (float(hi) <= G) { dst.write(col, gid); return; }
    float leaf = max(float(lo), G);
    bool floating = !trunk && leaf > G;
    dst.write(float4(float(hi), G, floating ? leaf : G, leaf), gid);
}

kernel void reduceMax(texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2;
    float m = max(max(src.read(s).r, src.read(s + uint2(1, 0)).r), max(src.read(s + uint2(0, 1)).r, src.read(s + uint2(1, 1)).r));
    dst.write(float4(m, 0, 0, 0), gid);
}

// MARK: - Traversal

struct Hit {
    bool hit;
    float t;
    int3 voxel;
    int3 normal;
    int iters;
};

// Terrain height under any tree.
static inline float groundAt(texture2d<float> hm, int2 c, int size) {
    c = clamp(c, int2(0), int2(size - 1));
    float4 col = hm.read(uint2(c), 0);
    return col.a > 0.0 ? col.g : col.r;
}

// Ground bilinearly interpolated between column centres: the continuous slope under the stairs.
static inline float groundSmooth(texture2d<float> hm, float2 p, int size) {
    float2 q = p - 0.5, f = fract(q);
    int2 i = int2(floor(q));
    return mix(mix(groundAt(hm, i, size), groundAt(hm, i + int2(1, 0), size), f.x),
               mix(groundAt(hm, i + int2(0, 1), size), groundAt(hm, i + int2(1, 1), size), f.x), f.y);
}

static inline float3 columnAt(texture2d<float> hm, int2 c, int size) {
    c = clamp(c, int2(0), int2(size - 1));
    return hm.read(uint2(c), 0).rgb;
}

static inline bool solid(texture2d<float> hm, int3 v, int size) {
    float3 col = columnAt(hm, v.xz, size);
    float y = float(v.y);
    return y < col.x && !(y >= col.y && y < col.z);
}

// Max-mip quadtree walk over the column heightfield; columns are stacks of unit voxels.
static Hit trace(texture2d<float> hm, float3 o, float3 d, float tMax, int maxIter, float globalMax, int size) {
    Hit h;
    h.hit = false;
    h.t = tMax;
    h.iters = -1;
    float3 inv = 1.0 / select(d, copysign(float3(1e-6), d), abs(d) < 1e-6);

    // Clip to the map box in xz and to [0, globalMax] in y.
    float3 lo = float3(0, -1, 0), hi = float3(size, globalMax, size);
    float3 t0 = (lo - o) * inv, t1 = (hi - o) * inv;
    float3 tn = min(t0, t1), tf = max(t0, t1);
    float tEnter = max(max(tn.x, tn.y), max(tn.z, 0.0));
    float tExit = min(min(tf.x, tf.y), min(tf.z, tMax));
    if (tEnter >= tExit) return h;

    float t = tEnter;
    int level = MAX_LEVEL;
    int lastAxis = (tn.x > tn.z) ? 0 : 2;
    // Spatial nudge across the boundary just crossed; a t-epsilon vanishes at large map coords.
    float2 nudge = float2(0);
    for (int i = 0; i < maxIter; i++) {
        h.iters = i;
        if (t >= tExit) return h;
        float3 p = o + d * t;
        int sz = 1 << level;
        int2 cell = int2(floor((p.xz + nudge) / float(sz)));
        float2 cLo = float2(cell * sz), cHi = cLo + float(sz);
        float2 tb = (select(cLo, cHi, d.xz > 0.0) - o.xz) * inv.xz;
        float tCell = min(min(tb.x, tb.y), tExit);
        float hmax = hm.read(uint2(clamp(cell, int2(0), int2((size >> level) - 1))), level).r;
        float yA = o.y + d.y * t, yB = o.y + d.y * tCell;
        if (min(yA, yB) >= hmax) {
            lastAxis = tb.x < tb.y ? 0 : 2;
            nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
            t = tCell;
            level = min(level + 1, MAX_LEVEL);
            continue;
        }
        if (level > 0) { level--; continue; }

        // Level 0: one column, solid in [0, gl) and [gh, H).
        float3 col = hm.read(uint2(clamp(cell, int2(0), int2(size - 1))), 0).rgb;
        float H = col.x, gl = col.y, gh = col.z;
        bool inGap = yA >= gl && yA < gh;
        if (yA < H && !inGap) {
            h.hit = true;
            h.t = t;
            h.normal = lastAxis == 0 ? int3(d.x > 0 ? -1 : 1, 0, 0) : int3(0, 0, d.z > 0 ? -1 : 1);
            h.voxel = int3(cell.x, int(floor(yA)), cell.y);
            return h;
        }
        float floorY = inGap ? gl : H;
        if (d.y < 0 && yB < floorY) {
            h.hit = true;
            h.t = (floorY - o.y) * inv.y;
            h.normal = int3(0, 1, 0);
            h.voxel = int3(cell.x, int(floorY) - 1, cell.y);
            return h;
        }
        if (inGap && d.y > 0 && yB > gh) {
            h.hit = true;
            h.t = (gh - o.y) * inv.y;
            h.normal = int3(0, -1, 0);
            h.voxel = int3(cell.x, int(gh), cell.y);
            return h;
        }
        lastAxis = tb.x < tb.y ? 0 : 2;
        nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
        t = tCell;
        level = 1;
    }
    return h;
}

// MARK: - Fullscreen

struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

// MARK: - Sky & light

struct Light {
    float3 key;     // direction to the current key light (sun, afterglow or moon)
    float3 keyCol;
    bool shadows;
    float3 zenith;   // terrain/cloud fill
    float3 horizon;
    float3 skyZen;   // sky radiance, kept darker and more saturated than the fill so twilight skies hold colour
    float3 skyHor;
    float3 sun;
    float3 moon;
    float sunAmt;
    float night;
};

// Keyframes by sun elevation: night, blue hour, sunrise, golden, midday.
static inline float3 key5(float e, float3 a, float3 b, float3 c, float3 d, float3 f) {
    float k0 = -0.30, k1 = -0.10, k2 = 0.0, k3 = 0.18, k4 = 0.32;
    if (e < k1) return mix(a, b, smoothstep(k0, k1, e));
    if (e < k2) return mix(b, c, smoothstep(k1, k2, e));
    if (e < k3) return mix(c, d, smoothstep(k2, k3, e));
    return mix(d, f, smoothstep(k3, k4, e));
}

static Light makeLight(float3 sunDir) {
    Light L;
    float e = sunDir.y;
    L.sun = sunDir;
    L.moon = normalize(float3(-sunDir.x, max(-sunDir.y, 0.08), -sunDir.z));
    L.zenith = key5(e, float3(0.018, 0.022, 0.045), float3(0.10, 0.10, 0.22), float3(0.13, 0.16, 0.32),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    L.horizon = key5(e, float3(0.022, 0.026, 0.045), float3(0.30, 0.24, 0.36), float3(0.80, 0.45, 0.28),
                     float3(0.95, 0.60, 0.32), float3(0.46, 0.62, 0.86));
    L.skyHor = key5(e, float3(0.022, 0.026, 0.045), float3(0.16, 0.10, 0.22), float3(0.50, 0.22, 0.09),
                    float3(0.80, 0.48, 0.22), float3(0.50, 0.66, 0.90));
    L.skyZen = key5(e, float3(0.010, 0.013, 0.028), float3(0.03, 0.04, 0.12), float3(0.07, 0.09, 0.24),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    float3 sunCol = key5(e, float3(0), float3(0), float3(1.8, 0.6, 0.22), float3(3.0, 1.45, 0.55), float3(3.0, 2.7, 2.2));
    L.sunAmt = smoothstep(-0.02, 0.05, e);
    L.night = smoothstep(-0.04, -0.20, e);
    if (e > -0.02) {
        L.key = sunDir;
        L.keyCol = sunCol * L.sunAmt + float3(1.3, 0.5, 0.3) * 0.4 * (1.0 - L.sunAmt);
        L.shadows = true;
    } else {
        // Below the horizon an afterglow from the sun's azimuth hands over to the moon.
        float3 glowDir = normalize(float3(sunDir.x, 0.05, sunDir.z));
        float3 glowCol = float3(1.3, 0.5, 0.3) * 0.4 * smoothstep(-0.20, -0.02, e);
        float m = smoothstep(-0.09, -0.16, e);
        L.key = normalize(mix(glowDir, L.moon, m));
        L.keyCol = mix(glowCol, float3(0.60, 0.66, 0.80) * L.night, m);
        L.shadows = m > 0.5;
    }
    return L;
}

static inline float hg(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (12.566 * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

static float3 skyColor(float3 d, Light L, bool withDiscs, uint seed, float time) {
    float up = saturate(d.y);
    float lowSun = 1.0 - smoothstep(0.0, 0.35, abs(L.sun.y));
    float3 hor = L.skyHor;
    // Low sun: gold toward the sun, a cooler pink-lavender band opposite it.
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(L.sun.xz + 1e-5);
    hor *= mix(float3(1.0), mix(float3(0.75, 0.6, 0.85), float3(1.0), 0.5 + 0.5 * dot(dh, sh2)), lowSun);
    float3 c = mix(hor, L.skyZen, 1.0 - exp(-up * mix(5.0, 7.0, saturate(L.sun.y / 0.4))));
    if (d.y < 0) c = hor * mix(1.0, 0.75, saturate(-d.y * 4));
    float s = max(dot(d, L.sun), 0.0);
    float3 glowCol = key5(L.sun.y, float3(0), float3(0.45, 0.18, 0.12), float3(1.2, 0.40, 0.12), float3(1.0, 0.65, 0.35), float3(0.5, 0.45, 0.4));
    c += glowCol * (pow(s, 6.0) * 0.35 + pow(s, 64.0) * 0.6) * (1.0 - 0.5 * up);
    if (!withDiscs) return c;
    // Disc takes the key light's hue (amber near the horizon) and blooms into the haze.
    float3 tint = L.keyCol / max(max(L.keyCol.r, max(L.keyCol.g, L.keyCol.b)), 1e-3);
    c += tint * 12.0 * smoothstep(0.99955, 0.9998, s) * L.sunAmt;
    c += L.keyCol * (pow(s, 1200.0) * 0.8 + pow(s, 180.0) * 0.25) * L.sunAmt;

    if (L.night > 0.0) {
        float md = max(dot(d, L.moon), 0.0);
        c += float3(0.10, 0.13, 0.22) * (pow(md, 24.0) * 0.6 + pow(md, 400.0) * 1.5) * L.night;
        // Stars: one hashed point per cell, thinned by low-frequency noise into clusters and voids and
        // denser along a band (a galactic plane at a seed-dependent tilt).
        float3 q = d * 260.0;
        int3 cell = int3(floor(q));
        uint h = hash3i(cell, seed + 77u);
        float3 gp = normalize(float3(0.6, 0.5, 0.3 + hashf(pcg(seed + 3u)) * 0.6));
        float bd = dot(d, gp);
        float band = exp(-pow(bd / 0.18, 2.0));
        // Milky Way: a wide glow and a brighter core built from layered noise, a fine grain of unresolved
        // stars, a meandering dark rift and ragged dust clouds.
        float n1 = perlin3(d * 3.0, seed + 80u), n2 = perlin3(d * 9.0, seed + 81u);
        float n3 = perlin3(d * 24.0, seed + 82u), n4 = perlin3(d * 60.0, seed + 85u);
        float wob = bd + 0.05 * n1;
        float core = exp(-pow(wob / 0.08, 2.0));
        float glow = (exp(-pow(wob / 0.22, 2.0)) * 0.45 + core) * saturate(0.7 + 0.5 * n1 + 0.4 * n2 + 0.3 * n3 + 0.2 * n4);
        float rift = exp(-pow((wob - 0.015 - 0.025 * n2 - 0.01 * n3) / 0.028, 2.0)) * smoothstep(-0.3, 0.2, n1 + 0.5 * n2);
        float murk = smoothstep(0.05, 0.35, n2 + 0.6 * n3 + 0.3 * n4) * exp(-pow(wob / 0.14, 2.0));
        float dust = saturate(rift * 0.6 + murk * 0.4);
        float3 mwCol = mix(float3(0.55, 0.62, 0.90), float3(0.95, 0.88, 0.78), saturate(core * (0.5 + 0.8 * n2)));
        float horizonFade = smoothstep(0.0, 0.12, d.y);
        float mw = glow * (1.0 - dust) * L.night * horizonFade;
        c += mwCol * mw * 0.07;
        if (mw > 0.02) {
            float3 fq = d * 500.0;
            uint fh = hash3i(int3(floor(fq)), seed + 86u);
            float3 fsp = floor(fq) + float3(hashf(pcg(fh)), hashf(pcg(fh + 1)), hashf(pcg(fh + 2)));
            c += mwCol * 0.25 * mw * step(0.75, hashf(pcg(fh + 3))) * smoothstep(0.75, 0.15, length(fq - fsp));
        }
        float clump = saturate(perlin3(d * 3.0, seed + 78u) * 0.9 + 0.5);
        float density = mix(0.004, 0.035, saturate(clump * 0.8 + band * 0.7 + glow * 0.6));
        if (hashf(h) > 1.0 - density) {
            float3 sp = float3(cell) + float3(hashf(pcg(h)), hashf(pcg(h + 1)), hashf(pcg(h + 2)));
            float dd = length(q - sp);
            float b = pow(hashf(pcg(h + 3)), 9.0) * 7.0 + 0.12;
            float twinkle = 0.7 + 0.3 * sin(time * 1.3 + hashf(pcg(h + 4)) * 40.0);
            c += float3(0.9, 0.95, 1.1) * b * twinkle * smoothstep(0.55, 0.0, dd) * L.night * smoothstep(0.0, 0.05, d.y) * (1.0 - 0.8 * dust);
        }
        // Moon: a clean disc whose maria are Perlin noise.
        if (md > 0.99) {
            float3 mu = normalize(cross(L.moon, float3(0, 1, 0)));
            float3 mv = cross(mu, L.moon);
            float2 xy = float2(dot(d, mu), dot(d, mv)) / 0.03;
            float disc = 1.0 - smoothstep(0.96, 1.0, length(xy));
            float lum = 0.92 - 0.3 * smoothstep(0.15, 0.55, fbm2(xy * 1.3 + 3.0, 3, 9u) + 0.25);
            c = mix(c, float3(0.75, 0.80, 0.92) * 2.4 * lum, L.night * disc);
        }
    }
    return c;
}

// MARK: - Surface

struct Surface {
    float3 albedo;
    float3 tread;   // the column's top colour, used to prefilter risers at distance
    float glint;    // step-lip highlight weight
    float4 nh;      // neighbour heights: -x, +x, -z, +z
    float2 lipDir;  // outward direction of the edge that glints
    float H;
    int mat;        // 0 terrain, 1 leaf, 2 wood
    float canopy;   // ambient left under a canopy overhead
};

// Two low-chroma tones per rock family, ramped smoothly by voxel row so strata read as gentle folds.
static inline float3 strataTone(float y, float2 w, uint seed, float desertAmt) {
    float k = 0.5 + 0.5 * sin(y * 0.38 + 2.2 * perlin2(w * 0.01, seed + 22u) + 1.3 * perlin2(w * 0.003, seed + 23u));
    k = smoothstep(0.25, 0.75, k);
    float3 grey = mix(float3(0.22, 0.23, 0.25), float3(0.35, 0.34, 0.33), k);
    float3 sand = mix(float3(0.56, 0.36, 0.24), float3(0.70, 0.52, 0.37), k);
    return mix(grey, sand, desertAmt);
}

static Surface surfaceAt(texture2d<float> hm, int size, int3 v, int3 n, float2 uv, float fp, int2 mapOrigin, uint seed) {
    Surface s;
    s.glint = 0;
    s.lipDir = float2(float(n.x), float(n.z));
    s.mat = 0;
    float4 col4 = hm.read(uint2(clamp(v.xz, int2(0), int2(size - 1))), 0);
    float3 col = col4.rgb;
    bool tree = col4.a > 0.0;
    float H = tree ? col.y : col.x;
    s.canopy = tree ? 0.55 : 1.0;
    s.H = H;
    float hx0 = groundAt(hm, v.xz + int2(-1, 0), size), hx1 = groundAt(hm, v.xz + int2(1, 0), size);
    float hz0 = groundAt(hm, v.xz + int2(0, -1), size), hz1 = groundAt(hm, v.xz + int2(0, 1), size);
    s.nh = float4(hx0, hx1, hz0, hz1);
    float slope = max(max(H - hx0, H - hx1), max(H - hz0, H - hz1));
    float2 w = float2(v.xz + mapOrigin);
    Climate c = climate(w, seed);
    float desert = desertAmount(c);
    float cold = smoothstep(0.42, 0.18, c.temp);
    int depth = int(H) - 1 - v.y;
    bool underGap = !tree && col.z > col.y && float(v.y) < col.z;

    float snowline = mix(160.0, 92.0, cold) + 5.0 * perlin2(w / 160.0, seed + 21u);
    float3 stone = strataTone(float(v.y), w, seed, desert);

    float3 grassWet = float3(0.12, 0.20, 0.13), grassDry = float3(0.40, 0.34, 0.16), grassCold = float3(0.20, 0.23, 0.17);
    float3 grass = mix(mix(grassDry, grassWet, smoothstep(0.3, 0.7, c.moist)), grassCold, cold);
    float3 sand = float3(0.47, 0.41, 0.29);
    float3 desertTop = float3(0.54, 0.34, 0.19);
    float3 tread = mix(grass, desertTop, desert) * (0.95 + 0.1 * perlin2(w / 40.0, seed + 30u));
    // Gentle hypsometric tint, like a printed topo map: sage valleys, paler ochre-grey uplands.
    float alt = saturate((H - SEA) / 110.0);
    tread *= mix(float3(0.93, 1.03, 0.90), float3(1.10, 1.04, 0.98), alt);
    tread = mix(tread, stone * 1.35, 0.30 * smoothstep(0.55, 1.0, alt));
    // Hypsometric banding, as on a printed topo map: each 6-voxel elevation band steps its tint slightly,
    // so broad hillsides read as layered contour bands rather than one repeated stair.
    float bandK = fract(floor(H / 6.0) * 0.618034);
    tread *= mix(float3(1.12, 1.04, 0.90), float3(0.90, 0.98, 1.10), bandK);
    s.tread = tread;
    int drops = int(H - hx0 >= 3.0) + int(H - hx1 >= 3.0) + int(H - hz0 >= 3.0) + int(H - hz1 >= 3.0);
    bool beach = H <= SEA + 1.2;
    if (beach) tread = sand;
    s.tread = tread;
    bool snowy = float(int(H)) > snowline;

    if (n.y > 0 && !underGap) {
        if (float(v.y + 1) > snowline) s.albedo = float3(0.63, 0.66, 0.71);
        else if (beach) s.albedo = sand;
        else if (drops >= 2) s.albedo = stone;
        else s.albedo = tread;
        // Step-lip glint along edges that drop away, at least a pixel wide at any distance.
        float edge = 9.0;
        if (hx0 < H && uv.x < edge) { edge = uv.x; s.lipDir = float2(-1, 0); }
        if (hx1 < H && 1.0 - uv.x < edge) { edge = 1.0 - uv.x; s.lipDir = float2(1, 0); }
        if (hz0 < H && uv.y < edge) { edge = uv.y; s.lipDir = float2(0, -1); }
        if (hz1 < H && 1.0 - uv.y < edge) { edge = 1.0 - uv.y; s.lipDir = float2(0, 1); }
        s.glint = edge > 8.0 ? 0.0 : 1.0 - smoothstep(0.0, min(max(1.0 / 16.0, 1.5 * fp), 0.4), edge);
    } else if (n.y == 0 && !underGap) {
        if (beach) s.albedo = sand * 0.85;
        else if (slope < 4.0 && !snowy) s.albedo = mix(tread, stone, 0.5) * 0.9;
        else s.albedo = snowy ? mix(stone, float3(0.63, 0.66, 0.71), 0.45) : stone;
        if (depth == 0) s.glint = 0.5 * (1.0 - smoothstep(0.0, min(max(1.0 / 16.0, 1.5 * fp), 0.4), 1.0 - uv.y));
    } else {
        s.albedo = stone * 0.9;
    }
    if (tree && float(v.y) >= col4.a) {
        // Leaves: deep, low-chroma greens that drift slowly across a grove; conifers cooler and darker.
        float conifer = smoothstep(0.50, 0.32, c.temp);
        float k = saturate(perlin2(w / 23.0, seed + 93u) * 0.9 + 0.5);
        float3 leaf = mix(float3(0.035, 0.07, 0.035), float3(0.065, 0.09, 0.04), k);
        leaf = mix(leaf, float3(0.025, 0.055, 0.045), conifer);
        // A 16x16 leaf texture: two-tone texels, faded out once a texel nears a pixel.
        int2 tx = int2(floor(uv * 16.0));
        int3 tk = n.y != 0 ? int3(v.x * 16 + tx.x, v.y * 2 + max(n.y, 0), v.z * 16 + tx.y)
                : n.x != 0 ? int3(v.x * 2 + max(n.x, 0), v.y * 16 + tx.y, v.z * 16 + tx.x)
                           : int3(v.x * 16 + tx.x, v.y * 16 + tx.y, v.z * 2 + max(n.z, 0));
        uint th = hash3i(tk, seed + 94u);
        float tex = hashf(th) < 0.35 ? 0.72 : 1.0 + 0.12 * hashf(pcg(th));
        leaf *= mix(tex, 0.92, smoothstep(0.015, 0.04, fp));
        // Upper leaves catch more sky.
        leaf *= mix(0.85, 1.15, saturate((float(v.y) - col4.a) / max(col.x - col4.a - 1.0, 1.0)));
        s.albedo = leaf;
        s.tread = leaf;
        s.glint = 0;
        s.H = col.x;
        s.mat = 1;
        s.canopy = 1.0;
    } else if (tree && float(v.y) >= col.y) {
        s.albedo = float3(0.06, 0.058, 0.052);
        s.tread = s.albedo;
        s.glint = 0;
        s.mat = 2;
        s.canopy = 1.0;
    }
    return s;
}

static inline float vao(float s1, float s2, float c) { return (s1 > 0.5 && s2 > 0.5) ? 0.0 : 3.0 - (s1 + s2 + c); }

// Minecraft-style per-corner AO, bilinearly interpolated across the face.
static float cornerAO(texture2d<float> hm, int size, int3 v, int3 n, float2 uv) {
    int3 a = v + n;
    int3 tu = n.x != 0 ? int3(0, 0, 1) : int3(1, 0, 0);
    int3 tv = n.y != 0 ? int3(0, 0, 1) : int3(0, 1, 0);
    float su0 = solid(hm, a - tu, size), su1 = solid(hm, a + tu, size);
    float sv0 = solid(hm, a - tv, size), sv1 = solid(hm, a + tv, size);
    float c00 = solid(hm, a - tu - tv, size), c10 = solid(hm, a + tu - tv, size);
    float c01 = solid(hm, a - tu + tv, size), c11 = solid(hm, a + tu + tv, size);
    float a00 = vao(su0, sv0, c00), a10 = vao(su1, sv0, c10), a01 = vao(su0, sv1, c01), a11 = vao(su1, sv1, c11);
    float ao = mix(mix(a00, a10, uv.x), mix(a01, a11, uv.x), uv.y) / 3.0;
    return mix(0.6, 1.0, pow(ao, 0.8));
}

// Fraction of open sky above a point, from how far the surrounding columns rise above y within 6 voxels.
// Heights are filtered, so a c on a column centre reads exact columns.
static float skyVisibility(texture2d<float> hm, float2 c, float y) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    float2 inv = 1.0 / float2(hm.get_width(), hm.get_height());
    float occ = 0.0;
    const int2 dirs[8] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1) };
    for (int i = 0; i < 8; i++) {
        float m = 0.0;
        for (int r = 1; r <= 3; r++) {
            float dist = float(r * 2) * (i < 4 ? 1.0 : 1.414);
            float rise = hm.sample(bl, (c + float2(dirs[i] * (r * 2))) * inv, level(0)).r - y;
            m = max(m, rise / sqrt(rise * rise + dist * dist));
        }
        occ += m;
    }
    return 1.0 - occ / 8.0;
}

static inline float2 faceUV(float3 p, int3 n) {
    float3 f = fract(p);
    if (n.y != 0) return f.xz;
    if (n.x != 0) return float2(f.z, f.y);
    return float2(f.x, f.y);
}

// MARK: - Clouds

constant float CLOUD_Y = 240.0; // above the tallest peaks

static inline float cloudDensity(float2 q, uint seed, float time, float cover) {
    q -= float2(cos(0.6), sin(0.6)) * time * 1.6;
    return fbm2(q / 300.0, 3, seed + 40u) + perlin3(float3(q / 520.0, time * 0.004), seed + 41u) * 0.35 + cover - 0.5;
}

static inline float cloudShadow(float3 p, float3 L, int2 mapOrigin, uint seed, float time, float cover) {
    if (L.y <= 0.02) return 1.0;
    float2 q = p.xz + L.xz / L.y * (CLOUD_Y - p.y) + float2(mapOrigin);
    return mix(1.0, mix(0.6, 0.32, smoothstep(0.3, 0.5, L.y)), smoothstep(0.0, 0.14, cloudDensity(q, seed, time, cover)));
}

struct Ctx {
    texture2d<float> hm;
    int size;
    int2 origin;
    uint seed;
    float time;
    float quality;
    float globalMax;
    float pixelAngle;
    float cover;
    Light L;
    float dither; // per-pixel, per-frame in [0, 1); TAA resolves it
};

// Mean light reaching a staircase of unit steps with this gradient, as seen along d: each step that rises
// toward the light shades the strip of tread behind it (to contact level ct) and its own riser (to cr).
// The smoothed gradient misses the local ups and downs of terraced ground; the gain of 3 matches the mean
// of the traced step shadows it replaces.
static float stairShadow(float2 grad, float3 L, float3 d, float ct, float cr) {
    float2 up = grad * L.xz;
    float2 f = saturate(max(up, 0.0) / max(L.y, 0.02));
    float tread = 1.0 - (1.0 - (1.0 - f.x) * (1.0 - f.y)) * (1.0 - ct);
    float2 riser = max(grad * d.xz, 0.0);
    float2 riserSh = select(float2(1.0), float2(cr), up > 0.0);
    return saturate(1.0 - 3.0 * (1.0 - (abs(d.y) * tread + dot(riser, riserSh)) / (abs(d.y) + riser.x + riser.y)));
}

static float3 shade(thread const Ctx& x, float3 o, float3 d, Hit h, bool shadows, bool lips = true) {
    float3 p = o + d * h.t;
    float3 n = float3(h.normal);
    float2 uv = faceUV(p, h.normal);
    float fp = h.t * x.pixelAngle; // voxels per pixel
    Surface s = surfaceAt(x.hm, x.size, h.voxel, h.normal, uv, fp, x.origin, x.seed);
    // Prefilter: from ~1/10 voxel per pixel, blend toward the smooth heightfield normal, relax AO and
    // shadows, and give risers their tread colour, so distant stairs read as contour bands, not moire.
    float wf = smoothstep(0.025, 0.20, fp);
    float3 nS = normalize(float3(s.nh.x - s.nh.y, 2.0, s.nh.z - s.nh.w));
    // Below ~2 pixels per voxel even per-column normals, step shadows and sky visibility alias into a hatch
    // that crawls under motion; shade the continuous slope under the stairs instead.
    float wf2 = s.mat == 0 ? smoothstep(0.2, 0.5, fp) : 0.0;
    float2 grad = 0.0;
    if (wf2 > 0.0) {
        float R = max(2.0, 1.5 * fp);
        grad = float2(groundSmooth(x.hm, p.xz + float2(R, 0), x.size) - groundSmooth(x.hm, p.xz - float2(R, 0), x.size),
                      groundSmooth(x.hm, p.xz + float2(0, R), x.size) - groundSmooth(x.hm, p.xz - float2(0, R), x.size)) / (2.0 * R);
        nS = normalize(mix(nS, normalize(float3(-grad.x, 1.0, -grad.y)), wf2));
    }
    float3 nl = normalize(mix(n, nS, wf));
    float ao = mix(cornerAO(x.hm, x.size, h.voxel, h.normal, uv), 0.85, wf) * s.canopy;
    float3 alb = h.normal.y == 0 ? mix(s.albedo, s.tread, wf) : s.albedo;
    // Moonlight drains colour.
    alb = mix(alb, float3(dot(alb, float3(0.3, 0.59, 0.11))), 0.4 * x.L.night);
    // Canopies would otherwise drop to cut-out silhouettes under the moon.
    if (s.mat != 0) alb *= 1.0 + 1.4 * x.L.night;
    float ndl = max(dot(nl, x.L.key), 0.0);
    float hiSun = smoothstep(0.15, 0.45, x.L.key.y) * (1.0 - x.L.night);
    float sh = 1.0;
    if (shadows && ndl > 0.0 && x.L.shadows) {
        // Keep the origin inside the hit voxel's footprint so boundary columns can't self-shadow.
        float3 so = p + n * 0.02;
        so.xz = clamp(so.xz, float2(h.voxel.xz) + 0.002, float2(h.voxel.xz) + 0.998);
        // Prefiltered, the ray starts a step above the column, and single steps shade by their expected cover.
        so.y = mix(so.y, max(so.y, s.H) + 1.0, wf2);
        Hit sHit = trace(x.hm, so, x.L.key, 700.0, x.quality > 0.5 ? 120 : 48, x.globalMax, x.size);
        // Contact shadows from single-voxel steps draw sawtooth noise; keep them faint, cliffs full.
        float contact = 0.35 * (1.0 - smoothstep(2.0, 8.0, 0.5 / x.L.key.y));
        sh = sHit.hit ? 0.35 * (1.0 - smoothstep(2.0, 8.0, sHit.t)) : 1.0;
        if (wf2 > 0.0) sh = mix(sh, min(sh, stairShadow(grad, x.L.key, d, contact, 0.35)), wf2);
        sh = mix(sh, mix(0.5, 0.25, hiSun) + mix(0.5, 0.75, hiSun) * sh, wf);
        sh = mix(sh, 1.0, 0.3 * x.L.night);
    }
    sh *= mix(1.0, cloudShadow(p, x.L.key, x.origin, x.seed, x.time, x.cover), x.L.sunAmt);
    float lowFill = mix(0.9, 1.0, smoothstep(0.1, 0.4, x.L.key.y));
    // Low light: shadows take the cool zenith, against warm key light.
    // High sun: fill is the deep zenith, so shade reads blue against warm key light.
    float zw = mix(0.55, 0.8, hiSun);
    float3 skyFill = mix(x.L.horizon, x.L.zenith, saturate(zw + 0.15 * nl.y)) * 0.9 * lowFill * mix(1.0, 0.85, hiSun);
    float2 svAt = mix(float2(h.voxel.xz + h.normal.xz) + 0.5, p.xz, wf2);
    float svY = wf2 > 0.0 ? mix(p.y, groundSmooth(x.hm, p.xz, x.size), wf2) : p.y;
    skyFill *= mix(1.0, skyVisibility(x.hm, svAt, svY), mix(0.55, 0.85, hiSun));
    float3 bounce = (1.0 - saturate(nl.y)) * 0.35 * alb * x.L.keyCol * max(x.L.key.y, 0.0);
    // High sun: risers sit in cooler shade so terraces keep their modelling at noon.
    float drop = s.H - min(min(s.nh.x, s.nh.y), min(s.nh.z, s.nh.w));
    float3 riserShade = mix(float3(1.0), float3(0.82, 0.86, 0.95),
                            (1.0 - saturate(nl.y)) * smoothstep(0.3, 0.6, x.L.key.y) * smoothstep(1.5, 4.0, drop));
    // Night: no pure-black riser feet; a faint indigo ambient keeps the steps readable as stone, not holes.
    ao = mix(ao, 1.0, 0.25 * x.L.night);
    skyFill += float3(0.09, 0.12, 0.25) * x.L.night;
    float3 col = alb * (x.L.keyCol * ndl * sh * mix(1.0, 1.1, x.L.night) + (skyFill * ao * riserShade + bounce));
    // Low sun: up-facing treads take a warm sky-dome tint instead of going to charcoal.
    col += alb * x.L.keyCol * 0.07 * (1.0 - smoothstep(0.05, 0.35, x.L.key.y)) * saturate(nl.y) * (1.0 - x.L.night);
    // Backlit leaves: a low sun glows through the canopy edges facing away from it.
    if (s.mat == 1) {
        float back = pow(saturate(dot(d, x.L.key)), 4.0) * (1.0 - smoothstep(0.08, 0.45, x.L.key.y)) * (1.0 - x.L.night);
        float graze = 1.0 - abs(dot(n, d));
        col += alb * float3(0.9, 1.3, 0.3) * x.L.keyCol * back * (1.0 - ndl) * (1.0 - abs(n.y)) * graze * graze * sh;
    }
    // Lips of steps catch low light and moonlight; these rise into EDR headroom.
    float low = max(1.0 - smoothstep(0.05, 0.4, x.L.key.y), 0.8 * x.L.night);
    // Lips facing toward or away from the light glint; lips running along it stay quiet (lost-and-found edges).
    float face = abs(dot(s.lipDir, normalize(x.L.key.xz + 1e-5)));
    float lipLight = mix(0.15, 1.0, smoothstep(0.15, 0.75, face));
    // Under a high moon, land sits cool and below the sky so night reads as night (won night A/B);
    // under a low moon it is already dark and keeps its contours (darkening lost the dusk A/B).
    col *= mix(float3(1.0), float3(0.5, 0.56, 0.72), x.L.night * smoothstep(0.32, 0.45, x.L.moon.y));
    // Moonlit lips stay silver: they are added after the cool tint.
    float3 lipTint = mix(mix(alb, float3(1.0), mix(0.5, 0.15, hiSun)), float3(0.85, 0.92, 1.1), x.L.night);
    // Moonlit lips carry further than the prefilter allows by day: they are the night's contour lines,
    // at least a pixel wide, so they hold steady under TAA.
    float lipFade = mix(1.0 - wf, 1.0 - smoothstep(0.08, 0.25, fp), x.L.night);
    if (lips) col += s.glint * lipLight * lipFade * x.L.keyCol * sh * (mix(0.3, 0.22, hiSun) + 1.25 * low) * lipTint;
    return col;
}

// Converges on the actual sky colour as fog thickens (T -> 0), so fogged land meets the sky with no seam.
static inline float3 fogColor(thread const Ctx& x, float3 d, float T) {
    float y = max(d.y, 0.0);
    float3 c = skyColor(float3(d.x, mix(y, y * 0.5 + 0.12, T), d.z), x.L, false, x.seed, x.time);
    // Aerial colour cools away from the sun so far ridges separate from the sky.
    float away = saturate(dot(normalize(d.xz + 1e-5), normalize(x.L.sun.xz + 1e-5)) * 0.5 + 0.5);
    return c * mix(float3(1.0), mix(float3(0.80, 0.82, 0.95), float3(1.0), away), T);
}

// Exponential height fog plus a slow-drifting noise modulation that pools mist in valleys.
static float fogTransmittance(thread const Ctx& x, float3 o, float3 d, float t) {
    float hf = 30.0;
    float a = 0.0009, b = 0.0, c2 = 1.3e-6;
    float k = exp(-(o.y - SEA) / hf);
    float dy = abs(d.y) < 1e-4 ? 1e-4 : d.y;
    float heightTerm = a * hf * k * (1.0 - exp(-dy * t / hf)) / dy;
    float mist = 0;
    int samples = x.quality > 0.5 ? 4 : 2;
    for (int i = 0; i < samples; i++) {
        float st = min(t, 400.0) * (float(i) + 0.5) / float(samples);
        float3 sp = o + d * st;
        float3 wp = float3(sp.x + float(x.origin.x), sp.y, sp.z + float(x.origin.y));
        mist += saturate(perlin3(wp * float3(0.012, 0.03, 0.012) + float3(x.time * 0.01, 0, x.time * 0.006), x.seed + 50u) + 0.2)
              * exp(-max(sp.y - SEA, 0.0) / 10.0);
    }
    mist = mist / float(samples) * min(t, 400.0) * 0.0014 * mix(0.7, 0.3, smoothstep(0.15, 0.45, x.L.sun.y));
    // Hide the map edge by distance to the edge itself, not to the camera.
    float3 ep = o + d * t;
    float edgeDist = min(min(ep.x, float(x.size) - ep.x), min(ep.z, float(x.size) - ep.z));
    // Clear air under a high sun; haze belongs to the low light.
    float clear = mix(1.0, 0.15, smoothstep(0.2, 0.45, x.L.sun.y));
    // Night air is crisp: moonlit ridges stay legible to the horizon.
    clear *= mix(1.0, 0.35, x.L.night);
    return exp(-((heightTerm + b * t + c2 * t * t + mist) * clear + 5.0 * smoothstep(110.0, 10.0, edgeDist)));
}

static float3 applyFog(thread const Ctx& x, float3 col, float3 o, float3 d, float t) {
    float T = fogTransmittance(x, o, d, t);
    // Thin haze in-scatters less than the full horizon radiance; only thick fog converges on the sky.
    float3 f = (fogColor(x, d, T) + x.L.keyCol * min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night)) * mix(0.3, 1.0, smoothstep(0.1, 0.9, 1.0 - T));
    return mix(f, col, T);
}

struct CloudSample { float3 col; float a; float t; };

// Clouds: a volumetric slab above the peaks, marched in a few steps. The same noise field sets the
// coverage; tops tower where it is dense, and each sample is self-shadowed toward the sun, so clouds have
// lit crowns and shaded bases instead of reading as flat cut-outs.
static CloudSample cloudSheet(thread const Ctx& x, float3 o, float3 d) {
    CloudSample c;
    c.col = 0; c.a = 0; c.t = 6e4;
    if (d.y < 0.01) return c;
    float y0 = CLOUD_Y - 10.0, y1 = CLOUD_Y + 34.0;
    float ta = (y0 - o.y) / d.y, tb = (y1 - o.y) / d.y;
    float t0 = max(min(ta, tb), 0.0), t1 = max(ta, tb);
    if (t0 > 7000.0) return c;
    // Night clouds have crisp edges, which need finer, dithered steps to show no slices at grazing angles.
    int steps = (x.quality > 0.5 ? 14 : 8) + int(x.L.night * 0.0);
    float dt = (t1 - t0) / float(steps);
    float lowSun = 1.0 - smoothstep(0.02, 0.28, abs(x.L.sun.y));
    float2 dh = normalize(d.xz + 1e-5), sh2 = normalize(x.L.sun.xz + 1e-5);
    float toSun = 0.5 + 0.5 * dot(dh, sh2);
    float2 sunStep = x.L.key.xz / max(x.L.key.y, 0.2) * 14.0;
    float3 ambient = x.L.zenith * 1.0 + x.L.horizon * 0.35;
    float glowAmt = lowSun * smoothstep(-0.14, -0.02, x.L.sun.y);
    float T = 1.0;
    float3 acc = 0;
    float silver = 0;
    float jit = 0.5 + x.L.night * (x.dither - 0.5);
    float2 lastQ = float2(1e9);
    float coarse = 0, towardC = -1e9;
    float reuse = x.L.night > 0.0 ? 400.0 : -1.0;
    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + jit) * dt;
        float3 p = o + d * t;
        float2 q = p.xz + float2(x.origin);
        // Night: steep rays cross the slab within a few units, so the coarse field is reused between
        // nearby steps.
        bool fresh = distance_squared(q, lastQ) > reuse;
        if (fresh) { lastQ = q; coarse = cloudDensity(q, x.seed, x.time, x.cover); towardC = -1e9; }
        float base = coarse;
        // Night: a finer octave frays the edges into wisps for the moon to light through.
        if (x.L.night > 0.0 && base > -0.3) {
            float2 qd = q - float2(x.time * 1.3, 0);
            base += x.L.night * (0.16 * perlin2(qd / 90.0, x.seed + 42u) + 0.07 * perlin2(qd / 33.0, x.seed + 43u) + 0.035 * perlin2(qd / 13.0, x.seed + 44u));
        }
        float h01 = (p.y - y0) / (y1 - y0);
        float top = 0.35 + 0.65 * saturate(base * 2.5);
        float dens = saturate((base - 0.04) * mix(3.0, 9.0, x.L.night)) * smoothstep(0.0, 0.2, h01) * (1.0 - smoothstep(top * 0.7, top, h01));
        if (dens < 0.01) continue;
        if (towardC < -1e8) towardC = cloudDensity(q + sunStep, x.seed, x.time, x.cover);
        float toward = towardC;
        float shadow = exp(-saturate((toward - 0.04) * 3.0) * 2.2) * (0.55 + 0.45 * h01);
        float3 lit = x.L.keyCol * (mix(0.15, 0.4, x.L.night) + mix(0.85, 0.6, x.L.night) * shadow) * mix(1.0, 0.32, x.L.night) * (0.6 + 0.4 * toSun) + ambient * (0.6 + 0.4 * h01);
        lit += float3(1.6, 0.55, 0.20) * glowAmt * toSun * (1.0 - h01) * 0.9;
        float a = 1.0 - exp(-dens * dt * mix(0.05, 0.08, x.L.night));
        acc += T * a * lit;
        silver += T * a * shadow;
        T *= 1.0 - a;
        if (c.t > 5e4 && T < 0.6) c.t = t;
        if (T < 0.03) break;
    }
    float cover = 1.0 - T;
    c.col = acc / max(cover, 1e-3);
    // Moonlit night: forward scatter through the thin, moon-facing fringes draws a silver lining around
    // the moon.
    float cm = dot(d, x.L.moon);
    float fringe = silver / max(cover, 1e-3) * (1.0 - cover);
    c.col += float3(0.78, 0.86, 1.0) * x.L.night * hg(cm, 0.9) * 2.0 * fringe;
    c.col *= mix(1.0, 0.9, smoothstep(0.3, 0.5, x.L.sun.y));
    c.a = cover * mix(1.0, 0.7, x.L.night) * exp(-t0 / 3000.0) * smoothstep(7000.0, 2500.0, t0) * smoothstep(0.02, 0.1, d.y);
    return c;
}

static float4 skyWithClouds(thread const Ctx& x, float3 o, float3 d) {
    float3 raw = skyColor(d, x.L, false, x.seed, x.time);
    float3 bare = raw;
    // Forward-scatter glow near the horizon, matching what applyFog adds to fogged land.
    bare += x.L.keyCol * min(hg(dot(d, x.L.key), 0.6) * 0.35, 0.15) * (1.0 - x.L.night) * smoothstep(0.1, 0.0, d.y);
    float3 discs = skyColor(d, x.L, true, x.seed, x.time) - raw;
    CloudSample c = cloudSheet(x, o, d);
    // Clouds hide the stars, moon and sun behind them.
    // At night thin cloud veils the moon instead of hiding it.
    float3 sky = bare + discs * mix(1.0 - saturate(c.a * 3.0), (1.0 - c.a) * (1.0 - c.a), x.L.night);
    return float4(mix(sky, c.col, c.a), c.a > 0.5 ? c.t : 6e4);
}

// Reflected sky: clouds softened (rippled water blurs them), no stars or discs; glitter handles the lights.
static float3 softCloudSky(thread const Ctx& x, float3 o, float3 r) {
    float3 sky = skyColor(r, x.L, false, x.seed, x.time);
    if (r.y < 0.02) return sky;
    float t = (CLOUD_Y - o.y) / r.y;
    float2 q = o.xz + r.xz * t + float2(x.origin);
    // Soft coverage, blurred as if by ripples, matching the volumetric sky.
    float dens = smoothstep(0.0, 0.3, cloudDensity(q, x.seed, x.time, x.cover));
    float3 lit = x.L.keyCol * 0.4 * mix(1.0, 0.35, x.L.night) + x.L.zenith * 1.2 + x.L.horizon * 0.4;
    return mix(sky, lit, dens * 0.4 * exp(-t / 2600.0));
}

fragment float4 sceneFragment(VOut in [[stage_in]], constant Frame& f [[buffer(0)]],
                              texture2d<float> hm [[texture(0)]]) {
    float2 ndc = in.uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    ndc += f.jitter.xy;
    float tanH = f.camPos.w;
    float3 d = normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
    float3 o = f.camPos.xyz;

    Ctx x { hm, f.mapInfo.z, f.mapInfo.xy, uint(f.mapInfo.w), f.camRight.w, f.camUp.w, f.misc.y,
            2.0 * tanH * f.resolution.w, f.misc.w, makeLight(f.sunDir.xyz),
            fract(52.9829189 * fract(dot(in.pos.xy + f.misc2.x * float2(5.588, 2.31), float2(0.06711056, 0.00583715)))) };

    Hit h = trace(hm, o, d, 4000.0, 400, x.globalMax, x.size);
    if (f.misc2.z == 1) return float4(float(h.iters) / 400.0, h.hit ? 0.3 : 0.0, h.iters >= 399 ? 1 : 0, 1);
    if (f.misc2.z == 5) return float4(skyColor(d, x.L, false, x.seed, x.time), 1);
    if (f.misc2.z == 6) return float4(skyWithClouds(x, o, d).rgb, 1);

    float tw = d.y < 0 ? (SEA - o.y) / d.y : 1e9;
    bool water = d.y < 0 && tw > 0 && (!h.hit || tw < h.t);
    float3 col;
    float tSurf;
    if (water) {
        float3 wp = o + d * tw;
        float2 ww = wp.xz + float2(x.origin);
        float e0 = perlin3(float3(ww * 0.3, x.time * 0.25), x.seed + 60u);
        float ex = perlin3(float3((ww + float2(0.05, 0)) * 0.3, x.time * 0.25), x.seed + 60u);
        float ez = perlin3(float3((ww + float2(0, 0.05)) * 0.3, x.time * 0.25), x.seed + 60u);
        float rs = max(0.35 * saturate(1.0 - tw * x.pixelAngle * 4.0), 0.08 * x.L.night);
        float3 n = normalize(float3(-(ex - e0) * rs, 1.0, -(ez - e0) * rs));
        float3 r = reflect(d, n);
        r.y = abs(r.y);
        float3 refl;
        if (x.quality > 0.5) {
            Hit rh = trace(hm, wp + float3(0, 0.02, 0), r, 1500.0, 220, x.globalMax, x.size);
            refl = rh.hit ? applyFog(x, shade(x, wp, r, rh, false, false), wp, r, rh.t)
                           : softCloudSky(x, wp, r);
        } else {
            refl = softCloudSky(x, wp, r);
        }
        float3 deep = x.L.zenith * 0.18 + x.L.horizon * 0.06 + float3(0.004, 0.014, 0.018);
        float3 under = deep;
        if (h.hit) {
            float path = h.t - tw;
            float3 lit = shade(x, o, d, h, false, false);
            under = mix(deep, lit * float3(0.55, 0.8, 0.85), exp(-path * 0.35));
        }
        under *= 1.0 - 0.4 * x.L.night;
        float cosi = saturate(-dot(d, n));
        float fres = 0.02 + 0.98 * pow(1.0 - cosi, 5.0);
        col = mix(under, refl, fres);
        // Glitter path toward the sun or moon: the main EDR highlight.
        col += x.L.keyCol * 6.0 * pow(max(dot(r, x.L.key), 0.0), mix(900.0, 4000.0, x.L.night)) * fres * (x.L.shadows ? 1.0 : 0.3) * (1.0 + 0.5 * x.L.night);
        if (x.L.night > 0.0) {
            // Moon path: the chance that some ripple facet mirrors the moon (a slope distribution, so it is
            // smooth and TAA-stable), broken into glitter by fine ripples that fade with pixel footprint.
            float3 hv = normalize(x.L.moon - d);
            float2 slope = hv.xz / max(hv.y, 0.05);
            float path = exp(-dot(slope, slope) / 0.03);
            float fine = perlin3(float3(ww * 1.6, x.time * 0.7), x.seed + 61u);
            float spark = mix(0.3, smoothstep(0.2, 0.55, fine), saturate(1.0 - tw * x.pixelAngle * 3.0));
            col += float3(0.75, 0.82, 1.0) * x.L.night * path * (0.12 + 2.5 * spark);
        }
        tSurf = tw;
    } else if (h.hit) {
        col = shade(x, o, d, h, true);
        tSurf = h.t;
    } else {
        return skyWithClouds(x, o, d);
    }
    if (f.misc2.z == 3) return float4(float3(fogTransmittance(x, o, d, tSurf)), 1);
    if (f.misc2.z != 2) col = applyFog(x, col, o, d, tSurf);
    return float4(col, tSurf);
}


static inline float3 tmap(float3 v) { return v / (1.0 + max(max(v.r, v.g), v.b)); }
static inline float3 itmap(float3 v) { return v / max(1.0 - max(max(v.r, v.g), v.b), 1e-4); }

// 9-tap Catmull-Rom via bilinear taps: bilinear history fetches blur edges a little more every frame the
// camera moves, which TAA then accumulates.
static float3 sampleCatmullRom(texture2d<float> t, sampler s, float2 uv) {
    float2 size = float2(t.get_width(), t.get_height());
    float2 sp = uv * size;
    float2 tc = floor(sp - 0.5) + 0.5;
    float2 f = sp - tc;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t0 = (tc - 1.0) / size, t3 = (tc + 2.0) / size, t12 = (tc + w2 / w12) / size;
    float3 c = 0;
    c += t.sample(s, float2(t0.x, t0.y)).rgb * w0.x * w0.y;
    c += t.sample(s, float2(t12.x, t0.y)).rgb * w12.x * w0.y;
    c += t.sample(s, float2(t3.x, t0.y)).rgb * w3.x * w0.y;
    c += t.sample(s, float2(t0.x, t12.y)).rgb * w0.x * w12.y;
    c += t.sample(s, float2(t12.x, t12.y)).rgb * w12.x * w12.y;
    c += t.sample(s, float2(t3.x, t12.y)).rgb * w3.x * w12.y;
    c += t.sample(s, float2(t0.x, t3.y)).rgb * w0.x * w3.y;
    c += t.sample(s, float2(t12.x, t3.y)).rgb * w12.x * w3.y;
    c += t.sample(s, float2(t3.x, t3.y)).rgb * w3.x * w3.y;
    return max(c, 0.0);
}

// Temporal resolve: reproject history through the previous camera (the world is static), clamp it to
// the current 3x3 neighbourhood, and blend. Alpha carries hit distance.
fragment float4 taaFragment(VOut in [[stage_in]], texture2d<float> cur [[texture(0)]],
                            texture2d<float> hist [[texture(1)]], constant Frame& f [[buffer(0)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    int2 px = int2(in.pos.xy);
    float4 c = cur.read(uint2(px));
    if (f.prevPos.w < 0.5 || f.misc2.z == 7) return c;
    float3 mn = tmap(c.rgb), mx = mn;
    int2 lim = int2(cur.get_width() - 1, cur.get_height() - 1);
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            float3 v = tmap(cur.read(uint2(clamp(px + int2(i, j), int2(0), lim))).rgb);
            mn = min(mn, v);
            mx = max(mx, v);
        }

    float2 ndc = in.uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float tanH = f.camPos.w;
    float3 d = normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
    float3 wp = f.camPos.xyz + d * min(c.a, 1e5);
    float3 v = wp - f.prevPos.xyz;
    float z = dot(v, f.prevFwd.xyz);
    if (z <= 0.01) return c;
    float2 pn = float2(dot(v, f.prevRight.xyz) / (z * tanH * f.camFwd.w), dot(v, f.prevUp.xyz) / (z * tanH));
    float2 puv = float2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
    if (any(puv < 0.0) || any(puv > 1.0)) return c;
    // Clamp and blend in a compressed space so sub-pixel EDR spikes can't smear or flicker.
    float3 h = clamp(tmap(sampleCatmullRom(hist, lin, puv)), mn, mx);
    return float4(itmap(mix(h, tmap(c.rgb), 0.12)), c.a);
}

static inline float3 acesFit(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

static inline float3 linearToSRGB(float3 c) {
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

static inline float3 srgbToLinear(float3 c) {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

// misc2.w is the display's EDR headroom; 0 means SDR output (sRGB-encoded 8-bit).
fragment float4 postFragment(VOut in [[stage_in]], texture2d<float> scene [[texture(0)]],
                             constant Frame& f [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float3 x = scene.sample(s, in.uv).rgb * f.misc.x;
    float3 c = acesFit(x);
    // Noon light is the flattest; restore a little chroma the tonemap shoulder takes away.
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = max(mix(float3(l), c, 1.0 + 0.10 * smoothstep(0.2, 0.45, f.sunDir.y)), 0.0);
    float ign = fract(52.9829189 * fract(dot(in.pos.xy + f.misc2.x * float2(5.588, 2.31), float2(0.06711056, 0.00583715))));
    float3 enc = linearToSRGB(saturate(c)) + (ign - 0.5) / 255.0;
    float headroom = f.misc2.w;
    if (headroom <= 1.0) return float4(enc, 1);
    // EDR: identical to SDR below scene value 1; energy above it rolls per channel into the headroom.
    // EDR: a hue-preserving luminance knee above diffuse white, so only lips, glitter, sun, moon and
    // bright stars rise into the headroom.
    float3 lin = srgbToLinear(saturate(enc));
    float Y = dot(x, float3(0.2126, 0.7152, 0.0722));
    float e = (headroom - 1.0) * (1.0 - exp(-max(Y - 1.6, 0.0) / 2.5));
    lin *= 1.0 + e / max(dot(lin, float3(0.2126, 0.7152, 0.0722)), 1e-3);
    if (f.misc2.z == 4) return float4(e > 0.05 ? float3(1, 0, 0) : lin, 1);
    return float4(lin, 1);
}
