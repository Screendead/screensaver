// Two world-anchored toroidal grids around the camera (SHADOW_N texels a side; SHADOW_CELL voxels and 4x that):
// each texel holds the height below which its column is in the shadow of terrain at least SHADOW_NEAR away toward
// the key light, from the coarse rings' max mips, and the heights [z, w] the islands toward the light shade (empty:
// z > w), so at a low sun an island's shadow falls clear of the ground beneath it. Rebaked a band per frame; shade()
// traces only the near ray.
constant float SHADOW_CELL = 8.0;
constant int SHADOW_N = 1024;
constant float SHADOW_NEAR = 96.0;

struct ShadowBake {
    float4 key;     // xyz key light direction, w horizontal start distance
    float4 cam;     // xy camera world xz, z cell size, w levels built
    int4 rows;      // x first row of this band, y camera cell x, z camera cell z
};

kernel void bakeShadowHeight(texture2d<float, access::write> out [[texture(0)]],
                             texture2d_array<float> ch [[texture(1)]],
                             texture2d_array<float> cb [[texture(2)]],
                             texture2d_array<float> cg [[texture(3)]],
                             constant ShadowBake& b [[buffer(0)]],
                             uint2 g [[thread_position_in_grid]]) {
    int2 t = int2(g) + int2(0, b.rows.x);
    if (t.x >= SHADOW_N || t.y >= SHADOW_N) return;
    // The window's cell for this texel: the one within half a window of the camera's cell.
    int2 lo = b.rows.yz - SHADOW_N / 2;
    int2 c = lo + ((t - lo) & (SHADOW_N - 1));
    float cell = b.cam.z;
    float2 P = (float2(c) + 0.5) * cell;
    float hl = length(b.key.xz);
    if (hl < 1e-3 || b.key.y <= 0.0) { out.write(float4(-1e4, -1e4, 1e4, -1e4), uint2(t)); return; }
    float2 dir = b.key.xz / hl;
    float tanE = b.key.y / hl;
    int levels = int(b.cam.w);
    // x: occluders from the near ray's reach on; y: from two texels out, for pixels that trace no near ray.
    float S = -1e4, Sa = -1e4, Ilo = 1e4, Ihi = -1e4;
    float d = 2.0 * cell;
    float step = cell;
    float iA = 1e9, iB = -1e9;
    for (int i = 0; i < 96; i++) {
        float2 X = P + dir * d;
        float r = length(X - b.cam.xy);
        // Finest ring that holds X, at the max mip whose cells are as wide as the step.
        int k = 1;
        while (k <= levels && r > 900.0 * float(1 << k)) k++;
        if (k > levels) break;
        int m = max(int(round(log2(step))) - k, 0);
        int2 mc = int2(floor(X / float(1 << k))) >> m;
        float hm = ch.read(uint2(mc & ((2048 >> m) - 1)), uint(k - 1), uint(m)).r;
        float w = float(1 << (k + m));
        // Under an island the ground stops at the band's floor, and the island fills the band's roof to the top.
        float2 isl = float2(1e4, -1e4);
        if (hm < 0.0) {
            hm = -hm;
            float2 bd = cb.read(uint2(mc & ((1024 >> (m - 1)) - 1)), uint(k - 1), uint(m - 1)).rg;
            isl = float2(bd.y, hm);
            hm = bd.x;
        } else if (m == 0 && k <= GAP_LEVELS) {
            float2 gp = cg.read(uint2(mc & 2047), uint(k - 1)).rg;
            if (gp.x < 0.0) { isl = float2(gp.y, hm); hm = -gp.x; }
        }
        // The cell's nearest part lies up to ~0.7 of its width closer than X.
        float hs = hm - max(d - 0.7 * w, cell) * tanE;
        Sa = max(Sa, hs);
        if (d >= b.key.w) S = max(S, hm - max(d - 0.7 * w, b.key.w) * tanE);
        if (isl.y > isl.x) { iA = min(iA, d - w); iB = max(iB, d + w); }
        if ((d - w) * tanE > 2100.0) break;
        d += step;
        if (d > 5.0 * step) step *= 2.0;
        if (d > 7000.0) break;
    }
    // The islands' shade, marched again over their stretch of the ray in cells at most 16 voxels wide (the coarse
    // steps' cells would cut its edge into blocks), from the near ray's reach on, as S: nearer, a cell's max over an
    // island's rolling top would shade the top itself.
    float st = max(cell, 16.0);
    d = max(iA, b.key.w);
    for (int i = 0; i < 160 && d <= iB; i++) {
        float2 X = P + dir * d;
        float r = length(X - b.cam.xy);
        int k = 1;
        while (k <= levels && r > 900.0 * float(1 << k)) k++;
        if (k > levels) break;
        int m = max(int(round(log2(st))) - k, 0);
        int2 mc = int2(floor(X / float(1 << k))) >> m;
        float hm = ch.read(uint2(mc & ((2048 >> m) - 1)), uint(k - 1), uint(m)).r;
        float w = float(1 << (k + m));
        float2 isl = float2(1e4, -1e4);
        if (hm < 0.0) isl = float2(cb.read(uint2(mc & ((1024 >> (m - 1)) - 1)), uint(k - 1), uint(m - 1)).g, -hm);
        else if (m == 0 && k <= GAP_LEVELS) {
            float2 gp = cg.read(uint2(mc & 2047), uint(k - 1)).rg;
            if (gp.x < 0.0) isl = float2(gp.y, hm);
        }
        if (isl.y > isl.x) {
            Ilo = min(Ilo, isl.x - (d + 0.7 * w) * tanE);
            Ihi = max(Ihi, isl.y - max(d - 0.7 * w, b.key.w) * tanE);
        }
        d += st;
    }
    out.write(float4(S, Sa, Ilo, Ihi), uint2(t));
}

// The light an island's shade (the maps' [z, w]) leaves at height y over world xz w, bilinear over the four texels
// round it, each lit or shaded on its own: a filtered interval would blend an empty one into a wrong one.
static float isleShadeLit(texture2d<float> sm, float2 w, float cell, float y, float pen) {
    float2 q = w / cell - 0.5, f = fract(q);
    int2 i0 = int2(floor(q));
    float lit = 0.0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            float2 I = sm.read(uint2((i0 + int2(i, j)) & (SHADOW_N - 1))).zw;
            float in = I.y > I.x ? smoothstep(I.x - pen, I.x + pen, y) * (1.0 - smoothstep(I.y - pen, I.y + pen, y)) : 0.0;
            lit += (1.0 - in) * (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
        }
    return lit;
}

// Fraction of the key light the maps let through to p (map-local); never lighter than a lit column.
static float shadowMapLit(thread const Ctx& x, float3 p, bool all) {
    constexpr sampler rs(filter::linear, address::repeat);
    float2 w = p.xz + float2(x.origin);
    float2 cw = x.tr.cam + float2(x.origin);
    float2 dv = abs(w - cw);
    float rn = max(dv.x, dv.y) / (SHADOW_CELL * float(SHADOW_N / 2));
    float bn = smoothstep(0.80, 0.92, rn);
    float2 Sn = x.shadowNear.sample(rs, w / (SHADOW_CELL * float(SHADOW_N)), level(0)).rg;
    if (bn > 0.0) Sn = mix(Sn, x.shadowFar.sample(rs, w / (4.0 * SHADOW_CELL * float(SHADOW_N)), level(0)).rg, bn);
    float S = all ? Sn.y : Sn.x;
    float pen = mix(6.0, 14.0, bn);
    float lit = smoothstep(S - pen, S + pen, p.y);
    if (SCENE_ISLES && x.isleShadows) {
        float li = isleShadeLit(x.shadowNear, w, SHADOW_CELL, p.y, pen);
        if (bn > 0.0) li = mix(li, isleShadeLit(x.shadowFar, w, 4.0 * SHADOW_CELL, p.y, pen), bn);
        lit = min(lit, li);
    }
    return mix(lit, 1.0, smoothstep(0.85, 0.95, rn * 0.25));
}

// Mean light reaching the stairs around p, as seen along d, averaged bilinearly over the four nearest
// columns: a higher neighbour toward the light shades a strip of a column's tread (to contact level ct), and
// risers sit at contact level cr, as the traced shadow had them: its origin, kept inside the hit column, starts
// a side face's ray inside the solid.
// far: a pixel spans several columns, so the nearest column alone stands for them (TAA averages the rest).
static float stairShadow(thread const Terrain& tr, int k, float2 p, float y, float3 L, float3 d, float ct, float cr, bool far = false) {
    float cs = tr.lod[k].w;
    float2 q = p / cs - 0.5, f = fract(q);
    int2 c0 = int2(floor(q));
    int2 sx = int2(L.x > 0.0 ? 1 : -1, 0), sz = int2(0, L.z > 0.0 ? 1 : -1);
    int2 vx = int2(d.x > 0.0 ? 1 : -1, 0), vz = int2(0, d.z > 0.0 ? 1 : -1);
    // Horizontal reach of a unit step's shadow along each axis, in columns.
    float2 reach = abs(L.xz) / max(L.y, 0.02) / cs;
    float lit = 0.0, area = 0.0;
    int n = far ? 0 : 1;
    if (far) { c0 = int2(floor(p / cs)); f = 0.0; }
    for (int j = 0; j <= n; j++)
        for (int i = 0; i <= n; i++) {
            int2 c = c0 + int2(i, j);
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            float h = groundL(tr, k, c, y);
            // A step j columns toward the light covers what its shadow has left after crossing j - 1 columns.
            float fx = 0.0, fz = 0.0;
            for (int j = 1; j <= 2; j++) {
                fx = max(fx, (groundL(tr, k, c + sx * j, y) - h) * reach.x - float(j - 1));
                fz = max(fz, (groundL(tr, k, c + sz * j, y) - h) * reach.y - float(j - 1));
            }
            fx = saturate(fx);
            fz = saturate(fz);
            float tread = 1.0 - (1.0 - (1.0 - fx) * (1.0 - fz)) * (1.0 - ct);
            // Risers on the camera's side of the column, by visible area.
            float rx = max(h - groundL(tr, k, c - vx, y), 0.0) * abs(d.x);
            float rz = max(h - groundL(tr, k, c - vz, y), 0.0) * abs(d.z);
            lit += w * (abs(d.y) * tread + (rx + rz) * cr);
            area += w * (abs(d.y) + rx + rz);
        }
    return lit / area;
}
