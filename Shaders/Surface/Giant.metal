// 0..1 where world xz w lies among mega relief, from the coarsest ring's max over ~128-voxel cells (lowland rarely rises
// past ~190). Interpolated between cell corners, each the max of the four cells meeting there, so every column rates at
// least its own cell's relief and the field stays continuous: interpolated between cell centres, a wall's foot columns
// past a cell edge rated partly as lowland, and a jogging face crossfaded column by column between the formations and the
// lowland's 16-voxel beds, which near eye level line up with the beds on every rock behind and read as see-through rock.
// Wider cells would carry the walls' rock a whole cell out over the lowland at their feet.
static inline float megaWall(thread const Terrain& tr, float2 w) {
    int kc = tr.levels - 1;
    if (kc < 1) return 0.0;
    int m = max(7 - kc, 0), mask = (COARSE_N >> m) - 1;
    float2 q = w / float(1 << (kc + m));
    int2 c = int2(floor(q));
    float2 f = q - float2(c);
    uint sl = uint(kc - 1), mp = uint(m);
    #define MW(i, j) tr.ch.read(uint2((c + int2(i, j)) & mask), sl, mp).r
    float a0 = MW(-1, -1), a1 = MW(0, -1), a2 = MW(1, -1);
    float b0 = MW(-1, 0), b1 = MW(0, 0), b2 = MW(1, 0);
    float c0 = MW(-1, 1), c1 = MW(0, 1), c2 = MW(1, 1);
    #undef MW
    float v00 = max(max(a0, a1), max(b0, b1)), v10 = max(max(a1, a2), max(b1, b2));
    float v01 = max(max(b0, b1), max(c0, c1)), v11 = max(max(b1, b2), max(c1, c2));
    return smoothstep(190.0, 270.0, mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y));
}

// A giant's bark, leaves or vines (surfaceAt), returned before the ground's shading: none of its tones apply.
static Surface giantSurface(thread const Terrain& tr, Surface s, int k, int3 v, int3 n, float2 uv, float fp, float2 wq,
                                   float wy, float3 col, uint gsb, float gLoc, Giant gi, float cs, int2 mapOrigin, uint seed) {
    float y = float(v.y);
    s.autumn = 0.0;
    float runLo = gLoc, runHi = col.x;
    int gm = GM_WOOD;
    if (k == 0) {
        float4 e = extraGaps(tr.gx.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))));
        // The extra gaps in either order (writeVeg), both below the principal one.
        bool sw = e.w > e.z && e.z < e.x;
        float3 g0 = float3(sw ? e.z : e.x, sw ? e.x : e.z, col.y), g1 = float3(sw ? e.w : e.y, sw ? e.y : e.w, col.z);
        int ri = 0, nG = 0;
        bool found = false;
        for (int i = 0; i < 3; i++) {
            if (g1[i] <= g0[i]) continue;
            nG++;
            if (found) continue;
            if (y >= g1[i]) { ri++; runLo = g1[i]; }
            else { runHi = g0[i]; found = true; }
        }
        float leafStart = float((gsb >> 11) & 2047u);
        int rm = ri < 3 ? int((gsb >> (22u + 2u * uint(ri))) & 3u) : GM_WOOD;
        gm = ri == nG && y >= leafStart ? GM_LEAF : rm;
        if (ri == nG && gm == GM_LEAF) runLo = leafStart;
    } else {
        float2 q = float2(v.xz * int(cs) + mapOrigin) + 0.5 * cs - gi.c;
        bool above = col.z > col.y && y >= col.z;
        // Bark only on the trunk; elsewhere a solid coarse column is the crown's skirt of leaves and vines.
        gm = above || length(q) > 40.0 || (y - gi.G > gi.yT * 0.55 && length(q) > 24.0) ? GM_LEAF : GM_WOOD;
        // Wood under the crown's air: aerial roots, buttresses, arching roots. Levels with a second gap mark its top
        // there (coarseColumn); further out it is guessed from the lowest limbs.
        float2 g2 = k <= GAP2_LEVELS ? secondGapW(tr, k, v.xz, 0.0) : float2(0);
        float woodTop = k <= GAP2_LEVELS ? g2.x : col.z > col.y ? min(col.y, gi.G + gi.yT * 0.42) : col.x < gi.G + 120.0 ? 1e9 : -1e9;
        if (!above && y < woodTop) gm = GM_WOOD;
        runLo = above ? col.z : gLoc;
    }
    float3 a;
    if (gm == GM_LEAF) {
        uint pal = k == 0 ? (gsb >> 28) & 7u : giantPalette(gi.h);
        s.up = 1.0 - saturate((runHi - 1.0 - y) / 14.0);
        if (pal != 0u || k != 0) {
            float kq = saturate(perlin2(wq / 31.0, seed + 601u) * 0.9 + 0.5);
            a = pal == 2u ? mix(float3(0.024, 0.070, 0.060), float3(0.050, 0.098, 0.066), kq)
                          : mix(float3(0.034, 0.078, 0.050), float3(0.075, 0.112, 0.042), kq);
        } else {
            // Old growth and the jungle: the wood's own leaves (provinces carry no autumn or painted hills).
            float2 w = float2(v.xz + mapOrigin);
            Climate c = climateL(tr, 0, float2(v.xz) + 0.5);
            c.temp = altitudeTemp(c.temp, gLoc);
            a = woodLeaf(w, c, 0.0, 0.0, 0.0, seed);
            float ct = 1.0 - smoothstep(0.5, 1.0, fp);
            if (ct > 0.0) {
                Crown cr = crownAt(w + 0.5, c, seed);
                a *= mix(float3(1.0), cr.tint, ct);
                if (cr.Ht > 0.0) s.up = saturate((y - gLoc - 0.45 * cr.Ht) / (0.55 * cr.Ht));
            }
        }
        // A rust flush drifts over part of the crown, strongest on its upper leaves.
        if (pal == 3u) a = mix(a, float3(0.115, 0.062, 0.030), 0.7 * smoothstep(0.05, 0.45, perlin2(wq / 23.0, seed + 604u) + 0.25 * saturate((y - runLo) / max(runHi - runLo, 1.0))));
        int2 tx = int2(floor(uv * 16.0));
        int3 tk = n.y != 0 ? int3(v.x * 16 + tx.x, v.y * 2 + max(n.y, 0), v.z * 16 + tx.y)
                : n.x != 0 ? int3(v.x * 2 + max(n.x, 0), v.y * 16 + tx.y, v.z * 16 + tx.x)
                           : int3(v.x * 16 + tx.x, v.y * 16 + tx.y, v.z * 2 + max(n.z, 0));
        uint th = hash3i(tk, seed + 94u);
        float tex = hashf(th) < 0.35 ? 0.72 : 1.0 + 0.12 * hashf(pcg(th));
        a *= mix(tex, 0.92, smoothstep(0.015, 0.04, fp));
        // Darker with depth under the run's top: a clump's side darkens down from its crest the way the dome turns
        // from the sky, rather than by each column's own span (which striped the clumps column by column).
        a *= mix(1.15, 0.62, saturate((runHi - 1.0 - y) / 14.0));
        s.mat = 1;
        s.canopy = 1.0;
    } else if (gm == GM_VINE) {
        a = float3(0.036, 0.050, 0.028) * (0.85 + 0.3 * hashf(hash2i(v.xz, seed + 602u)));
        s.mat = 1;
        s.vine = true;
        s.canopy = mix(1.0, 0.75, smoothstep(6.0, 80.0, col.x - y));
    } else {
        // Bark in vertical ridges, mossed where it faces up and low down.
        a = float3(0.115, 0.098, 0.082);
        float fade = 1.0 - smoothstep(0.1, 0.5, fp);
        if (fade > 0.0) a *= 1.0 + 0.40 * fade * perlin3(float3(wq.x / 1.6, wy / 12.0, wq.y / 1.6), seed + 600u);
        float moss = saturate((n.y > 0 ? 0.75 : 0.05) + 0.6 * perlin3(float3(wq.x, wy * 0.7, wq.y) / 6.0, seed + 603u)
                              - 0.5 * smoothstep(gLoc + 15.0, gLoc + 140.0, wy));
        a = mix(a, float3(0.062, 0.092, 0.040), moss);
        s.mat = 2;
        s.canopy = mix(1.0, 0.75, smoothstep(6.0, 80.0, col.x - y));
    }
    s.albedo = a;
    s.tread = a;
    s.face = a;
    s.H = runHi;
    s.pale = 0;
    s.slope = 0;
    s.level = 0;
    return s;
}
