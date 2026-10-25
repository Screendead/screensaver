// Whether a giant (the map's table) stands within d of world xz c.
static inline bool isleNearGiant(device const GiantRec* T, float2 c, float d) {
    int2 c0 = int2(floor((c - d) / GIANT_CELL)), c1 = int2(floor((c + d) / GIANT_CELL));
    for (int j = c0.y; j <= c1.y; j++)
        for (int i = c0.x; i <= c1.x; i++) {
            GiantRec r = T[giantSlot(int2(i, j))];
            if (r.G > 0.0 && all(r.cell == int2(i, j)) && length_squared(c - r.c) < d * d) return true;
        }
    return false;
}

// The highest ground under an island of the given reach round c: its centre and three rings.
static inline float isleGround(float2 c, float reach, uint seed) {
    float B = terrainColumnBase(int2(floor(c)), seed).h;
    for (int i = 0; i < 24; i++) {
        float rr = reach * (i < 8 ? 0.35 : i < 16 ? 0.7 : 1.0);
        float an = float(i) * 0.785398 + (i < 8 ? 0.0 : i < 16 ? 0.39 : 0.2);
        B = max(B, terrainColumnBase(int2(floor(c + float2(cos(an), sin(an)) * rr)), seed).h);
    }
    return B;
}

// A waterfall for some of the bigger islands, at a hashed bearing: its lip the last column of the island's ground out
// along it, its water two under the ground there (the stream's channel), its foot ISLE_AIR and more over the highest
// ground under the island (floor), so it never reaches the ground and every column of it stays an island's air.
static void isleFallPlace(thread IsleRec& r, float floor0, uint seed) {
    r.fa = 0;
    r.f = 0;
    if (r.R < 60.0 || hashf(pcg(r.h + 16u)) >= 0.45) return;
    float az = hashf(pcg(r.h + 17u)) * 6.2831853;
    float2 u = float2(cos(az), sin(az));
    float re = isleReach(r.R, r.h);
    float2 lip = 0;
    float top = 0;
    uint k;
    for (float t = 0.3 * r.R; t < re; t += 1.0) {
        float2 p = floor(r.c + u * t) + 0.5;
        float3 sp = isleSpan(r, p, seed, 4, k);
        if (sp.y <= sp.x) break;
        lip = p;
        top = sp.z > 0.0 ? sp.z : sp.y;
    }
    if (top <= 0.0 || length(lip - r.c) + 10.0 > re) return;
    float water = top - 2.0;
    float L = mix(70.0, 230.0, hashf(pcg(r.h + 18u))) * clamp(r.R / 220.0, 0.55, 1.15);
    float foot = floor(max(water - L, floor0 + ISLE_AIR + 25.0));
    if (water - foot < 40.0) return;
    r.fa = float2(az, clamp(r.R * 0.03, 2.5, 6.0) * mix(0.8, 1.2, hashf(pcg(r.h + 19u))));
    r.f = float4(lip, water, foot);
}

// The test field (VR_ISLE_FIELD): islands in a disc round (-3500, -2500) whatever the country, one massif among them.
constant float2 ISLE_FIELD_C = float2(-3500.0, -2500.0);
constant float ISLE_FIELD_R = 5200.0;

// One thread per slot: the candidate of its cell in the window round the map's middle (info.xy, world), unless the slot
// holds that cell already. info.z bit 0: the test field; bit 1: none (island country without its islands). A candidate stands where island country is deep enough at its
// centre (a massif deeper still), clear of the mega relief's provinces, the tepuis, the towers and the giants, over
// ground low across its whole reach. Sparse, and none small: a view holds a few big ones, not a fleet of chunks. Their
// heights spread wide, so no two near ones float level; a big keel clears the ground by 100, a smaller one by 60.
kernel void placeIslands(device IsleRec* out [[buffer(0)]],
                         constant int4& info [[buffer(1)]],
                         device const GiantRec* giants [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(ISLE_SLOTS)) return;
    uint seed = uint(info.w);
    bool massif = gid >= uint(ISLE_WIN * ISLE_WIN);
    int W = massif ? ISLE_MWIN : ISLE_WIN;
    int i = int(gid) - (massif ? ISLE_WIN * ISLE_WIN : 0);
    float cw = massif ? ISLE_MCELL : ISLE_CELL;
    int2 cell = int2(floor(float2(info.xy) / cw)) - W / 2 + int2(i % W, i / W);
    int slot = isleSlot(cell, massif);
    if (all(out[slot].cell == cell)) return;
    IsleRec r;
    r.cell = cell; r.c = 0; r.R = 0; r.A = 0; r.D = 0; r.B = 0; r.K = 0; r.h = 0; r.fa = 0; r.f = 0;
    if ((info.z & 2) != 0) { out[slot] = r; return; }
    if ((info.z & 1) != 0) {
        if (massif) {
            float2 bc = ISLE_FIELD_C + float2(600.0, 900.0);
            if (all(int2(floor(bc / ISLE_MCELL)) == cell)) {
                r.c = bc; r.R = 560.0; r.A = 520.0; r.h = 0x5bd1e995u; r.D = 380.0 - isleTilt(r.R, r.h);
            }
        } else {
            uint h = hash2i(cell, seed + 700u);
            float2 c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h)), hashf(pcg(h + 1u)))) * ISLE_CELL;
            if (hashf(pcg(h + 2u)) <= 0.55 && length(c - ISLE_FIELD_C) <= ISLE_FIELD_R) {
                float u = hashf(pcg(h + 3u));
                r.c = c; r.h = h;
                r.R = 18.0 + 300.0 * u * u * u;
                r.A = floor(190.0 + 420.0 * hashf(pcg(h + 4u)));
                r.D = min(isleDepth(r.R, h), r.A - 120.0 - isleTilt(r.R, h));
            }
        }
        r.K = r.A + 0.06 * r.R - 8.5 - (r.R > 150.0 ? 1.3 : 1.0) * r.D - 18.0 - isleTilt(r.R, r.h);
        if (r.R > 0.0) isleFallPlace(r, 200.0, seed);
        out[slot] = r;
        return;
    }
    uint h = hash2i(cell, seed + (massif ? 721u : 711u));
    float2 c = floor((float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h)), hashf(pcg(h + 1u)))) * cw) + 0.5;
    Climate cl = climate(c, seed);
    float chance = massif ? 0.4 * smoothstep(0.3, 0.7, cl.isle) : 0.15 * smoothstep(0.15, 0.5, cl.isle);
    bool ok = hashf(pcg(h + 2u)) < chance && desertAmount(cl) < 0.1 && saltAmount(cl) < 0.1;
    float u = hashf(pcg(h + 3u));
    float R = massif ? 420.0 + 230.0 * u : 60.0 + 240.0 * u * u;
    float reach = isleReach(R, h);
    float2 p = c + seedOffset(seed);
    for (int k = 0; k < 4 && ok; k++) {
        float a = float(k) * 1.5707963 + 0.785398;
        ok = towerProvinceField(p + float2(cos(a), sin(a)) * 2500.0, seed) <= 0.5;
    }
    ok = ok && !towerNearTepui(p, reach + 600.0, seed) && !towerNear(c, reach + 300.0, seed)
            && !isleNearGiant(giants, c, reach + 300.0 + GIANT_REACH);
    float B = ok ? max(isleGround(c, reach, seed), SEA + 1.0) : 0.0;
    ok = ok && B <= (massif ? 130.0 : 150.0);
    if (ok) {
        float v = hashf(pcg(h + 4u));
        float rise = massif ? 260.0 + 160.0 * v : R > 150.0 ? 240.0 + 300.0 * v : 260.0 + 360.0 * v;
        float drip = R > 150.0 ? 1.3 : 1.0;
        float A = floor(B + rise);
        float topC = A + 0.06 * R - 8.5;
        float tilt = isleTilt(R, h);
        float D = min(isleDepth(R, h), (topC - 18.0 - B - (R > 150.0 ? 100.0 : 60.0) - tilt) / drip);
        r.c = c; r.R = R; r.A = A; r.D = D; r.B = B; r.h = h;
        r.K = topC - drip * D - 18.0 - tilt;
        isleFallPlace(r, B, seed);
    }
    out[slot] = r;
}

// Drops every candidate that would overlap a stronger one in plan (a massif, else the larger), judged against the raw
// candidates round it, so no chain of drops depends on the order slots are visited. Reads |R| alone, which it never
// changes.
kernel void resolveIslands(device IsleRec* T [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(ISLE_SLOTS)) return;
    IsleRec r = T[gid];
    float R = abs(r.R);
    if (R <= 0.0) return;
    bool massif = gid >= uint(ISLE_WIN * ISLE_WIN);
    float reach = isleReach(R, r.h);
    float pr = (massif ? 1e5 : 0.0) + R + 0.001 * hashf(r.h);
    bool drop = false;
    int2 n0 = int2(floor(r.c / ISLE_CELL));
    for (int j = -2; j <= 2 && !drop && !massif; j++)
        for (int i = -2; i <= 2 && !drop; i++) {
            int2 cell = n0 + int2(i, j);
            int sl = isleSlot(cell, false);
            if (sl == int(gid)) continue;
            IsleRec q = T[sl];
            float Rq = abs(q.R);
            if (Rq <= 0.0 || any(q.cell != cell)) continue;
            float pq = Rq + 0.001 * hashf(q.h);
            if (pq < pr || (pq == pr && sl > int(gid))) continue;
            drop = length(q.c - r.c) < reach + isleReach(Rq, q.h) + 30.0;
        }
    int2 m0 = int2(floor(r.c / ISLE_MCELL));
    for (int j = -1; j <= 1 && !drop; j++)
        for (int i = -1; i <= 1 && !drop; i++) {
            int2 cell = m0 + int2(i, j);
            int sl = isleSlot(cell, true);
            if (sl == int(gid)) continue;
            IsleRec q = T[sl];
            float Rq = abs(q.R);
            if (Rq <= 0.0 || any(q.cell != cell)) continue;
            float pq = 1e5 + Rq + 0.001 * hashf(q.h);
            if (pq < pr || (pq == pr && sl > int(gid))) continue;
            drop = length(q.c - r.c) < reach + isleReach(Rq, q.h) + 30.0;
        }
    T[gid].R = drop ? -R : R;
}
