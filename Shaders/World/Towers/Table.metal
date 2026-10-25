// The shading table (fragment buffer 9; stage 3's giants hold 8): the towers standing nearest the camera (Sources/Strata/World/Towers.swift fills it from placeTowers).
// a: axis xz, R0, Hb; b: Ht, K, N, rot; c: th0, th1, Sy, bw; d: hash, kind, lit at night, member index | moisture << 8 |
// temperature << 16 (bytes; for listings).
struct TowerRec { float4 a; float4 b; float4 c; uint4 d; };
struct TowerTable { int4 n; TowerRec r[TOWER_TABLE]; };

// The table record of the standing tower whose plinth holds world point w, or -1.
static inline int towerIndexAt(constant TowerTable* tt, float2 w) {
    int n = min(tt->n.x, TOWER_TABLE);
    for (int i = 0; i < n; i++) {
        float4 a = tt->r[i].a;
        float2 d = w - a.xy;
        float R = a.z * 1.45;
        if (dot(d, d) < R * R) return i;
    }
    return -1;
}


// One thread per world cell info.xy + gid of an info.z-square grid: its group's members, TOWER_MEMBERS records a cell
// (Ht 0: none).
kernel void placeTowers(device TowerRec* out [[buffer(0)]],
                        constant int4& info [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(info.z) || gid.y >= uint(info.z)) return;
    uint seed = uint(info.w);
    uint base = (gid.y * uint(info.z) + gid.x) * uint(TOWER_MEMBERS);
    TowerRec none;
    none.a = 0; none.b = 0; none.c = 0; none.d = 0;
    for (int m = 0; m < TOWER_MEMBERS; m++) out[base + uint(m)] = none;
    TowerGroup g;
    if (!towerGroupIn(info.xy + int2(gid), seed, g) || !towerGroupStands(g, seed)) return;
    for (int m = 0; m < TOWER_MEMBERS; m++) {
        if (m >= g.n) break;
        Tower T = towerMember(g, m);
        T.Hb = towerBase(T, seed);
        if (T.Hb > TOWER_BASE_MAX) continue;
        TowerRec r;
        r.a = float4(T.c, T.R0, T.Hb);
        r.b = float4(T.Ht, float(T.K), float(T.N), T.rot);
        r.c = float4(T.th0, T.th1, T.Sy, T.bw);
        Climate cl = climate(T.c, seed);
        r.d = uint4(T.h, uint(T.kind), hashf(pcg(T.h + 60u)) < 0.35 ? 1u : 0u,
                    uint(m) | (uint(cl.moist * 255.0) << 8) | (uint(cl.temp * 255.0) << 16));
        out[base + uint(m)] = r;
    }
}
