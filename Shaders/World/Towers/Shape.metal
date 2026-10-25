// The tower's share of column wf: its plinth, stages, shell and openings over the terrain column col. s: the column
// width this is sampled for (coarse level k: 2^k); from 4 on, no hollows or openings, which the samples would alias.
static inline Col towerShape(Col col, Tower T, float2 wf, int s, uint seed) {
    float2 d0 = wf - T.c;
    TowerPlan P0 = towerPlan(T, d0);
    // Rubble of the fallen crown in a lobe toward th0: mounds only (loose blocks read as toys).
    float2 f0 = float2(cos(T.th0), sin(T.th0));
    float along = dot(d0, f0), across = abs(dot(d0, float2(-f0.y, f0.x)));
    float lobe = smoothstep(T.R0 * 0.9, T.R0 * 1.4, along) * smoothstep(T.R0 * 2.7, T.R0 * 1.7, along)
               * smoothstep(T.R0 * (0.8 + 0.003 * along), T.R0 * 0.3, across);
    if (lobe > 0.0) {
        float mnd = perlin2(wf / 23.0, seed + 510u) * 0.7 + perlin2(wf / 9.0, seed + 511u) * 0.3;
        col.h = max(col.h, floor(col.h + 16.0 * lobe * saturate(mnd + 0.4)));
    }
    // The plinth: two broad platforms, cut and filled level, the outer one broken up and gone in places.
    if (P0.reff >= T.R0 * 1.42) return col;
    float inner = T.R0 * (1.2 + 0.08 * perlin2(P0.dir * 3.7, seed + 520u));
    // Where the ground falls away there is no plinth: built out over it, it stood on a sheer pedestal of ground.
    float ground = col.h;
    if (ground > T.Hb - 12.0) col.h = P0.reff < inner ? T.Hb + 6.0 : perlin2(wf / 37.0, seed + 521u) > -0.15 ? T.Hb + 3.0 : col.h;
    col.gl = 0; col.gh = 0; col.lo1 = 0; col.hi1 = 0; col.lo2 = 0; col.hi2 = 0; col.carved = false;
    // The highest stage whose plan covers the column, and how far outside each stage's plan it lies.
    int m = -1;
    float Rm = 0.0;
    TowerPlan Pm = P0;
    float out[6] = { 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < 6; k++) {
        if (k >= T.K) break;
        TowerPlan Pk = towerPlan(T, wf - towerStageAxis(T, k));
        float Rk = tierR(T, Pk, k);
        out[k] = Pk.reff - Rk;
        if (Pk.reff < Rk) { m = k; Rm = Rk; Pm = Pk; }
    }
    if (m < 0) {
        float root = s < 4 ? towerRootH(T, P0, seed) : 0.0;
        float2 tal = towerTalus(T, P0, wf, s, seed);
        col.h = max(col.h + root, floor(mix(min(ground, T.Hb + 3.0), T.Hb + 3.0, tal.y) + tal.x));
        return col;
    }
    bool shell = m >= T.K - 2;
    float yS = tierY(T, Pm, T.K - 3, seed);                       // the shell stands on this
    float y0 = m == 0 ? T.Hb + 6.0 : shell ? yS : tierY(T, Pm, m - 1, seed);
    float H = shell ? crownY(T, Pm, yS, T.Hb + T.Ht, seed) : tierY(T, Pm, m, seed);
    float wallIn = Rm - Pm.reff;
    // Whole sections of a stage's outer wall have fallen from the top, in broad bites, so no two stages share an outline;
    // each bite a few dozen courses at most (cut down a whole stage, a bite's sides ran down the face as crisp lines).
    float bite = 30.0 * saturate(perlin2(Pm.dir * 2.6 + float(m) * 2.3, seed + 506u) * 1.7 - 0.2) * (m == 0 ? 0.6 : 1.0);
    if (!shell && bite > 4.0 && wallIn < bite * (0.7 + 0.5 * perlin2(wf / 9.0, seed + 533u)))
        H = max(y0 + 3.0, H - floor(min((H - y0) * (0.15 + 0.01 * bite), 25.0 + 2.0 * bite) / 5.0) * 5.0);
    // Chunks bitten out of the corners, part of a stage's height, their edges ragged. (Stepped down toward the corner,
    // they drew more setbacks.)
    float seg = 6.2831853 / float(T.N);
    uint ch = hash3i(int3((Pm.face + (Pm.am > 0.0 ? 1 : 0)) % T.N, m, int(T.h & 1023u)), seed + 515u);
    if (hashf(ch) < 0.65) {
        float hf = Rm * tan(0.5 * seg);
        float drop = min((H - y0) * (0.25 + 0.5 * hashf(pcg(ch + 1u))), 20.0 + 50.0 * hashf(pcg(ch + 2u)));
        float dc = hf - abs(Pm.t) + (s < 8 ? 7.0 * perlin2(wf / 7.0, seed + 523u) : 0.0) + 0.3 * drop * perlin2(wf / 13.0, seed + 530u);
        if (dc < hf * (0.25 + 0.45 * hashf(pcg(ch))))
            H = max(y0, H - floor(drop / 5.0) * 5.0);
    }
    // The parapet along each rim broken down a course or three, in runs of a few columns.
    if (!shell && wallIn < 8.0 + 0.05 * T.R0)
        H = max(y0, H - 5.0 * floor(3.5 * saturate(perlin2(wf / 11.0, seed + 531u) + (s < 4 ? 0.6 * perlin2(wf / 4.0, seed + 532u) : 0.0) + 0.15)));
    // A great collapse on most towers: a wedge of the upper storeys fallen on the crown's side, its floor broken in
    // courses, the outer skin gone deeper than the core.
    uint hc = pcg(T.h + 30u);
    if (hashf(hc) < 0.75) {
        float thc = T.th0 + 0.9 * (hashf(pcg(hc)) - 0.5);
        float da = P0.a - thc;
        da -= 6.2831853 * floor(da / 6.2831853 + 0.5);
        float u = abs(da) / (0.3 + 0.4 * hashf(pcg(hc + 1u))) + 0.25 * perlin2(wf / 23.0, seed + 525u);
        if (u < 1.0) {
            float yF = T.Hb + T.Ht * (0.3 + 0.35 * hashf(pcg(hc + 2u)));
            float yc = yF + (T.Hb + T.Ht - yF) * u * u + 1.2 * wallIn;
            uint bh = hash2i(int2(floor(wf / 6.0)), seed + 526u);
            yc = floor(yc / 5.0) * 5.0 + 5.0 * floor(3.0 * hashf(bh) - 1.0);
            H = min(H, max(yc, T.Hb + 6.0));
        }
    }
    // Weathering: the top crumbles back course by course where the worn skin has gone, a few courses at most: run down a
    // whole stage, the worn skin cut grooves down the face, a crisp line each. (Pockets cut in the face read as dark
    // bricks or windows punched in it, so none are.)
    if (wallIn < 16.0 && s < 8) {
        float yl = max(y0, H - 25.0);
        for (float yc = y0 + 5.0 * (ceil((H - y0) / 5.0) - 1.0); yc > yl; yc -= 5.0) {
            if (wallIn >= towerErosion(T, Pm, Rm, yc + 2.5, seed)) break;
            H = yc;
        }
    }
    // Rubble heaped in mounds on the tread at the foot of the stage above.
    if (!shell && H > y0 + 3.0) {
        float L = max(8.0, 0.45 * T.R0 * (towerTierScale(T, m) - towerTierScale(T, m + 1)));
        float f = saturate(1.0 - out[m + 1] / L);
        if (f > 0.0)
            H += floor((3.0 + 0.1 * T.R0) * f * sqrt(f) * saturate(0.35 + perlin2(wf / 19.0, seed + 527u) + (s < 8 ? 0.4 * perlin2(wf / 7.0, seed + 528u) : 0.0)));
    }
    float gaps[6] = { 0, 0, 0, 0, 0, 0 };
    int ng = 0;
    // A stage fallen away beneath the one above (the wound): a sector of stage kb's outer mass is gone, the stages over
    // it overhang.
    if (T.kb <= m && T.kb < T.K - 2) {
        TowerPlan Pb = towerPlan(T, wf - towerStageAxis(T, T.kb));
        float Rk = tierR(T, Pb, T.kb);
        float lobeW = 0.5 + 0.5 * cos(Pb.a - T.th1);
        float depth = T.R0 * 0.35 * smoothstep(0.82, 0.97, lobeW) * saturate(0.7 + perlin2(Pb.dir * 8.0, seed + 513u));
        if (Pb.reff > Rk - depth) {
            float yb0 = tierY(T, Pb, T.kb - 1, seed) + 8.0 + floor(10.0 * abs(perlin2(Pb.dir * 6.0, seed + 514u)));
            float yb1 = tierY(T, Pb, T.kb, seed) - 4.0;
            if (m == T.kb) H = min(H, yb0);
            else if (s < 4 && yb1 - yb0 > 6.0) { gaps[0] = yb0; gaps[1] = yb1; ng = 1; }
        }
    }
    if (shell && Pm.reff < Rm - max(10.0, 1.5 * float(s))) {
        // The shell is open to the sky, its floor a court where a grove can root. Every ring keeps the court: filled, its
        // top would follow the crown's broken rim and show a radial pattern from above. The wall thickens with the
        // samples' spacing, or whole runs of the rim would fall between them and the crown lose its silhouette.
        H = min(H, yS + 3.0);
    }
    if (s < 4) {
        // A passage through the base under arches, in some towers.
        if ((T.h & 128u) != 0u && ng < 3) {
            float2 pd = float2(cos(T.th1 + 1.5708), sin(T.th1 + 1.5708));
            float off = abs(d0.x * pd.y - d0.y * pd.x);
            if (off < 7.0 && P0.reff < T.R0) {
                float hi = T.Hb + 36.0 + floor(0.2 * T.R0) - floor(6.0 * (off / 7.0) * (off / 7.0));
                gaps[2 * ng] = T.Hb + 6.0; gaps[2 * ng + 1] = min(hi, H - 4.0); ng++;
            }
        }
        // Great portals in the base's faces.
        if (m == 0 && P0.face % (T.N > 8 ? 3 : 2) == 0 && abs(Pm.t) < 7.0 && wallIn < 14.0 && ng < 3) {
            float hi = T.Hb + 40.0 - floor(4.0 * (Pm.t / 7.0) * (Pm.t / 7.0));
            gaps[2 * ng] = T.Hb + 6.0; gaps[2 * ng + 1] = min(hi, H - 4.0); ng++;
        }
        bool through = shell && Pm.reff >= Rm - 10.0;
        float bwF = towerBayW(T.h, m, Pm.face, T.bw);
        float bay = floor(Pm.t / bwF + 0.5);
        float dt = abs(Pm.t - bay * bwF);
        int ns = int((H - y0) / T.Sy);
        // A loggia under each stage's rim: a storey of deep arches round the stage, wherever its top still stands.
        if (!shell && wallIn < 6.0 && dt < 0.32 * bwF && ng < 3 && towerLoggiaBay(T.h, m, Pm.face, bay)) {
            float top = tierY(T, Pm, m, seed);
            float hi = top - 4.0 - (dt > 0.25 * bwF ? 2.0 : dt > 0.18 * bwF ? 1.0 : 0.0);
            float lo = top - 4.0 - floor(T.Sy * 1.1);
            if (hi <= H - 3.0 && lo > y0 + 4.0) {
                if (towerBayFallen(T.h, m, Pm.face, bay)) H = lo;
                else { gaps[2 * ng] = lo; gaps[2 * ng + 1] = hi; ng++; }
            }
        }
        // Arcades through the shell's wall: on each face a few of every other storey, bays missing from them. (Single
        // windows in the stage walls read as slots punched in a slab, a grid of them as a block of flats.)
        if (through && dt < 3.0) {
            for (int st = 40; st >= 0; st--) {
                if (ng >= 3) break;
                if (st >= ns || (st & 1) != 0 || hashf(hash3i(int3(Pm.face, st, 9), T.h + 125u)) > 0.4
                    || hashf(hash3i(int3(Pm.face, int(bay), st + 16), T.h + 121u)) < 0.3) continue;
                float lo = y0 + float(st) * T.Sy + 3.0;
                float hi = lo + floor(T.Sy * 0.75) - (dt > 2.0 ? 2.0 : dt > 1.0 ? 1.0 : 0.0);
                if (hi > H - 3.0) continue;
                gaps[2 * ng] = lo; gaps[2 * ng + 1] = hi; ng++;
            }
        }
    }
    col.h = min(max(col.h, floor(H)), TERRAIN_CEIL);
    // Highest gap first.
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++)
            if (gaps[2 * j + 1] > gaps[2 * i + 1]) {
                float a0 = gaps[2 * i], a1 = gaps[2 * i + 1];
                gaps[2 * i] = gaps[2 * j]; gaps[2 * i + 1] = gaps[2 * j + 1];
                gaps[2 * j] = a0; gaps[2 * j + 1] = a1;
            }
    col.gl = floor(gaps[0]); col.gh = floor(gaps[1]);
    col.lo1 = floor(gaps[2]); col.hi1 = floor(gaps[3]);
    col.lo2 = floor(gaps[4]); col.hi2 = floor(gaps[5]);
    col.carved = col.gh > col.gl;
    return col;
}

static inline Col towerColumn(Col col, int2 w, uint seed, int s) {
    float2 wf = float2(w) + 0.5;
    Tower T;
    TowerGroup g;
    if (!towerFind(wf, 0.0, seed, T, g)) return col;
    // Most of a member's reach is open ground, off its plinth and its rubble: those columns are let go before asking
    // whether the group stands, which costs two terrain columns and the province fields.
    float2 d0 = wf - T.c, f0 = float2(cos(T.th0), sin(T.th0));
    float along = dot(d0, f0), across = abs(dot(d0, float2(-f0.y, f0.x)));
    bool rubble = along > T.R0 * 0.9 && along < T.R0 * 2.7 && across < T.R0 * (0.8 + 0.003 * along);
    if (!rubble && towerPlan(T, d0).reff >= T.R0 * 1.42) return col;
    if (!towerGroupStands(g, seed)) return col;
    T.Hb = towerBase(T, seed);
    if (T.Hb > TOWER_BASE_MAX) return col;
    return towerShape(col, T, wf, s, seed);
}

// Within margin of a standing tower's reach (sites keep off them).
static inline bool towerNear(float2 wf, float margin, uint seed) {
    Tower T;
    TowerGroup g;
    return towerFind(wf, margin, seed, T, g) && towerGroupStands(g, seed) && towerBase(T, seed) <= TOWER_BASE_MAX;
}

// How wet a tower's country is, for its greenery.
static inline float towerWet(Climate c) {
    return smoothstep(0.25, 0.65, c.moist) * (1.0 - smoothstep(0.35, 0.7, desertAmount(c)));
}

// How far the ivy hangs below a rim at height rim, at most room, at world xz w on a face of the tower whose axis is c
// (plan rotation rot, N sides, hash th): mats several columns wide with bare stone between, their lower edges drifting
// smoothly along the face, a whole column at a time on the 1-voxel map (q quantised by the caller).
static inline float towerIvyDrop(float2 w, float2 c, float rot, int N, uint th, float rim, float room, float wet, uint seed) {
    float2 d = w - c;
    float seg = 6.2831853 / float(N);
    float a = atan2(d.y, d.x) - rot;
    float fi = floor(a / seg);
    float t = length(d) * sin(a - seg * fi - 0.5 * seg) + 97.0 * fi;
    float2 q = float2(t, rim * 0.4 + float(th & 255u) * 7.3);
    float mat = perlin2(q / 34.0, seed + 560u) + 0.35 * perlin2(q / 14.0, seed + 561u);
    return min(room, 20.0 + 90.0 * wet) * smoothstep(-0.25, 0.2, mat + 0.4 * wet - 0.1) * (0.75 + 0.25 * perlin2(q / 9.0, seed + 562u));
}
