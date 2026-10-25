// The flicker key of the lit room at w, stage j's loggia (j = 7: the portals), or 0 where the room is dark. A tower that
// lights any lights one or two, a near the middle of a face, so the open rooms near by and the painted bays far off light
// the same ones. (A share of every bay lit read as an inhabited block.)
static inline uint towerLitRoom(Tower T, float2 w, int j) {
    TowerPlan P = towerPlan(T, w - (j < 7 ? towerStageAxis(T, j) : T.c));
    int bay = j < 7 ? int(floor(P.t / towerBayW(T.h, j, P.face, T.bw) + 0.5)) : 0;
    int n = hashf(pcg(T.h + 61u)) < 0.4 ? 2 : 1;
    int step = T.N > 8 ? 3 : 2;
    // Candidates in turn, skipping bays walled up or fallen, until n are chosen.
    for (int i = 0; i < 8; i++) {
        if (n <= 0) break;
        uint r = pcg(T.h + 62u + 5u * uint(i));
        int jj = hashf(r) < 0.3 ? 7 : int(pcg(r + 1u) % uint(max(T.K - 2, 1)));
        int ff = jj == 7 ? int(pcg(r + 2u) % uint(T.N / step)) * step : int(pcg(r + 2u) % uint(T.N));
        int bb = jj == 7 ? 0 : int(pcg(r + 3u) % 3u) - 1;
        if (jj != 7 && (!towerLoggiaBay(T.h, jj, ff, float(bb)) || towerBayFallen(T.h, jj, ff, float(bb)))) continue;
        if (j == jj && P.face == ff && bay == bb) return r | 1u;
        n--;
    }
    return 0u;
}

static inline Tower towerFromRec(TowerRec R) {
    Tower T;
    T.c = R.a.xy; T.R0 = R.a.z; T.Hb = R.a.w;
    T.Ht = R.b.x; T.K = int(R.b.y); T.N = int(R.b.z); T.rot = R.b.w;
    T.th0 = R.c.x; T.th1 = R.c.y; T.Sy = R.c.z; T.bw = R.c.w;
    T.h = R.d.x; T.kind = int(R.d.y);
    T.kb = 1 + int(pcg(T.h + 9u) % 2u);
    T.reach = T.R0 * 2.8;
    return T;
}

// The plane a side face of tower column wc (normal n) shades as, or 0 where it keeps its own (broken masonry). Every
// plane off the grid is a run of voxel jogs; each jog's faces take the plane they make up: the polygon face's (out of
// the wall, or into the court). A jog's face nearly across the plane would otherwise turn to the light on its own and
// rule the wall with a line per jog.
static float3 towerWallPlane(Tower T, float2 wc, float3 n) {
    float seg = 6.2831853 / float(T.N);
    float2 d0 = wc - T.c;
    // The highest stage whose plan holds the column (none: the plinth, the base's polygon).
    int m = -1;
    float Rm = 0.0;
    TowerPlan Pm = towerPlan(T, d0);
    float2 ax = T.c;
    for (int k = 0; k < 6; k++) {
        if (k >= T.K) break;
        float2 a = towerStageAxis(T, k);
        TowerPlan Pk = towerPlan(T, wc - a);
        float Rk = tierR(T, Pk, k);
        if (Pk.reff < Rk + 0.75) { m = k; Rm = Rk; Pm = Pk; ax = a; }
    }
    // Off the stages (plinth, talus, roots) and deep in a stage (the breaks and the rubble on its treads) the masonry is
    // broken, its faces their own.
    if (m < 0 || (m < T.K - 2 && Rm - Pm.reff > 3.0)) return 0.0;
    float fa = T.rot + seg * (float(Pm.face) + 0.5);
    float2 pn = float2(cos(fa), sin(fa));
    // At a corner a jog's face takes the plane its own centre lies nearer: by the column's centre, or by the plane it is
    // nearer square to, a line of the corner's faces lit as the other face ran down it.
    float fb = fa + (Pm.am > 0.0 ? seg : -seg);
    float2 pb = float2(cos(fb), sin(fb)), q = wc + 0.5 * n.xz - ax;
    if (dot(q, pb) > dot(q, pn)) pn = pb;
    float cp = dot(n.xz, pn);
    if (cp > 0.1) return float3(pn.x, 0.0, pn.y);
    if (cp < -0.1) return float3(-pn.x, 0.0, -pn.y);
    // Nearly across the plane: a jog of the outer face, or of the court's wall within it.
    bool inner = m >= T.K - 2 && Pm.reff < Rm - 4.0;
    return inner ? float3(-pn.x, 0.0, -pn.y) : float3(pn.x, 0.0, pn.y);
}

// A lit room's light: faint, a lamp deep inside.
constant float3 TOWER_ROOM = float3(1.0, 0.55, 0.22) * 0.5;

// The light of a lit room on the faces of an opening of tower R at voxel v of level k (0 where none): only the great
// portals and the loggias' bays, a storey and a tenth tall just under a rim; weathered pockets stay dark. Out of line:
// it runs only at night, and inlined it slows every pixel's shading by day.
static __attribute__((noinline)) float3 towerRoomLight(thread const Ctx& x, TowerRec R, int k, int3 v, int3 n, float2 w, float night) {
    float cs = x.tr.lod[k].w;
    float Hb = R.a.w, Sy = R.c.z;
    int K = int(R.b.y);
    uint seed = x.seed;
    float yv = float(v.y);
    float4 here = columnL(x.tr, k, v.xz);
    float4 e = k == 0 ? extraGaps(x.tr.gx.read(uint2(clamp(v.xz, int2(0), int2(x.tr.size - 1))))) : float4(0);
    float2 g = float2(-1.0);
    if (n.y > 0) {
        if (here.b > here.g && here.g == yv + 1.0) g = here.gb;
        else if (e.y > e.x && e.x == yv + 1.0) g = e.xy;
        else if (e.w > e.z && e.z == yv + 1.0) g = e.zw;
    } else if (n.y < 0) {
        if (here.b > here.g && here.b == yv) g = here.gb;
        else if (e.y > e.x && e.y == yv) g = e.xy;
        else if (e.w > e.z && e.w == yv) g = e.zw;
    } else {
        int2 q = v.xz + n.xz;
        float4 f = columnL(x.tr, k, q);
        float4 fe = k == 0 ? extraGaps(x.tr.gx.read(uint2(clamp(q, int2(0), int2(x.tr.size - 1))))) : float4(0);
        if (f.b > f.g && yv >= f.g && yv < f.b) g = f.gb;
        else if (fe.y > fe.x && yv >= fe.x && yv < fe.y) g = fe.xy;
        else if (fe.w > fe.z && yv >= fe.z && yv < fe.w) g = fe.zw;
    }
    int jr = -1;
    Tower T = towerFromRec(R);
    float2 wc = w + 0.5 * cs;
    if (g.y > g.x) {
        for (int j = 0; j < 6; j++) {
            if (j >= K - 2) break;
            float top = tierY(T, towerPlan(T, wc - towerStageAxis(T, j)), j, seed);
            if (g.y <= top - 3.0 && g.y >= top - 7.0 && abs(g.y - g.x - floor(Sy * 1.1)) <= 2.0 + float(k)) jr = j;
        }
        if (abs(g.x - (Hb + 6.0)) <= float(k) && g.y - g.x >= 26.0) jr = 7;
    }
    if (jr >= 0) {
        uint lh = towerLitRoom(T, wc, jr);
        if (lh != 0u) return TOWER_ROOM * night * (0.8 + 0.2 * hashf(pcg(lh))) * flicker(x.time, float(lh & 255u), seed);
    }
    return 0.0;
}

// A tower's masonry in place of the terrain's surface s: each tower its own old stone laid in storey courses, stained
// under its ledges, lichened, darkened by age at the foot; greened on its treads, hung with ivy under its rims (painted
// where the coarse rings carry no leaves), roots over its plinth, and at night a few rooms lit. Every term is taken in
// the face's own plane and fades to its mean with the footprint.
static void towerSurface(thread const Ctx& x, thread Surface& s, int k, int3 v, int3 n, float2 uv, float fp, float fpA, float3 d) {
    if (s.mat != 0) return;
    float cs = x.tr.lod[k].w;
    float2 w = float2(v.xz * int(cs) + x.tr.origin);
    int i = towerIndexAt(x.towers, w + 0.5 * cs);
    if (i < 0) return;
    TowerRec R = x.towers->r[i];
    float Hb = R.a.w, Ht = R.b.x, Sy = R.c.z, bw = R.c.w, R0 = R.a.z;
    int K = int(R.b.y);
    uint th = R.d.x;
    uint seed = x.seed;
    float2 wq = w + (n.y != 0 ? uv : n.x != 0 ? float2(n.x > 0 ? 1.0 : 0.0, uv.x) : float2(uv.x, n.z > 0 ? 1.0 : 0.0)) * cs;
    float wy = n.y != 0 ? float(v.y + max(n.y, 0)) : float(v.y) + uv.y;
    // Along a side face, the distance along the plan's face it lies on: by the face's own axis, the faces of each jog of
    // a diagonal face would take different stone.
    float2 dq = wq - R.a.xy;
    float aq = atan2(dq.y, dq.x) - R.b.w, sq = 6.2831853 / R.b.z;
    float fq = floor(aq / sq);
    float along = n.y != 0 ? wq.x : length(dq) * sin(aq - sq * fq - 0.5 * sq) + 97.0 * fq;
    Climate c = climateL(x.tr, k, (float2(v.xz) + 0.5) * cs);
    float wet = towerWet(c);
    float dry = smoothstep(0.35, 0.7, desertAmount(c));
    // The plan at the column: out from the base's face, and whether the column stands outside the base (roots).
    float2 dc0 = w + 0.5 * cs - R.a.xy;
    float seg = 6.2831853 / R.b.z;
    float ang = atan2(dc0.y, dc0.x) - R.b.w;
    float am = ang - seg * floor(ang / seg) - 0.5 * seg;
    float reff = length(dc0) * cos(am);
    bool off = reff > R0 + 0.5 * cs;
    // Below the plinth's foot it is the ground it was cut from; the base's walls go down to the ground wherever it falls
    // away.
    if (float(v.y) < Hb - 1.0 && off) return;
    bool root = false;
    if (off && k <= 1) {
        // Roots stand over the plinth and over the talus heaped on it.
        Tower T = towerFromRec(R);
        TowerPlan P0 = towerPlan(T, dc0);
        float inner = R0 * (1.2 + 0.08 * perlin2(P0.dir * 3.7, seed + 520u));
        root = wy > Hb + max(reff < inner ? 6.0 : 3.0, 3.0 + towerTalus(T, P0, w + 0.5 * cs, 1, seed).x) + 0.5;
    }
    // A face of the plan's polygon is a plane: its voxel jogs shade as that plane, and every detail fades with the
    // footprint across the plane, not across the jog's own face, or the jogs' two orientations fade apart in stripes.
    float3 pl = n.y == 0 && !root ? towerWallPlane(towerFromRec(R), w + 0.5 * cs, float3(n)) : float3(0);
    if (any(pl != 0.0)) fpA = fp / max(abs(dot(pl, d)), 0.15);
    // Each tower its own stone: pale limestone, warm sandstone, grey granite or rose.
    float u = hashf(pcg(th + 70u));
    float3 stone = u < 0.4 ? float3(0.27, 0.25, 0.215) : u < 0.7 ? float3(0.29, 0.24, 0.185) : u < 0.9 ? float3(0.24, 0.235, 0.225)
                 : float3(0.265, 0.235, 0.215);
    // Broad drifts of tone over the tower and up it.
    float3 m = stone * mix(1.06, 0.9, saturate(perlin2(wq / 90.0 + float(th & 255u), seed + 530u) + 0.5));
    m = mix(m, float3(0.42, 0.33, 0.24), 0.5 * dry);
    m *= (1.0 + 0.12 * perlin2(float2(along / 37.0, wy / 29.0), seed + 531u)) * (1.0 + 0.2 * perlin2(float2(along / 130.0, wy / 95.0), seed + 546u));
    // Age darkens the foot and the wet country's stone, and grime lies over it in broad blotches.
    m *= mix(1.0, 0.72, smoothstep(Hb + 120.0, Hb, wy) * (0.4 + 0.6 * wet));
    m *= mix(0.78, 1.0, smoothstep(-0.35, 0.25, perlin2(float2(along / 30.0, wy / 70.0), seed + 529u) + 0.4 * perlin2(float2(along / 10.0, wy / 25.0), seed + 538u)));
    // Courses: each storey a different tone, a voxel block's own tone close up; the fallen masonry off the stages more so.
    // (Far off, storey bands read as strata.)
    if (fpA < 0.6) {
        float course = hashf(hash2i(int2(int(floor((wy - Hb) / Sy)), int(th & 1023u)), seed + 532u)) - 0.5;
        m *= 1.0 + 0.12 * course * (1.0 - smoothstep(0.15, 0.6, fpA));
    }
    if (fp < 0.3) {
        uint bh = hash3i(v, seed + 533u);
        m *= 1.0 + (off ? 0.3 : 0.16) * (hashf(bh) - 0.5) * (1.0 - smoothstep(0.05, 0.3, fp));
    }
    // Ashlar on the walls: courses a third of a storey high, blocks of 6 to 13 laid in bond, each its own tone; and the
    // weathered crust spalled off in broad patches of paler, fresher stone. No joints are drawn (a grid of lines reads as
    // toy blocks); both fade to their mean with the footprint.
    if (n.y == 0) {
        float fade = 1.0 - smoothstep(0.3, 1.1, fpA);
        if (fade > 0.0) {
            float ch = max(4.0, floor(Sy / 3.0));
            float row = floor((wy - Hb) / ch);
            float rh = hashf(hash2i(int2(int(row), int(th & 1023u)), seed + 554u));
            float len = 6.0 + floor(8.0 * rh);
            float blk = floor((along + row * 3.7) / len);
            m *= (1.0 + 0.12 * (rh - 0.5) * fade) * (1.0 + 0.26 * (hashf(hash3i(int3(int(blk), int(row), int(th & 1023u)), seed + 555u)) - 0.5) * fade);
        }
        float sp = perlin2(float2(along, wy) / 23.0, seed + 556u) + 0.45 * perlin2(float2(along, wy) / 8.0, seed + 557u);
        float spall = mix(smoothstep(0.15, 0.5, sp), 0.15, smoothstep(0.6, 4.0, fpA));
        m *= mix(0.96, 1.07, spall);
    }
    // Lichen in pale crusts on the drier stone, faded to its mean with the footprint.
    float lich = smoothstep(0.25, 0.45, perlin2(float2(along, wy) / 7.0 + (n.y != 0 ? 3.1 : 0.0), seed + 547u) + 0.25 * perlin2(float2(along, wy) / 31.0, seed + 548u));
    lich = mix(lich, 0.2, smoothstep(0.15, 0.6, fpA)) * (1.0 - 0.6 * wet);
    m *= 1.0 + 0.07 * lich;
    float3 moss = mix(float3(0.06, 0.085, 0.04), float3(0.09, 0.11, 0.055), saturate(perlin2(wq / 13.0, seed + 534u) + 0.5));
    float conifer = smoothstep(0.50, 0.32, c.temp);
    float3 leaf = mix(mix(float3(0.035, 0.07, 0.035), float3(0.065, 0.09, 0.04), saturate(perlin2(wq / 23.0, seed + 93u) * 0.9 + 0.5)),
                      float3(0.025, 0.055, 0.045), conifer);
    float snowline = snowlineAt(c.temp);
    // The ledge (stage rim) above this height, by the stages' nominal tops.
    float ledge = Hb + Ht;
    int jl = -1;
    for (int j = 0; j < 6; j++) {
        if (j >= K - 2) break;
        float yk = Hb + Ht * tierTop(th, j, K);
        if (yk >= wy - (n.y > 0 ? 1.0 : 0.0)) { ledge = yk; jl = j; break; }
    }
    if (root) {
        // Roots: grey bark under moss, mossed over their tops.
        float3 bark = mix(float3(0.17, 0.145, 0.115), float3(0.12, 0.11, 0.09), saturate(perlin2(wq / 5.0, seed + 549u) + 0.5));
        m = n.y > 0 ? mix(bark, moss, 0.6 + 0.3 * wet) : mix(bark, moss, 0.25 * wet);
    } else if (n.y > 0) {
        if (wy > snowline) m = float3(0.31, 0.325, 0.35);
        else {
            // Moss and turf over the treads, more in wet country.
            float cov = smoothstep(-0.25, 0.25, perlin2(wq / 6.0, seed + 535u) + 0.5 * perlin2(wq / 23.0, seed + 536u) + 0.6 * wet - 0.3);
            cov = mix(cov, saturate(0.2 + 0.6 * wet), smoothstep(0.1, 0.6, fpA));
            m = mix(m, moss, cov * (0.25 + 0.75 * wet));
            // Past the 1-voxel map the treads carry no crowns: their share of the tread takes the leaves' colour.
            if (k > 0 && wy < treelineAt(snowline)) m = mix(m, leaf, 0.45 * (0.25 + 0.6 * wet));
        }
    } else if (n.y == 0) {
        float drop = max(ledge - wy, 0.0);
        // Rain off each rim stains the stone below it in soft, broad runs.
        float run = saturate(0.55 + 0.6 * perlin2(float2(along / 13.0, wy / 70.0), seed + 550u));
        run = mix(run, 0.55, smoothstep(0.3, 1.5, fpA));
        m *= 1.0 - 0.42 * run * exp(-drop / 80.0) * step(wy, ledge);
        // The ivy below the rim: hung as leaves on the 1-voxel map, painted past it, over a halo of moss on the stone.
        // Taken in the column in front of the face, as the leaves are hung.
        float2 wf = wq + float2(n.xz) * 0.5;
        if (k == 0) wf = floor(wf) + 0.5;
        float L = towerIvyDrop(wf, R.a.xy, R.b.w, int(R.b.z), th, ledge, drop + 200.0, wet, seed);
        if (k == 0) L = floor(L);
        float e = 0.5 * max(fpA, cs);
        // The mat's edge is ragged in leafy clumps, with stone showing through toward it; both fade to its mean with the
        // footprint (whole voxels on the 1-voxel map).
        float2 lq = k == 0 ? floor(float2(along, wy)) + 0.5 : float2(along, wy);
        float rag = (0.6 * perlin2(lq / 5.0, seed + 552u) + 0.4 * perlin2(lq / 2.2, seed + 553u)) * (1.0 - smoothstep(0.3, 2.0, fpA));
        float cover = (L - drop) / max(L, 1.0) + 0.45 * rag + 0.05;
        float ec = 0.5 * max(fpA, cs) / max(L, 4.0) + 0.04;
        float vine = smoothstep(-ec, ec, cover) * step(1.0, L) * step(drop, L * 1.5 + 2.0) * mix(1.0, 0.8, smoothstep(0.3, 2.0, fpA));
        float halo = smoothstep(L * 1.3 + 8.0 + e, L * 1.3 + 8.0 - e, drop) * step(1.0, L);
        m = mix(m, moss * 1.1, 0.4 * halo);
        if (wy < treelineAt(snowline)) {
            // Leaves in three tones voxel by voxel close up (sunlit tips, mid, the dark between), their mean past that.
            float lt = hashf(hash3i(int3(int(floor(wq.x)), int(floor(wy)), int(floor(wq.y))), seed + 551u));
            float tex = lt < 0.25 ? 0.65 : lt < 0.75 ? 1.0 : 1.25;
            float3 ivy = mix(leaf, float3(0.06, 0.075, 0.04), 0.5);
            m = mix(m, ivy * mix(tex, 1.0, smoothstep(0.05, 0.3, fp)), vine);
        }
        // A damp band of moss along the foot.
        m = mix(m, moss * 1.1, smoothstep(Hb + 30.0, Hb + 6.0, wy) * 0.6 * wet);
        // Past level 1 the coarse columns hold no loggias: their dark bays are painted under each rim, eased in across
        // level 1's morph band as its gaps close.
        float paint = k >= 2 ? 1.0 : k == 1 ? morphAt(x.tr, 1, (float2(v.xz) + 0.5) * cs) : 0.0;
        if (paint > 0.0) {
            Tower T = towerFromRec(R);
            float2 wc = w + 0.5 * cs;
            float hole = 0.0;
            int jr = -1;
            if (jl >= 0) {
                TowerPlan Pj = towerPlan(T, wc - towerStageAxis(T, jl));
                float bwF = towerBayW(th, jl, Pj.face, bw);
                float bayI = floor(Pj.t / bwF + 0.5);
                if (towerLoggiaBay(th, jl, Pj.face, bayI) && !towerBayFallen(th, jl, Pj.face, bayI)) {
                    float dt = abs(Pj.t - bwF * bayI);
                    float hi = tierY(T, Pj, jl, seed) - 4.0, lo = hi - floor(Sy * 1.1);
                    float band = smoothstep(lo - e, lo + e, wy) * smoothstep(hi + e, hi - e, wy);
                    float bay = mix(smoothstep(0.32 * bwF + e, 0.32 * bwF - e, dt), 0.64, smoothstep(0.15 * bwF, 0.4 * bwF, max(fpA, cs)));
                    hole = band * bay;
                    jr = jl;
                }
            }
            // The great portals in the base's faces, and the passage under arches.
            TowerPlan P0 = towerPlan(T, wc - T.c);
            if (wy < Hb + 40.0 + e && P0.reff < R0 + 6.0 + cs) {
                float tw = 7.0 + 0.5 * cs;
                float hp = smoothstep(Hb + 40.0 + e, Hb + 40.0 - e, wy + 4.0 * (P0.t / 7.0) * (P0.t / 7.0));
                if (P0.face % (T.N > 8 ? 3 : 2) == 0) {
                    float por = smoothstep(tw + e, tw - e, abs(P0.t)) * hp;
                    if (por > hole) { hole = por; jr = 7; }
                }
                if ((th & 128u) != 0u) {
                    float2 pd = float2(cos(T.th1 + 1.5708), sin(T.th1 + 1.5708));
                    float off = abs((wc - T.c).x * pd.y - (wc - T.c).y * pd.x);
                    float hpass = Hb + 36.0 + floor(0.2 * R0);
                    float pas = smoothstep(tw + e, tw - e, off) * smoothstep(hpass + e, hpass - e, wy + 6.0 * (off / 7.0) * (off / 7.0));
                    hole = max(hole, pas);
                }
            }
            m = mix(m, m * 0.22, hole * paint);
            // Its lit rooms, as the open bays near by light them, glowing on far into the night.
            float night = saturate(x.lights.w * 4.0) * paint * (1.0 - smoothstep(2500.0, 4500.0, length((float2(v.xz) + 0.5) * cs - x.tr.cam)));
            if (night > 0.0 && R.d.z != 0u && jr >= 0) {
                uint lh = towerLitRoom(T, wc, jr);
                if (lh != 0u)
                    s.emit = TOWER_ROOM * night * hole * (0.8 + 0.2 * hashf(pcg(lh))) * flicker(x.time, float(lh & 255u), seed);
            }
        }
    } else {
        m *= 0.85;
    }
    s.albedo = m;
    s.tread = m;
    s.face = m;
    s.pale = n.y > 0 && wy > snowline && !root ? 1.0 : 0.0;
    s.slope = 0.0;
    s.wall = 1.0;
    if (any(pl != 0.0)) s.nrm = pl;
    s.level = n.y > 0 ? 1.0 : 0.0;
    // A few rooms lit at night on some towers: the faces of an opening (back wall, reveals, sill and lintel) on the
    // 1-voxel map, faded out across its morph band, where the painted bays take over.
    float night = saturate(x.lights.w * 4.0) * (k == 0 ? 1.0 : k == 1 ? 1.0 - morphAt(x.tr, 1, (float2(v.xz) + 0.5) * cs) : 0.0);
    if (night > 0.0 && R.d.z != 0u) {
        float3 e = towerRoomLight(x, R, k, v, n, w, night);
        if (any(e > 0.0)) s.emit = e;
    }
}
