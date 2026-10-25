// Heights are rgba16Float/r16Float everywhere (map, rings, gaps, first leaf, the CPU copy): integers are exact only
// up to 2048, so land stops at 2000 and trees on the highest summit still fit.
constant float TERRAIN_CEIL = 2000.0;

// Lift over the lowland, and the signals a later wall/overhang pass needs. Provinces about 30% of the land, each
// one landform across its extent, separated by lowland 6-12k wide: alpine ranges, tablelands with sheer
// escarpments, karst pillar forests; a low-frequency river net cuts gorges through them, and a rare tepui stands
// on its own. U is 0 outside them, so the lowland there is exactly as before. p is seed-offset world position.
// lipU: the lift the column would have on the cap rock behind an escarpment's edge (0: no lip here); lipR: how far the
// cap reaches out over the soft beds, lipU01: the column's place under it (0 at the wall line, 1 at the lip's edge).
// notchFoot, notchH: a notch cut round a karst tower's foot, its floor that far above the lowland ground (-1: at the
// water line) and its height at this column (0: none).
struct Mega { float U; float wK; float lipU; float lipR; float lipU01; float notchFoot; float notchH; };

// The tableland field, also tapped beside a column to measure its distance to an escarpment in voxels.
static inline float tableField(float2 p, float M, uint seed) {
    float2 tw = float2(fbm2(p / 1500.0, 3, seed + 220u), fbm2(p / 1500.0 + 5.3, 3, seed + 221u)) * 700.0
              + float2(perlin2(p / 260.0, seed + 222u), perlin2(p / 260.0 + 3.1, seed + 223u)) * 90.0;
    return fbm2((p + tw) / 6000.0, 3, seed + 224u) * 1.4 + 0.5 + 0.5 * (M - 0.5);
}

// The warped position the provinces are laid out on, and their field m there: they stand where it passes 0.5.
static inline float megaField(float2 p, uint seed, thread float2& q) {
    q = p + float2(fbm2(p / 7000.0, 2, seed + 200u), fbm2(p / 7000.0 + 71.3, 2, seed + 201u)) * 3000.0;
    return fbm2(q / 16000.0, 3, seed + 202u) * 1.6 + 0.5;
}

// A pinned biome keeps only its own landforms: tableland and tepuis in desert and painted hills (4, 5), karst in lagoons
// and jungle (7, 8), ranges alone in the rest.
static inline bool pinnedTable(uint b) { return b == 4 || b == 5; }
static inline bool pinnedKarst(uint b) { return b == 7 || b == 8; }

// A province's alpine share (the rest is tableland and karst), at its warped position q.
static inline float megaAlpine(float2 q, uint seed) {
    uint b = pinnedBiome(seed);
    if (b != 0 && !pinnedTable(b) && !pinnedKarst(b)) return 1.0;
    return smoothstep(-0.12, 0.12, perlin2(q / 22000.0, seed + 203u));
}

// The tepui of p's 30k cell, if one stands there: its centre c, rim radius R and hash hc.
static inline bool tepuiIn(float2 p, uint seed, thread float2& c, thread float& R, thread uint& hc) {
    const float CELL = 30000.0;
    int2 cell = int2(floor(p / CELL));
    hc = hash2i(cell, seed + 240u);
    uint b = pinnedBiome(seed);
    if (hashf(hc) >= 0.35 || (b != 0 && !pinnedTable(b))) return false;
    c = (float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(hc)), hashf(pcg(hc + 1u)))) * CELL;
    R = 900.0 + 600.0 * hashf(pcg(hc + 2u));
    return true;
}

static inline Mega megaLift(float2 p, float lagoon, uint seed) {
    Mega g;
    g.U = 0.0;
    g.wK = 0.0;
    g.lipU = 0.0; g.lipR = 0.0; g.lipU01 = 0.0; g.notchFoot = 0.0; g.notchH = 0.0;
    float lipU = 0.0, uT = 0.0;
    float2 q;
    float m = megaField(p, seed, q);
    float M = smoothstep(0.50, 0.80, m);
    if (M > 0.0) {
        float b = perlin2(q / 22000.0 + 9.7, seed + 204u);
        float wA = megaAlpine(q, seed);
        uint pb = pinnedBiome(seed);
        float wT = (1.0 - wA) * (pinnedTable(pb) ? 1.0 : pinnedKarst(pb) ? 0.0 : smoothstep(-0.12, 0.12, b));
        float wK = (1.0 - wA) * (1.0 - wT);
        float core = smoothstep(0.55, 0.95, m);
        float U = 0.0;
        if (wA > 0.0) {
            // Ridged multifractal, domain-warped into ranges; its finest octave is 200 wide, so the 16-voxel ring
            // carries every octave. The pow keeps the valleys between ridges low.
            float2 aw = float2(fbm2(p / 1800.0, 3, seed + 210u), fbm2(p / 1800.0 + 13.1, 3, seed + 211u)) * 500.0;
            float r = ridged2((p + aw) / 3200.0, 5, seed + 212u);
            U += wA * M * (120.0 + 1450.0 * pow(saturate(r), 1.8)) * mix(0.55, 1.0, core);
        }
        if (wT > 0.0) {
            // Two tiers, each a sloping talus apron (a quarter of its height) under a sheer cap rock; outliers of the
            // field near a threshold stand as mesas, then buttes.
            float sv = tableField(p, M, seed);
            float s1 = 0.25 * smoothstep(0.38, 0.45, sv) + 0.75 * smoothstep(0.45, 0.456, sv);
            float s2 = 0.25 * smoothstep(0.55, 0.62, sv) + 0.75 * smoothstep(0.62, 0.626, sv);
            float roll = 18.0 * fbm2(p / 400.0, 2, seed + 225u);
            float fT = wT * smoothstep(0.0, 0.3, M);
            uT = fT * (430.0 * s1 + 380.0 * s2 + roll * s1);
            U += uT;
            // Caprock lips: where the cap rock is hard it reaches out over the soft beds of the wall below, by a distance in
            // voxels (the field's offset from the top of the wall over its gradient), so the overhang is a slab of even
            // reach rather than one that thins to nothing where the field is flat. The column takes the cap's own height.
            float reachK = smoothstep(0.30, 0.75, perlin2(p / 520.0, seed + 262u) * 0.9 + 0.5);
            float thr = sv < 0.54 ? 0.456 : 0.626;
            if (fT > 0.3 && reachK > 0.0 && sv < thr && sv > thr - 0.03) {
                float e = 3.0;
                float gx = tableField(p + float2(e, 0), M, seed) - sv, gz = tableField(p + float2(0, e), M, seed) - sv;
                float grad = max(length(float2(gx, gz)) / e, 1e-6);
                float delta = (thr - sv) / grad;
                float R = 40.0 * reachK;
                if (delta < R && R > 4.0) {
                    float c1 = thr < 0.5 ? 1.0 : s1, c2 = thr < 0.5 ? s2 : 1.0;
                    lipU = fT * (430.0 * c1 + 380.0 * c2 + roll * c1);
                    g.lipR = R;
                    g.lipU01 = delta / R;
                }
            }
        }
        if (wK > 0.0) {
            // Tower karst. A slow field gathers the towers into clusters (fengcong: peaks crowding on a shared rocky
            // pedestal) with open plains between, where towers stand alone (fenglin). Two populations: slender
            // towers on a ~50 lattice and massive ones on a ~130 lattice, so girth and spacing vary; each tower's
            // height comes from noise on its own scale, so neighbours differ. Each stands on a talus apron, its sheer
            // shaft ends in a rounded, craggy crown, and thin needles rise among the fringe towers.
            float P = 280.0 + 260.0 * saturate(fbm2(q / 5000.0, 2, seed + 230u) * 1.5 + 0.5);
            float2 kw = float2(perlin2(p / 60.0, seed + 231u), perlin2(p / 60.0 + 7.7, seed + 232u)) * 14.0;
            float clus = saturate(fbm2(q / 750.0, 2, seed + 239u) * 3.0 + 0.3 + 0.2 * (core - 0.5));
            float dens = clus * clus * (3.0 - 2.0 * clus);
            float c1 = perlin2((p + kw) / 52.0, seed + 233u);
            float c2 = perlin2((p + kw) / 23.0, seed + 235u);
            float cB = perlin2((p + kw * 2.5) / 130.0, seed + 245u);
            float thr = mix(0.5, 0.22, dens) + 0.16 * perlin2(p / 170.0, seed + 248u);
            float skirt = smoothstep(thr - 0.14, thr, c1);
            float big = smoothstep(thr, thr + 0.02, c1);
            float crown = smoothstep(thr + 0.01, thr + 0.18, c1);
            float thrB = mix(0.42, 0.27, dens);
            float skirtB = smoothstep(thrB - 0.16, thrB, cB);
            float bigB = smoothstep(thrB, thrB + 0.015, cB);
            float crownB = smoothstep(thrB + 0.01, thrB + 0.16, cB);
            float thin = smoothstep(0.52, 0.54, c2) * (1.0 - dens) * smoothstep(0.1, 0.4, perlin2(p / 900.0, seed + 236u) + 0.3);
            float hv = 0.25 + 0.75 * saturate(perlin2(p / 61.0, seed + 234u) * 1.3 + 0.5);
            float hB = 0.45 + 0.5 * saturate(perlin2(p / 150.0, seed + 246u) * 1.2 + 0.5);
            float crag = perlin2(p / 11.0, seed + 237u) + 0.5 * perlin2(p / 4.7, seed + 238u);
            float shaft = P * M * hv * (0.10 * skirt + 0.62 * big + 0.28 * crown * (2.0 - crown)) * 0.5;
            float shaftB = P * M * hB * (0.12 * skirtB + 0.58 * bigB + 0.30 * crownB * (2.0 - crownB));
            // The cluster's pedestal: a rolling rock base the crowded towers share.
            float ped = P * M * 0.22 * dens * smoothstep(-0.15, 0.35, perlin2((p + kw * 3.0) / 210.0, seed + 247u));
            float kh = max(max(shaft, shaftB) + 9.0 * crag * max(big, bigB), P * M * hv * 0.8 * thin);
            kh = max(kh, ped + 0.5 * kh);
            U += wK * max(kh, 0.0);
            // Notches: the rock just inside a big tower's foot is undercut, a rounded nip at the water line or above the
            // talus, its depth a distance in voxels inside the tower's edge (the field over its gradient), so a needle
            // thinner than two notches is pierced right through.
            bool useB = shaftB > shaft;
            float baseT = P * M * (useB ? hB : hv);
            float F = useB ? cB : c1, tF = (useB ? thrB : thr) + 0.01;
            if (wK > 0.5 && baseT > 110.0 && F > tF && F < tF + 0.4 && thin < 0.5) {
                float2 e1 = float2(1, 0), e2 = float2(0, 1);
                float2 k1 = float2(perlin2((p + e1) / 60.0, seed + 231u), perlin2((p + e1) / 60.0 + 7.7, seed + 232u)) * 14.0;
                float2 k2 = float2(perlin2((p + e2) / 60.0, seed + 231u), perlin2((p + e2) / 60.0 + 7.7, seed + 232u)) * 14.0;
                float Fx = useB ? perlin2((p + e1 + k1 * 2.5) / 130.0, seed + 245u) : perlin2((p + e1 + k1) / 52.0, seed + 233u);
                float Fz = useB ? perlin2((p + e2 + k2 * 2.5) / 130.0, seed + 245u) : perlin2((p + e2 + k2) / 52.0, seed + 233u);
                float grad = max(length(float2(Fx - F, Fz - F)), 1e-6);
                float delta = (F - tF) / grad;
                float D = 4.0 + 8.0 * saturate(perlin2(p / 150.0, seed + 265u) + 0.5);
                if (delta < D) {
                    float v = delta / D;
                    float hn = 8.0 + 12.0 * saturate(perlin2(p / 120.0, seed + 266u) + 0.5);
                    g.notchH = hn * sqrt(1.0 - v * v);
                    g.notchFoot = wK * max(baseT * (useB ? 0.12 : 0.05), ped + baseT * (useB ? 0.06 : 0.025));
                }
            }
        }
        // Lagoon country keeps only its pillars standing in the water.
        float keepU = mix(1.0, wK, lagoon);
        // Rivers cut gorges: slots through tableland and karst, broader valleys through ranges.
        float gorge = abs(fbm2(q / 5000.0, 3, seed + 250u));
        keepU *= smoothstep(0.008, mix(0.035, 0.12, wA), gorge);
        // No lips over a gorge's slot, where the cap would roof the river.
        if (lipU > 0.0) lipU = keepU > 0.98 ? (U - uT + lipU) * keepU : 0.0;
        U *= keepU;
        g.notchFoot *= keepU;
        if (keepU < 0.98) g.notchH = 0.0;
        g.U = U;
        g.wK = wK;
        g.lipU = lipU;
    }
    // Tepui: in about one 30k cell in three, a sheer-walled table mountain over a talus skirt.
    float2 c;
    float R;
    uint hc;
    if (tepuiIn(p, seed, c, R, hc)) {
        float dd = length(p - c);
        if (dd < R + 1600.0) {
            // Lobes and buttresses, so the rim reads as eroded rock, not a drum.
            float d = dd + 520.0 * fbm2(p / 1100.0, 3, seed + 241u) + 140.0 * perlin2(p / 330.0, seed + 244u)
                    + 50.0 * perlin2(p / 140.0, seed + 242u)
                    + 110.0 * smoothstep(0.16, 0.0, abs(perlin2(p / 280.0, seed + 245u)));
            float Hl = 1450.0 + 300.0 * hashf(pcg(hc + 3u));
            // Seen from far off a sheer drum is a rectangle: the wall steps back at two ledges whose heights and depths
            // wander round the rim, and canyons notch the crown, so the outline is never two verticals under a flat top.
            float a1 = 0.46 + 0.12 * perlin2(p / 900.0, seed + 246u);
            float a2 = 0.70 + 0.12 * perlin2(p / 700.0 + 9.3, seed + 246u);
            float r1 = R - 6.0 - R * (0.04 + 0.22 * saturate(perlin2(p / 600.0, seed + 247u) + 0.5));
            float r2 = r1 - 16.0 - R * (0.04 + 0.22 * saturate(perlin2(p / 520.0 + 4.7, seed + 248u) + 0.5));
            float crown = 240.0 * fbm2(p / 800.0, 3, seed + 243u) - 380.0 * smoothstep(0.09, 0.0, abs(perlin2(p / 520.0, seed + 249u)));
            crown = floor(crown / 70.0) * 70.0 + 70.0 * smoothstep(0.7, 1.0, fract(crown / 70.0));
            float t = 0.32 * pow(smoothstep(R + 900.0, R + 10.0, d), 1.6) + (a1 - 0.32) * smoothstep(R + 10.0, R - 6.0, d)
                    + 0.02 * smoothstep(R - 6.0, r1, d) + (a2 - a1 - 0.02) * smoothstep(r1 + 16.0, r1, d)
                    + 0.02 * smoothstep(r1, r2, d);
            float T = Hl * t + (Hl * (0.96 - a2) + crown) * smoothstep(r2 + 16.0, r2, d);
            if (T > g.U) { g.U = T; g.wK = 0.0; g.lipU = 0.0; g.notchH = 0.0; }
        }
    }
    return g;
}
