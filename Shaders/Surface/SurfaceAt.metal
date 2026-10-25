// Surface of level k's column v (xz in level cells) on face n; uv is the face position in [0, 1) of the column.
// Coarse columns are s voxels wide but still unit voxels tall, and carry no structures.
// lite: seen in rippled water, so detail finer than the ripples' blur is shaded at its mean.
static Surface surfaceAt(thread const Terrain& tr, texture2d<uint> st, device const Site* sites, device const float4* lampState,
                         int k, int3 v, int3 n, float2 uv, float fp, float fpA, float3 p, uint seed, float time, bool lite = false,
                         bool bio = false) {
    if (lite) { fp = max(fp, 4.0); fpA = max(fpA, 4.0); }
    Surface s;
    s.up = 1.0;
    s.vine = false;
    s.bio = 0;
    s.bioG = 0;
    s.bioY = 0;
    s.bioJ = 0;
    s.bioNb = 0u;
    s.slope = 0;
    s.mat = 0;
    s.level = 0;
    s.pale = 0;
    s.wall = 0;
    s.face = 0;
    s.nrm = 0;
    s.emit = 0;
    s.isle = 0;
    s.isleUnder = false;
    s.gy = p.y;
    s.cave = 0;
    s.caveG = -1.0;
    int2 mapOrigin = tr.origin;
    float cs = tr.lod[k].w;
    float4 col4 = columnL(tr, k, v.xz);
    float3 col = col4.rgb;
    bool tree = col4.a > 0.0;
    // A structure stands on this column from voxel `base` up; below it (or below its gap) is the terrain.
    float base = -col4.a;
    bool built = col4.a < 0.0 && base < col.x;
    // A giant: on the 1-voxel map its columns carry GIANT_A, with the ground and materials in st; further out its
    // columns are those within its reach, wood and leaf from a little above the ground at its axis.
    Giant gi;
    float2 cdK = coverDepthL(tr, k, v.xz);
    bool giant = k == 0 ? col4.a == GIANT_A : cdK.y >= 64.0 && giantAt(tr.giants, float2(v.xz * int(cs) + mapOrigin) + 0.5 * cs, cs, gi);
    uint gsb = giant && k == 0 ? st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r : 0u;
    float gLoc = k == 0 ? float(gsb & 2047u) : 0.0;
    if (giant && k > 0) gLoc = gi.G + 6.0;
    bool gWood = giant && float(v.y) >= gLoc;
    // An island's body: its top is meadow, coloured as the lowland under it is (its height would read as upland rock).
    bool isleCol = ISLE_SHAPE && !tree && !built && !giant && col.z > col.y && isleGapL(tr, k, v.xz, col4);
    bool isleBody = isleCol && float(v.y) >= col.z;
    // Its canopy, stream, fall or vine: the column's ST_ISLE word on the 1-voxel map, its leaf depth further out.
    // Falls and streams run out to level 2 and the 1-voxel map holds the rest, all well inside the islands' pipeline
    // variant's reach: compiled out of the others.
    uint isw = SCENE_NEAR && isleBody && k == 0 ? st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r : 0u;
    bool isleWaterCol = SCENE_NEAR && isleBody && (k == 0 ? (isw & ST_ISLE_WATER) != 0u : cdK.y == ISLE_WATER_DEPTH || cdK.y == ISLE_STREAM_DEPTH);
    // A stream's water is its bed alone; the rock under it is the island's.
    bool isleWater = isleWaterCol && !(float(v.y) < col.x - 3.0 && (k == 0 ? (isw & 2047u) == 0u : cdK.y == ISLE_STREAM_DEPTH));
    float isleLo = (isw & (ST_ISLE | ST_ISLE_WATER | ST_ISLE_VINE)) == ST_ISLE ? float(isw & 2047u) : col.x;
    bool isleLeaf = k == 0 && isleBody && float(v.y) >= isleLo;
    // A coarse forested column's top is its leaves (coarseVeg).
    float vegLo = col.x - cdK.y;
    bool vegLeaf = k > 0 && !giant && !isleWaterCol && cdK.y > 0.0 && float(v.y) >= vegLo;
    if (k > 0 && isleBody && !isleWaterCol) isleLo = vegLo;
    // Bark and leaves: the ground's own tones (rock strata, painted beds) would be worked out only to be replaced.
    bool vegVox = gWood || vegLeaf || isleLeaf || (tree && float(v.y) >= col.y);
    float H = giant ? (gWood ? col.x : gLoc) : tree ? col.y : built ? (col.z > col.y ? col.y : base) : col.x;
    // Just under a rock gap the voxel belongs to the floor below it: its height is the gap's floor. Further down it is
    // part of the wall the column makes as a whole, which shades as one with its neighbours'.
    if (!tree && !built && !giant && col.z > col.y && float(v.y) < col.y) {
        float seg = col.y;
        if (k == 0) {
            float4 e = extraGaps(tr.gx.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))));
            if (e.y > e.x && float(v.y) < e.x) seg = e.x;
            if (e.w > e.z && float(v.y) < e.z) seg = e.z;
        }
        if (float(v.y) >= seg - 6.0) H = seg;
    }
    s.canopy = tree ? 0.55 : 1.0;
    s.H = H;
    float Hc = isleBody ? col.y + 4.0 : H;
    bool isleRoot = false;
    s.isleUnder = isleCol && float(v.y) < col.z;
    // An island's own rock sees its neighbours whole: the ground under them is the ground's business.
    if (isleCol && float(v.y) >= col.z) s.gy = 1e9;
    float hx0, hx1, hz0, hz1;
    float4 frontCol = 0.0;
    if (k == 0) {
        // The neighbours' whole columns: their ground, and whether any holds a gap, from the same reads.
        float4 a0 = columnL(tr, 0, v.xz + int2(-1, 0)), a1 = columnL(tr, 0, v.xz + int2(1, 0));
        float4 a2 = columnL(tr, 0, v.xz + int2(0, -1)), a3 = columnL(tr, 0, v.xz + int2(0, 1));
        hx0 = a0.a > 0.0 ? a0.g : a0.r; hx1 = a1.a > 0.0 ? a1.g : a1.r;
        hz0 = a2.a > 0.0 ? a2.g : a2.r; hz1 = a3.a > 0.0 ? a3.g : a3.r;
        if (ISLE_SHAPE) {
            // Under an island's edge the ground sees the island's own ground beside it, not a cliff to its top.
            float vy = min(float(v.y), s.gy);
            if (vy < a0.b && a0.a == 0.0 && a0.b - a0.g >= ISLE_AIR) hx0 = a0.g;
            if (vy < a1.b && a1.a == 0.0 && a1.b - a1.g >= ISLE_AIR) hx1 = a1.g;
            if (vy < a2.b && a2.a == 0.0 && a2.b - a2.g >= ISLE_AIR) hz0 = a2.g;
            if (vy < a3.b && a3.a == 0.0 && a3.b - a3.g >= ISLE_AIR) hz1 = a3.g;
            // A root hangs well below a neighbour's underside (a keel's beds step by seven at most).
            if (isleCol && vy >= col.z) {
                float hang = max(max(a0.a == 0.0 && a0.b > a0.g ? a0.b : 0.0, a1.a == 0.0 && a1.b > a1.g ? a1.b : 0.0),
                                 max(a2.a == 0.0 && a2.b > a2.g ? a2.b : 0.0, a3.a == 0.0 && a3.b > a3.g ? a3.b : 0.0));
                isleRoot = vy < hang - ((isw & ST_ISLE_VINE) != 0u ? 3.0 : 8.0);
            }
        }
        // Beside a multi-run column (a giant's, an emergent's) the ground does not see its top as a wall: it would
        // light and paint a ring round every crown's footprint as a cliff foot.
        if (!giant) {
            if (a0.a == GIANT_A) hx0 = H;
            if (a1.a == GIANT_A) hx1 = H;
            if (a2.a == GIANT_A) hz0 = H;
            if (a3.a == GIANT_A) hz1 = H;
        }
        // Rock gaps only: canopies and buildings hold theirs with a non-zero alpha, and an island's air is no roof.
        s.gapSpan = float2(1e9, -1e9);
        float4 cs5[5] = { col4, a0, a1, a2, a3 };
        for (int i = 0; i < 5; i++)
            if (cs5[i].b > cs5[i].g && cs5[i].a == 0.0 && (!ISLE_SHAPE || cs5[i].b - cs5[i].g < ISLE_AIR))
                s.gapSpan = float2(min(s.gapSpan.x, cs5[i].g), max(s.gapSpan.y, cs5[i].b));
        s.gapNear = s.gapSpan.y > s.gapSpan.x;
        frontCol = n.x < 0 ? a0 : n.x > 0 ? a1 : n.z < 0 ? a2 : a3;
        if (bio && n.y > 0 && !tree && !giant)
            s.bioNb = uint(a0.a > 0.0 || a0.a == GIANT_A) | uint(a1.a > 0.0 || a1.a == GIANT_A) << 1
                    | uint(a2.a > 0.0 || a2.a == GIANT_A) << 2 | uint(a3.a > 0.0 || a3.a == GIANT_A) << 3;
    } else {
        float vy = min(float(v.y), s.gy);
        hx0 = groundL(tr, k, v.xz + int2(-1, 0), vy); hx1 = groundL(tr, k, v.xz + int2(1, 0), vy);
        hz0 = groundL(tr, k, v.xz + int2(0, -1), vy); hz1 = groundL(tr, k, v.xz + int2(0, 1), vy);
        float4 fr = n.y == 0 ? columnL(tr, k, v.xz + n.xz) : float4(0);
        frontCol = fr;
        bool cg0 = col.z > col.y && !isleCol, fg = fr.z > fr.y && !(ISLE_SHAPE && isleGapL(tr, k, v.xz + n.xz, fr));
        s.gapNear = cg0 || fg;
        s.gapSpan = float2(min(cg0 ? col.y : 1e9, fg ? fr.y : 1e9), max(cg0 ? col.z : -1e9, fg ? fr.z : -1e9));
    }
    s.nh = float4(hx0, hx1, hz0, hz1);
    float slope = max(max(H - hx0, H - hx1), max(H - hz0, H - hz1));
    float2 w = float2(v.xz * int(cs) + mapOrigin);
    // Continuous world position on the face, so tonal detail flows across block edges.
    float2 wq = w + (n.y != 0 ? uv : n.x != 0 ? float2(n.x > 0 ? 1.0 : 0.0, uv.x) : float2(uv.x, n.z > 0 ? 1.0 : 0.0)) * cs;
    float wy = n.y != 0 ? float(v.y + max(n.y, 0)) : float(v.y) + uv.y;
    float nb = n.x < 0 ? hx0 : n.x > 0 ? hx1 : n.z < 0 ? hz0 : hz1;
    s.riser = n.y == 0 ? H - nb : 0.0;
    if (gWood) {
        if (bio) { s.bio = -1.0; s.bioY = gLoc; }
        return giantSurface(tr, s, k, v, n, uv, fp, wq, wy, col, gsb, gLoc, gi, cs, mapOrigin, seed);
    }
    Climate c = climateL(tr, k, (float2(v.xz) + 0.5) * cs);
    // An island is green to its rim, wherever its edge reaches out over drier land.
    if (Hc != H) { c.temp = clamp(c.temp, 0.48, 0.6); c.moist = max(c.moist, 0.66); c.fall = 0.0; }
    float desert = desertAmount(c);
    // Beaches and beds take sand or crust, faces included, so they need none of the ground and rock tones.
    bool beach = H <= SEA + 1.2;
    // Bark and leaves take their own colours below: of the ground's tones they keep only a salt pan's level, and on a top
    // face the sand's ripples, which the snowline can rule out.
    // Painted-hill beds belong to low mounds; on the great walls their 11-voxel bands would read as stripes.
    float wallM = vegVox && paintedAmount(c) <= 0.0 ? 0.0 : megaWall(tr, w);
    s.wall = wallM;
    float painted = smoothstep(0.0, 0.6, paintedAmount(c)) * (1.0 - wallM);
    float snowline = 1e9;
    if (!vegVox || (n.y > 0 && (beach || desert > 0.35))) {
        snowline = snowlineAt(c.temp) + 5.0 * perlin2(w / 160.0, seed + 21u);
        // On the ranges rock breaks through the lower snowfields in drifts, so a massif is not one white sheet.
        if (Hc > 150.0) snowline += 60.0 * perlin2(w / 700.0, seed + 28u) * smoothstep(150.0, 400.0, Hc)
                                 + 260.0 * saturate(perlin2(w / 90.0, seed + 33u) * 0.9 + 0.45) * smoothstep(150.0, 450.0, Hc);
    }
    c.temp = altitudeTemp(c.temp, Hc);
    float cold = smoothstep(0.42, 0.18, c.temp);
    float autumn = autumnAmount(c);
    s.autumn = autumn;
    float birch = 0.0;
    float3 fallLeaf = 0.0;
    if (autumn > 0.0) {
        birch = birchAmount(w, seed);
        fallLeaf = autumnLeaf(w, birch, seed);
    }
    // A side face looking into a rock gap (a shelter's back wall) is rock under a roof too: by its neighbours' tops it
    // would read as a low riser and take the tread colour, column by column.
    // An island's air is no roof.
    bool underGap = !tree && !built && !giant && ((col.z > col.y && float(v.y) < col.z && !isleCol)
                    || (n.y == 0 && frontCol.a == 0.0 && frontCol.b > frontCol.g && float(v.y) >= frontCol.g && float(v.y) < frontCol.b
                        && !(ISLE_SHAPE && isleGapL(tr, k, v.xz + n.xz, frontCol))));

    float salt = saltAmount(c);
    float rimW = smoothstep(0.05, 0.4, salt);

    // Rock tone shows on side and under faces, and on high treads; a low tread never reads it.
    bool needStone = !beach && !vegVox && (!(n.y > 0 && !underGap) || Hc - SEA > 0.55 * 110.0);
    // Branches, not selects: a select would work out both tones for every pixel.
    float3 stone = 0;
    if (needStone) {
        if (isleBody) stone = isleRock(wy, wq, seed, fpA);
        else stone = strataTone(wy, wq, seed, max(desert, rimW), fpA, wallM, lite);
    }

    float lagoon = smoothstep(0.0, 0.35, lagoonAmount(c));
    // Pale ground (sand, salt, snow) sits well below the clouds, and drifts in tone so open flats never read as one sheet.
    float drift = vegVox ? 1.0 : 1.0 + 0.12 * perlin2(wq / 110.0, seed + 32u);
    float3 sand = mix(float3(0.33, 0.29, 0.21), float3(0.37, 0.33, 0.245), lagoon) * drift;
    float3 snow = float3(0.31, 0.325, 0.35) * drift;
    float3 tread = sand;
    if (!beach && !vegVox) {
        float3 grassWet = float3(0.12, 0.20, 0.13), grassDry = float3(0.40, 0.34, 0.16), grassCold = float3(0.20, 0.23, 0.17);
        float3 grass = mix(mix(grassDry, grassWet, smoothstep(0.3, 0.7, c.moist)), grassCold, cold);
        // Autumn ground: tawny grass, with fallen leaves drifting out from the groves in the canopy's own hues.
        if (autumn > 0.0) {
            float litter = smoothstep(0.08, 0.36, perlin2((w + seedOffset(seed)) / 300.0, seed + 90u));
            float3 fallen = mix(fallLeaf * 1.5, float3(0.30, 0.15, 0.07), 0.45);
            grass = mix(grass, mix(float3(0.22, 0.18, 0.09), fallen, litter), autumn * 0.8);
        }
        grass = mix(grass, float3(0.14, 0.20, 0.12), lagoon);
        float3 desertTop = float3(0.54, 0.34, 0.19);
        tread = mix(grass, desertTop, desert) * (0.95 + 0.1 * perlin2(wq / 40.0, seed + 30u));
        // Shoreline benches alternate rust and ochre with height, drifting slowly along the rim.
        float benchK = saturate(0.5 + 0.9 * perlin2(float2(H * 0.23 + w.x / 400.0, w.y / 400.0), seed + 18u));
        tread = mix(tread, mix(float3(0.48, 0.25, 0.15), float3(0.62, 0.46, 0.25), benchK), rimW);
        // Broad, soft mottling: a little warmer and drier in one drift, cooler and lusher in the next.
        float mott = fpA < 2.0 ? perlin2(wq / 13.0, seed + 31u) * (1.0 - smoothstep(0.6, 2.0, fpA)) : 0.0;
        tread *= 1.0 + mott * float3(0.05, 0.035, -0.01);
        // Gentle hypsometric tint, like a printed topo map: sage valleys, paler ochre-grey uplands.
        float alt = saturate((Hc - SEA) / 110.0);
        tread *= mix(float3(0.93, 1.03, 0.90), float3(1.10, 1.04, 0.98), alt);
        tread = mix(tread, stone * 1.35, 0.30 * smoothstep(0.55, 1.0, alt));
        // Hypsometric banding, as on a printed topo map: each 6-voxel elevation band steps its tint slightly,
        // so broad hillsides read as layered contour bands rather than one repeated stair.
        float bandK = fract(floor(H / 6.0) * 0.618034);
        tread *= mix(float3(1.12, 1.04, 0.90), float3(0.90, 0.98, 1.10), bandK);
        if (painted > 0.0 && !vegVox) {
            float2 p = w + seedOffset(seed);
            // Alluvium on the valley floors: pale, dusty sage.
            float fl = paintedFloor(p, seed);
            float floorK = smoothstep(fl + 2.0, fl + 0.5, H);
            float2 lat = paintedLateral(w, seed);
            float3 bed = mix(paintedTone(H - 1.0, lat, seed, fpA), float3(0.44, 0.40, 0.28), floorK);
            tread = mix(tread, bed, painted);
            if (needStone) stone = mix(stone, paintedTone(float(v.y), lat, seed, fpA), painted);
        }
    }
    // Salt crust: cream with slow Perlin drifts, and a faint polygon net from two crossing sets of
    // Perlin zero-lines that fades long before it could alias.
    float crustW = beach ? smoothstep(0.3, 0.6, salt) : 0.0;
    if (crustW > 0.0 && !vegVox) {
        float3 crust = mix(float3(0.32, 0.305, 0.27), float3(0.33, 0.29, 0.27), saturate(perlin2(wq / 210.0, seed + 23u) + 0.5))
                     * (drift + 0.07 * perlin2(wq / 60.0, seed + 19u) + 0.04 * perlin2(wq / 17.0, seed + 20u));
        if (fpA < 0.2) {
            float lw = 0.06 + fpA * 0.6;
            float net = max(1.0 - smoothstep(0.0, lw, abs(perlin2(wq / 2.4, seed + 21u))),
                            1.0 - smoothstep(0.0, lw, abs(perlin2(OCT_ROT * wq / 2.4 + 13.7, seed + 22u))));
            crust *= 1.0 - 0.045 * net * (1.0 - smoothstep(0.05, 0.2, fpA));
        }
        // Broader pressure ridges carry the net into the middle distance.
        if (fpA < 0.8) {
            float ridge = 1.0 - smoothstep(0.0, 0.04 + fpA * 0.08, abs(perlin2(wq / 15.0, seed + 24u)));
            crust *= 1.0 - 0.035 * ridge * (1.0 - smoothstep(0.3, 0.8, fpA));
        }
        // Wet salt is greyer: pool floors, and a damp margin that shades into each pool.
        float lvl = poolLevel(salt);
        float damp = H < PAN ? 1.0 : smoothstep(lvl - 0.14, lvl, poolNoise(wq + seedOffset(seed), seed));
        crust *= mix(float3(1.0), float3(0.80, 0.82, 0.85), damp);
        tread = mix(tread, crust, crustW);
    }
    // The pan is dead level: lit by the smoothed heightfield, or with the lip and pool floor lit as walls and corners,
    // every pool would be outlined.
    s.level = crustW;
    // Coarse levels carry no trees: the share of a column under canopy tints its ground with their leaves.
    if (k > 0 && !vegVox) {
        float2 cd = cdK;
        float cov = saturate(cd.x * COVER_GAIN);
        if (cov > 0.0) {
            float conifer = smoothstep(0.50, 0.32, c.temp);
            float3 leaf = mix(float3(0.05, 0.08, 0.0375), float3(0.025, 0.055, 0.045), conifer);
            leaf = mix(leaf, float3(0.11, 0.125, 0.09), painted);
            leaf = mix(leaf, fallLeaf, autumn * (1.0 - conifer * 0.5)) * 0.92;
            tread = mix(tread, leaf, cov);
            s.canopy = mix(1.0, 0.7, cov);
        }
    }
    // Worn ground round a dwelling: paths between doors and hearth, and the apron round a fire.
    // A giant's column (GIANT_A) holds its own word in st, not a clearing's.
    if (k == 0 && col4.a < 0.0 && !built && !giant) {
        uint mk = (st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r >> S_MARK) & 3u;
        if (mk != 0u) {
            float3 earth = mix(mix(float3(0.20, 0.135, 0.08), float3(0.16, 0.13, 0.10), cold), float3(0.50, 0.36, 0.22), desert);
            earth = mix(earth, sand * 0.88, beach ? 1.0 : lagoon * 0.6);
            float br = fpA < 0.3 ? perlin2(wq / 1.3, seed + 342u) * (1.0 - smoothstep(0.05, 0.3, fpA)) : 0.0;
            // Worn patchily, column by column, thinning out to the fringe.
            float wear = mix(0.55 + 0.6 * hashf(hash2i(v.xz + mapOrigin, seed + 343u)), 0.85, smoothstep(0.1, 0.4, fpA));
            float w = (mk == MK_APRON ? 0.62 : mk == MK_FRINGE ? 0.26 : 0.75) * wear;
            tread = mix(tread, earth * 0.9, saturate(w + 0.2 * br));
        }
    }
    s.tread = tread;
    bool snowy = float(int(Hc)) > snowline;
    // Snow risers prefilter to snow, except on the great walls, whose faces stay rock under snowy ledges.
    if (snowy && wallM < 0.5) s.riser = 0.0;
    // Snow risers prefilter to snow; eased across the snowline, which far off is one straight contour.
    s.tread = mix(tread, snow, smoothstep(snowline - 6.0, snowline + 6.0, Hc));
    s.face = needStone ? (snowy ? mix(stone, snow, 0.45 * (1.0 - smoothstep(6.0, 30.0, slope))) : stone * 0.95) : s.tread;
    // Night glow: damp, mild, vegetated country, in groves, most in the woods and the damp lowlands; none on sand, salt,
    // desert, snow, the great walls or the bare uplands. Cover comes from the 2-voxel map on the 1-voxel one too, so
    // both sides of their handoff see the same woods.
    if (bio && !underGap && !built && !(isleBody && n.y <= 0 && float(v.y) < isleLo - 6.0)) {
        float d = max(smoothstep(0.1, 0.5, provinceAmount(c)), smoothstep(0.25, 0.55, c.moist)) * smoothstep(0.18, 0.4, c.temp)
                * (1.0 - smoothstep(0.1, 0.4, desert)) * (1.0 - rimW) * (1.0 - 0.5 * painted) * (1.0 - wallM)
                * (1.0 - smoothstep(SEA + 60.0, SEA + 150.0, Hc)) * (1.0 - smoothstep(snowline - 8.0, snowline, Hc));
        if (d > 0.0) {
            float2 wp = p.xz + float2(mapOrigin);
            float g = bioNoise(wp / 170.0, seed + 710u) + 0.5 * bioNoise(wp / 47.0, seed + 711u);
            s.bioG = g;
            s.bioJ = smoothstep(0.65, 0.9, c.jungle);
            d *= smoothstep(-0.35, 0.05, g) * (1.0 + 0.7 * smoothstep(SEA + 28.0, SEA + 3.0, Hc)) * (1.0 - 0.45 * s.bioJ);
            int kc = max(k, 1);
            constexpr sampler cb(filter::linear, address::repeat);
            float cov = tr.cv.sample(cb, (p.xz / tr.lod[kc].w + float2(levelOrigin(tr, kc))) / float(COARSE_N), uint(kc - 1), level(0)).r;
            d *= 0.6 + 0.4 * saturate(cov * COVER_GAIN);
        }
        s.bio = d;
        if (tree) s.bioY = col.y;
    }

    if (n.y > 0 && !underGap) {
        if (snowy) s.albedo = snow;
        else s.albedo = tread;
        float sandy = snowy ? 0.0 : beach ? 1.0 - crustW : smoothstep(0.35, 0.7, desert);
        s.pale = snowy || beach ? 1.0 : sandy;
        if (sandy > 0.0) {
            float3 rp = (k == 0 ? sandRipples(v.xz + mapOrigin, uv, w, fpA, time, seed)
                                : sandRipples(int2(floor(p.xz)) + mapOrigin, fract(p.xz), float2(int2(floor(p.xz)) + mapOrigin), fpA, time, seed)) * sandy;
            s.albedo *= 1.0 + rp.x;
            s.slope = rp.yz;
        }
    } else if (n.y == 0 && !underGap) {
        s.pale = beach || snowy ? 1.0 : 0.0;
        if (beach) s.albedo = tread * 0.85;
        else {
            // Low risers half take the tread; a column's deepest drop alone would make a staircase wall of mixed step
            // heights alternate per column (dashes down the face), so on steep relief the slope across the
            // neighbours decides too.
            float steepC = 0.5 * length(float2(hx0 - hx1, hz0 - hz1));
            float rockW = snowy ? 1.0 : max(step(4.0, slope), max(smoothstep(1.0, 2.5, steepC), wallM));
            // Snow lies on the treads; a tall wall stays rock under it.
            float3 rock = snowy ? mix(stone, snow, 0.45 * (1.0 - smoothstep(6.0, 30.0, slope))) : stone;
            s.albedo = mix(mix(mix(tread, stone, 0.5) * 0.9, stone * 0.95, painted), rock, rockW);
        }
    } else {
        s.albedo = stone * 0.9;
    }
    // An island's flanks and keel are rock below a lip of earth: far off, their risers prefilter to it, not the meadow.
    if (isleBody && float(v.y) < isleLo - 6.0 && n.y <= 0) {
        s.albedo = stone * 0.9;
        s.tread = s.albedo;
        s.face = s.albedo;
    }
    // Roots and vines under an island: umber and olive, each its own tone, darker toward the tip.
    if (isleRoot) {
        float rk = hashf(hash2i(v.xz, seed + 712u));
        s.albedo = ((isw & ST_ISLE_VINE) != 0u ? mix(float3(0.045, 0.075, 0.03), float3(0.07, 0.085, 0.035), rk) * (0.85 + 0.3 * hashf(hash2i(int2(v.x, v.y), seed + 718u)))
                                              : mix(float3(0.075, 0.058, 0.04), float3(0.058, 0.07, 0.035), rk))
                 * mix(0.7, 1.0, smoothstep(0.0, 10.0, wy - col.z));
        s.tread = s.albedo;
        s.face = s.albedo;
    }
    // Water: a stream's bed, ripples running out to the lip; a fall's sheet, foam streaming down it, whiter in the spray
    // at its foot.
    if (isleWater) {
        // Whiter toward a fall's foot (its height in the word; a stream's is 0), evenly so on the coarser rings.
        float fy = float(isw & 2047u);
        float foot = k > 0 || fy <= 0.0 ? 0.3 : 1.0 - saturate((wy - fy) / 45.0);
        float fine = 1.0 - smoothstep(1.0, 3.0, fpA);
        // A fall's sheet is foam on every face; only a stream's bed ripples.
        if (n.y > 0 && (k == 0 ? fy <= 0.0 : cdK.y == ISLE_STREAM_DEPTH)) {
            float rip = fine > 0.0 ? perlin2(float2(wq.x + wq.y, wq.x - wq.y) / 2.5 + float2(time * 1.3, 0.0), seed + 734u) * fine : 0.0;
            s.albedo = mix(float3(0.07, 0.12, 0.13), float3(0.30, 0.36, 0.38), saturate(0.25 + 0.6 * rip));
        } else {
            float lat = n.x != 0 ? wq.y : wq.x;
            float broad = fpA < 8.0 ? perlin2(float2(lat * 0.45, (wy + time * 20.0) / 16.0), seed + 736u) * (1.0 - smoothstep(4.0, 8.0, fpA)) : 0.0;
            float str = broad + (fine > 0.0 ? (0.8 * perlin2(float2(lat * 0.9, (wy + time * 24.0) / 7.0), seed + 733u)
                                               + 0.4 * perlin2(float2(lat * 2.1, (wy + time * 31.0) / 3.0), seed + 735u)) * fine : 0.0);
            s.albedo = mix(float3(0.26, 0.33, 0.37), float3(0.62, 0.66, 0.68), saturate(0.4 + 0.65 * str + 0.7 * foot));
        }
        s.tread = s.albedo;
        s.face = s.albedo;
    }
    if (k == 0 && isleBody) s.isle = float2(float((isleRoot ? 1 : 0) | (isleWater ? 2 : 0) | (float(v.y) < isleLo - 6.0 ? 4 : 0) | 8),
                                            isleWater && (isw & 2047u) != 0u ? float(isw & 2047u) : col.z);
    // Only beside a rock gap: every face a cave dresses is one of its columns' or their neighbours'.
    if (tr.caves > 0.0 && s.gapNear && !tree && !built && !vegLeaf && !isleBody) {
        float green = smoothstep(0.25, 0.5, c.moist) * smoothstep(0.18, 0.32, c.temp) * (1.0 - smoothstep(0.15, 0.45, desert))
                    * (1.0 - smoothstep(snowline - 20.0, snowline, H));
        s.caveG = green + (underGap ? 2.0 : 0.0);
    }
    if ((tree && float(v.y) >= col4.a) || vegLeaf || isleLeaf) {
        float leafLo = vegLeaf ? vegLo : isleLeaf ? isleLo : col4.a;
        // An island's leaves are the jungle's, with no autumn.
        Climate lc = c;
        if (isleBody) lc.jungle = 1.0;
        float3 leaf = woodLeaf(w, lc, painted, fallLeaf, autumn, seed);
        s.up = saturate((float(v.y) - leafLo) / max(col.x - leafLo - 1.0, 1.0));
        // Gone, like the land's glow, before the 1-voxel ring's morph band.
        float ct = k == 0 && !isleLeaf ? (1.0 - smoothstep(0.5, 1.0, fp)) * (1.0 - 0.5 * autumn) : 0.0;
        if (ct > 0.0) {
            Crown cr = crownAt(w + 0.5, c, seed);
            leaf *= mix(float3(1.0), cr.tint, ct);
            // An old-growth crown's columns run its whole height (multi-run ones too): by each column's own span they
            // would stripe it.
            if (cr.Ht > 0.0) s.up = saturate((float(v.y) - col.y - 0.45 * cr.Ht) / (0.55 * cr.Ht));
        }
        // A 16x16 leaf texture: two-tone texels, faded out once a texel nears a pixel.
        if (fp < 0.04) {
        int2 tx = int2(floor(uv * 16.0));
        int3 tk = n.y != 0 ? int3(v.x * 16 + tx.x, v.y * 2 + max(n.y, 0), v.z * 16 + tx.y)
                : n.x != 0 ? int3(v.x * 2 + max(n.x, 0), v.y * 16 + tx.y, v.z * 16 + tx.x)
                           : int3(v.x * 16 + tx.x, v.y * 16 + tx.y, v.z * 2 + max(n.z, 0));
        uint th = hash3i(tk, seed + 94u);
        float tex = hashf(th) < 0.35 ? 0.72 : 1.0 + 0.12 * hashf(pcg(th));
        leaf *= mix(tex, 0.92, smoothstep(0.015, 0.04, fp));
        } else leaf *= 0.92;
        // Upper leaves catch more sky.
        leaf *= mix(0.85, 1.15, saturate((float(v.y) - leafLo) / max(col.x - leafLo - 1.0, 1.0)));
        s.albedo = leaf;
        s.tread = leaf;
        s.H = col.x;
        s.mat = 1;
        s.canopy = 1.0;
    } else if (tree && float(v.y) >= col.y) {
        s.albedo = float3(0.06, 0.058, 0.052);
        s.mat = 2;
        if (autumn > 0.3 && birch > 0.5) {
            // Birch bark: chalk white, broken by dark lenticel bands.
            uint bh = hash3i(int3(v.x, int(floor(wy * 3.0)), v.z), seed + 98u);
            s.albedo = mix(hashf(bh) < 0.22 ? float3(0.06, 0.055, 0.05) : float3(0.30, 0.29, 0.27), float3(0.24, 0.23, 0.215), smoothstep(0.04, 0.12, fp));
            s.mat = 3;
        }
        s.tread = s.albedo;
        s.canopy = 1.0;
    }
    if (giant && !gWood && col.x > gLoc + 4.0) s.canopy = 0.7;
    // Over a gap the structure starts part way up its lowest voxel.
    if (built && float(v.y) >= (col.z > col.y ? floor(base) : base))
        structureSurface(s, tr, st, sites, lampState, v, n, uv, fp, fpA, col, base, nb, w, wq, wy, tread, painted, desert, autumn,
                         cold, snowy, seed, time);
    return s;
}
