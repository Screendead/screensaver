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
    if (built && float(v.y) >= (col.z > col.y ? floor(base) : base)) {
        uint sb = st.read(uint2(clamp(v.xz, int2(0), int2(tr.size - 1)))).r;
        int mat = int(sb & 15u);
        uint pk = (sb >> S_PART) & 31u;
        device const Site& site = sites[(sb >> S_SITE) & 511u];
        Part P = unpackPart(site.parts[pk]);
        uint sh = uint(site.head.w);
        float age = site.info.y;
        float Ht = col.x;
        int style = (P.flags >> 12) & 15;
        int feat = int((sb >> S_FEAT) & 15u);
        bool chim = (sb & S_CHIMNEY) != 0u;
        bool tent = P.kind == P_HOUSE && (P.flags & F_TENT) != 0;
        bool house = P.kind == P_HOUSE && !tent;
        int2 sc = site.head.xy - mapOrigin;
        float floorY = float(P.y0);
        float k = hashf(pcg(sh + pk * 7919u + uint(P.y2)));
        // Fine pattern fades to its mean before it nears a pixel.
        float fade = 1.0 - smoothstep(0.02, 0.07, fp);
        // This voxel's material, and the plane a roof or a round wall shades as (zero: its own face).
        int vm = mat;
        bool roof = false;
        float3 rn = 0;
        // Roof coordinates: along the fall line, and along the eave.
        float2 ra = wq;
        HouseAt ha;
        if (house) {
            houseAt(P, v.xz, sc, ha);
            if (chim && (ha.jw == (ha.Dw - 1) / 2 || wy < Ht - ha.f.lip)) vm = M_STONE;
            else if (feat == FT_VIGA || (feat == FT_POST && (style != ST_CABIN || wy < Ht - roofStep(ha) - 0.25))) vm = M_TIMBER;
            else if (feat == FT_DECK) vm = M_PLANK;
            else if (float(v.y) < floorY) vm = style == ST_STILT ? M_LOG : M_STONE;
            else if (!ha.f.flat) {
                bool inWall = ha.iw >= 0 && ha.jw >= 0 && ha.iw < ha.Lw && ha.jw < ha.Dw;
                // The roof course is as deep as the steepest step, so every riser of the roof is roof.
                float thick = max(1.0, roofStep(ha)) + 0.3;
                bool gableWall = !ha.f.hip && ha.dG == 0;
                float under = gableWall ? Ht - thick : min(floorY + ha.f.wall, Ht - thick);
                if (!inWall || wy >= under) {
                    roof = true;
                    vm = style == ST_TIMBER || style == ST_CABIN ? M_SHINGLE : M_THATCH;
                    bool along = ha.f.hip && ha.dG < ha.dA;
                    float side = along ? (ha.iw * 2 < ha.Lw - 1 ? -1.0 : 1.0)
                                       : (ha.jw * 2 < ha.Dw - 1 ? -1.0 : ha.jw * 2 > ha.Dw - 1 ? 1.0 : 0.0);
                    float slope = ha.dA < -1 ? 0.5 : roofProfile(ha).y;
                    bool useX = ha.axisZ != along;
                    rn = normalize(float3(useX ? side * slope : 0.0, 1.0, useX ? 0.0 : side * slope));
                    ra = useX ? wq : wq.yx;
                }
            }
        } else if (tent) {
            int4 ij = partIJ(P, v.xz, sc);
            // Posts show as timber under the roof they carry.
            if (feat == FT_POST && (style == ST_AFRAME || wy < Ht - 0.5)) vm = M_TIMBER;
            else {
                if (feat == FT_POST) vm = style == ST_PALM ? M_THATCH : M_FELT;
                roof = true;
                bool axisZ = (P.flags & F_AXISZ) != 0;
                float side = 0.0, slope = 0.0;
                if (style == ST_AFRAME) { side = sign(float(ij.y - ij.w / 2)); slope = 1.0; }
                else if (style == ST_LEANTO) {
                    float2 mid = float2(P.o) + float2(P.sz) * 0.5;
                    side = (axisZ ? mid.x : mid.y) > 0.0 ? 1.0 : -1.0;
                    slope = 0.65;
                }
                rn = normalize(float3(axisZ ? side * slope : 0.0, 1.0, axisZ ? 0.0 : side * slope));
                ra = axisZ ? wq : wq.yx;
            }
        } else if (P.kind == P_ROUND) {
            float2 dq = wq - float2(mapOrigin) - float2(sc + P.o) - 0.5;
            float r = length(dq);
            float2 rd = dq / max(r, 1e-3);
            float2 pr = roundProfile(P, r);
            if (feat == FT_POLE && wy >= Ht - 0.75) vm = M_TIMBER;
            // The wall ring shades as a drum, the roof as a dome or cone.
            bool wallFace = n.y == 0 && wy < floorY + float(P.y1) * 0.25;
            rn = wallFace ? float3(rd.x, 0.0, rd.y) : normalize(float3(rd.x * pr.y, 1.0, rd.y * pr.y));
            roof = !wallFace;
        }
        if (P.kind == P_FIRE && (float(v.y) < Ht - 1.0 || mat == M_STONE)) vm = M_STONE;
        if (P.kind == P_PROP && feat == FT_POST) vm = M_TIMBER;
        float fy = wy - floorY;
        // Along a wall face, and the glow of an open door (lit from within at night).
        float hs = n.x != 0 ? wq.y : wq.x;
        float doorM = 0.0;
        float3 a;
        if (vm == M_STONE) {
            // The hill's own rock, lichened with age.
            a = strataTone(wy, wq, seed, desert, fpA) * 0.8;
            if (painted > 0.0) a = mix(a, paintedTone(wy, paintedLateral(w, seed), seed, fpA) * 0.85, painted);
            float3 lichen = mix(mix(float3(0.17, 0.19, 0.14), float3(0.26, 0.15, 0.07), autumn), float3(0.26, 0.27, 0.27), cold);
            float ln = fp < 0.15 ? perlin3(float3(wq.x, wy * 1.3, wq.y) / 1.4, seed + 310u) : 0.0;
            float cov = smoothstep(0.0, 0.3, ln + 0.45 * age - 0.3 + (n.y > 0 ? 0.15 : 0.0));
            cov = mix(cov, 0.35 * age, smoothstep(0.04, 0.15, fp));
            a = mix(a, lichen, cov * (1.0 - desert) * 0.55);
            // Built stone (walls and houses, not the standing stones) is laid in courses, darker at the joints.
            if (P.kind != P_HENGE && P.kind != P_BOX && fp < 0.08) {
                float jy = abs(fract(wy * 2.0) - 0.5), jx = abs(fract((n.x != 0 ? wq.y : wq.x) * 1.5 + floor(wy * 2.0) * 0.5) - 0.5);
                float joint = max(smoothstep(0.42, 0.48, jy), n.y == 0 ? smoothstep(0.44, 0.49, jx) : 0.0);
                a *= 1.0 - 0.25 * joint * (1.0 - smoothstep(0.03, 0.08, fp));
            }
            if (P.kind == P_FIRE) a = mix(a, float3(0.16, 0.135, 0.10), 0.75) * (0.8 + 0.35 * hashf(hash2i(v.xz + mapOrigin, seed + 344u)));
            else if (chim) a = mix(a, float3(dot(a, float3(0.3, 0.59, 0.11))), 0.5) * (wy > Ht - 0.4 ? 0.3 : 0.6);
            else if (P.kind != P_HENGE && P.kind != P_BOX) a *= house && ha.f.flat ? 0.6 : 0.85;
        } else if (vm == M_TIMBER) {
            a = (P.kind == P_PROP ? float3(0.17, 0.125, 0.08) : float3(0.10, 0.075, 0.055)) * (1.0 + 0.15 * sin(wy * 9.0 + perlin2(wq * 3.0, seed + 311u) * 3.0) * (1.0 - smoothstep(0.02, 0.06, fp)));
        } else if (vm == M_THATCH) {
            if (P.kind == P_PROP) a = float3(0.232, 0.202, 0.148) * (0.9 + 0.2 * k);
            else if (style == ST_STILT || style == ST_PALM) a = mix(float3(0.27, 0.23, 0.14), float3(0.19, 0.18, 0.15), 0.5 * age + 0.4 * k);
            else if (style == ST_CROFT) a = mix(float3(0.12, 0.15, 0.08), float3(0.14, 0.12, 0.08), saturate(0.4 * k + 0.25 * cold));
            else a = mix(mix(float3(0.24, 0.18, 0.09), float3(0.16, 0.15, 0.11), 0.7 * k + 0.4 * age), float3(0.15, 0.12, 0.07), 0.4 * autumn);
            // Bundles laid in courses up the slope, combed down it.
            float course = hashf(hash2i(int2(int(floor((n.y != 0 ? ra.x : wy) * 2.0)), int(floor(ra.y * 0.8))), seed + 330u));
            a *= 1.0 + fade * (0.14 * (course - 0.5) + 0.10 * perlin2(float2(ra.y * 4.0, wy * 0.7), seed + 312u));
        } else if (vm == M_SHINGLE) {
            a = (style == ST_CABIN ? float3(0.25, 0.165, 0.10) : float3(0.20, 0.12, 0.075)) * (0.9 + 0.2 * k);
            float moss = saturate(perlin2(wq / 2.5, seed + 331u) + 0.5) * age;
            a = mix(a, float3(0.09, 0.11, 0.05), (style == ST_CABIN ? 0.25 : 0.5) * mix(moss, 0.5 * age, 1.0 - fade));
            // Rows of shingles, each row's joints offset half a shingle.
            float rw = (n.y != 0 ? ra.x : wy) * 2.0;
            float u = ra.y * 1.5 + 0.5 * float(int(floor(rw)) & 1);
            float seam = max(smoothstep(0.40, 0.48, abs(fract(rw) - 0.5)), smoothstep(0.44, 0.49, abs(fract(u) - 0.5)));
            a *= 1.0 - 0.3 * seam * (1.0 - smoothstep(0.03, 0.08, fp));
        } else if (vm == M_LOG) {
            float along = n.x != 0 ? wq.y : wq.x;
            if (style == ST_STILT) {
                // Bamboo: pale canes standing side by side.
                a = float3(0.26, 0.21, 0.13) * (0.9 + 0.15 * k);
                a *= 1.0 - 0.18 * fade * smoothstep(0.36, 0.48, abs(fract(along * 3.0) - 0.5));
            } else {
                // Log courses with dark chinking between them.
                float lc = fract(wy * 2.0);
                a = (P.kind == P_PROP ? float3(0.14, 0.105, 0.075) : float3(0.21, 0.14, 0.085)) * (0.9 + 0.2 * k);
                a *= mix(1.0, 0.8 + 0.25 * sin(lc * 3.14159265) - 0.3 * smoothstep(0.38, 0.5, abs(lc - 0.5)), fade);
            }
        } else if (vm == M_PLASTER) {
            a = float3(0.24, 0.22, 0.19) * (0.95 + 0.08 * perlin2(wq / 3.0 + wy, seed + 313u));
            float3 beam = float3(0.08, 0.055, 0.04);
            // A dark timber sill; timber-framed houses show their posts and rails over the infill.
            float frame = smoothstep(0.25, 0.2, fy);
            if (style == ST_TIMBER) {
                float along = n.x != 0 ? wq.y : wq.x;
                frame = max(frame, max(smoothstep(0.42, 0.36, abs(fract(along * 0.5) - 0.5)) , smoothstep(2.8, 2.85, fy)));
                frame = max(frame, smoothstep(0.08, 0.03, abs(fy - 1.6)));
            }
            a = mix(a, beam, n.y == 0 ? mix(frame, 0.2, 1.0 - fade) : 0.0);
        } else if (vm == M_ADOBE) {
            a = mix(float3(0.44, 0.28, 0.16), float3(0.50, 0.36, 0.22), smoothstep(0.5, 2.5, fy));
            if (style == ST_PUEBLO) a = mix(a, paintedTone(wy, paintedLateral(w, seed), seed, fpA), 0.35);
            a *= (0.94 + 0.08 * perlin2(wq / 2.0 + wy * 0.5, seed + 313u)) * mix(0.82, 1.0, smoothstep(0.2, 0.5, fy));
        } else if (vm == M_FELT) {
            float3 band = 0;
            bool banded = false;
            if (P.kind == P_ROUND && style == ST_TIPI) {
                a = float3(0.32, 0.25, 0.17) * (0.9 + 0.15 * k);
                band = k < 0.5 ? float3(0.22, 0.06, 0.03) : float3(0.05, 0.07, 0.12);
                banded = fy > 0.75 && fy < 1.25;
            } else if (P.kind == P_ROUND) {
                a = float3(0.28, 0.26, 0.22) * (0.92 + 0.1 * k);
                band = k < 0.5 ? float3(0.24, 0.05, 0.04) : float3(0.06, 0.07, 0.18);
                banded = n.y == 0 && fy > float(P.y1) * 0.25 - 0.5 && fy < float(P.y1) * 0.25;
                if (feat == FT_CROWN && wy >= Ht - 0.5) a *= 0.5;
            } else if (style == ST_CARAVAN) {
                a = k < 0.5 ? float3(0.22, 0.09, 0.06) : float3(0.09, 0.09, 0.15);
            } else {
                a = k < 0.4 ? float3(0.15, 0.13, 0.09) : k < 0.75 ? float3(0.11, 0.12, 0.07) : float3(0.18, 0.09, 0.05);
            }
            if (banded) a = mix(a, band, fade * 0.85 + 0.15);
            // A painted door in a yurt's wall.
            if (P.kind == P_ROUND && style != ST_TIPI && n.y == 0) {
                float2 dq = wq - float2(mapOrigin) - float2(sc + P.o) - 0.5;
                float da = hashf(pcg(sh + pk)) * 6.2831853;
                if (dot(normalize(dq), float2(cos(da), sin(da))) > 0.96 && fy < 1.6) { a = float3(0.30, 0.12, 0.03); doorM = 0.6; }
            }
            // An A-frame's flap stands open on the gable facing the fire.
            if (style == ST_AFRAME && n.y == 0) {
                int4 ij = partIJ(P, v.xz, sc);
                bool az = (P.flags & F_AXISZ) != 0;
                float2 mid = float2(P.o) + float2(P.sz) * 0.5;
                bool gable = az ? float(n.z) * mid.y < 0.0 : float(n.x) * mid.x < 0.0;
                int rj = abs(ij.y - ij.w / 2);
                if (gable && rj < 2 && wy - base < (rj == 0 ? 1.75 : 0.75)) { a = float3(0.03, 0.025, 0.02); doorM = 1.0; }
            }
        } else if (vm == M_PLANK) {
            float seam = n.y > 0 ? 1.0 - smoothstep(0.0, 0.08, abs(fract(((P.flags & F_AXISZ) != 0 ? wq.y : wq.x) * 3.0) - 0.5) - 0.42) : 0.0;
            a = float3(0.12, 0.10, 0.08) * (1.0 - 0.3 * seam * (1.0 - smoothstep(0.02, 0.06, fp)));
        } else if (vm == M_CLOTH) {
            uint ch = pcg(sh + uint(P.y2)) % 3u;
            a = ch == 0u ? float3(0.22, 0.20, 0.17) : ch == 1u ? float3(0.24, 0.16, 0.08) : float3(0.10, 0.11, 0.15);
        } else {
            // A fire's bed: charred ends in grey ash.
            a = n.y > 0 && length(uv - 0.5) < 0.3 ? float3(0.045, 0.035, 0.025) : float3(0.12, 0.105, 0.09);
        }
        if (P.kind == P_PROP) {
            a = propShade(P, a, feat, v, n, uv, wq, wy, hs, Ht, fade, sc, mapOrigin, sh, seed, autumn);
            // Yard clutter settles toward the ground's tone once a voxel nears a pixel.
            a = mix(a, tread, 0.6 * smoothstep(0.2, 0.8, fpA));
        }
        // Roofs snow where the ground they stand on does: a few voxels up they would wear a white lid over green land.
        if (roof && (vm == M_THATCH || vm == M_SHINGLE || vm == M_FELT) && snowy) a = float3(0.29, 0.30, 0.33);
        if (P.kind == P_BOAT && n.y > 0) a *= 0.55;
        // Windows and doors on the wall faces of houses.
        float winM = 0.0;
        if (house && n.y == 0 && !roof && !chim && float(v.y) >= floorY && feat == 0) {
            uint side = n.x < 0 ? 0u : n.x > 0 ? 1u : n.z < 0 ? 2u : 3u;
            int op = houseOpening(P, ha, side);
            float wfy = fy - (op == 3 ? 3.0 : 0.0) - (ha.f.wall >= 3.0 ? 0.2 : 0.0);
            float e = clamp(fpA * 0.6, 0.03, 0.25);
            if (op == 1 || op == 3) {
                float mx = smoothstep(0.28 - e, 0.28 + e, uv.x) * smoothstep(0.72 + e, 0.72 - e, uv.x);
                // The height band is not foreshortened when the wall is seen edge-on, so the glow keeps it and
                // only the band across the wall fades to its mean.
                float ey = clamp(fp * 0.6, 0.03, 0.25);
                float my = smoothstep(0.55 - ey, 0.55 + ey, wfy) * smoothstep(1.45 + ey, 1.45 - ey, wfy);
                float m = mix(mx * my, 0.2, smoothstep(0.05, 0.3, fpA));
                a = mix(a, float3(0.025, 0.025, 0.03), m);
                winM = mix(mx, 0.44, smoothstep(0.05, 0.3, fpA)) * mix(my, 0.3, smoothstep(0.3, 1.0, fp));
            }
            if (op == 2) {
                float m = smoothstep(0.25 - e, 0.25 + e, uv.x) * smoothstep(0.75 + e, 0.75 - e, uv.x) * smoothstep(1.8 + e, 1.8 - e, wfy);
                a = mix(a, float3(0.045, 0.035, 0.028), mix(m, 0.4, smoothstep(0.05, 0.3, fpA)));
                doorM = mix(m, 0.4, smoothstep(0.05, 0.3, fpA)) * 0.6;
            }
        }
        int3 ld = house && ha.f.flat && n.y == 0 ? flatLadder(style, uint(P.y2), ha.Lw, ha.Dw) : int3(-1);
        if (ld.x >= 0 && ha.iw == ld.x && ha.jw == ld.y && fy >= float(ld.z)) {
            bool want = style == ST_PUEBLO ? (ha.axisZ ? n.x > 0 : n.z > 0) : (ha.axisZ ? n.z > 0 : n.x > 0);
            if (want) {
                float rail = step(abs(abs(uv.x - 0.5) - 0.28), 0.07);
                float rung = step(abs(uv.x - 0.5), 0.28) * step(fract(fy * 2.5), 0.2);
                float shadow = step(abs(abs(uv.x - 0.5) - 0.38), 0.05);
                float3 wood = float3(0.20, 0.14, 0.085);
                a = mix(a * mix(1.0, 0.92, 1.0 - fade), mix(a * (1.0 - 0.35 * shadow), wood, max(rail, rung)), fade);
                winM = 0.0;
            }
        }
        // Lamps: flame on a fire's top course, a lantern atop its post, light in the windows.
        int li = ((P.flags >> 8) & 15) - 1;
        if (li >= 0) {
            int lk = int(site.lamps[li].w) % 16;
            float3 E = lampState[((sb >> S_SITE) & 511u) * uint(MAX_LAMPS) + uint(li)].rgb * LAMP_EMIT[lk];
            bool topCourse = float(v.y) >= Ht - 1.0;
            if (P.kind == P_FIRE && topCourse && mat == M_FIRE) {
                float flame = perlin3(float3(wq * 2.5, wy * 1.6 - time * 1.4), seed + 320u) * 0.5 + 0.5;
                flame = mix(flame, 0.5, smoothstep(0.05, 0.3, fpA));
                // Per-channel tonemapping bleaches a hot core to white; held this low, the core stays orange and
                // the flame noise carries it.
                s.emit = E * float3(1.0, 0.7, 0.5) * (n.y > 0 ? mix(0.05, 0.24, flame) : mix(0.02, 0.17, flame) * (0.45 + 0.55 * fract(wy)));
            } else if (P.kind == P_BOX && topCourse) {
                s.emit = E * float3(1.0, 0.8, 0.6) * (n.y > 0 ? 0.16 : 0.3 * smoothstep(0.3, 0.8, fract(wy)));
            } else if (house || P.kind == P_ROUND || tent) {
                float canvas = lk == L_TENT && (vm == M_FELT || vm == M_THATCH) ? 0.12 : 0.0;
                s.emit = E * (winM + canvas) + E * doorM * float3(1.0, 0.72, 0.45);
            }
        }
        s.albedo = a;
        s.tread = a;
        s.mat = max(vm, 4);
        s.canopy = 1.0;
        s.H = Ht;
        s.pale = 0;
        s.slope = 0;
        s.riser = n.y == 0 ? Ht - nb : 0.0;
        s.level = (sb & S_FLAT) != 0u && n.y > 0 ? 1.0 : 0.0;
        if (any(rn != 0.0)) s.nrm = rn;
    }
    return s;
}
