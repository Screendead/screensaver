// The surface of a built voxel (a structure's part, from its word in st) seen on face n, over the ground's own tones
// (tread, painted, desert, autumn, cold, snowy) and the column's top col.x, its structure's base and the neighbour
// height nb toward n.
static void structureSurface(thread Surface& s, thread const Terrain& tr, texture2d<uint> st, device const Site* sites,
                             device const float4* lampState, int3 v, int3 n, float2 uv, float fp, float fpA, float3 col,
                             float base, float nb, float2 w, float2 wq, float wy, float3 tread, float painted, float desert,
                             float autumn, float cold, bool snowy, uint seed, float time) {
    int2 mapOrigin = tr.origin;
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
        else if (chim) a = mix(a, float3(dot(a, LUMA)), 0.5) * (wy > Ht - 0.4 ? 0.3 : 0.6);
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
            float da = hashf(pcg(sh + pk)) * TAU;
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
