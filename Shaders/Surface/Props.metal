constant float3 PROP_DYES[5] = { float3(0.36, 0.35, 0.32), float3(0.235, 0.097, 0.085), float3(0.08, 0.092, 0.158),
                                 float3(0.319, 0.247, 0.145), float3(0.116, 0.158, 0.11) };
constant float3 WOOL[3] = { float3(0.17, 0.08, 0.055), float3(0.12, 0.12, 0.08), float3(0.19, 0.14, 0.075) };

static float3 propShade(Part P, float3 a, int feat, int3 v, int3 n, float2 uv, float2 wq, float wy, float hs, float Ht,
                        float fade, int2 sc, int2 mapOrigin, uint sh, uint seed, float autumn) {
    float floorY = float(P.y0), fy = wy - floorY;
    int style = (P.flags >> 12) & 15;
    if (feat != FT_POST) {
        int type = P.y1;
        bool az = P.sz.y > P.sz.x;
        int2 pr0 = sc + P.o;
        int pi = az ? v.z - pr0.y : v.x - pr0.x, pj = az ? v.x - pr0.x : v.z - pr0.y;
        int PL = max(P.sz.x, P.sz.y);
        bool endFace = n.y == 0 && (az ? n.z != 0 : n.x != 0);
        uint ch = pcg(sh + uint(P.y2) * 13u + uint(pi) * 7u + uint(pj));
        if (type == PR_WOODPILE || type == PR_LOGSEAT || type == PR_STUMP) {
            // Sawn ends: heartwood rings in bark; a woodpile's logs lie across it, so its long faces show their ends.
            bool ends = type == PR_WOODPILE ? n.y == 0 && !endFace : type == PR_LOGSEAT ? endFace : n.y > 0;
            if (ends) {
                float2 e = type == PR_WOODPILE ? float2(fract(hs * 2.0 + 0.5 * fmod(floor(wy * 2.0), 2.0)) - 0.5, fract(wy * 2.0) - 0.5) * 2.0
                         : type == PR_LOGSEAT ? float2(fract(hs) - 0.5, wy - floorY - 0.35) * 2.4 : (fract(wq) - 0.5) * 2.2;
                float r = length(e);
                float3 wood = float3(0.31, 0.25, 0.17) * (0.92 + 0.08 * sin(r * 22.0));
                float3 endc = r < 0.62 ? wood : r < 0.8 ? float3(0.15, 0.11, 0.07) : float3(0.05, 0.04, 0.03);
                a = mix(type == PR_WOODPILE ? float3(0.19, 0.13, 0.08) : float3(0.24, 0.17, 0.10), endc, fade);
            }
        } else if (type == PR_BALES) {
            a *= 1.0 - 0.5 * fade * (n.y == 0 && !endFace ? smoothstep(0.08, 0.02, abs(fract(hs * 1.5) - 0.5) - 0.2) : 0.0);
        } else if (type == PR_WASH) {
            a = PROP_DYES[ch % 5u] * (1.0 - 0.12 * fade * step(0.5, fract(hs * 3.0)));
        } else if (type == PR_RACK) {
            float3 item = style == ST_ADOBE || style == ST_PUEBLO ? float3(0.26, 0.08, 0.05)
                        : style == ST_TIPI ? float3(0.32, 0.24, 0.15) : float3(0.24, 0.24, 0.23);
            float strand = step(fract(hs * 3.0), 0.6) * (n.y == 0 ? 1.0 : 0.5);
            a = mix(mix(float3(0.04, 0.035, 0.03), item, 0.6), mix(float3(0.04, 0.035, 0.03), item, strand), fade);
        } else if (type == PR_BEDROLL) {
            float3 wool = WOOL[uint(P.y2) % 3u];
            a = pi == 0 ? wool * 0.85 : wool;
            if (pi != 0 && n.y > 0) a = mix(a, wool * 0.7, 0.6 * fade * step(abs(fract(az ? wq.y : wq.x) - 0.3), 0.08));
        } else if (type == PR_RUG) {
            float2 lq = wq - float2(mapOrigin) - float2(pr0);
            float2 hsz = float2(P.sz) * 0.5;
            float edge = min(hsz.x - abs(lq.x - hsz.x), hsz.y - abs(lq.y - hsz.y));
            float dm = abs(lq.x - hsz.x) + abs(lq.y - hsz.y);
            float3 field = PROP_DYES[1u + 2u * (uint(P.y2) % 2u)], border = PROP_DYES[(uint(P.y2) % 2u) != 0u ? 1 : 3], motif = PROP_DYES[0];
            float3 pat = edge < 0.3 ? border : dm < 0.7 ? motif : field;
            a = mix(field * 0.8 + border * 0.2, pat, fade);
        } else if (type == PR_CANOE) {
            bool inside = n.y > 0 && pi > 0 && pi < PL - 1;
            a = inside ? float3(0.04, 0.03, 0.02) : mix(float3(0.13, 0.09, 0.06), float3(0.24, 0.17, 0.10), step(Ht - 0.15, wy) * fade);
        } else if (type == PR_BARREL) {
            a = float3(0.16, 0.10, 0.055);
            float hoop = n.y == 0 ? max(step(abs(wy - floorY - 0.2), 0.06), step(abs(wy - floorY - 0.8), 0.06)) : step(0.35, length(fract(wq) - 0.5)) * step(length(fract(wq) - 0.5), 0.42);
            float stave = n.y == 0 ? step(0.44, abs(fract(hs * 4.0) - 0.5)) : 0.0;
            a *= 1.0 - fade * max(0.65 * hoop, 0.35 * stave);
        } else if (type == PR_PEAT) {
            a = float3(0.075, 0.055, 0.04) * (1.0 - 0.3 * fade * step(0.42, abs(fract(wy * 3.0) - 0.5)));
        } else if (type == PR_GARDEN) {
            if (feat == FT_FENCE) {
                a = float3(0.16, 0.12, 0.08);
            } else if (Ht - floorY < 0.4) {
                a = float3(0.12, 0.085, 0.055) * (0.9 + 0.2 * hashf(ch));
            } else {
                uint ck = hash2i(int2(floor(wq * 2.0)), seed + 350u);
                float3 leafc = float3(0.08, 0.15, 0.05) * (0.8 + 0.4 * hashf(ck));
                if (autumn > 0.2 && hashf(pcg(ck)) < 0.18) leafc = float3(0.40, 0.17, 0.03);
                a = mix(float3(0.09, 0.14, 0.05), leafc, fade);
            }
        }
    }
    if (P.y1 == PR_OVEN && n.y == 0) {
        // The oven's mouth, a dark arch low on one side.
        int m = int(uint(P.y2) & 3u);
        int2 md = m == 0 ? int2(1, 0) : m == 1 ? int2(-1, 0) : m == 2 ? int2(0, 1) : int2(0, -1);
        int2 rc = v.xz - (sc + P.o + 1);
        if (all(n.xz == md) && all(rc == md) && fy < 0.8 && abs(uv.x - 0.5) < 0.3)
            a = mix(a, float3(0.03, 0.02, 0.015), fade + (1.0 - fade) * 0.5);
    }
    return a;
}
