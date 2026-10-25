struct SeaLife { half3 litMul; half underMul; half3 add; half wake; };

// The reef and its fish at a water pixel: litMul tints the bed's sand shading to the reef's colours, underMul darkens
// the water under fish and their shadows, add is the glints and glowing corals over the surface; wake, for seaGlow to
// light, is the fish wakes' strength.
static inline SeaLife seaLife(thread const Ctx& x, float3 o, float3 d, Hit h, float tw, float2 ww, float lagRaw, float lagoon, float fishAmt, float fres, float bedShows, bool reefOn, bool fishOn) {
    float3 litMul = 1, add = 0;
    float underMul = 1, wake = 0;
    float bedPath = h.t - tw;
    if (reefOn) {
        // Reefs: the bed was shaded as sand; colonies and the turf between them take their colours as a ratio to it.
        float3 bp = o + d * h.t;
        float2 bw = bp.xz + float2(x.origin);
        float reg = 0.0, hand = 1.0;
        uint code = 0u;
        bool bed = true;
        float colTop = 0.0;
        if (h.level == 0) {
            uint2 vc = uint2(clamp(h.voxel.xz, int2(0), int2(x.tr.size - 1)));
            float4 hc = x.tr.hm.read(vc);
            bed = hc.r < SEA && hc.a == 0.0;
            colTop = hc.r;
            code = reefCode(x.tr.gx.read(vc));
            reg = float(code & 15u) / 15.0;
            hand = max(max(smoothstep(0.8, 2.0, h.t * x.pixelAngle), smoothstep(260.0, 420.0, h.t)), morphAt(x.tr, 0, bp.xz));
        } else {
            reg = reefRegion(bw + seedOffset(x.seed), lagRaw, x.seed);
        }
        if (reg > 0.0 && bed) {
            reg *= smoothstep(REEF_FAR, REEF_FAR - 250.0, h.t);
            float cw = smoothstep(0.5, 0.7, reg);
            float3 sandA = mix(float3(0.33, 0.29, 0.21), float3(0.37, 0.33, 0.245), lagoon);
            float3 alb = mix(sandA, REEF_TURF, 0.8 * smoothstep(0.3, 0.7, reg));
            float3 cc = alb;
            bool colony = (code & 16u) != 0u;
            // A colony's face in world units (a side's along it and up), and a mottle across its voxels: shaded per
            // voxel, each face reads as a tile.
            float2 fu = h.normal.y > 0 ? bw : float2(h.normal.x != 0 ? bw.y : bw.x, bp.y);
            float mot = colony ? seaNoise(fu * 0.6 + (h.normal.y > 0 ? 0.0 : 31.0), x.seed + 154u) + 0.5 : 0.5;
            if (colony) {
                Coral cr;
                cr.rise = 1.0; cr.kind = int((code >> 5) & 3u); cr.tip = float((code >> 7) & 1u); cr.h = code >> 8;
                cc = coralColour(cr);
                // Relief the water would flatten: sides in shade, each voxel down the colony darker and browner.
                float down = colTop - h.voxel.y - 1.0;
                cc *= (0.78 + float3(0.28, 0.22, 0.16) * mot)
                    * (h.normal.y > 0 ? 1.0 : 0.72) * mix(float3(1.0), float3(0.82, 0.76, 0.7), saturate(down * 0.5));
            }
            alb = mix(alb, mix(cc, REEF_MEAN, hand), cw);
            litMul = alb / sandA;
            if (x.L.night > 0.0 && cw > 0.0) {
                // Glowing colonies: polyps, points of light in clusters across the colony, blurred by the water over
                // them; sides fainter downward (fans fainter still, which side-on would stand as lit posts).
                float3 g = 0;
                float3 gc = hand < 1.0 && colony ? coralGlowCode(code, x.time) : float3(0.0);
                if (any(gc > 0.0)) {
                    // A point in each cell of 0.7 voxel, off the voxel grid; a pixel asks the 2x2 cells nearest it,
                    // which hold every point within half a cell, the widest a kernel grows.
                    float2 pq = fu * 1.4 + (h.normal.y > 0 ? 0.0 : 17.0);
                    int2 p0 = int2(floor(pq - 0.5));
                    float fpc = 1.4 * h.t * x.pixelAngle * rsqrt(max(abs(dot(d, float3(h.normal))), 0.1));
                    float Rw = sqrt(0.13 + 0.012 * max(SEA - bp.y, 0.0) + 0.6 * fpc * fpc);
                    float R = min(Rw, 0.5);
                    float occ = saturate(1.3 * mot - 0.25);
                    float pt = 0.0;
                    for (int j = 0; j < 2; j++)
                        for (int i = 0; i < 2; i++) {
                            int2 pc = p0 + int2(i, j);
                            uint ph = reefHash(pc, x.seed + 155u);
                            float2 pd = pq - float2(pc) - float2(byteOf(ph, 0), byteOf(ph, 1));
                            float k = saturate(1.0 - dot(pd, pd) / (R * R));
                            pt += step(byteOf(ph, 2), occ) * (0.45 + 0.55 * byteOf(ph, 3))
                                * (0.75 + 0.25 * sin(x.time * 0.7 + TAU * byteOf(ph, 3))) * (k * k * k + 0.3 * k);
                        }
                    // Energy kept as the kernel widens; far off, where points near a pixel, their mean.
                    pt = mix(pt * (0.13 / 1.3) / (R * R), 0.068 * occ, smoothstep(0.45, 0.5, Rw)) + 0.02 * occ;
                    float face = h.normal.y > 0 ? 1.0 : (((code >> 5) & 3u) == uint(CORAL_FAN) ? 0.3 : 0.6) * exp(-(colTop - bp.y) * 0.8);
                    g = 1.2 * gc * pt * face;
                }
                add += (g * (1.0 - hand) + mix(BIO_LAND[1], BIO_LAND[0], 0.3) * 0.012 * hand) * cw * x.L.night
                            * exp(-bedPath * float3(0.30, 0.09, 0.065)) * (1.0 - fres);
            }
        }
    }
    if (fishOn) {
        // By day fish are shadows and glints; from dusk, once they have faded, the wakes they leave glow in their stead.
        bool wakes = x.L.night > 0.3;
        float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
        float3 rd = refract(d, float3(0, 1, 0), 0.75);
        float2 q = ww + rd.xz / max(-rd.y, 0.2) * FISH_DEPTH;
        bool shadows = !wakes && h.hit && x.L.sunAmt > 0.0 && bedShows > 0.01;
        float2 bs = q;
        if (shadows) {
            float3 bp = o + d * h.t;
            float2 kh = x.L.key.xz / 1.33;
            float3 kr = float3(kh.x, sqrt(max(1.0 - dot(kh, kh), 0.0)), kh.y);
            bs = bp.xz + float2(x.origin) + kr.xz / kr.y * max(SEA - FISH_DEPTH - bp.y, 0.0);
        }
        FishSample fs = fishAt(x, q, bs, shadows, fpv, wakes);
        if (wakes) {
            wake = fs.wake * fishAmt * smoothstep(300.0, 150.0, tw) * smoothstep(0.3, 0.6, x.L.night) * x.L.night * (1.0 - 0.5 * fres);
        } else {
            float nb = fishAmt * smoothstep(200.0, 100.0, tw) * (1.0 - smoothstep(0.25, 0.6, fpv)) * (1.0 - smoothstep(0.0, 0.3, x.L.night));
            underMul = (1.0 - 0.5 * fs.body * nb * (1.0 - 0.5 * x.L.night)) * (1.0 - 0.3 * fs.shade * nb * x.L.sunAmt);
            add += x.L.keyCol * 0.12 * fs.glint * nb * x.L.sunAmt * (1.0 - fres);
        }
    }
    return SeaLife { half3(litMul), half(underMul), half3(add), half(wake) };
}
