// A cave's dressing at world xz wq, height wy on a face of terrain column w: in green country (green 0...1) moss on its
// floor and the foot of its walls toward the light, ferns in the moss, ivy hanging over the mouth and roots under its lip;
// deep in, a few mosses glowing at night; and how much of the sky and the bounce its depth keeps from the face (s.cave).
static void caveDress(thread const Terrain& tr, thread Surface& s, int3 n, float2 w, float2 wq, float wy, float cs,
                      float fpA, bool underGap, float green, uint seed, float time) {
    CaveRec r;
    if (!caveFind(caveTable(tr.giants), w + 0.5 * cs, r)) return;
    float2 d = wq - r.m;
    float a = dot(d, r.u), b = dot(d, float2(-r.u.y, r.u.x));
    if (a < -10.0 || abs(b) > 2.0 * r.W + 6.0) return;
    float y = wy - r.yF;
    bool inside = underGap || n.y < 0;
    // Light from the mouth fades over a mouth's width or two.
    if (inside) s.cave = mix(0.45, 0.95, smoothstep(0.2 * r.W, 1.7 * r.W + 6.0, a));
    // Leaf texels a quarter voxel wide, at their mean once one nears a pixel.
    float near = 1.0 - smoothstep(0.08, 0.3, fpA);
    float3 tq = float3(wq.x, wy, wq.y) * 4.0;
    float leaf = mix(0.8, hashf(hash3i(int3(floor(tq)), seed + 929u)), near);
    if (green > 0.0) {
        float pn = perlin2(wq / 4.0, seed + 920u) + 0.5 * perlin2(wq / 1.7, seed + 921u);
        float lit = 1.0 - smoothstep(0.3 * r.W, 1.3 * r.W + 8.0, a);
        // Moss on the floor and the foot of the walls toward the light, and on the ground before the mouth.
        float moss = green * lit * smoothstep(-9.0, -3.0, a) * smoothstep(-0.35, 0.2, pn + 0.35 * lit);
        moss *= n.y > 0 ? step(abs(y - 0.5), 2.0) : n.y == 0 && inside ? step(-1.5, y) * (1.0 - smoothstep(0.5, 3.0, y)) : 0.0;
        float3 mossC = mix(float3(0.07, 0.10, 0.04), float3(0.10, 0.13, 0.05), saturate(pn * 0.5 + 0.5)) * (0.85 + 0.3 * leaf);
        s.albedo = mix(s.albedo, mossC, 0.9 * moss);
        s.tread = mix(s.tread, mossC, 0.9 * moss);
        // Ferns: brighter fronds in the moss.
        if (n.y > 0 && moss > 0.15) {
            float fr = hashf(hash2i(int2(floor(wq * 2.0)), seed + 922u));
            float fern = mix(0.3, step(0.6, fr) * (0.6 + 0.6 * leaf), near);
            s.albedo = mix(s.albedo, float3(0.09, 0.16, 0.045), fern * moss);
        }
        if (n.y == 0 && !inside && abs(b) < 1.3 * r.W + 2.0 && a > -4.0 && a < 5.0 && s.H >= r.yF + 0.5 * r.Hc) {
            // Ivy and creepers trailing from a ledge over the mouth, strands of every length, thinning out to either side.
            float along = b + 0.37 * a;
            uint sh = hash2i(int2(int(floor(along)), 3), seed + 924u);
            // Each strand from its own ledge, in whole voxels; its leaves end in quarter voxels.
            float rim = min(s.H, floor(r.yF + r.Hc + 4.0 + 3.0 * perlin2(wq / 9.0, seed + 928u)) + floor(3.0 * hashf(pcg(sh + 2u)) - 1.0));
            float side = green * (1.0 - smoothstep(0.6 * r.W, 1.3 * r.W + 2.0, abs(b)));
            float len = (r.Hc + 2.0) * (0.1 + 0.9 * hashf(pcg(sh)) * hashf(pcg(sh + 1u))) * side * step(0.45, hashf(sh));
            len = max(len, 1.5 * side * smoothstep(-0.2, 0.3, perlin2(float2(along / 4.0, 1.5), seed + 923u)));
            float hang = floor((rim - wy) * 4.0) * 0.25;
            // Leaves thin toward a strand's end.
            float ivy = step(0.0, hang) * step(hang, len) * step(hang / max(len, 1.0) * 0.8, leaf * 1.1);
            float3 ivyC = mix(float3(0.07, 0.11, 0.04), float3(0.12, 0.16, 0.055), leaf);
            s.albedo = mix(s.albedo, ivyC, mix(ivy, 0.5 * ivy + 0.3 * step(0.0, hang) * step(hang, len), 1.0 - near));
        }
        // Roots and creepers under the lip.
        if (n.y < 0 && a < 0.4 * r.W + 3.0) {
            float rt = green * smoothstep(0.0, 0.4, perlin2(wq / 2.2, seed + 926u) + 0.2);
            s.albedo = mix(s.albedo, float3(0.075, 0.075, 0.04) * (0.8 + 0.4 * leaf), 0.8 * rt);
        }
    }
    // How far past the 1-voxel ring's lamps (updateLamps' far fade) this face lies.
    float farW = smoothstep(tr.lod[0].x, tr.lod[0].y, length(wq - float2(tr.origin) - tr.cam));
    // A home's furnishing: the faces of its rise over the floor.
    float rise;
    int prop = (r.flags & CV_HOME) != 0u ? caveProp(r, dot(w + 0.5 * cs - r.m, r.u), dot(w + 0.5 * cs - r.m, float2(-r.u.y, r.u.x)), rise) : 0;
    if (prop != 0 && cs == 1.0 && wy >= r.yF - 0.01 && wy <= r.yF + rise + 0.01 && n.y >= 0) {
        float3 c = prop == CP_STONE ? float3(0.20, 0.19, 0.18) * (0.8 + 0.4 * leaf)
                 : prop == CP_FIRE ? float3(0.07, 0.055, 0.045)
                 : prop == CP_WOOD ? mix(float3(0.20, 0.13, 0.07), float3(0.29, 0.21, 0.12), step(0.5, fract(wy * 3.0)) * (n.y == 0 ? 1.0 : 0.3))
                 : mix(float3(0.30, 0.21, 0.13), float3(0.36, 0.30, 0.20), step(0.7, fract(dot(wq, r.u) * 1.5)));
        s.albedo = c;
        s.tread = c;
        s.level = n.y > 0 ? 1.0 : 0.0;
        if (prop == CP_FIRE) {
            // The flame, flickering as the site's hearth lamp does (placeSites: lamp 0 of the hearth's cell).
            int2 sc = int2(floor((w + 0.5) / float(SITE_CELL)));
            uint lh = pcg(hash2i(sc, seed + 200u));
            float fl = flicker(time, hashf(lh) * 40.0, seed);
            float flame = perlin3(float3(wq * 2.5, wy * 1.6 - time * 1.4), seed + 320u) * 0.5 + 0.5;
            flame = mix(flame, 0.5, smoothstep(0.05, 0.3, fpA));
            s.emit += LAMP_COL[L_HEARTH] * LAMP_EMIT[L_HEARTH] * fl * tr.caveFire * float3(1.0, 0.8, 0.6)
                    * (n.y > 0 ? mix(0.1, 0.45, flame) : mix(0.04, 0.3, flame)) * (1.0 - farW);
        }
    }
    // Past the 1-voxel ring the fire's lamp is gone (updateLamps): its light in the mouth glows on, so a lived-in cave
    // stays a warm spark in the mountainside at night out to the haze.
    if ((r.flags & CV_HOME) != 0u && farW > 0.0 && tr.caveFire > 0.03 && a > -2.0 && a < caveHearthA(r) + 2.0 && y < r.Hc + 1.0) {
        float g = (1.0 - smoothstep(0.6 * r.W, 1.2 * r.W, abs(b))) * (1.0 - smoothstep(0.5 * r.Hc, r.Hc + 1.0, y));
        // Gone with the mouth itself across level 3's morph band (its gap closes from the top there).
        g *= 1.0 - smoothstep(tr.lod[3].x, tr.lod[3].z, length(wq - float2(tr.origin) - tr.cam));
        s.emit += LAMP_COL[L_HEARTH] * LAMP_I[L_HEARTH] * 0.08 * g * farW * tr.caveFire;
    }
    // Deep in, a few mosses and fungi glow at night, in the land's glow hues, slow to swell.
    // Not in a lived-in cave: there the fire lights it.
    if (inside && tr.caveNight > 0.0 && (r.flags & CV_HOME) == 0u && a > 0.3 * r.L && y < 0.6 * r.Hc) {
        // In clumps on the floor and the foot of the walls: specks a quarter voxel across.
        float clump = smoothstep(0.15, 0.45, perlin2(wq / 4.5 + wy * 0.13, seed + 930u))
                    * (n.y > 0 ? 1.0 : n.y == 0 ? 1.0 - smoothstep(0.0, 0.5 * r.Hc, y) : 0.25);
        uint gh = hash3i(int3(floor(tq)), seed + 927u);
        float spot = clump * mix(step(hashf(gh), 0.3) * (0.5 + hashf(pcg(gh))), 0.3, smoothstep(0.08, 0.4, fpA));
        float pulse = bioPulse(wq, time, seed, 0.0);
        s.emit += bioHue(r.m, seed) * 0.45 * spot * pulse * tr.caveNight * (0.4 + 0.6 * green) * smoothstep(0.3 * r.L, 0.55 * r.L, a);
    }
}
