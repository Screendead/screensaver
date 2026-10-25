// Screen tiles for the lamp haze and smoke lists (Renderer.lampTiles): a LAMP_TILES x LAMP_TILES grid over the frame.
constant int LAMP_TILES = 16;

fragment float4 sceneFragment(VOut in [[stage_in]], constant Frame& f [[buffer(0)]],
                              texture2d<float> hm [[texture(0)]], texture2d<float> cloudTex [[texture(1)]],
                              texture2d<float> clim [[texture(2)]], texture2d<uint> st [[texture(3)]],
                              texture2d<float> lightMap [[texture(4)]], texture2d<float> lightDir [[texture(5)]],
                              texture2d_array<float> coarse [[texture(6)]], texture2d_array<float> coarseClim [[texture(7)]],
                              texture2d_array<float> coarseCover [[texture(8)]],
                              texture2d<float> shadowNear [[texture(9)]], texture2d<float> shadowFar [[texture(10)]],
                              texture2d<float> slowTex [[texture(11)]], texture2d<float> hx [[texture(12)]],
                              texture2d<uint> gx [[texture(13)]], texture2d_array<float> cg [[texture(14)]],
                              texture2d_array<float> cg2 [[texture(17)]],
                              texture2d<float> hb [[texture(15)]], texture2d<float> pvTex [[texture(16)]],
                              texture2d_array<float> cbTex [[texture(18)]],
                              device const Site* sites [[buffer(1)]], device const float4* lampPos [[buffer(2)]],
                              device const float4* lampState [[buffer(3)]], constant int4* refs [[buffer(4)]],
                              device const float4* birds [[buffer(5)]], device const float* horizon [[buffer(6)]],
                              constant uint* tiles [[buffer(7)]], device const GiantRec* giants [[buffer(8)]], constant TowerTable* towers [[buffer(9)]],
                              constant IsleTable* isleTab [[buffer(10)]]) {
    float2 ndc = screenNdc(in.uv) + f.jitter.xy;
    float tanH = f.camPos.w;
    float3 d = viewRay(f, ndc);
    float3 o = f.camPos.xyz;

    int levels = int(f.fade.z);
    float gmax = f.misc.y;
    for (int k = 1; k < levels; k++) gmax = max(gmax, abs(coarse.read(uint2(0), uint(k - 1), coarse.get_num_mip_levels() - 1).r) + 1.0);
    if (int(f.shadow.y) & 2) gmax = min(gmax, 260.0);
    Terrain tr { hm, coarse, coarseCover, coarseClim, clim, f.lod, f.mapInfo.xy, f.mapInfo.z, levels, f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w, f.fade.y, f.fade.x, gmax, hx, gx, cg, cg2, hb, uint(f.mapInfo.w), giants, pvTex, towers, sceneTowers ? towers->n.x : 0, cbTex };
    tr.caves = f.birds.w;
    tr.caveNight = f.birds.w > 0.0 ? bioLandRamp(f.lights.x) : 0.0;
    tr.caveFire = mix(1.0, 0.02, smoothstep(-0.02, 0.05, f.lights.x)) * (1.0 - 0.75 * smoothstep(0.6, 0.66, f.lights.y));
    Ctx x = makeCtx(tr, f, in.pos.xy, cloudTex, st, sites, lampPos, lampState, lightMap, lightDir, shadowNear, shadowFar,
                    slowTex);
    {
        int2 ti = clamp(int2(in.uv * float(LAMP_TILES)), int2(0), int2(LAMP_TILES - 1));
        x.tiles = tiles[ti.y * LAMP_TILES + ti.x];
        x.towers = towers;
        x.isles = isleTab;
        x.isleShadows = f.shadow.w > 0.0;
    }

    // Debug map view: 11 level 4 (+-10k tall, +-16k wide), 12 level 2, 13 level 1; hillshade, height ramp, 100-voxel
    // contours, 2k grid, yellow 10k lines, magenta world origin, red camera.
    if (f.misc2.z >= 11 && f.misc2.z <= 13) {
        int k = f.misc2.z == 11 ? 4 : f.misc2.z == 12 ? 2 : 1;
        float cs = float(1 << k);
        float2 uv = in.uv - 0.5;
        uv.x *= f.resolution.x / f.resolution.y;
        float2 cell = f.camPos.xz / cs + float2(f.mapInfo.xy >> k) + uv * 1300.0;
        int2 ci = int2(floor(cell));
        float h0 = coarse.read(uint2(ci & (COARSE_N - 1)), uint(k - 1), 0).r;
        float hx = coarse.read(uint2((ci + int2(1, 0)) & (COARSE_N - 1)), uint(k - 1), 0).r;
        float hz = coarse.read(uint2((ci + int2(0, 1)) & (COARSE_N - 1)), uint(k - 1), 0).r;
        float3 n = normalize(float3(-(hx - h0) / cs, 1.0, -(hz - h0) / cs));
        float lit = saturate(dot(n, normalize(float3(-1, 1.2, 1))));
        float3 c = h0 <= SEA ? float3(0.15, 0.3, 0.55)
                 : h0 < 120 ? mix(float3(0.3, 0.5, 0.25), float3(0.55, 0.5, 0.3), h0 / 120.0)
                 : h0 < 600 ? mix(float3(0.55, 0.5, 0.3), float3(0.5, 0.35, 0.3), (h0 - 120.0) / 480.0)
                 : h0 < 1200 ? mix(float3(0.5, 0.35, 0.3), float3(0.6, 0.6, 0.65), (h0 - 600.0) / 600.0) : float3(0.95);
        c *= 0.35 + 0.9 * lit;
        if (floor(h0 / 100.0) != floor(hx / 100.0) || floor(h0 / 100.0) != floor(hz / 100.0)) c *= 0.6;
        // Gaps (overhangs, caves, bridges) in magenta.
        if (k <= GAP_LEVELS) { float2 gp = cg.read(uint2(ci & (COARSE_N - 1)), uint(k - 1)).rg; if (gp.y > gp.x) c = float3(1, 0, 1); }
        float2 wc = cell * cs;
        // Forest provinces: jungle warm, old growth cool; giants white.
        float3 pv = provincesBaked(pvTex, wc);
        c = mix(c, float3(0.9, 0.5, 0.1), 0.45 * smoothstep(0.15, 0.6, pv.x));
        c = mix(c, float3(1.0, 0.25, 0.55), 0.45 * smoothstep(0.2, 0.6, pv.z));
        c = mix(c, float3(0.2, 0.4, 1.0), 0.45 * smoothstep(0.2, 0.6, pv.y));
        { Giant gm; if (giantAt(giants, wc, 0.0, gm)) c = float3(1); }
        float px = 1300.0 * cs / f.resolution.y;
        float2 g1 = abs(wc - 2000.0 * floor(wc / 2000.0 + 0.5)), g2 = abs(wc - 10000.0 * floor(wc / 10000.0 + 0.5));
        if (any(g1 < px * 0.6)) c = mix(c, float3(0), 0.35);
        if (any(g2 < px * 1.2)) c = float3(1, 1, 0);
        if (all(abs(wc) < px * 4.0)) c = float3(1, 0, 1);
        if (length(uv) < 0.004) c = float3(1, 0, 0);
        return float4(c * c, 1);
    }
    const int primaryIters = 520;
    Hit h;
    h.hit = false; h.t = x.tr.rmax; h.iters = 0; h.level = 0;
    if (f.shadow.z == 0.0 || !aboveHorizon(horizon, d)) h = trace(x.tr, o, d, x.tr.rmax, primaryIters, 0);
    if (f.misc2.z == 1) return float4(float(h.iters) / float(primaryIters), h.hit ? 0.3 : 0.0, h.iters >= primaryIters - 1 ? 1 : 0, 1);
    if (f.misc2.z == 5) return float4(skyColor(d, x.L), 1);

    float tw = d.y < 0 ? (SEA - o.y) / d.y : 1e9;
    bool water = d.y < 0 && tw > 0 && (!h.hit || tw < h.t);
    if (f.misc2.z == 9 && (water || !h.hit)) return float4(0, 0, 0, water ? tw : 6e4);
    int nFlocks = int(f.birds.x);
    // Birds in front of land, water and sky. Their pixels carry negative depth, which tells the temporal resolve
    // that they move.
    if (nFlocks > 0) {
        // A trace out of steps has not shown the way clear: a bird past where it stopped could be behind a hill.
        float tFront = !h.hit && h.iters >= primaryIters ? 0.0 : water ? tw : (h.hit ? h.t : 6000.0);
        BirdHit bh = traceBirds(birds, nFlocks, o, d, tFront, 2.0 * tanH * f.resolution.w, 0.0);
        if (bh.t < tFront) {
            float3 bc = shadeBird(x, o + d * bh.t, bh);
            float seaB = cloudSeaAmount(x);
            if (seaB > 0.0 && f.misc2.z != 2) { SeaFog sf = cloudSea(x, seaB, o, d, bh.t); bc = bc * sf.T + sf.col; }
            if (f.misc2.z != 2) bc = applyFog(x, bc, o, d, bh.t);
            if (SCENE_MIST && x.isles->n.x > 0 && !(x.prof & 128)) {
                float4 m = isleMist(x, o, d, bh.t);
                bc = mix(bc, m.rgb, m.a);
            }
            bc = hazeAndSmoke(x, bc, sites, lampPos, lampState, refs, f.refs, o, d, bh.t);
            return float4(bc, -bh.t);
        }
    }
    // Sea past the world's edge is fully hazed: it takes the fog colour alone, with no ring of ocean around the land.
    if (water && length((o + d * tw).xz - x.tr.cam) >= x.tr.rmax) {
        if (f.misc2.z == 3) return float4(0, 0, 0, 1);
        float3 fogged = hazeAndSmoke(x, applyFog(x, float3(0), o, d, tw), sites, lampPos, lampState, refs, f.refs, o, d, tw);
        return float4(fogged, min(tw, 6e4));
    }
    bool birdRefl = false;
    float3 col;
    float tSurf;
    float Tfog = -1.0;
    // The lagoon bed's shading and the land's share one call: each call site inlines all of shade.
    float3 wp = 0, n = 0, r = 0;
    float2 ww = 0;
    float e0 = 0, lagoon = 0, lagRaw = 0, salt = 0, fres = 0, bedShows = 0, warm = 0;
    int wl = 0;
    Light Ls = x.L;
    bool shadeHit = false;
    SeaLife life = { half3(1), 1.0h, half3(0), 0.0h };
    float seaAmt = f.misc2.z != 2 ? cloudSeaAmount(x) : 0.0;
    // What the sea of cloud hides needs no shading. Only that is kept through the shading; the sea is taken again
    // after it, as its transmittance and colour held across it cost the scene pass registers on every pixel.
    bool seaHides = seaAmt > 0.0 && (water || h.hit) && cloudSea(x, seaAmt, o, d, water ? tw : h.t).T < 0.01;
    if (water) {
        wp = o + d * tw;
        ww = wp.xz + float2(x.origin);
        e0 = perlin3(float3(ww * 0.3, x.time * 0.25), x.seed + 60u);
        float ex = perlin3(float3((ww + float2(0.05, 0)) * 0.3, x.time * 0.25), x.seed + 60u);
        float ez = perlin3(float3((ww + float2(0, 0.05)) * 0.3, x.time * 0.25), x.seed + 60u);
        wl = levelAt(x.tr, length(wp.xz - x.tr.cam));
        Climate wc = climateL(x.tr, wl, wp.xz);
        lagRaw = lagoonAmount(wc);
        lagoon = smoothstep(0.0, 0.35, lagRaw);
        // Pools on a salt pan are a still skin over pale crust: a near-perfect mirror.
        salt = saltAmount(wc);
        warm = smoothstep(0.45, 0.6, wc.temp) * (1.0 - salt);
        float rs = max(0.35 * saturate(1.0 - tw * x.pixelAngle * 4.0) * (1.0 - 0.6 * lagoon), 0.08 * x.L.night) * (1.0 - 0.7 * salt);
        n = normalize(float3(-(ex - e0) * rs, 1.0, -(ez - e0) * rs));
        r = reflect(d, n);
        r.y = abs(r.y);
        float cosi = saturate(-dot(d, n));
        fres = 0.02 + 0.98 * pow(1.0 - cosi, 5.0);
        float bedPath = h.t - tw;
        // How much of the lit bed survives the water and the surface reflection; below a few thousandths it
        // can't show, and its shade() is the costliest part of a water pixel.
        bedShows = (1.0 - fres) * max(exp(-bedPath * mix(0.35, 0.08, salt)) * (1.0 - lagoon),
                                      lagoon * exp(-(bedPath + SEA - (o.y + d.y * h.t)) * 0.065));
        // Lagoon beds are lit by sunlight refracted steeply down through the surface. Flux per horizontal
        // area is kept (less what the surface reflects), so the light is redistributed from risers to treads.
        float2 kh = x.L.key.xz / 1.33;
        float3 kr = float3(kh.x, sqrt(max(1.0 - dot(kh, kh), 0.0)), kh.y);
        float ky = max(x.L.key.y, 0.0);
        Ls.key = normalize(mix(x.L.key, kr, lagoon));
        Ls.keyCol *= mix(1.0, ky / kr.y * (0.98 - 0.98 * pow(1.0 - ky, 5.0)), lagoon);
        shadeHit = h.hit && bedShows > 0.003 && !seaHides;
        // Reefs and fish, before the bed and the reflection are shaded: their result is small, and kept as halves.
        float fishAmt = warm * (0.4 + 0.6 * lagoon);
        bool wakes = x.L.night > 0.3;
        bool reefOn = lagRaw > 0.45 && h.hit && h.t - tw < 16.0 && h.t < REEF_FAR && (shadeHit || x.L.night > 0.0) && !(x.prof & 512);
        bool fishOn = fishAmt > 0.0 && (wakes ? tw < 300.0 : tw < 200.0 && bedShows > 0.01 && x.L.night < 0.3) && !(x.prof & 256);
        if (reefOn || fishOn) life = seaLife(x, o, d, h, tw, ww, lagRaw, lagoon, fishAmt, fres, bedShows, reefOn, fishOn);
    } else if (h.hit) {
        if (f.misc2.z != 2) Tfog = fogTransmittance(x, o, d, h.t);
        shadeHit = !((x.prof & 64) || (Tfog >= 0.0 && Tfog < 0.01) || seaHides);
    }
    float3 lit = shadeHit ? shadeL(x, Ls, o, d, h, water ? 0 : 1) : float3(0);
    if (water) {
        float3 refl;
        // Land in the reflected ray hides the sun or moon its glitter would mirror.
        float open = 1.0;
        if (x.quality > 0.5 && (int(f.shadow.y) & 4) == 0 && !seaHides) {
            // Rippled reflections of water past 300 trace from the 2-voxel ring: its far shore reads the same, at half
            // the steps.
            Hit rh = trace(x.tr, wp + float3(0, 0.02, 0), r, x.tr.rmax, 260, max(wl, tw > 600.0 ? 2 : tw > 300.0 ? 1 : 0));
            // Shadowed: unshadowed, a sun-facing wall in the shade of the far rim mirrors as a pale streak.
            if (rh.hit) {
                open = 0.0;
                // Land the haze all but hides needs no shading.
                float T = fogTransmittance(x, wp, r, rh.t, 2);
                refl = applyFogT(x, T < 0.01 || (x.prof & 8) ? float3(0.0) : shadeL(x, x.L, wp, r, rh, x.shadowMode == 1.0 ? 2 : 1, true), r, T,
                                 landSolid(x, wp + r * rh.t));
            } else {
                refl = softCloudSky(x, wp, r);
            }
            if (nFlocks > 0) {
                float tB = rh.hit ? rh.t : (rh.iters >= 260 ? 0.0 : 3000.0);
                BirdHit bh = traceBirds(birds, nFlocks, wp, r, tB, 2.0 * tanH * f.resolution.w, tw);
                if (bh.t < tB) { refl = applyFog(x, shadeBird(x, wp + r * bh.t, bh), wp, r, bh.t); birdRefl = true; }
            }
        } else {
            refl = softCloudSky(x, wp, r);
        }
        lit *= float3(life.litMul);
        float3 deep = x.L.zenith * 0.18 + x.L.horizon * 0.06 + float3(0.004, 0.014, 0.018);
        float3 under = deep;
        float bedPath = h.t - tw;
        if (h.hit) {
            float path = bedPath;
            under = mix(deep, lit * mix(float3(0.55, 0.8, 0.85), float3(0.9, 0.95, 0.97), salt), exp(-path * mix(0.35, 0.08, salt)));
            if (lagoon > 0.0) {
                // Clear water over sand: red is absorbed first, so the bed turns aquamarine, then teal, then blue.
                // Light crosses the water twice: down to the bed, then up along the view path.
                float3 T = exp(-(path + SEA - (o.y + d.y * h.t)) * float3(0.30, 0.09, 0.065));
                float3 glow = (x.L.zenith * 0.6 + x.L.keyCol * max(x.L.key.y, 0.0) * 0.12) * float3(0.06, 0.30, 0.50);
                float3 lag = lit * T + glow * (1.0 - T.g);
                // Keep the shallows soft rather than swimming-pool cyan; moonlit, only a hint of colour remains.
                lag = mix(lag, float3(dot(lag, LUMA)), 0.35 + 0.25 * x.L.night) * 0.9;
                under = mix(under, lag, lagoon);
            }
        }
        under *= 1.0 - 0.4 * x.L.night;
        under *= float(life.underMul);
        col = mix(under, refl, fres) + float3(life.add);
        // Glitter path toward the sun or moon: the main EDR highlight.
        col += x.L.keyCol * 6.0 * pow(max(dot(r, x.L.key), 0.0), mix(mix(900.0, 7000.0, x.L.night), 350.0, x.L.gold)) * fres * (x.L.shadows ? (x.L.shade < 1.0 ? mix(0.3, 1.0, x.L.shade) : 1.0) : 0.3) * (1.0 + 0.5 * x.L.night) * open;
        // Golden hour: a broad sheen of the aureole on the water under the sun.
        col += x.L.keyCol * 0.5 * pow(max(dot(r, x.L.key), 0.0), 40.0) * fres * x.L.gold * open;
        if (x.L.night > 0.0 && open > 0.0) {
            // Moon path: the chance that some ripple facet mirrors the moon (a slope distribution, so it is
            // smooth and TAA-stable). Slopes along the view spread wider than across it, so the path runs out
            // toward the horizon; a broad skirt under the core lets it fade out rather than stop.
            float3 hv = normalize(x.L.moon - d);
            float2 slope = hv.xz / max(hv.y, 0.05);
            float2 vh = normalize(d.xz + 1e-5);
            float sa = dot(slope, vh), sc = slope.x * vh.y - slope.y * vh.x;
            float q = sa * sa / 0.035 + sc * sc / 0.006;
            float cover = 0.55 * exp(-q) + 0.18 * exp(-0.25 * q);
            // Glitter: a facet lights where the ripple field clears a threshold set by the cover, so the core is a
            // broken sheen and the skirt thins to scattered sparks. Each octave hands over to the next coarser as
            // its features shrink toward a pixel, measured along the foreshortened view; far out, swells long
            // across the moon's bearing break the path into bands, and past them the cover itself remains.
            float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
            float fine = perlin3(float3(ww * 1.6, x.time * 0.7), x.seed + 61u);
            // Crests face the moon's midnight bearing (opposite SkyClock.azimuth at phase 0.75), within about 80
            // degrees of it while it shows. Facing the moon itself turned the axes with every step of its bearing about
            // a world origin some 10^5 voxels off, so the swell under the path jumped frame to frame.
            constexpr float az0 = 1.5 * M_PI_F + 0.9;
            const float2 mz = -float2(sin(az0), cos(az0));
            float swell = perlin3(float3(dot(ww, mz) * 0.07, (ww.x * mz.y - ww.y * mz.x) * 0.02, x.time * 0.1), x.seed + 62u);
            // An octave between the ripples and the fine facets makes the glints smaller where it resolves.
            float e1 = mix(e0, perlin3(float3(ww * 0.6, x.time * 0.4), x.seed + 64u), saturate(2.0 - fpv * 3.75));
            float rip = mix(swell, mix(e1, fine, saturate(2.0 - fpv * 10.0)), saturate(2.0 - fpv * 1.8));
            // The ripple noise is near-Gaussian (sd ~0.22); a logistic quantile turns cover into its threshold.
            float thr = 0.13 * log((1.0 - cover) / max(cover, 1e-4));
            // Facets brighten toward their crests rather than switching on flat, so each glint has a peak.
            float glint = mix(cover, smoothstep(thr - 0.08, thr + 0.40, rip), saturate(2.0 - fpv * 0.43));
            col += float3(0.75, 0.82, 1.0) * x.L.night * (0.05 * cover + 0.8 * glint) * x.L.moonAmt;
        }
        if (x.L.night > 0.0) {
            // Moonlit water rolls off toward a peak just under white (post-exposure 1.25, about half the dither
            // cells at the top level) instead of clipping: the glints keep a falloff rather than a flat plateau.
            float ex = f.misc.x, k = 0.55 / ex, P = 1.25 / ex;
            float m = max3(col.r, col.g, col.b);
            if (m > k) col *= mix(1.0, (k + (P - k) * (1.0 - exp(-(m - k) / (P - k)))) / m, x.L.night);
        }
        // Water past the 1-voxel map lies beyond the light map, which would clamp its edge onto it.
        if (x.lights.w > 0.0 && all(wp.xz > 0.0) && all(wp.xz < float(x.tr.size))) {
            // Lamplight on the water: a soft pool, and a broken streak drawn toward the viewer as ripples
            // stretch the moon's path.
            constexpr sampler bl(filter::linear, address::clamp_to_edge);
            float2 vh = normalize(d.xz + 1e-5);
            float inv = 1.0 / float(x.tr.size);
            float3 E = x.lightMap.sample(bl, wp.xz * inv, level(0)).rgb * 0.5 + x.lightMap.sample(bl, (wp.xz + vh * 4.0) * inv, level(0)).rgb * 0.3
                     + x.lightMap.sample(bl, (wp.xz + vh * 10.0) * inv, level(0)).rgb * 0.2;
            if (any(E > 1e-4)) {
                // Ripple facets on fixed world axes; the streak toward the viewer comes from the offset samples
                // above. A basis turned to each pixel's view direction would multiply the jitter by the world
                // coordinates and crawl. Facets hand over to their mean before they shrink to a pixel along the
                // foreshortened view.
                float ripple = perlin3(float3(ww * 1.6, x.time * 0.7), x.seed + 63u) + 0.3 * e0;
                float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
                float spark = mix(smoothstep(0.08, 0.38, ripple), 0.22, saturate(fpv * 2.0));
                col += E * (0.06 + 0.5 * spark) * x.lights.w;
            }
        }
        float glowAmt = smoothstep(0.03, -0.14, x.L.sun.y);
        if (glowAmt > 0.0 && !(x.prof & 128))
            col += seaGlow(x, d, wp, ww, wl, tw, e0, fres, glowAmt, salt, float(life.wake));
        tSurf = tw;
    } else if (h.hit) {
        col = lit;
        tSurf = h.t;
        // Debug: flat level colours, morph bands white.
        if (f.misc2.z == 9) {
            const float3 flat[5] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0), float3(1, 0, 1) };
            return float4(morphAt(x.tr, h.level, (o + d * h.t).xz) > 0.0 ? float3(1) : flat[h.level], h.t);
        }
        // Debug: levels tinted, morph bands striped toward the next level's tint.
        if (f.misc2.z == 8) {
            const float3 tint[5] = { float3(1, 0.3, 0.3), float3(0.3, 1, 0.3), float3(0.3, 0.5, 1), float3(1, 1, 0.3), float3(1, 0.3, 1) };
            float w = morphAt(x.tr, h.level, (o + d * h.t).xz);
            float l = dot(col, LUMA);
            return float4(mix(tint[h.level], tint[min(h.level + 1, 4)], w) * (0.15 + l), h.t);
        }
    } else {
        // seen: about half the dither's finest step near black, after exposure.
        float4 sky = skyWithClouds(x, f, o, d, meteor(f, x.L.night, d, x.time, x.seed), 0.004 / f.misc.x, in.pos.xy);
        if (SCENE_MIST && x.isles->n.x > 0 && !(x.prof & 128)) {
            float4 m = isleMist(x, o, d, 1e5);
            sky.rgb = mix(sky.rgb, m.rgb, m.a);
        }
        if (x.lights.w > 0.0 && f.refs.x > 0) sky.rgb += lampHaze(x, lampPos, lampState, refs, f.refs.x, o, d, 2500.0) * 0.05 * x.lights.w;
        if (f.refs.y > 0) {
            float4 sm = smoke(x, sites, lampPos, lampState, refs, f.refs.y, o, d, 1e9);
            sky.rgb = sky.rgb * (1.0 - sm.a) + sm.rgb;
        }
        return sky;
    }
    if (f.misc2.z == 10) {
        // Site footprints tinted by type.
        float3 sp = o + d * tSurf;
        int2 q = int2(floor(sp.xz));
        int2 home = int2(floor(float2(q + x.origin) / float(SITE_CELL))) - siteCell0(x.origin);
        const float3 tint[7] = { float3(0), float3(1, 0.2, 1), float3(0.6, 0.6, 0.6), float3(1, 0.5, 0), float3(1, 1, 0.2),
                                 float3(0.2, 0.6, 1), float3(1, 0.2, 0.2) };
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                int2 gi = home + int2(i, j);
                if (any(gi < 0) || any(gi >= SITE_GRID)) continue;
                Site st = sites[gi.y * SITE_GRID + gi.x];
                if (st.head.z == 0) continue;
                float dd = length(float2(q + x.origin - st.head.xy));
                if (dd < max(st.info.x, 6.0)) col = mix(col, tint[st.head.z] * 0.6, dd < 3.0 ? 0.9 : 0.45);
            }
    }
    if (f.misc2.z == 3) return float4(float3(fogTransmittance(x, o, d, tSurf)), 1);
    if (seaAmt > 0.0 && (water || h.hit)) { SeaFog sea = cloudSea(x, seaAmt, o, d, water ? tw : h.t); col = col * sea.T + sea.col; }
    if (f.misc2.z != 2) col = Tfog >= 0.0 ? applyFogT(x, col, d, Tfog, landSolid(x, o + d * h.t)) : applyFog(x, col, o, d, tSurf);
    // A summit above the cloud base is seen through the slab in front of it.
    if (!(x.prof & 32) && !water && h.hit && o.y < CLOUD_Y0 && o.y + d.y * h.t > CLOUD_Y0) {
        CloudSample cl = cloudSheet(x, o, d, h.t, 6);
        // A flat base would cut a level line across every wall that rises into it: the veil thickens up the wall.
        // Thin cover would wash the whole wall evenly, which reads as terrain gone translucent rather than as cloud:
        // the veil is drawn only where the cloud in front is dense, as an opaque body with soft edges.
        col = mix(col, cl.col, smoothstep(0.35, 0.8, cl.a) * smoothstep(CLOUD_Y0, CLOUD_Y0 + 110.0, o.y + d.y * h.t));
    }
    if (SCENE_MIST && x.isles->n.x > 0 && !(x.prof & 128)) {
        float4 m = isleMist(x, o, d, tSurf);
        col = mix(col, m.rgb, m.a);
    }
    col = hazeAndSmoke(x, col, sites, lampPos, lampState, refs, f.refs, o, d, tSurf);
    return float4(col, birdRefl ? -tSurf : tSurf);
}
