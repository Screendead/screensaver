static float3 shade(thread const Ctx& x, thread const Light& L, float3 o, float3 d, Hit h, int shadows, bool inWater = false) {
    float3 p = o + d * h.t;
    float3 n = float3(h.normal);
    int k = h.level;
    float cs = x.tr.lod[k].w;
    float2 uv = faceUV(p, h.normal, cs);
    float fp = h.t * x.pixelAngle; // voxels per pixel
    // Footprint stretched across a grazing face, for detail that must fade before it aliases along the view.
    float fpA = fp / max(abs(dot(n, d)), 0.15);
    float bio = shadows != 0 && !(x.prof & 4096) ? bioLandRamp(L.sun.y) : 0.0;
    Surface s = surfaceAt(x.tr, x.st, x.sites, x.lampState, k, h.voxel, h.normal, uv, fp, fpA, p, x.seed, x.time, inWater, bio > 0.0);
    // Dressed here, not in surfaceAt: inlined there, among its many live values, it slowed every pixel.
    if (s.caveG >= 0.0) {
        int3 n3 = h.normal;
        float2 w = float2(h.voxel.xz * int(cs) + x.tr.origin);
        float2 wq = w + (n3.y != 0 ? uv : n3.x != 0 ? float2(n3.x > 0 ? 1.0 : 0.0, uv.x) : float2(uv.x, n3.z > 0 ? 1.0 : 0.0)) * cs;
        float wy = n3.y != 0 ? float(h.voxel.y + max(n3.y, 0)) : float(h.voxel.y) + uv.y;
        bool under = s.caveG >= 2.0;
        caveDress(x.tr, s, n3, w, wq, wy, cs, fpA, under, s.caveG - (under ? 2.0 : 0.0), x.seed, x.time);
    }
    if (sceneTowers && x.towers->n.x > 0) towerSurface(x, s, k, h.voxel, h.normal, uv, inWater ? max(fp, 4.0) : fp, inWater ? max(fpA, 4.0) : fpA, d);
    // Only a flag is kept through the shading below, which takes the wash again at its end: three floats held across
    // it cost the scene pass registers on every pixel.
    bool washOn = false;
    if (s.mat == 0 && h.normal.y > 0 && k == 0 && shadows == 1 && !inWater && s.pale > 0.5 && s.H > SEA && s.H <= SEA + 1.2) {
        float3 wash = shoreWash(x, p, s.H, fp);
        washOn = any(wash > 0.0);
        s.albedo *= mix(float3(1.0), float3(0.6, 0.55, 0.47), wash.x) * mix(1.0, 0.85, wash.y);
    }
    // A tower wall shading as its plane (s.nrm; terrain has none of its own): no corner AO, which would line every jog of
    // a diagonal face.
    bool planar = sceneTowers && s.mat == 0 && any(s.nrm != 0.0);
    // Prefilter: from ~1/10 voxel per pixel, relax AO and shadows and give risers their tread colour, so
    // distant stairs read as contour bands, not moire.
    float wf = smoothstep(0.025, 0.20, fp);
    float3 nS = normalize(float3(s.nh.x - s.nh.y, 2.0 * cs, s.nh.z - s.nh.w));
    // Below ~2 pixels per voxel step shadows and sky visibility alias into a hatch that crawls under motion;
    // shade the continuous slope under the stairs instead.
    // Taken just over half a column into the air in front of the face: on the face, or exactly half a column out, the
    // footprint's split between rock and air turns on rounding.
    // Floors are asked just above, undersides just below.
    float roofW = 0.0;
    // Only near the gaps' heights: a lip's top or the sheer wall under a shelter lies outside every gap around it.
    // Extra gaps lie under the principal ones, so below those only the floor of the map bounds them.
    if (s.mat == 0 && s.gapNear && p.y < s.gapSpan.y + 1.5 && (p.y > s.gapSpan.x - 1.5 || k == 0)) {
        // A wall is under a roof when the column in front of it is hollow at its height: every face of a jogging back
        // wall then agrees, where a share sampled at the face fell part way at each jog and striped the wall.
        if (h.normal.y == 0) roofW = hollowAtMorph(x.tr, k, h.voxel.xz + h.normal.xz, float(h.voxel.y) + 0.5) ? 1.0 : 0.0;
        else if (h.normal.y > 0) roofW = roofField(x.tr, k, p.xz, p.y + 0.5).x;
    }
    // An underside is its own roof: a share fading toward the lip's edge would light the edge's undersides as a pale line.
    if (s.mat == 0 && h.normal.y < 0) roofW = 1.0;
    // Faces under a roof, and undersides, are not stairs on a heightfield: no heightfield prefilter applies to them. A
    // lip's front and top are ordinary wall and ground.
    float ov = s.mat == 0 ? max(roofW, h.normal.y < 0 ? 1.0 : 0.0) : 0.0;
    float wf2 = s.mat == 0 ? smoothstep(0.2, 0.5, fp) * (1.0 - ov) : 0.0;
    bool built = s.mat >= 4;
    // Tops a few pixels wide keep their exact normal: a column's central difference tilts only the tops at a
    // ledge, so they would shade unlike the rest of their plane. Toward 2 pixels per voxel the slope under the
    // stairs takes over, smooth across ledges. Side faces blend with their tread colour (wf), toward the column
    // normal until that slope takes over: a riser in tread colour lit as a wall reads as a veil over the hill
    // behind it. Canopies keep their column normal blend.
    float wt = smoothstep(0.33, 0.6, fp);
    // Structures keep their own planes: a column-difference normal would tilt a house's edge columns.
    float wn = (built ? 0.0 : s.mat == 0 && h.normal.y != 0 ? wt : wf) * (1.0 - ov);
    // An island's top sees the lowland past its rim no lower than a step down: that drop would turn every tread within
    // the smoothing's reach of the rim outward, a lit ring at one inset round the top.
    float gFloor = ISLE_SHAPE && s.gy >= 1e8 && h.normal.y > 0 ? s.H - 6.0 : -1e9;
    if (s.mat == 0 && wt > 0.0 && s.level < 1.0 && shadows != 2) {
        // Side faces take the wall's direction over several columns: a single column's jog in a wall would otherwise
        // turn a narrow face to the light and draw a pale dash down the wall.
        float R = max(max(h.normal.y != 0 ? 2.0 : 6.0, 1.5 * fp), cs);
        float2 grad = float2(max(groundSmoothL(x.tr, k, p.xz + float2(R, 0), s.gy), gFloor) - max(groundSmoothL(x.tr, k, p.xz - float2(R, 0), s.gy), gFloor),
                             max(groundSmoothL(x.tr, k, p.xz + float2(0, R), s.gy), gFloor) - max(groundSmoothL(x.tr, k, p.xz - float2(0, R), s.gy), gFloor)) / (2.0 * R);
        float3 g = normalize(float3(-grad.x, 1.0, -grad.y));
        nS = h.normal.y != 0 ? g : normalize(mix(nS, g, wt));
    }
    float3 nl = normalize(mix(normalize(n - float3(s.slope.x, 0.0, s.slope.y)), nS, wn));
    if (any(s.nrm != 0.0)) nl = s.nrm;
    nl = normalize(mix(nl, float3(0, 1, 0), s.level));
    // Coarse levels are always past the footprint where corner AO is fully replaced.
    // Seen in water (reflections, the bed) the surface is rippled out of focus: no corner AO, a coarse sky visibility.
    bool lite = shadows != 1;
    // A wall under a roof skips corner AO too: every jog of a back wall is a concave corner, a dark line per column.
    float ao = (s.level < 1.0 ? mix(k == 0 && !lite && !(h.normal.y == 0 && roofW > 0.0) && !planar ? mix(cornerAO(x.tr.hm, x.tr.size, h.voxel, h.normal, uv), 0.85, wf) : 0.85, 1.0, s.level) : 1.0) * s.canopy;
    // A pitched roof's treads and risers are one plane: no corner AO lines along its steps, and (below) one shadow.
    bool roofPlane = built && s.nrm.y > 0.1;
    if (roofPlane) ao = 0.9;
    float drop = s.H - min(min(s.nh.x, s.nh.y), min(s.nh.z, s.nh.w));
    // Cliffs several pixels tall keep their strata; only sub-pixel stairs take the tread colour. Keyed to the
    // column's deepest drop, not this face's: the faces of a wall running diagonal to the grid alternate
    // between shallow and deep neighbours, which would stripe it strata, tread, strata. A wall built of steps a
    // few voxels each is as much a cliff as a sheer one; by drop alone it turns a pale tread-coloured band.
    float steep = 0.5 * length(float2(s.nh.x - s.nh.y, s.nh.z - s.nh.w));
    float cliff = s.riser > 0.0 ? max(drop, 2.0 * steep) : 0.0;
    float keep = smoothstep(6.0, 14.0, cliff / fp) * smoothstep(3.0, 5.0, cliff);
    // Among the great walls every face is rock: column by column, some faces of a wall would otherwise take the
    // tread colour and stripe it.
    // So is every riser of a tall, steep staircase wall: kept by its own drop alone, its deeper columns would show as
    // strata dashes among tread-coloured ones.
    keep = max(keep, max(s.wall, smoothstep(1.0, 2.5, steep) * smoothstep(100.0, 160.0, s.H)) * smoothstep(1.0, 3.0, cliff));
    // Overhangs keep their strata and their own faces' normals.
    keep = max(keep, ov);
    // A slope steep over a wide span (+-16 voxels and more) is seen almost edge-on to its treads: from afar it is a
    // rock face. Its risers painted in the tread colour read as a veil of the ground above (grass, snow, hypsometric
    // bands) hung across the rock, and its treads as pale lines ruled across it, so both take the face's rock.
    float sus = 0.0;
    // A planar wall is its own plane already: the slope under the stairs is the ground's, and column by column it would
    // turn each face of a jogging wall its own way.
    if (s.mat == 0 && wf > 0.0 && s.H > SEA + 20.0 && shadows != 2 && !planar) {
        float R2 = max(16.0, 2.0 * cs);
        float2 g2 = float2(max(groundSmoothL(x.tr, k, p.xz + float2(R2, 0), s.gy), gFloor) - max(groundSmoothL(x.tr, k, p.xz - float2(R2, 0), s.gy), gFloor),
                           max(groundSmoothL(x.tr, k, p.xz + float2(0, R2), s.gy), gFloor) - max(groundSmoothL(x.tr, k, p.xz - float2(0, R2), s.gy), gFloor)) / (2.0 * R2);
        sus = smoothstep(0.6, 1.1, length(g2)) * (1.0 - ov);
        if (h.normal.y == 0) keep = max(keep, sus);
        // Treads and risers alike light as the face they make up, or every tread is a lit line across it.
        nl = normalize(mix(nl, normalize(float3(-g2.x, 1.0, -g2.y)), wf * sus));
    }
    // A riser face only a column wide (a jog in a step edge) is a lone cube face: lit, it is a bright cube; in shade,
    // stone against sand, a dark notch. With the footprint it fades to the tread it stands for, colour and light.
    float lone = 0.0;
    // A planar wall's jogs are its plane's.
    if (h.normal.y == 0 && s.mat == 0 && wn < 1.0 && shadows != 2 && !planar) {
        int2 t = h.normal.x != 0 ? int2(0, 1) : int2(1, 0), o = h.normal.xz;
        float vy = min(float(h.voxel.y), s.gy);
        float run = float(groundL(x.tr, k, h.voxel.xz + t, vy) > groundL(x.tr, k, h.voxel.xz + t + o, vy) + 0.5)
                  + float(groundL(x.tr, k, h.voxel.xz - t, vy) > groundL(x.tr, k, h.voxel.xz - t + o, vy) + 0.5);
        lone = (1.0 - 0.5 * run) * smoothstep(0.01, 0.05, fp) * (1.0 - wn) * (1.0 - keep) * (1.0 - ov);
    }
    float3 alb = h.normal.y == 0 ? mix(s.albedo, s.tread, max(wf * (1.0 - keep), lone)) : mix(s.albedo, s.face, wf * sus);
    float deep = L.night * smoothstep(0.32, 0.45, -L.sun.y);
    float dusk = s.mat == 1 ? leafDusk(L) : 0.0;
    float3 albK;
    alb = gradeAlbedo(L, alb, deep, s.mat == 1 || s.mat == 2, s.autumn, dusk, albK);
    float ndl = mix(max(dot(nl, L.key), 0.0), max(L.key.y, 0.0), lone);
    // A low moon lights a step's riser several times brighter than its tread, drawing a highlight line along
    // every step; under the moon only cliffs outshine the ground they stand in. A canopy's steps are steps too: no
    // strata to keep, and its moon-facing risers would rule a pale line along every crest.
    if (h.normal.y == 0 && s.mat == 0) ndl = mix(ndl, min(ndl, max(L.key.y, 0.0)), L.night * (1.0 - keep));
    if (h.normal.y == 0 && s.mat == 1) ndl = mix(ndl, min(ndl, max(L.key.y, 0.0)), L.night);
    // Low structure faces (cairns, boats, deck edges, fallen stones) are steps too; uprights and house walls stay walls.
    if (h.normal.y == 0 && built && drop <= 2.0 && !any(s.nrm != 0.0)) ndl = mix(ndl, min(ndl, max(L.key.y, 0.0)), L.night);
    float hiSun = smoothstep(0.15, 0.45, L.key.y) * (1.0 - L.night);
    // Under the moon a shelter's back wall takes the moon where the shadow ray reaches it, lit as the cliff's direction
    // (across the ground past the deepest lip, 40): its own faces' normals alternate light and dark along every jog of
    // a back wall, and the ground a few columns round it is the lip's flat top. By day the roof shades it whole.
    float ndlR = 0.0;
    if (roofW > 0.0 && L.night > 0.0 && shadows > 0 && L.shadows && x.shadowMode != 3.0) {
        float3 nR = nl;
        if (h.normal.y == 0) {
            float R = max(48.0, 2.0 * cs);
            float2 gr = float2(groundSmoothL(x.tr, k, p.xz + float2(R, 0)) - groundSmoothL(x.tr, k, p.xz - float2(R, 0)),
                               groundSmoothL(x.tr, k, p.xz + float2(0, R)) - groundSmoothL(x.tr, k, p.xz - float2(0, R))) / (2.0 * R);
            nR = normalize(float3(-gr.x, 1.0, -gr.y));
        }
        ndlR = max(dot(nR, L.key), 0.0) * L.shade;
    }
    float sh = 1.0, shRay = 1.0;
    if (shadows > 0 && (ndl > 0.0 || ndlR > 0.0) && L.shadows && x.shadowMode != 3.0) {
        // Keep the origin inside the hit voxel's footprint so boundary columns can't self-shadow. A planar wall's ray
        // starts out in front of its plane: from a jog's own face it would cross the next column and stripe the wall.
        float3 so = p + (planar ? s.nrm * 1.5 : float3(n) * 0.02);
        if (!planar) so.xz = clamp(so.xz, float2(h.voxel.xz) * cs + 0.002, float2(h.voxel.xz) * cs + (cs - 0.002));
        if (roofPlane && h.normal.y == 0) so.y = max(so.y, s.H + 0.02);
        // Prefiltered, the ray starts a step above the column, and single steps shade by their expected cover.
        so.y = mix(so.y, max(so.y, s.H) + 1.0, wf2);
        // With the shadow-height maps the ray only reaches SHADOW_NEAR (contact and cliff shadows); the maps hold every
        // occluder beyond.
        bool maps = x.shadowMode == 1.0, brute = x.shadowMode == 2.0;
        int shadowIters = brute ? 4000 : maps ? (x.quality > 0.5 ? 48 : 32) : (x.quality > 0.5 ? 120 : 48);
        Hit sHit;
        sHit.hit = false; sHit.t = 1e9; sHit.iters = 0;
        // Past the 1-voxel map a pixel spans columns: the maps alone, with occluders from two texels out.
        // Under the moon, whose shadows are soft and faint, only near ground keeps the traced ray.
        bool mapOnly = maps && (shadows == 2 || k >= 1 || h.t > (L.night > 0.5 ? 120.0 : 500.0));
        // Where the maps shade the point fully, the traced ray can only agree. Maps baked for a light on or under the
        // horizon shade nothing: there the lookup waits until after the ray.
        bool mapFirst = maps && !mapOnly && L.key.y > 0.0;
        float mapLit = mapFirst ? shadowMapLit(x, so, false) : 1.0;
        if (!mapOnly && mapLit > 0.0) sHit = trace(x.tr, so, L.key, brute ? 6000.0 : maps ? SHADOW_NEAR : 700.0, shadowIters, k);
        // A ray that runs out of steps is still threading terrain toward the sun: count it as blocked there, so a
        // shaded spot never comes out lit.
        sHit.hit = sHit.hit || sHit.iters >= shadowIters;
        // Contact shadows from single-voxel steps draw sawtooth noise; keep them faint, cliffs full.
        float contact = 0.35 * (1.0 - smoothstep(2.0, 8.0, 0.5 / L.key.y));
        // The lift moves the ray's start about 1 / L.y along it; contact reach moves with it.
        float reach = wf2 / max(L.key.y, 0.05);
        // A roof overhead is never a contact shadow.
        sh = sHit.hit ? (sHit.normal.y < 0 && sHit.iters < shadowIters ? 0.0 : 0.35 * (1.0 - smoothstep(2.0 + reach, 8.0 + reach, sHit.t))) : 1.0;
        if (maps) sh = min(sh, mapFirst ? mapLit : shadowMapLit(x, so, mapOnly));
        // Under a roof the roof stands in for the ray: the maps see an overhang only from two texels out, and a traced ray
        // that slips out under the edge on one row and not the next stripes the back wall with sun.
        if (wf2 > 0.0 && !planar) sh = mix(sh, min(sh, stairShadow(x.tr, k, p.xz, s.gy, L.key, d, contact, 0.35, fp > 3.0 || shadows == 2)), wf2);
        float floorSh = mix(mix(0.5, 0.25, hiSun), 0.32, L.gold);
        sh = mix(sh, floorSh + (1.0 - floorSh) * sh, wf);
        if (L.shade < 1.0) sh = mix(1.0, sh, L.shade);
        shRay = sh;
        sh = mix(sh, 1.0, 0.3 * L.night);
        // After the lifts, which stand for light slipping between distant steps: none slips under a roof.
        sh = min(sh, 1.0 - roofW * L.shade);
    }
    if (!(x.prof & 32) && L.sunAmt > 0.0) sh *= mix(1.0, cloudShadow(x, p, L.key), L.sunAmt);
    float lowFill = mix(0.9, 1.0, smoothstep(0.1, 0.4, L.key.y));
    // Low light: shadows take the cool zenith, against warm key light.
    // High sun: fill is the deep zenith, so shade reads blue against warm key light.
    float zw = mix(mix(0.55, 0.8, hiSun), 0.85, L.gold);
    float3 skyFill = mix(L.horizon, L.zenith, saturate(zw + 0.15 * nl.y)) * 0.9 * lowFill * mix(1.0, 0.85, hiSun);
    // Golden hour: shade is the violet dome opposite the sun, well below the key, so hillsides keep dark values.
    skyFill *= mix(float3(1.0), float3(0.62, 0.58, 0.85), L.gold * L.sunAmt);
    // Under a roof the visibility is taken where the point is, not at its column's centre: column by column it would
    // jump between reaching past the roof's edge and not, striping the wall.
    float2 svAt = planar ? p.xz + s.nrm.xz * 1.5 : mix((float2(h.voxel.xz + h.normal.xz) + 0.5) * cs, p.xz, max(wf2, ov));
    float svY = wf2 > 0.0 ? mix(p.y, groundSmoothL(x.tr, k, p.xz, s.gy), wf2) : p.y;
    // Far off, an island's flanks and keel take the open sky and lowland round and under it: its own disc, read as a
    // shelter's roof, would make every distant keel a black cut-out.
    float svIsle = ISLE_SHAPE && s.gy >= 1e8 ? smoothstep(0.8, 2.5, fp) : 0.0;
    float sv = x.prof & 16 || roofW >= 1.0 || s.isleUnder || svIsle >= 1.0 ? 1.0 : mix(skyVisibility(x.tr, k, svAt, svY, lite || fp > 5.0 ? 2 : fp > 2.0 ? 1 : 0), 1.0, svIsle);
    // Under a roof the sky comes only through the mouth, an even share well under open shade's: the visibility samples
    // only six voxels out, so in a shelter it reads the roof all round and varies column by column with the back wall's
    // jogs, striping it.
    // By night a shelter's back wall takes open shade's share: at the roof's it fell below the open wall beside it, a
    // black band under every lip. Toward the roof it keeps the roof's share, the lip's shadow along the top; the roof is
    // a level bedding plane, so its height reads alike for every jog of the wall.
    float roofSky = ROOF_SKY;
    // From well into dusk: before that the night fill the roof withholds is a sliver.
    if (h.normal.y == 0 && roofW > 0.0 && L.night > 0.1)
        roofSky = mix(ROOF_SKY, ROOF_SKY_NIGHT, smoothstep(0.1, 1.0, L.night) * smoothstep(0.0, max(8.0, 2.0 * cs), s.gapSpan.y - p.y));
    skyFill *= mix(1.0 - mix(0.55, 0.85, hiSun) * (1.0 - sv), roofSky, roofW);
    // Dusk: a crown's top still holds the glow while its foot and the gaps between crowns sink deeper and cooler than
    // open shade, so the crowns stand apart.
    if (dusk > 0.0) skyFill *= mix(float3(1.0), mix(float3(0.52, 0.58, 0.8), float3(1.12, 1.08, 1.0), saturate(sv * sv * (0.4 + 0.75 * s.up))), dusk);
    // An island's own rock is the occluder itself (s.gy): under its own disc its keel would go black.
    if (SCENE_ISLES && x.isles->n.x > 0 && s.gy < 1e8) skyFill *= 1.0 - 0.85 * isleSkyOcc(x, p);
    // A wall sees half the dome, and the half it sees is the dimmer horizon side; without this a riser in shade
    // outshines the sunlit tread above it under a low sun.
    skyFill *= built ? 0.7 + 0.3 * saturate(nl.y) : 0.5 + 0.5 * saturate(nl.y);
    // Light bounced into a hollow comes in through its mouth, off the lit ground outside, which a low sun still lights
    // face on where it reaches the shelter's own floor: dim, never black; an underside stays well under the lit top of
    // its slab.
    float keyUp = mix(max(L.key.y, 0.0), max(L.key.y, 0.3) * saturate(L.key.y * 10.0), ov);
    // An island's keel hangs over sunlit lowland, which lights it from below more than a shelter's mouth does.
    float3 bounce = (1.0 - saturate(nl.y)) * (built || (ISLE_SHAPE && s.gy >= 1e8) ? 0.6 : 0.35) * alb * L.keyCol * keyUp;
    // High sun: risers sit in cooler shade so terraces keep their modelling at noon.
    float3 riserShade = mix(float3(1.0), float3(0.82, 0.86, 0.95),
                            (h.normal.y == 0 ? 1.0 - saturate(nl.y) : 0.0) * smoothstep(0.3, 0.6, L.key.y) * smoothstep(1.5, 4.0, drop) * (1.0 - ov));
    // Night: no pure-black riser feet; a faint indigo ambient keeps the steps readable as stone, not holes.
    ao = mix(ao, 1.0, 0.25 * L.night);
    // Night: a cave mouth stays a dark hole.
    skyFill += float3(0.09, 0.12, 0.25) * L.night * mix(1.0, roofSky, roofW);
    // Deep in the night, land sits below the sky so night reads as night (won night A/B): moonlight is graded
    // to a cool silver and the fill to a deep blue well under it, so slopes facing the moon still model. In
    // early night it is already dark and keeps its contours (darkening lost the dusk A/B).
    // Deep in a cave neither the sky nor the light bounced in at its mouth reaches.
    // A cave's floor sees the sky only level through the mouth, off the horizon and the ground before it: grey, not the
    // zenith's blue.
    if (s.cave > 0.0) skyFill = mix(skyFill, float3(dot(skyFill, float3(0.3, 0.55, 0.15))), 0.75 * saturate(nl.y)) * (1.0 - s.cave);
    bounce *= 1.0 - s.cave;
    float3 col = albK * L.keyCol * ndl * sh * mix(1.0, 1.1, L.night) * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep)
               + alb * (skyFill * ao * riserShade + bounce) * mix(float3(1.0), float3(0.19, 0.23, 0.38), deep);
    // Where the ray finds the lip, the night lift's share of the moon at the roof's sky share.
    if (ndlR > 0.0) {
        float top = ndlR * max(shRay, 0.3 * ROOF_SKY) * L.night * roofW - ndl * sh;
        if (top > 0.0) col += albK * L.keyCol * top * mix(1.0, 1.1, L.night) * mix(float3(1.0), float3(0.40, 0.42, 0.47), deep);
    }
    // Low sun: up-facing treads take a warm sky-dome tint instead of going to charcoal.
    col += albK * L.keyCol * 0.07 * (1.0 - smoothstep(0.05, 0.35, L.key.y)) * saturate(nl.y) * (1.0 - L.night);
    // Backlit leaves: a low sun glows through the canopy edges facing away from it.
    if (s.mat == 1) {
        float back = pow(saturate(dot(d, L.key)), 4.0) * (1.0 - smoothstep(0.08, 0.45, L.key.y)) * (1.0 - L.night);
        float graze = 1.0 - abs(dot(n, d));
        col += alb * mix(float3(0.9, 1.3, 0.3), float3(1.3, 0.9, 0.4), s.autumn) * L.keyCol * back * (1.0 - ndl) * (1.0 - abs(n.y)) * graze * graze * sh;
    }
    // The glowing band low over the sun's side of the sky warms the crowns' flanks that face it, the upper ones most,
    // even where the sun itself is hidden: leaves model warm against cool, and seen toward the sun only their rims catch it.
    if (dusk > 0.0) {
        float face = saturate(dot(nl, normalize(float3(L.sun.x, 0.25, L.sun.z)))) * (1.0 - 0.6 * saturate(nl.y));
        col += albK * L.skyHor * float3(1.0, 0.85, 0.7) * face * (0.1 + 0.9 * s.up * s.up) * sv * 0.8 * L.gold * (1.0 - L.night) * (1.0 - 0.5 * L.sunAmt);
    }
    // Waxy crown tops mirror the low sky: amber toward the sun, lavender away from it. Tops only: on side faces the
    // grazing sheen would rule a pale line along every step.
    if (dusk > 0.0) {
        float3 rv = reflect(d, nl);
        float tw = 0.5 + 0.5 * dot(normalize(rv.xz + 1e-5), normalize(L.sun.xz + 1e-5));
        float3 env = mix(L.skyHor * mix(float3(0.72, 0.66, 1.0), float3(1.0, 0.8, 0.55), tw * tw), L.skyZen, saturate(rv.y * 3.0));
        float fr = min(0.02 + 0.3 * pow(1.0 - saturate(-dot(d, nl)), 5.0), 0.08);
        col += env * fr * saturate(nl.y) * sv * (0.6 + 0.4 * sh) * dusk * s.up * s.up;
    }
    if (washOn) {
        float3 wash = shoreWash(x, p, s.H, fp);
        // Wet sand and the film on it mirror the sky, the film as water does; the foam takes the sand's light.
        float3 rv = reflect(d, float3(0, 1, 0));
        float3 env = mix(L.skyHor, L.skyZen, saturate(rv.y * 2.5));
        // The night sky is graded brighter than the moonlit ground it lights: mirrored at grazing Fresnel the band would
        // glow above the dry sand, so by night it takes the matte ground's cap and only the moon's glint shows.
        float fr = min(0.02 + 0.98 * pow(1.0 - saturate(-d.y), 5.0), mix(1.0, 0.06, L.night));
        float gloss = 0.2 * wash.x + 0.8 * wash.y;
        col += (env * fr + L.keyCol * 0.8 * pow(max(dot(rv, L.key), 0.0), 24.0) * fr * sh) * gloss * sv;
        float3 irr = col / max(dot(alb, LUMA), 0.02);
        // Foam is pale: under the moon it takes the same knee as the sand's albedo.
        float kn = 4.5 * L.night;
        col = mix(col, irr * 0.6 * (1.0 + 0.1 * kn) / (1.0 + 0.6 * kn), 0.8 * saturate(wash.z));
    }
    // Lamplight is added in absolute terms, outside the moonlight grade, so firelit ground stays warm in a blue night.
    if (x.lights.w > 0.0) {
        constexpr sampler bl(filter::linear, address::clamp_to_edge);
        float2 mu = p.xz / float(x.tr.size);
        // The light map spans the 1-voxel map; coarse ground past it would take its clamped edge.
        float3 Lm = all(mu > 0.0) && all(mu < 1.0) ? x.lightMap.sample(bl, mu, level(0)).rgb : float3(0);
        if (any(Lm > 1e-4)) {
            float4 Ld = x.lightDir.sample(bl, mu, level(0));
            float3 lp = Ld.xyz / max(Ld.w, 1e-6);
            // Hills above a fire and valleys below it stay dark.
            float dy = (p.y - lp.y) / 5.0;
            float face = 0.3 + 0.7 * saturate(dot(nl, normalize(lp - p + float3(0, 0.3, 0))));
            float glow = s.mat == M_CLOTH || s.mat == M_FELT ? 0.2 : 0.0;
            col += s.albedo * Lm * ((face * mix(1.0, ao, 0.5)) / (1.0 + dy * dy) + glow) * x.lights.w;
        }
    }
    // From early twilight, out to the haze. Reflections shade with a wide footprint, so they take the means.
    if (bio > 0.0 && s.bio != 0.0 && s.mat < 4 && !planar && s.pale < 0.5) {
        // Light given off loses less to the haze than light reflected: fogged as T^0.5 rather than T (the valley mist
        // left out), and as 2.5 T in thick haze, so it still fades into the haze at the very horizon.
        float lift = rsqrt(max(fogTransmittance(x, o, d, h.t, 0), 0.16));
        col += bioLight(x, s, k, h.normal, p, alb, fp, sv) * bio * lift;
    }
    // An island's glow, from late dusk; gone before the 1-voxel ring's morph band, and not in rippled water.
    if (SCENE_NEAR && shadows == 1 && s.isle.x > 0.0 && L.sun.y < 0.04 && fp < 1.0) {
        uint ib = uint(s.isle.x);
        col += isleGlow(h.voxel, h.normal, p.xz + float2(x.origin), p.y, s.isle.y, (ib & 1u) != 0u, (ib & 2u) != 0u, (ib & 4u) != 0u, fp, x.seed, x.time)
             * smoothstep(0.04, -0.12, L.sun.y);
    }
    return col + s.emit;
}

// In a morph band the shading also slides to the next level's, reaching it where the geometry does, so the switch
// at the handoff circle is seamless in colour and light as well as in shape.
static float3 shadeL(thread const Ctx& x, thread const Light& L, float3 o, float3 d, Hit h, int shadows, bool inWater = false) {
    float3 p = o + d * h.t;
    float w = morphAt(x.tr, h.level, p.xz);
    // One shading per pixel: the next level's with probability w, varying per pixel and frame for TAA to blend.
    // Each shading traces its own level's shadow: the outer one must match the next level's hits past the circle.
    if (w > 0.0 && x.dither < w) {
        int2 c = int2(floor((p.xz - float2(h.normal.xz) * 0.01) / x.tr.lod[h.level + 1].w));
        h.level += 1;
        h.voxel = int3(c.x, h.voxel.y, c.y);
    }
    return shade(x, L, o, d, h, shadows, inWater);
}
