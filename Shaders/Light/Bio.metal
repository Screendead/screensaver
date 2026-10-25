// The glowing bugs by sun height: faint from just before sunset, full from mid twilight (Renderer.bioSun must match the
// start).
constant float BIO_SUN = 0.07;
static inline float bioRamp(float sunY) { return smoothstep(BIO_SUN, -0.12, sunY); }
// The specks on the land's surfaces come in later than the bugs: against a still bright sky they look stuck on.
static inline float bioLandRamp(float sunY) { return smoothstep(0.02, -0.12, sunY); }

// The glow on land and vines is 40% lower within ~60 voxels of the camera, where it filled the frame, and untouched
// past ~220, so the far glow keeps its level.
static inline float bioNear(float dist) { return mix(0.6, 1.0, smoothstep(60.0, 220.0, dist)); }

// Means of the near patterns below, per unit of glow; far off each surface takes its mean, so the land keeps one level
// of light from the camera's feet to the haze.
constant float BIO_FLOOR = 0.052, BIO_FLOOR_HALO = 0.68, BIO_CROWN = 0.038;
// Distant glow: x(1 + gain) reached at FAR1 voxels from the camera, none within FAR0.
constant float BIO_FAR0 = 48.0, BIO_FAR1 = 1400.0, BIO_FAR_GAIN = 4.5;

// The light the vegetation gives off at night: moss and undergrowth carpets on the ground, pooled round trunk feet,
// ragged moss hanging from terrace lips, fungus collars hugging trunk bases, crowns lit from within (undersides and
// lower leaves most), and the soft light all of it casts on what is around it. Everything is fixed by place; only the
// pulse moves.
static float3 bioLight(thread const Ctx& x, thread const Surface& s, int k, int3 n, float3 p, float3 alb, float fp, float sv) {
    float3 wp = p + float3(float(x.origin.x), 0.0, float(x.origin.y));
    float D = s.bio, g = s.bioG;
    // A giant's own surfaces: the provinces' groves.
    if (D < 0.0) {
        Climate c = climateL(x.tr, k, p.xz);
        g = bioNoise(wp.xz / 170.0, x.seed + 710u) + 0.5 * bioNoise(wp.xz / 47.0, x.seed + 711u);
        D = 0.6 * smoothstep(-0.35, 0.05, g) * (1.0 - 0.5 * smoothstep(0.65, 0.9, c.jungle));
    }
    float jw = s.bioJ;
    if (D <= 0.0) return 0;
    float3 hue = bioHue(wp.xz, x.seed);
    float pulse = bioPulse(wp.xz, x.time, x.seed, g);
    // Pattern to mean as the footprint grows, before any of it nears a pixel, and across the 1-voxel map's morph band:
    // the coarser maps hold no trunks to pool round, so there the pattern is all mean.
    float band = k == 0 ? morphAt(x.tr, 0, p.xz) : 1.0;
    float pf = x.prof & 8192 || k > 0 ? 1.0 : max(smoothstep(0.1, 0.5, fp), band);
    float e = 0.0, halo = 0.5;
    if (s.mat == 0 && n.y > 0) {
        // Clumps one to a few voxels across, bright at heart and fading out ragged, grained with small plants close up,
        // in a pool of their own light; round trunk feet they close into a pool.
        float2 q = wp.xz;
        float n1 = bioNoise(q / 3.3, x.seed + 712u);
        float b = n1 + (fp < 0.2 ? 0.55 * (1.0 - smoothstep(0.04, 0.2, fp)) * bioNoise(q / 1.1, x.seed + 713u) : 0.0);
        float pool = 0.0;
        if (s.bioNb != 0u) {
            float2 f = fract(p.xz);
            float4 dd = float4(f.x, 1.0 - f.x, f.y, 1.0 - f.y);
            for (int i = 0; i < 4; i++) if ((s.bioNb >> uint(i)) & 1u) pool = max(pool, exp(-2.2 * dd[i]));
        }
        float th = 0.08 - 0.4 * pool, core = smoothstep(th, th + 0.3, b);
        float gr = fp < 0.04 ? mix(0.5 + smoothstep(-0.15, 0.2, bioNoise(q / 0.4, x.seed + 719u)), 1.0, smoothstep(0.01, 0.04, fp)) : 1.0;
        e = mix(core * core * gr * 0.4 + 0.05 * pool, BIO_FLOOR, pf);
        halo = mix(smoothstep(th - 0.45, th + 0.05, n1), BIO_FLOOR_HALO, pf);
    } else if (s.mat == 0) {
        // Moss over a terrace lip: hanging clumps, gone round steps too low to hold them.
        float along = n.x != 0 ? wp.z : wp.x;
        float dd = s.H - p.y;
        float len = (0.25 + 0.9 * saturate(bioNoise(float2(along / 1.7, s.H), x.seed + 714u) + 0.35))
                  * smoothstep(-0.15, 0.1, bioNoise(float2(along / 7.0, s.H * 0.37), x.seed + 715u));
        float lip = (1.0 - smoothstep(0.5 * len, len + 0.05, dd)) * smoothstep(1.5, 2.5, s.riser);
        e = mix(lip * 0.16, 0.012, pf);
    } else if (s.mat == 1) {
        // Lit from within: undersides and the crown's lower leaves, the tops faintly; veins of light close up.
        // Sides fade up each voxel as well as up the crown, so a crown is not lit as a stack of boxes.
        float low = (1.0 - s.up) * (1.0 - s.up) * (n.y == 0 ? 1.0 - 0.7 * fract(p.y) : 1.0);
        float lo = n.y < 0 ? 0.07 : n.y > 0 ? 0.014 : 0.008 + 0.035 * low;
        halo = 0.25 * low;
        float vf = 1.0 - smoothstep(0.02, 0.08, fp);
        if (vf > 0.0 && n.y <= 0) {
            float3 vq = wp * 1.4;
            float vn = bioNoise(n.y != 0 ? vq.xz : n.x != 0 ? vq.zy : vq.xy, x.seed + 716u);
            lo *= mix(1.0, 0.45 + 2.2 * (1.0 - smoothstep(0.02, 0.07, abs(vn))), vf);
        }
        // Glowing sprays among the leaves, a few voxels across with soft hearts, which also leave the undersides and low
        // sides dark between them (mean 1); both at their means before a spray nears a few pixels, or they read as flecks.
        float ps = max(pf, smoothstep(0.05, 0.25, fp)), bl = 0.0;
        if (ps < 1.0) {
            float2 q2 = n.y != 0 ? wp.xz : n.x != 0 ? wp.zy : wp.xy;
            bl = smoothstep(0.05, 0.35, bioNoise(q2 / 2.6, x.seed + 728u));
            if (n.y <= 0) lo *= mix(0.5 + 2.4 * bl, 1.0, ps);
        }
        e = lo + 0.15 * mix(bl * bl, 0.159, ps);
        // Far crowns stand for the floor between them as well as themselves.
        if (n.y > 0) e = mix(e, BIO_CROWN, max(band, x.prof & 8192 ? 1.0 : 0.0));
    } else if (n.y == 0) {
        // Fungus hugging a trunk's foot: a ragged collar of caps, brightest at the ground.
        float hb = p.y - s.bioY;
        float along = n.x != 0 ? wp.z : wp.x;
        float tall = s.bioY > 0.0 && s.bio < 0.0 ? 4.0 : 1.0;
        float ht = tall * (0.5 + 1.1 * saturate(bioNoise(float2(along * 1.6, wp.x + wp.z), x.seed + 717u) + 0.4));
        float cap = 0.55 + 0.9 * smoothstep(-0.05, 0.25, bioNoise(float2(along, wp.y) * 2.4, x.seed + 718u));
        e = (1.0 - smoothstep(0.55 * ht, ht, hb)) * mix(cap, 1.0, pf) * 0.22 * (1.0 - smoothstep(0.08, 0.35, fp / tall));
    }
    // The light it casts: a tinted lift on everything in the grove, more in hollows and under crowns.
    // Glades of it a dozen voxels across and dark ground between, out to where they near a few pixels (mean 1).
    float gf = x.prof & 8192 ? 1.0 : smoothstep(2.0, 6.0, fp), glade = 1.0;
    if (gf < 1.0) {
        glade = 0.45 + 1.72 * smoothstep(-0.05, 0.3, bioNoise(wp.xz / 14.0, x.seed + 726u) + 0.5 * bioNoise(wp.xz / 5.5, x.seed + 727u));
        // Deep in the jungle the light gathers into sparse bright clusters with dim ground between (mean 1).
        if (jw > 0.0) glade *= mix(1.0, 0.3 + 5.2 * smoothstep(0.15, 0.4, bioNoise(wp.xz / 9.0, x.seed + 730u) + 0.4 * bioNoise(wp.xz / 3.5, x.seed + 731u)), jw);
        glade = mix(glade, 1.0, gf);
    }
    // Far off, a grove's many small lights merge into one glow brighter than their literal mean, rising with distance
    // alone (so alike on either side of every map boundary and the pattern-to-mean fade); the swells calm toward
    // their mean as they shrink.
    float far = smoothstep(BIO_FAR0, BIO_FAR1, fp / x.pixelAngle);
    pulse = mix(pulse, 0.58, 0.5 * far);
    // Violet pockets gain half as much: far off they would crowd out the regional teal and blue.
    float gain = 1.0 + BIO_FAR_GAIN * far * (1.0 - 0.5 * saturate(3.0 * (hue.r - hue.g)));
    return hue * D * pulse * glade * gain * bioNear(fp / x.pixelAngle) * (e + alb * halo * 0.2 * (1.3 - 0.5 * sv));
}
