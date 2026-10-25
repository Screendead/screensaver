// The bed's depth on the 8-voxel level: a broad ramp off the shore, which a lake's one-step shelf lacks. Surf, the
// shore fade and the creatures follow it; on the fine depth they would trace the bed's voxel stairs.
static float seaBroad(thread const Ctx& x, int k, float2 p) {
    int kc = max(k, 3);
    float b = SEA - groundSmoothL(x.tr, kc, p);
    float mc = kc < x.tr.levels - 1 ? morphAt(x.tr, kc, p) : 0.0;
    if (mc > 0.0) b = mix(b, SEA - groundSmoothL(x.tr, kc + 1, p), mc);
    return b;
}

// Bioluminescence from dusk to night: bacteria blooming in some seas, lit along the swell's crests and brightest where
// they break over the shallows; below the surface sparse jellies, plankton swirls and rare
// rays. The same bacteria flare in a fish's wake. wp: the surface point (frame-local), ww: its world xz, k: its level;
// salt: the pan's saltAmount; wake: seaLife's.
static float3 seaGlow(thread const Ctx& x, float3 d, float3 wp, float2 ww, int k, float tw, float e0, float fres, float amt, float salt, float wake) {
    float t = x.time;
    // Voxels per pixel along the foreshortened view: detail hands over to its mean before it shrinks to a pixel.
    float fpv = tw * x.pixelAngle / max(-d.y, 0.1);
    float3 c = 0;
    float big = seaNoise(ww * (1.0 / 900.0), x.seed + 144u);
    // b2 lies in [-0.5, 0.5]: under -0.5 big alone rules out a bloom.
    float b2 = big > -0.5 ? seaNoise(ww * (1.0 / 230.0), x.seed + 145u) : -0.5;
    float bloom = smoothstep(-0.275, -0.075, big + 0.45 * b2);
    float near = smoothstep(420.0, 180.0, tw) * (1.0 - fres);
    if (bloom <= 0.0 && near <= 0.0 && wake <= 0.0) return 0;
    // The land's hue for the region, but mostly cyan-blue: violet streaks on water read as dye.
    float3 hue0 = mix(bioHue(ww, x.seed), mix(BIO_LAND[0], BIO_LAND[1], 0.3 + 0.3 * smoothstep(-0.15, 0.15, b2)), 0.55);
    // Read only where something uses it: most open water near the camera holds no creature.
    float broad = bloom > 0.0 || salt > 0.0 ? seaBroad(x, k, wp.xz) : -1.0;
    // A pan's pools are a still skin a voxel deep; deeper water in a dry basin is open sea and glows in full.
    if (salt > 0.0) amt *= 1.0 - 0.7 * salt * (1.0 - smoothstep(1.2, 2.2, broad));
    if (bloom > 0.0) {
        // The fine depth, smooth across its voxel steps and through the morph band, catches the last voxel of shore.
        float fine = 3.0;
        if (broad < 1.2) {
            fine = SEA - groundSmoothL(x.tr, k, wp.xz);
            float mw = k < x.tr.levels - 1 ? morphAt(x.tr, k, wp.xz) : 0.0;
            if (mw > 0.0) fine = mix(fine, SEA - groundSmoothL(x.tr, k + 1, wp.xz), mw);
        }
        // Surf over the shallows; the glow dies out before the waterline.
        float sh = smoothstep(0.7, 1.5, broad) * (1.0 - smoothstep(1.6, 2.7, broad));
        float shore = smoothstep(0.35, 1.3, broad) * smoothstep(0.0, 0.9, fine);
        // Patches tens of voxels across, so no crest or stretch of surf is lit end to end.
        float patch = smoothstep(-0.4, 0.1, seaNoise(ww * 0.035 + float2(t * 0.01, 0), x.seed + 81u));
        // Far out the flecks' mean; the surf no brighter than open water, or it would trace the shore.
        float crest = 0.04;
        float surf = sh * (1.0 - saturate(fpv * 0.8));
        if (fpv < 1.0 / 0.6) {
            // Two swells from directions fixed per seed, one or the other ruling each region with a calm seam between.
            // Each swell's crests are narrow ridges bent by the warp; a second train at an angle groups them, so only
            // stretches of uneven length and spacing are lit.
            float a0 = hashf(pcg(x.seed + 73u)) * 6.2831853;
            float2 wd1 = float2(cos(a0), sin(a0)), wd2 = float2(wd1.x * 0.765 - wd1.y * 0.644, wd1.x * 0.644 + wd1.y * 0.765);
            float2 wd1b = float2(wd1.x * 0.765 + wd1.y * 0.644, -wd1.x * 0.644 + wd1.y * 0.765);
            float2 wd2b = float2(wd2.x * 0.814 + wd2.y * 0.581, -wd2.x * 0.581 + wd2.y * 0.814);
            float warp = 6.0 * seaNoise(ww * 0.012, x.seed + 72u) + 3.0 * seaNoise(ww * 0.05, x.seed + 79u);
            float sel = smoothstep(0.19, 0.21, big);
            // Flecks are near round, so they shrink to a pixel on a footprint between the foreshortened and the square one.
            float fa = tw * x.pixelAngle * rsqrt(max(-d.y, 0.1));
            // More of each wave breaks in the surf; close up a crest is a narrow line of foam.
            float lo = 0.7 - 0.3 * sh + 0.12 * (1.0 - saturate(fa * 3.0));
            float cr = 0;
            if (sel < 1.0) cr += (1.0 - sel) * smoothstep(lo, lo + 0.25, sin(dot(ww, wd1) * 0.33 + warp - t * 0.9))
                                            * smoothstep(-0.3, 0.6, sin(dot(ww, wd1b) * 0.11 + 0.6 * warp - t * 0.2));
            if (sel > 0.0) cr += sel * smoothstep(lo, lo + 0.25, sin(dot(ww, wd2) * 0.27 - warp - t * 0.75))
                                     * smoothstep(-0.3, 0.6, sin(dot(ww, wd2b) * 0.09 - 0.6 * warp - t * 0.17));
            // Foam flecks a few voxels long along the crest, with gaps, each of its own brightness.
            float2 wd = sel < 0.5 ? wd1 : wd2;
            float2 fq = float2(dot(ww, wd) * 0.35, (ww.x * wd.y - ww.y * wd.x) * 0.26);
            // Runs of a crest a dozen voxels long, the rest of it calm.
            if (cr > 0.0) cr *= smoothstep(-0.25, 0.1, seaNoise(fq * float2(0.3, 0.3) + float2(t * 0.05, 7.3), x.seed + 83u));
            // The surf breaks up again every few voxels whatever way the shore runs, so it never draws its outline.
            if (cr > 0.0 && sh > 0.0) cr *= mix(1.0, smoothstep(-0.1, 0.2, seaNoise(ww * 0.16 + float2(0, t * 0.1), x.seed + 85u)), sh);
            float fleck = 0.0;
            if (cr > 0.0) {
                float fl = seaNoise(fq + float2(-t * 0.6, t * 0.25), x.seed + 82u);
                fleck = smoothstep(0.05 - 0.15 * sh, 0.28, fl) * (0.3 + 1.1 * smoothstep(0.05, 0.4, e0));
                // Close up the foam is a grain of sparks rather than smooth blobs.
                if (fa < 0.4) fleck *= mix(0.45 + 1.1 * smoothstep(0.0, 0.45, seaNoise(ww * 2.3 + float2(t * 0.3, -t * 0.2), x.seed + 84u)), 1.0, fa * 2.5);
                fleck = mix(fleck, 0.25, saturate(fa * 0.8));
            }
            crest = mix(cr * fleck, crest, saturate(fpv * 0.45));
        }
        float glow = crest * (0.3 + 0.9 * surf) * shore * patch * (1.0 + 2.0 * wake);
        float3 hue = mix(hue0, mix(BIO_LAND[0], BIO_LAND[1], 0.6) * 1.3, saturate(glow * 2.0 - 0.6));
        c += hue * bloom * glow * 1.4;
    }
    if (wake > 0.002) {
        // Sparks of the crest foam's kind, finer and sparser, thicker in a bloom; bright cores whiten to turquoise as a
        // crest's do.
        float fa = tw * x.pixelAngle * rsqrt(max(-d.y, 0.1));
        float gr = seaNoise(ww * 3.6 + float2(t * 0.3, -t * 0.2), x.seed + 84u);
        float wg = wake * mix(0.15 + 2.6 * smoothstep(0.08, 0.38, gr), 1.0, saturate(fa * 4.0)) * (0.7 + 0.6 * bloom);
        c += mix(hue0, mix(BIO_LAND[0], BIO_LAND[1], 0.6) * 1.3, saturate(wg * 1.5 - 0.3)) * 0.42 * wg;
    }
    if (near > 0.0) {
        float3 g = 0;
        float3 rd = refract(d, float3(0, 1, 0), 0.75);
        float2 slant = rd.xz / max(-rd.y, 0.2);
        // Jellies: one in some 16-voxel cells, 0.8 to 3 under the surface, drifting about the cell's middle; the
        // deeper, the dimmer, bluer and more blurred.
        {
            float2 q = ww + slant * 1.9;
            int2 cell = int2(floor(q / 16.0));
            uint h = hash2i(cell, x.seed + 75u);
            if (hashf(h) < 0.07) {
                float ph = hashf(pcg(h)) * 6.2831853;
                float dj = 0.8 + 2.2 * hashf(pcg(h + 2u));
                float2 cc = (float2(cell) + 0.5) * 16.0 + 4.5 * float2(sin(t * 0.05 + ph), cos(t * 0.037 + ph * 1.7)) - slant * (dj - 1.9);
                float fj = 0.0;
                if (length_squared(q - cc) < 64.0) {
                    if (broad < -0.5) broad = seaBroad(x, k, wp.xz);
                    fj = smoothstep(1.5, 2.3, broad);
                }
                if (fj > 0.0) {
                    float pulse = 0.5 + 0.5 * sin(t * 0.9 + ph * 3.0);
                    float r = (0.9 + 0.7 * hashf(pcg(h + 1u))) * (1.0 + 0.1 * pulse);
                    float blur = 0.2 + 0.22 * dj + 0.5 * fpv;
                    float re = sqrt(r * r + blur * blur);
                    float2 v = (q - cc) / re;
                    float rho2 = length_squared(v);
                    // A dome lit from within, brightest at its crown, its margin a soft shoulder rather than a ring;
                    // faint tendrils trail behind its drift.
                    float2 back = -normalize(float2(cos(t * 0.05 + ph), -sin(t * 0.037 + ph * 1.7) * 0.74) + 1e-4);
                    float along = dot(v, back), side = v.x * back.y - v.y * back.x;
                    float tent = smoothstep(0.6, 1.1, along) * smoothstep(4.5, 1.0, along)
                               * smoothstep(0.1, 0.9, cos(side * 8.0 + sin(along * 1.7 - t * 1.2) * 1.4)) * smoothstep(0.75, 0.3, abs(side));
                    float body = exp(-rho2 * 2.2) * (0.7 + 0.3 * pulse) + 0.2 * exp(-rho2 * 0.6);
                    float3 jc = (h & 192u) == 192u ? mix(BIO_LAND[2], BIO_LAND[0], 0.35) : mix(BIO_LAND[1], BIO_LAND[0], 0.3);
                    jc = mix(jc, BIO_LAND[0], saturate(dj * 0.25));
                    float lum = (r * r) / (re * re) * exp(-dj * 0.3);
                    g += jc * (body + 0.22 * tent) * lum * 0.3 * smoothstep(260.0, 140.0, tw) * fj;
                }
            }
        }
        // Plankton: a swirl of sparks in some 48-voxel cells, just under the surface.
        {
            float2 q = ww + slant * 0.6;
            int2 cell = int2(floor(q / 48.0));
            uint h = hash2i(cell, x.seed + 76u);
            if (hashf(h) < 0.12) {
                float ph = hashf(pcg(h)) * 6.2831853;
                float2 cc = (float2(cell) + 0.5) * 48.0 + 11.0 * float2(sin(t * 0.03 + ph), cos(t * 0.023 + ph * 1.3));
                float R = 5.0 + 3.0 * hashf(pcg(h + 1u));
                float2 v = q - cc;
                float rr = length(v) / R;
                float fp = 0.0;
                if (rr < 1.6) {
                    if (broad < -0.5) broad = seaBroad(x, k, wp.xz);
                    fp = smoothstep(0.8, 1.5, broad);
                }
                if (fp > 0.0) {
                    float a = atan2(v.y, v.x);
                    float arms = smoothstep(-0.2, 1.0, sin(3.0 * a - 5.0 * rr + t * 0.5 + ph));
                    float sa = t * 0.15 / (0.4 + rr);
                    float2 vr = float2(v.x * cos(sa) - v.y * sin(sa), v.x * sin(sa) + v.y * cos(sa));
                    float dots = mix(smoothstep(0.22, 0.42, seaNoise(vr * 3.5, x.seed + 77u)), 0.1, saturate(fpv * 2.5));
                    float body = exp(-rr * rr * 2.0);
                    float3 pc = mix(BIO_LAND[1], BIO_LAND[0], 0.4 + 0.6 * saturate(rr));
                    g += pc * body * (0.05 + (0.3 + 0.7 * arms) * dots) * 0.11 * smoothstep(320.0, 160.0, tw) * fp;
                }
            }
        }
        // Rays: rare, gliding slow circles 2 under the surface.
        {
            float2 q = ww + slant * 2.0;
            int2 cell = int2(floor(q / 140.0));
            uint h = hash2i(cell, x.seed + 78u);
            if (hashf(h) < 0.22) {
                float ph = hashf(pcg(h)) * 6.2831853;
                float dir = (h & 128u) ? 1.0 : -1.0;
                float an = t * 0.012 * dir + ph;
                float2 cc = (float2(cell) + 0.5) * 140.0 + 38.0 * float2(cos(an), sin(an));
                float2 v = q - cc;
                float fr = 0.0;
                if (dot(v, v) < 196.0) {
                    if (broad < -0.5) broad = seaBroad(x, k, wp.xz);
                    fr = smoothstep(2.1, 2.7, broad);
                }
                if (fr > 0.0) {
                    float2 fw = dir * float2(-sin(an), cos(an));
                    float u = dot(v, fw), s = v.x * fw.y - v.y * fw.x;
                    float W = 7.5 * (1.0 + 0.1 * sin(t * 0.7 + ph)), L = 4.5;
                    // Wingtips swept back from a rounded head, the trailing edge cut in between tip and body; dv: the distance
                    // outside the outline, in voxels (inside negative). The ray itself is dark: the plankton it stirs light
                    // its margin in sparks, most at the wingtips, with a faint wash over its back; filled, it reads as a
                    // lit kite. The edge is blurred by the water above.
                    float a = min(abs(s) / W, 1.0);
                    float uf = L * (1.0 - a * a) - 2.3 * a, ub = -2.4 + 1.1 * sin(3.14159 * a);
                    float dv = max((abs(s) - W), max(u - uf, ub - u));
                    float blur = 1.2 + 0.6 * fpv;
                    float inner = smoothstep(blur, -blur - 1.5, dv);
                    float spine = exp(-s * s / (W * W * 0.12));
                    float edge = exp(-2.0 * dv * dv / (blur * blur));
                    float spk = mix(smoothstep(-0.05, 0.3, seaNoise(q * 1.8 + float2(t * 0.3, -t * 0.2), x.seed + 86u)), 0.35, saturate(fpv * 1.2));
                    float tail = smoothstep(1.2, 0.0, abs(s)) * smoothstep(-0.5, -2.4, u) * smoothstep(-10.0, -2.4, u);
                    float flap = 0.85 + 0.15 * sin(t * 0.7 + ph - abs(s) * 0.3);
                    float3 rc = mix(BIO_LAND[1], BIO_LAND[0], 0.35 + 0.4 * a)
                              * (edge * spk * (0.3 + 0.7 * a) * flap + 0.08 * inner * spine + 0.06 * tail * spk);
                    g += rc * 0.16 * smoothstep(420.0, 220.0, tw) * fr;
                }
            }
        }
        c += g * near;
    }
    return c * amt;
}
