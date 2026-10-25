// Fish: in some FISH_CELL cells a school (4 to 7) or a single larger fish, wandering about the cell's middle and now
// and then darting. Everything is a smooth function of world position and time, and a school with its last second of
// track stays inside its cell, so a pixel asks its own cell alone.
constant float FISH_CELL = 40.0;
constant float FISH_DEPTH = 1.2;

struct FishSample { float wake; float body; float shade; float glint; };

static inline float2 fishWander(float t, uint h) {
    uint g = pcg(h + 1u);
    float4 ph = float4(byteOf(g, 0), byteOf(g, 1), byteOf(g, 2), byteOf(g, 3)) * 6.2831853;
    return 5.5 * float2(sin(0.31 * t + ph.x) + 0.45 * sin(0.83 * t + ph.y), sin(0.27 * t + ph.z) + 0.45 * sin(0.71 * t + ph.w));
}

// A dart: out 4.5 voxels in a twelfth of the period, drifting back over the rest; zero with zero slope at its ends.
static inline float2 fishDart(float t, uint h, float period) {
    float u = t / period + hashf(pcg(h + 5u));
    float k = floor(u), f = u - k;
    uint hk = hash2i(int2(int(k), 7), h);
    if (hashf(hk) > 0.65) return 0;
    float a = hashf(pcg(hk)) * 6.2831853;
    return float2(cos(a), sin(a)) * 4.5 * (smoothstep(0.0, 0.08, f) - smoothstep(0.3, 1.0, f));
}

// A fin flick: a quarter-second pulse every ten to sixteen seconds, from half the fish of a school (hash hi).
constant float FLICK_W = 0.025;
static inline float fishFlickRate(uint hi) { return 0.0625 + 0.0375 * byteOf(hi, 3); }
static inline float fishFlick(float t, uint hi) {
    float ph = fract(t * fishFlickRate(hi) + byteOf(hi, 2));
    float b = saturate(1.0 - abs(ph - 0.5) / FLICK_W);
    return b * b * (3.0 - 2.0 * b);
}

// A school's formation: points in the unit disc, turned and scaled per school.
constant float2 FISH_SPOTS[8] = { float2(0.0, 0.0), float2(0.62, 0.21), float2(-0.48, 0.55), float2(-0.2, -0.71),
                                  float2(0.37, -0.52), float2(0.93, -0.18), float2(-0.85, -0.1), float2(0.15, 0.92) };

// q: world xz at the fishes' depth along the view; bs: the bed point's xz moved toward the sun to where a fish shadows
// it (unused when shadows is false); fpv: voxels per pixel there. wake (at night): the glowing track fast movement
// leaves in the water (fin flicks, darts), about [0, 1]; else body: a fish's silhouette; shade: its shadow on the bed;
// glint: a flank catching the light as it flicks. Kept lean in registers: inlined in the scene pass, a heavier body
// slowed every pixel.
static inline FishSample fishAt(thread const Ctx& x, float2 q, float2 bs, bool shadows, float fpv, bool wake) {
    FishSample r = { 0, 0, 0, 0 };
    int2 cell = int2(floor(q / FISH_CELL));
    uint h = hash2i(cell, x.seed + 160u);
    if (hashf(h) > 0.18) return r;
    float2 C = (float2(cell) + 0.5) * FISH_CELL;
    bool single = (h & 7u) < 3u;
    float t = x.time, per = single ? 3.4 : 5.5;
    float spread = single ? 0.0 : 3.2;
    int n = single ? 1 : 4 + int((h >> 4) % 4u);
    float L = single ? 1.3 : 0.55;
    uint hs = pcg(h + 9u);
    float sa = byteOf(hs, 0) * 6.2831853;
    float2 rot = float2(cos(sa), sin(sa)) * spread;
    if (wake) {
        // Calm water stays dark: unless a dart's outward rush or a fin flick fell within the last second, nothing is
        // lit, and the track need not be worked out.
        float u = t / per + hashf(pcg(h + 5u));
        float k = floor(u);
        bool lit = (u - k) * per < 0.08 * per + 0.95 && hashf(hash2i(int2(int(k), 7), h)) <= 0.65;
        for (int i = 0; i < n && !lit; i++) {
            uint hi = pcg(h + uint(i) * 31u + 11u);
            float rate = fishFlickRate(hi);
            float ph = fract(t * rate + byteOf(hi, 2));
            lit = (single || (hi & 0x100000u) != 0u) && ph > 0.5 - FLICK_W && ph < 0.5 + FLICK_W + rate;
        }
        if (!lit) return r;
    }
    float2 c0 = C + fishWander(t, h) + fishDart(t, h, per);
    if (wake) {
        // A dart runs out and back along one line, so the last second keeps within the school's spread, the glow's
        // reach and a little bow of the chord from then to now.
        float2 d5 = fishDart(t - 0.95, h, per);
        float2 c5 = C + fishWander(t - 0.95, h) + d5;
        float2 a0 = q - c0, s0 = c5 - c0;
        // Each fish's track is the school's turned and stretched about where it is now, by up to 0.62 of its reach.
        if (length(a0 - s0 * saturate(dot(a0, s0) / max(dot(s0, s0), 1e-4))) > 1.25 * spread + 3.0 + 2.5 * fpv + 0.65 * (length(s0) + 2.0)) return r;
        const float tau[6] = { 0.0, 0.12, 0.25, 0.42, 0.65, 0.95 };
        float2 ca = c0, da = fishDart(t, h, per);
        for (int j = 0; j < 5; j++) {
            float2 db = j == 4 ? d5 : fishDart(t - tau[j + 1], h, per);
            float2 cb = j == 4 ? c5 : C + fishWander(t - tau[j + 1], h) + db;
            float tm = 0.5 * (tau[j] + tau[j + 1]);
            float2 sg = cb - ca;
            // A dart's outward rush lights the water, never the wander or the drift back: then the window above
            // holds every lit step.
            float Id = saturate((length(db - da) / (tau[j + 1] - tau[j]) - 2.5) / 10.0);
            da = db;
            float E = exp(-tm / 0.3);
            float wv = (single ? 0.6 : 0.4) + 0.6 * tm;
            float2 A = ca - c0;
            for (int i = 0; i < n; i++) {
                uint hi = pcg(h + uint(i) * 31u + 11u);
                float I = Id + ((single || (hi & 0x100000u) != 0u) ? 0.6 * fishFlick(t - tm, hi) : 0.0);
                if (I * E < 0.01) continue;
                // Fish of a school bolt apart: each its own heading (about 25 degrees either way), reach, width and brightness.
                uint hv = single ? 0x80808080u : reefMix(hi);
                float2 m = (0.75 + 0.5 * byteOf(hv, 0)) * float2(1.0, 0.9 * byteOf(hv, 1) - 0.45);
                float wi = wv * (0.8 + 0.45 * byteOf(hv, 2));
                float w2 = wi * wi + 0.25 * fpv * fpv;
                float2 sp = FISH_SPOTS[i] * (0.75 + 0.5 * byteOf(hi, 0));
                float2 a = q - c0 - float2(sp.x * rot.x - sp.y * rot.y, sp.x * rot.y + sp.y * rot.x) - float2(m.x * A.x - m.y * A.y, m.x * A.y + m.y * A.x);
                float2 sgi = float2(m.x * sg.x - m.y * sg.y, m.x * sg.y + m.y * sg.x);
                float2 e = a - sgi * saturate(dot(a, sgi) / max(dot(sgi, sgi), 1e-4));
                r.wake += I * (single ? 1.8 : 0.8 * (0.6 + 0.7 * byteOf(hv, 3))) * E * wi * wi / w2 * exp(-dot(e, e) / w2);
            }
            ca = cb;
        }
    } else {
        float dB = length(q - c0);
        if (shadows) dB = min(dB, length(bs - c0));
        if (dB > 1.25 * spread + 2.0) return r;
        // The school swims as one, heading along its track.
        float2 hd = normalize(c0 - (C + fishWander(t - 0.12, h) + fishDart(t - 0.12, h, per)) + float2(1e-4, 0));
        for (int i = 0; i < n; i++) {
            uint hi = pcg(h + uint(i) * 31u + 11u);
            float wf = 0.8 + 0.5 * byteOf(hi, 3), pf = 6.2831853 * byteOf(hi, 2);
            float2 sp = FISH_SPOTS[i] * (0.75 + 0.5 * byteOf(hi, 0));
            float2 P0 = c0 + float2(sp.x * rot.x - sp.y * rot.y, sp.x * rot.y + sp.y * rot.x);
            float2 a = q - P0;
            float u = dot(a, hd), sv = a.x * hd.y - a.y * hd.x;
            sv -= 0.25 * L * sin(t * 9.0 * wf + pf) * saturate(-u / L);
            float m = (u * u) / (L * L) + (sv * sv) / (0.1 * L * L);
            float body = smoothstep(1.0, 0.45, m);
            r.body = max(r.body, body);
            r.glint = max(r.glint, body * fishFlick(t, hi));
            if (shadows) {
                float2 b = bs - P0;
                float ub = dot(b, hd), vb = b.x * hd.y - b.y * hd.x;
                r.shade = max(r.shade, smoothstep(1.6, 0.4, (ub * ub) / (L * L) + (vb * vb) / (0.14 * L * L)));
            }
        }
    }
    if (r.wake + r.body + r.shade <= 0.0) return r;
    // Fish keep off the shallows, fading on the bed's broad depth under the school.
    float fade = smoothstep(1.5, 2.6, SEA - groundSmoothL(x.tr, min(3, x.tr.levels - 1), c0 - float2(x.origin)));
    r.wake *= fade; r.body *= fade; r.shade *= fade; r.glint *= fade;
    return r;
}
