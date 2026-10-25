// Moonlight's strength at new moon (lit share 0), as a share of full moon's.
constant float MOON_FLOOR = 0.15;

struct Light {
    float3 key;     // direction to the current key light (sun, afterglow or moon)
    float3 keyCol;
    bool shadows;
    float shade;    // how much of its shadows land: fades in from where they switch on, with the sun and with the moon
    float3 zenith;   // terrain/cloud fill
    float3 horizon;
    float3 skyZen;   // sky radiance, kept darker and more saturated than the fill so twilight skies hold colour
    float3 skyHor;
    float3 sun;
    float3 moon;
    float sunAmt;
    float night;
    float gold;     // golden and blue hour; 0 at noon and deep night
    float moonAmt;  // moonlight's strength on land, cloud and water: floored so thin-moon nights stay readable
};

// The moon's phase and the planet live outside Light, read from the Frame where the sky needs them: every pixel
// carries Light through its shading, and their seven fields there cost the scene pass registers.
// x the sun's angle around the moon as seen from here (pi is full), y the moon's lit share (exactly 1 at full).
static inline float2 moonPhase(uint seed, float nightIndex) {
    // An 8-night cycle of phases with full moon at k = 4. It stops short of new moon (lit share 0.13), which would
    // leave no moon at all. Full is pinned to 1 so full-moon nights render exactly as before phases.
    uint k = (uint(int(nightIndex) + 8000) + pcg(seed + 5u)) & 7u;
    float th = 3.14159265 + 0.6 * (float(k) - 4.0);
    return float2(th, k == 4u ? 1.0 : 0.5 - 0.5 * cos(th));
}

struct Planet {
    float3 dir;   // direction to the evening or morning star
    float amt;
    float2 px;    // in scene pixels
    float reach;  // scene pixels from px to the core's edge
};

static inline Planet planetOf(constant Frame& f) {
    // Every scene pixel some cell of the two-cell block takes as its own (taaFragment), and no more at full scale.
    return { f.planet.xyz, smoothstep(0.02, -0.05, f.sunDir.y), f.planetPx.xy * f.planetPx.z, f.planetPx.z + 0.5 };
}

// Keyframes by sun elevation: night, blue hour, sunrise, golden, midday.
static inline float3 key5(float e, float3 a, float3 b, float3 c, float3 d, float3 f) {
    float k0 = -0.30, k1 = -0.10, k2 = 0.0, k3 = 0.18, k4 = 0.32;
    if (e < k1) return mix(a, b, smoothstep(k0, k1, e));
    if (e < k2) return mix(b, c, smoothstep(k1, k2, e));
    if (e < k3) return mix(c, d, smoothstep(k2, k3, e));
    return mix(d, f, smoothstep(k3, k4, e));
}

static inline float goldAmount(float e) { return smoothstep(0.30, 0.12, e) * smoothstep(-0.16, -0.04, e); }

// nightIndex is Frame.lights.z.
static Light makeLight(float3 sunDir, uint seed, float nightIndex) {
    Light L;
    float e = sunDir.y;
    L.sun = sunDir;
    // The moon rides lower than the anti-sun (at most ~20 degrees), so a frame that holds it still has room
    // for land below the horizon; a high moon left land a strip along the bottom.
    L.moon = normalize(float3(-sunDir.x, max(-sunDir.y * 0.7, 0.08), -sunDir.z));
    L.zenith = key5(e, float3(0.018, 0.022, 0.045), float3(0.10, 0.10, 0.22), float3(0.13, 0.16, 0.32),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    L.horizon = key5(e, float3(0.022, 0.026, 0.045), float3(0.30, 0.24, 0.36), float3(0.80, 0.45, 0.28),
                     float3(0.95, 0.60, 0.32), float3(0.46, 0.62, 0.86));
    L.skyHor = key5(e, float3(0.022, 0.026, 0.045), float3(0.16, 0.10, 0.22), float3(0.50, 0.22, 0.09),
                    float3(0.80, 0.48, 0.22), float3(0.50, 0.66, 0.90));
    L.skyZen = key5(e, float3(0.010, 0.013, 0.028), float3(0.03, 0.04, 0.12), float3(0.07, 0.09, 0.24),
                    float3(0.10, 0.20, 0.48), float3(0.06, 0.20, 0.62));
    float3 sunCol = key5(e, float3(0), float3(0), float3(1.8, 0.6, 0.22), float3(3.0, 1.45, 0.55), float3(3.0, 2.7, 2.2));
    L.sunAmt = smoothstep(-0.02, 0.05, e);
    L.night = smoothstep(-0.04, -0.20, e);
    L.gold = goldAmount(e);
    L.moonAmt = 1.0 - (1.0 - MOON_FLOOR) * (1.0 - moonPhase(seed, nightIndex).y);
    if (e > -0.02) {
        // Held at the afterglow's height until the sun clears it, and its shadows faded in: the key dropped from 0.05
        // to the sun's -0.02 at the handover, shadows switched on there and again at 0 (the maps bake only for a light
        // above the horizon), and sunlit faces darkened by a tenth in one frame each time. Must match SkyClock.keyLight (Sources/StrataCore/Sky/KeyLight.swift).
        L.key = e >= 0.05 ? sunDir : normalize(float3(sunDir.x, 0.05, sunDir.z));
        L.keyCol = sunCol * L.sunAmt + float3(1.3, 0.5, 0.3) * 0.4 * (1.0 - L.sunAmt);
        L.shadows = true;
        L.shade = smoothstep(-0.02, 0.03, e);
    } else {
        // Below the horizon an afterglow from the sun's azimuth hands over to the moon.
        float3 glowDir = normalize(float3(sunDir.x, 0.05, sunDir.z));
        float3 glowCol = float3(1.3, 0.5, 0.3) * 0.4 * smoothstep(-0.20, -0.02, e);
        float m = smoothstep(-0.09, -0.16, e);
        L.key = normalize(mix(glowDir, L.moon, m));
        L.keyCol = mix(glowCol, float3(0.60, 0.66, 0.80) * L.night * L.moonAmt, m);
        // The moon's shadows switch on halfway through the blend (with the key near the zenith) and fade in over the rest:
        // switched on at once they darkened every moonlit face by up to 10/255 in one frame. Must match SkyClock.keyLight.
        L.shadows = m > 0.5;
        L.shade = smoothstep(0.5, 1.0, m);
    }
    return L;
}
