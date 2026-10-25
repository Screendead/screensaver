// How far a leaf takes the dusk lift (gradeAlbedo): from the golden hour to night.
static inline float leafDusk(thread const Light& L) { return L.gold * (1.0 - L.night) * mix(0.5, 1.0, L.sunAmt); }

// An albedo graded for the light, and in albK the one the key light sees. deep: how far into the night (darker than
// moonlight's first hour); canopy: leaves, autumn their fall share; dusk: the lift a leaf takes at dusk (leafDusk).
static inline float3 gradeAlbedo(thread const Light& L, float3 alb, float deep, bool canopy, float autumn, float dusk,
                                 thread float3& albK) {
    // Moonlight drains colour.
    alb = mix(alb, float3(dot(alb, LUMA)), 0.4 * L.night + 0.1 * deep);
    // Canopies would otherwise drop to cut-out silhouettes under the moon: lifted toward the luminance of a lifted
    // green crown, never past it, or fall foliage, already several times brighter, glows like candy.
    if (canopy) {
        float l = dot(alb, LUMA);
        alb = mix(alb, float3(l), 0.4 * L.night * autumn);
        alb *= 1.0 + L.night * (clamp(0.17 / max(l, 1e-3), 1.0, 2.4) - 1.0);
    }
    // Pale sand, salt and snow would glow under the moon brighter than the sky above them. A knee on albedo
    // above mid-tone, not a cut keyed to the material, so tones keep their order: a stone riser never outshines
    // its sand tread.
    float kn = 4.5 * L.night;
    alb *= (1.0 + 0.1 * kn) / (1.0 + kn * dot(alb, LUMA));
    // Low light: leaves this dark under a warm key and a violet dome go to brown mud. Light passed and scattered through
    // a crown takes the leaf's own hue, so the lift saturates as it brightens (autumn crowns are bright already).
    // A sun on the horizon, dim and amber, lights leaves unlifted and warmed toward bronze: lifted toward green first, it
    // turns them mustard. A golden-hour sun, and the afterglow, a broad patch of sky, light them lifted like the dome.
    albK = alb;
    if (dusk > 0.0) {
        float la = max(dot(alb, LUMA), 1e-4);
        alb *= mix(float3(1.0), clamp(1.5 * sqrt(alb / la), 0.7, 2.0), dusk * (1.0 - 0.6 * autumn));
        albK = mix(albK, la * float3(1.5, 0.95, 0.55), 0.5 * dusk * (1.0 - 0.6 * autumn)) * (1.0 + 0.3 * dusk);
        albK = mix(alb, albK, smoothstep(-0.04, 0.0, L.sun.y) * (1.0 - L.sunAmt));
    }
    return alb;
}
