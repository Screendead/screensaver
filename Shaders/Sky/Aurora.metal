// Aurora on about one night in four: one pleated curtain 9-24k up toward a poleward bearing fixed per seed, building
// after the stars are out and gone before dawn. The sheet is a curve in polar form around the camera (range from
// azimuth), so flight and map recentring never move it and each ray crosses it once; it brightens where it turns
// edge-on (the path through it lengthens as 1/sin of the angle to the sheet), which draws the folds. The path saturates
// smoothly at the sheet's thickness; a hard cap draws flat-topped bars. It spans about 55 degrees of azimuth, full over
// the middle 25, and the fold and ray scales stay several cells wide so a turning camera doesn't make them shimmer in
// the dither. Its brightest folds stay under the Milky Way's brightest parts.
static float3 aurora(thread const Ctx& x, float3 d) {
    uint n = uint(max(int(x.lights.z), 0));
    if (x.L.night <= 0.0 || d.y < 0.02 || hashf(pcg(n * 7919u + x.seed + 900u)) >= 0.25) return 0;
    float env = x.L.night * smoothstep(0.05, 0.2, x.lights.y) * smoothstep(0.97, 0.86, x.lights.y);
    float a = hashf(pcg(x.seed + 901u)) * 6.2831853;
    float2 P = float2(sin(a), cos(a)), Q = float2(-P.y, P.x);
    float dp = dot(d.xz, P);
    if (env <= 0.0 || dp <= 0.0) return 0;
    float k = dot(d.xz, Q) / dp;
    float span = smoothstep(0.55, 0.22, abs(k));
    if (span <= 0.0) return 0;
    const float D = 70000.0, A = 14000.0, e = 0.01;
    float t = x.time, fn = float(n) * 3.7;
    float2 m0 = float2(k, k + e) * (D / 9000.0) + float2(t * 0.002 + fn);
    float2 m1 = float2(k, k + e) * (D / 4500.0) - float2(t * 0.005 - fn);
    float2 m = A * float2(perlin2(float2(m0.x, 0.5), x.seed + 902u), perlin2(float2(m0.y, 0.5), x.seed + 902u));
    if (x.quality > 0.5)
        m += 0.35 * A * float2(perlin2(float2(m1.x, 0.5), x.seed + 903u), perlin2(float2(m1.y, 0.5), x.seed + 903u));
    float u = D + m.x, du = (m.y - m.x) / e;
    float g = du * du * (1.0 + k * k) / (u * u);
    float fold = sqrt(1.0 + g / (1.0 + g / 35.0));
    float y = d.y / length(d.xz) * u * sqrt(1.0 + k * k);
    float hem = 8500.0 + 0.3 * m.x;
    float z = (y - hem) / 15000.0;
    float prof = smoothstep(0.0, 0.05, z) * exp(-4.0 * max(z, 0.0));
    float rays = 0.6 + 0.4 * saturate(perlin2(float2(k * (D / 2400.0), t * 0.03), x.seed + 904u) + 0.5);
    float3 col = mix(float3(0.2, 1.0, 0.45), float3(0.7, 0.25, 0.9), smoothstep(0.1, 0.7, z));
    return col * 0.052 * fold * prof * rays * span * env * smoothstep(0.02, 0.06, d.y);
}

// Shooting stars on a dark night: one in each 30 s slot of the clock, everything about it hashed from the slot, so
// nothing is kept between frames. The clock is wall time and dark sky lasts about 2 minutes of each 12-minute day, so
// that is 3 to 5 a night, never two within 10 s; tools/meteors.py mirrors the schedule. It is placed on the screen's
// 864-cell dither grid and starts in the upper part of the frame, where the sky is. Its width is at least a cell and
// at least a scene sample, so the battery path's coarser scene still draws it unbroken.
static float3 meteor(constant Frame& f, float night, float3 d, float time, uint seed) {
    if (night < 0.6) return 0;
    float k = floor(time / 30.0);
    uint h = pcg(uint(int(k)) * 2654435761u ^ (seed + 4242u));
    float dur = mix(0.4, 0.8, hashf(pcg(h + 1u)));
    float u = (time - k * 30.0 - mix(5.0, 25.0 - dur, hashf(pcg(h + 2u)))) / dur;
    float z = dot(d, f.camFwd.xyz);
    if (u <= 0.0 || u >= 1.0 || z <= 0.0) return 0;
    float cell = 2.0 * f.camPos.w * f.camFwd.w / 864.0;
    float px = max(1.0, 864.0 / f.resolution.x);
    float2 p = float2(dot(d, f.camRight.xyz), dot(d, f.camUp.xyz)) / (z * cell);
    float2 s = float2(mix(-0.65, 0.65, hashf(pcg(h + 3u))) * 432.0, mix(0.5, 0.92, hashf(pcg(h + 4u))) * 432.0 / f.camFwd.w);
    float a = mix(-2.7, -0.45, hashf(pcg(h + 5u)));
    float2 dir = float2(cos(a), sin(a));
    float headAt = mix(60.0, 120.0, hashf(pcg(h + 6u))) * u;
    float tail = min(headAt, mix(12.0, 22.0, hashf(pcg(h + 7u))) * (0.4 + u));
    float2 r = p - s;
    float along = headAt - dot(r, dir), across = abs(dot(r, float2(-dir.y, dir.x))) / px;
    if (along < -1.0 || along > tail || across > 1.0) return 0;
    // Bright through its middle; it flares in and burns out.
    float life = smoothstep(0.0, 0.2, u) * (1.0 - smoothstep(0.55, 1.0, u));
    float fall = saturate(1.0 - along / max(tail, 1e-3));
    float w = mix(0.55, 1.0, fall * fall);
    float body = smoothstep(w, w - 0.3, across) * smoothstep(-1.0, 0.0, along);
    float3 col = mix(float3(1.0, 0.62, 0.35), float3(0.85, 0.95, 1.1), fall) * (0.1 + 2.6 * fall * fall * fall);
    return col * body * life * smoothstep(0.6, 1.0, night) * smoothstep(0.0, 0.05, d.y);
}
