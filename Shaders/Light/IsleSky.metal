// shadows: 0 none, 1 traced near ray and maps, 2 maps only (reflections: ripples hide contact shadows).
// L: the light (the bed under water takes it refracted); inWater: seen in rippled water.
// The share of the sky the islands overhead hide from a point at map-local p: each a disc of its radius at its
// underside's mean height (the view factor of a disc to a level patch), out to 2-3k from the camera and to where it hides
// a five-hundredth of the sky.
static float isleSkyOcc(thread const Ctx& x, float3 p) {
    int n = min(x.isles->n.x, ISLE_TABLE);
    float2 w = p.xz + float2(x.origin), cam = x.tr.cam + float2(x.origin);
    float occ = 0.0;
    for (int i = 0; i < n; i++) {
        constant IsleShade& s = x.isles->r[i];
        float hh = s.a.w - p.y;
        float2 dv = w - s.a.xy;
        float r2 = dot(dv, dv);
        if (hh < 4.0 || r2 > s.o.x) continue;
        float R2 = s.b.x * s.b.x, h2 = hh * hh;
        float q = h2 + r2 + R2;
        float F = 0.5 * (1.0 - (h2 + r2 - R2) / sqrt(max(q * q - 4.0 * r2 * R2, 1e-3)));
        occ += F * (1.0 - smoothstep(2000.0, 3000.0, length(s.a.xy - cam)));
    }
    return saturate(occ);
}
