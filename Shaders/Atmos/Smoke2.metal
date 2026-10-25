// Lamp haze and smoke between the eye and a fogged surface at tSurf.
static float3 hazeAndSmoke(thread const Ctx& x, float3 col, device const Site* sites, device const float4* lampPos,
                           device const float4* lampState, constant int4* refs, int4 counts, float3 o, float3 d, float tSurf) {
    if (x.lights.w > 0.0 && counts.x > 0) col += lampHaze(x, lampPos, lampState, refs, counts.x, o, d, tSurf) * 0.05 * x.lights.w;
    if (counts.y > 0) {
        float4 sm = smoke(x, sites, lampPos, lampState, refs, counts.y, o, d, tSurf);
        if (sm.a > 0.0) {
            float T = fogTransmittance(x, o, d, min(tSurf, 600.0));
            col = col * (1.0 - sm.a * T) + sm.rgb * T;
        }
    }
    return col;
}
