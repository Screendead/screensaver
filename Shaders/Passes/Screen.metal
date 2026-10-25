struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

// A screen position in [0, 1] (y down) in normalised device coordinates (y up).
static inline float2 screenNdc(float2 uv) {
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return ndc;
}

// The eye ray through ndc.
static inline float3 viewRay(constant Frame& f, float2 ndc) {
    float tanH = f.camPos.w;
    return normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
}

// Interleaved gradient noise at pixel pos, moved on every frame.
static inline float pixelDither(constant Frame& f, float2 pos) {
    return fract(52.9829189 * fract(dot(pos + f.misc2.x * float2(5.588, 2.31), float2(0.06711056, 0.00583715))));
}
