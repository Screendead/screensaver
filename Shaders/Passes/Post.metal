// misc2.w is the display's EDR headroom; 0 means SDR output (sRGB-encoded 8-bit).
fragment float4 postFragment(VOut in [[stage_in]], texture2d<float> scene [[texture(0)]],
                             constant Frame& f [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    // Golden hour sits lower on the curve, below the shoulder where ACES bleaches warm skies to cream.
    float sunUp = smoothstep(-0.04, 0.04, f.sunDir.y);
    // Retro pixels: the history is the cell grid; each output pixel takes its cell's flat colour.
    float2 grid = float2(scene.get_width(), scene.get_height());
    float2 cellId = min(floor(in.uv * grid), grid - 1.0);
    float2 uv = (cellId + 0.5) / grid;
    float3 x = scene.sample(s, uv).rgb * f.misc.x * mix(1.0, 0.75, goldAmount(f.sunDir.y) * sunUp);
    float3 c = acesFit(x);
    // Noon light is the flattest; restore a little chroma the tonemap shoulder takes away.
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = max(mix(float3(l), c, 1.0 + 0.10 * smoothstep(0.2, 0.45, f.sunDir.y)), 0.0);
    // 8x8 Bayer, fixed to the screen; 8 levels per channel in sRGB, and four times as many below 2/7. Under motion the
    // picture slides through the fixed pattern: in near-black foliage, with no edge for the steps to follow, whole steps
    // from black to 1/7 boiled.
    uint2 b = uint2(cellId) & 7u;
    uint bx = b.x ^ b.y;
    uint bayer = ((bx & 1u) << 5) | ((b.y & 1u) << 4) | ((bx & 2u) << 2) | ((b.y & 2u) << 1) | ((bx & 4u) >> 1) | ((b.y & 4u) >> 2);
    float3 enc = linearToSRGB(saturate(c));
    float th = (float(bayer) + 0.5) / 64.0;
    if (DITHER_FC) enc = select(floor(enc * 7.0 + th) / 7.0, floor(enc * 28.0 + th) / 28.0, enc < 2.0 / 7.0);
    float headroom = f.misc2.w;
    if (headroom <= 1.0) return float4(enc, 1);
    // EDR: identical to SDR below scene value 1; energy above it rolls per channel into the headroom.
    // EDR: a hue-preserving luminance knee above diffuse white, so only glitter, sun, moon and
    // bright stars rise into the headroom.
    float3 lin = srgbToLinear(saturate(enc));
    float Y = dot(x, float3(0.2126, 0.7152, 0.0722));
    float e = (headroom - 1.0) * (1.0 - exp(-max(Y - 1.6, 0.0) / 2.5));
    lin *= 1.0 + e / max(dot(lin, float3(0.2126, 0.7152, 0.0722)), 1e-3);
    if (f.misc2.z == 4) return float4(e > 0.05 ? float3(1, 0, 0) : lin, 1);
    return float4(lin, 1);
}
