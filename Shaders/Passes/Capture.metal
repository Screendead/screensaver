// Bug-report capture: the presented cell grid as NV12 (BT.709, video range) for the hardware encoder, one 2x2 block per
// thread. `linear` marks an EDR drawable, whose SDR part re-encodes exactly to what postFragment quantised.
kernel void captureNV12(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> luma [[texture(1)]],
                        texture2d<float, access::write> chroma [[texture(2)]],
                        constant uint& linear [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= chroma.get_width() || gid.y >= chroma.get_height()) return;
    const float3 k = float3(0.2126, 0.7152, 0.0722);
    float3 sum = 0.0;
    for (uint i = 0; i < 4; i++) {
        uint2 p = gid * 2 + uint2(i & 1u, i >> 1);
        float3 c = saturate(src.read(p).rgb);
        if (linear != 0) c = linearToSRGB(c);
        luma.write(float4((16.0 + 219.0 * dot(c, k)) / 255.0), p);
        sum += c;
    }
    float3 c = sum * 0.25;
    float y = dot(c, k);
    chroma.write(float4((128.0 + 224.0 * (c.b - y) / 1.8556) / 255.0, (128.0 + 224.0 * (c.r - y) / 1.5748) / 255.0, 0, 0), gid);
}
