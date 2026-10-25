// Value noise in about [-0.5, 0.5]: perlin2's gradients cost a sine and cosine per corner, too dear for every water pixel.
static inline float seaNoise(float2 p, uint s) {
    float2 i = floor(p), f = p - i;
    int2 c = int2(i);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hashf(hash2i(c, s)), b = hashf(hash2i(c + int2(1, 0), s));
    float cc = hashf(hash2i(c + int2(0, 1), s)), d = hashf(hash2i(c + int2(1, 1), s));
    return mix(mix(a, b, u.x), mix(cc, d, u.x), u.y) - 0.5;
}

// seaNoise at half the hashing (one pcg a corner), for the night glow's many fields.
static inline float bioNoise(float2 p, uint s) {
    float2 i = floor(p), f = p - i;
    int2 c = int2(i);
    float2 u = f * f * (3.0 - 2.0 * f);
    uint x0 = uint(c.x) * 1597334677u, x1 = x0 + 1597334677u, y0 = uint(c.y) * 3812015801u ^ s, y1 = (uint(c.y) * 3812015801u + 3812015801u) ^ s;
    float a = hashf(pcg(x0 ^ y0)), b = hashf(pcg(x1 ^ y0)), cc = hashf(pcg(x0 ^ y1)), d = hashf(pcg(x1 ^ y1));
    return mix(mix(a, b, u.x), mix(cc, d, u.x), u.y) - 0.5;
}
