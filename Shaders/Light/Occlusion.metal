static inline float vao(float s1, float s2, float c) { return (s1 > 0.5 && s2 > 0.5) ? 0.0 : 3.0 - (s1 + s2 + c); }

// Minecraft-style per-corner AO, bilinearly interpolated across the face.
static float cornerAO(texture2d<float> hm, int size, int3 v, int3 n, float2 uv) {
    int3 a = v + n;
    int3 tu = n.x != 0 ? int3(0, 0, 1) : int3(1, 0, 0);
    int3 tv = n.y != 0 ? int3(0, 0, 1) : int3(0, 1, 0);
    float su0 = solid(hm, a - tu, size), su1 = solid(hm, a + tu, size);
    float sv0 = solid(hm, a - tv, size), sv1 = solid(hm, a + tv, size);
    float c00 = solid(hm, a - tu - tv, size), c10 = solid(hm, a + tu - tv, size);
    float c01 = solid(hm, a - tu + tv, size), c11 = solid(hm, a + tu + tv, size);
    float a00 = vao(su0, sv0, c00), a10 = vao(su1, sv0, c10), a01 = vao(su0, sv1, c01), a11 = vao(su1, sv1, c11);
    float ao = mix(mix(a00, a10, uv.x), mix(a01, a11, uv.x), uv.y) / 3.0;
    return mix(0.6, 1.0, pow(ao, 0.8));
}

// Fraction of open sky above a point, from how far the surrounding columns rise above y within 6 voxels.
// Heights are filtered, so a c on a column centre reads exact columns.
// Coarse levels sample their own slice at the same world radii.
// mode 0: 8 directions at 2, 4 and 6; 1 (far: a pixel spans columns): 8 directions at 4; 2 (seen in water): 4 at 4.
static float skyVisibility(thread const Terrain& tr, int k, float2 c, float y, int mode = 0) {
    constexpr sampler bl(filter::linear, address::clamp_to_edge);
    constexpr sampler blr(filter::linear, address::repeat);
    float2 inv = 1.0 / float2(tr.hm.get_width(), tr.hm.get_height());
    float cs = tr.lod[k].w;
    float2 fo = float2(levelOrigin(tr, k) & (COARSE_N - 1));
    float occ = 0.0;
    for (int i = 0; i < 8; i += mode == 2 ? 2 : 1) {
        // (1,0) (-1,0) (0,1) (0,-1) (1,1) (-1,1) (1,-1) (-1,-1), computed: a local table indexed in a loop of dynamic
        // stride lives in scratch memory.
        int2 dir = i < 2 ? int2(i == 0 ? 1 : -1, 0) : i < 4 ? int2(0, i == 2 ? 1 : -1) : int2((i & 1) == 0 ? 1 : -1, i < 6 ? 1 : -1);
        float m = 0.0;
        for (int r = mode > 0 ? 2 : 1; r <= (mode > 0 ? 2 : 3); r++) {
            float dist = float(r * 2) * (i < 4 ? 1.0 : 1.414);
            float2 q = c + float2(dir * (r * 2));
            float rise = (k == 0 ? tr.hx.sample(bl, q * inv, level(0)).r
                                 : tr.ch.sample(blr, (q / cs + fo) / float(COARSE_N), uint(k - 1), level(0)).r) - y;
            m = max(m, rise / sqrt(rise * rise + dist * dist));
        }
        occ += m;
    }
    return 1.0 - occ / (mode == 2 ? 4.0 : 8.0);
}

// Position on the face in [0, 1) of a column s voxels wide (y stays in unit voxels).
static inline float2 faceUV(float3 p, int3 n, float s) {
    float3 f = fract(p / float3(s, 1.0, s));
    if (n.y != 0) return f.xz;
    if (n.x != 0) return float2(f.z, f.y);
    return float2(f.x, f.y);
}
