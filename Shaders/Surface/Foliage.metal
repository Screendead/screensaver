// Autumn foliage: a slow Perlin hue field runs lingering green, gold, orange, scarlet to crimson;
// birch groves lean gold. Kept dark and earthy so the colour reads as paint, not candy.
static inline float3 autumnLeaf(float2 w, float birch, uint seed) {
    float k = saturate(perlin2(w / 75.0, seed + 96u) * 1.1 + 0.55 + 0.12 * perlin2(w / 19.0, seed + 97u));
    k = mix(k, 0.28, 0.75 * birch);
    float3 green = float3(0.055, 0.075, 0.03), gold = float3(0.22, 0.15, 0.035), orange = float3(0.21, 0.09, 0.035);
    float3 scarlet = float3(0.18, 0.04, 0.025), crimson = float3(0.12, 0.018, 0.018);
    if (k < 0.2) return mix(green, gold, smoothstep(0.0, 0.2, k));
    if (k < 0.45) return mix(gold, orange, smoothstep(0.2, 0.45, k));
    if (k < 0.7) return mix(orange, scarlet, smoothstep(0.45, 0.7, k));
    return mix(scarlet, crimson, smoothstep(0.7, 1.0, k));
}

// A wood's leaves at world column w: deep, low-chroma greens that drift slowly across a grove; conifers cooler and
// darker. A tree's single-run columns and its multi-run ones (old growth, the jungle's roof) must agree, or one crown
// shows two palettes.
static float3 woodLeaf(float2 w, Climate c, float painted, float3 fallLeaf, float autumn, uint seed) {
    float jw = smoothstep(0.1, 0.4, c.jungle);
    float3 jl = 0;
    if (jw > 0.0) {
        // The jungle's crowns differ one from the next: deep green, olive, now and then a cool teal.
        float jk = perlin2(w / 9.0, seed + 340u) + 0.6 * perlin2(w / 31.0, seed + 341u);
        jl = mix(float3(0.028, 0.062, 0.034), float3(0.072, 0.094, 0.032), smoothstep(-0.3, 0.5, jk));
        jl = mix(jl, float3(0.024, 0.068, 0.058), smoothstep(0.25, 0.6, perlin2(w / 17.0 + 7.3, seed + 342u)));
        if (jw >= 1.0) return jl;
    }
    float conifer = smoothstep(0.50, 0.32, c.temp);
    float k = saturate(perlin2(w / 23.0, seed + 93u) * 0.9 + 0.5);
    float3 leaf = mix(float3(0.035, 0.07, 0.035), float3(0.065, 0.09, 0.04), k);
    leaf = mix(leaf, float3(0.025, 0.055, 0.045), conifer);
    leaf = mix(leaf, float3(0.11, 0.125, 0.09), painted);
    leaf = mix(leaf, fallLeaf, autumn * (1.0 - conifer * 0.5));
    return mix(leaf, jl, jw);
}

// Crown by crown, species and age: a third cooler and darker, one in five yellower, one in ten light and warm, each
// lighter or darker again. Over many crowns the mean is 1, the wood's own colour that the coarse rings show. The crown is
// the nearest trunk (the jungle's: the dome standing highest) among the four cells round w (world column centre); Ht its
// height when it is old growth, else 0.
struct Crown { float3 tint; float Ht; };

static Crown crownAt(float2 w, Climate c, uint seed) {
    float best = 1e9, bestH = 0.0;
    uint id = 0u;
    bool jungle = c.jungle >= 0.15, old = !jungle && c.old >= 0.2;
    float cell = jungle ? JUNGLE_CELL : old ? OLD_CELL : float(TREE_CELL);
    int2 c0 = int2(floor(w / cell - 0.5));
    for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++) {
            int2 cc = c0 + int2(i, j);
            float2 tc;
            float R = 1.0, Ht = 0.0;
            uint h;
            if (jungle) {
                h = hash2i(cc, seed + 322u);
                tc = (float2(cc) + 0.2 + 0.6 * float2(hashf(h), hashf(pcg(h)))) * JUNGLE_CELL;
                R = 3.5 + 4.0 * hashf(pcg(h + 1u));
            } else if (old) {
                float tr;
                if (!oldTreeIn(cc, seed, c.old, tc, Ht, R, tr, h)) continue;
            } else {
                h = hash2i(cc, seed + 92u);
                tc = float2(cc * TREE_CELL + int2(h % uint(TREE_CELL), (h / 7u) % uint(TREE_CELL))) + 0.5;
            }
            float d = length_squared(w - tc) / (R * R);
            if (d < best) { best = d; id = h; bestH = Ht; }
        }
    uint h = pcg(id + 361u);
    float u = hashf(h);
    float3 t = u < 0.35 ? float3(0.80, 0.88, 1.05) : u < 0.70 ? float3(1.0) : u < 0.90 ? float3(1.35, 1.15, 0.75) : float3(1.45, 1.3, 0.7);
    Crown o;
    o.tint = t / float3(1.045, 1.018, 0.9375) * (0.75 + 0.5 * hashf(pcg(h)));
    o.Ht = bestH;
    return o;
}
