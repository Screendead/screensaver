// Whether column c of level k is hollow at height y (inside one of its rock gaps).
// An island's air is open sky, not a shelter.
static inline bool hollowAt(thread const Terrain& tr, int k, int2 c, float y) {
    float4 col = columnL(tr, k, c);
    if (col.z <= col.y || col.a != 0.0) return false;
    if (y >= col.y && y < col.z) return !(ISLE_SHAPE && isleGapL(tr, k, c, col));
    if (k > 0 || y >= col.y) return false;
    float4 e = extraGaps(tr.gx.read(uint2(clamp(c, int2(0), int2(tr.size - 1)))));
    return (y >= e.x && y < e.y) || (y >= e.z && y < e.w);
}

// The share of open shade's sky light reaching under a roof.
constant float ROOF_SKY = 0.55;
// A shelter's back wall by night: about open shade's share on a wall.
constant float ROOF_SKY_NIGHT = 1.0;

// hollowAt as the trace sees it: the column morphed toward the next level's.
static inline bool hollowAtMorph(thread const Terrain& tr, int k, int2 c, float y) {
    float4 c4 = columnL(tr, k, c);
    if (c4.a != 0.0) return false;
    float3 col = columnMorph(tr, k, c);
    if (col.z <= col.y) return false;
    if (y >= col.y && y < col.z) return !ISLE_SHAPE || (!isleGapL(tr, k, c, c4) && parentGapS(tr, k, c).x >= 0.0);
    if (k > 0 || y >= col.y) return false;
    float4 e = extraGapsMorph(tr, c);
    return (y >= e.x && y < e.y) || (y >= e.z && y < e.w);
}

// Near p at height y: x, the share of the ground under a roof (per column, whether it or a 4-neighbour is hollow at y,
// bilinear over the four columns nearest p); yz, which way the hollow lies (its share's gradient over those four). A
// function of position alone, never of which way a face turns, so the jogging faces of a wall under a roof all agree,
// and it fades over a column or two at a roof's edge. Twelve columns, each read once.
static float3 roofField(thread const Terrain& tr, int k, float2 p, float y) {
    float cs = tr.lod[k].w;
    float2 q = p / cs - 0.5, f = fract(q);
    int2 i0 = int2(floor(q));
    // The 4x4 block from i0 - 1, less its corners; bit (j * 4 + i).
    uint hb = 0u;
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            if ((i == 0 || i == 3) && (j == 0 || j == 3)) continue;
            if (hollowAt(tr, k, i0 + int2(i - 1, j - 1), y)) hb |= 1u << uint(j * 4 + i);
        }
    if (hb == 0u) return float3(0);
    float share = 0.0;
    float hv[4];
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            uint b = uint((j + 1) * 4 + i + 1);
            uint near = (1u << b) | (1u << (b - 1u)) | (1u << (b + 1u)) | (1u << (b - 4u)) | (1u << (b + 4u));
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            if ((hb & near) != 0u) share += w;
            hv[j * 2 + i] = float((hb >> b) & 1u);
        }
    float2 g = float2(mix(hv[1] - hv[0], hv[3] - hv[2], f.y), mix(hv[2] - hv[0], hv[3] - hv[1], f.x));
    return float3(share, g);
}
