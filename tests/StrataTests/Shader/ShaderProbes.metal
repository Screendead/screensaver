// Kernels that expose the shader's own functions, constants and layouts to ShaderFunctionTests and ShaderLayoutTests,
// which build this file against Shaders/ at test time. Never part of the app's metallib.
#include "Saver.metal"

// x pcg(v), y hash2i((a, b), s), z pinnedBiome(v) for each (v, a, b, s).
kernel void probeHash(device const uint4* in [[buffer(0)]], device uint4* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint4 q = in[i];
    out[i] = uint4(pcg(q.x), hash2i(int2(int(q.y), int(q.z)), q.w), pinnedBiome(q.x), 0);
}

// The jungle and old-growth provinces at world xz, as the shader evaluates them for a seed.
kernel void probeProvinces(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]], constant uint& seed [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    out[i] = provinces(in[i] + seedOffset(seed), seed);
}

// makeLight's key direction (xyz) and, in w, 1 + shade where it casts shadows and 0 where it casts none, for each sun direction.
kernel void probeLight(device const float3* in [[buffer(0)]], device float4* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    Light L = makeLight(in[i], 1u, 0.0);
    out[i] = float4(L.key, L.shadows ? 1.0 + L.shade : 0.0);
}

// cloudSeaAmount for each (sun elevation, night progress); built with SEA_FC on.
kernel void probeCloudSea(device const float2* in [[buffer(0)]], device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    Ctx x;
    x.L.sun = float3(sqrt(1.0 - in[i].x * in[i].x), in[i].x, 0.0);
    x.lights = float4(in[i].x, in[i].y, 0.0, 0.0);
    out[i] = cloudSeaAmount(x);
}

// lampLevel deep in the night (sun far down, night half through) for each (site hash, lamp index, night, kind), with
// the per-lamp hash updateLamps gives it.
kernel void probeLamp(device const uint4* in [[buffer(0)]], device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint4 q = in[i];
    out[i] = lampLevel(int(q.w), pcg(q.x + q.y * 7717u), float4(-0.5, 0.5, float(int(q.z)), 0.0));
}

// The constants the CPU mirrors, in ShaderLayoutTests.constants' order.
kernel void probeConstants(device float* out [[buffer(0)]]) {
    const float v[] = {
        SEA, float(SITE_GRID), float(SITE_CELL), float(MAX_LAMPS), float(MAX_PARTS), float(GIANT_WIN), GIANT_CELL,
        float(CAVE_WIN), CAVE_CELL, CAVE_REACH, float(ISLE_WIN), float(ISLE_MWIN), float(ISLE_SLOTS), float(ISLE_TABLE),
        float(ISLE_MIST_PUFFS), TOWER_CELL, float(TOWER_MEMBERS), float(TOWER_TABLE), float(VINE_SLOTS), float(VINE_W),
        float(MAX_FLOCKS), float(MAX_CLUSTERS), CLOUD_Y0, float(MOTE_N), float(LAMP_TILES), BIO_SUN,
        LAMP_R[0], LAMP_R[1], LAMP_R[2], LAMP_R[3], LAMP_R[4], LAMP_R[5], LAMP_R[6], LAMP_R[7],
    };
    for (uint i = 0; i < sizeof(v) / sizeof(float); i++) out[i] = v[i];
}

#define SIZE(T) int(sizeof(T))
#define AT(T, f) int(__builtin_offsetof(T, f))

// Sizes and field offsets of the records the CPU writes or reads, in ShaderLayoutTests.layouts' order.
kernel void probeLayouts(device int* out [[buffer(0)]]) {
    const int v[] = {
        SIZE(Frame), AT(Frame, sunDir), AT(Frame, mapInfo), AT(Frame, lights), AT(Frame, refs), AT(Frame, birds), AT(Frame, lod),
        AT(Frame, fade), AT(Frame, shadow), AT(Frame, slowCloud), AT(Frame, planet), AT(Frame, planetPx),
        SIZE(GiantRec), AT(GiantRec, c), AT(GiantRec, G), AT(GiantRec, yT), AT(GiantRec, h), AT(GiantRec, kind),
        SIZE(CaveRec), AT(CaveRec, m), AT(CaveRec, u), AT(CaveRec, yF), AT(CaveRec, W), AT(CaveRec, L), AT(CaveRec, Hc),
        AT(CaveRec, h), AT(CaveRec, flags),
        SIZE(IsleRec), AT(IsleRec, c), AT(IsleRec, R), AT(IsleRec, A), AT(IsleRec, D), AT(IsleRec, B), AT(IsleRec, K),
        AT(IsleRec, h), AT(IsleRec, fa), AT(IsleRec, f),
        SIZE(Site), AT(Site, info), AT(Site, lamps), AT(Site, parts),
        SIZE(TowerRec), SIZE(IsleShade), SIZE(IsleTable), SIZE(TowerTable),
        SIZE(ShadowBake), AT(ShadowBake, cam), AT(ShadowBake, rows),
        SIZE(VineBake), AT(VineBake, b), AT(VineBake, c), SIZE(VineRec), SIZE(VineTable),
    };
    for (uint i = 0; i < sizeof(v) / sizeof(int); i++) out[i] = v[i];
}
