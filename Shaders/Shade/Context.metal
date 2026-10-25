struct Ctx {
    Terrain tr;
    int2 origin;
    uint seed;
    float time;
    float quality;
    float pixelAngle;
    float cover;
    Light L;
    float dither; // per-pixel, per-frame in [0, 1); TAA resolves it
    texture2d<float> cloudTex;
    float3 cloudInfo; // as Frame.cloud
    texture2d<uint> st;
    device const Site* sites;
    device const float4* lampPos;
    device const float4* lampState;
    texture2d<float> lightMap;
    texture2d<float> lightDir;
    float4 lights;
    texture2d<float> shadowNear;  // shadow-height maps, SHADOW_CELL and 4x that, toroidal
    texture2d<float> shadowFar;
    float shadowMode;             // 0 none, 1 maps, 2 brute-force rays (debug)
    bool prof1;
    int prof;
    texture2d<float> slowTex;
    float4 slowInfo;
    uint tiles;                   // haze lamps (bits 0-15) and smoke plumes (16-27) that may touch this pixel's screen tile
    constant TowerTable* towers;  // the ruined towers nearest the camera
    bool isleShadows;             // an island stands in view: the shadow maps hold island shade
    constant IsleTable* isles;    // the islands nearest the camera
};

// The context of the scene and vine passes' pixel at pos; the tables (tiles, towers, isles) are left for the pass.
static inline Ctx makeCtx(thread const Terrain& tr, constant Frame& f, float2 pos, texture2d<float> cloudTex,
                          texture2d<uint> st, device const Site* sites, device const float4* lampPos,
                          device const float4* lampState, texture2d<float> lightMap, texture2d<float> lightDir,
                          texture2d<float> shadowNear, texture2d<float> shadowFar, texture2d<float> slowTex) {
    float tanH = f.camPos.w;
    return Ctx { tr, f.mapInfo.xy, uint(f.mapInfo.w), f.camRight.w, f.camUp.w,
                 2.0 * tanH * f.resolution.z, f.misc.w, makeLight(f.sunDir.xyz, uint(f.mapInfo.w), f.lights.z),
                 pixelDither(f, pos),
                 cloudTex, f.cloud.xyz, st, sites, lampPos, lampState, lightMap, lightDir, f.lights, shadowNear, shadowFar,
                 f.misc2.z == 14 ? 2.0 : f.shadow.x, (int(f.shadow.y) & 1) != 0, int(f.shadow.y), slowTex,
                 f.slowCloud.w == f.camRight.w ? f.slowCloud : float4(0), 0u };
}

// Baked fine cloud field where it covers q, else evaluated.
