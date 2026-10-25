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

// Baked fine cloud field where it covers q, else evaluated.
