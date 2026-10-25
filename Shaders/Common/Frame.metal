// Must match `Frame` in Sources/Renderer.swift.
struct Frame {
    float4 camPos;     // xyz map-local position, w tan(fovY/2)
    float4 camFwd;     // w aspect
    float4 camRight;   // w wrapped time (s)
    float4 camUp;      // w quality (0 battery, 1 full)
    float4 sunDir;     // xyz direction to sun, w day phase
    float4 resolution; // xy pixels, z inverse height of the shading footprint's pixel, w inverse height
    int4   mapInfo;    // xy world xz of texel 0, z size, w seed
    float4 misc;       // x exposure, y global max height, z wind angle, w cloud cover
    float4 misc2;      // x frame index, y fog density scale, z debug mode, w EDR headroom (0 = SDR)
    float4 prevPos;    // previous frame's camera, in this frame's map-local coords; w history valid
    float4 prevFwd;
    float4 prevRight;
    float4 prevUp;
    float4 jitter;     // xy subpixel jitter in NDC
    float4 cloud;      // xy cloud field origin in drifted world xz, z field texel size (0: no field)
    float4 lights;     // x sun elevation, y night progress (0 dusk .. 1 dawn), z day index, w lamp light on the land (0 by day)
    int4   refs;       // x lamps in the haze list, y smoke plumes
    float4 birds;      // x bird flock count
    float4 lod[5];     // per level: x morph band start, y handoff radius (the last level's is the world's edge), z morph end, w column width
    float4 fade;       // x fog fade start, y world radius, z level count, w ring centre's offset ahead of the camera (tests)
    float4 shadow;     // x 1 when the shadow-height maps hold the current key light (3 no shadows), y profiling flags, z 1 when
                       // the horizon buffer is valid, w 1 when an island is in view
    float4 slowCloud;  // xy slow cloud field origin in world xz, z texel size (0: no field), w the time it was baked for
    float4 planet;     // xyz direction to the evening or morning star
    float4 planetPx;   // xy the cell corner the star's core is centred on, unjittered; z scene pixels per cell
};
