// Glowing bugs drifting in and around the foliage after dusk: at most one in each MOTE_CELL cube, drawn as a small
// sprite over the scene (Renderer: one instance per cube of the MOTE_N^3 block round the camera). Whether a bug is there
// is decided at its anchor, so drifting over a column edge never switches it.
constant float MOTE_CELL = 6.0;
constant float MOTE_REACH = 66.0;
constant int MOTE_N = 23; // cubes along each side of the block, which must hold the sphere of MOTE_REACH; Renderer.moteCount

struct MoteOut { float4 pos [[position]]; float3 m [[flat]]; float re [[flat]]; float3 col [[flat]]; };

vertex MoteOut bioVertex(uint vid [[vertex_id]], uint iid [[instance_id]], constant Frame& f [[buffer(0)]],
                         texture2d<float> hm [[texture(0)]], texture2d<float> clim [[texture(2)]], texture2d<float> pv [[texture(16)]]) {
    MoteOut o;
    o.pos = float4(0, 0, -1, 1);
    o.m = 0; o.re = 1; o.col = 0;
    float sky = bioRamp(f.sunDir.y);
    if (sky <= 0.0) return o;
    int2 origin = f.mapInfo.xy;
    uint seed = uint(f.mapInfo.w);
    float time = f.camRight.w;
    float2 org = float2(origin);
    float3 wo = f.camPos.xyz + float3(org.x, 0.0, org.y);
    int3 c = int3(floor((wo - MOTE_REACH) / MOTE_CELL)) + int3(int(iid) % MOTE_N, (int(iid) / MOTE_N) % MOTE_N, int(iid) / (MOTE_N * MOTE_N));
    uint hh = pcg(uint(c.x) * 1597334677u ^ uint(c.y) * 3812015801u ^ uint(c.z) * 2798796415u ^ (seed + 710u));
    // 30% of cells hold a bug: at half, the near swarm was too bright.
    if (hashf(hh) >= 0.3) return o;
    uint h1 = pcg(hh), h2 = pcg(h1), h3 = pcg(h2), h4 = pcg(h3), h5 = pcg(h4);
    float3 a = float3(c) * MOTE_CELL + 1.3 + (MOTE_CELL - 2.6) * float3(hashf(h1), hashf(h2), hashf(h3));
    float ph = 6.283 * hashf(h4);
    float3 m = a + 0.8 * float3(sin(time * 0.31 + ph), 0.5 * sin(time * 0.47 + 1.7 * ph), sin(time * 0.23 + 2.3 * ph));
    float3 v = m - wo;
    float dist = length(v), z = dot(v, f.camFwd.xyz);
    float fade = smoothstep(3.0, 10.0, dist) * (1.0 - smoothstep(0.65 * MOTE_REACH, MOTE_REACH, dist));
    if (fade <= 0.0 || z < 0.5) return o;
    float2 ml = a.xz - org;
    float4 col4 = hm.read(uint2(clamp(int2(floor(ml)), int2(0), int2(hm.get_width() - 1))), 0);
    bool veg = col4.a > 0.0 || col4.a == GIANT_A;
    float above = a.y - col4.r;
    if (above >= 8.0 || above <= (veg ? -25.0 : -1.0)) return o;
    Climate cl = climateAt(clim, ml, pv, origin);
    float w = max(smoothstep(0.1, 0.5, provinceAmount(cl)), (veg ? 0.7 : 0.35) * smoothstep(0.25, 0.55, cl.moist))
            * (1.0 - smoothstep(0.42, 0.18, cl.temp));
    if (hashf(h5) >= w) return o;
    float tanH = f.camPos.w;
    float re = max(0.1, 0.8 * dist * 2.0 * tanH * f.resolution.w);
    float2 corner = float2(float(vid & 1u), float(vid >> 1)) * 2.0 - 1.0;
    float3 p = v + (f.camRight.xyz * corner.x + f.camUp.xyz * corner.y) * 4.5 * re;
    float pz = dot(p, f.camFwd.xyz);
    float2 ndc = float2(dot(p, f.camRight.xyz) / (tanH * f.camFwd.w), dot(p, f.camUp.xyz) / tanH) / pz;
    o.pos = float4(ndc - f.jitter.xy, 0.5, 1.0);
    o.m = m - float3(org.x, 0.0, org.y);
    o.re = re;
    // Each flashes on its own slow rhythm, fading in and out over a second or so.
    float blink = 0.15 + 0.85 * smoothstep(0.3, 1.0, sin(time * (0.5 + 0.6 * hashf(pcg(h5))) + ph));
    // The glow's hue where it flies, a little paler or deeper bug by bug.
    float3 hue = bioHue(a.xz, seed) * (0.8 + 0.4 * hashf(pcg(h5 + 1u)));
    o.col = hue * blink * fade * sky;
    return o;
}

// Adds a bug's glow along the pixel's ray up to the depth the scene left in alpha.
fragment float4 bioFragment(MoteOut in [[stage_in]], float4 dst [[color(0)]], constant Frame& f [[buffer(0)]]) {
    float2 ndc = in.pos.xy * f.resolution.zw * 2.0 - 1.0;
    ndc.y = -ndc.y;
    ndc += f.jitter.xy;
    float tanH = f.camPos.w;
    float3 d = normalize(f.camFwd.xyz + f.camRight.xyz * ndc.x * tanH * f.camFwd.w + f.camUp.xyz * ndc.y * tanH);
    float3 v = in.m - f.camPos.xyz;
    float tm = dot(v, d);
    if (tm > abs(dst.a)) return dst;
    float e2 = (dot(v, v) - tm * tm) / (in.re * in.re);
    float g = (3.0 * exp(-e2) + 0.4 * exp(-e2 / 9.0)) * (0.1 * 0.1) / (in.re * in.re) * (1.0 - smoothstep(9.0, 20.0, e2));
    return float4(dst.rgb + in.col * g, dst.a);
}
