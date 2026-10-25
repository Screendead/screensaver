// The strands' pass, drawn over the scene while a giant stands within the world's edge: dst is the scene's pixel
// (its distance in alpha, a bird's negative), and the strands in front of it replace it or cover it. Its own pass:
// compiled into sceneFragment, the walk's registers cost every pixel of a giant scene a third of a millisecond unrun.
fragment float4 vineFragment(VOut in [[stage_in]], float4 dst [[color(0)]], constant Frame& f [[buffer(0)]],
                             texture2d<float> hm [[texture(0)]], texture2d<float> cloudTex [[texture(1)]],
                             texture2d<float> clim [[texture(2)]], texture2d<uint> st [[texture(3)]],
                             texture2d<float> lightMap [[texture(4)]], texture2d<float> lightDir [[texture(5)]],
                             texture2d_array<float> coarse [[texture(6)]], texture2d_array<float> coarseClim [[texture(7)]],
                             texture2d_array<float> coarseCover [[texture(8)]],
                             texture2d<float> shadowNear [[texture(9)]], texture2d<float> shadowFar [[texture(10)]],
                             texture2d<float> slowTex [[texture(11)]], texture2d<float> hx [[texture(12)]],
                             texture2d<uint> gx [[texture(13)]], texture2d_array<float> cg [[texture(14)]],
                             texture2d_array<float> cg2 [[texture(17)]],
                             texture2d<float> hb [[texture(15)]], texture2d<float> pvTex [[texture(16)]],
                             device const Site* sites [[buffer(1)]], device const float4* lampPos [[buffer(2)]],
                             device const float4* lampState [[buffer(3)]], constant uint* tiles [[buffer(7)]],
                             device const GiantRec* giants [[buffer(8)]], constant TowerTable* towers [[buffer(9)]],
                             constant VineTable* vines [[buffer(10)]], texture2d_array<float> vineFine [[texture(18)]],
                             texture2d_array<uint> vineCoarse [[texture(19)]], texture2d_array<uint> vineSuper [[texture(20)]],
                             texture2d_array<uint> vineBands [[texture(21)]]) {
    float2 ndc = screenNdc(in.uv) + f.jitter.xy;
    float tanH = f.camPos.w;
    float3 d = viewRay(f, ndc);
    float3 o = f.camPos.xyz;
    int levels = int(f.fade.z);
    float gmax = f.misc.y;
    for (int k = 1; k < levels; k++) gmax = max(gmax, coarse.read(uint2(0), uint(k - 1), coarse.get_num_mip_levels() - 1).r + 1.0);
    if (int(f.shadow.y) & 2) gmax = min(gmax, 260.0);
    Terrain tr { hm, coarse, coarseCover, coarseClim, clim, f.lod, f.mapInfo.xy, f.mapInfo.z, levels, f.camPos.xz + normalize(f.camFwd.xz + 1e-6) * f.fade.w, f.fade.y, f.fade.x, gmax, hx, gx, cg, cg2, hb, uint(f.mapInfo.w), giants, pvTex, towers, 0 };
    Ctx x { tr, f.mapInfo.xy, uint(f.mapInfo.w), f.camRight.w, f.camUp.w,
            2.0 * tanH * f.resolution.z, f.misc.w, makeLight(f.sunDir.xyz, uint(f.mapInfo.w), f.lights.z),
            pixelDither(f, in.pos.xy),
            cloudTex, f.cloud.xyz, st, sites, lampPos, lampState, lightMap, lightDir, f.lights, shadowNear, shadowFar,
            f.misc2.z == 14 ? 2.0 : f.shadow.x, (int(f.shadow.y) & 1) != 0, int(f.shadow.y), slowTex,
            f.slowCloud.w == f.camRight.w ? f.slowCloud : float4(0), 0u };
    x.towers = towers;
    if (f.misc2.z >= 1 && f.misc2.z != 2) return dst;
    float tSurf = abs(dst.a);
    StrandHit vh = traceStrands(vineFine, vineCoarse, vineSuper, vineBands, vines, x.origin, o, d, tSurf, x.pixelAngle);
    float vineA = mix(vh.hit ? 1.0 : 0.0, vh.alpha, vh.w);
    if (vineA <= 0.0) return dst;
    float tv = vh.hit ? vh.t : vh.tm;
    // The cover over a bird in front of it is nothing of the bird's.
    if (!vh.hit && dst.a < 0.0 && tv > tSurf) return dst;
    float3 nv = vh.hit ? float3(vh.n) : normalize(float3(-d.x, 0.0, -d.z) + float3(1e-5, 0, 0));
    float top = vh.hit ? vh.top : vh.topm;
    float draw = vh.hit ? hashf(hash2i(vh.cell, x.seed + 602u)) : 0.5;
    float3 col = shadeVine(x, x.L, o, d, tv, nv, top, draw, !vh.hit);
    float seaAmt = cloudSeaAmount(x);
    if (seaAmt > 0.0 && f.misc2.z != 2) { SeaFog sf = cloudSea(x, seaAmt, o, d, tv); col = col * sf.T + sf.col; }
    if (f.misc2.z != 2) col = applyFog(x, col, o, d, tv);
    return float4(coverMix(dst.rgb, col, vineA), vh.hit && vh.w <= 0.0 ? vh.t : dst.a);
}
