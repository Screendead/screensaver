// Max-mip quadtree walk over the column heightfields; columns are stacks of unit voxels. A ray never moves to a
// finer level: it starts at the coarser of startLevel and its origin's ring, and steps out a level exactly where
// it leaves each handoff circle. Every level covers the whole disc inside its circle, so a ray turning back toward
// the camera still has data under it.
static Hit trace(thread const Terrain& tr, float3 o, float3 d, float tMax, int maxIter, int startLevel) {
    Hit h;
    h.hit = false;
    h.t = tMax;
    h.iters = -1;
    h.level = 0;
    float3 inv = 1.0 / select(d, copysign(float3(1e-6), d), abs(d) < 1e-6);

    // Clip to [-1, globalMax] in y and to the world's circle.
    float ty0 = (-1.0 - o.y) * inv.y, ty1 = (tr.globalMax - o.y) * inv.y;
    float tEnter = max(min(ty0, ty1), 0.0);
    float tExit = min(min(max(ty0, ty1), tMax), circleExit(o.xz, d.xz, tr.cam, tr.rmax));
    if (tEnter >= tExit) return h;

    float t = tEnter;
    int k = max(startLevel, levelAt(tr, length(o.xz + d.xz * t - tr.cam)));
    if (k >= tr.levels) return h;
    float tSwitch = k < tr.levels - 1 ? circleExit(o.xz, d.xz, tr.cam, tr.lod[k].y) : 1e30;
    float s = tr.lod[k].w, invS = 1.0 / s;
    int2 fo = levelOrigin(tr, k);
    int level = MAX_LEVEL;
    // A ray that starts inside a column hits the side it last crossed into.
    float2 tbox = min((0.0 - o.xz) * inv.xz, (float(tr.size) - o.xz) * inv.xz);
    int lastAxis = (tbox.x > tbox.y) ? 0 : 2;
    // Spatial nudge across the boundary just crossed; a t-epsilon vanishes at large map coords.
    float2 nudge = float2(0);
    for (int i = 0; i < maxIter; i++) {
        h.iters = i;
        if (t >= tExit) return h;
        if (t >= tSwitch) {
            k++;
            tSwitch = k < tr.levels - 1 ? circleExit(o.xz, d.xz, tr.cam, tr.lod[k].y) : 1e30;
            s = tr.lod[k].w;
            invS = 1.0 / s;
            fo = levelOrigin(tr, k);
            level = MAX_LEVEL;
            nudge = float2(0);
            continue;
        }
        float3 p = o + d * t;
        int2 mc = (int2(floor((p.xz + nudge) * invS)) + fo) >> level;
        int2 lc = (mc << level) - fo;
        float2 cLo = float2(lc) * s, cHi = cLo + s * float(1 << level);
        float2 tb = (select(cLo, cHi, d.xz > 0.0) - o.xz) * inv.xz;
        float tCell = min(min(tb.x, tb.y), min(tExit, tSwitch));
        float yA = o.y + d.y * t, yB = o.y + d.y * tCell;
        if (level > 0) {
            // Mips from 1 up also bound the next level's columns, which morph-band columns rise or sink toward.
            // A negative max flags a cell with an open band: under a crown or an island (bandFlag, coarseBandFlag).
            float mx = k == 0 ? tr.hx.read(uint2(clamp(mc, int2(0), int2((tr.size >> level) - 1))), level).r
                              : tr.ch.read(uint2(mc & ((COARSE_N >> level) - 1)), uint(k - 1), uint(level)).r;
            bool clear = min(yA, yB) >= abs(mx);
            // The 1-voxel map's canopy bands are the columns' unmorphed gaps, which close across the morph band: only a
            // cell wholly inside its start may skip through one. Island bands, and every coarse level's, hold through it.
            if (!clear && BAND_SKIP && mx < 0.0) {
                // Only an island's air holds outside the disc, and only tiles with an island's surfaces meet it.
                bool disc = k == 0 && length(0.5 * (cLo + cHi) - tr.cam) + 0.71 * (cHi.x - cLo.x) < tr.lod[0].x;
                if (ISLE_SHAPE || disc) {
                    float2 bd = k == 0 ? tr.hb.read(uint2(clamp(mc, int2(0), int2((tr.size >> level) - 1))), level - 1).rg
                                       : tr.cb.read(uint2(mc & ((COARSE_N >> level) - 1)), uint(k - 1), uint(level - 1)).rg;
                    float2 lo = bandLo(bd.x);
                    if (k > 0 || lo.y > 0.0 || disc) clear = min(yA, yB) >= lo.x && max(yA, yB) < bd.y;
                }
            }
            if (clear) {
                lastAxis = tb.x < tb.y ? 0 : 2;
                nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
                t = tCell;
                level = min(level + 1, MAX_LEVEL);
                continue;
            }
            level--;
            continue;
        }

        // One column, solid below H except in its gaps (the principal one, and on the 1-voxel map up to two more below
        // it); a ray passing over it moves on as a skip would. A ray above the column never needs its gaps, which lie
        // at least two voxels under its top: coarse levels read them only for a ray inside.
        float wm = morphAt(tr, k, (float2(lc) + 0.5) * s);
        float4 ext;
        bool got;
        float4 col = columnMorphE(tr, k, lc, wm, ext, got);
        float H = col.x, gl = col.y, gh = col.z;
        bool inGap = yA >= gl && yA < gh;
        // Under the principal gap: the 1-voxel map's extra gaps, a coarse giant column's air under its crown.
        if (!inGap && k <= GAP2_LEVELS && yA < col.w) {
            float4 e = ext;
            if (!got && k == 0) {
                e = extraGaps(tr.gx.read(uint2(lc)));
                if (wm > 0.0) e.yw = mix(e.yw, e.xz, wm);
            } else if (!got) e = float4(secondGapW(tr, k, lc, wm), 0, 0);
            bool g1 = yA >= e.x && yA < e.y, g2 = yA >= e.z && yA < e.w;
            inGap = g1 || g2;
            gl = g1 ? e.x : e.z;
            gh = g1 ? e.y : e.w;
        }
        h.level = k;
        if (yA < H && !inGap) {
            h.hit = true;
            h.t = t;
            h.normal = lastAxis == 0 ? int3(d.x > 0 ? -1 : 1, 0, 0) : int3(0, 0, d.z > 0 ? -1 : 1);
            h.voxel = int3(lc.x, int(floor(yA)), lc.y);
            return h;
        }
        float floorY = inGap ? gl : H;
        if (d.y < 0 && yB < floorY) {
            h.hit = true;
            h.t = (floorY - o.y) * inv.y;
            h.normal = int3(0, 1, 0);
            h.voxel = int3(lc.x, int(ceil(floorY)) - 1, lc.y);
            return h;
        }
        if (inGap && d.y > 0 && yB > gh) {
            h.hit = true;
            h.t = (gh - o.y) * inv.y;
            h.normal = int3(0, -1, 0);
            h.voxel = int3(lc.x, int(gh), lc.y);
            return h;
        }
        lastAxis = tb.x < tb.y ? 0 : 2;
        nudge = select(float2(0), sign(d.xz) * 0.01, tb <= tCell + 1e-4);
        t = tCell;
        level = 1;
    }
    // Out of steps, not out of the world.
    h.iters = maxIter;
    return h;
}
