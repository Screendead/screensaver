static inline float snowlineAt(float temp);
static inline float altitudeTemp(float temp, float H);

// Finds a cave on a cliff at P: the steepest rise along the fall line through it (a riser of a range's cliff band, a
// karst tower's or tepui's wall, an escarpment), its foot the floor, rock behind deep enough to roof it.
static bool caveTry(float2 P, uint h, uint seed, thread CaveRec& r) {
    Climate cl = climate(P, seed);
    if (cl.jungle > 0.1) return false;
    float U = megaLift(P + seedOffset(seed), lagoonAmount(cl), seed).U;
    if (hashf(h) >= (U > 30.0 ? 0.8 : 0.12)) return false;
    if (towerNear(P, CAVE_REACH + 16.0, seed)) return false;
    float hx = terrainColumnBase(int2(floor(P + float2(6, 0))), seed).h - terrainColumnBase(int2(floor(P - float2(6, 0))), seed).h;
    float hz = terrainColumnBase(int2(floor(P + float2(0, 6))), seed).h - terrainColumnBase(int2(floor(P - float2(0, 6))), seed).h;
    float2 gr = float2(hx, hz) / 12.0;
    if (length(gr) < 0.4) return false;
    // The terraced field's differences lean to the grid's axes; a cave need not.
    float2 u = normalize(gr), v;
    float tw = 0.35 * (hashf(pcg(h + 8u)) - 0.5);
    u = float2(u.x * cos(tw) - u.y * sin(tw), u.x * sin(tw) + u.y * cos(tw));
    v = float2(-u.y, u.x);
    // The profile up the fall line, every two voxels from 16 below P to 16 above.
    float pr[17];
    for (int i = 0; i < 17; i++) {
        Col q = terrainColumnBase(int2(floor(P + u * float(2 * i - 16))), seed);
        if (q.gh > q.gl) return false;
        pr[i] = q.h;
    }
    float best = 0.0;
    int bi = -1;
    for (int i = 1; i < 14; i++) {
        float rise = pr[i + 3] - pr[i];
        if (rise > best) { best = rise; bi = i; }
    }
    // A sheer face, not a staircase slope: a stair's mouth would open as a ragged notch down its steps.
    float sheer = max(max(pr[bi + 1] - pr[bi], pr[bi + 2] - pr[bi + 1]), pr[bi + 3] - pr[bi + 2]);
    if (best < 11.0 || sheer < 7.0) return false;
    // The foot: the last sample before the rise starts.
    int j = bi;
    while (j < bi + 2 && pr[j + 1] - pr[j] < 2.0) j++;
    float yF = pr[j];
    if (yF < SEA + 1.5 || pr[j - 1] > yF + 1.0) return false;
    float tm = float(2 * j - 16) + 1.0;
    float2 m = floor(P + u * tm) + 0.5;
    // The mouth's line: the first column up from the foot that stands well over it, so the face holds the arch.
    for (int t = 0; t < 6; t++) {
        float2 mt = floor(m + u * float(t)) + 0.5;
        if (terrainColumnBase(int2(floor(mt)), seed).h >= yF + 8.0) { m = mt; break; }
    }
    bool grotto = U > 250.0 && best > 30.0 && hashf(pcg(h + 4u)) < 0.3;
    float rr = hashf(pcg(h + 3u));
    float W = grotto ? 10.0 + 6.0 * rr : 4.0 + 7.0 * rr;
    float Hc = clamp(W * (1.1 + 0.6 * hashf(pcg(h + 5u))), 6.0, grotto ? 24.0 : 15.0);
    // Under three voxels of roof at the mouth's crown, so its arch shows whole on the face.
    float face = 1e9;
    for (int t = 0; t < 3; t++) face = min(face, terrainColumnBase(int2(floor(m + u * float(t))), seed).h);
    Hc = min(min(Hc, best - 3.0), face - yF - 3.0);
    if (Hc < 5.0) return false;
    float Lmax = grotto ? 56.0 : 18.0 + 26.0 * hashf(pcg(h + 6u));
    float L = 0.0;
    for (float dd = 4.0; dd <= Lmax; dd += 4.0) {
        Col q = terrainColumnBase(int2(floor(m + u * dd)), seed);
        if (q.gh > q.gl || q.h < yF + Hc * (1.0 - 0.45 * smoothstep(0.25, 1.0, dd / Lmax)) + 3.0) break;
        L = dd;
    }
    if (L < 10.0) return false;
    // Rock either side of the mouth.
    for (int t = 0; t < 2; t++) {
        float s2 = t == 0 ? 1.0 : -1.0;
        Col q = terrainColumnBase(int2(floor(m + u * 4.0 + v * (s2 * W * 0.8))), seed);
        if (q.h < yF + 0.6 * Hc + 3.0) { W *= 0.6; Hc = min(Hc, max(5.0, W * 1.2)); }
    }
    // Level ground before the mouth across its width: a ledge narrower than the mouth would roof the ground beside it.
    for (int t = 0; t < 2; t++) {
        float s2 = t == 0 ? 1.0 : -1.0;
        Col q = terrainColumnBase(int2(floor(m - u * 2.0 + v * (s2 * W * 0.7))), seed);
        if (q.h > yF + 2.0) { W *= 0.6; Hc = min(Hc, max(5.0, W * 1.2)); }
    }
    // Smaller mouths read as notches in the slope.
    if (W < 4.5 || Hc < 6.0) return false;
    // Nothing of it may leave P's cell: columns ask their own cell only.
    int2 cell = int2(floor(P / CAVE_CELL));
    for (; L >= 10.0; L -= 4.0) {
        bool inside = true;
        for (int k = 0; k < 4; k++) {
            float2 e = m + u * ((k & 1) != 0 ? L + 2.0 : -4.0) + v * ((k & 2) != 0 ? 1.0 : -1.0) * (1.5 * W + 3.0);
            inside = inside && all(int2(floor(e / CAVE_CELL)) == cell);
        }
        if (inside) break;
    }
    if (L < 10.0) return false;
    r.m = m; r.u = u; r.yF = yF; r.W = W; r.L = L; r.Hc = Hc; r.h = h;
    r.flags = CV_ON | (grotto && W >= 8.0 ? CV_GROTTO : 0u) | (U > 30.0 ? 0u : CV_LOW);
    // Someone lives in about one cave in three whose floor is level with the ground in front, below the snow.
    float snow = snowlineAt(altitudeTemp(cl.temp, yF));
    if (W >= 4.0 && Hc >= 6.0 && L >= 14.0 && yF < snow - 10.0 && abs(pr[j - 1] - yF) <= 1.0 && hashf(pcg(h + 7u)) < 0.35)
        r.flags |= CV_HOME;
    return true;
}

// One thread per world cell info.xy + gid (info.z: the first row of this dispatch).
kernel void placeCaves(device CaveRec* out [[buffer(0)]],
                       constant int4& info [[buffer(1)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint2 gid = g + uint2(0, uint(info.z));
    if (gid.x >= uint(CAVE_WIN) || gid.y >= uint(CAVE_WIN)) return;
    uint seed = uint(info.w);
    int2 cell = info.xy + int2(gid);
    // A slot already holding this cell keeps it (World.Buffer starts every slot with no cell).
    if (all(out[caveSlot(cell)].cell == cell)) return;
    CaveRec r;
    r.cell = cell; r.m = 0; r.u = 0; r.yF = 0; r.W = 0; r.L = 0; r.Hc = 0; r.h = 0; r.flags = 0;
    uint hc = hash2i(cell, seed + 900u);
    for (int a = 0; a < 2; a++) {
        uint h = pcg(hc + uint(a) * 7919u);
        float2 P = floor((float2(cell) + 0.25 + 0.5 * float2(hashf(pcg(h + 1u)), hashf(pcg(h + 2u)))) * CAVE_CELL) + 0.5;
        CaveRec t = r;
        if (caveTry(P, h, seed, t)) { r = t; break; }
    }
    out[caveSlot(cell)] = r;
}
