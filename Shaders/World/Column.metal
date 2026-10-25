// A generated column: solid where y < h, except inside its gaps. [gl, gh) is the principal (highest) gap, [lo1, hi1) and
// [lo2, hi2) up to two more below it, highest first (empty when hi <= lo). carved: the gaps are rock carved by the mega
// relief (overhangs, caves, bridges), whose roofs rest on any neighbour rather than all four; canyon arches are not.
// island: the principal gap is a floating island's air; leaf: its canopy's first leaf (0 none); isk: what its column
// holds besides rock (ISK_*).
struct Col { float h, gl, gh, lo1, hi1, lo2, hi2; bool carved; uint reef; bool island; float leaf; uint isk; bool cave = false; };

// Extra gaps (World gx, rgba16Uint): x lo1, y hi1, z lo2, w hi2 in the low 11 bits (heights stay under 2048); the top 5
// bits of z hold flags (bit 0: carved). An empty gap is lo = hi = 0.
constant uint GAP_BITS = 0x7ffu;
constant uint GAP_CARVED = 1u;
constant uint GAP_ISLAND = 4u;
// An island's air is at least this tall on the 1-voxel map (its keel clears the ground by 60); no other rock gap is.
constant float ISLE_AIR = 59.0;

static inline uint4 packGaps(Col c) {
    uint flags = (c.carved ? GAP_CARVED : 0u) | (c.island ? GAP_ISLAND : 0u);
    bool g1 = c.hi1 > c.lo1, g2 = c.hi2 > c.lo2;
    return uint4(g1 ? uint(c.lo1) : 0u, g1 ? uint(c.hi1) : 0u, (g2 ? uint(c.lo2) : 0u) | (flags << 11), g2 ? uint(c.hi2) : 0u);
}

// (lo1, hi1, lo2, hi2) of packed extra gaps.
static inline float4 extraGaps(uint4 g) { return float4(g & GAP_BITS); }
static inline uint gapFlags(uint4 g) { return g.z >> 11; }

// Lifted land is re-terraced into cliff bands, so mountain faces read as contours, not a stair lattice. Karst keeps its
// sheer pillar walls.
static inline float megaTerrace(float h, float U, float wK, float td, uint seed) {
    float hm = h + U;
    float tdm = mix(td, 14.0, smoothstep(40.0, 400.0, U));
    // Beds differ in thickness from one formation to the next, so a tall wall is not one even ruled stripe.
    tdm *= mix(1.0, 0.6 + 0.8 * hashf(hash2i(int2(int(floor(hm / 70.0)), 7), seed + 260u)), smoothstep(60.0, 200.0, U));
    float hn3 = hm / tdm, lv3 = floor(hn3);
    // Tall relief gets short risers between wide treads: a long ramp of single steps reads as a stair lattice.
    float e3 = mix(0.6, 0.85, smoothstep(100.0, 400.0, U));
    return min(mix(hm, (lv3 + smoothstep(e3, 1.0, hn3 - lv3)) * tdm, smoothstep(0.5, 30.0, U) * (1.0 - wK)), TERRAIN_CEIL);
}

static inline Col terrainColumnBase(int2 w, uint seed, int s = 1) {
    float2 p = float2(w) + seedOffset(seed);
    float2 warp = float2(fbm2(p / 900.0, 2, seed + 1), fbm2(p / 900.0 + 37.1, 2, seed + 2)) * 300.0;
    float c = saturate(fbm2((p + warp) / 2000.0, 3, seed + 3) * 1.4 + 0.5);
    float hi = smoothstep(0.35, 0.75, c);
    float rough = fbm2((p + warp * 0.3) / 350.0, 3, seed + 4) * 0.5 + 0.5;
    float ridge = ridged2((p + warp * 0.2) / 520.0, 3, seed + 5);
    float r = mix(rough, ridge, hi);
    float h = mix(16.0, 62.0, c) + mix(26.0, 115.0, hi) * r;
    // Crenellated ridgelines instead of soft domes.
    h += (ridged2((p + warp * 0.4) / 110.0, 3, seed + 14) - 0.45) * mix(0.0, 8.0, smoothstep(0.3, 0.7, hi));
    Climate cl = climate(float2(w), seed);
    // Island country lies low: hills are pressed down toward meadow, so islands ride high over it with room beneath.
    if (cl.isle > 0.0) {
        float L = 62.0 + 20.0 * fbm2(p / 650.0, 2, seed + 303u);
        h = mix(h, h > L ? L + (h - L) * 0.15 : h, cl.isle);
    }
    // Soft terracing everywhere: risers cluster into short cliff bands between wide treads,
    // so slopes read as contour lines instead of a 1:1 stair lattice.
    float td = mix(4.0, 8.0, saturate(perlin2(p / 900.0, seed + 13) * 0.8 + 0.5));
    float hn2 = h / td;
    float lv = floor(hn2);
    h = (lv + smoothstep(mix(0.55, 0.9, hi), 1.0, hn2 - lv)) * td;

    // Canyon country: in hot, dry regions the land steps into flat-topped mesas with sheer risers.
    float salt = saltAmount(cl);
    float canyon = smoothstep(0.35, 0.7, desertAmount(cl)) * (1.0 - salt);
    float tier = 13.0 + 4.0 * perlin2(p / 700.0, seed + 7);
    float hn = h / tier;
    float level = floor(hn);
    float mesa = (level + smoothstep(0.62, 0.9, hn - level)) * tier;
    h = mix(h, mesa, canyon);

    // Painted hills: rounded mounds from billowed Perlin noise, untreaded so every voxel step shows a bed,
    // rising from flat floors where the billow's creases are filled in.
    float painted = paintedAmount(cl);
    // Low islands in lagoons, salt basins and painted country: a short-wavelength warp and roughening, or smooth
    // domes floor into concentric rings that look built.
    float lagoon = lagoonAmount(cl);
    float eroded = max(max(lagoon, smoothstep(0.0, 0.3, salt)), smoothstep(0.0, 0.6, painted));
    float2 ew = eroded > 0.0 ? float2(perlin2(p / 38.0, seed + 120u), perlin2(p / 38.0 + 19.1, seed + 121u)) * 14.0 * eroded : 0.0;
    if (painted > 0.0) {
        float b = abs(fbm2((p + warp * 0.3 + ew) / 160.0, 3, seed + 111u));
        float amp = mix(22.0, 52.0, saturate(perlin2(p / 600.0, seed + 112u) * 0.8 + 0.5));
        float mound = smoothstep(0.07, 0.5, b);
        h = mix(h, paintedFloor(p, seed) + amp * mound * (2.0 - mound) * 0.75, smoothstep(0.0, 0.6, painted));
    }

    // Long meandering valleys with flat water floors: the camera follows these.
    float river = abs(fbm2((p + warp * 0.6) / 1400.0, 3, seed + 6));
    float valley = smoothstep(0.02, 0.30 + 0.10 * hi, river);
    h = mix(SEA - 3.0, h, valley);
    // Lagoon shallows: the land sinks to a sandy bed rolling a few voxels either side of the surface,
    // so sand bars and islets break it and channels run deeper between them.
    float lagBed = 0.0;
    uint reef = 0u;
    if (lagoon > 0.0) {
        float bed = fbm2((p + warp * 0.3 + ew) / 210.0, 3, seed + 15u) + 0.25 * perlin2(p / 600.0, seed + 16u);
        lagBed = SEA - 2.5 + bed * 14.0;
        h = mix(h, lagBed, lagoon);
    }
    Mega mg = megaLift(p, lagoon, seed);
    float U = mg.U;
    float h0 = h;
    if (U > 0.5) h = megaTerrace(h0, U, mg.wK, td, seed);
    // The cap rock's height over a lip column: the plateau behind it by the same formula, so the lip meets it level.
    float hCap = mg.lipU > U + 6.0 ? floor(megaTerrace(h0, mg.lipU, mg.wK, td, seed)) : 0.0;
    // A pan's rim would otherwise chop a massif flat.
    salt *= 1.0 - smoothstep(10.0, 80.0, U);
    if (eroded > 0.0) h += eroded * (1.6 * perlin2(p / 21.0 + ew / 14.0, seed + 122u) + 0.9 * perlin2(p / 9.0, seed + 125u)) * smoothstep(SEA + 40.0, SEA + 2.0, h);
    h = floor(max(h, 2.0));

    // Salt pans: the land is cut down to old shorelines, low benches climbing out of a dead-flat floor,
    // and rivers that reach the pan end at its edge.
    bool pool = false;
    if (salt > 0.02) {
        float bench = 3.0;
        float rs = salt + 0.05 * perlin2(p / 45.0, seed + 123u) + 0.025 * perlin2(p / 16.0, seed + 124u);
        float rim = PAN + floor(70.0 * pow(1.0 - smoothstep(0.1, 0.85, rs), 1.5) / bench) * bench;
        h = min(h, rim);
        h = max(h, PAN - floor(6.0 * (1.0 - smoothstep(0.6, 0.85, salt))));
        if (h == PAN && poolNoise(p, seed) > poolLevel(salt)) { h = PAN - 1.0; pool = true; }
    }
    // Lake beds are one flat shelf: submerged terraces read as stepping stones under the surface.
    if (h < SEA && !pool) h = floor(mix(floor(SEA) - 2.0, h, lagoon) + 0.5);
    // Reefs: a flat behind a crest along the shallows (a fringing reef, its crest at the drop-off), patch reefs and
    // bommies rising from the deeper bed, colonies of coral on all of them; never breaking the surface.
    if (lagoon > 0.45 && !pool && h < SEA && h >= SEA - 12.0 && U <= 0.5) {
        float reg = reefRegion(p, lagoon, seed);
        if (reg > 0.0) {
            float D = SEA - lagBed;
            float top = floor(SEA);
            float fringe = smoothstep(-0.05, 0.08, seaNoise(p / 180.0, seed + 149u) + 0.1);
            float flat = smoothstep(0.4, 1.0, D) * (1.0 - smoothstep(3.6, 5.0, D)) * fringe;
            float crest = exp(-(D - 3.9) * (D - 3.9) / 0.6) * fringe;
            float blob = seaNoise(p / 34.0, seed + 143u) + 0.35 * seaNoise(p / 11.0, seed + 148u);
            float patch = smoothstep(0.16, 0.32, blob) * smoothstep(4.0, 6.0, D);
            // Spur and groove: the flat rolls a voxel up and down across the shallows.
            float groove = step(0.08, seaNoise(p / 9.0, seed + 146u) + 0.5 * seaNoise(p / 4.0, seed + 147u));
            float hr = max(max(mix(h, top - 2.0 - groove, flat), mix(h, top - 1.0, crest)), mix(h, top - 2.0 - floor(2.5 * (1.0 - smoothstep(0.24, 0.4, blob))), patch));
            h = max(h, floor(mix(h, hr, smoothstep(0.3, 0.7, reg)) + 0.5));
            reef = uint(saturate(reg) * 15.0 + 0.5);
            if (reg > 0.5 && s <= 2 && h >= REEF_FLOOR) {
                Coral cr = coralAt(w, seed);
                h = max(h, min(h + cr.rise, top - 1.0));
                if (cr.rise > 0.0) reef |= 16u | (uint(cr.kind) << 5) | (cr.tip > 0.5 ? 128u : 0u) | (cr.h << 8);
            }
        }
    }

    // Arches and undercuts: noise blobs hollow out a band low in each mesa tier.
    float gl = 0, gh = 0;
    if (canyon > 0.2 && U <= 0.5) {
        float blob = fbm2(p / 34.0, 3, seed + 8);
        float g = (blob - 0.2) * canyon;
        if (g > 0.0) {
            float base = max(level * tier - tier, SEA + 1.0);
            gl = floor(base + 2.0 + 2.0 * perlin2(p / 400.0, seed + 10));
            gh = gl + floor(min(g * 40.0, tier * 0.8));
            if (gh > h - 3.0 || gh - gl < 3.0 || gl < SEA + 1.0) { gl = 0; gh = 0; }
        }
    }
    Col col;
    col.lo1 = 0; col.hi1 = 0; col.lo2 = 0; col.hi2 = 0;
    col.carved = false;
    col.island = false;
    col.leaf = 0.0;
    col.isk = 0u;
    // A caprock lip: the column rises to the cap, hollow beneath it down to its own ground. The underside is the cap
    // rock's bedding plane, level: a sloping one would be an upturned staircase, every step a lit line. The wall below
    // runs up under it into a rock shelter.
    if (hCap > h + 6.0) {
        float t0 = 12.0 + 10.0 * saturate(perlin2(p / 400.0, seed + 263u) + 0.5);
        float ceil0 = floor(hCap - t0);
        // The shelter is a recess of bounded height under the slab: below it the band is the escarpment's sheer wall.
        float V = floor(18.0 + 40.0 * saturate(perlin2(p / 300.0, seed + 264u) * 0.9 + 0.5));
        float floor0 = max(h, ceil0 - V);
        if (ceil0 - floor0 >= 3.0) { gl = floor0; gh = ceil0; col.carved = true; }
        h = hCap;
    }
    // A notch round a karst tower's foot: at the water line for a tower standing in water, else on its talus.
    if (mg.notchH >= 3.0 && gh <= gl) {
        float lo = h0 < SEA ? floor(SEA) - 1.0 : floor(h0 + mg.notchFoot) + 1.0;
        float hi = lo + floor(mg.notchH);
        if (hi < h - 8.0 && hi - lo >= 3.0) { gl = lo; gh = hi; col.carved = true; }
    }
    col.h = h; col.gl = gl; col.gh = gh;
    col.reef = reef;
    return col;
}
