// Snow above this height: cold country as ever, while warm provinces lift it far up their ranges (hot ones never
// snow), so massifs are rock and forest under a cap rather than white sheets.
static inline float snowlineAt(float temp) {
    return mix(160.0, 92.0, smoothstep(0.42, 0.18, temp)) + 1400.0 * smoothstep(0.42, 0.75, temp);
}

// Forest gives out well under a high snowline, leaving a band of bare rock.
static inline float treelineAt(float snowline) { return snowline - 150.0 * smoothstep(200.0, 700.0, snowline); }

// Air cools up a range: deciduous, then conifer, then bare rock and snow.
static inline float altitudeTemp(float temp, float H) { return temp - 0.22 * smoothstep(150.0, 1400.0, H); }

static inline float coniferAmount(Climate c) { return smoothstep(0.50, 0.32, c.temp) * (1.0 - 0.8 * autumnAmount(c)); }

// Whole groves of birch within the autumn woods.
static inline float birchAmount(float2 w, uint seed) {
    return smoothstep(0.22, 0.38, perlin2((w + seedOffset(seed)) / 110.0, seed + 95u));
}
