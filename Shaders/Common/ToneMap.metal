static inline float3 tmap(float3 v) { return v / (1.0 + max(max(v.r, v.g), v.b)); }
static inline float3 itmap(float3 v) { return v / max(1.0 - max(max(v.r, v.g), v.b), 1e-4); }

// A part-cover blend as the temporal resolve averages jittered samples: in tone-mapped space, so a cover of thin
// strands matches what the strands themselves resolve to.
static inline float3 coverMix(float3 a, float3 b, float w) { return itmap(mix(tmap(a), tmap(b), w)); }
