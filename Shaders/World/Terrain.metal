// The land without its islands or caves (placing the giants, which the islands then keep clear of); terrainColumnC adds
// them.
static inline Col terrainColumnBareS(int2 w, uint seed, int s) { return towerColumn(terrainColumnBase(w, seed, s), w, seed, s); }
static inline Col terrainColumnBare(int2 w, uint seed) { return terrainColumnBareS(w, seed, 1); }
