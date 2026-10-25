// The land, its caves, then its islands over them.
static inline Col terrainColumnC(int2 w, uint seed, int s, device const CaveRec* caves, device const IsleRec* T) {
    Col c = terrainColumnBareS(w, seed, s);
    CaveRec r;
    if (s <= CAVE_MAX_S && caveFind(caves, float2(w) + 0.5, r)) c = caveColumn(c, w, r, seed, s);
    return islandColumn(c, w, seed, s, T);
}
