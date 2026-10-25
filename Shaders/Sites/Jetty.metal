// A jetty from the shore along `dir` (a unit grid step) into the water, with a lantern post and moored boats.
static bool buildJetty(texture2d<float, access::read> hm, thread SiteGen& G, int2 c, uint h, int2 dir, int maxReach) {
    int k = 0;
    for (; k < maxReach; k++) if (colH(hm, c + dir * k) < SEA) break;
    if (k == maxReach || k == 0) return false;
    int2 shore = c + dir * (k - 1);
    if (colH(hm, shore) > SEA + 3.0 || colArch(hm, shore)) return false;
    int run = 0;
    for (int m = 1; m <= 16; m++) {
        int2 q = shore + dir * m;
        if (colH(hm, q) >= SEA || colArch(hm, q)) break;
        run++;
    }
    if (run < 5 || G.nl >= MAX_LAMPS || G.np > MAX_PARTS - 3) return false;
    int len = min(run - 1, 6 + int(h % 9u));
    int2 perp = int2(abs(dir.y), abs(dir.x));
    int w = (h >> 5 & 3u) == 0u ? 2 : 1;
    int2 head = shore + dir * len;
    int2 r0 = min(shore, head), r1 = max(shore, head) + 1 + perp * (w - 1);
    int axis = dir.y != 0 ? F_AXISZ : 0;
    int lamp = addLamp(G, float3(float(head.x - c.x) + 0.5, 22.6, float(head.y - c.y) + 0.5), L_LANTERN, 0);
    Part post = makePart(P_BOX, M_TIMBER, head - c, int2(1), 22, 0, 0, F_WATER | ((lamp + 1) << 8));
    addPart(G, post, float(partReach(post)));
    Part deck = makePart(P_DECK, M_PLANK, r0 - c, r1 - r0, 21, 0, int(h & 0x7fffu), axis | F_WATER);
    addPart(G, deck, float(partReach(deck)));
    int boats = int((h >> 9) % 3u);
    for (int b = 0; b < boats && G.np < MAX_PARTS; b++) {
        int side = ((h >> (12 + b)) & 1u) != 0u ? -2 : 1 + w;
        int2 bs = shore + dir * (2 + 3 * b) + perp * side;
        int2 be = bs + dir * 2;
        bool wet = true;
        for (int m = 0; m < 3; m++) wet = wet && colH(hm, bs + dir * m) < SEA - 0.5 && !colArch(hm, bs + dir * m);
        if (!wet) continue;
        int2 b0 = min(bs, be);
        Part boat = makePart(P_BOAT, M_TIMBER, b0 - c, max(bs, be) + 1 - b0, 0, 0, 0, axis | F_WATER);
        addPart(G, boat, float(partReach(boat)));
    }
    return true;
}
