struct Surface {
    float3 albedo;
    float3 tread;   // the column's top colour, used to prefilter risers at distance
    float3 face;    // the colour a steep face of this column shows from afar: rock, snow-flecked over the snowline
    float4 nh;      // neighbour heights: -x, +x, -z, +z
    float H;
    int mat;        // 0 terrain, 1 leaf, 2 wood, 3 pale bark
    float canopy;   // ambient left under a canopy overhead
    float2 slope;   // heightfield gradient of fine relief on top faces
    float riser;    // drop below this face, for side faces
    float autumn;
    float level;    // 1 shades the face as dead-level ground: up normal, no corner AO
    bool gapNear;   // this column or the one in front of the face has a rock gap (on the 1-voxel map: any 4-neighbour)
    float2 gapSpan; // the lowest floor and highest roof of those gaps (extra gaps lie lower still: floor 0 then)
    float pale;     // sand, salt crust or snow
    float wall;     // megaWall: among the great walls
    float3 nrm;     // the plane a structure's roof voxel or a tower's wall shades as (zero: the face normal)
    float3 emit;    // fire, lanterns and lit windows
    float cave;     // how much of the sky and bounce light a cave's depth keeps from this face (0 outside caves)
    float caveG;    // a face caveDress may dress (shade()): its green share, plus 2 under a gap; -1 none
    float up;       // leaves: height up the crown, 0 at its foot, 1 at its top
    bool vine;      // a giant's hanging vines
    float2 isle;    // on the 1-voxel map an island's body: (1 root, 2 water, 4 rock under its lip, 8 any; its gap's top), for its glow
    bool isleUnder; // under an island: the sky it hides is isleSkyOcc's
    float gy;       // the height the ground round it is judged from (groundL): above any island for an island's own rock
    float bio;      // night glow share of this ground (-1: a giant's, worked out in bioLight)
    float bioG;     // the groves' noise, which also bends the glow's pulse
    float bioY;     // a trunk's foot
    float bioJ;     // jungle weight, which gathers the glow into clusters
    uint bioNb;     // ground beside a tree column: -x, +x, -z, +z bits
};
