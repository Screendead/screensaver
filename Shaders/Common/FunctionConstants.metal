// Function constant 1: false compiles the towers out of sceneFragment, for frames with none in view. Compiled in, even
// unused, they slow every pixel's shading.
constant bool sceneTowers [[function_constant(1)]];

// The band test in the DDA: compiled in only while a canopy band is on the map (Renderer picks the pipeline); off in
// every pipeline built without the constant.
constant bool BAND_FC [[function_constant(0)]];
constant bool BAND_SKIP = is_function_constant_defined(BAND_FC) && BAND_FC;
// Renderer picks the pipeline per tile of the frame (Renderer.sceneTile) by what of the islands may show there.
// The islands' shade and sky occlusion: compiled in only where an island within isleShadeReach casts them.
constant bool ISLE_FC [[function_constant(3)]];
constant bool SCENE_ISLES = is_function_constant_defined(ISLE_FC) && ISLE_FC;
// Close by an island within isleShadeReach (its column of air), with its shade: the ground under it, its falls, streams,
// canopies and glow.
constant bool NEAR_FC [[function_constant(6)]];
constant bool SCENE_NEAR = SCENE_ISLES && is_function_constant_defined(NEAR_FC) && NEAR_FC;
// The islands' surfaces (their rock, roots and vines, air that is no shelter's roof): wherever an island at any distance
// could show.
constant bool SHAPE_FC [[function_constant(4)]];
constant bool ISLE_SHAPE = SCENE_NEAR || (is_function_constant_defined(SHAPE_FC) && SHAPE_FC);
// The islands' mist (isleMist), close by them.
constant bool MIST_FC [[function_constant(5)]];
constant bool SCENE_MIST = SCENE_NEAR && is_function_constant_defined(MIST_FC) && MIST_FC;

// Compiled in only while the sea of cloud is out (SEA_FC, Renderer picks the pipeline): held off in the scene and vine
// passes it costs every pixel about 0.3 ms (pin606n, dusk).
constant bool SEA_FC [[function_constant(8)]];
constant bool SCENE_SEA = is_function_constant_defined(SEA_FC) && SEA_FC;

constant bool STAR_FC [[function_constant(7)]];
constant bool STAR_TAA = is_function_constant_defined(STAR_FC) && STAR_FC;

constant bool DITHER_FC [[function_constant(2)]];
