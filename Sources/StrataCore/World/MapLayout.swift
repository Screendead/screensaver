// The 1-voxel map and the tables placed with it.
package enum MapLayout {
    package static let size = 2048
    package static let cpuLevel = 2
    // Must match SITE_GRID, MAX_LAMPS and MAX_PARTS (the Site struct) in Shaders/Saver.metal.
    package static let siteGrid = 18, siteCell = 128, maxLamps = 8, maxParts = 32
    package static let siteStride = 32 + 16 * maxLamps + 16 * maxParts
    // Must match GIANT_WIN and GIANT_CELL in Shaders/Saver.metal.
    package static let giantWin = 24, giantCell = 1536.0
    // Must match CAVE_WIN and CAVE_CELL in Shaders/Saver.metal; the cave table follows the giant table in its buffer.
    package static let caveWin = 96, caveCell = 256.0, caveReach = 64.0
    package static let caveOffset = giantWin * giantWin * MemoryLayout<GiantRec>.stride
}
