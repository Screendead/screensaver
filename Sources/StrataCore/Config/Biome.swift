import Foundation

// The world as one biome: pinnedBiome in Shaders/Saver.metal reads it back from the seed.
enum Biome {
    // Index + 1 is the biome's number in the shader.
    static let names = ["Grassland", "Autumn Woods", "Pine & Snow", "Desert", "Painted Hills", "Salt Flats", "Lagoons", "Jungle", "Old Growth"]

    static func number(_ name: String?) -> Int {
        guard let name else { return 0 }
        if let n = Int(name), (0...names.count).contains(n) { return n }
        return (names.firstIndex { $0.caseInsensitiveCompare(name) == .orderedSame } ?? -1) + 1
    }

    static func pinned(_ seed: UInt32) -> Int { seed >> 24 == 0xFF ? Int((seed >> 20) & 15) : 0 }

    // Seeds whose top byte is 0xFF are reserved for this, so random seeds stay below it.
    static let maxSeed: UInt32 = 0xFEFF_FFFF
    static func seed(_ base: UInt32, biome: Int) -> UInt32 {
        biome == 0 ? base : 0xFF00_0000 | UInt32(biome) << 20 | (base & 0xF_FFFF)
    }
}
