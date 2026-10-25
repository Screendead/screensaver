@testable import Strata
@testable import StrataCore
import XCTest

// The sizes, offsets and constants the CPU shares with the shader, read from the compiled shader.
final class ShaderLayoutTests: XCTestCase {
    // Each shader constant (in probeConstants' order) beside the CPU value that must equal it.
    private let constants: [(String, Double)] = [
        ("SEA", Sight.sea),
        ("SITE_GRID", Double(MapLayout.siteGrid)), ("SITE_CELL", Double(MapLayout.siteCell)),
        ("MAX_LAMPS", Double(MapLayout.maxLamps)), ("MAX_PARTS", Double(MapLayout.maxParts)),
        ("GIANT_WIN", Double(MapLayout.giantWin)), ("GIANT_CELL", MapLayout.giantCell),
        ("CAVE_WIN", Double(MapLayout.caveWin)), ("CAVE_CELL", MapLayout.caveCell), ("CAVE_REACH", MapLayout.caveReach),
        ("ISLE_WIN", Double(Islands.win)), ("ISLE_MWIN", Double(Islands.mwin)), ("ISLE_SLOTS", Double(Islands.slots)),
        ("ISLE_TABLE", Double(Islands.tableSize)), ("ISLE_MIST_PUFFS", Double(Islands.mistPuffs)),
        ("TOWER_CELL", Towers.cell), ("TOWER_MEMBERS", Double(Towers.members)), ("TOWER_TABLE", Double(Towers.tableSize)),
        ("VINE_SLOTS", Double(Vines.slots)), ("VINE_W", Double(Vines.window)),
        ("MAX_FLOCKS", Double(Wildlife.maxFlocks)), ("MAX_CLUSTERS", Double(Wildlife.maxClusters)),
        // The flight's ceiling is 80 under the cloud base.
        ("CLOUD_Y0", FlightCamera.ceiling + 80),
        ("MOTE_N", cbrt(Double(Renderer.moteCount))), ("LAMP_TILES", Double(Renderer.lampTiles)), ("BIO_SUN", Double(Renderer.bioSun)),
    ] + Renderer.lampReach.enumerated().map { ("LAMP_R[\($0.offset)]", $0.element) }

    func testConstantsMatchTheShader() throws {
        let gpu = try ShaderProbes.get().run("probeConstants", [Float]?.none, count: constants.count, as: Float.self)
        for ((name, cpu), g) in zip(constants, gpu) {
            XCTAssertEqual(Float(cpu), g, accuracy: abs(g) * 1e-6, name)
        }
    }

    // Each size or field offset (in probeLayouts' order) beside the CPU's.
    private var layouts: [(String, Int)] {
        let f = MemoryLayout<Frame>.self, g = MemoryLayout<GiantRec>.self, c = MemoryLayout<CaveRec>.self, i = MemoryLayout<IsleRec>.self
        let s = MemoryLayout<ShadowBake>.self, v = MemoryLayout<VineBake>.self
        return [
            ("Frame", f.stride), ("Frame.sunDir", f.offset(of: \.sunDir)!), ("Frame.mapInfo", f.offset(of: \.mapInfo)!),
            ("Frame.lights", f.offset(of: \.lights)!), ("Frame.refs", f.offset(of: \.refs)!), ("Frame.birds", f.offset(of: \.birds)!),
            ("Frame.lod", f.offset(of: \.lod0)!), ("Frame.fade", f.offset(of: \.fade)!), ("Frame.shadow", f.offset(of: \.shadow)!),
            ("Frame.slowCloud", f.offset(of: \.slowCloud)!), ("Frame.planet", f.offset(of: \.planet)!), ("Frame.planetPx", f.offset(of: \.planetPx)!),
            ("GiantRec", g.stride), ("GiantRec.c", g.offset(of: \.c)!), ("GiantRec.G", g.offset(of: \.g)!), ("GiantRec.yT", g.offset(of: \.yT)!),
            ("GiantRec.h", g.offset(of: \.h)!), ("GiantRec.kind", g.offset(of: \.kind)!),
            ("CaveRec", c.stride), ("CaveRec.m", c.offset(of: \.m)!), ("CaveRec.u", c.offset(of: \.u)!), ("CaveRec.yF", c.offset(of: \.yF)!),
            ("CaveRec.W", c.offset(of: \.w)!), ("CaveRec.L", c.offset(of: \.l)!), ("CaveRec.Hc", c.offset(of: \.hc)!),
            ("CaveRec.h", c.offset(of: \.h)!), ("CaveRec.flags", c.offset(of: \.flags)!),
            ("IsleRec", i.stride), ("IsleRec.c", i.offset(of: \.c)!), ("IsleRec.R", i.offset(of: \.r)!), ("IsleRec.A", i.offset(of: \.a)!),
            ("IsleRec.D", i.offset(of: \.d)!), ("IsleRec.B", i.offset(of: \.b)!), ("IsleRec.K", i.offset(of: \.k)!),
            ("IsleRec.h", i.offset(of: \.h)!), ("IsleRec.fa", i.offset(of: \.fa)!), ("IsleRec.f", i.offset(of: \.f)!),
            // MapTables.sites reads the head at 0, info at 16 and lamp l at 32 + 16 l.
            ("Site", MapLayout.siteStride), ("Site.info", 16), ("Site.lamps", 32), ("Site.parts", 32 + 16 * MapLayout.maxLamps),
            ("TowerRec", TowerInfo.recordSize), ("IsleShade", Islands.entryBytes), ("IsleTable", Islands.tableBytes),
            ("TowerTable", Towers.tableBytes),
            ("ShadowBake", s.stride), ("ShadowBake.cam", s.offset(of: \.cam)!), ("ShadowBake.rows", s.offset(of: \.rows)!),
            ("VineBake", v.stride), ("VineBake.b", v.offset(of: \.b)!), ("VineBake.c", v.offset(of: \.c)!),
            ("VineRec", Vines.recordSize), ("VineTable", Vines.tableBytes),
        ]
    }

    func testLayoutsMatchTheShader() throws {
        let cpu = layouts
        let gpu = try ShaderProbes.get().run("probeLayouts", [Int32]?.none, count: cpu.count, as: Int32.self)
        for ((name, c), g) in zip(cpu, gpu) { XCTAssertEqual(c, Int(g), name) }
    }
}
