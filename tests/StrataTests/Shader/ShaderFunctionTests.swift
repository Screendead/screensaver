import Metal
@testable import StrataCore
import XCTest
import simd

// The CPU's ports of shader functions, run against the shader's own on the GPU.
final class ShaderFunctionTests: XCTestCase {
    // A fixed spread of 32-bit words, sign bits and the pinned-biome seeds among them.
    private func words(_ n: Int) -> [UInt32] {
        var s: UInt64 = 0x2545_F491_4F6C_DD1D
        return (0..<n).map { _ in s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407; return UInt32(truncatingIfNeeded: s >> 32) }
    }

    // Bit for bit: the camera's lamp choices and the islands' mist ride on these.
    func testHashesMatchBitForBit() throws {
        let w = words(4 * 4096)
        var input = (0..<4096).map { SIMD4(w[4 * $0], w[4 * $0 + 1], w[4 * $0 + 2], w[4 * $0 + 3]) }
        input += (0..<16).map { SIMD4(0xFF00_0000 | UInt32($0) << 20 | 0x1234, 0, 0, 0) }
        let gpu = try ShaderProbes.get().run("probeHash", input, count: input.count, as: SIMD4<UInt32>.self)
        for (q, g) in zip(input, gpu) {
            XCTAssertEqual(Provinces.pcg(q.x), g.x)
            XCTAssertEqual(Provinces.hash2i(Int32(bitPattern: q.y), Int32(bitPattern: q.z), q.w), g.y)
            XCTAssertEqual(UInt32(Biome.pinned(q.x)), g.z)
        }
    }

    // The renderer reads the jungle and old-growth provinces on the CPU to decide what the frame holds; the shader draws
    // them. Over a 40k square for three seeds, one of them pinned to the jungle.
    func testProvincesMatchTheShader() throws {
        let points = stride(from: -20000.0, through: 20000.0, by: 250.0).flatMap { z in stride(from: -20000.0, through: 20000.0, by: 250.0).map { SIMD2(Float($0), Float(z)) } }
        var jungle = 0, old = 0
        for seed: UInt32 in [1, 303, Biome.seed(77, biome: 8)] {
            let gpu = try ShaderProbes.get().run("probeProvinces", points, count: points.count, as: SIMD2<Float>.self, bytes: [seed])
            for (p, g) in zip(points, gpu) {
                let c = Provinces.at(Double(p.x), Double(p.y), seed: seed)
                XCTAssertEqual(c.0, Double(g.x), accuracy: 2e-3, "jungle at \(p), seed \(seed)")
                XCTAssertEqual(c.1, Double(g.y), accuracy: 2e-3, "old growth at \(p), seed \(seed)")
                if c.0 > 0.5 { jungle += 1 }
                if c.1 > 0.5 { old += 1 }
            }
        }
        XCTAssertGreaterThan(jungle, points.count / 20)
        XCTAssertGreaterThan(old, 50)
    }

    // The shadow maps bake for SkyClock.keyLight, the scene shades with makeLight's key: over a whole day the two agree in
    // direction and on whether it casts shadows.
    func testKeyLightMatchesMakeLight() throws {
        let suns = (0..<4000).map { SkyClock.direction(Double($0) / 4000) }
        let gpu = try ShaderProbes.get().run("probeLight", suns.map { SIMD3<Float>($0) }, count: suns.count, as: SIMD4<Float>.self)
        var shadowed = 0
        for (sun, g) in zip(suns, gpu) {
            let (key, casts) = SkyClock.keyLight(sun)
            XCTAssertLessThan(simd_length(SIMD3<Float>(key) - SIMD3(g.x, g.y, g.z)), 1e-4, "sun.y \(sun.y)")
            // The moon takes over at m = 0.5 (sun.y -0.125), where float and Double may round apart.
            if abs(sun.y + 0.125) > 1e-4 { XCTAssertEqual(casts, g.w == 1, "sun.y \(sun.y)") }
            if casts { shadowed += 1 }
        }
        XCTAssertGreaterThan(shadowed, 0)
        XCTAssertLessThan(shadowed, suns.count)
    }

    // The renderer compiles the sea of cloud in only while SkyClock.cloudSea is above 0: it must be 0 wherever the
    // shader's is, and agree elsewhere.
    func testCloudSeaMatchesTheShader() throws {
        var clock = SkyClock()
        let samples = (0..<4000).map { i -> (Double, SIMD2<Float>) in
            clock.phase = Double(i) / 4000
            return (clock.cloudSea, SIMD2(Float(SkyClock.elevation(clock.phase)), Float(clock.nightProgress)))
        }
        let on = MTLFunctionConstantValues()
        var yes = true
        on.setConstantValue(&yes, type: .bool, index: 8)
        let gpu = try ShaderProbes.get().run("probeCloudSea", samples.map(\.1), count: samples.count, as: Float.self, constants: on)
        for ((cpu, q), g) in zip(samples, gpu) {
            XCTAssertEqual(cpu, Double(g), accuracy: 1e-4, "elevation \(q.x), night \(q.y)")
            XCTAssertEqual(cpu > 0, g > 0, "elevation \(q.x), night \(q.y)")
        }
        XCTAssertGreaterThan(samples.map(\.0).max()!, 0.5)
    }

    // The autopilot frames a hamlet at night by its beacon, and a henge only when its rite burns: on exactly the nights
    // lampLevel lights them.
    func testLitSitesMatchLampLevel() throws {
        let camera = FlightCamera(seed: 1, terrain: Terrain(), far: FarHeights())
        let ids = words(64)
        var input: [SIMD4<UInt32>] = []
        for id in ids { for l: UInt32 in [0, 3, 7] { for night in -2...30 { for kind: UInt32 in [5, 6] {
            input.append(SIMD4(id, l, UInt32(bitPattern: Int32(night)), kind))
        } } } }
        let gpu = try ShaderProbes.get().run("probeLamp", input, count: input.count, as: Float.self)
        var lit = 0
        for (q, level) in zip(input, gpu) {
            let lamp = SiteInfo.Lamp(pos: .zero, kind: Int(q.w), rise: 0, index: 9 * MapLayout.maxLamps + Int(q.y), smoke: .zero)
            let site = SiteInfo(index: 9, id: q.x, kind: q.w == 5 ? .hamlet : .henge, pos: .zero, radius: 10, lamps: [lamp])
            camera.nightIndex = Int(Int32(bitPattern: q.z))
            let burns = q.w == 5 ? camera.litInterest(site) == 1.0 : camera.litInterest(site) > 0
            XCTAssertEqual(burns, level > 0.5, "site \(q.x) lamp \(q.y) night \(Int32(bitPattern: q.z)) kind \(q.w)")
            if burns { lit += 1 }
        }
        XCTAssertGreaterThan(lit, input.count / 5)
        XCTAssertLessThan(lit, input.count / 2)
    }
}
