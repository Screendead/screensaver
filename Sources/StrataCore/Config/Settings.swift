import CoreGraphics
import Foundation

package struct Settings {
    package var idleThreshold: Double?
    package var menu = false
    package var noDither = false
    package var preview = false
    package var previewDuration: Double = 0
    package var screenshotPath: String?
    package var screenshotTimes: [Double] = [8]
    package var screenshotSize = CGSize(width: 1728, height: 1117)
    package var forceBattery = false
    package var seed: UInt32 = 0
    package var debugMode: Float = 0
    package var renderScale: Double?
    package var phase: Double?
    package var biome: Int?
    package var timeScale: Double = 1
    package var pitch: Double?
    package var headroom: Float?
    package var servePath: String?

    package static func parse(_ args: [String]) -> Settings {
        var s = Settings()
        var i = 1
        func next() -> String { i += 1; return i < args.count ? args[i] : "" }
        while i < args.count {
            switch args[i] {
            case "--idle": s.idleThreshold = Double(next()) ?? 30
            case "--menu": s.menu = true
            case "--nodither": s.noDither = true
            case "--preview": s.preview = true
            case "--duration": s.previewDuration = Double(next()) ?? 0
            case "--screenshot": s.screenshotPath = next()
            case "--times": s.screenshotTimes = next().split(separator: ",").compactMap { Double($0) }
            case "--size":
                let p = next().split(separator: "x").compactMap { Double($0) }
                if p.count == 2 { s.screenshotSize = CGSize(width: p[0], height: p[1]) }
            case "--battery": s.forceBattery = true
            case "--seed": s.seed = UInt32(next()) ?? 0
            case "--debug": s.debugMode = Float(next()) ?? 0
            case "--scale": s.renderScale = Double(next())
            case "--phase": s.phase = Double(next())
            case "--biome": s.biome = Biome.number(next())
            case "--speed": s.timeScale = Double(next()) ?? 1
            case "--pitch": s.pitch = Double(next())
            case "--headroom": s.headroom = Float(next())
            case "--serve": s.servePath = next()
            default: break
            }
            i += 1
        }
        return s
    }
}
