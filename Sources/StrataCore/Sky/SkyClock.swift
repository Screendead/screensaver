import Foundation
import simd

// Day cycle with a time-warp that lingers in golden and blue hour.
package struct SkyClock {
    package var phase: Double = -0.035 // just before sunrise: open on blue hour
    package var day = 0
    package static let dayLength: Double = 12 * 60

    package init() {}

    // Peak day is held to a long-afternoon sun (flat high noon judged weakest); night keeps its full depth
    // (a shallower night judged worse). Both halves meet at the horizon.
    static func elevation(_ phase: Double) -> Double {
        let w = sin(2 * .pi * phase)
        return w * (w > 0 ? 0.42 : 0.56) + 0.07
    }

    // A full turn a day: the sun sets opposite where it rose, and it and the moon opposite it are continuous across the
    // wrap (phase 1 to 0).
    static func azimuth(_ phase: Double) -> Double {
        0.9 + 2 * .pi * (phase - floor(phase))
    }

    static func direction(_ phase: Double) -> SIMD3<Double> {
        let el = asin(SkyClock.elevation(phase))
        let az = SkyClock.azimuth(phase)
        return SIMD3(cos(el) * sin(az), sin(el), cos(el) * cos(az))
    }

    package var sunDirection: SIMD3<Double> { SkyClock.direction(phase) }

    // The evening star trails the sun along its path and sets this much later; the morning star leads it and rises
    // this much earlier.
    static let planetLag = 0.09

    package var planetDirection: SIMD3<Double> {
        let evening = SkyClock.direction(phase - SkyClock.planetLag), morning = SkyClock.direction(phase + SkyClock.planetLag)
        return evening.y > morning.y ? evening : morning
    }

    // Runs 0 at dusk to 1 at dawn and eases back to 0 through the morning, so the burn-down schedule never steps.
    package var nightProgress: Double {
        let ph = phase - floor(phase)
        return ph < 0.25 ? 1 - { let s = min(max((ph - 0.08) / 0.12, 0), 1); return s * s * (3 - 2 * s) }() : min(max((ph - 0.52) / 0.46, 0), 1)
    }

    // The sea of cloud's amount: must match cloudSeaAmount in Shaders/Atmos/CloudSea.metal, 0 at dusk, by day and at night.
    package var cloudSea: Double {
        let e = SkyClock.elevation(phase)
        return smoothstep(0.5, 0.75, nightProgress) * smoothstep(-0.24, -0.08, e) * (1 - smoothstep(0.08, 0.30, e))
    }

    // A night keeps one index from dusk through to the morning after: it steps at mid-morning rather than at the dawn
    // wrap, so a night keeps its beacons and rites until daylight.
    package var nightIndex: Int {
        let n = day + Int(floor(phase))
        return phase - floor(phase) < 0.25 ? n - 1 : n
    }

    package mutating func advance(_ dt: Double) {
        let y = SkyClock.elevation(phase)
        let s = min(max((abs(y) - 0.22) / 0.35, 0), 1)
        // Linger in golden and blue hour, hurry through midday and deep night.
        let warp = y < -0.25 ? 5.0 : 0.5 + 1.5 * s * s * (3 - 2 * s)
        phase += dt / SkyClock.dayLength * warp * 1.05
        if phase > 1 { phase -= 1; day += 1 }
    }
}
