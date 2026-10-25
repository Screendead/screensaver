import Foundation

// The VR_* test knobs: the process environment's, or a served request's.
struct Knobs {
    let env: [String: String]
    let timing, fullCoarse, chunkedCoarse, pieces, sites, popLog, birdTrace: Bool
    let profile: String
    let lodShift: Float

    init(_ env: [String: String]) {
        self.env = env
        timing = env["VR_TIMING"] != nil
        fullCoarse = env["VR_COARSE_FULL"] != nil
        chunkedCoarse = env["VR_CHUNKED"] != nil
        pieces = env["VR_PIECES"] != nil
        sites = env["VR_SITES"] != nil
        popLog = env["VR_POPLOG"] != nil
        birdTrace = env["VR_BIRDTRACE"] != nil
        profile = env["VR_PROFILE"] ?? ""
        lodShift = Float(env["VR_LODSHIFT"] ?? "") ?? 0
    }

    static var current = Knobs(ProcessInfo.processInfo.environment)
}
