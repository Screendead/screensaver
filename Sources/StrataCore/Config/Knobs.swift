import Foundation

// The VR_* test switches, the process environment's or a served request's (Server swaps `current` per request). VR_DEBUG
// is read once at launch instead (dlog).
package struct Knobs {
    // VR_TIMING: log GPU frame times and world regeneration.
    package let timing: Bool
    // VR_COARSE_FULL: rebuild the LOD rings whole every frame.
    package let fullCoarse: Bool
    // VR_CHUNKED: snapshots update the coarse rings a chunk per frame, as live.
    package let chunkedCoarse: Bool
    // VR_PIECES: regenerate a map one command buffer per piece, each timed (the regeneration budget is per piece).
    package let pieces: Bool
    // VR_SITES: log each new map's giants and sites; VR_SITES=3 also each site's parts.
    package let sites: Bool, siteParts: Bool
    // VR_CAVES=r: list the caves within r of each new map's centre (not a number: 1500).
    package let caveList: Float?
    // VR_NOCAVES: an empty cave table, which must render as before caves.
    package let noCaves: Bool
    // VR_ISLE_FIELD=1: the test field round (-3500, -2500) in place of island country; 2: no islands.
    package let isleField: Int32
    // VR_POPLOG: log birds that appear or vanish in view (PopLog).
    package let popLog: Bool
    // VR_BIRDTRACE: log the flocks every simulated second.
    package let birdTrace: Bool
    // VR_PROFILE: "noshadow", "oldshadow", "nobake", "nohorizon", "noband", "nearclouds", "lowmax", "norefl", or "bitsN"
    // (Renderer.bakeShadows).
    package let profile: String
    // VR_PROFILE=bitsN: N, the shader's profiling flags (Renderer.bakeShadows); else 0.
    package let profileBits: Int
    // VR_LODSHIFT=v: move the LOD rings' centre v voxels ahead of the camera; frames differing only in it isolate pops.
    package let lodShift: Float
    // VR_START=x,z,yawDeg: the flight starts there.
    package let start: [Double]?
    // VR_TOWERS=n: list the towers of the n-cell square of tower cells round the start (Snapshot.listTowers).
    package let towerList: Int?
    // VR_ISLES=n: list the n islands nearest the start (Snapshot.listIsles).
    package let isleList: Int?
    // VR_SHOW and VR_CAVE: hold the camera on a site or a cave (Snapshot.showSite, showCave).
    package let show: String?, cave: String?
    // VR_CAM=x,z,height,yawDeg,pitchDeg[,abs[,rollDeg]]: hold the camera at world (x, z), `height` above the ground there
    // (absolute when abs is 1).
    package let cam: [Double]?
    // VR_FLIGHT=minutes: fly the autopilot that long, logging it (Snapshot.fly); VR_FTRACE: also every 0.25 s.
    package let flight: Double?
    package let flightTrace: Bool
    // VR_BUGTEST=seconds: fly that long past the first time, recording as the saver does, then write a bug report beside
    // the screenshot path, whose last frame is the screenshot.
    package let bugTest: Double?
    // VR_EXPOSURE=e: the exposure a bug report recorded (snapshots jump straight to the target the saver eases toward).
    package let exposure: Double?
    // VR_PATH=file: render a recorded camera path frame by frame (Snapshot.replay).
    package let path: String?
    // VR_BIRDTEST=d and VR_MURMUR=d: replace the flocks with a test pattern or a murmuration d ahead.
    package let birdTest: Double?, murmur: Double?
    // VR_MOTION: advance the simulation 1/60 s between frames, VR_MOTION_FRAMES of them (48); without it, VR_BENCH still
    // frames (12; gpu= is their minimum: more frames ride out the GPU's clock ramp).
    package let motion: Bool
    package let motionFrames: Int?, bench: Int?
    // VR_FRAMES=n: also write each of the last n frames, for measuring frame-to-frame crawl.
    package let frames: Int?
    // VR_POWER_FLIP=i: switch between battery and AC power at frame i (render scale kept).
    package let powerFlip: Int?
    // VR_DOLLY=v: a held camera moves v voxels a frame along its heading, level (approaches across the rings).
    package let dolly: Double?

    package init(_ env: [String: String]) {
        timing = env["VR_TIMING"] != nil
        fullCoarse = env["VR_COARSE_FULL"] != nil
        chunkedCoarse = env["VR_CHUNKED"] != nil
        pieces = env["VR_PIECES"] != nil
        sites = env["VR_SITES"] != nil
        siteParts = env["VR_SITES"] == "3"
        caveList = env["VR_CAVES"].map { Float($0) ?? 1500 }
        noCaves = env["VR_NOCAVES"] != nil
        isleField = Int32(env["VR_ISLE_FIELD"] ?? "") ?? 0
        popLog = env["VR_POPLOG"] != nil
        birdTrace = env["VR_BIRDTRACE"] != nil
        profile = env["VR_PROFILE"] ?? ""
        profileBits = profile.hasPrefix("bits") ? Int(profile.dropFirst(4)) ?? 0 : 0
        lodShift = Float(env["VR_LODSHIFT"] ?? "") ?? 0
        start = env["VR_START"]?.split(separator: ",").compactMap { Double($0) }
        towerList = Int(env["VR_TOWERS"] ?? "")
        isleList = Int(env["VR_ISLES"] ?? "")
        show = env["VR_SHOW"]
        cave = env["VR_CAVE"]
        cam = env["VR_CAM"].map { $0.split(separator: ",").compactMap { Double($0) } }
        flight = Double(env["VR_FLIGHT"] ?? "")
        flightTrace = env["VR_FTRACE"] != nil
        bugTest = Double(env["VR_BUGTEST"] ?? "")
        exposure = Double(env["VR_EXPOSURE"] ?? "")
        path = env["VR_PATH"]
        birdTest = Double(env["VR_BIRDTEST"] ?? "")
        murmur = Double(env["VR_MURMUR"] ?? "")
        motion = env["VR_MOTION"] != nil
        motionFrames = Int(env["VR_MOTION_FRAMES"] ?? "")
        bench = Int(env["VR_BENCH"] ?? "")
        frames = Int(env["VR_FRAMES"] ?? "")
        powerFlip = Int(env["VR_POWER_FLIP"] ?? "")
        dolly = Double(env["VR_DOLLY"] ?? "")
    }

    package static var current = Knobs(ProcessInfo.processInfo.environment)
}
