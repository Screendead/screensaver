import Foundation
@testable import Strata
@testable import StrataCore
import XCTest
import simd

// state.json is read back by bugclip.sh, which replays a report's camera path through a build to prove a fix.
final class BugReportStateTests: XCTestCase {
    private func record(_ i: Int) -> BugRecorder.Record {
        BugRecorder.Record(frame: i, time: Double(i) / 60, dt: i == 0 ? .nan : 1.0 / 60, simTime: 100 + Double(i) / 30,
                           pos: SIMD3(1234.5 + Double(i), 87.25, -3456.75), yaw: 0.3 + 0.01 * Double(i), pitch: -0.2, roll: 0.05,
                           phase: 0.62, day: 3, exposure: 1.7, battery: true, renderScale: 0.75, coarsePending: false,
                           coarseGenerations: [1, 2, 3, 4], mapOrigin: SIMD2(1024, -4096), gpuMs: i == 0 ? nil : 4.5)
    }

    // The keys the recorder's session info carries (SaverController, BugReport.test), passed through as given.
    private let session: [String: Any] = ["seed": UInt32(303), "drawable": [1728, 1117], "dither": true, "screenPixels": [3456, 2234]]

    // As write() writes it, read back.
    private func written() throws -> [String: Any] {
        var clock = SkyClock()
        clock.phase = 0.62
        clock.day = 3
        let state = BugReport.state(info: session, records: (0..<90).map(record), clip: (1728, 1116), timeScale: 2, simTime: 103,
                                    clock: clock, battery: true, renderScale: 0.75, headroom: 1.5, exposure: 1.7)
        XCTAssertTrue(JSONSerialization.isValidJSONObject(state), "a non-finite number reached state.json")
        let data = try JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // Every key bugclip.sh's reader asks for, at the level it asks: s['k'] and s.get('k') on the state, s['clip']['k'] on
    // the clip, and anything else (x['k'], sel[0]['k'], f[-1]['k']) on a frame.
    func testHoldsEveryKeyBugclipReads() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let script = try String(contentsOf: root.appendingPathComponent("bugclip.sh"), encoding: .utf8)
        func keys(_ pattern: String) -> Set<String> {
            let re = try! NSRegularExpression(pattern: pattern)
            return Set(re.matches(in: script, range: NSRange(script.startIndex..., in: script)).map { String(script[Range($0.range(at: 1), in: script)!]) })
        }
        let top = keys(#"\bs(?:\['|\.get\(')(\w+)'"#), clip = keys(#"\['clip'\]\['(\w+)'\]"#)
        let frame = keys(#"(?:\bx|\]|\))\['(\w+)'\]"#).subtracting(clip)
        XCTAssertTrue(top.isSuperset(of: ["frames", "seed", "drawable", "clip", "renderScale", "battery"]), "the reader moved: \(top)")
        XCTAssertTrue(frame.isSuperset(of: ["time", "simTime", "position", "yawDeg", "VR_CAM"]), "the reader moved: \(frame)")

        let state = try written()
        for k in top { XCTAssertNotNil(state[k], "state.\(k)") }
        let c = try XCTUnwrap(state["clip"] as? [String: Any])
        for k in clip { XCTAssertNotNil(c[k] as? Int, "clip.\(k)") }
        let frames = try XCTUnwrap(state["frames"] as? [[String: Any]])
        XCTAssertEqual(frames.count, 90)
        for k in frame { XCTAssertNotNil(frames[1][k], "frames[].\(k)") }
    }

    // bugclip.sh turns each frame into a VR_PATH line (simTime / timeScale, x, z, y, yaw, pitch and roll in degrees, exposure,
    // phase, day) and Snapshot.replay holds the camera there: the pose that comes back is the one recorded.
    func testFramesReplayAsRecorded() throws {
        let state = try written()
        let ts = try XCTUnwrap(state["timeScale"] as? Double)
        let frames = try XCTUnwrap(state["frames"] as? [[String: Any]])
        for (i, f) in frames.enumerated() {
            let p = try XCTUnwrap(f["position"] as? [Double])
            let line = [f["simTime"] as! Double / ts, p[0], p[2], p[1], f["yawDeg"] as! Double, f["pitchDeg"] as! Double,
                        f["rollDeg"] as! Double, f["exposure"] as! Double, f["phase"] as! Double, Double(f["day"] as! Int)]
            let r = record(i)
            XCTAssertEqual(line[0] * ts, r.simTime, accuracy: 1e-9)
            XCTAssertEqual(SIMD3(line[1], line[3], line[2]), r.pos)
            XCTAssertEqual(line[4] * .pi / 180, r.yaw, accuracy: 1e-12)
            XCTAssertEqual(line[5] * .pi / 180, r.pitch, accuracy: 1e-12)
            XCTAssertEqual(line[6] * .pi / 180, r.roll, accuracy: 1e-12)
            XCTAssertEqual(line[7], r.exposure)
            XCTAssertEqual(line[8], r.phase)
            XCTAssertEqual(Int(line[9]), r.day)
        }
        XCTAssertTrue(frames[0]["dt"] is NSNull)
        XCTAssertTrue(frames[0]["gpuMs"] is NSNull)
    }

    // A frame's VR_CAM, as bugclip.sh and replay.txt pass it, holds a snapshot's camera at the recorded pose: absolute
    // height (the sixth field 1), angles in degrees, roll last (Snapshot.run).
    func testCamHoldsTheRecordedPose() throws {
        let r = record(7)
        let v = try XCTUnwrap(Knobs(["VR_CAM": r.cam]).cam)
        XCTAssertEqual(v.count, 7)
        XCTAssertEqual(v[5], 1)
        XCTAssertEqual(SIMD3(v[0], v[2], v[1]), r.pos)
        XCTAssertEqual(v[3] * .pi / 180, r.yaw, accuracy: 1e-12)
        XCTAssertEqual(v[4] * .pi / 180, r.pitch, accuracy: 1e-12)
        XCTAssertEqual(v[6] * .pi / 180, r.roll, accuracy: 1e-12)
    }
}
