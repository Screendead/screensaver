import CoreGraphics
import Foundation
import Metal
import QuartzCore
import StrataCore

enum BugReport {
    static let key: CGKeyCode = 11 // kVK_ANSI_B

    static var folder: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("STRATA", isDirectory: true).appendingPathComponent("Bug Reports", isDirectory: true)
    }

    static func makeDirectory(in base: URL = folder) -> URL? {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        let dir = base.appendingPathComponent(f.string(from: Date()), isDirectory: true)
        return (try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)) != nil ? dir : nil
    }

    private static func num(_ d: Double) -> Any { d.isFinite ? d : NSNull() }

    private static func json(_ r: BugRecorder.Record) -> [String: Any] {
        let d = 180 / Double.pi
        return ["frame": r.frame, "time": num(r.time), "dt": num(r.dt), "simTime": num(r.simTime),
                "position": [num(r.pos.x), num(r.pos.y), num(r.pos.z)], "yawDeg": num(r.yaw * d), "pitchDeg": num(r.pitch * d),
                "rollDeg": num(r.roll * d), "VR_CAM": r.cam, "phase": num(r.phase), "day": r.day, "exposure": num(r.exposure),
                "battery": r.battery, "renderScale": num(r.renderScale), "gpuMs": r.gpuMs.map(num) ?? NSNull(),
                "coarsePending": r.coarsePending, "coarseGenerations": r.coarseGenerations,
                "mapOrigin": [Int(r.mapOrigin.x), Int(r.mapOrigin.y)]]
    }

    private static func quoted(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    // Writes state.json and replay.txt; the clip follows from the recorder.
    static func write(to dir: URL, renderer r: Renderer, recorder: BugRecorder, binary: String) {
        let records = recorder.recentRecords()
        var state: [String: Any] = recorder.info
        state["created"] = ISO8601DateFormatter().string(from: Date())
        state["timeScale"] = num(r.timeScale)
        state["simTime"] = num(r.simTime)
        state["clock"] = ["phase": num(r.clock.phase), "day": r.clock.day, "nightIndex": r.clock.nightIndex,
                          "sunY": num(r.clock.sunDirection.y), "dayLength": SkyClock.dayLength]
        state["battery"] = r.battery
        state["renderScale"] = num(r.renderScale)
        state["headroom"] = num(Double(r.headroom))
        state["exposure"] = num(r.exposure)
        state["clip"] = ["file": "clip.mp4", "width": recorder.width, "height": recorder.height]
        state["frames"] = records.map(json)
        state["log"] = BugLog.recent
        if let data = try? JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: dir.appendingPathComponent("state.json"))
        }

        let seed = recorder.info["seed"] as? UInt32 ?? 0
        let screen = recorder.info["screenPixels"] as? [Int] ?? [recorder.width, recorder.height]
        let dither = recorder.info["dither"] as? Bool ?? true
        func command(_ rec: BugRecorder.Record, _ name: String, size: [Int]) -> String {
            var a = ["VR_CAM=\(rec.cam)", "VR_EXPOSURE=\(rec.exposure)", quoted(binary), "--screenshot", quoted(dir.appendingPathComponent(name + ".png").path),
                     "--size", "\(size[0])x\(size[1])", "--seed", "\(seed)", "--times", "\(rec.simTime / r.timeScale)"]
            if r.timeScale != 1 { a += ["--speed", "\(r.timeScale)"] }
            if rec.battery { a.append("--battery") }
            a += ["--scale", "\(rec.renderScale)"]
            if !dither { a.append("--nodither") }
            if let p = r.pinnedPhase { a += ["--phase", "\(p)"] }
            return a.joined(separator: " ")
        }
        var text = "STRATA bug report \(dir.lastPathComponent)\n\n"
        text += "clip.mp4: about the last 6 s on screen \(recorder.info["screenIndex"] ?? 0), \(recorder.width)x\(recorder.height)"
        text += " (one pixel per dither cell). state.json: session settings, per-frame camera, clock and GPU time for the last"
        text += " \(Int(BugRecorder.recordSeconds)) s, and the event log. note.txt, if present: what the user saw.\n\n"
        if let last = records.last {
            var picks: [(String, BugRecorder.Record)] = [("last frame", last)]
            for back in [1.0, 3.0] {
                if let rec = records.last(where: { $0.time <= last.time - back }) { picks.append(("\(Int(back)) s earlier", rec)) }
            }
            text += "Re-render: sim time, exposure and camera pose as recorded, with the camera held still from the start. Sky,"
            text += " clouds and terrain match; birds are not replayed (the still camera grows its own), and temporal smoothing and"
            text += " streamed detail settle as for a still, so expect dither flips along edges.\n"
            for (i, (label, rec)) in picks.enumerated() {
                text += String(format: "\n# %@: frame %d, %.2f s into the show, sim time %.3f\n", label, rec.frame, rec.time, rec.simTime)
                text += command(rec, "replay_\(i)", size: screen) + "\n"
            }
            // The drawable's own grid; the clip drops an odd last row or column of it.
            let cells = recorder.info["drawable"] as? [Int] ?? [recorder.width, recorder.height]
            text += "\n# last frame on the cell grid (one pixel per cell; clip frames are this less any odd last row or column)\n"
            text += command(last, "replay_0_cells", size: cells) + "\n"
        } else {
            text += "No frames were recorded.\n"
        }
        try? text.write(to: dir.appendingPathComponent("replay.txt"), atomically: true, encoding: .utf8)
    }

    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576 : -1
    }

    // VR_BUGTEST: records `seconds` of autopilot flight at 60 fps through the saver's recorder and
    // writes a report beside `path`. The capture is committed in its own command buffer here so its GPU time shows apart.
    static func test(_ r: Renderer, gpu: GPU, settings: Settings, output tex: MTLTexture, path: String, seconds: Double) {
        let w = tex.width, h = tex.height
        r.simulate(settings.screenshotTimes.min() ?? 0)
        let info: [String: Any] = ["seed": r.world.seed, "screenIndex": 0, "screenPixels": [w, h], "drawable": [w, h], "dither": !settings.noDither,
                                   "test": true]
        guard let rec = BugRecorder(gpu: gpu, width: w, height: h, info: info) else { print("bugtest: recorder failed"); return }
        let before = footprintMB()
        var frameMs: [Double] = [], captureMs: [Double] = [], peak = before
        let frames = max(1, Int(seconds * 60))
        for i in 0..<frames {
            // Paced as the display would, so the encoder sees the live frame rate.
            let wait = rec.started + Double(i) / 60 - CACurrentMediaTime()
            if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            r.simulate(1.0 / 60)
            let cb = r.queue.makeCommandBuffer()!
            r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
            cb.commit()
            let cap = r.queue.makeCommandBuffer()!
            rec.capture(r, into: cap, source: tex, time: rec.started + Double(i) / 60)
            cap.commit()
            cap.waitUntilCompleted()
            frameMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            captureMs.append((cap.gpuEndTime - cap.gpuStartTime) * 1000)
            if i % 30 == 0 { peak = max(peak, footprintMB()) }
        }
        _ = Snapshot.save(tex, to: path)
        let base = URL(fileURLWithPath: path).deletingPathExtension().path + "_reports"
        guard let dir = makeDirectory(in: URL(fileURLWithPath: base)) else { print("bugtest: no report dir"); return }
        write(to: dir, renderer: r, recorder: rec, binary: URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path)
        // AVFoundation's loading needs the main run loop turning.
        var ok: Bool?
        rec.finish(to: dir.appendingPathComponent("clip.mp4")) { r in DispatchQueue.main.async { ok = r } }
        while ok == nil { RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.05)) }
        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        print(String(format: "bugtest: %d frames, frame gpu median %.3f ms, capture gpu median %.3f ms (max %.3f), footprint %.1f MB before recording, peak %.1f MB while recording, clip %@ -> %@",
                     frames, median(frameMs), median(captureMs), captureMs.max() ?? 0, before, peak, ok == true ? "ok" : "FAILED", dir.path))
    }
}
