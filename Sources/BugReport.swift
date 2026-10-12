import AppKit
import AVFoundation
import CoreVideo
import Metal
import VideoToolbox

// MARK: - Event log

// The last notable events of this process, for bug reports.
enum BugLog {
    static let capacity = 200
    private static let lock = NSLock()
    private static var lines: [String] = []
    private static var head = 0
    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static func note(_ s: String) {
        dlog(s)
        lock.lock()
        defer { lock.unlock() }
        let line = stamp.string(from: Date()) + " " + s
        if lines.count < capacity { lines.append(line) } else { lines[head] = line; head = (head + 1) % capacity }
    }

    static var recent: [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(lines[head...] + lines[..<head])
    }
}

// MARK: - Recorder

// Records the first screen while the saver shows: the presented frames go to the hardware H.264 encoder in rolling ~2 s
// segment files, and the last ~10 s of per-frame state stay in a ring. Nothing holds raw frames beyond the encoder's
// few pooled buffers.
final class BugRecorder {
    struct Record {
        var frame: Int
        var time: Double
        var dt: Double
        var simTime: Double
        var pos: SIMD3<Double>
        var yaw: Double, pitch: Double, roll: Double
        var phase: Double
        var day: Int
        var exposure: Double
        var battery: Bool
        var renderScale: Double
        var coarsePending: Bool
        var coarseGenerations: [Int]
        var mapOrigin: SIMD2<Int32>
        var gpuMs: Double?

        // The VR_CAM form (absolute height, angles in degrees, roll last).
        var cam: String {
            let d = 180 / Double.pi
            return "\(pos.x),\(pos.z),\(pos.y),\(yaw * d),\(pitch * d),1,\(roll * d)"
        }
    }

    private final class Segment {
        let url: URL
        let writer: AVAssetWriter
        let input: AVAssetWriterInput
        let adaptor: AVAssetWriterInputPixelBufferAdaptor
        let start: CMTime
        var last: CMTime?
        init(url: URL, writer: AVAssetWriter, input: AVAssetWriterInput, adaptor: AVAssetWriterInputPixelBufferAdaptor, start: CMTime) {
            (self.url, self.writer, self.input, self.adaptor, self.start) = (url, writer, input, adaptor, start)
        }
    }

    static let segmentSeconds = 2.0, keptSegments = 4, recordSeconds = 10.0
    let width: Int, height: Int
    let info: [String: Any]
    let started = CACurrentMediaTime()
    private let pipeline: MTLComputePipelineState
    private let cache: CVMetalTextureCache
    private let pool: CVPixelBufferPool
    private let dir: URL
    private let queue = DispatchQueue(label: "strata.bugreport.video")
    private let finishing = DispatchGroup()
    // Queue-only.
    private var segments: [Segment] = []
    private var serial = 0
    private var dropped = 0
    private var openFailures = 0
    private var stoppedQueue = false
    // Main-only.
    private var stopped = false
    private var frame = 0
    private var lastTime: Double?
    private var lastOrigin: SIMD2<Int32>?
    private var pendingSince: Int?
    // Shared with completion handlers.
    private let lock = NSLock()
    private var records: [Record?] = Array(repeating: nil, count: 1200)

    // `info` is the session's fixed state for the report (seed, screen, dither...).
    init?(gpu: GPU, width w: Int, height h: Int, info: [String: Any]) {
        width = w & ~1
        height = h & ~1
        self.info = info
        var c: CVMetalTextureCache?
        let poolAttrs = [kCVPixelBufferPoolMinimumBufferCountKey: 3] as CFDictionary
        let bufAttrs = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                        kCVPixelBufferWidthKey: width, kCVPixelBufferHeightKey: height,
                        kCVPixelBufferMetalCompatibilityKey: true,
                        kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()] as [CFString: Any] as CFDictionary
        var p: CVPixelBufferPool?
        guard width >= 16, height >= 16, let pl = gpu.pipeline("captureNV12"),
              CVMetalTextureCacheCreate(nil, nil, gpu.device, nil, &c) == kCVReturnSuccess, let c,
              CVPixelBufferPoolCreate(nil, poolAttrs, bufAttrs, &p) == kCVReturnSuccess, let p else { return nil }
        pipeline = pl
        cache = c
        pool = p
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("STRATA-capture-\(ProcessInfo.processInfo.processIdentifier)-\(Int(started * 1000))")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    private func texture(_ pb: CVPixelBuffer, plane: Int, format: MTLPixelFormat) -> CVMetalTexture? {
        var t: CVMetalTexture?
        let attrs = [kCVMetalTextureUsage: MTLTextureUsage([.shaderRead, .shaderWrite]).rawValue] as CFDictionary
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, attrs, format, CVPixelBufferGetWidthOfPlane(pb, plane),
                                                  CVPixelBufferGetHeightOfPlane(pb, plane), plane, &t)
        return t
    }

    // Called after the frame is encoded into `cb`, before it is committed; `time` is the frame's host time in seconds.
    func capture(_ r: Renderer, into cb: MTLCommandBuffer, source: MTLTexture, time: Double) {
        guard !stopped else { return }
        frame += 1
        let dt = lastTime.map { time - $0 } ?? 0
        lastTime = time
        let gens = (1...Clipmap.levels).map { r.clipmap.generation($0) }
        let rec = Record(frame: frame, time: time - started, dt: dt, simTime: r.simTime, pos: r.camera.position, yaw: r.camera.yaw,
                         pitch: r.camera.pitch, roll: r.camera.roll, phase: r.clock.phase, day: r.clock.day, exposure: r.exposure,
                         battery: r.battery, renderScale: r.renderScale, coarsePending: !r.clipmap.idle, coarseGenerations: gens,
                         mapOrigin: r.world.origin)
        noteChanges(rec)
        let slot = frame % records.count, n = frame
        lock.lock()
        records[slot] = rec
        lock.unlock()

        var pb: CVPixelBuffer?
        let aux = [kCVPixelBufferPoolAllocationThresholdKey: 6] as CFDictionary
        let src = source.width >= width && source.height >= height ? source : nil
        guard let src, CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pool, aux, &pb) == kCVReturnSuccess, let pb,
              let y = texture(pb, plane: 0, format: .r8Unorm), let c = texture(pb, plane: 1, format: .rg8Unorm),
              let yt = CVMetalTextureGetTexture(y), let ct = CVMetalTextureGetTexture(c), let enc = cb.makeComputeCommandEncoder() else {
            queue.async { self.dropped += 1 }
            return
        }
        var linear: UInt32 = src.pixelFormat == .rgba16Float ? 1 : 0
        enc.setComputePipelineState(pipeline)
        enc.setTexture(src, index: 0)
        enc.setTexture(yt, index: 1)
        enc.setTexture(ct, index: 2)
        enc.setBytes(&linear, length: 4, index: 0)
        enc.dispatchThreads(MTLSize(width: width / 2, height: height / 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        let t = CMTime(seconds: time, preferredTimescale: 60000)
        cb.addCompletedHandler { [self] cb in
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            lock.lock()
            if records[slot]?.frame == n { records[slot]?.gpuMs = ms }
            lock.unlock()
            queue.async { [y, c] in
                withExtendedLifetime((y, c)) { self.append(pb, at: t) }
            }
        }
    }

    private func noteChanges(_ r: Record) {
        if r.dt > 0.1 { BugLog.note(String(format: "frame %d: hitch, %.0f ms since the last", r.frame, r.dt * 1000)) }
        if let o = lastOrigin, o != r.mapOrigin { BugLog.note("frame \(r.frame): 1-voxel map recentred, origin \(r.mapOrigin.x),\(r.mapOrigin.y)") }
        if r.coarsePending, pendingSince == nil { pendingSince = r.frame }
        if !r.coarsePending, let since = pendingSince {
            BugLog.note("frame \(r.frame): coarse job done after \(r.frame - since) frames, generations \(r.coarseGenerations)")
            pendingSince = nil
        }
        lastOrigin = r.mapOrigin
    }

    private static let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoEncoderSpecificationKey: [kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String: true],
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 12_000_000, AVVideoExpectedSourceFrameRateKey: 60,
                                          AVVideoAllowFrameReorderingKey: false, AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
        AVVideoColorPropertiesKey: [AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                                    AVVideoTransferFunctionKey: kCVImageBufferTransferFunction_sRGB as String,
                                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2],
    ]

    private func openSegment(at t: CMTime) -> Segment? {
        serial += 1
        let url = dir.appendingPathComponent(String(format: "seg-%05d.mov", serial))
        var settings = BugRecorder.videoSettings
        settings[AVVideoWidthKey] = width
        settings[AVVideoHeightKey] = height
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return nil }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard w.canAdd(input) else { return nil }
        w.add(input)
        guard w.startWriting() else {
            BugLog.note("capture: writer failed to start: \(w.error?.localizedDescription ?? "?")")
            return nil
        }
        w.startSession(atSourceTime: t)
        return Segment(url: url, writer: w, input: input, adaptor: adaptor, start: t)
    }

    private func close(_ s: Segment, at end: CMTime) {
        guard s.writer.status == .writing else { return }
        s.input.markAsFinished()
        s.writer.endSession(atSourceTime: end)
        finishing.enter()
        s.writer.finishWriting { self.finishing.leave() }
    }

    private func append(_ pb: CVPixelBuffer, at t: CMTime) {
        guard !stoppedQueue else { return }
        if segments.last.map({ (t - $0.start).seconds >= BugRecorder.segmentSeconds }) ?? true {
            // Without a working encoder, stop retrying rather than build a writer every frame.
            guard openFailures < 3 else { dropped += 1; return }
            if let cur = segments.last { close(cur, at: t) }
            guard let s = openSegment(at: t) else { openFailures += 1; dropped += 1; return }
            segments.append(s)
            while segments.count > BugRecorder.keptSegments { try? FileManager.default.removeItem(at: segments.removeFirst().url) }
        }
        guard let s = segments.last, s.input.isReadyForMoreMediaData, s.adaptor.append(pb, withPresentationTime: t) else {
            dropped += 1
            return
        }
        s.last = t
    }

    // Records from the last `recordSeconds`, oldest first.
    func recentRecords() -> [Record] {
        lock.lock()
        let all = records.compactMap { $0 }.sorted { $0.frame < $1.frame }
        lock.unlock()
        guard let last = all.last else { return [] }
        return all.filter { $0.time >= last.time - BugRecorder.recordSeconds }
    }

    // Stops recording and writes the kept segments, joined without re-encoding, to `url`; `done` runs on a background
    // queue with whether a clip was written.
    func finish(to url: URL, done: @escaping (Bool) -> Void) {
        stopped = true
        queue.async { [self] in
            stoppedQueue = true
            if let cur = segments.last {
                if let last = cur.last { close(cur, at: last + CMTime(value: 1000, timescale: 60000)) } else { cur.writer.cancelWriting() }
            }
            finishing.notify(queue: queue) { [self] in
                let parts = segments.filter { $0.writer.status == .completed }.map(\.url)
                BugLog.note("capture: \(parts.count) segments, \(dropped) frames dropped")
                let dir = dir
                Task {
                    let ok = await BugRecorder.join(parts, to: url)
                    try? FileManager.default.removeItem(at: dir)
                    done(ok)
                }
            }
        }
    }

    // Stops recording and drops what was recorded.
    func discard() {
        stopped = true
        queue.async { [self] in
            stoppedQueue = true
            for s in segments where s.writer.status == .writing { s.writer.cancelWriting() }
            segments.removeAll()
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private static func join(_ parts: [URL], to url: URL) async -> Bool {
        let comp = AVMutableComposition()
        guard !parts.isEmpty, let track = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else { return false }
        var at = CMTime.zero
        for part in parts {
            do {
                // A track keeps no hold on its asset.
                let asset = AVURLAsset(url: part)
                guard let src = try await asset.loadTracks(withMediaType: .video).first else { continue }
                let range = try await src.load(.timeRange)
                try withExtendedLifetime(asset) { try track.insertTimeRange(range, of: src, at: at) }
                at = at + range.duration
            } catch {
                BugLog.note("capture: \(part.lastPathComponent) unreadable: \(error.localizedDescription)")
            }
        }
        guard at > .zero, let export = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetPassthrough) else {
            BugLog.note("capture: nothing to export")
            return false
        }
        export.outputURL = url
        export.outputFileType = .mp4
        await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in export.exportAsynchronously { k.resume() } }
        if export.status != .completed { BugLog.note("capture: export failed: \(export.error?.localizedDescription ?? "?")") }
        return export.status == .completed
    }
}

// MARK: - Reports

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
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        Snapshot.write(bytes: bytes, width: w, height: h, to: path)
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

// MARK: - Note window

// Asks what the user saw after a report is captured: Return saves note.txt, Option-Return starts a new line, Escape or
// the close button skips it. `done` runs once the window closes, or after five minutes.
final class BugNoteWindow: NSObject, NSTextViewDelegate, NSWindowDelegate {
    private let window: NSWindow
    private let text = NSTextView()
    private let dir: URL
    private var done: (() -> Void)?
    private var timeout: Timer?

    init(dir: URL, done: @escaping () -> Void) {
        self.dir = dir
        self.done = done
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 440, height: 210), styleMask: [.titled, .closable], backing: .buffered, defer: false)
        super.init()
        window.title = "What did you see?"
        window.isReleasedWhenClosed = false
        window.delegate = self
        let content = window.contentView!
        let hint = NSTextField(labelWithString: "Saved to Bug Reports/\(dir.lastPathComponent). Return saves; Option-Return for a new line.")
        hint.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        hint.textColor = .secondaryLabelColor
        hint.frame = NSRect(x: 16, y: 182, width: 408, height: 16)
        let scroll = NSScrollView(frame: NSRect(x: 16, y: 52, width: 408, height: 122))
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        text.frame = NSRect(origin: .zero, size: scroll.contentSize)
        text.autoresizingMask = [.width]
        text.isRichText = false
        text.font = .systemFont(ofSize: NSFont.systemFontSize)
        text.delegate = self
        scroll.documentView = text
        let save = NSButton(title: "Save", target: self, action: #selector(saveNote))
        save.keyEquivalent = "\r"
        save.frame = NSRect(x: 336, y: 12, width: 88, height: 32)
        let skip = NSButton(title: "Skip", target: self, action: #selector(skipNote))
        skip.keyEquivalent = "\u{1b}"
        skip.frame = NSRect(x: 244, y: 12, width: 88, height: 32)
        for v in [hint, scroll, save, skip] { content.addSubview(v) }
    }

    func show() {
        window.center()
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(text)
        timeout = Timer.scheduledTimer(withTimeInterval: 300, repeats: false) { [weak self] _ in self?.window.close() }
    }

    func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
        switch selector {
        case #selector(NSResponder.insertNewline(_:)): saveNote()
        case #selector(NSResponder.cancelOperation(_:)): skipNote()
        default: return false
        }
        return true
    }

    @objc private func saveNote() {
        let note = text.string.trimmingCharacters(in: .whitespacesAndNewlines)
        if !note.isEmpty { try? (note + "\n").write(to: dir.appendingPathComponent("note.txt"), atomically: true, encoding: .utf8) }
        BugLog.note("bug report: note \(note.isEmpty ? "empty" : "saved")")
        window.close()
    }

    @objc private func skipNote() { window.close() }

    func windowWillClose(_ notification: Notification) {
        timeout?.invalidate()
        timeout = nil
        let d = done
        done = nil
        d?()
    }
}
