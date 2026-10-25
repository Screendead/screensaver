import AVFoundation
import CoreVideo
import Foundation
import Metal
import QuartzCore
import StrataCore
import VideoToolbox
import simd

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
