import AppKit
import Metal
import StrataCore
import simd

// Offscreen render to PNG, for checking visuals without a display.
enum Snapshot {
    // False if the renderer could not start or a PNG could not be written.
    static func run(settings: Settings, path: String, gpu shared: GPU? = nil) -> Bool {
        guard let gpu = shared ?? MTLCreateSystemDefaultDevice().flatMap({ GPU(device: $0) }),
              let r = Renderer(gpu: gpu, outputFormat: .rgba8Unorm, seed: Biome.seed(settings.seed == 0 ? 1 : settings.seed, biome: settings.biome ?? 0),
                               screenIndex: 0, battery: settings.forceBattery, dither: !settings.noDither) else {
            FileHandle.standardError.write("renderer init failed\n".data(using: .utf8)!)
            return false
        }
        let device = gpu.device
        var wrote = true
        r.debugMode = settings.debugMode
        r.timeScale = settings.timeScale
        r.world.synchronous = true
        r.camera.pitchOverride = settings.pitch
        let w = Int(settings.screenshotSize.width), h = Int(settings.screenshotSize.height)
        r.renderScale = settings.renderScale ?? r.renderScale
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        let tex = device.makeTexture(descriptor: td)!
        r.pinnedPhase = settings.phase
        r.prepare(outputSize: CGSize(width: w, height: h))
        r.headroom = settings.headroom ?? 0
        let env = Knobs.current.env
        if let n = Int(env["VR_TOWERS"] ?? "") { listTowers(r, cells: n) }
        if let n = Int(env["VR_ISLES"] ?? "") { listIsles(r, count: n) }
        if let spec = env["VR_SHOW"] { showSite(r, spec: spec, pitch: settings.pitch) }
        if let spec = env["VR_CAVE"] { showCave(r, spec: spec, pitch: settings.pitch) }
        // VR_CAM=x,z,height,yawDeg,pitchDeg[,abs[,rollDeg]]: holds the camera at world (x, z), `height` above the ground
        // there (absolute when abs is 1).
        let camSpec = env["VR_CAM"].map { $0.split(separator: ",").compactMap { Double($0) } }
        if let v = camSpec, v.count >= 5 {
            r.camera.freeze(position: SIMD3(v[0], 400, v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            r.camera.pitchOverride = nil
        }
        if let mins = Double(env["VR_FLIGHT"] ?? "") { return fly(r, minutes: mins, output: tex, path: path) }
        // VR_BUGTEST=seconds: flies that long past the first time, recording as the saver does, then writes a bug report
        // beside the screenshot path, whose last frame is the screenshot.
        if let secs = Double(env["VR_BUGTEST"] ?? "") {
            BugReport.test(r, gpu: gpu, settings: settings, output: tex, path: path, seconds: secs)
            return wrote
        }
        var simulated = 0.0
        for t in settings.screenshotTimes.sorted() {
            r.simulate(t - simulated)
            if let v = camSpec, v.count >= 5 {
                let g = (v.count > 5 && v[5] == 1) ? 0 : max(r.world.terrain.height(x: v[0], z: v[1]) ?? 20, 19.85)
                r.camera.freeze(position: SIMD3(v[0], g + v[2], v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180,
                                roll: v.count > 6 ? v[6] * .pi / 180 : 0)
            }
            if let phase = settings.phase { r.clock.phase = phase }
            // VR_EXPOSURE=e: the exposure a bug report recorded (snapshots jump straight to the target the saver eases toward).
            if let e = Double(env["VR_EXPOSURE"] ?? "") { r.exposure = e }
            simulated = t
            if let pf = env["VR_PATH"], let text = try? String(contentsOfFile: pf, encoding: .utf8) {
                return replay(r, path: text, from: simulated, output: tex, to: path)
            }
            if let d = Double(env["VR_BIRDTEST"] ?? "") { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = Double(env["VR_MURMUR"] ?? "") { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            // VR_BENCH=n renders n still frames (gpu= is their minimum): more frames ride out the GPU's clock ramp.
            let frames = env["VR_MOTION"] != nil ? Int(env["VR_MOTION_FRAMES"] ?? "") ?? 48 : Int(env["VR_BENCH"] ?? "") ?? 12
            // VR_FRAMES=n also writes each of the last n frames, for measuring frame-to-frame crawl.
            let dump = min(Int(env["VR_FRAMES"] ?? "") ?? 0, frames)
            let out = settings.screenshotTimes.count > 1 ? path.replacingOccurrences(of: ".png", with: "_t\(Int(t)).png") : path
            for i in 0..<frames {
                // VR_POWER_FLIP=i switches between battery and AC power at frame i (render scale kept).
                if Int(env["VR_POWER_FLIP"] ?? "") == i { r.battery.toggle() }
                if env["VR_MOTION"] != nil { r.simulate(1.0 / 60) }
                // VR_DOLLY=v: a held camera moves v voxels a frame along its heading, level (approaches across the rings).
                if let v = Double(env["VR_DOLLY"] ?? ""), i > 0 {
                    let c = r.camera
                    c.freeze(position: c.position + SIMD3(sin(c.yaw), 0, cos(c.yaw)) * v, yaw: c.yaw, pitch: c.pitch)
                }
                let cb = render(r, into: tex)
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if Renderer.logTiming { print(String(format: "frame %d gpu %.2f ms cam (%.1f, %.1f, %.1f)%@", i, times.last!, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.clipmap.idle ? "" : " (coarse job pending)")) }
                if i >= frames - dump {
                    wrote = save(tex, to: out.replacingOccurrences(of: ".png", with: "_f\(i - frames + dump).png")) && wrote
                }
            }
            wrote = save(tex, to: out) && wrote
            let seen = r.wildlife.inView(camera: r.camera, aspect: Double(w) / Double(h), tanHalf: Renderer.tanV)
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) yaw=%.1f pitch=%.1f gpu=%.2fms mem=%.0fMB birds=%d flocks=%d near=%.0f %@ -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.camera.yaw * 180 / .pi, r.camera.pitch * 180 / .pi, times.min()!,
                         Double(device.currentAllocatedSize) / 1048576, seen.count, seen.flocks, seen.nearest.isFinite ? seen.nearest : -1, Snapshot.sitesInView(r), out))
        }
        if let log = r.popLog { print(log.summary(r.wildlife)) }
        return wrote
    }
}
