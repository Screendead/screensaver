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
        let knobs = Knobs.current
        if let n = knobs.towerList { listTowers(r, cells: n) }
        if let n = knobs.isleList { listIsles(r, count: n) }
        if let spec = knobs.show { showSite(r, spec: spec, pitch: settings.pitch) }
        if let spec = knobs.cave { showCave(r, spec: spec, pitch: settings.pitch) }
        let camSpec = knobs.cam
        if let v = camSpec, v.count >= 5 {
            r.camera.freeze(position: SIMD3(v[0], 400, v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            r.camera.pitchOverride = nil
        }
        if let mins = knobs.flight { return fly(r, minutes: mins, output: tex, path: path) }
        if let secs = knobs.bugTest {
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
            if let e = knobs.exposure { r.exposure = e }
            simulated = t
            if let pf = knobs.path, let text = try? String(contentsOfFile: pf, encoding: .utf8) {
                return replay(r, path: text, from: simulated, output: tex, to: path)
            }
            if let d = knobs.birdTest { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = knobs.murmur { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            let frames = knobs.motion ? knobs.motionFrames ?? 48 : knobs.bench ?? 12
            let dump = min(knobs.frames ?? 0, frames)
            let out = settings.screenshotTimes.count > 1 ? path.replacingOccurrences(of: ".png", with: "_t\(Int(t)).png") : path
            for i in 0..<frames {
                if knobs.powerFlip == i { r.battery.toggle() }
                if knobs.motion { r.simulate(1.0 / 60) }
                if let v = knobs.dolly, i > 0 {
                    let c = r.camera
                    c.freeze(position: c.position + SIMD3(sin(c.yaw), 0, cos(c.yaw)) * v, yaw: c.yaw, pitch: c.pitch)
                }
                let cb = render(r, into: tex)
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if Knobs.current.timing { print(String(format: "frame %d gpu %.2f ms cam (%.1f, %.1f, %.1f)%@", i, times.last!, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.clipmap.idle ? "" : " (coarse job pending)")) }
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
