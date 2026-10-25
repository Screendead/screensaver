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
        // VR_TOWERS=n: lists the towers of the n-cell square of tower cells round the camera's start, nearest first.
        if let n = Int(env["VR_TOWERS"] ?? "") {
            let here = SIMD2(r.camera.position.x, r.camera.position.z)
            for t in r.towers.scan(around: here, cells: n, queue: r.queue).sorted(by: { simd_length($0.pos - here) < simd_length($1.pos - here) }) {
                print(String(format: "tower %.0f,%.0f dist %.0f kind %d member %d R0 %.0f base %.0f height %.0f top %.0f lit %d moist %.2f temp %.2f",
                             t.pos.x, t.pos.y, simd_length(t.pos - here), t.kind, t.member, t.radius, t.base, t.height, t.top, t.lit ? 1 : 0, t.moist, t.temp))
            }
        }
        // VR_ISLES=n: lists the n floating islands nearest the camera's start on its map's table.
        if let n = Int(env["VR_ISLES"] ?? "") {
            let here = SIMD2(r.camera.position.x, r.camera.position.z)
            let list = r.world.terrain.isles.sorted { simd_length($0.pos - here) < simd_length($1.pos - here) }
            print("islands on the table: \(list.count) (massifs \(list.filter(\.massif).count))")
            for s in list.prefix(n) {
                print(String(format: "island %.0f,%.0f dist %.0f R %.0f reach %.0f top %.0f keel %.0f base %.0f%@", s.pos.x, s.pos.y,
                             simd_length(s.pos - here), s.radius, s.reach, s.top, s.keel, s.base, (s.massif ? " massif" : "")
                             + (s.fall.map { String(format: " fall lip %.0f,%.0f top %.0f foot %.0f", $0.x, $0.y, $0.z, $0.w) } ?? "")))
            }
        }
        // VR_SHOW=kind,distance,height,bearing[,rank]: hold the camera on the rank-th nearest site of a kind
        // (1 henge .. 6 camp), `distance` from it along `bearing` degrees, `height` above the ground there.
        if let spec = env["VR_SHOW"] {
            let v = spec.split(separator: ",").compactMap { Double($0) }
            if v.count >= 4 {
                let here = SIMD2(r.camera.position.x, r.camera.position.z)
                let pick = r.world.terrain.sites.filter { $0.kind.rawValue == Int(v[0]) }
                    .sorted { simd_length($0.pos - here) < simd_length($1.pos - here) }
                let rank = v.count > 4 ? Int(v[4]) : 0
                if rank < pick.count {
                    let site = pick[rank]
                    let b = v[3] * .pi / 180
                    let cam = site.pos + SIMD2(sin(b), cos(b)) * v[1]
                    let g = r.world.terrain.height(x: cam.x, z: cam.y) ?? 20
                    let sg = r.world.terrain.height(x: site.pos.x, z: site.pos.y) ?? 20
                    let y = max(g, 19.85) + v[2]
                    let yaw = atan2(site.pos.x - cam.x, site.pos.y - cam.y)
                    let pitch = atan2(sg + 2 - y, v[1]) + (settings.pitch ?? 0)
                    r.camera.freeze(position: SIMD3(cam.x, y, cam.y), yaw: yaw, pitch: pitch)
                    r.camera.pitchOverride = nil
                    print(String(format: "showing %@ site %u at (%.0f, %.0f), %d of %d", "\(site.kind)", site.id, site.pos.x, site.pos.y, rank + 1, pick.count))
                } else {
                    print("no site of kind \(Int(v[0])) rank \(rank) (\(pick.count) on this map)")
                }
            }
        }
        // VR_CAVE=distance,height,bearing[,rank[,homes]]: hold the camera on the rank-th nearest cave (with homes 1, of those
        // someone lives in), `distance` out from its mouth turned `bearing` degrees off its axis, `height` over the ground
        // there or the cave's floor, whichever is higher, looking at the mouth.
        if let spec = env["VR_CAVE"] {
            let v = spec.split(separator: ",").compactMap { Double($0) }
            if v.count >= 3 {
                let here = SIMD2<Float>(Float(r.camera.position.x), Float(r.camera.position.z))
                // homes: 1 lived-in caves only; 2 any cave of a good size seen clear across `distance` (by the CPU heights); 3 grottoes.
                let b0 = v[2] * .pi / 180
                func clear(_ c: CaveRec) -> Bool {
                    let m = SIMD2(Double(c.m.x), Double(c.m.y)), u = SIMD2(Double(c.u.x), Double(c.u.y))
                    let dir = SIMD2(u.x * cos(b0) - u.y * sin(b0), u.x * sin(b0) + u.y * cos(b0))
                    guard c.w >= 6, c.hc >= 8, let g = r.world.terrain.height(x: m.x - dir.x * v[0], z: m.y - dir.y * v[0]) else { return false }
                    let y0 = max(g, Double(c.yF)) + v[1], y1 = Double(c.yF) + 0.5 * Double(c.hc)
                    for t in stride(from: 12.0, to: v[0], by: 6.0) {
                        let q = m - dir * t
                        guard let h = r.world.terrain.height(x: q.x, z: q.y) else { return false }
                        if h > y1 + (y0 - y1) * t / v[0] { return false }
                    }
                    return true
                }
                let mode = v.count > 4 ? Int(v[4]) : 0
                let pick = r.world.terrain.caves.filter { mode == 0 || (mode == 1 && $0.flags & 2 != 0) || (mode == 2 && clear($0)) || (mode == 3 && $0.flags & 4 != 0) }
                    .sorted { simd_length($0.m - here) < simd_length($1.m - here) }
                let rank = v.count > 3 ? Int(v[3]) : 0
                if rank < pick.count {
                    let c = pick[rank]
                    let m = SIMD2(Double(c.m.x), Double(c.m.y)), u = SIMD2(Double(c.u.x), Double(c.u.y))
                    let b = v[2] * .pi / 180
                    let dir = SIMD2(u.x * cos(b) - u.y * sin(b), u.x * sin(b) + u.y * cos(b))
                    let cam = m - dir * v[0]
                    let g = r.world.terrain.height(x: cam.x, z: cam.y) ?? 20
                    let y = max(max(g, 19.85), Double(c.yF)) + v[1]
                    let yaw = atan2(m.x - cam.x, m.y - cam.y)
                    let pitch = atan2(Double(c.yF) + 0.4 * Double(c.hc) - y, v[0]) + (settings.pitch ?? 0)
                    r.camera.freeze(position: SIMD3(cam.x, y, cam.y), yaw: yaw, pitch: pitch)
                    r.camera.pitchOverride = nil
                    print(String(format: "showing cave at (%.0f, %.0f) floor %.0f W %.1f H %.0f L %.0f%@, %d of %d; VR_CAM=%.1f,%.1f,%.1f,%.2f,%.2f,1", c.m.x, c.m.y, c.yF, c.w, c.hc, c.l,
                                 c.flags & 2 != 0 ? " home" : "", rank + 1, pick.count, cam.x, cam.y, y, yaw * 180 / .pi, pitch * 180 / .pi))
                } else {
                    print("no cave rank \(rank) (\(pick.count) in the table)")
                }
            }
        }
        // VR_CAM=x,z,height,yawDeg,pitchDeg[,abs[,rollDeg]]: holds the camera at world (x, z), `height` above the ground
        // there (absolute when abs is 1).
        let camSpec = env["VR_CAM"].map { $0.split(separator: ",").compactMap { Double($0) } }
        if let v = camSpec, v.count >= 5 {
            r.camera.freeze(position: SIMD3(v[0], 400, v[1]), yaw: v[3] * .pi / 180, pitch: v[4] * .pi / 180)
            r.camera.pitchOverride = nil
        }
        // VR_FLIGHT=minutes: flies the autopilot that long, logging altitude, clearance and climb every 12 s and
        // writing a frame each minute next to the screenshot path. VR_FTRACE=1 also prints every 0.25 s: time, x, z, y, ground,
        // yaw, the altitude terms (target, climb rate, need (-1: floor clamp), rim lift, regional floor, ahead, planned climb)
        // and the routing (blocked ahead, route offset).
        if let mins = Double(env["VR_FLIGHT"] ?? "") {
            r.camera.tracing = env["VR_FTRACE"] != nil
            var maxClimb = 0.0, minClear = Double.infinity, maxY = 0.0, lows = 0
            var walled = 0, walledRun = 0, maxWalledRun = 0, fastClimb = 0, prevY = r.camera.position.y
            var lastY = r.camera.position.y
            var underCrown = 0, giantsSeen = Set<Int>()
            // Ruined towers: the closest the flight came to a plinth's edge below its top, and the seconds a tower stood in
            // frame within 9k.
            var towerEdge = Double.infinity, towerSeen = 0, towerClose = 0
            var towersMet = Set<SIMD2<Int>>()
            // Islands: the closest the flight came to one's rock (from the CPU cells within 24), the seconds it was nearer
            // than 25, the seconds it spent beneath one, and the islands it passed under.
            var isleClear = Double.infinity, isleScrape = 0, isleUnder = 0, birdsInRock = 0, birdChecks = 0
            var islesUnder = Set<SIMD2<Int>>()
            // Seconds with an island's reach within 400 on either side, and flocks circling an island (by serial).
            var isleBetween = 0, circlingSeconds = 0
            var circlers = Set<Int>()
            for i in 1...Int(mins * 60) {
                for q in 0..<4 {
                    r.simulate(0.25)
                    if env["VR_FTRACE"] != nil {
                        let c = r.camera.position
                        print(String(format: "ft %.2f %.1f %.1f %.2f %.2f %.3f", Double(i - 1) + Double(q + 1) * 0.25, c.x, c.z, c.y, r.world.terrain.height(x: c.x, z: c.z) ?? -1, r.camera.yaw)
                              + " " + r.camera.terms.map { String(format: "%.1f", $0) }.joined(separator: " "))
                    }
                }
                let p = r.camera.position
                let col = r.world.terrain.column(x: p.x, z: p.z)
                let g = col.map { c in c.under.map { p.y < $0 ? c.ground : c.top } ?? c.top } ?? r.world.terrain.height(x: p.x, z: p.z) ?? 0
                var near = Double.infinity
                for (dx, dz) in [(0.0, 0.0)] + (0..<16).map({ (k: Int) -> (Double, Double) in
                    let a = Double(k % 8) * .pi / 4, rr = k < 8 ? 12.0 : 24.0
                    return (sin(a) * rr, cos(a) * rr)
                }) {
                    guard let c = r.world.terrain.column(x: p.x + dx, z: p.z + dz), let u = c.under else { continue }
                    let v = p.y < u ? u - p.y : p.y > c.top ? p.y - c.top : 0
                    near = min(near, (dx * dx + dz * dz + v * v).squareRoot())
                }
                isleClear = min(isleClear, near)
                // Birds inside an island's rock (its cell's span), which Sight cannot see as hidden.
                for f in r.wildlife.flocks {
                    for b in f.birds {
                        birdChecks += 1
                        if let c = r.world.terrain.column(x: b.pos.x, z: b.pos.z), let u = c.under, b.pos.y >= u, b.pos.y < c.top { birdsInRock += 1 }
                    }
                }
                if near < 25 { isleScrape += 1; print(String(format: "flight island scrape t=%d pos=(%.0f,%.0f) y=%.0f clearance %.0f", i, p.x, p.z, p.y, near)) }
                do {
                    // Between two islands: one abreast on each side (its reach spans the camera along the track), within 400 of its rock.
                    let side = SIMD2(cos(r.camera.yaw), -sin(r.camera.yaw)), fwd = SIMD2(sin(r.camera.yaw), cos(r.camera.yaw))
                    var l = false, rt = false
                    for s in r.world.terrain.isles where s.radius > 40 {
                        let v = s.pos - SIMD2(p.x, p.z), across = simd_dot(v, side), along = simd_dot(v, fwd)
                        if abs(along) < s.reach, abs(across) > s.reach * 0.5, abs(across) < s.reach + 400 { if across > 0 { rt = true } else { l = true } }
                    }
                    if l && rt { isleBetween += 1 }
                    let circ = r.wildlife.flocks.filter { $0.orbit != nil && !$0.leaving }
                    if !circ.isEmpty { circlingSeconds += 1 }
                    for f in circ { circlers.insert(f.serial) }
                }
                if let u = col?.under, p.y < u {
                    isleUnder += 1
                    if let s = r.world.terrain.isles.first(where: { simd_length($0.pos - SIMD2(p.x, p.z)) < $0.reach }) { islesUnder.insert(SIMD2(Int(s.pos.x), Int(s.pos.y))) }
                }
                if p.y - lastY > 6 { print(String(format: "flight jump t=%d pos=(%.0f,%.0f) %.0f -> %.0f", i, p.x, p.z, lastY, p.y)) }
                maxClimb = max(maxClimb, p.y - lastY)
                lastY = p.y
                minClear = min(minClear, p.y - g)
                maxY = max(maxY, p.y)
                if p.y < g + 2 { lows += 1 }
                var seen = false
                for t in r.towers.list {
                    let v = t.pos - SIMD2(p.x, p.z), d = simd_length(v)
                    if p.y < t.top { towerEdge = min(towerEdge, d - t.radius * 1.42) }
                    if d < 9000, abs(atan2(sin(atan2(v.x, v.y) - r.camera.yaw), cos(atan2(v.x, v.y) - r.camera.yaw))) < 0.6 {
                        seen = true
                        if towersMet.insert(SIMD2(Int(t.pos.x), Int(t.pos.y))).inserted {
                            print(String(format: "flight tower seen t=%d at %.0f,%.0f member %d dist %.0f", i, t.pos.x, t.pos.y, t.member, d))
                        }
                    }
                    if d < 1500 { towerClose += 1 }
                }
                if seen { towerSeen += 1 }
                let wall = r.camera.wallInFrame()
                if wall >= 0.4 { walled += 1; walledRun += 1; maxWalledRun = max(maxWalledRun, walledRun) } else { walledRun = 0 }
                if p.y - (i == 1 ? p.y : prevY) > 6 { fastClimb += 1 }
                for g in r.world.terrain.giants {
                    let dg = simd_length(SIMD2(p.x, p.z) - g.pos)
                    if dg < GiantInfo.reach && p.y < g.top { underCrown += 1 }
                    if dg < 1500 { giantsSeen.insert(Int(g.pos.x) * 100_003 + Int(g.pos.y)) }
                }
                prevY = p.y
                if i % 12 == 0 {
                    fflush(stdout)
                    let nearG = r.world.terrain.giants.map { simd_length(SIMD2(p.x, p.z) - $0.pos) }.min() ?? -1
                    let nearI = r.world.terrain.isles.map { simd_length(SIMD2(p.x, p.z) - $0.pos) - $0.reach }.min() ?? -1
                    print(String(format: "flight t=%d pos=(%.0f,%.0f) y=%.0f ground=%.0f wall=%.2f far=%.0f yaw=%.0f giant=%.0f isle=%.0f", i, p.x, p.z, p.y, g, wall,
                                 r.farHeight(x: p.x + sin(r.camera.yaw) * 600, z: p.z + cos(r.camera.yaw) * 600) ?? -1, r.camera.yaw * 180 / .pi, nearG, nearI))
                }
                if i % 60 == 0 {
                    for _ in 0..<4 {
                        let cb = r.queue.makeCommandBuffer()!
                        r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                        cb.commit()
                        cb.waitUntilCompleted()
                    }
                    var bytes = [UInt8](repeating: 0, count: w * h * 4)
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    wrote = write(bytes: bytes, width: w, height: h, to: path.replacingOccurrences(of: ".png", with: "_m\(i / 60).png")) && wrote
                }
            }
            print(String(format: "flight towers: closest plinth edge %.0f, seconds with a tower ahead within 9k %d, tower-seconds within 1.5k %d", towerEdge, towerSeen, towerClose))
            print(String(format: "flight islands: closest %.0f, seconds nearer than 25 %d, seconds beneath one %d, islands passed under %d, on the table %d, bird-seconds in island cells %d of %d",
                         isleClear, isleScrape, isleUnder, islesUnder.count, r.world.terrain.isles.count, birdsInRock, birdChecks))
            print(String(format: "flight islands 2: seconds between two %d, seconds with a flock circling one %d, flocks that circled %d", isleBetween, circlingSeconds, circlers.count))
            print(String(format: "flight done: max climb %.2f/s, min clearance %.1f, max y %.0f (ceiling %.0f), low seconds %d, walled seconds %d (longest %d), seconds climbing over 6/s %d, seconds under a crown %d, giants passed within 1500 %d", maxClimb, minClear, maxY, FlightCamera.ceiling, lows, walled, maxWalledRun, fastClimb, underCrown, giantsSeen.count))
            if let log = r.popLog { print(log.summary(r.wildlife)) }
            return wrote
        }
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
            // VR_PATH=file: renders each line "simTime x z y yawDeg pitchDeg rollDeg exposure [phase day]" (a bug report's
            // recorded frames, bugclip.sh) in turn as <screenshot>_pNNNN.png, carrying the temporal history across them as live.
            // Phase and day hold the sky clock where the report's frame had it: the menu's Time of Day pins the phase in the
            // saver, and the day picks each night's lit beacons and rites.
            if let pf = env["VR_PATH"], let text = try? String(contentsOfFile: pf, encoding: .utf8) {
                var bytes = [UInt8](repeating: 0, count: w * h * 4)
                var n = 0
                for line in text.split(separator: "\n") {
                    let v = line.split(separator: " ").compactMap { Double($0) }
                    guard v.count == 8 || v.count == 10 else { continue }
                    if v.count == 10 { r.pinnedPhase = v[8] }
                    r.simulate(max(v[0] - simulated, 0))
                    simulated = v[0]
                    if v.count == 10 {
                        r.clock.phase = v[8]
                        r.clock.day = Int(v[9])
                        r.camera.sun = r.clock.sunDirection
                        r.camera.nightIndex = r.clock.nightIndex
                        r.camera.cloudSea = r.clock.cloudSea
                    }
                    r.camera.freeze(position: SIMD3(v[1], v[3], v[2]), yaw: v[4] * .pi / 180, pitch: v[5] * .pi / 180, roll: v[6] * .pi / 180)
                    r.exposure = v[7]
                    let cb = r.queue.makeCommandBuffer()!
                    r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                    cb.commit()
                    cb.waitUntilCompleted()
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    wrote = write(bytes: bytes, width: w, height: h, to: path.replacingOccurrences(of: ".png", with: String(format: "_p%04d.png", n))) && wrote
                    n += 1
                }
                return wrote
            }
            if let d = Double(env["VR_BIRDTEST"] ?? "") { r.wildlife.testPattern(camera: r.camera, distance: d) }
            if let d = Double(env["VR_MURMUR"] ?? "") { r.wildlife.testMurmuration(camera: r.camera, distance: d) }
            var times: [Double] = []
            // VR_BENCH=n renders n still frames (gpu= is their minimum): more frames ride out the GPU's clock ramp.
            let frames = env["VR_MOTION"] != nil ? Int(env["VR_MOTION_FRAMES"] ?? "") ?? 48 : Int(env["VR_BENCH"] ?? "") ?? 12
            // VR_FRAMES=n also writes each of the last n frames, for measuring frame-to-frame crawl.
            let dump = min(Int(env["VR_FRAMES"] ?? "") ?? 0, frames)
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
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
                let cb = r.queue.makeCommandBuffer()!
                r.encode(into: cb, output: tex, outputSize: CGSize(width: w, height: h))
                cb.commit()
                cb.waitUntilCompleted()
                times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                if Renderer.logTiming { print(String(format: "frame %d gpu %.2f ms cam (%.1f, %.1f, %.1f)%@", i, times.last!, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.clipmap.idle ? "" : " (coarse job pending)")) }
                if i >= frames - dump {
                    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                    wrote = write(bytes: bytes, width: w, height: h, to: out.replacingOccurrences(of: ".png", with: "_f\(i - frames + dump).png")) && wrote
                }
            }
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            wrote = write(bytes: bytes, width: w, height: h, to: out) && wrote
            let seen = r.wildlife.inView(camera: r.camera, aspect: Double(w) / Double(h), tanHalf: tan(25.0 * .pi / 180))
            print(String(format: "t=%.0f phase=%.3f sunY=%.2f exposure=%.2f cam=(%.0f,%.0f,%.0f) yaw=%.1f pitch=%.1f gpu=%.2fms mem=%.0fMB birds=%d flocks=%d near=%.0f %@ -> %@", t, r.clock.phase,
                         r.clock.sunDirection.y, r.exposure, r.camera.position.x, r.camera.position.y, r.camera.position.z, r.camera.yaw * 180 / .pi, r.camera.pitch * 180 / .pi, times.min()!,
                         Double(device.currentAllocatedSize) / 1048576, seen.count, seen.flocks, seen.nearest.isFinite ? seen.nearest : -1, Snapshot.sitesInView(r), out))
        }
        if let log = r.popLog { print(log.summary(r.wildlife)) }
        return wrote
    }

    // Sites within 700 voxels whose centre falls inside the frame (not tested for occlusion).
    static func sitesInView(_ r: Renderer) -> String {
        let b = r.camera.basis
        let o = r.camera.position
        let tanV = tan(25.0 * .pi / 180), tanH = tanV * Double(r.aspect)
        var seen: [String] = []
        for site in r.world.terrain.sites {
            let g = r.world.terrain.height(x: site.pos.x, z: site.pos.y) ?? 20
            let v = SIMD3(site.pos.x, g + 1, site.pos.y) - o
            let z = simd_dot(v, b.fwd), dist = simd_length(v)
            guard z > 0, dist < 700 else { continue }
            if abs(simd_dot(v, b.right)) / z < tanH && abs(simd_dot(v, b.up)) / z < tanV {
                seen.append("\(site.kind)@\(Int(dist))")
            }
        }
        return "sites=\(seen.count)[\(seen.joined(separator: " "))]"
    }
}
