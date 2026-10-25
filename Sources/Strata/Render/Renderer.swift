import AppKit
import MetalKit
import simd

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    // sceneFragment specialised by [sea][towers][GPU.sceneVariants]: constant 8 the sea of cloud, constant 1 the towers (only frames
    // with one in view need them), constant 0 the band test in the DDA (BAND_FC, only while a giant or the jungle is near,
    // or close by an island), and the islands' constants (GPU.SceneVariant). Each costs every pixel a little even unused,
    // so each tile of the scene pass takes the variant its pixels need.
    let scenePipelines: [[[MTLRenderPipelineState]]]
    // The giants' hanging strands, a pass over the scene while a giant stands within the world's edge (vineFragment), by [sea].
    let vinePipelines: [MTLRenderPipelineState]
    let postPipeline: MTLRenderPipelineState
    let taaPipeline: MTLRenderPipelineState
    let taaStarPipeline: MTLRenderPipelineState?
    // The glowing bugs (bioFragment), drawn over the scene in a pass of their own from sunset.
    let bioPipeline: MTLRenderPipelineState
    let cloudPipeline: MTLComputePipelineState
    // Shadow-height maps (see bakeShadowHeight in Shaders/Saver.metal): near and far cascades, a band rebaked per frame.
    let shadowPipeline: MTLComputePipelineState
    let shadowMaps: [MTLTexture]
    static let shadowN = 1024, shadowCell = 8.0
    static let shadowBands = [256, 512]
    var shadowBand = [0, 0]
    var shadowValid = false
    var shadowFrom: (cam: SIMD2<Double>, key: SIMD3<Double>)?
    // The terrain horizon around the camera and the islands' tangents (bakeHorizon): sky pixels above it and clear of
    // the islands skip the trace.
    let horizonPipeline: MTLComputePipelineState
    let horizonBuf: MTLBuffer
    // The province fields around the camera (bakeProvinces in Shaders/Saver.metal: jungle, old growth, islands), which
    // shading samples rather than evaluating per pixel; rebaked whole once the camera has moved an eighth of the window
    // (+-24.6k), so the coarsest ring (13.8k) never reads past it.
    let provPipeline: MTLComputePipelineState
    let provTex: MTLTexture
    var provCentre: SIMD2<Double>?
    static let provN = 512, provCell = 96.0
    let far = FarHeights()
    var farAge = 0
    // Level 3's generation last read back, and the one far.heights holds.
    var farRequested = -1, farHeld = -1
    private var farSimClock = -1e9
    // The fine cloud field only drifts with the wind, so it is baked once around the camera (in drifted
    // coordinates) and re-baked when the camera nears its edge.
    var cloudTex: MTLTexture?
    var cloudCenter: SIMD2<Double>?
    static let cloudTexels = 2048, cloudTexel = 20.0
    var slowTex: MTLTexture?
    let slowPipeline: MTLComputePipelineState
    static let slowTexels = 512, slowTexel = 100.0
    var history: [MTLTexture] = []
    // Lamps: per-frame position and state (schedule, flicker), and their light baked onto the map.
    let lampPipeline: MTLComputePipelineState
    let lightPipeline: MTLComputePipelineState
    let lampPos: MTLBuffer
    let lampState: MTLBuffer
    let lightMap: MTLTexture
    let lightDir: MTLTexture
    static let lightTexels = 512, hazeLamps = 16
    // Lamp indices: 16 haze lamps, then 12 smoke sources (and 4 spare).
    var hazeRefs = [Int32](repeating: 0, count: 32)
    static let smokePlumes = 12
    var smokeCount = 0
    // Per screen tile (LAMP_TILES x LAMP_TILES in Shaders/Saver.metal), the haze lamps (bits 0-15) and smoke plumes
    // (bits 16-27) whose bounding spheres the tile's rays may cross; the shader loops only over those.
    static let lampTiles = 16
    static let lampReach: [Double] = [1, 20, 16, 7, 10, 24, 22, 6]   // LAMP_R in Shaders/Saver.metal
    var tileMasks = [UInt32](repeating: 0, count: 16 * 16)
    static let ditherCells = 864.0
    // Sun height the glowing bugs start from (BIO_SUN in Shaders/Saver.metal).
    static let bioSun: Float = 0.07
    // Bug sprites: one per cube of the MOTE_N^3 block round the camera (bioVertex in Shaders/Saver.metal).
    static let moteCount = 23 * 23 * 23
    // Islands farther than this need no band test (past it the coarsest rings' columns are wide enough to walk under
    // them), shade without their sky occlusion and cast no island shade (only a long dusk shadow reaches back).
    static let isleShadeReach = 4000.0
    var isleVariant = false
    var isleShapeVariant = false
    // The scene pass is drawn in sceneTile-pixel squares, each with the cheapest variant its pixels need: the islands'
    // surfaces only where an island's boxes (Islands.boxes) cover it, their shade, sky occlusion and mist only where those
    // reach.
    static let sceneTile = 16
    var historyIndex = 0
    var prevCamera: (pos: SIMD3<Double>, fwd: SIMD3<Double>, right: SIMD3<Double>, up: SIMD3<Double>)?
    let world: World
    let clipmap: Clipmap
    let towers: Towers
    // The giants' hanging strands: their slots of baked maps and the frame's table.
    let vines: Vines
    let camera: FlightCamera
    let wildlife: Wildlife
    // Ring of bird buffers, one per frame in flight; the semaphore keeps the CPU from overwriting one the GPU reads.
    var birdBuffers: [MTLBuffer] = []
    var birdIndex = 0
    let birdSlots = DispatchSemaphore(value: 3)
    var clock = SkyClock()
    var battery: Bool
    // The world's edge, a circle around the camera: it eases between the battery and AC radii rather than snapping
    // when the power source changes.
    var edgeRadius: Double
    var timeScale: Double = 1
    var debugMode: Float = 0
    var headroom: Float = 0
    var sceneTex: MTLTexture?
    private(set) var simTime: Double = 0
    var lastFrame: CFTimeInterval?
    var frameIndex: Float = 0
    var gpuTimes: [Double] = []
    var sceneSize = (width: 0, height: 0)
    let popLog: PopLog? = PopLog.enabled ? PopLog() : nil
    var aspect: Float = 16.0 / 10.0
    static var logTiming: Bool { Knobs.current.timing }
    static var profile: String { Knobs.current.profile }
    static var fullCoarse: Bool { Knobs.current.fullCoarse }
    // VR_CHUNKED: snapshots update the coarse rings a chunk per frame, as live.
    static var chunkedCoarse: Bool { Knobs.current.chunkedCoarse }
    // VR_LODSHIFT=v moves the LOD rings' centre v voxels ahead of the camera: frames differing only in it isolate pops.
    static var lodShift: Float { Knobs.current.lodShift }
    var exposure: Double = 1.5
    // Holds the time of day fixed for snapshots, so exposure and the shot director see the rendered sun.
    var pinnedPhase: Double?
    var renderScale: Double
    let dither: Bool
    // The screen's size in pixels: the dither grid follows it, while the drawable holds just one pixel per cell.
    var screenPixels = CGSize.zero
    var recorder: BugRecorder?
    // Set by the S key: the next frame is saved here as a PNG at screen size.
    var shotPath: String?

    // Cover drifts between clear and broken skies over tens of minutes.
    var cloudCover: Double {
        0.52 + 0.12 * sin(simTime * 0.0021 + 1.0) + 0.06 * sin(simTime * 0.0057 + 4.0)
    }

    var hazeCount = 0

    init?(gpu: GPU, outputFormat: MTLPixelFormat, seed: UInt32, screenIndex: Int, battery: Bool, dither: Bool = true) {
        let device = gpu.device
        guard let queue = device.makeCommandQueue() else { return nil }
        let sceneStates = [false, true].map { s in
            [false, true].map { t in GPU.sceneVariants.compactMap { gpu.scene(towers: t, sea: s, $0) } }
        }
        let vineStates = [false, true].compactMap { gpu.vine(sea: $0) }
        guard sceneStates.allSatisfy({ $0.allSatisfy { $0.count == GPU.sceneVariants.count } }),
              vineStates.count == 2,
              let tp = gpu.taa(),
              let bp = gpu.bio(),
              let pp = gpu.post(outputFormat, dither: dither),
              let cf = gpu.pipeline("cloudField"),
              let csf = gpu.pipeline("cloudSlowField"),
              let shP = gpu.pipeline("bakeShadowHeight"),
              let hzP = gpu.pipeline("bakeHorizon"),
              let hzB = device.makeBuffer(length: 3 * 1024 * 4, options: .storageModePrivate),
              let pvP = gpu.pipeline("bakeProvinces"),
              let lampP = gpu.pipeline("updateLamps"),
              let lightP = gpu.pipeline("bakeLights"),
              let lpos = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let lstate = device.makeBuffer(length: World.siteGrid * World.siteGrid * World.maxLamps * 16, options: .storageModePrivate),
              let world = World(gpu: gpu, seed: seed),
              let clipmap = Clipmap(gpu: gpu, seed: seed),
              let towers = Towers(gpu: gpu, seed: seed),
              let vines = Vines(gpu: gpu) else { return nil }
        self.device = device
        self.queue = queue
        scenePipelines = sceneStates
        postPipeline = pp
        taaPipeline = tp
        taaStarPipeline = gpu.taa(star: true)
        bioPipeline = bp
        cloudPipeline = cf
        slowPipeline = csf
        shadowPipeline = shP
        horizonPipeline = hzP
        horizonBuf = hzB
        provPipeline = pvP
        let pd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Renderer.provN, height: Renderer.provN, mipmapped: false)
        pd.usage = [.shaderRead, .shaderWrite]
        pd.storageMode = .private
        guard let pt = device.makeTexture(descriptor: pd) else { return nil }
        provTex = pt
        // (S, Sa, island shade's floor and roof): halves hold heights to a voxel, and the sentinels.
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Renderer.shadowN, height: Renderer.shadowN, mipmapped: false)
        sd.usage = [.shaderRead, .shaderWrite]
        sd.storageMode = .private
        guard let sm0 = device.makeTexture(descriptor: sd), let sm1 = device.makeTexture(descriptor: sd) else { return nil }
        shadowMaps = [sm0, sm1]
        lampPipeline = lampP
        lightPipeline = lightP
        lampPos = lpos
        lampState = lstate
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Renderer.lightTexels, height: Renderer.lightTexels, mipmapped: false)
        ld.usage = [.shaderRead, .shaderWrite]
        ld.storageMode = .private
        guard let lmT = device.makeTexture(descriptor: ld), let ldT = device.makeTexture(descriptor: ld) else { return nil }
        lightMap = lmT
        lightDir = ldT
        self.world = world
        self.clipmap = clipmap
        self.towers = towers
        self.vines = vines
        vinePipelines = vineStates
        // Spec offset displays by 90 s of the same flight; steering reads GPU heights, so displays
        // instead start at different places in the same world.
        camera = FlightCamera(seed: seed &+ UInt32(screenIndex) &* 7919, world: world, far: far)
        wildlife = Wildlife(seed: seed &+ UInt32(screenIndex) &* 7919)
        birdBuffers = (0..<3).compactMap { _ in device.makeBuffer(length: Wildlife.bufferLength, options: .storageModeShared) }
        guard birdBuffers.count == 3 else { return nil }
        self.battery = battery
        self.dither = dither
        renderScale = battery ? 0.7 : 1.0
        edgeRadius = Clipmap.handoff(battery ? Clipmap.levels - 1 : Clipmap.levels)
        super.init()
        wildlife.heightAt = { [unowned(unsafe) world] x, z in world.height(x: x, z: z) }
        wildlife.giantsAt = { [unowned(unsafe) world] in world.giants }
        wildlife.columnAt = { [unowned(unsafe) world] x, z in world.column(x: x, z: z) }
        wildlife.islesAt = { [unowned(unsafe) world] in world.isles }
        camera.towersAt = { [unowned(unsafe) towers] in towers.list }
        camera.islesAt = { [unowned(unsafe) world] in world.isles }
        // VR_START=x,z,yawDeg: the flight starts there (tests).
        if let v = Knobs.current.env["VR_START"]?.split(separator: ",").compactMap({ Double($0) }), v.count >= 3 {
            camera.start(at: SIMD2(v[0], v[1]), yaw: v[2] * .pi / 180)
        }
        world.placeGiants(around: SIMD2(camera.position.x, camera.position.z))
        if Knobs.current.env["VR_START"] == nil, camera.startNearGiant() {
            world.placeGiants(around: SIMD2(camera.position.x, camera.position.z))
        }
        let pre = SIMD2(camera.position.x, camera.position.z)
        towers.update(center: pre, queue: queue, wait: true)
        // Placing the camera reads only the far heights, which level 3 holds: the CPU waits for that level alone.
        refreshFar(only: 3)
        camera.moveToOpenGround()
        let post = SIMD2(camera.position.x, camera.position.z)
        // The lift reads the map centred on `pre` where it reaches round `post`, else the far heights: that map is built only
        // if it reaches.
        let o = World.mapOrigin(pre), r = FlightCamera.liftRadius
        if post == pre || (post.x + r >= Double(o.x) && post.x - r < Double(o.x) + Double(World.size)
                           && post.y + r >= Double(o.y) && post.y - r < Double(o.y) + Double(World.size)) {
            world.ensureCentered(on: pre)
        }
        camera.liftOverGround()
        // A start moved off the massifs needs its own map now: on the first frame it would wait behind the coarse rings.
        world.ensureCentered(on: post)
        // The other rings are built ahead of the first frame on the same queue, which orders that frame after them;
        // nothing waits on the CPU. Afterwards only strips are rewritten.
        if let cb = queue.makeCommandBuffer() {
            let t0 = CACurrentMediaTime()
            clipmap.encode(into: cb, center: SIMD2(camera.position.x, camera.position.z), drain: true, active: activeLevels, giants: world.giantBuffer, isles: world.isleBuffer)
            if Renderer.logTiming {
                cb.addCompletedHandler { cb in
                    print(String(format: "coarse build %.1f ms (gpu %.1f)", (CACurrentMediaTime() - t0) * 1000, (cb.gpuEndTime - cb.gpuStartTime) * 1000))
                }
            }
            cb.commit()
        }
    }

    // Coarse levels kept up to date: on battery the outermost is dropped once the edge has eased inside it.
    var activeLevels: Int {
        battery && edgeRadius <= Clipmap.handoff(Clipmap.levels - 1) ? Clipmap.levels - 1 : Clipmap.levels
    }

    // Advances the simulation; large steps are subdivided so steering stays stable.
    func simulate(_ dt: Double) {
        var remaining = dt * timeScale
        // Snapshots simulate long stretches outside any run loop: without a pool per step, the autoreleased command
        // buffers of each map rebuild, and the world buffers they retain, pile up (1 GB per simulated half hour).
        while remaining > 0 { autoreleasepool {
            let step = min(remaining, 1.0 / 30.0)
            if let p = pinnedPhase { clock.phase = p }
            camera.sun = clock.sunDirection
            camera.nightIndex = clock.nightIndex
            camera.cloudSea = clock.cloudSea
            camera.step(dt: step)
            wildlife.step(dt: step, camera: camera, sight: sight(), sunY: clock.sunDirection.y, phase: clock.phase)
            popLog?.check(wildlife, sight: sight(), time: simTime + dt * timeScale - remaining + step, dt: step,
                          period: PopLog.period(sunY: clock.sunDirection.y, phase: clock.phase))
            if Renderer.logTiming, let g = world.height(x: camera.position.x, z: camera.position.z), camera.position.y < g + 2 {
                print(String(format: "camera low: y=%.1f ground=%.1f", camera.position.y, g))
            }
            clock.advance(step)
            if let p = pinnedPhase { clock.phase = p }
            world.ensureCentered(on: SIMD2(camera.position.x, camera.position.z))
            towers.update(center: SIMD2(camera.position.x, camera.position.z), queue: queue, wait: world.synchronous)
            // Snapshots simulate long stretches without frames: keep the far heights following the camera.
            if world.synchronous && simTime + (dt * timeScale - remaining) - farSimClock > 2 {
                farSimClock = simTime + (dt * timeScale - remaining)
                refreshFar()
            }
            remaining -= step
        } }
        simTime += dt * timeScale
        // Out to the AC edge only once its level is built (after battery mode it is rebuilt in pieces).
        let full = Clipmap.handoff(Clipmap.levels), part = Clipmap.handoff(Clipmap.levels - 1)
        let edgeTarget = battery || !clipmap.ready(Clipmap.levels) ? part : full
        let edgeStep = (full - part) / 4 * dt * timeScale
        edgeRadius = edgeRadius < edgeTarget ? min(edgeTarget, edgeRadius + edgeStep) : max(edgeTarget, edgeRadius - edgeStep)
        let e = clock.sunDirection.y
        let target = e > 0 ? 0.62 + 1.0 * max(0, 0.3 - e) / 0.3 + 0.08 * min(1, max(0, (e - 0.3) / 0.2)) : (e > -0.1 ? 2.02 + 0.4 * (-e / 0.1) : 1.6 + 0.4 * max(0, 0.3 + e) / 0.2)
        exposure += (target - exposure) * min(1, dt * timeScale * 0.2)
    }

    // What a frame rendered now could show; the frustum follows the scene size the last encode (or prepare) set.
    func sight() -> Sight {
        let b = camera.basis
        let tanV = tan(25.0 * .pi / 180)
        // As makeFrame's level-0 ring: its morph band starts 140 inside the handoff, less a cell for the floor's 4x4 grain.
        let p = camera.position - SIMD3(Double(world.origin.x), 0, Double(world.origin.y)), half = Double(World.size / 2)
        let edge = half - 8 - max(abs(p.x - half), abs(p.z - half)) - abs(Double(Renderer.lodShift))
        return Sight(pos: camera.position, fwd: b.fwd, right: b.right, up: b.up, tanV: tanV, aspect: Double(aspect),
                     pixelAngle: 2 * tanV / Double(max(sceneSize.height, 1)), sunY: clock.sunDirection.y,
                     worldEdge: Clipmap.handoff(Clipmap.levels), floorReach: min(880, edge) - 140 - 6 - abs(Double(Renderer.lodShift)),
                     world: world)
    }

    // Scene and history sizes for an output size: the retro dither shows one flat colour per cell, so the temporal
    // history only needs the cell grid, and the scene renders at (a fraction of) it. Without the dither the history
    // is every pixel.
    func sizes(_ outputSize: CGSize) -> (ow: Int, oh: Int, w: Int, h: Int) {
        let cell = dither ? max(1.0, (outputSize.width / Renderer.ditherCells).rounded()) : 1
        let ow = max(16, Int((outputSize.width / cell).rounded())), oh = max(16, Int((outputSize.height / cell).rounded()))
        return (ow, oh, max(16, Int(Double(ow) * renderScale)), max(16, Int(Double(oh) * renderScale)))
    }

    // Sets the frustum the simulation sees before the first frame is encoded.
    func prepare(outputSize: CGSize) {
        let s = sizes(outputSize)
        sceneSize = (s.w, s.h)
        aspect = Float(s.w) / Float(s.h)
    }

    // A queue runs its command buffers in order, so an empty one completing means all committed before it have.
    func drain() {
        for q in [queue, world.genQueue] {
            guard let cb = q.makeCommandBuffer() else { continue }
            cb.commit()
            cb.waitUntilCompleted()
        }
    }
}
