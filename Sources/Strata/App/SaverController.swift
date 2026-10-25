import AppKit
import MetalKit
import StrataCore

final class SaverController {
    private var windows: [SaverWindow] = []
    private var renderers: [(Renderer, MTKView)] = []

    func prefsChanged() {
        let phase = Prefs.timeOfDay
        for (r, _) in renderers { r.pinnedPhase = phase }
        // A new biome is a new world: fade out and start again in it.
        if isShowing, !dismissing, settings.biome == nil, Prefs.biome != shownBiome {
            restartAfterDismiss = true
            dismiss()
        }
    }
    private var shownBiome = 0
    private var restartAfterDismiss = false
    private var watchTimer: Timer?
    private var shownAt: TimeInterval = 0 // monotonic
    private var previousApp: NSRunningApplication?
    private var dismissing = false
    private var battery = false
    // Bug reports: the first screen's recorder while showing, and the report a B press is finishing.
    private var recorder: BugRecorder?
    private var pendingReport: URL?
    private var noteWindow: BugNoteWindow?
    private var awaitingReport = false
    let settings: Settings
    var onDismiss: (() -> Void)?

    var isShowing: Bool { !windows.isEmpty }
    private(set) var screenFrames: [CGRect] = []

    init(settings: Settings) { self.settings = settings }

    private static var now: TimeInterval { ProcessInfo.processInfo.systemUptime }

    // A manual start follows a click or key press that is still within the idle clock's reach, so input counts only once
    // the grace has passed.
    private var inputGrace: TimeInterval = 0

    private func inputReceived() {
        if SaverController.now - shownAt >= inputGrace { dismiss() }
    }

    func show(manual: Bool = false) {
        let screens = NSScreen.screens
        guard !isShowing, !awaitingReport, !screens.isEmpty else { return }
        dismissing = false
        let front = NSWorkspace.shared.frontmostApplication
        previousApp = front?.processIdentifier == ProcessInfo.processInfo.processIdentifier ? nil : front
        battery = settings.forceBattery || SystemState.onBattery()
        // A different seed per session keeps the world non-repeating across activations.
        shownBiome = settings.biome ?? Prefs.biome
        let sessionSeed = Biome.seed(settings.seed != 0 ? settings.seed : UInt32.random(in: 1...Biome.maxSeed), biome: shownBiome)
        let gpu = MTLCreateSystemDefaultDevice().flatMap { GPU(device: $0) }

        screenFrames = screens.map(\.frame)
        for (index, screen) in screens.enumerated() {
            let window = SaverWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.level = .screenSaver
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
            window.backgroundColor = .black
            window.isOpaque = true
            window.hasShadow = false
            window.alphaValue = 0
            window.isReleasedWhenClosed = false
            window.setFrame(screen.frame, display: false)
            window.onInput = { [weak self] in self?.inputReceived() }
            window.onBugKey = { [weak self] in self?.reportBug() }
            window.onShotKey = { [weak self] in self?.screenshot() }

            // A screen whose renderer can't start still gets covered, in black.
            if let gpu {
                let view = MTKView(frame: CGRect(origin: .zero, size: screen.frame.size), device: gpu.device)
                if let renderer = Renderer(view: view, gpu: gpu, screen: screen, seed: sessionSeed, screenIndex: index, battery: battery, dither: !settings.noDither && Prefs.dither) {
                    renderer.timeScale = settings.timeScale
                    renderer.pinnedPhase = settings.phase ?? Prefs.timeOfDay
                    window.contentView = view
                    renderers.append((renderer, view))
                    let ds = view.drawableSize
                    // The recorder and the S key read the first screen's drawable.
                    if index == 0 { view.framebufferOnly = false }
                    if index == 0, let rec = BugRecorder(gpu: gpu, width: Int(ds.width), height: Int(ds.height), info: [
                        "seed": sessionSeed, "screenIndex": index, "screenCount": screens.count, "screenName": screen.localizedName,
                        "screenPoints": [Int(screen.frame.width), Int(screen.frame.height)], "backingScale": screen.backingScaleFactor,
                        "screenPixels": [Int(renderer.screenPixels.width), Int(renderer.screenPixels.height)],
                        "drawable": [Int(ds.width), Int(ds.height)], "edr": view.colorPixelFormat == .rgba16Float,
                        "dither": !settings.noDither && Prefs.dither, "manualStart": manual,
                    ]) {
                        renderer.recorder = rec
                        recorder = rec
                    }
                }
            }
            windows.append(window)
            window.orderFrontRegardless()
        }

        NSApp.activate(ignoringOtherApps: true)
        windows.first?.makeKey()
        SystemState.allowCursorHidingInBackground()
        NSCursor.hide()
        NSApp.presentationOptions = [.hideDock, .hideMenuBar]

        BugLog.note("show: seed \(sessionSeed), screens \(screens.count), renderers \(renderers.count), battery \(battery), manual \(manual), active \(NSApp.isActive)")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { dlog("show+1s: active=\(NSApp.isActive) keyWindow=\(NSApp.keyWindow != nil)") }
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 2.0
            ctx.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            for w in windows { w.animator().alphaValue = 1 }
        }

        shownAt = SaverController.now
        inputGrace = manual ? 1.5 : 0
        inputFloor = shownAt + inputGrace
        var lastPowerCheck = shownAt
        watchTimer?.invalidate()
        watchTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let age = SaverController.now - self.shownAt
            // Any physical input after the moment we appeared, including during the fade-in.
            let idle = SystemState.idleSeconds()
            let shotHeld = CGEventSource.keyState(.hidSystemState, key: SaverController.shotKey)
            if idle < SaverController.now - self.inputFloor - 0.15 {
                // The key event itself may not arrive (macOS can refuse to activate the agent), so B and S are read from the HID state.
                if CGEventSource.keyState(.hidSystemState, key: BugReport.key) { self.reportBug(); return }
                if shotHeld { self.screenshot(); return }
                BugLog.note(String(format: "input detected: idle %.2f s, shown %.1f s", idle, age))
                self.dismiss()
                return
            }
            // While S is held its autorepeat and release are not input.
            if shotHeld { self.inputFloor = SaverController.now }
            if self.settings.previewDuration > 0 && age > self.settings.previewDuration { self.dismiss(); return }
            if SaverController.now - lastPowerCheck > 5 {
                lastPowerCheck = SaverController.now
                let b = self.settings.forceBattery || SystemState.onBattery()
                if b != self.battery {
                    BugLog.note("power: battery \(b)")
                    self.battery = b
                    for (r, v) in self.renderers { r.setBattery(b, view: v) }
                }
            }
        }
        watchTimer?.tolerance = 0.005
        RunLoop.main.add(watchTimer!, forMode: .common)
    }

    static let shotKey: CGKeyCode = 1 // kVK_ANSI_S
    static var shotFolder: URL {
        FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0].appendingPathComponent("STRATA", isDirectory: true)
    }
    // Input older than this is not a reason to dismiss.
    private var inputFloor: TimeInterval = 0
    private var lastShot: TimeInterval = -1

    // S: saves the first screen's next frame and keeps the saver showing.
    private func screenshot() {
        let now = SaverController.now
        inputFloor = now
        guard isShowing, !dismissing, now - lastShot > 0.5, let r = renderers.first?.0 else { return }
        lastShot = now
        let dir = SaverController.shotFolder
        guard (try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)) != nil else { return }
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        r.shotPath = dir.appendingPathComponent("STRATA \(f.string(from: Date())).png").path
    }

    // B: writes the report's state now, finishes the clip in the background, and asks for a note once the saver is gone.
    private func reportBug() {
        guard isShowing, !dismissing else { return }
        guard let rec = recorder, let r = renderers.first?.0, let dir = BugReport.makeDirectory() else { dismiss(); return }
        BugLog.note("bug report: B pressed, \(dir.lastPathComponent)")
        BugReport.write(to: dir, renderer: r, recorder: rec, binary: Bundle.main.executablePath ?? CommandLine.arguments[0])
        recorder = nil
        pendingReport = dir
        awaitingReport = true
        reportWait.enter()
        rec.finish(to: dir.appendingPathComponent("clip.mp4")) { [weak self] ok in
            DispatchQueue.main.async {
                BugLog.note("bug report: clip \(ok ? "written" : "failed")")
                self?.reportWait.leave()
            }
        }
        dismiss()
    }

    private let reportWait = DispatchGroup()
    private var reportSerial = 0

    private func askForNote(_ dir: URL) {
        reportWait.enter()
        let note = BugNoteWindow(dir: dir) { [weak self] in
            self?.noteWindow = nil
            self?.reportWait.leave()
        }
        noteWindow = note
        note.show()
        reportSerial += 1
        let serial = reportSerial
        let end = { [weak self] in
            guard let self, self.awaitingReport, self.reportSerial == serial else { return }
            self.awaitingReport = false
            if NSApp.isActive {
                if let prev = self.previousApp, !prev.isTerminated { prev.activate() } else { NSApp.hide(nil) }
            }
            self.onDismiss?()
        }
        reportWait.notify(queue: .main, execute: end)
        // The note window gives up after 5 minutes; this covers a clip export that never returns.
        DispatchQueue.main.asyncAfter(deadline: .now() + 310, execute: end)
    }

    func dismiss(animated: Bool = true) {
        guard isShowing, !dismissing else { return }
        dismissing = true
        watchTimer?.invalidate()
        watchTimer = nil
        let closing = windows
        for w in closing { w.ignoresMouseEvents = true }
        // Hand focus back immediately so typing during the fade lands where the user expects, but only
        // if we still hold it (the user may already have switched apps).
        if NSApp.isActive && pendingReport == nil {
            if let prev = previousApp, !prev.isTerminated { prev.activate() } else { NSApp.hide(nil) }
        }
        NSCursor.unhide()
        let finish = { [weak self] in
            guard let self else { return }
            for (_, v) in self.renderers { v.isPaused = true; v.delegate = nil }
            for w in closing { w.orderOut(nil); w.contentView = nil; w.close() }
            for (r, _) in self.renderers { r.drain() }
            self.windows.removeAll()
            self.renderers.removeAll()
            NSApp.presentationOptions = []
            self.dismissing = false
            self.recorder?.discard()
            self.recorder = nil
            let restart = self.restartAfterDismiss && self.pendingReport == nil
            self.restartAfterDismiss = false
            if restart {
                self.show(manual: true)
            } else if let dir = self.pendingReport {
                self.pendingReport = nil
                self.askForNote(dir)
            } else {
                self.onDismiss?()
            }
        }
        guard animated else { finish(); return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.25
            for w in closing { w.animator().alphaValue = 0 }
        }, completionHandler: finish)
    }
}
