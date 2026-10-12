import AppKit
import CoreGraphics
import IOKit.ps
import IOKit.pwr_mgt
import MetalKit

// MARK: - System state probes

let debugLog = ProcessInfo.processInfo.environment["VR_DEBUG"] != nil
func dlog(_ s: @autoclosure () -> String) { if debugLog { print(s()); fflush(stdout) } }

enum SystemState {
    static func idleSeconds() -> Double {
        // kCGAnyInputEventType is ~0; Swift doesn't import it.
        CGEventSource.secondsSinceLastEventType(.hidSystemState, eventType: CGEventType(rawValue: ~0)!)
    }

    // Video players, calls and presentations hold one of these to keep the display awake.
    static func somethingWantsDisplayAwake() -> Bool {
        var status: Unmanaged<CFDictionary>?
        guard IOPMCopyAssertionsStatus(&status) == kIOReturnSuccess,
              let dict = status?.takeRetainedValue() as? [String: Int] else { return false }
        return (dict["PreventUserIdleDisplaySleep"] ?? 0) > 0 || (dict["NoDisplaySleepAssertion"] ?? 0) > 0
    }

    static func screenIsLockedOrInactive() -> Bool {
        guard let session = CGSessionCopyCurrentDictionary() as? [String: Any] else { return true }
        if (session["CGSSessionScreenIsLocked"] as? Bool) == true { return true }
        if (session[kCGSessionOnConsoleKey as String] as? Bool) == false { return true }
        return false
    }

    static func displaysAsleep() -> Bool {
        CGDisplayIsAsleep(CGMainDisplayID()) != 0
    }

    // NSCursor.hide only takes effect for the active app, and macOS often refuses to activate a background agent.
    // This private connection property lets it hide the cursor anyway.
    static func allowCursorHidingInBackground() {
        typealias MainConn = @convention(c) () -> Int32
        typealias SetProp = @convention(c) (Int32, Int32, CFString, CFTypeRef) -> Int32
        guard let h = dlopen(nil, RTLD_NOW), let m = dlsym(h, "_CGSDefaultConnection"), let p = dlsym(h, "CGSSetConnectionProperty") else { return }
        let conn = unsafeBitCast(m, to: MainConn.self)()
        _ = unsafeBitCast(p, to: SetProp.self)(conn, conn, "SetsCursorInBackground" as CFString, kCFBooleanTrue)
    }

    static func onBattery() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return false }
        return type == kIOPMBatteryPowerKey || ProcessInfo.processInfo.isLowPowerModeEnabled
    }
}

// MARK: - Settings

struct Settings {
    var idleThreshold: Double?
    var menu = false
    var noDither = false
    var preview = false
    var previewDuration: Double = 0
    var screenshotPath: String?
    var screenshotTimes: [Double] = [8]
    var screenshotSize = CGSize(width: 1728, height: 1117)
    var forceBattery = false
    var seed: UInt32 = 0
    var debugMode: Float = 0
    var renderScale: Double?
    var phase: Double?
    var timeScale: Double = 1
    var pitch: Double?
    var headroom: Float?

    static func parse(_ args: [String]) -> Settings {
        var s = Settings()
        var i = 1
        func next() -> String { i += 1; return i < args.count ? args[i] : "" }
        while i < args.count {
            switch args[i] {
            case "--idle": s.idleThreshold = Double(next()) ?? 30
            case "--menu": s.menu = true
            case "--nodither": s.noDither = true
            case "--preview": s.preview = true
            case "--duration": s.previewDuration = Double(next()) ?? 0
            case "--screenshot": s.screenshotPath = next()
            case "--times": s.screenshotTimes = next().split(separator: ",").compactMap { Double($0) }
            case "--size":
                let p = next().split(separator: "x").compactMap { Double($0) }
                if p.count == 2 { s.screenshotSize = CGSize(width: p[0], height: p[1]) }
            case "--battery": s.forceBattery = true
            case "--seed": s.seed = UInt32(next()) ?? 0
            case "--debug": s.debugMode = Float(next()) ?? 0
            case "--scale": s.renderScale = Double(next())
            case "--phase": s.phase = Double(next())
            case "--speed": s.timeScale = Double(next()) ?? 1
            case "--pitch": s.pitch = Double(next())
            case "--headroom": s.headroom = Float(next())
            default: break
            }
            i += 1
        }
        return s
    }
}

// MARK: - Saver windows

final class SaverWindow: NSWindow {
    var onInput: (() -> Void)?
    var onBugKey: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    // Swallow input here: unhandled keys would otherwise reach NSApp and beep.
    override func keyDown(with event: NSEvent) { event.keyCode == BugReport.key ? onBugKey?() : onInput?() }
    override func keyUp(with event: NSEvent) {}
    override func flagsChanged(with event: NSEvent) { onInput?() }
    override func mouseDown(with event: NSEvent) { onInput?() }
    override func rightMouseDown(with event: NSEvent) { onInput?() }
    override func otherMouseDown(with event: NSEvent) { onInput?() }
    override func scrollWheel(with event: NSEvent) { onInput?() }
}

final class SaverController {
    private var windows: [SaverWindow] = []
    private var renderers: [(Renderer, MTKView)] = []
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
        let sessionSeed = settings.seed != 0 ? settings.seed : UInt32.random(in: 1...UInt32.max)
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

            // A screen whose renderer can't start still gets covered, in black.
            if let gpu {
                let view = MTKView(frame: CGRect(origin: .zero, size: screen.frame.size), device: gpu.device)
                if let renderer = Renderer(view: view, gpu: gpu, screen: screen, seed: sessionSeed, screenIndex: index, battery: battery, dither: !settings.noDither && Prefs.dither) {
                    renderer.timeScale = settings.timeScale
                    window.contentView = view
                    renderers.append((renderer, view))
                    let ds = view.drawableSize
                    if index == 0, let rec = BugRecorder(gpu: gpu, width: Int(ds.width), height: Int(ds.height), info: [
                        "seed": sessionSeed, "screenIndex": index, "screenCount": screens.count, "screenName": screen.localizedName,
                        "screenPoints": [Int(screen.frame.width), Int(screen.frame.height)], "backingScale": screen.backingScaleFactor,
                        "screenPixels": [Int(renderer.screenPixels.width), Int(renderer.screenPixels.height)],
                        "drawable": [Int(ds.width), Int(ds.height)], "edr": view.colorPixelFormat == .rgba16Float,
                        "dither": !settings.noDither && Prefs.dither, "manualStart": manual,
                    ]) {
                        // The recorder reads the drawable.
                        view.framebufferOnly = false
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
        let grace = inputGrace
        var lastPowerCheck = shownAt
        watchTimer?.invalidate()
        watchTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let age = SaverController.now - self.shownAt
            // Any physical input after the moment we appeared, including during the fade-in.
            let idle = SystemState.idleSeconds()
            if idle < age - grace - 0.15 {
                // The key event itself may not arrive (macOS can refuse to activate the agent), so B is read from the HID state.
                if CGEventSource.keyState(.hidSystemState, key: BugReport.key) { self.reportBug(); return }
                BugLog.note(String(format: "input detected: idle %.2f s, shown %.1f s", idle, age))
                self.dismiss()
                return
            }
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
            if let dir = self.pendingReport {
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

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    let settings: Settings
    lazy var saver = SaverController(settings: settings)
    var pollTimer: Timer?
    private var menuController: MenuController?
    private var systemSaverRunning = false

    init(settings: Settings) { self.settings = settings }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if let path = settings.screenshotPath {
            Snapshot.run(settings: settings, path: path)
            exit(0)
        }
        if settings.menu {
            let menu = MenuController()
            menu.start()
            menuController = menu
            return
        }
        if settings.preview {
            saver.onDismiss = { NSApp.terminate(nil) }
            saver.show()
            return
        }

        let ws = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.screensDidSleepNotification, NSWorkspace.sessionDidResignActiveNotification, NSWorkspace.willSleepNotification] {
            ws.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in self?.saver.dismiss(animated: false) }
        }
        // Asleep or locked, the poll backs off: these bring the next check forward.
        for name in [NSWorkspace.screensDidWakeNotification, NSWorkspace.sessionDidBecomeActiveNotification, NSWorkspace.didWakeNotification] {
            ws.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in self?.schedule(after: 0) }
        }
        let dnc = DistributedNotificationCenter.default()
        // If the system screensaver starts anyway, it owns the screen until it stops.
        dnc.addObserver(forName: NSNotification.Name("com.apple.screensaver.didstart"), object: nil, queue: .main) { [weak self] _ in
            self?.systemSaverRunning = true
            self?.saver.dismiss(animated: false)
        }
        dnc.addObserver(forName: NSNotification.Name("com.apple.screensaver.didstop"), object: nil, queue: .main) { [weak self] _ in
            self?.systemSaverRunning = false
            self?.schedule(after: 0)
        }
        dnc.addObserver(forName: Prefs.changed, object: nil, queue: .main) { [weak self] _ in self?.schedule(after: 0) }
        dnc.addObserver(forName: Prefs.startNow, object: nil, queue: .main) { [weak self] _ in
            guard let self, !self.saver.isShowing, !SystemState.screenIsLockedOrInactive(), !SystemState.displaysAsleep() else { return }
            self.saver.show(manual: true)
        }
        dnc.addObserver(forName: NSNotification.Name("com.apple.screenIsLocked"), object: nil, queue: .main) { [weak self] _ in
            self?.saver.dismiss(animated: false)
        }
        dnc.addObserver(forName: NSNotification.Name("com.apple.screenIsUnlocked"), object: nil, queue: .main) { [weak self] _ in
            self?.schedule(after: 0)
        }
        // Hiding the menu bar also posts this; only a real display change (frames differ) needs a rebuild.
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self, self.saver.isShowing, self.saver.screenFrames != NSScreen.screens.map(\.frame) else { return }
            self.saver.dismiss(animated: false)
        }

        // A session leaves ~20 MB of freed malloc pages dirty that nothing returns to the system, so the agent exits and
        // launchd's KeepAlive starts it afresh. Run by hand, it carries on instead.
        if let label = Bundle.main.bundleIdentifier, ProcessInfo.processInfo.environment["XPC_SERVICE_NAME"] == label {
            saver.onDismiss = { DispatchQueue.main.async { exit(0) } }
        } else {
            saver.onDismiss = { [weak self] in self?.schedule(after: 0) }
        }
        systemSaverRunning = !NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.ScreenSaver.Engine").isEmpty
        schedule(after: 1)
    }

    private func schedule(after delay: Double) {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in self?.tick() }
        pollTimer?.tolerance = 0.2
    }

    // Idle time grows no faster than the clock, so nothing can start before it could reach the threshold: the poll
    // sleeps until then rather than waking every second.
    private func tick() {
        dlog(String(format: "tick idle=%.1f wantsAwake=%d locked=%d asleep=%d showing=%d sys=%d", SystemState.idleSeconds(),
                     SystemState.somethingWantsDisplayAwake() ? 1 : 0, SystemState.screenIsLockedOrInactive() ? 1 : 0,
                     SystemState.displaysAsleep() ? 1 : 0, saver.isShowing ? 1 : 0, systemSaverRunning ? 1 : 0))
        if saver.isShowing { return }
        if systemSaverRunning { schedule(after: 10); return }
        if !Prefs.autoStart { schedule(after: 30); return }
        if let until = Prefs.pauseUntil { schedule(after: min(30, until.timeIntervalSinceNow + 0.5)); return }
        let threshold = settings.idleThreshold ?? Prefs.idleDelay
        let idle = SystemState.idleSeconds()
        if idle < threshold { schedule(after: max(0.25, threshold - idle)); return }
        if SystemState.screenIsLockedOrInactive() || SystemState.displaysAsleep() { schedule(after: 10); return }
        if SystemState.somethingWantsDisplayAwake() { schedule(after: 1); return }
        saver.show()
        if !saver.isShowing { schedule(after: 1) }
    }
}

let settings = Settings.parse(CommandLine.arguments)
let app = NSApplication.shared
let delegate = AppDelegate(settings: settings)
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
