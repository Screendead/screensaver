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

    static func onBattery() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return false }
        return type == kIOPMBatteryPowerKey || ProcessInfo.processInfo.isLowPowerModeEnabled
    }
}

// MARK: - Settings

struct Settings {
    var idleThreshold: Double = 30
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

    static func parse(_ args: [String]) -> Settings {
        var s = Settings()
        var i = 1
        func next() -> String { i += 1; return i < args.count ? args[i] : "" }
        while i < args.count {
            switch args[i] {
            case "--idle": s.idleThreshold = Double(next()) ?? 30
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
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    // Swallow input here: unhandled keys would otherwise reach NSApp and beep.
    override func keyDown(with event: NSEvent) { onInput?() }
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
    let settings: Settings
    var onDismiss: (() -> Void)?

    var isShowing: Bool { !windows.isEmpty }
    private(set) var screenFrames: [CGRect] = []

    init(settings: Settings) { self.settings = settings }

    private static var now: TimeInterval { ProcessInfo.processInfo.systemUptime }

    func show() {
        let screens = NSScreen.screens
        guard !isShowing, !screens.isEmpty else { return }
        dismissing = false
        let front = NSWorkspace.shared.frontmostApplication
        previousApp = front?.processIdentifier == ProcessInfo.processInfo.processIdentifier ? nil : front
        battery = settings.forceBattery || SystemState.onBattery()
        // A different seed per session keeps the world non-repeating across activations.
        let sessionSeed = settings.seed != 0 ? settings.seed : UInt32.random(in: 1...UInt32.max)
        let device = MTLCreateSystemDefaultDevice()

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
            window.onInput = { [weak self] in self?.dismiss() }

            // A screen whose renderer can't start still gets covered, in black.
            if let device {
                let view = MTKView(frame: CGRect(origin: .zero, size: screen.frame.size), device: device)
                if let renderer = Renderer(view: view, screen: screen, seed: sessionSeed, screenIndex: index, battery: battery) {
                    renderer.timeScale = settings.timeScale
                    window.contentView = view
                    renderers.append((renderer, view))
                }
            }
            windows.append(window)
            window.orderFrontRegardless()
        }

        NSApp.activate(ignoringOtherApps: true)
        windows.first?.makeKey()
        NSCursor.hide()
        NSApp.presentationOptions = [.hideDock, .hideMenuBar]

        dlog("show: active=\(NSApp.isActive) screens=\(screens.count) renderers=\(renderers.count)")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { dlog("show+1s: active=\(NSApp.isActive) keyWindow=\(NSApp.keyWindow != nil)") }
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 2.0
            ctx.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            for w in windows { w.animator().alphaValue = 1 }
        }

        shownAt = SaverController.now
        var lastPowerCheck = shownAt
        watchTimer?.invalidate()
        watchTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let age = SaverController.now - self.shownAt
            // Any physical input after the moment we appeared, including during the fade-in.
            let idle = SystemState.idleSeconds()
            if idle < age - 0.15 { dlog("input detected: idle \(idle), shown \(age)"); self.dismiss(); return }
            if self.settings.previewDuration > 0 && age > self.settings.previewDuration { self.dismiss(); return }
            if SaverController.now - lastPowerCheck > 5 {
                lastPowerCheck = SaverController.now
                let b = self.settings.forceBattery || SystemState.onBattery()
                if b != self.battery {
                    self.battery = b
                    for (r, v) in self.renderers { r.setBattery(b, view: v) }
                }
            }
        }
        watchTimer?.tolerance = 0.005
        RunLoop.main.add(watchTimer!, forMode: .common)
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
        if NSApp.isActive {
            if let prev = previousApp, !prev.isTerminated { prev.activate() } else { NSApp.hide(nil) }
        }
        NSCursor.unhide()
        let finish = { [weak self] in
            guard let self else { return }
            for w in closing { w.orderOut(nil); w.contentView = nil }
            self.windows.removeAll()
            self.renderers.removeAll()
            NSApp.presentationOptions = []
            self.dismissing = false
            self.onDismiss?()
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
    private var systemSaverRunning = false

    init(settings: Settings) { self.settings = settings }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if let path = settings.screenshotPath {
            Snapshot.run(settings: settings, path: path)
            exit(0)
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
        let dnc = DistributedNotificationCenter.default()
        // If the system screensaver starts anyway, it owns the screen until it stops.
        dnc.addObserver(forName: NSNotification.Name("com.apple.screensaver.didstart"), object: nil, queue: .main) { [weak self] _ in
            self?.systemSaverRunning = true
            self?.saver.dismiss(animated: false)
        }
        dnc.addObserver(forName: NSNotification.Name("com.apple.screensaver.didstop"), object: nil, queue: .main) { [weak self] _ in
            self?.systemSaverRunning = false
        }
        dnc.addObserver(forName: NSNotification.Name("com.apple.screenIsLocked"), object: nil, queue: .main) { [weak self] _ in
            self?.saver.dismiss(animated: false)
        }
        // Hiding the menu bar also posts this; only a real display change (frames differ) needs a rebuild.
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self, self.saver.isShowing, self.saver.screenFrames != NSScreen.screens.map(\.frame) else { return }
            self.saver.dismiss(animated: false)
        }

        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in self?.tick() }
        pollTimer?.tolerance = 0.2
    }

    private func tick() {
        dlog(String(format: "tick idle=%.1f wantsAwake=%d locked=%d asleep=%d showing=%d sys=%d", SystemState.idleSeconds(),
                     SystemState.somethingWantsDisplayAwake() ? 1 : 0, SystemState.screenIsLockedOrInactive() ? 1 : 0,
                     SystemState.displaysAsleep() ? 1 : 0, saver.isShowing ? 1 : 0, systemSaverRunning ? 1 : 0))
        guard !saver.isShowing, !systemSaverRunning else { return }
        guard SystemState.idleSeconds() >= settings.idleThreshold else { return }
        if SystemState.somethingWantsDisplayAwake() || SystemState.screenIsLockedOrInactive() || SystemState.displaysAsleep() { return }
        saver.show()
    }
}

let settings = Settings.parse(CommandLine.arguments)
let app = NSApplication.shared
let delegate = AppDelegate(settings: settings)
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
