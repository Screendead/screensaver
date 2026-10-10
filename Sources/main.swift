import AppKit
import CoreGraphics
import IOKit.ps
import IOKit.pwr_mgt
import MetalKit

// MARK: - System state probes

enum SystemState {
    static func idleSeconds() -> Double {
        // kCGAnyInputEventType is ~0; Swift doesn't import it.
        CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: ~0)!)
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
            default: break
            }
            i += 1
        }
        return s
    }
}

// MARK: - Saver windows

final class SaverWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class SaverController {
    private var windows: [SaverWindow] = []
    private var renderers: [Renderer] = []
    private var watchTimer: Timer?
    private var shownAt = Date()
    private var previousApp: NSRunningApplication?
    private var dismissing = false
    let settings: Settings
    var onDismiss: (() -> Void)?

    var isShowing: Bool { !windows.isEmpty }

    init(settings: Settings) { self.settings = settings }

    func show() {
        guard !isShowing, let device = MTLCreateSystemDefaultDevice() else { return }
        dismissing = false
        previousApp = NSWorkspace.shared.frontmostApplication
        let battery = settings.forceBattery || SystemState.onBattery()
        // A different seed per session keeps the world non-repeating across activations.
        let sessionSeed = settings.seed != 0 ? settings.seed : UInt32.random(in: 1...UInt32.max)

        for (index, screen) in NSScreen.screens.enumerated() {
            let window = SaverWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.level = .screenSaver
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
            window.backgroundColor = .black
            window.isOpaque = true
            window.hasShadow = false
            window.alphaValue = 0
            window.setFrame(screen.frame, display: false)

            let view = MTKView(frame: CGRect(origin: .zero, size: screen.frame.size), device: device)
            guard let renderer = Renderer(view: view, screen: screen, seed: sessionSeed, screenIndex: index, battery: battery) else { continue }
            renderer.timeScale = settings.timeScale
            window.contentView = view
            renderers.append(renderer)
            windows.append(window)
            window.orderFrontRegardless()
        }

        NSApp.activate(ignoringOtherApps: true)
        windows.first?.makeKey()
        NSCursor.hide()
        NSApp.presentationOptions = [.hideDock, .hideMenuBar]

        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 2.0
            ctx.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            for w in windows { w.animator().alphaValue = 1 }
        }

        shownAt = Date()
        var lastIdle = SystemState.idleSeconds()
        watchTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let idle = SystemState.idleSeconds()
            // Grace period: activating our windows can itself register as an event.
            let settled = Date().timeIntervalSince(self.shownAt) > 0.6
            if settled && idle + 0.05 < lastIdle { self.dismiss() }
            lastIdle = idle
            if self.settings.previewDuration > 0 && Date().timeIntervalSince(self.shownAt) > self.settings.previewDuration {
                self.dismiss()
            }
        }
        RunLoop.main.add(watchTimer!, forMode: .common)
    }

    func dismiss(animated: Bool = true) {
        guard isShowing, !dismissing else { return }
        dismissing = true
        watchTimer?.invalidate()
        watchTimer = nil
        let closing = windows
        let finish = { [weak self] in
            guard let self else { return }
            for w in closing { w.orderOut(nil) }
            self.windows.removeAll()
            self.renderers.removeAll()
            NSApp.presentationOptions = []
            NSCursor.unhide()
            self.previousApp?.activate()
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
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { [weak self] _ in
            self?.saver.dismiss(animated: false)
        }

        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in self?.tick() }
    }

    private func tick() {
        guard !saver.isShowing else { return }
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
