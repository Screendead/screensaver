import AppKit
import StrataCore

package final class AppDelegate: NSObject, NSApplicationDelegate {
    let settings: Settings
    lazy var saver = SaverController(settings: settings)
    var pollTimer: Timer?
    private var menuController: MenuController?
    private var systemSaverRunning = false

    package init(settings: Settings) { self.settings = settings }

    package func applicationDidFinishLaunching(_ notification: Notification) {
        if let socket = settings.servePath {
            Server.run(socket: socket)
            exit(0)
        }
        if let path = settings.screenshotPath {
            _ = Snapshot.run(settings: settings, path: path)
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
        dnc.addObserver(forName: Prefs.changed, object: nil, queue: .main) { [weak self] _ in
            self?.saver.prefsChanged()
            self?.schedule(after: 0)
        }
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
