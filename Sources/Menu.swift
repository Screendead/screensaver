import AppKit

// MARK: - Shared preferences

// Read by the saver agent and written by the menu process; CFPreferences makes both see the same file.
enum Prefs {
    static let domain = "com.jacklusher.voxelreverie" as CFString
    static let changed = Notification.Name("com.jacklusher.voxelreverie.changed")
    static let startNow = Notification.Name("com.jacklusher.voxelreverie.startNow")

    private static func read<T>(_ key: String) -> T? {
        CFPreferencesAppSynchronize(domain)
        return CFPreferencesCopyAppValue(key as CFString, domain) as? T
    }

    static func set(_ key: String, _ value: Any?) {
        CFPreferencesSetAppValue(key as CFString, value as CFPropertyList?, domain)
        CFPreferencesAppSynchronize(domain)
        DistributedNotificationCenter.default().postNotificationName(changed, object: nil, userInfo: nil, deliverImmediately: true)
    }

    static var autoStart: Bool { read("autoStart") ?? true }
    static var idleDelay: Double { read("idleDelay") ?? 30 }
    static var dither: Bool { read("dither") ?? true }
    // A SkyClock phase the clock is held at; nil lets the day cycle run.
    static var timeOfDay: Double? { read("timeOfDay") }
    static var pauseUntil: Date? {
        guard let d: Date = read("pauseUntil"), d > Date() else { return nil }
        return d
    }
}

// MARK: - Menu bar item

final class MenuController: NSObject, NSMenuDelegate {
    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    private var expiry: Timer?
    private static let times: [(String, Double)] = [("Dawn", 0.965), ("Sunrise", 0.03), ("Morning", 0.12), ("Midday", 0.25),
                                                    ("Afternoon", 0.36), ("Golden Hour", 0.46), ("Sunset", 0.505), ("Dusk", 0.535),
                                                    ("Night", 0.62), ("Midnight", 0.75)]
    private static let delays: [(String, Double)] = [("15 seconds", 15), ("30 seconds", 30), ("1 minute", 60), ("2 minutes", 120),
                                                     ("5 minutes", 300), ("10 minutes", 600), ("15 minutes", 900)]

    func start() {
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        DistributedNotificationCenter.default().addObserver(forName: Prefs.changed, object: nil, queue: .main) { [weak self] _ in
            self?.refreshIcon()
        }
        refreshIcon()
    }

    private func refreshIcon() {
        let pause = Prefs.pauseUntil
        let active = Prefs.autoStart && pause == nil
        let image = NSImage(systemSymbolName: active ? "mountain.2.fill" : "mountain.2", accessibilityDescription: "Voxel Reverie")
        image?.isTemplate = true
        item.button?.image = image
        item.button?.appearsDisabled = !active
        expiry?.invalidate()
        expiry = nil
        if let pause {
            expiry = Timer.scheduledTimer(withTimeInterval: pause.timeIntervalSinceNow + 0.5, repeats: false) { [weak self] _ in self?.refreshIcon() }
        }
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        func add(_ title: String, _ action: Selector?, on: Bool = false, value: Any? = nil, to m: NSMenu = menu) {
            let i = m.addItem(withTitle: title, action: action, keyEquivalent: "")
            i.target = self
            i.state = on ? .on : .off
            i.representedObject = value
        }
        add("Start Now", #selector(startNow))
        menu.addItem(.separator())
        add("Start Automatically", #selector(toggleAuto), on: Prefs.autoStart)
        if let until = Prefs.pauseUntil {
            let f = DateFormatter()
            f.dateFormat = "HH:mm"
            add("Paused until \(f.string(from: until)) — Resume", #selector(resume))
        } else {
            add("Pause for 1 Hour", #selector(pause(_:)), value: Date().addingTimeInterval(3600))
            let tomorrow = Calendar.current.nextDate(after: Date(), matching: DateComponents(hour: 6, minute: 0), matchingPolicy: .nextTime)
            add("Pause Until Tomorrow", #selector(pause(_:)), value: tomorrow)
        }
        let after = NSMenuItem(title: "Start After", action: nil, keyEquivalent: "")
        let sub = NSMenu()
        let current = Prefs.idleDelay
        for (title, secs) in MenuController.delays { add(title, #selector(setDelay(_:)), on: secs == current, value: secs, to: sub) }
        after.submenu = sub
        menu.addItem(after)
        let tod = NSMenuItem(title: "Time of Day", action: nil, keyEquivalent: "")
        let todSub = NSMenu()
        let pinned = Prefs.timeOfDay
        add("Day Cycle", #selector(setTime(_:)), on: pinned == nil, to: todSub)
        todSub.addItem(.separator())
        for (title, phase) in MenuController.times { add(title, #selector(setTime(_:)), on: phase == pinned, value: phase, to: todSub) }
        tod.submenu = todSub
        menu.addItem(tod)
        add("Dithering", #selector(toggleDither), on: Prefs.dither)
        menu.addItem(.separator())
        add("Bug Reports…", #selector(openBugReports))
        menu.addItem(.separator())
        add("Quit Menu Item", #selector(quit))
    }

    @objc private func startNow() {
        DistributedNotificationCenter.default().postNotificationName(Prefs.startNow, object: nil, userInfo: nil, deliverImmediately: true)
    }
    @objc private func toggleAuto() { Prefs.set("autoStart", !Prefs.autoStart); refreshIcon() }
    @objc private func toggleDither() { Prefs.set("dither", !Prefs.dither) }
    @objc private func resume() { Prefs.set("pauseUntil", nil); refreshIcon() }
    @objc private func pause(_ sender: NSMenuItem) { Prefs.set("pauseUntil", sender.representedObject as? Date); refreshIcon() }
    @objc private func setDelay(_ sender: NSMenuItem) { Prefs.set("idleDelay", sender.representedObject as? Double) }
    @objc private func setTime(_ sender: NSMenuItem) { Prefs.set("timeOfDay", sender.representedObject as? Double) }
    @objc private func openBugReports() {
        try? FileManager.default.createDirectory(at: BugReport.folder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(BugReport.folder)
    }
    @objc private func quit() { NSApp.terminate(nil) }
}
