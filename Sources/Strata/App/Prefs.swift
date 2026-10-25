import Foundation

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
    static var biome: Int { Biome.number(read("biome")) }
    static var pauseUntil: Date? {
        guard let d: Date = read("pauseUntil"), d > Date() else { return nil }
        return d
    }
}
