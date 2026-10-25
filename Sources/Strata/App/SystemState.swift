import CoreGraphics
import Foundation
import IOKit.ps
import IOKit.pwr_mgt

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
