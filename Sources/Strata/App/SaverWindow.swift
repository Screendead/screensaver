import AppKit

final class SaverWindow: NSWindow {
    var onInput: (() -> Void)?
    var onBugKey: (() -> Void)?
    var onShotKey: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    // Swallow input here: unhandled keys would otherwise reach NSApp and beep.
    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case BugReport.key: onBugKey?()
        case SaverController.shotKey: if !event.isARepeat { onShotKey?() }
        default: onInput?()
        }
    }
    override func keyUp(with event: NSEvent) {}
    override func flagsChanged(with event: NSEvent) { onInput?() }
    override func mouseDown(with event: NSEvent) { onInput?() }
    override func rightMouseDown(with event: NSEvent) { onInput?() }
    override func otherMouseDown(with event: NSEvent) { onInput?() }
    override func scrollWheel(with event: NSEvent) { onInput?() }
}
