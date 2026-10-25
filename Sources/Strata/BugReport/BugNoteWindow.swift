import AppKit
import StrataCore

// Asks what the user saw after a report is captured: Return saves note.txt, Option-Return starts a new line, Escape or
// the close button skips it. `done` runs once the window closes, or after five minutes.
final class BugNoteWindow: NSObject, NSTextViewDelegate, NSWindowDelegate {
    private let window: NSWindow
    private let text = NSTextView()
    private let dir: URL
    private var done: (() -> Void)?
    private var timeout: Timer?

    init(dir: URL, done: @escaping () -> Void) {
        self.dir = dir
        self.done = done
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 440, height: 210), styleMask: [.titled, .closable], backing: .buffered, defer: false)
        super.init()
        window.title = "What did you see?"
        window.isReleasedWhenClosed = false
        window.delegate = self
        let content = window.contentView!
        let hint = NSTextField(labelWithString: "Saved to Bug Reports/\(dir.lastPathComponent). Return saves; Option-Return for a new line.")
        hint.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        hint.textColor = .secondaryLabelColor
        hint.frame = NSRect(x: 16, y: 182, width: 408, height: 16)
        let scroll = NSScrollView(frame: NSRect(x: 16, y: 52, width: 408, height: 122))
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        text.frame = NSRect(origin: .zero, size: scroll.contentSize)
        text.autoresizingMask = [.width]
        text.isRichText = false
        text.font = .systemFont(ofSize: NSFont.systemFontSize)
        text.delegate = self
        scroll.documentView = text
        let save = NSButton(title: "Save", target: self, action: #selector(saveNote))
        save.keyEquivalent = "\r"
        save.frame = NSRect(x: 336, y: 12, width: 88, height: 32)
        let skip = NSButton(title: "Skip", target: self, action: #selector(skipNote))
        skip.keyEquivalent = "\u{1b}"
        skip.frame = NSRect(x: 244, y: 12, width: 88, height: 32)
        for v in [hint, scroll, save, skip] { content.addSubview(v) }
    }

    func show() {
        window.center()
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(text)
        timeout = Timer.scheduledTimer(withTimeInterval: 300, repeats: false) { [weak self] _ in self?.window.close() }
    }

    func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
        switch selector {
        case #selector(NSResponder.insertNewline(_:)): saveNote()
        case #selector(NSResponder.cancelOperation(_:)): skipNote()
        default: return false
        }
        return true
    }

    @objc private func saveNote() {
        let note = text.string.trimmingCharacters(in: .whitespacesAndNewlines)
        if !note.isEmpty { try? (note + "\n").write(to: dir.appendingPathComponent("note.txt"), atomically: true, encoding: .utf8) }
        BugLog.note("bug report: note \(note.isEmpty ? "empty" : "saved")")
        window.close()
    }

    @objc private func skipNote() { window.close() }

    func windowWillClose(_ notification: Notification) {
        timeout?.invalidate()
        timeout = nil
        let d = done
        done = nil
        d?()
    }
}
