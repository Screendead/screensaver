import Foundation

// The last notable events of this process, for bug reports.
package enum BugLog {
    static let capacity = 200
    private static let lock = NSLock()
    private static var lines: [String] = []
    private static var head = 0
    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    package static func note(_ s: String) {
        dlog(s)
        lock.lock()
        defer { lock.unlock() }
        let line = stamp.string(from: Date()) + " " + s
        if lines.count < capacity { lines.append(line) } else { lines[head] = line; head = (head + 1) % capacity }
    }

    package static var recent: [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(lines[head...] + lines[..<head])
    }
}
