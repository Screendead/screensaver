import Foundation
import Metal
import QuartzCore
import StrataCore

// --serve <socket>: renders --screenshot requests (tools/vrr) in one long-lived process, so the device, library and
// pipelines are made once. Each request still gets a fresh Renderer, exactly as a new process would.
//
// Protocol: one connection per request. The client sends one JSON line {"args": [...], "env": {...}, "cwd": "..."} with its
// stdout and stderr passed as SCM_RIGHTS descriptors; the request prints to them. The reply is one JSON line {"ok": Bool}.
enum Server {
    static let idleLimit = 600.0

    static func run(socket path: String) {
        signal(SIGPIPE, SIG_IGN)
        let started = CACurrentMediaTime()
        guard let device = MTLCreateSystemDefaultDevice(), let gpu = GPU(device: device) else {
            log("no Metal device or library")
            return
        }
        gpu.prewarm(output: .rgba8Unorm)
        let builds = buildStamps()
        guard let listener = listen(on: path) else { return }
        var inode = stat()
        stat(path, &inode)
        log(String(format: "serving %@ (ready in %.0f ms)", path, (CACurrentMediaTime() - started) * 1000))
        var last = CACurrentMediaTime()
        while true {
            var p = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
            let n = poll(&p, 1, 30_000)
            if n < 0 && errno != EINTR { log("poll: \(String(cString: strerror(errno)))"); break }
            if n <= 0 {
                if CACurrentMediaTime() - last > idleLimit { log("idle, exiting"); break }
                if buildStamps() != builds { log("binary or metallib changed, exiting"); break }
                continue
            }
            let c = accept(listener, nil, nil)
            guard c >= 0 else { continue }
            handle(c, gpu)
            close(c)
            last = CACurrentMediaTime()
        }
        var now = stat()
        if stat(path, &now) == 0 && now.st_ino == inode.st_ino { unlink(path) }
    }

    private static func handle(_ c: Int32, _ gpu: GPU) {
        guard let (data, fds) = receive(c) else { return }
        defer { fds.forEach { close($0) } }
        guard fds.count == 2, let req = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let args = req["args"] as? [String], let env = req["env"] as? [String: String], let cwd = req["cwd"] as? String else {
            reply(c, ok: false)
            return
        }
        drainMain()
        let t0 = CACurrentMediaTime()
        fflush(stdout)
        fflush(stderr)
        let saved = (dup(1), dup(2))
        dup2(fds[0], 1)
        dup2(fds[1], 2)
        FileManager.default.changeCurrentDirectoryPath(cwd)
        Knobs.current = Knobs(env)
        let settings = Settings.parse(["voxelreverie"] + args)
        var ok = false
        if let path = settings.screenshotPath {
            ok = autoreleasepool { Snapshot.run(settings: settings, path: path, gpu: gpu) }
        } else {
            // vrr looks for this message in the binary to tell whether a build has --serve.
            FileHandle.standardError.write("vrr: --screenshot is required\n".data(using: .utf8)!)
        }
        fflush(stdout)
        fflush(stderr)
        dup2(saved.0, 1)
        dup2(saved.1, 2)
        close(saved.0)
        close(saved.1)
        drainMain()
        log(String(format: "%@ %.0f ms, %.0f MB allocated after: %@", ok ? "ok" : "FAILED", (CACurrentMediaTime() - t0) * 1000,
                   Double(gpu.device.currentAllocatedSize) / 1048576, args.joined(separator: " ")))
        reply(c, ok: ok)
    }

    // Completion handlers queue main-thread blocks that a --screenshot process never runs (it exits first). Run here,
    // between requests, they find their renderer gone.
    private static func drainMain() {
        for _ in 0..<4 { CFRunLoopRunInMode(.defaultMode, 0, false) }
    }

    private static func reply(_ c: Int32, ok: Bool) {
        let line = "{\"ok\": \(ok)}\n"
        _ = line.withCString { write(c, $0, strlen($0)) }
    }

    // Reads up to the first newline, collecting any descriptors passed alongside.
    private static func receive(_ c: Int32) -> (Data, [Int32])? {
        var data = Data(), fds: [Int32] = []
        var buf = [UInt8](repeating: 0, count: 65536), control = [UInt8](repeating: 0, count: 256)
        while !data.contains(10) {
            var n = 0
            buf.withUnsafeMutableBytes { b in
                control.withUnsafeMutableBytes { cb in
                    var iov = iovec(iov_base: b.baseAddress, iov_len: b.count)
                    withUnsafeMutablePointer(to: &iov) { iovp in
                        var msg = msghdr(msg_name: nil, msg_namelen: 0, msg_iov: iovp, msg_iovlen: 1,
                                         msg_control: cb.baseAddress, msg_controllen: socklen_t(cb.count), msg_flags: 0)
                        n = recvmsg(c, &msg, 0)
                        // cmsghdr is {len, level, type} as 32-bit fields, its data 4-byte aligned after them.
                        var at = 0
                        while n > 0 && at + 12 <= Int(msg.msg_controllen) {
                            let len = Int(cb.load(fromByteOffset: at, as: UInt32.self))
                            if cb.load(fromByteOffset: at + 4, as: Int32.self) == SOL_SOCKET && cb.load(fromByteOffset: at + 8, as: Int32.self) == SCM_RIGHTS {
                                for k in 0..<max(0, (len - 12) / 4) { fds.append(cb.load(fromByteOffset: at + 12 + 4 * k, as: Int32.self)) }
                            }
                            guard len >= 12 else { break }
                            at += (len + 3) & ~3
                        }
                    }
                }
            }
            if n <= 0 { fds.forEach { close($0) }; return nil }
            data.append(contentsOf: buf[0..<n])
        }
        return (data, fds)
    }

    private static func listen(on path: String) -> Int32? {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard fd >= 0, bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else { log("bad socket path \(path)"); return nil }
        withUnsafeMutableBytes(of: &addr.sun_path) { $0.copyBytes(from: bytes) }
        unlink(path)
        let bound = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) } }
        guard bound == 0, Darwin.listen(fd, 64) == 0 else { log("bind \(path): \(String(cString: strerror(errno)))"); return nil }
        chmod(path, 0o600)
        return fd
    }

    // The executable's and the metallib's modification times: a rebuild retires the server.
    private static func buildStamps() -> [Date?] {
        let exe = Bundle.main.executablePath ?? CommandLine.arguments[0]
        let lib = Bundle.main.url(forResource: "Saver", withExtension: "metallib")?.path
            ?? URL(fileURLWithPath: exe).deletingLastPathComponent().appendingPathComponent("Saver.metallib").path
        return [exe, lib].map { (try? FileManager.default.attributesOfItem(atPath: $0))?[.modificationDate] as? Date }
    }

    private static func log(_ s: String) {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        FileHandle.standardError.write("\(f.string(from: Date())) \(s)\n".data(using: .utf8)!)
    }
}
