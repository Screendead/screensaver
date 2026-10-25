import Foundation

let debugLog = ProcessInfo.processInfo.environment["VR_DEBUG"] != nil
func dlog(_ s: @autoclosure () -> String) { if debugLog { print(s()); fflush(stdout) } }
