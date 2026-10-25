// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "STRATA",
    platforms: [.macOS(.v14)],
    // The tests live in tests/, beside the golden renders: git tracks that spelling, which a case-sensitive checkout needs.
    targets: [
        // No Metal or AppKit: the sky clock, the autopilot, the birds, the map's CPU copy and tables, settings.
        .target(name: "StrataCore"),
        // Map generation and rendering on the GPU, snapshots, the saver's windows, menu and bug reports.
        .target(name: "Strata", dependencies: ["StrataCore"]),
        .executableTarget(name: "voxelreverie", dependencies: ["Strata", "StrataCore"]),
        .testTarget(name: "StrataCoreTests", dependencies: ["StrataCore"], path: "tests/StrataCoreTests"),
        // The shader's contracts with the CPU (built from Shaders/Saver.metal at test time) and the bug report's state.
        .testTarget(name: "StrataTests", dependencies: ["Strata", "StrataCore"], path: "tests/StrataTests", exclude: ["Shader/ShaderProbes.metal"]),
    ]
)
