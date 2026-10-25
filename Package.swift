// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "STRATA",
    platforms: [.macOS(.v14)],
    targets: [
        // No Metal or AppKit: the sky clock, the autopilot, the birds, the map's CPU copy and tables, settings.
        .target(name: "StrataCore"),
        // Map generation and rendering on the GPU, snapshots, the saver's windows, menu and bug reports.
        .target(name: "Strata", dependencies: ["StrataCore"]),
        .executableTarget(name: "voxelreverie", dependencies: ["Strata", "StrataCore"]),
        .testTarget(name: "StrataCoreTests", dependencies: ["StrataCore"]),
    ]
)
