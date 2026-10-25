// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "STRATA",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "voxelreverie", path: "Sources"),
    ]
)
