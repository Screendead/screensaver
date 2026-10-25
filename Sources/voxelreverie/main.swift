import AppKit
import Strata
import StrataCore

let settings = Settings.parse(CommandLine.arguments)
let app = NSApplication.shared
let delegate = AppDelegate(settings: settings)
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
