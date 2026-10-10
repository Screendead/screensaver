#!/bin/zsh
# Builds the app, copies it to ~/Applications and registers a login agent that watches for idle.
set -euo pipefail
cd "$(dirname "$0")"
make
LABEL=com.jacklusher.voxelreverie
DEST="$HOME/Applications/VoxelReverie.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Applications" "$HOME/Library/LaunchAgents"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
[[ -e "$DEST" ]] && trash "$DEST"
cp -R build/VoxelReverie.app "$DEST"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$DEST/Contents/MacOS/voxelreverie</string>
		<string>--idle</string><string>30</string>
	</array>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>LimitLoadToSessionType</key><string>Aqua</string>
	<key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed. Voxel Reverie starts after 30s idle. Preview now: \"$DEST/Contents/MacOS/voxelreverie\" --preview"
