#!/bin/zsh
# Builds the app, copies it to ~/Applications and registers a login agent that watches for idle.
set -euo pipefail
cd "$(dirname "$0")"
make
LABEL=com.jacklusher.voxelreverie
MENU_LABEL=$LABEL.menu
DEST="$HOME/Applications/VoxelReverie.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
MENU_PLIST="$HOME/Library/LaunchAgents/$MENU_LABEL.plist"
mkdir -p "$HOME/Applications" "$HOME/Library/LaunchAgents"
# bootout returns before launchd has torn the job down; wait so bootstrap doesn't race it.
for l in $LABEL $MENU_LABEL; do
  launchctl bootout "gui/$(id -u)/$l" 2>/dev/null || true
  for _ in {1..50}; do launchctl print "gui/$(id -u)/$l" >/dev/null 2>&1 || break; sleep 0.1; done
done
[[ -e "$DEST" ]] && trash "$DEST"
cp -R build/VoxelReverie.app "$DEST"
agent() {
  local plist=$1 label=$2 keep=$3; shift 3
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>$label</string>
	<key>ProgramArguments</key>
	<array>
$(for a; do print -r -- "		<string>$a</string>"; done)
	</array>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key>$keep
	<key>LimitLoadToSessionType</key><string>Aqua</string>
	<key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
PLIST
}
agent "$PLIST" $LABEL '<true/>' "$DEST/Contents/MacOS/voxelreverie"
# The menu bar item runs apart from the saver so its icon survives the saver agent's restarts; a clean Quit stays quit.
agent "$MENU_PLIST" $MENU_LABEL '<dict><key>SuccessfulExit</key><false/></dict>' "$DEST/Contents/MacOS/voxelreverie" --menu
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl bootstrap "gui/$(id -u)" "$MENU_PLIST"
echo "Installed. Voxel Reverie starts after the idle delay (30s by default; change it from the menu bar item). Preview now: \"$DEST/Contents/MacOS/voxelreverie\" --preview"
