#!/bin/zsh
LABEL=com.jacklusher.voxelreverie
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
for f in "$HOME/Library/LaunchAgents/$LABEL.plist" "$HOME/Applications/VoxelReverie.app"; do
  [[ -e "$f" ]] && trash "$f"
done
echo "Uninstalled."
