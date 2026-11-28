#!/bin/bash
# stop.sh — stop the fleet: the watchdog first (it would start the fleet again), then the container on every node.
# MIT License, Copyright (c) 2026 Aiden Le.
. "$(cd "$(dirname "$0")" && pwd)/settings.sh"
WLOG=$HOME/inkling2-watchdog.log
wp=$(grep -o "watchdog started (pid [0-9]*" "$WLOG" 2>/dev/null | tail -1 | grep -o "[0-9]*$")
if [ -n "$wp" ] && ps -p "$wp" -o args= 2>/dev/null | grep -q "watchdog-inkling2.sh"; then kill "$wp" && echo "watchdog $wp stopped"; fi
rm -f "$HOME/.inkling2-watchdog.lock"
exec bash "$BASE_DIR/launch/launch-inkling2.sh" --stop
