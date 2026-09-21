#!/bin/bash
# production.sh — the production profile of launch-sgl-dsv41.sh: the published image and the production port (8210).
# Everything else, including SERVED_ALIASES (extra model ids to list on /v1/models), comes from launch/fleet.env or the
# environment. Takes the launcher's flags (--dry-run, --stop). The fleet watchdog (launch/watchdog.sh) recovers this
# profile; use launch/relaunch.sh to restart it by hand so the watchdog is disarmed and re-armed around the boot.
# MIT License, Copyright (c) 2026 Aiden Le.
export PORT="${PORT:-8210}"
export PRODUCTION_IMAGE="aidendle94/sparkrun-sglang-dsv41-gb10:production-1.1"   # used unless IMAGE is set in the environment or fleet.env
exec bash "$(dirname "$0")/launch-sgl-dsv41.sh" "$@"
