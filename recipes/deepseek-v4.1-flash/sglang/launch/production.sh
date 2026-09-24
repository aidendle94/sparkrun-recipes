#!/bin/bash
# production.sh — the production profile of launch-sgl-dsv41.sh: the published image and the production port (8210).
# Everything else, including SERVED_ALIASES (extra model ids to list on /v1/models), comes from launch/fleet.env or the
# environment. Takes the launcher's flags (--dry-run, --stop). The fleet watchdog (launch/watchdog.sh) recovers this
# profile; use launch/relaunch.sh to restart it by hand so the watchdog is disarmed and re-armed around the boot.
# MIT License, Copyright (c) 2026 Aiden Le.
export PORT="${PORT:-8210}"
export PRODUCTION_IMAGE="aidendle94/sparkrun-sglang-dsv41-gb10:production-1.4"   # used unless IMAGE is set in the environment or fleet.env
# production-1.4 switches (overlay/late_tail.py, overlay/sm120_prefill_pages.py): prefill skips the late layers on chunks
# that do not reach the prompt's last 128 tokens; decode-sized attention runs on the rank's real heads, not padded to 64
export LATE_TAIL_SKIP="${LATE_TAIL_SKIP:-1}" SM120_REAL_HEADS="${SM120_REAL_HEADS:-1}"
exec bash "$(dirname "$0")/launch-sgl-dsv41.sh" "$@"
