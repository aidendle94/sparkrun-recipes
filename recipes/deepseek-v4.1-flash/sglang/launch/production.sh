#!/bin/bash
# production.sh — the production profile of launch-sgl-dsv41.sh: the published image and the production port (8210).
# Everything else, including SERVED_ALIASES (extra model ids to list on /v1/models), comes from launch/fleet.env or the
# environment. Takes the launcher's flags (--dry-run, --stop). The fleet watchdog (launch/watchdog.sh) recovers this
# profile; use launch/relaunch.sh to restart it by hand so the watchdog is disarmed and re-armed around the boot.
# MIT License, Copyright (c) 2026 Aiden Le.
export PORT="${PORT:-8210}"
export PRODUCTION_IMAGE="aidendle94/sparkrun-sglang-dsv41-gb10:production-1.7.1"   # used unless IMAGE is set in the environment or fleet.env
# production switches (overlay/late_tail.py, overlay/sm120_prefill_pages.py): prefill skips the late layers on chunks that
# end before the prompt's last 128 tokens (1.4 ran them on one token, 1.5 skips them); decode-sized attention runs on the
# rank's real heads, not padded to 64
export LATE_TAIL_SKIP="${LATE_TAIL_SKIP:-1}" LATE_TAIL_SKIP_ALL="${LATE_TAIL_SKIP_ALL:-1}" SM120_REAL_HEADS="${SM120_REAL_HEADS:-1}"
# Prompt-lookup wide mode (overlay/lookup_wide.py): WIDE=1 turns it on (block 15, compact ragged verify, the step-cost
# fixes). Off by default since 2026-09-29 (a lookup continuation ran into image placeholder ids and a device-side assert
# took the fleet down; fixed in 1.7.1, re-enable after its verification window). Any knob stays overridable.
if [ "${WIDE:-0}" = 1 ]; then
  export SPEC_K="${SPEC_K:-15}" LOOKUP_DRAFT="${LOOKUP_DRAFT:-1}" LOOKUP_MODE="${LOOKUP_MODE:-wide}" \
         RAGGED_VERIFY_MODE="${RAGGED_VERIFY_MODE:-compact}" LOOKUP_SYNC_GATE="${LOOKUP_SYNC_GATE:-1}" \
         LOOKUP_NARROW_SLOTS="${LOOKUP_NARROW_SLOTS:-1}" LOOKUP_FUSED_C2="${LOOKUP_FUSED_C2:-1}"
fi
# CUDA_LOG=1 puts CUDA driver error details in the container log (off by default; turn it on to chase a driver error)
exec bash "$(dirname "$0")/launch-sgl-dsv41.sh" "$@"
