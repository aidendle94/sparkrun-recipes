#!/bin/bash
# relaunch.sh — start or restart the fleet with this recipe's settings (launch/settings.sh): the launcher of
# ../sglang with gated transient system-prompt reinforcement (SYSTEM_REINFORCE=user), the NVFP4 KV cache and the
# pre-sharded weights. Stops a running fleet first, waits for /health and arms the watchdog.
#   launch/relaunch.sh                      this recipe
#   SYSTEM_REINFORCE= launch/relaunch.sh    the same stack without system-prompt reinforcement
#   KV_NVFP4= launch/relaunch.sh            the same stack with the FP8 KV cache (793,000 pool tokens)
# MIT License, Copyright (c) 2026 Aiden Le.
. "$(cd "$(dirname "$0")" && pwd)/settings.sh"
exec bash "$BASE_DIR/launch/relaunch-inkling2.sh" "$@"
