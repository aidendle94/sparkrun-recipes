#!/bin/bash
# relaunch-inkling2.sh — orchestrated relaunch of the Inkling-2-Small fleet (after the DeepSeek-V4.1 recipe's
# launch/relaunch.sh): disarm the watchdog, launch (launch-inkling2.sh removes a previous boot of this stack first),
# wait for /health (<= 40 min), print the boot facts, re-arm the watchdog. Log: $LOG_DIR/inkling2-relaunch-HHMMSS.log
# The launcher's env (CTX, MEMFRAC, EXTRA_ARGS, ...) passes through, also to the watchdog started here: its own
# relaunches boot the same configuration.
# MIT License, Copyright (c) 2026 Aiden Le.
HERE=$(cd "$(dirname "$0")" && pwd); FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
case "${LOG_DIR:-inkling2-logs}" in /*) L=$LOG_DIR;; *) L=$HOME/${LOG_DIR:-inkling2-logs};; esac
mkdir -p "$L" && [ -w "$L" ] || { echo "ABORT: cannot write to LOG_DIR=$L (fleet.env)" >&2; exit 1; }
WLOG=$HOME/inkling2-watchdog.log; PORT=${PORT:-8210}
out=$L/inkling2-relaunch-$(date +%H%M%S).log; echo "relaunch start $(date +%T)" | tee $out
wp=$(grep -o "watchdog started (pid [0-9]*" $WLOG 2>/dev/null | tail -1 | grep -o "[0-9]*$")
if [ -n "$wp" ] && ps -p $wp -o args= | grep -q "watchdog-inkling2.sh"; then kill $wp && echo "watchdog $wp disarmed" | tee -a $out; fi
rm -f $HOME/.inkling2-watchdog.lock
# Another engine (OTHER_CONTAINER in fleet.env) must be off: launch-inkling2.sh refuses to start while it runs anywhere.
fail() { echo "$1 — the fleet is DOWN and the watchdog is DISARMED; fix the cause, then run relaunch-inkling2.sh again (log: $out)" | tee -a $out; exit $2; }
# The head's memory through the boot (bench/boot_mem.sh, /proc only, 15 min): the KV pool is sized from what the head has left, and that varies.
bm=$L/inkling2-bootmem-$(date +%m%d-%H%M%S).tsv; [ -f $HERE/../bench/boot_mem.sh ] && { nohup setsid bash $HERE/../bench/boot_mem.sh $bm 900 2 > /dev/null 2>&1 & echo "boot memory samples: $bm" | tee -a $out; }
PORT=$PORT bash $HERE/launch-inkling2.sh >> $out 2>&1 || fail "launch failed" 1
t0=$(date +%s)
until curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null; do
  sleep 20; [ $(( $(date +%s)-t0 )) -gt 2400 ] && fail "not healthy after 40 min (docker logs inkling2 on each node)" 2
  docker ps --format "{{.Names}}" | grep -q "^inkling2$" || fail "head container exited $(date +%T) (docker logs inkling2)" 3
done
echo "healthy $(date +%T) after $(( $(date +%s)-t0 )) s" | tee -a $out
docker logs inkling2 2>&1 | grep -E "Load weight end|KV Cache is allocated|Served-model aliases|The server is fired up" | tail -4 | cut -c1-200 | tee -a $out
PORT=$PORT nohup setsid bash $HERE/watchdog-inkling2.sh > /dev/null 2>&1 & sleep 3; echo "watchdog: $(tail -1 $WLOG)" | tee -a $out
