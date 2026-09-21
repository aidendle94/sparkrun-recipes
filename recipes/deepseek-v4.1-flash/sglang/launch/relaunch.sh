#!/bin/bash
# relaunch.sh — orchestrated relaunch of the production fleet: disarm the watchdog, stop whatever serves the production
# port (a previous boot of this stack, or another engine via PREVIOUS_STACK_STOP in fleet.env), launch production.sh,
# wait for /health (<= 40 min), print the boot facts, re-arm the watchdog. Log: $LOG_DIR/dsv41-sgl-relaunch-HHMMSS.log
# MIT License, Copyright (c) 2026 Aiden Le.
HERE=$(cd "$(dirname "$0")" && pwd); FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
case "${LOG_DIR:-sgl-dsv41-logs}" in /*) L=$LOG_DIR;; *) L=$HOME/${LOG_DIR:-sgl-dsv41-logs};; esac
mkdir -p "$L" && [ -w "$L" ] || { echo "ABORT: cannot write to LOG_DIR=$L (fleet.env)" >&2; exit 1; }
WLOG=$HOME/sgl-watchdog.log; PORT=${PORT:-8210}
out=$L/dsv41-sgl-relaunch-$(date +%H%M%S).log; echo "relaunch start $(date +%T)" | tee $out
wp=$(grep -o "watchdog started (pid [0-9]*" $WLOG 2>/dev/null | tail -1 | grep -o "[0-9]*$")
if [ -n "$wp" ] && ps -p $wp -o args= | grep -q "launch/watchdog.sh"; then kill $wp && echo "watchdog $wp disarmed" | tee -a $out; fi
rm -f $HOME/.sgl-watchdog.lock
[ -n "${PREVIOUS_STACK_STOP:-}" ] && { bash -c "$PREVIOUS_STACK_STOP" >> $out 2>&1; sleep 5; }
fail() { echo "$1 — the fleet is DOWN and the watchdog is DISARMED; fix the cause, then run relaunch.sh again (log: $out)" | tee -a $out; exit $2; }
PORT=$PORT bash $HERE/production.sh >> $out 2>&1 || fail "launch failed" 1
t0=$(date +%s)
until curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null; do
  sleep 20; [ $(( $(date +%s)-t0 )) -gt 2400 ] && fail "not healthy after 40 min (docker logs sgldsv41 on each node)" 2
  docker ps --format '{{.Names}}' | grep -q '^sgldsv41$' || fail "head container exited $(date +%T) (docker logs sgldsv41)" 3
done
echo "healthy $(date +%T) after $(( $(date +%s)-t0 )) s" | tee -a $out
docker logs sgldsv41 2>&1 | grep -E "DSV4 memory calculation|RoCE collectives on|Served-model aliases|The server is fired up" | tail -4 | cut -c1-200 | tee -a $out
PORT=$PORT nohup setsid bash $HERE/watchdog.sh > /dev/null 2>&1 & sleep 3; echo "watchdog: $(tail -1 $WLOG)" | tee -a $out
