#!/bin/bash
# window-sgl.sh — deadline-guarded test window on the production fleet: stop production -> launch this stack on a test
# port with the given knobs -> wait for /health (<= 40 min) -> smoke + optional benches -> collect rank logs -> stop ->
# relaunch production (launch/relaunch.sh; RESTORE=previous runs PREVIOUS_STACK_RELAUNCH from fleet.env instead).
# Run a COPY of this file: bash reads scripts incrementally, so editing a running one breaks it.
# env: DEADLINE (6600 s) BENCH (0|1) RESTORE (sgl|previous) PORT (8888) plus the launcher's knobs (CHUNK, SPEC_K, ...).
# MIT License, Copyright (c) 2026 Aiden Le.
HERE=$(cd "$(dirname "$0")" && pwd); FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
DEADLINE="${DEADLINE:-6600}"; BENCH="${BENCH:-0}"; SGL_PORT=${PORT:-8888}; SGL=http://127.0.0.1:$SGL_PORT; MODEL=${SERVED:-deepseek-v4.1-flash}
case "${LOG_DIR:-sgl-dsv41-logs}" in /*) L=$LOG_DIR;; *) L=$HOME/${LOG_DIR:-sgl-dsv41-logs};; esac; mkdir -p "$L"
B=$HERE/../../../../bench; WLOG=$HOME/sgl-watchdog.log   # the shared benches live at the repository root
out=$L/dsv41-sgl-$(date +%H%M%S).log; echo "start $(date +%T) deadline ${DEADLINE}s chunk=${CHUNK:-2048} k=${SPEC_K:-5} mxfp8=${MXFP8:-b12x} roce=${ROCE_AR:-1} bench=$BENCH" > $out; echo $out > $HOME/.dsv41_window_current
FLAG=$HOME/.dsv41_sgl_done; rm -f $FLAG $FLAG.restoring
restore_prod() {
  [ -f $FLAG.restoring ] && return; touch $FLAG.restoring  # the deadline subshell and the main flow must not both restore
  echo "=== stop the window stack + restore production (${RESTORE:-sgl}) $(date +%T) ===" >> $out; bash $HERE/launch-sgl-dsv41.sh --stop >> $out 2>&1
  unset IMAGE ROCE_AR ROCE_HCA PORT CHUNK SPEC_K SERVED SERVED_ALIASES KVTOK MAXREQ CTX MEMFRAC EXTRA_ARGS NCCL_EXTRA MXFP8 THINKING_DEFAULT; sleep 10
  if [ "${RESTORE:-sgl}" = previous ]; then [ -n "${PREVIOUS_STACK_RELAUNCH:-}" ] && bash -c "$PREVIOUS_STACK_RELAUNCH" >> $out 2>&1 || echo "RESTORE=previous but PREVIOUS_STACK_RELAUNCH is empty in fleet.env: nothing restored" >> $out; else bash $HERE/relaunch.sh >> $out 2>&1; fi
}
( sleep $DEADLINE; if [ ! -f $FLAG ]; then echo "DEADLINE HIT $(date +%T)" >> $out; restore_prod; fi ) & DL=$!
finish() { touch $FLAG; kill $DL 2>/dev/null; kill $MS 2>/dev/null; }
wp=$(grep -o "watchdog started (pid [0-9]*" $WLOG 2>/dev/null | tail -1 | grep -o "[0-9]*$")
if [ -n "$wp" ] && ps -p $wp -o args= | grep -q "launch/watchdog.sh"; then kill $wp && echo "production watchdog $wp disarmed" >> $out; fi
rm -f $HOME/.sgl-watchdog.lock
( while [ ! -f $FLAG ]; do m="$(date +%T) memavail rank0=$(free -g | awk '/Mem:/{print $7}')"; for r in 1 2 3; do m="$m rank$r=$(ssh -o BatchMode=yes -o ConnectTimeout=4 ${USERS[$r]}@${NODES[$r]} "free -g | awk '/Mem:/{print \$7}'" 2>/dev/null)"; done; echo "$m" >> $out; sleep 15; done ) & MS=$!
[ -n "${PREVIOUS_STACK_STOP:-}" ] && bash -c "$PREVIOUS_STACK_STOP" >> $out 2>&1
bash $HERE/launch-sgl-dsv41.sh --stop >> $out 2>&1; echo "production stopped $(date +%T)" >> $out; sleep 10
echo "=== launch the window stack on :$SGL_PORT $(date +%T) ===" >> $out
PORT=$SGL_PORT bash $HERE/launch-sgl-dsv41.sh >> $out 2>&1 || { echo "launch failed" >> $out; finish; restore_prod; echo "WINDOW END $(date +%T)" >> $out; exit 1; }
t0=$(date +%s); seen=0
until curl -sf -m 3 $SGL/health >/dev/null; do
  sleep 20; [ $(( $(date +%s)-t0 )) -gt 2400 ] && { echo "not healthy after 40 min" >> $out; break; }
  docker ps --format '{{.Names}}' | grep -q '^sgldsv41$' || { echo "head container exited $(date +%T)" >> $out; break; }
  n=$(docker logs sgldsv41 2>&1 | grep -E "Traceback|Error|DSV4 memory calculation|Capture|engram layer|engram_store installed|MXFP8|indexer schedule|prefill flush|RoCE collectives|SM12x|Load weight end|The server is fired up|watchdog" | grep -vE "Warning|warn" | wc -l)
  if [ "$n" != "$seen" ]; then docker logs sgldsv41 2>&1 | grep -E "Traceback|Error|DSV4 memory calculation|Capture|engram layer|engram_store installed|MXFP8|indexer schedule|prefill flush|RoCE collectives|SM12x|Load weight end|The server is fired up|watchdog" | grep -vE "Warning|warn" | tail -n $((n - seen)) | cut -c1-220 >> $out; seen=$n; fi
done
if curl -sf -m 3 $SGL/health >/dev/null; then
  echo "healthy $(date +%T) after $(( $(date +%s)-t0 )) s" >> $out
  echo "=== smoke ===" >> $out
  curl -sf -m 120 $SGL/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 20, comma separated, nothing else.\"}],\"max_tokens\":80,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" | python3 -c "import sys,json; r=json.load(sys.stdin); print('  count:', r['choices'][0]['message']['content'][:90].replace(chr(10),' '), '| tokens', r['usage']['completion_tokens'])" >> $out 2>&1
  NEEDLE_SALT=$(date +%s%N) timeout 600 python3 $B/needle.py --base $SGL/v1 --model $MODEL --targets 32768 >> $out 2>&1
  if [ "$BENCH" = "1" ]; then
    NEEDLE_SALT=$(date +%s%N) timeout 900 python3 $B/needle.py --base $SGL/v1 --model $MODEL --targets 131072 >> $out 2>&1
    timeout 900 python3 $B/prefill_repetitive.py $SGL 32768,131072 >> $out 2>&1
    timeout 900 python3 $B/decode_bench.py --model $MODEL $SGL >> $out 2>&1
  fi
  docker logs sgldsv41 2>&1 | grep -E "engram layer|engram_store installed|MXFP8|refused" | tail -6 | cut -c1-220 >> $out
fi
for r in 0 1 2 3; do if [ $r = 0 ]; then docker logs sgldsv41 > $L/dsv41-sgl-rank0-$(date +%H%M%S).log 2>&1; else ssh -o BatchMode=yes ${USERS[$r]}@${NODES[$r]} 'docker logs sgldsv41 2>&1' > $L/dsv41-sgl-rank$r-$(date +%H%M%S).log; fi; done
finish; restore_prod
echo "WINDOW END $(date +%T)" >> $out
