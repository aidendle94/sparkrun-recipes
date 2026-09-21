#!/bin/bash
# watchdog.sh — fleet watchdog for the production DeepSeek-V4.1-Flash stack: SGLang TP4/EP4 on four DGX Sparks, one
# container (sgldsv41) per node, the head on rank 0. Runs on rank 0. launch/relaunch.sh starts it once a boot is
# healthy (PORT=8210 nohup setsid bash launch/watchdog.sh) and disarms it by killing the pid from this log's
# "watchdog started (pid N, ...)" line and removing the lock file. Start it by hand the same way; stop it with kill.
#
# Every CHECK_INTERVAL s it asks the head whether it is alive. After FAIL_THRESHOLD consecutive failed checks it
# (1) captures evidence from all four ranks into $LOG_DIR/postmortem/<timestamp>/ and (2) relaunches the fleet with
# launch/production.sh, waits up to READY_TIMEOUT s for /health, then goes back to probing.
#
# Why Docker restart policies cannot do this job on this fleet:
#   - the three workers are headless torch.distributed peers; when the head dies they lose the rendezvous and exit
#     with status 0, so `--restart on-failure` never fires on the ranks that need restarting;
#   - a wedged head (scheduler stuck in a collective, HTTP server still up) is a running container as far as Docker
#     can tell; only a request from outside notices;
#   - `--restart always` on a worker brings it back on its own and it re-joins --dist-init-addr while the old head
#     is still dying or before the new head owns the rendezvous: it then sits on the GPU with a stale rendezvous
#     member and the next fleet launch dies with "CUDA device busy" or hangs in torch.distributed init.
#   So every container runs with --restart no and there is one recovery path: tear all four ranks down, wait until
#   every GPU is free, start workers 3..1 and the head last (launch-sgl-dsv41.sh does that), then wait for /health.
#
# Liveness = GET /health answers 200 AND, once armed, a 3-token chat completion returns a "choices" field within
# PROBE_TIMEOUT s. SGLang's /health answers 503 while it is starting or shutting down and when its scheduler has
# been silent for 20 s; the scheduler acks health probes between prefill chunks, so a long prefill keeps /health at
# 200 while the chat probe waits in the queue. A probe timeout with /health fine therefore means "busy" (a
# multi-minute prefill or a full queue holds the scheduler) and counts as a failure only after BUSY_GRACE s of
# continuous busy. The watchdog arms itself on the first check that passes both tests and never relaunches before
# that: a boot takes ~10 min, and a watchdog that probes a booting fleet would tear it down forever.
#
# Files: $HOME/sgl-watchdog.log (kept below 20000 lines: trimmed to the last 5000 with the started line restated, so
# relaunch.sh keeps finding the pid) and $HOME/.sgl-watchdog.lock (flock; carries the pid). When the lock file
# vanishes or carries another pid the watchdog stands down by itself, so a relaunch that failed to kill it stays safe.
# env: FLEET_ENV (launch/fleet.env: NODES USERS LOG_DIR) PORT (8210) CHECK_INTERVAL (60) FAIL_THRESHOLD (3)
#      PROBE_TIMEOUT (45) BUSY_GRACE (600) READY_TIMEOUT (3600) RELAUNCH_LIMIT (3: consecutive relaunch attempts
#      without a healthy period in between before it gives up and waits for a human) MODEL (model id for the chat
#      probe; default: the first id listed by /v1/models)
# MIT License, Copyright (c) 2026 Aiden Le.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "watchdog: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
for v in NODES USERS; do eval "n=\${#$v[@]}"; [ "$n" = 4 ] || { echo "watchdog: $v in $FLEET_ENV must list exactly 4 ranks" >&2; exit 1; }; done
export PORT="${PORT:-8210}"   # exported: production.sh reads it
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"; FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-45}"
BUSY_GRACE="${BUSY_GRACE:-600}"; READY_TIMEOUT="${READY_TIMEOUT:-3600}"; RELAUNCH_LIMIT="${RELAUNCH_LIMIT:-3}"; MODEL="${MODEL:-}"
case "${LOG_DIR:-sgl-dsv41-logs}" in /*) L=$LOG_DIR;; *) L=$HOME/${LOG_DIR:-sgl-dsv41-logs};; esac
LOG=$HOME/sgl-watchdog.log; LOCK=$HOME/.sgl-watchdog.lock; NAME=sgldsv41; BASE=http://127.0.0.1:$PORT
LOG_MAX=20000; LOG_KEEP=5000
mkdir -p "$L" || { echo "watchdog: cannot create LOG_DIR $L" >&2; exit 1; }
TMP=$(mktemp -d) || exit 1

ts() { date '+%F %T'; }
log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }
still_owner() { [ "$(cat "$LOCK" 2>/dev/null)" = "$$" ]; }

# One instance per home directory. Append-open, so an instance that loses the race does not truncate the winner's pid.
exec 9>>"$LOCK" || { echo "watchdog: cannot open $LOCK" >&2; exit 1; }
if ! flock -n 9; then
  msg="watchdog not started: another instance (pid $(cat "$LOCK" 2>/dev/null)) holds $LOCK — kill it first, or use launch/relaunch.sh"
  echo "$msg" >&2; log "$msg"; exit 1
fi
echo $$ > "$LOCK"
START_INFO="port $PORT, check every ${CHECK_INTERVAL}s, relaunch after $FAIL_THRESHOLD failures, probe timeout ${PROBE_TIMEOUT}s, busy grace ${BUSY_GRACE}s, ready timeout ${READY_TIMEOUT}s, log dir $L, fleet $FLEET_ENV"
printf 'watchdog started (pid %s, %s, %s)\n' "$$" "$(ts)" "$START_INFO" >> "$LOG"

# Every step that can take a while runs as a background child + wait, so a kill from relaunch.sh takes effect at
# once instead of after the current curl/ssh/launcher returns (bash defers traps while a foreground child runs).
CHILD=""
run() { "$@" & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return $rc; }
on_signal() {
  log "watchdog stopped (pid $$, signal): no automatic recovery until launch/relaunch.sh arms a new one"
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; still_owner && rm -f "$LOCK"; exit 0
}
trap on_signal TERM INT HUP
trap 'rm -rf "$TMP"' EXIT

trim_log() {  # bounded log; the started line is restated at the end so "grep | tail -1" in relaunch.sh still yields this pid
  local n; n=$(wc -l < "$LOG" 2>/dev/null) || n=0
  [ "$n" -gt "$LOG_MAX" ] || return 0
  { tail -n "$LOG_KEEP" "$LOG"
    printf '%s log trimmed from %s to %s lines; watchdog started (pid %s, %s, %s) [restated after the trim]\n' "$(ts)" "$n" "$LOG_KEEP" "$$" "$(ts)" "$START_INFO"
  } > "$TMP/trim" && cat "$TMP/trim" > "$LOG"   # cat, not mv: keeps the inode for anyone tailing the log
}

head_state() {  # one phrase about the head container, for messages
  docker inspect --format 'head container {{.State.Status}} (exit code {{.State.ExitCode}}, oom killed {{.State.OOMKilled}})' "$NAME" 2>/dev/null \
    || printf 'no container named %s on this node' "$NAME"
}
head_running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; }

REASON=""; RESULT=""
# fetch TIMEOUT [curl args] URL -> $TMP/code (HTTP status), $TMP/body, $TMP/err; returns curl's status (28 = --max-time hit)
fetch() { local t=$1; shift; run curl -sS --max-time "$t" -o "$TMP/body" -w '%{http_code}' "$@" > "$TMP/code" 2> "$TMP/err"; }
http_code() { cat "$TMP/code" 2>/dev/null; }
curl_err() { tr -d '\n' < "$TMP/err" 2>/dev/null | head -c 160; }

check_health() {  # 0 = HTTP 200; 1 = another status; 2 = no answer at all
  fetch "$PROBE_TIMEOUT" "$BASE/health"; local rc=$?
  if [ $rc = 0 ] && [ "$(http_code)" = 200 ]; then return 0; fi
  if [ $rc = 0 ]; then REASON="/health answered HTTP $(http_code) (SGLang: 503 = starting, shutting down, or scheduler silent for 20 s); $(head_state)"; return 1; fi
  REASON="/health on $BASE gave no answer (curl $rc: $(curl_err)); $(head_state)"; return 2
}
discover_model() {
  fetch 10 "$BASE/v1/models"
  MODEL=$(grep -o '"id": *"[^"]*"' "$TMP/body" 2>/dev/null | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  [ -n "$MODEL" ] || MODEL=deepseek-v4.1-flash
  log "chat probe uses model id '$MODEL' (first id on /v1/models; set MODEL to override)"
}
check_probe() {  # 0 = choices came back; 1 = answered without choices; 2 = timed out (busy); 3 = no answer
  local body='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with the word OK."}],"max_tokens":3,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
  fetch "$PROBE_TIMEOUT" -H 'content-type: application/json' -d "$body" "$BASE/v1/chat/completions"; local rc=$?
  if [ $rc = 0 ]; then
    [ "$(http_code)" = 200 ] && grep -q '"choices"' "$TMP/body" && return 0
    REASON="chat probe answered HTTP $(http_code) without choices: $(tr -d '\n' < "$TMP/body" | head -c 200) (wrong model id '$MODEL'? compare /v1/models)"; return 1
  fi
  [ $rc = 28 ] && { REASON="3-token chat probe unanswered within ${PROBE_TIMEOUT}s while /health is 200"; return 2; }
  REASON="chat probe got no answer (curl $rc: $(curl_err))"; return 3
}
check() {  # RESULT = ok | busy | fail; REASON says why
  REASON=""
  if ! check_health; then RESULT=fail; return; fi
  [ -n "$MODEL" ] || discover_model
  check_probe; case $? in 0) RESULT=ok;; 2) RESULT=busy;; *) RESULT=fail;; esac
}

postmortem() {  # $1 = directory, $2 = trigger text. Best effort, every step bounded: a dead node must not stall the relaunch.
  local d=$1 r
  { echo "watchdog postmortem $(ts) (pid $$, port $PORT)"; echo "trigger: $2"; echo; echo "last 40 watchdog log lines:"; tail -n 40 "$LOG"; } > "$d/summary.txt"
  { echo "== uptime"; uptime; echo "== docker ps -a"; docker ps -a --filter "name=$NAME" --format '{{.Names}} {{.Status}} {{.Image}}'
    echo "== docker inspect .State"; docker inspect --format '{{json .State}}' "$NAME"; echo "== free -g"; free -g
    echo "== nvidia-smi"; timeout 20 nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv
  } > "$d/rank0-state.txt" 2>&1
  run timeout 120 docker logs "$NAME" > "$d/rank0-docker.log" 2>&1
  # dmesg is often root-only (kernel.dmesg_restrict=1); the file then just holds the "Operation not permitted" line.
  timeout 20 dmesg -T 2>&1 | tail -300 > "$d/rank0-dmesg.txt"
  for r in 1 2 3; do
    run timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=10 "${USERS[$r]}@${NODES[$r]}" \
      "echo '== uptime'; uptime; echo '== docker ps -a'; docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}} {{.Image}}' 2>&1
       echo '== docker inspect .State'; docker inspect --format '{{json .State}}' $NAME 2>&1; echo '== free -g'; free -g
       echo '== nvidia-smi'; timeout 20 nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv 2>&1
       echo '== docker logs --tail 200'; docker logs --tail 200 $NAME 2>&1; echo '== dmesg -T | tail -300 (root-only on many hosts)'; dmesg -T 2>&1 | tail -300" \
      > "$d/rank$r.txt" 2>&1 || echo "(ssh ${USERS[$r]}@${NODES[$r]} failed or timed out, rc $? — node down, or ssh keys/NODES/USERS in $FLEET_ENV wrong)" >> "$d/rank$r.txt"
  done
  log "evidence captured in $d: $(ls "$d" | tr '\n' ' ')"
}

recover() {  # $1 = trigger text. Returns 0 once the fleet answers /health again, 1 after RELAUNCH_LIMIT failed attempts.
  local attempt d t0 waited rc
  for attempt in $(seq 1 "$RELAUNCH_LIMIT"); do
    still_owner || { log "lock $LOCK no longer mine before relaunch attempt $attempt (a relaunch/window script took over): standing down"; exit 0; }
    d=$L/postmortem/$(date +%Y%m%d-%H%M%S); mkdir -p "$d"
    log "RECOVERY attempt $attempt/$RELAUNCH_LIMIT: capturing evidence into $d, then relaunching the fleet"
    postmortem "$d" "$1"
    log "running launch/production.sh (log: $d/relaunch.log); it tears every rank down before starting any"
    run bash "$HERE/production.sh" > "$d/relaunch.log" 2>&1; rc=$?
    if [ $rc != 0 ]; then log "launch/production.sh exited $rc — read $d/relaunch.log (preflight: image, engram rows, ssh to the workers, GPU still busy)"; run sleep 30; continue; fi
    t0=$(date +%s)
    while :; do
      run sleep 20; waited=$(( $(date +%s) - t0 ))
      if check_health; then log "fleet healthy ${waited}s after relaunch attempt $attempt; arming again on the next successful probe"; return 0; fi
      still_owner || { log "lock $LOCK no longer mine while waiting for the boot: standing down"; exit 0; }
      if ! head_running; then log "head container exited ${waited}s into the boot: $(head_state) — read $d/relaunch.log and 'docker logs $NAME'"; break; fi
      if [ $waited -ge "$READY_TIMEOUT" ]; then log "not healthy ${waited}s after relaunch attempt $attempt (READY_TIMEOUT ${READY_TIMEOUT}s): $REASON — read $d/relaunch.log and 'docker logs $NAME'"; break; fi
      [ $((waited % 300)) -lt 20 ] && log "booting: ${waited}s, head container running, $REASON"
    done
  done
  log "GIVING UP after $RELAUNCH_LIMIT relaunch attempts: the fleet needs a human. Read the newest directories under $L/postmortem, fix the cause, then run launch/relaunch.sh. This watchdog keeps probing and arms itself again once the fleet is healthy"
  return 1
}

armed=0; fails=0; busy_since=0; unarmed_checks=0
while :; do
  still_owner || { log "lock $LOCK no longer carries pid $$ (removed or taken over by a relaunch/window script): standing down"; exit 0; }
  trim_log
  check; now=$(date +%s)
  case $RESULT in
    ok)
      if [ $armed = 0 ]; then armed=1; unarmed_checks=0; log "armed: /health 200 and the chat probe answered (model '$MODEL'); checking every ${CHECK_INTERVAL}s, relaunch after $FAIL_THRESHOLD consecutive failures"; fi
      [ $fails -gt 0 ] && log "healthy again after $fails failed check(s); counter reset"
      [ $busy_since != 0 ] && log "busy cleared after $((now - busy_since))s"
      fails=0; busy_since=0;;
    busy)
      if [ $armed = 0 ]; then log "healthy but busy, not armed yet: $REASON"
      else
        [ $busy_since = 0 ] && busy_since=$now
        if [ $((now - busy_since)) -ge "$BUSY_GRACE" ]; then fails=$((fails + 1)); log "busy for $((now - busy_since))s, past BUSY_GRACE ${BUSY_GRACE}s: failure $fails/$FAIL_THRESHOLD ($REASON)"
        else log "busy $((now - busy_since))s of ${BUSY_GRACE}s grace: $REASON (a long prefill or a full queue; not a failure yet)"; fi
      fi;;
    fail)
      busy_since=0
      if [ $armed = 0 ]; then
        unarmed_checks=$((unarmed_checks + 1))
        [ $unarmed_checks -le 3 ] || [ $((unarmed_checks % 10)) = 0 ] && log "not healthy, not armed (check $unarmed_checks): $REASON — normal during a boot (~10 min); if nothing is booting, run launch/relaunch.sh"
      else fails=$((fails + 1)); log "check failed $fails/$FAIL_THRESHOLD: $REASON"; fi;;
  esac
  if [ $armed = 1 ] && [ $fails -ge "$FAIL_THRESHOLD" ]; then
    recover "$fails consecutive failed checks; last: $REASON"; armed=0; fails=0; busy_since=0; unarmed_checks=0
  fi
  run sleep "$CHECK_INTERVAL"
done
