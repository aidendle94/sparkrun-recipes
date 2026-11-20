#!/bin/bash
# stall_evidence.sh <out-dir> [tag] — what every rank is doing right now, for a stalled fleet (bench/stall_hunt.py).
# Per rank into <out-dir>/<tag>-rank<N>.txt: the [stall] lines of the collective recorder (SPARK_STALL_DUMP), the last
# log lines, GPU power and utilisation, the CPU time of every thread of the scheduler over 3 s (which threads spin),
# the RDMA counters, and py-spy stacks (plain and --native) of the scheduler. Read-only; run it as often as wanted
# (tag = when: t60, t330, ...). FLEET_ENV as for the launcher. MIT License, Copyright (c) 2026 Aiden Le.
out=${1:?out dir}; tag=${2:-now}; NAME=inkling2
FLEET_ENV="${FLEET_ENV:-$(dirname "$0")/../launch/fleet.env}"; . "$FLEET_ENV"
mkdir -p "$out"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cat > "$T/in.sh" <<'IN'
p=$(ps -eo pid,args | grep 'sglang::scheduler' | grep -v grep | head -1 | while read p rest; do echo $p; done)
echo "== scheduler pid $p; threads: cpu ticks now and 3 s later (utime+stime), state, name"
snap() { for t in /proc/$p/task/*; do printf '%s %s %s\n' "${t##*/}" "$(awk '{print $14+$15, $3}' $t/stat 2>/dev/null)" "$(cat $t/comm 2>/dev/null)"; done; }
snap > /tmp/.se1; sleep 3; snap > /tmp/.se2
join /tmp/.se1 /tmp/.se2 | awk '{d=$5-$2; if (d>0) print "  tid " $1 " +" d " ticks  state " $6 "  " $7 " " $8}' | sort -k3 -nr | head -20
echo "  (threads total: $(wc -l < /tmp/.se2))"
echo "== py-spy dump"; timeout 30 py-spy dump --locals --pid $p 2>&1 | grep -vE '^\s+(self|args|kwargs|module|forward_batch): ' | cut -c1-260 | sed -n '1,160p'
echo "== py-spy dump --native"; timeout 60 py-spy dump --native --pid $p 2>&1 | sed -n '1,45p'
IN
cat > "$T/host.sh" <<'HOST'
echo "== $(date -u '+%F %T') $(hostname)"; uptime
echo "== nvidia-smi"; timeout 20 nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv,noheader 2>&1
echo "== [stall] lines"; docker logs inkling2 2>&1 | grep -F '[stall]' | grep -v 'watch armed' | tail -6
echo "== last log lines"; docker logs --tail 400 inkling2 2>&1 | grep -vE 'GET /metrics|GET /health' | tail -12 | cut -c1-400
echo "== rdma counters (non-zero)"; for h in /sys/class/infiniband/*; do printf '%s:' "${h##*/}"; for c in $h/ports/1/hw_counters/*; do v=$(cat $c 2>/dev/null); case "${c##*/}" in out_of_sequence|packet_seq_err|local_ack_timeout_err|rnr_nak_retry_err|roce_adp_retrans|roce_adp_retrans_to|req_cqe_error|resp_cqe_error|implied_nak_seq_err|duplicate_request) [ "$v" != 0 ] && printf ' %s=%s' "${c##*/}" "$v";; esac; done; echo; done
HOST
for r in 0 1 2 3; do
  f="$out/$tag-rank$r.txt"
  if [ $r = 0 ]; then
    ( bash "$T/host.sh"; timeout 150 docker exec --privileged -i $NAME sh -s < "$T/in.sh" ) > "$f" 2>&1 &
  else
    ( timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "${USERS[$r]}@${NODES[$r]}" 'bash -s' < "$T/host.sh"
      timeout 150 ssh -o BatchMode=yes -o ConnectTimeout=10 "${USERS[$r]}@${NODES[$r]}" "docker exec --privileged -i $NAME sh -s" < "$T/in.sh" ) > "$f" 2>&1 &
  fi
done
wait; ls -la "$out" | grep "$tag-" | awk '{print $5, $9}'
