#!/bin/bash
# quickstart.sh — from four DGX Sparks to a serving endpoint in one command. Run it on the node that will be rank 0.
#
#   ./quickstart.sh NODE0 NODE1 NODE2 NODE3    first run: the four nodes' fabric addresses, this node first
#                                               (each as ADDRESS or USER@ADDRESS when the login differs from yours)
#   ./quickstart.sh                             later runs: reads launch/fleet.env
#   ./quickstart.sh --check                     only the checks: starts nothing, downloads nothing
#   ./quickstart.sh --no-launch                 everything up to the launch (image, module, weights)
#   ./quickstart.sh --status                    is it serving, and with which KV cache
#   ./quickstart.sh --stop                      stop the fleet and its watchdog
#
# What it does, in order. Every step is skipped when its result is already there, so the script can be run again
# after any failure:
#   1. writes launch/fleet.env from the four addresses (interface, RDMA devices, subnet and home directories are
#      read from the nodes), unless the file exists;
#   2. checks every node: ssh without a password, docker with the GPU, memory, disk, the RDMA devices;
#   3. pulls the image on every node;
#   4. copies the prebuilt GEMM module out of the image (the launcher runs without it, with less free memory);
#   5. downloads every node's own weights folder (about 93 GB a node) into ~/inkling2-weights on that node;
#   6. launches the fleet and waits until it answers (about 5 minutes; about 17 the first time, while kernels compile);
#   7. checks that every node runs the NVFP4 KV cache and sends one chat request.
# MIT License, Copyright (c) 2026 Aiden Le.
set -o pipefail
. "$(cd "$(dirname "$0")" && pwd)/launch/settings.sh"
MODE=run; ARGS=()
for a in "$@"; do case "$a" in
  --check) MODE=check;; --no-launch) MODE=nolaunch;; --status) MODE=status;; --stop) MODE=stop;;
  -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  -*) echo "unknown option $a (see --help)" >&2; exit 2;;
  *) ARGS+=("$a");; esac; done
PORT=${PORT:-8210}; WEIGHTS_GB=93; IMAGE_GB=37; MIN_BYTES=98000000000; SHARDS=59
say() { printf '\n== %s\n' "$*"; }
ok() { printf '   ok    %s\n' "$*"; }
bad() { printf '   FAIL  %s\n' "$*"; FAILED=1; }
die() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

# ---- 1. the site file ------------------------------------------------------------------------------------------
write_fleet_env() {
  local users=() addrs=() homes=() ifaces=() cidrs=() r spec out
  [ ${#ARGS[@]} = 4 ] || die "give the four nodes' fabric addresses, this node first:  ./quickstart.sh 192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14
         (USER@ADDRESS where the login on a node differs from '$USER'). The fabric address is the one on the
         200G port: run 'ip -br addr' on a node and take the address of its enp1s0f... interface."
  for spec in "${ARGS[@]}"; do case "$spec" in *@*) users+=("${spec%@*}"); addrs+=("${spec#*@}");; *) users+=("$USER"); addrs+=("$spec");; esac; done
  ip -o -4 addr show | awk '{print $4}' | grep -q "^${addrs[0]}/" \
    || die "${addrs[0]} is not an address of this node. Run the script on the node you list first (rank 0)."
  for r in 0 1 2 3; do
    if [ $r = 0 ]; then out=$(echo "$HOME"; ip -o -4 addr show | awk -v a="${addrs[0]}" '$4 ~ "^"a"/" {print $2, $4}')
    else out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "${users[$r]}@${addrs[$r]}" \
        "echo \$HOME; ip -o -4 addr show | awk -v a=${addrs[$r]} '\$4 ~ \"^\"a\"/\" {print \$2, \$4}'" 2>&1) \
      || die "cannot log in to ${users[$r]}@${addrs[$r]} without a password. From this node run:
         ssh-copy-id ${users[$r]}@${addrs[$r]}
         then run this script again. ($(echo "$out" | tail -1))"; fi
    homes+=("$(echo "$out" | sed -n 1p)"); ifaces+=("$(echo "$out" | sed -n 2p | cut -d' ' -f1)"); cidrs+=("$(echo "$out" | sed -n 2p | cut -d' ' -f2)")
    [ -n "${ifaces[$r]}" ] || die "node ${addrs[$r]} has no interface with the address ${addrs[$r]} ('ip -br addr' there)."
  done
  for r in 1 2 3; do [ "${ifaces[$r]}" = "${ifaces[0]}" ] || die "the fabric interface must have the same name on every node: ${ifaces[0]} on ${addrs[0]}, ${ifaces[$r]} on ${addrs[$r]}."; done
  local subnet hcas first rest d
  subnet=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "${cidrs[0]}") || die "cannot read the subnet of ${cidrs[0]}"
  first=""; rest=""
  for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue
    grep -q ACTIVE "$d/ports/1/state" 2>/dev/null || continue
    if [ -e "$d/device/net/${ifaces[0]}" ]; then first=$(basename "$d"); else rest="$rest,$(basename "$d")"; fi
  done
  [ -n "$first" ] || die "no active RDMA device on ${ifaces[0]} (ls /sys/class/infiniband; rdma link show). The fabric port needs RoCE."
  hcas="$first$rest"
  cat > "$FLEET_ENV" <<EOT
# launch/fleet.env — written by quickstart.sh on $(date -u +%F). The site file of this fleet (git-ignored). The
# meaning of every line is in ../sglang/launch/fleet.env.example. Delete this file to have quickstart.sh write it again.
NODES=(${addrs[*]})
USERS=(${users[*]})
HOMES=(${homes[*]})
FABRIC_IFACE=${ifaces[0]}
RDMA_HCAS=$hcas
FABRIC_SUBNET=$subnet
LOG_DIR=inkling2-logs
SERVED_ALIASES="\${SERVED_ALIASES:-}"
EOT
  ok "wrote $FLEET_ENV: interface ${ifaces[0]}, RDMA devices $hcas, subnet $subnet"
}

on() { local r=$1; shift; if [ "$r" = 0 ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "${USERS[$r]}@${NODES[$r]}" "$*"; fi; }

load_fleet_env() {
  [ -f "$FLEET_ENV" ] || die "$FLEET_ENV not found. First run:  ./quickstart.sh NODE0 NODE1 NODE2 NODE3"
  . "$FLEET_ENV"
  [ "${#NODES[@]}" = 4 ] && [ "${#USERS[@]}" = 4 ] && [ "${#HOMES[@]}" = 4 ] || die "$FLEET_ENV must list exactly 4 NODES, USERS and HOMES"
}

status() {
  load_fleet_env
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; then echo "serving on http://${NODES[0]}:$PORT/v1 (model inkling-2-small)"
  else echo "not serving on port $PORT"; fi
  for r in 0 1 2 3; do
    echo "  rank $r (${NODES[$r]}): $(on $r "docker ps --filter name=^inkling2\$ --format '{{.Status}}' | grep . || echo 'no container'; docker logs inkling2 2>&1 | grep -o '\[nvfp4_kv\] target KV pool: NVFP4' | head -1 | sed 's/.*/NVFP4 KV cache/'" 2>&1 | tr '\n' ' ')"
  done
  local wp; wp=$(grep -o "watchdog started (pid [0-9]*" "$HOME/inkling2-watchdog.log" 2>/dev/null | tail -1 | grep -o "[0-9]*$")
  if [ -n "$wp" ] && ps -p "$wp" > /dev/null 2>&1; then echo "  watchdog running (pid $wp)"; else echo "  no watchdog"; fi
}

case $MODE in
  status) status; exit 0;;
  stop) exec bash "$RECIPE_DIR/launch/stop.sh";;
esac

say "1/7 the site file"
if [ -f "$FLEET_ENV" ]; then ok "$FLEET_ENV exists (delete it to write it again from the addresses)"; else write_fleet_env; fi
load_fleet_env

# ---- 2. the checks ---------------------------------------------------------------------------------------------
say "2/7 checking the four nodes"
FAILED=
for t in curl flock python3 ssh scp docker; do command -v $t > /dev/null || bad "this node has no '$t' (install it)"; done
probe() {  # what a node says about itself, one key=value a line (stderr dropped: it would break the lines)
  local r=$1 w=${HOMES[$1]}/$INKLING_REPO_REL/rank$1
  on $r "echo ssh=ok; echo arch=\$(uname -m); echo docker=\$(docker info --format '{{.ServerVersion}}' 2>/dev/null | head -1)
    echo gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo mem=\$(awk '/MemTotal/{print int(\$2/1048576)}' /proc/meminfo); echo rdma=\$(ls /sys/class/infiniband 2>/dev/null | wc -l)
    echo rdmadev=\$(ls /dev/infiniband 2>/dev/null | wc -l); echo free=\$(df -BG --output=avail ${HOMES[$r]} 2>/dev/null | tail -1 | tr -dc 0-9)
    echo shards=\$(ls $w/*.safetensors 2>/dev/null | wc -l); echo bytes=\$( (du -sb $w 2>/dev/null || echo 0) | cut -f1)
    if docker image inspect $INKLING_IMAGE > /dev/null 2>&1; then echo image=yes
      docker run --rm --gpus all --entrypoint true $INKLING_IMAGE > /dev/null 2>&1 && echo gpudocker=ok || echo gpudocker=no
    else echo image=no; fi" 2> /dev/null
}
val() { echo "$OUT" | sed -n "s/^$1=//p" | head -1; }
for r in 0 1 2 3; do
  n="rank $r (${USERS[$r]}@${NODES[$r]})"; OUT=$(probe $r); was=$FAILED; FAILED=
  if [ "$(val ssh)" != ok ]; then bad "$n: no ssh without a password. From this node run: ssh-copy-id ${USERS[$r]}@${NODES[$r]}"; continue; fi
  [ "$(val arch)" = aarch64 ] || bad "$n: architecture '$(val arch)', expected aarch64 (a DGX Spark)"
  case "$(val docker)" in [0-9]*) ;; *) bad "$n: docker does not answer for user ${USERS[$r]}. On that node: sudo usermod -aG docker ${USERS[$r]}, then log in again";; esac
  case "$(val gpu)" in *GB10*) ;; *) bad "$n: GPU '$(val gpu)', expected NVIDIA GB10 (nvidia-smi must work for this user)";; esac
  [ "$(val mem)" -ge 115 ] 2> /dev/null || bad "$n: $(val mem) GB of memory, expected a 128 GB Spark"
  [ "$(val rdma)" -ge 1 ] 2> /dev/null && [ "$(val rdmadev)" -ge 1 ] 2> /dev/null || bad "$n: no RDMA device (/sys/class/infiniband, /dev/infiniband): the RoCE fabric is not set up on this node"
  have_w=no; [ "$(val shards)" = $SHARDS ] && [ "$(val bytes)" -ge $MIN_BYTES ] 2> /dev/null && have_w=yes
  need=0; [ $have_w = yes ] || need=$((need + WEIGHTS_GB + 5)); [ "$(val image)" = yes ] || need=$((need + IMAGE_GB))
  [ "$(val free)" -ge $need ] 2> /dev/null || bad "$n: $(val free) GB free under ${HOMES[$r]}, $need GB needed (weights $WEIGHTS_GB GB, image $IMAGE_GB GB)"
  [ "$(val image)" != yes ] || [ "$(val gpudocker)" = ok ] || bad "$n: docker cannot start a container with the GPU. 'docker run --rm --gpus all $INKLING_IMAGE' must work (the NVIDIA container runtime)"
  HAVE_IMAGE[$r]=$(val image); HAVE_W[$r]=$have_w
  [ -n "$FAILED" ] || ok "$n: $(val gpu), $(val mem) GB memory, $(val free) GB free, image $(val image), weights $have_w"
  FAILED=$FAILED$was
done
[ -z "$FAILED" ] || die "fix the lines marked FAIL, then run the script again."
[ $MODE = check ] && { echo; echo "All checks passed. Run ./quickstart.sh to download and launch."; exit 0; }

# ---- 3. the image ----------------------------------------------------------------------------------------------
say "3/7 the image $INKLING_IMAGE on every node (about $IMAGE_GB GB the first time)"
pids=()
for r in 0 1 2 3; do
  if [ "${HAVE_IMAGE[$r]}" = yes ]; then ok "rank $r has it"; continue; fi
  ( on $r "docker pull -q $INKLING_IMAGE" > /dev/null 2>&1 && echo "   ok    rank $r pulled" || { echo "   FAIL  rank $r: docker pull $INKLING_IMAGE failed (network? disk?)"; exit 1; } ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || FAILED=1; done
[ -z "$FAILED" ] || die "the image is not on every node. Run 'docker pull $INKLING_IMAGE' on the failing node to see why."

# ---- 4. the prebuilt GEMM module -------------------------------------------------------------------------------
say "4/7 the prebuilt GEMM module (keeps 3 GB a node free)"
SO=$BASE_DIR/patches/spark_csf_gemm_sm120.so
if [ -s "$SO" ]; then ok "$SO exists"
else
  docker run --rm --entrypoint cat "$INKLING_IMAGE" /opt/inkling2-spark/prebuilt/spark_csf_gemm_sm120.so > "$SO.tmp" 2> /dev/null \
    && [ "$(head -c 4 "$SO.tmp" | tail -c 3)" = ELF ] && mv "$SO.tmp" "$SO" && ok "copied out of the image" \
    || { rm -f "$SO.tmp"; echo "   note  not found in the image: the launcher starts without the packed block scales (3 GB less free memory a node)"; }
fi

# ---- 5. the weights --------------------------------------------------------------------------------------------
say "5/7 the weights: each node's own folder of $WEIGHTS_REPO (about $WEIGHTS_GB GB a node)"
pids=(); started=
for r in 0 1 2 3; do
  if [ "${HAVE_W[$r]}" = yes ]; then ok "rank $r has ${HOMES[$r]}/$INKLING_REPO_REL/rank$r"; continue; fi
  # In the image (it has the hf client), as the node's user, under a memory limit: on a Spark the page cache of a
  # large download competes with everything else for the unified memory.
  ( on $r "docker run --rm --network host --memory 6g --memory-swap 6g --user \$(id -u):\$(id -g) -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -e HF_XET_HIGH_PERFORMANCE=0 -v ${HOMES[$r]}/$INKLING_REPO_REL:/w --entrypoint hf $INKLING_IMAGE download $WEIGHTS_REPO --include 'rank$r/*' --local-dir /w" > "/tmp/inkling2-download-rank$r.log" 2>&1 \
      || { echo "   FAIL  rank $r: the download stopped (log: /tmp/inkling2-download-rank$r.log on this node). Run the script again: it resumes."; exit 1; } ) & pids+=($!); started=1
  echo "   ...   rank $r downloading to ${HOMES[$r]}/$INKLING_REPO_REL/rank$r"
done
if [ -n "$started" ]; then
  while :; do
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null && alive=1; done; [ $alive = 1 ] || break
    sleep 60
    line=""; for r in 0 1 2 3; do line="$line rank$r $(on $r "du -sBG ${HOMES[$r]}/$INKLING_REPO_REL/rank$r 2>/dev/null | cut -f1" 2> /dev/null || echo ?)"; done
    echo "   ...  $line of ${WEIGHTS_GB}G each"
  done
  for p in "${pids[@]}"; do wait "$p" || FAILED=1; done
  [ -z "$FAILED" ] || die "a download did not finish. Run the script again: finished files are kept."
fi
for r in 0 1 2 3; do
  out=$(on $r "ls ${HOMES[$r]}/$INKLING_REPO_REL/rank$r/*.safetensors 2>/dev/null | wc -l; du -sb ${HOMES[$r]}/$INKLING_REPO_REL/rank$r 2>/dev/null | cut -f1; test -f ${HOMES[$r]}/$INKLING_REPO_REL/rank$r/model.safetensors.index.json && echo index")
  mapfile -t L <<< "$out"
  [ "${L[0]}" = $SHARDS ] && [ "${L[1]:-0}" -ge $MIN_BYTES ] 2>/dev/null && [ "${L[2]}" = index ] \
    || bad "rank $r: ${HOMES[$r]}/$INKLING_REPO_REL/rank$r is incomplete (${L[0]} of $SHARDS weight files, ${L[1]:-0} bytes)"
done
[ -z "$FAILED" ] || die "the weights are incomplete. Run the script again: the download resumes."
ok "all four folders are complete"
[ $MODE = nolaunch ] && { echo; echo "Everything is in place. Launch with ./quickstart.sh or launch/relaunch.sh."; exit 0; }

# ---- 6. the launch ---------------------------------------------------------------------------------------------
say "6/7 launching (about 5 minutes; about 17 the first time, while kernels compile)"
bash "$BASE_DIR/launch/launch-inkling2.sh" --dry-run > /dev/null 2> /tmp/inkling2-dryrun.err \
  || die "the launcher refuses this site file: $(tail -2 /tmp/inkling2-dryrun.err | tr '\n' ' ')"
bash "$RECIPE_DIR/launch/relaunch.sh" || die "the launch failed. The lines above say where; 'docker logs inkling2' on each node has the details
         (the first lines with 'Error' or 'ABORT'). Fix it and run the script again."

# ---- 7. the proof ----------------------------------------------------------------------------------------------
say "7/7 checking the running fleet"
for r in 0 1 2 3; do
  on $r "docker logs inkling2 2>&1 | grep -q '\[nvfp4_kv\] target KV pool: NVFP4'" && ok "rank $r runs the NVFP4 KV cache" || bad "rank $r: no NVFP4 KV cache in its log (docker logs inkling2 | grep nvfp4_kv)"
done
pool=$(docker logs inkling2 2>&1 | grep -o "full size: [0-9]*" | head -1); [ -n "$pool" ] && ok "KV pool, $pool tokens"
answer=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":400}' \
  | python3 -c "import json,sys; print((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip()[:80])" 2> /dev/null)
[ -n "$answer" ] && ok "a chat request answered: $answer" || bad "a chat request got no answer (curl http://127.0.0.1:$PORT/health; docker logs inkling2)"
[ -z "$FAILED" ] || die "the fleet started but a check failed (lines marked FAIL)."
cat <<EOT

READY. OpenAI-compatible endpoint:  http://${NODES[0]}:$PORT/v1   (model name: inkling-2-small)

  try it     curl -s http://${NODES[0]}:$PORT/v1/chat/completions -H 'content-type: application/json' \\
               -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Is 91 prime?"}],"max_tokens":400}'
  status     ./quickstart.sh --status
  stop       ./quickstart.sh --stop
  restart    launch/relaunch.sh          (the watchdog does this by itself when the fleet stops answering)
  FP8 cache  KV_NVFP4= launch/relaunch.sh
EOT
