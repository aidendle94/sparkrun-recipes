#!/bin/bash
# launch-sgl-dsv41.sh — DeepSeek-V4.1-Flash with SGLang (TP4/EP4) on four DGX Sparks over a switched RoCE fabric.
# One container per node, host networking, RDMA passthrough. Image: the Dockerfile in this repo on lmsysorg/sglang:dev-dsv41
# (published as aidendle94/sparkrun-sglang-dsv41-gb10:production-1.5; production.sh selects it). Engram rows come from
# node-local NVMe (tools/engram_local.py) or, on the node that holds the checkpoint, straight from the shards.
#   ./launch-sgl-dsv41.sh [--dry-run|--stop]
# Site facts — nodes, users, home directories, network devices, model paths — come from launch/fleet.env: copy
# fleet.env.example and fill it in. The copy is git-ignored and the launcher refuses to run without it.
# env: PORT (8888) IMAGE (sglang-dsv41-spark:local = a local build; production.sh uses the published tag) DIST_PORT (20100) CHUNK (2048) MAXREQ (16) KVTOK (0 = let SGLang size it) CTX (524288) MEMFRAC (0.80) SPEC_K (5)
#      NCCL_TRIM (1: 1 MiB buffers, no LL128, 8 channels) MXFP8 (b12x|auto|cutlass) SERVED (deepseek-v4.1-flash)
#      ROCE_AR (1|0: b12x one-shot RDMA collectives for the TP group; ROCE_AR_MAX 1MB, ROCE_AG_MAX 16MB, ROCE_HCA, ROCE_SPIN)
#      SERVED_ALIASES (comma-separated extra model ids listed by /v1/models; SGLang serves one name)
#      THINKING_DEFAULT (1|0: thinking on unless a request passes "thinking": false; effort max via SGLANG_DSV41_REASONING_EFFORT)
#      ENGRAM_EARLY (1|0: start the Engram host gathers on a side stream as soon as the hash ids exist)
#      ENGRAM_EARLY_VERIFY (0|1: on eager steps also gather inline and compare byte for byte; diagnosis only)
#      PAGE_CACHE_RELEASE (1|0: drop the checkpoint's page cache after the KV pool is allocated; see overlay/page_cache_release.py)
#      LATE_TAIL_SKIP (0|1: run the late layers only over the prompt positions decode reads) LATE_TAIL_TIMERS (0|1: time them)
#      LATE_TAIL_SKIP_ALL (0|1: with LATE_TAIL_SKIP, skip the late layers entirely on chunks no request needs them for)
#      WO_A_W8A16 (0|1: attention output projection wo_a with exact FP8 weights, BF16 activations; see overlay/wo_a_w8a16.py)
#      SM120_REAL_HEADS (0|1: decode-sized attention on the rank's 16 real heads instead of padded to 64)
#      STEP_TIMERS (0|1: per-graph decode-step timing in the log, for diagnosis windows; see overlay/step_timers.py)
#      STEP_TIMERS_ATTN (0|1: with STEP_TIMERS=1, also split attention into projections, compressor, indexer, kernel, output)
#      ENGRAM_MODE (hostnode|staged: Engram rows gathered inside the graph by the C store, or staged before each forward
#      by the pure-Python port; see docs/design.md) ENGRAM_PREFETCH (1|0: scheduler-side next-chunk row prefetch) EXTRA_ARGS (appended to sglang.launch_server)
#      NCCL_EXTRA (extra docker -e flags) FLEET_ENV (path of the fleet file)
# Container flags assume a 128 GB Spark: --memory 112g --memory-swap 112g --shm-size 32g --oom-score-adj 500 (the container is
# the first thing the kernel OOM killer takes). Every rank runs the same image; rank 0 is where this script runs.
# MIT License, Copyright (c) 2026 Aiden Le.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
for v in NODES USERS HOMES; do eval "n=\${#$v[@]}"; [ "$n" = 4 ] || { echo "ABORT: $v in $FLEET_ENV must list exactly 4 ranks" >&2; exit 1; }; done
for v in FABRIC_IFACE RDMA_HCAS FABRIC_SUBNET MODEL_REPO SNAP ENGRAM_LOCAL_DIR; do eval "x=\${$v:-}"; [ -n "$x" ] || { echo "ABORT: $v is not set in $FLEET_ENV" >&2; exit 1; }; done
MODEL_HOST_RANK="${MODEL_HOST_RANK:-}"; MODEL_REPO_HOST="${MODEL_REPO_HOST:-$MODEL_REPO}"; ENGRAM_ROWS_HOST="${ENGRAM_ROWS_HOST:-}"; CACHE_DIR="${CACHE_DIR:-sgl-dsv41-cache}"
[ -n "$MODEL_HOST_RANK" ] && [ -z "$ENGRAM_ROWS_HOST" ] && { echo "ABORT: ENGRAM_ROWS_HOST (LAYER:START:END,...) is required when MODEL_HOST_RANK is set" >&2; exit 1; }
HEAD=${NODES[0]}; PORT=${PORT:-8888}; DIST_PORT=${DIST_PORT:-20100}; NAME=sgldsv41
ip -o addr 2>/dev/null | grep -q " $HEAD/" || { echo "ABORT: this node does not carry the rank-0 fabric address $HEAD (NODES[0] in $FLEET_ENV); run the launcher on rank 0" >&2; exit 1; }
IMAGE="${IMAGE:-${PRODUCTION_IMAGE:-sglang-dsv41-spark:local}}"   # env > fleet.env > production.sh > local build
CHUNK="${CHUNK:-2048}"; MAXREQ="${MAXREQ:-16}"; KVTOK="${KVTOK:-0}"; CTX="${CTX:-524288}"; MEMFRAC="${MEMFRAC:-0.80}"; SPEC_K="${SPEC_K:-5}"
NCCL_TRIM="${NCCL_TRIM:-1}"; MXFP8="${MXFP8:-b12x}"; SERVED="${SERVED:-deepseek-v4.1-flash}"; SERVED_ALIASES="${SERVED_ALIASES:-}"; THINKING_DEFAULT="${THINKING_DEFAULT:-1}"   # aliases come from fleet.env or the environment
ROCE_AR="${ROCE_AR:-1}"; ROCE_AR_MAX="${ROCE_AR_MAX:-1MB}"; ROCE_AG_MAX="${ROCE_AG_MAX:-16MB}"; ROCE_HCA="${ROCE_HCA:-$RDMA_HCAS}"; ROCE_SPIN="${ROCE_SPIN:-300000000}"
# Engram row partition: every rank reads one contiguous range of each table — from its node-local sparse copy (the
# manifest written by tools/engram_local.py carries the range), or on MODEL_HOST_RANK from the shards (ENGRAM_ROWS_HOST).
# The four ranges must partition each table; SGLang's own row formula is not used.
abs() { case "$2" in /*) printf '%s' "$2";; *) printf '%s/%s' "${HOMES[$1]}" "$2";; esac; }  # rank, path -> absolute on that rank
DRY=0; STOP=0; for a in "$@"; do case $a in --dry-run) DRY=1;; --stop) STOP=1;; esac; done
on_rank() { local r=$1; shift; if [ $r = 0 ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=20 ${USERS[$r]}@${NODES[$r]} "$*"; fi; }
# Every remote step (GID probe, preflight, launch) goes over ssh; say so plainly when a node cannot be reached.
reach_all() { local r err; for r in 1 2 3; do if ! err=$(ssh -o BatchMode=yes -o ConnectTimeout=10 ${USERS[$r]}@${NODES[$r]} true 2>&1); then
  echo "ABORT: rank $r: cannot ssh to ${USERS[$r]}@${NODES[$r]}: ${err:-no error text} (check NODES/USERS in $FLEET_ENV and key-based ssh from this node)" >&2; return 1; fi; done; }
# RoCE-v2 GID index of this rank's fabric IPv4 on the first HCA. It differs between nodes (3 on some, 4 on others in
# the same fleet) and a wrong index fails every RDMA connect, so it is probed on the node, never assumed.
gid_snippet() {
  local a b c d hex hca=${RDMA_HCAS%%,*}; IFS=. read -r a b c d <<< "${NODES[$1]}"; hex=$(printf 'ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d")
  printf '%s' "for i in \$(seq 0 15); do g=\$(cat /sys/class/infiniband/$hca/ports/1/gids/\$i 2>/dev/null) || break; t=\$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/\$i 2>/dev/null); case \"\$g\" in *:$hex) [ \"\$t\" = \"RoCE v2\" ] && { echo \$i; break; };; esac; done"
}
if [ $STOP = 1 ]; then
  wp=$(grep -o "watchdog started (pid [0-9]*" $HOME/sgl-watchdog.log 2>/dev/null | tail -1 | grep -o "[0-9]*$")
  [ -n "$wp" ] && ps -p "$wp" >/dev/null 2>&1 && echo "NOTE: the fleet watchdog (pid $wp) is armed and will relaunch production within minutes; kill it first, or use launch/relaunch.sh" >&2
  reach_all || exit 1; for r in 0 1 2 3; do on_rank $r "docker rm -f $NAME >/dev/null 2>&1 && echo '  stopped rank $r' || echo '  rank $r: nothing to stop'"; done; exit 0
fi

if [ "$NCCL_TRIM" = "1" ]; then NCCL_BUF="-e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8"; else NCCL_BUF="-e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_ALGO=Ring -e NCCL_PROTO=LL,LL128,Simple"; fi
[ "$KVTOK" != "0" ] && KV_ARGS="--max-total-tokens $KVTOK" || KV_ARGS=""
[ "$THINKING_DEFAULT" = "1" ] && THINK_ARGS="--default-chat-template-kwargs '{\"thinking\":true}'" || THINK_ARGS=""

run_cmd() {  # rank -> the docker run command (single-quoted JSON survives the remote bash -c)
  local r=$1 repo gid engram cache; cache=$(abs $r "$CACHE_DIR")
  gid=$(on_rank $r "$(gid_snippet $r)" 2>/dev/null | tail -1); case "$gid" in [0-9]|[0-9][0-9]) ;; *) echo "ABORT: rank $r: no RoCE v2 GID for ${NODES[$r]} on ${RDMA_HCAS%%,*}" >&2; return 1;; esac
  if [ "$r" = "$MODEL_HOST_RANK" ]; then repo=$MODEL_REPO_HOST; engram="-e SPARK_ENGRAM_DIR=/models/repo/$SNAP -e SPARK_ENGRAM_ROWS=$ENGRAM_ROWS_HOST"
  else repo=$MODEL_REPO; engram="-v $(abs $r "$ENGRAM_LOCAL_DIR"):/engram-local:ro -e SPARK_ENGRAM_DIR=/engram-local"; fi
  printf '%s' "mkdir -p $cache; docker run -d --name $NAME --restart no --network host --ipc host --cap-add IPC_LOCK --gpus all \
 --shm-size 32g --memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --ulimit stack=67108864 --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
 -v $repo:/models/repo:ro -v $cache:/root/.cache $engram \
 -e SPARK_ENGRAM_EARLY=${ENGRAM_EARLY:-1} -e SPARK_ENGRAM_EARLY_VERIFY=${ENGRAM_EARLY_VERIFY:-0} -e SPARK_PAGE_CACHE_RELEASE=${PAGE_CACHE_RELEASE:-1} -e SPARK_LATE_TAIL_SKIP=${LATE_TAIL_SKIP:-0} -e SPARK_LATE_TAIL_SKIP_ALL=${LATE_TAIL_SKIP_ALL:-0} -e SPARK_WO_A_W8A16=${WO_A_W8A16:-0} -e SPARK_LATE_TAIL_TIMERS=${LATE_TAIL_TIMERS:-0} -e SPARK_SM120_REAL_HEADS=${SM120_REAL_HEADS:-0} -e SPARK_STEP_TIMERS=${STEP_TIMERS:-0} -e SPARK_STEP_TIMERS_ATTN=${STEP_TIMERS_ATTN:-0} -e SPARK_ENGRAM_MODE=${ENGRAM_MODE:-hostnode} -e SPARK_ENGRAM_PREFETCH=${ENGRAM_PREFETCH:-1} -e SPARK_ENGRAM_THREADS=64 -e SPARK_ENGRAM_MAX_IDS=$(( (CHUNK > MAXREQ * 8 ? CHUNK : MAXREQ * 8) * 32 )) -e SPARK_MXFP8_BACKEND=$MXFP8 \
 -e SPARK_SERVED_ALIASES=$SERVED_ALIASES -e SPARK_ROCE_AR=$ROCE_AR -e SPARK_ROCE_AR_MAX=$ROCE_AR_MAX -e SPARK_ROCE_AG_MAX=$ROCE_AG_MAX -e B12X_ROCE_HCA=$ROCE_HCA -e B12X_ROCE_GID_INDEX=$gid -e B12X_ROCE_SPIN_LIMIT=$ROCE_SPIN \
 -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0 -e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 -e SGLANG_DSV41_REASONING_EFFORT=100 \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOST_IP=${NODES[$r]} \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$RDMA_HCAS -e NCCL_SOCKET_IFNAME=$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE \
 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=$FABRIC_SUBNET -e NCCL_IB_SUBNET_PREFIX_LEN=${FABRIC_SUBNET#*/} -e NCCL_IB_GID_INDEX=$gid \
 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=WARN $NCCL_BUF ${NCCL_EXTRA:-} \
 --entrypoint python3 $IMAGE -m sglang.launch_server --model-path /models/repo/$SNAP --served-model-name $SERVED --trust-remote-code --load-format safetensors \
 --tp 4 --ep-size 4 --nnodes 4 --node-rank $r --dist-init-addr $HEAD:$DIST_PORT --host 0.0.0.0 --port $PORT \
 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --fp8-gemm-backend flashinfer_cutlass \
 --mem-fraction-static $MEMFRAC --chunked-prefill-size $CHUNK --context-length $CTX --max-running-requests $MAXREQ --cuda-graph-max-bs-decode $MAXREQ $KV_ARGS \
 --random-seed 0 --enable-decoder-swa-bounded-replay --speculative-algorithm DSPARK --speculative-dspark-block-size $SPEC_K \
 --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 --watchdog-timeout 1800 --min-free-slots-delay 1 $THINK_ARGS ${EXTRA_ARGS:-}"
}
# --dry-run also probes each node's GID index, so it needs ssh to every rank.
reach_all || exit 1
if [ $DRY = 1 ]; then for r in 0 1 2 3; do echo "== rank $r"; run_cmd $r || exit 1; echo; done; exit 0; fi
echo "== preflight"; bad=""
for r in 0 1 2 3; do
  if [ "$r" = "$MODEL_HOST_RANK" ]; then what="checkpoint index $MODEL_REPO_HOST/$SNAP/model.safetensors.index.json"; chk="test -f $MODEL_REPO_HOST/$SNAP/model.safetensors.index.json"
  else what="engram rows $(abs $r "$ENGRAM_LOCAL_DIR")/engram-local.json"; chk="test -f $(abs $r "$ENGRAM_LOCAL_DIR")/engram-local.json"; fi
  # The overlay is inert unless the image reads the env names this launcher passes (SPARK_*); an older image would
  # silently run stock SGLang, which loads the 100 GB Engram tables into memory and takes the nodes down.
  if out=$(on_rank $r "docker image inspect $IMAGE >/dev/null 2>&1 && echo image-ok || echo image-MISSING; docker run --rm --entrypoint grep $IMAGE -q SPARK_ENGRAM_DIR /opt/dsv41-spark/overlay/sitecustomize.py 2>/dev/null && echo overlay-ok || echo overlay-MISMATCH; $chk && echo rows-ok || echo rows-MISSING" 2>&1); then
    case "$out" in *MISSING*|*MISMATCH*) echo "  rank $r: $(echo $out) — image $IMAGE (docker pull it on that node; overlay-MISMATCH = the image's overlay does not read SPARK_* and would run stock SGLang) / $what"; bad=1;; *) echo "  rank $r: image + overlay + engram rows OK";; esac
  else echo "  rank $r: unreachable (${USERS[$r]}@${NODES[$r]}): $out"; bad=1; fi
done
[ -n "$bad" ] && { echo "ABORT preflight (nothing was stopped or started)"; exit 1; }
for r in 0 1 2 3; do on_rank $r "docker rm -f $NAME >/dev/null 2>&1"; done; sleep 5
# GPU must be free on every node (a rank that starts while the previous engine still holds the device dies with
# "CUDA-capable device(s) is/are busy or unavailable" and the other ranks hang in torch.distributed init).
for r in 0 1 2 3; do for try in 1 2 3 4 5 6; do
  busy=$(on_rank $r "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; nvidia-smi --query-gpu=memory.used --format=csv,noheader >/dev/null 2>&1 || echo nvsmi-fail" | tr '\n' ' ')
  case "$busy" in "0 ") echo "  rank $r: GPU free"; break;; *) echo "  rank $r: GPU busy ($busy), waiting"; sleep 10;; esac
  [ $try = 6 ] && { echo "ABORT: rank $r GPU still busy"; exit 1; }
done; done
for r in 3 2 1 0; do cmd=$(run_cmd $r) || exit 1; on_rank $r "$cmd" >/dev/null && echo "  rank $r launched" || { echo "rank $r launch FAILED"; exit 1; }; done
echo "== launched: head API http://$HEAD:$PORT (health: /health); follow: docker logs -f $NAME"
