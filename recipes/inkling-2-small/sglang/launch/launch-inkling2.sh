#!/bin/bash
# launch-inkling2.sh — Inkling-2-Small (thinkingmachines, NVFP4 checkpoint) with SGLang (TP4/EP4) on the four DGX Sparks
# of launch/fleet.env (copy fleet.env.example: nodes, users, homes, fabric). Image lmsysorg/sglang:dev-inkling-2-small
# plus the GB10 hooks in patches/sitecustomize.py (copied to every node, put on PYTHONPATH).
#   ./launch-inkling2.sh [--dry-run|--stop]
# Nothing else may hold the GPUs: the model needs nearly all memory of every node. OTHER_CONTAINER in fleet.env names
# another engine's container; while it runs on any node the preflight aborts instead of starting next to it.
# env: PORT (8210) DIST_PORT (20200) INKLING_IMAGE (the image; pin the digest of the README when the tag moves) INKLING_REPO_REL INKLING_SNAP (where the checkpoint is under each home, and its revision) CTX (262144) CHUNK (4096) MAXREQ (8; needs PAGED_DECODE=1 on GB10: without it two requests in a decode batch generate garbage, see README) MEMFRAC (0.93) SERVED (inkling-2-small)
#      ROCE_AR (1|0: b12x one-shot RDMA collectives for the TP group, ~8% faster decode steps)
#      FI_MOE (1|0: NVFP4 experts on FlashInfer NVFP4 grouped GEMMs, patches/fi_moe.py; prefill 442 -> 1156 tok/s) ROCE_AR_MAX (1MB) ROCE_SPIN (300000000)
#      PAGED_DECODE (1|0; 0 is SGLang's kernel, wrong at page size 1: decode/verify attention with a page lookup per key, patches/paged_decode.py)
#      KV_DTYPE (fp8_e4m3|auto: e4m3 KV pool read through patches/fp8_kv.py + the e4m3 decode kernel, 1.9x the tokens at the same speed; auto = BF16)
#      GATE (0.4 | empty = off: confidence-gated verify rows, patches/spark_gate.py: a draft row is verified only while the draft is confident; ~1.15x per stream, greedy output moves only at near-ties, by less than a second request in the batch moves it, see README)
#      DENSE_I8 (draft,dhead,dfree | empty = off: the draft reads int8 copies of its own weights and frees their BF16 copies, patches/dense_i8.py; ~1.05x per stream on top of the gate, the target computes what it computes without it (held-out NLL identical), ~5% of the KV pool, see README)
#      DENSE_P12 (attn,head,free, the default since 2026-11-15 | empty = off, the rollback: the target's attention projections and unembedding packed into 12 bits a weight without rounding, read by a Triton kernel on forwards of up to 64 rows, BF16 copies of attn freed, patches/dense_p12.py; the same weights, a different order of the sum; see README)
#      TP_SHARD (half,fp8, the default since 2026-11-15 | 1 | empty = off, the rollback (TP_SHARD= DENSE_P12= is the stack before it); needs FI_MOE=1: every rank holds all 384 experts at a quarter of their width instead of 96 whole ones, so no rank waits for the busiest, patches/tp_shard.py; `half` sums the shards' partial rows as FP16, `fp8` shards the MXFP8 layers too; with DENSE_P12 -18% step time for one stream, +20% / +14% at four / eight; not bit-identical to expert parallelism (rounding), held-out NLL unchanged)
#      REASONING_CACHE (1 | empty = off | an absolute path = on while that file exists in the head's container: the API server puts an answer's reasoning back into history turns that arrive without it, patches/reasoning_cache.py; without it a client that drops reasoning gets stale answers over long contexts)
#      SERVED_ALIASES (empty: extra ids listed by /v1/models, comma-separated) MTP (1|0: DSpark with the bundled draft, ~1.7x decode) EXTRA_ARGS
# MIT License, Copyright (c) 2026 Aiden Le.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
REPO_REL=${INKLING_REPO_REL:-.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Small}; SNAP=${INKLING_SNAP:-snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b}
HEAD=${NODES[0]}; PORT=${PORT:-8210}; DIST_PORT=${DIST_PORT:-20200}; NAME=inkling2
IMAGE=${INKLING_IMAGE:-lmsysorg/sglang:dev-inkling-2-small}; CTX=${CTX:-262144}; CHUNK=${CHUNK:-4096}; MAXREQ=${MAXREQ:-8}
MEMFRAC=${MEMFRAC:-0.93}; SERVED=${SERVED:-inkling-2-small}; MTP=${MTP:-1}; KV_DTYPE=${KV_DTYPE:-fp8_e4m3}; GATE=${GATE-0.4}; DENSE_I8=${DENSE_I8-draft,dhead,dfree}; DENSE_P12=${DENSE_P12-attn,head,free}; REASONING_CACHE=${REASONING_CACHE-1}; TP_SHARD=${TP_SHARD-half,fp8}
ROCE_AR=${ROCE_AR:-1}; FI_MOE=${FI_MOE:-1}; PAGED_DECODE=${PAGED_DECODE:-1}; ROCE_AR_MAX=${ROCE_AR_MAX:-1MB}; ROCE_SPIN=${ROCE_SPIN:-300000000}
SHARD_ENV=""; if [ -n "$TP_SHARD" ]; then
  [ "$FI_MOE" = 1 ] || { echo "ABORT: TP_SHARD needs FI_MOE=1" >&2; exit 1; }
  SHARD_ENV="-e SPARK_TP_SHARD=1"; case ",$TP_SHARD," in *,half,*) SHARD_ENV="$SHARD_ENV -e SPARK_TP_SHARD_HALF=1";; esac; case ",$TP_SHARD," in *,fp8,*) SHARD_ENV="$SHARD_ENV -e SPARK_TP_SHARD_FP8=1";; esac
fi
DRY=0; STOP=0; for a in "$@"; do case $a in --dry-run) DRY=1;; --stop) STOP=1;; esac; done
on_rank() { local r=$1; shift; if [ $r = 0 ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=20 ${USERS[$r]}@${NODES[$r]} "$*"; fi; }
gid_snippet() {  # RoCE-v2 GID index of the rank's fabric IPv4 on the first HCA (differs between nodes)
  local a b c d hex hca=${RDMA_HCAS%%,*}; IFS=. read -r a b c d <<< "${NODES[$1]}"; hex=$(printf 'ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d")
  printf '%s' "for i in \$(seq 0 15); do g=\$(cat /sys/class/infiniband/$hca/ports/1/gids/\$i 2>/dev/null) || break; t=\$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/\$i 2>/dev/null); case \"\$g\" in *:$hex) [ \"\$t\" = \"RoCE v2\" ] && { echo \$i; break; };; esac; done"
}
if [ $STOP = 1 ]; then for r in 0 1 2 3; do on_rank $r "docker rm -f $NAME >/dev/null 2>&1 && echo '  stopped rank $r' || echo '  rank $r: nothing to stop'"; done; exit 0; fi
[ "$MTP" = 1 ] && SPEC_ARGS="--speculative-algorithm DSPARK" || SPEC_ARGS=""

run_cmd() {
  local r=$1 gid repo=${HOMES[$r]}/$REPO_REL patches=${HOMES[$r]}/inkling2-patches
  gid=$(on_rank $r "$(gid_snippet $r)" 2>/dev/null | tail -1); case "$gid" in [0-9]|[0-9][0-9]) ;; *) echo "ABORT: rank $r: no RoCE v2 GID" >&2; return 1;; esac
  printf '%s' "docker run -d --name $NAME --restart no --network host --ipc host --cap-add IPC_LOCK --gpus all \
 --shm-size 32g --memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --ulimit stack=67108864 --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
 -v $repo:/models/repo:ro -v $patches:/patches:ro -v ${HOMES[$r]}/inkling2-cache:/root/.cache -e PYTHONPATH=/patches -e SPARK_SERVED_ALIASES=${SERVED_ALIASES:-} \
 -e SPARK_ROCE_AR=$ROCE_AR -e SPARK_FI_MOE=$FI_MOE -e SPARK_PAGED_DECODE=$PAGED_DECODE -e SPARK_ROCE_AR_MAX=$ROCE_AR_MAX -e B12X_ROCE_HCA=$RDMA_HCAS -e B12X_ROCE_GID_INDEX=$gid -e B12X_ROCE_SPIN_LIMIT=$ROCE_SPIN \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOST_IP=${NODES[$r]} \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$RDMA_HCAS -e NCCL_SOCKET_IFNAME=$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE \
 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=$FABRIC_SUBNET -e NCCL_IB_SUBNET_PREFIX_LEN=${FABRIC_SUBNET#*/} -e NCCL_IB_GID_INDEX=$gid \
 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=WARN \
 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8 ${GATE:+-e SPARK_GATE=$GATE} ${DENSE_I8:+-e SPARK_DENSE_I8=$DENSE_I8} ${DENSE_P12:+-e SPARK_DENSE_P12=$DENSE_P12} $SHARD_ENV ${NCCL_EXTRA:-} ${REASONING_CACHE:+-e SPARK_REASONING_CACHE=$REASONING_CACHE} \
 --entrypoint python3 $IMAGE -m sglang.launch_server --model-path /models/repo/$SNAP --served-model-name $SERVED --trust-remote-code \
 --tp 4 --ep-size 4 --nnodes 4 --node-rank $r --dist-init-addr $HEAD:$DIST_PORT --host 0.0.0.0 --port $PORT \
 --attention-backend fa4 --mem-fraction-static $MEMFRAC --chunked-prefill-size $CHUNK --context-length $CTX \
 --max-running-requests $MAXREQ --cuda-graph-max-bs-decode $MAXREQ --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-prefill $MAXREQ \
 --watchdog-timeout 1800 --enable-metrics --kv-cache-dtype $KV_DTYPE $SPEC_ARGS ${EXTRA_ARGS:-}"
}
if [ $DRY = 1 ]; then for r in 0 1 2 3; do echo "== rank $r"; run_cmd $r || exit 1; echo; done; exit 0; fi
echo "== preflight"; bad=""
for r in 0 1 2 3; do
  # The GB10 hooks must be on every node (a rank without them runs stock code and fails at the first collective).
  if [ $r = 0 ]; then mkdir -p ${HOMES[0]}/inkling2-patches && cp $HERE/../patches/*.py ${HOMES[0]}/inkling2-patches/
  else on_rank $r "mkdir -p ${HOMES[$r]}/inkling2-patches" && scp -q -o BatchMode=yes $HERE/../patches/*.py ${USERS[$r]}@${NODES[$r]}:${HOMES[$r]}/inkling2-patches/; fi
  out=$(on_rank $r "mkdir -p ${HOMES[$r]}/inkling2-cache; docker image inspect $IMAGE >/dev/null 2>&1 && echo image-ok || echo image-MISSING; test -f ${HOMES[$r]}/$REPO_REL/$SNAP/model.safetensors.index.json && echo weights-ok || echo weights-MISSING; ${OTHER_CONTAINER:+docker ps -q -f name=^$OTHER_CONTAINER\$ | grep -q . && echo other-engine-RUNNING || echo other-engine-off}" 2>&1)
  case "$out" in *MISSING*|*RUNNING*) echo "  rank $r: $(echo $out)"; bad=1;; *) echo "  rank $r: image + weights OK";; esac
done
[ -n "$bad" ] && { echo "ABORT preflight (nothing started)"; exit 1; }
for r in 0 1 2 3; do on_rank $r "docker rm -f $NAME >/dev/null 2>&1"; done
for r in 0 1 2 3; do for try in 1 2 3 4 5 6; do
  busy=$(on_rank $r "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l")
  [ "$busy" = 0 ] && break; echo "  rank $r: GPU busy, waiting"; sleep 10; [ $try = 6 ] && { echo "ABORT: rank $r GPU still busy"; exit 1; }
done; done
for r in 3 2 1 0; do cmd=$(run_cmd $r) || exit 1; on_rank $r "$cmd" >/dev/null && echo "  rank $r launched" || { echo "rank $r launch FAILED"; exit 1; }; done
echo "== launched: http://$HEAD:$PORT (health /health); docker logs -f $NAME"
