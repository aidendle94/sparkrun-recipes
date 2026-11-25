#!/bin/bash
# launch-inkling2.sh — Inkling-2-Small (thinkingmachines, NVFP4 checkpoint) with SGLang (TP4/EP4) on the four DGX Sparks
# of launch/fleet.env (copy fleet.env.example: nodes, users, homes, fabric). Image lmsysorg/sglang:dev-inkling-2-small
# plus the GB10 hooks in patches/sitecustomize.py (copied to every node, put on PYTHONPATH).
#   ./launch-inkling2.sh [--dry-run|--stop]
# Nothing else may hold the GPUs: the model needs nearly all memory of every node. OTHER_CONTAINER in fleet.env names
# another engine's container; while it runs on any node the preflight aborts instead of starting next to it.
# env: PORT (8210) DIST_PORT (20200) INKLING_IMAGE (the image; pin the digest of the README when the tag moves) INKLING_REPO_REL INKLING_SNAP (where the checkpoint is under each home, and its revision) CTX (524288; 262144 until 2026-11-17) CHUNK (4096) MAXREQ (8; needs PAGED_DECODE=1 on GB10: without it two requests in a decode batch generate garbage, see README) MEMFRAC (0.94; 0.93 until 2026-11-17) SERVED (inkling-2-small)
#      ROCE_AR (1|0: b12x one-shot RDMA collectives for the TP group, ~8% faster decode steps)
#      FI_MOE (1|0: NVFP4 experts on FlashInfer NVFP4 grouped GEMMs, patches/fi_moe.py; prefill 442 -> 1156 tok/s) ROCE_AR_MAX (1MB) ROCE_SPIN (300000000)
#      PAGED_DECODE (1|0; 0 is SGLang's kernel, wrong at page size 1: decode/verify attention with a page lookup per key, patches/paged_decode.py)
#      KV_DTYPE (fp8_e4m3|auto: e4m3 KV pool read through patches/fp8_kv.py + the e4m3 decode kernel, 1.9x the tokens at the same speed; auto = BF16)
#      GATE (0.4 | empty = off: confidence-gated verify rows, patches/spark_gate.py: a draft row is verified only while the draft is confident; ~1.15x per stream, greedy output moves only at near-ties, by less than a second request in the batch moves it, see README)
#      DENSE_I8 (draft,dhead,dfree | empty = off: the draft reads int8 copies of its own weights and frees their BF16 copies, patches/dense_i8.py; ~1.05x per stream on top of the gate, the target computes what it computes without it (held-out NLL identical), ~5% of the KV pool, see README)
#      DENSE_P12 (attn,head,free, the default since 2026-11-15 | empty = off, the rollback: the target's attention projections and unembedding packed into 12 bits a weight without rounding, read by a Triton kernel on forwards of up to 64 rows, BF16 copies of attn freed, patches/dense_p12.py; the same weights, a different order of the sum; see README)
#      TP_SHARD (half,fp8, the default since 2026-11-15 | 1 | empty = off, the rollback (TP_SHARD= DENSE_P12= is the stack before it); needs FI_MOE=1: every rank holds all 384 experts at a quarter of their width instead of 96 whole ones, so no rank waits for the busiest, patches/tp_shard.py; `half` sums the shards' partial rows as FP16, `fp8` shards the MXFP8 layers too; with DENSE_P12 -18% step time for one stream, +20% / +14% at four / eight; not bit-identical to expert parallelism (rounding), held-out NLL unchanged)
#      REASONING_CACHE (1 | empty = off | an absolute path = on while that file exists in the head's container: the API server puts an answer's reasoning back into history turns that arrive without it, patches/reasoning_cache.py; without it a client that drops reasoning gets stale answers over long contexts)
#      TOOL_PARSE_ERROR (1, the default since 2026-11-25 | empty = off, SGLang's behaviour: a tool call the model wrote with a payload that does not parse fails the chat-completions request, HTTP 500 or an error event in a stream, instead of reaching the client as assistant text (chat completions, Anthropic messages, Responses API), patches/tool_parse_error.py; a call cut off by max_tokens is not affected)
#      TOOL_CHOICE_FIX (parallel,none, the default since 2026-11-25 | empty = off, SGLang's behaviour: parallel = parallel_tool_calls false keeps the answer's first tool call; none = an explicit tool_choice "none" biases the two tool-invocation tokens by -100 so the model answers in text; patches/tool_choice_fix.py)
#      DRAFT_KV_FP8 (1, the default since 2026-11-17 | empty = off: the draft's KV cache in e4m3 like the target's, read by patches/draft_decode.py; 793,000 pool tokens where 716,000 fitted and a faster draft behind long contexts, for about 1% of step time on one short stream)
#      WMARK_BOOST (1, the default since 2026-11-17 | empty = off: the pool is sized without the kernel's temporary watermark boost, patches/wmark_boost.py, so every boot gets the same pool)
#      SWA_RATIO (0.05, the default since 2026-11-17 | empty = the engine's 0.1: --swa-full-tokens-ratio, the windowed layers' share of the pool; 0.05 halves their cache, so fewer cached conversation ends keep their window)
#      POOL (--max-total-tokens | empty = no cap, the engine sizes the pool at boot. Unset, it follows the three settings before it, the same bytes each time, so the rank-0 node keeps the memory it had: 793000 with all three, 716000 with DRAFT_KV_FP8=, 600000 with SWA_RATIO= or 0.1, no cap with anything else)
#      PREFILL_GLUE (epilogue,stream,gate,qkvr,relbias, the default since 2026-11-17 | empty = off: the per-layer glue of a prefill chunk in fewer passes over memory, patches/prefill_glue.py; exact; cold prefill +11% at 30,000 tokens, +10% at 120,000)
#      BIAS_FAR (f2, the default since 2026-11-17 | empty = off: the SM120 attention kernel derived from the image's source, its far key blocks compiled without the bias state, patches/fa4_bias_far.py; exact; 2% of a 120,000-token prefill, 3.6% of a 300,000-token one)
#      FI_MXFP8 (1, the default since 2026-11-17 | empty = off, which keeps the target's arithmetic as it was: the MXFP8 expert layers on FlashInfer's MXFP8 grouped GEMM instead of Marlin, patches/fi_mxfp8.py; does nothing without FI_MOE=1 and fp8 in TP_SHARD; not bit-identical (the order of the sums), held-out NLL unchanged; cold prefill +11% at 30,000 tokens, +10% at 120,000)
#      PREFILL_DECODE_INTERVAL (2, the default since 2026-11-17 | empty = off, the engine's own scheduling: --prefill-decode-interval, the decode rounds the scheduler runs after a prefill batch before it takes the next one, so that requests already decoding keep getting tokens through a long prefill; skipped while nothing is decoding; a new request still waits for the whole prefill; not measured against a boot without it)
#      STABLE_KERNELS (tau,gate_sink, the default since 2026-11-17 | empty = off: two of the image's Triton kernels with the batch's row count and strides as arguments, patches/stable_kernels.py; exact; without it every prefill of a new size compiles and loads a CUDA module while serving, which is where both outages of 2026-11-17 began)
#      KERNEL_LOAD_GUARD (5, the default since 2026-11-17 | empty = off: a Triton kernel load the driver refuses with "operation not permitted" is tried again up to this many times, patches/kernel_load_guard.py)
#      SCALE_PACK (1, the default since 2026-11-19 | empty = off: the NVFP4 expert layers' block scales stay packed in memory, patches/scale_pack.py, and the forked grouped GEMM patches/spark_csf_gemm_sm120.so reads them in the kernel (kernels/csf_gemm/build_csf_gemm.py builds it from the image's sources; without the file the launcher turns the packing off). Exact: same bytes into the GEMM. 3.0-3.4 GB a node back, one stream's step about 1% longer (0.5 to 0.7 ms on three boots), NLL identical. Needs FI_MOE=1 and TP_SHARD. The .so belongs to the image it was built from: rebuild it when INKLING_IMAGE changes)
#      GATE_HEAD (gate-head-mlp-reg.pt, the default since 2026-11-20 | empty = off, the gate's own rule: the gate's row count comes from a small trained confidence head, a file in patches/, computed in the draft graph's tail, patches/spark_gate_head.py. Exact for sampled requests like the gate (it reads the draft's hidden state and distributions, never the target); greedy text identical on 16 of 16 prompts. Measured in one boot against the gate's own rule, both arms carrying the head's cost: x1.02 greedy, x1.05 at temperature 1.0; its tail costs about 0.75 ms a step (most of it the statistics); net, measured across boots, x1.017 greedy at one stream. Every rank must hold the same file: the ranks vote at boot and no rank arms otherwise. Needs GATE; without the file the launcher leaves the gate on its own rule. bench/cc_export_head.py writes a head from a capture)
#      GATE_FUSED (1, the default since 2026-11-24 | empty = off: with GATE_HEAD, the head's draft statistics in one Triton pass over the vocabulary for the whole step, patches/gate_stats.py, instead of eight passes a request; the same five numbers to about 1e-5, the same row counts on every test step; 0.11 ms against 0.34 ms a step at one stream, 0.11 against 1.2 at four; -0.34 ms a step measured at one stream)
#      ROCE_LIVE (1, the default since 2026-11-24 | empty = off: with ROCE_AR and GATE, the verify forward's per-row all-reduces, 48 attention outputs and 48 expert combines a step, send only the rows the gate verifies instead of all 8 a request, patches/roce_live.py, a fork of the b12x one-shot kernel that reads a row table the gate writes; the live rows get the same bits, greedy text identical on 32 of 32 prompts at one stream; -0.78 ms a step at one stream at temperature 1.0, 2 to 6 ms a step at four and eight. It raises ROCE_AR_MAX to 1200KB so that the eight-request expert all-reduce, 1.18 MB before the cut, stays on the RoCE runtime)
#      GATE_THR (empty, off by default | a table '<batch>:<7 thresholds joined by />,...': with GATE_HEAD, the head's survival thresholds by the number of requests in the step, linear between the listed batch sizes, instead of the head file's one-request thresholds. bench/cc_batch_sim.py made a table from the step times of 2026-11-24 (1:0.449/0.35/0.35/0.277/0.277/0.277/0.277,4:0.61/0.405/0.405/0.276/0.276/0.276/0.276,8:0.593/0.385/0.385/0.239/0.239/0.239/0.239) and simulated x1.0015 / x1.0071 / x1.0047 at 1 / 4 / 8 requests; on the fleet that table was x0.960 at four and at eight requests (64 requests each, temperature 1.0), so it is not a default: the simulation prices a request's second row at what a second row of every request costs together, and a row added to a mixed step costs less)
#      PRESHARDED (empty = off, the official checkpoint | 1: every rank loads its own pre-sharded folder, $INKLING_REPO_REL/rank<R> (patches/preshard.py): about 93 GB a node instead of 394, written by an export boot (SPARK_PRESHARD_EXPORT) or downloaded; the folders are tied to 4 ranks and the TP_SHARD / FI_MOE / FI_MXFP8 settings they were written with, and a rank refuses to boot on any difference)
#      KV_NVFP4 (empty = off, the FP8 pools | target,draft | target: the KV cache in NVFP4, patches/nvfp4_kv.py: a head row of 128 values in 72 bytes instead of 128 (E2M1 codes, an E4M3 scale per 16 values, the checkpoint's 4-over-6 rule), read by NVFP4 versions of the decode, draft and prefill readers; lossy (K/V rounded to 4 bits; held-out NLL unchanged on average). Needs KV_DTYPE=fp8_e4m3, and DRAFT_KV_FP8=1 for draft. With POOL unset the cap keeps the bytes of the FP8 pool: 1409000 tokens with target,draft, 1301000 with target. The recipe ../sglang-nvfp4-kv launches with target,draft)
#      IMAGES (1, the default since 2026-11-21 | empty = off, text only: picture input, the server's --enable-multimodal with patches/vision_sm12x.py, which runs the vision encoder's attention on GB10 and caps a picture's long side; about 0.7 GB more GPU memory a node, the pool and the text path unchanged)
#      IMAGE_MAX_EDGE (empty = 2048: the long side a picture is downscaled to before encoding; 4096 is the model card's limit and takes about six times the head's memory for a burst of large pictures; 0 = no cap)
#      SERVED_ALIASES (empty: extra ids listed by /v1/models, comma-separated) MTP (1|0: DSpark with the bundled draft, ~1.7x decode) EXTRA_ARGS
# MIT License, Copyright (c) 2026 Aiden Le.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FLEET_ENV="${FLEET_ENV:-$HERE/fleet.env}"
[ -f "$FLEET_ENV" ] || { echo "ABORT: $FLEET_ENV not found — copy $HERE/fleet.env.example to fleet.env and fill in this site's nodes" >&2; exit 1; }
. "$FLEET_ENV"
REPO_REL=${INKLING_REPO_REL:-.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Small}; SNAP=${INKLING_SNAP:-snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b}
HEAD=${NODES[0]}; PORT=${PORT:-8210}; DIST_PORT=${DIST_PORT:-20200}; NAME=inkling2
IMAGE=${INKLING_IMAGE:-lmsysorg/sglang:dev-inkling-2-small}; CTX=${CTX:-524288}; CHUNK=${CHUNK:-4096}; MAXREQ=${MAXREQ:-8}
MEMFRAC=${MEMFRAC:-0.94}; SERVED=${SERVED:-inkling-2-small}; MTP=${MTP:-1}; KV_DTYPE=${KV_DTYPE:-fp8_e4m3}; GATE=${GATE-0.4}; DENSE_I8=${DENSE_I8-draft,dhead,dfree}; DENSE_P12=${DENSE_P12-attn,head,free}; REASONING_CACHE=${REASONING_CACHE-1}; TOOL_PARSE_ERROR=${TOOL_PARSE_ERROR-1}; TOOL_CHOICE_FIX=${TOOL_CHOICE_FIX-parallel,none}; TP_SHARD=${TP_SHARD-half,fp8}
ROCE_AR=${ROCE_AR:-1}; FI_MOE=${FI_MOE:-1}; PAGED_DECODE=${PAGED_DECODE:-1}; ROCE_AR_MAX=${ROCE_AR_MAX:-1MB}; ROCE_SPIN=${ROCE_SPIN:-300000000}
DRAFT_KV_FP8=${DRAFT_KV_FP8-1}; WMARK_BOOST=${WMARK_BOOST-1}; FI_MXFP8=${FI_MXFP8-1}; SWA_RATIO=${SWA_RATIO-0.05}; PREFILL_GLUE=${PREFILL_GLUE-epilogue,stream,gate,qkvr,relbias}; BIAS_FAR=${BIAS_FAR-f2}; PREFILL_DECODE_INTERVAL=${PREFILL_DECODE_INTERVAL-2}; STABLE_KERNELS=${STABLE_KERNELS-tau,gate_sink}; KERNEL_LOAD_GUARD=${KERNEL_LOAD_GUARD-5}
[ "$DRAFT_KV_FP8" = 1 ] || DRAFT_KV_FP8=; [ "$WMARK_BOOST" = 1 ] || WMARK_BOOST=; [ "$FI_MXFP8" = 1 ] || FI_MXFP8=   # the hooks read "1" and nothing else as on
KV_NVFP4=${KV_NVFP4-}
if [ -n "$KV_NVFP4" ]; then
  case "$KV_NVFP4" in target,draft|draft,target) KV_NVFP4=target,draft;; target) ;; *) echo "ABORT: KV_NVFP4=$KV_NVFP4 (target,draft | target)" >&2; exit 1;; esac
  [ "$KV_DTYPE" = fp8_e4m3 ] || { echo "ABORT: KV_NVFP4 needs KV_DTYPE=fp8_e4m3" >&2; exit 1; }
  [ "$KV_NVFP4" = target,draft ] && [ -z "$DRAFT_KV_FP8" ] && { echo "note: KV_NVFP4 draft needs DRAFT_KV_FP8=1: the draft's pool stays BF16 for this launch" >&2; KV_NVFP4=target; }
  [ -f "$HERE/../patches/nvfp4_kv.py" ] || { echo "ABORT: patches/nvfp4_kv.py is missing" >&2; exit 1; }
fi
if [ -z "${POOL+set}" ]; then case "$KV_DTYPE/$DRAFT_KV_FP8/$SWA_RATIO/$KV_NVFP4" in
  fp8_e4m3/1/0.05/target,draft) POOL=1409000;; fp8_e4m3/1/0.05/target) POOL=1301000;;
  fp8_e4m3/1/0.05/) POOL=793000;; fp8_e4m3//0.05/) POOL=716000;; fp8_e4m3/*//|fp8_e4m3/*/0.1/) POOL=600000;;
  *) POOL=; echo "note: no pool cap is known for KV_DTYPE=$KV_DTYPE DRAFT_KV_FP8=$DRAFT_KV_FP8 SWA_RATIO=$SWA_RATIO KV_NVFP4=$KV_NVFP4: the engine sizes the pool at boot; the caps were measured with FP8 KV at MEMFRAC=0.94 (set POOL, or MEMFRAC=0.93 WMARK_BOOST= for the sizing before them)" >&2;;
esac; fi
SCALE_PACK=${SCALE_PACK-1}; CSF_SO=spark_csf_gemm_sm120.so; [ "$SCALE_PACK" = 1 ] || SCALE_PACK=
if [ -n "$SCALE_PACK" ]; then
  if [ "$FI_MOE" != 1 ] || [ -z "$TP_SHARD" ]; then echo "note: SCALE_PACK needs FI_MOE=1 and TP_SHARD: the packing is off for this launch" >&2; SCALE_PACK=
  elif [ ! -f "$HERE/../patches/$CSF_SO" ]; then echo "note: patches/$CSF_SO is not built (kernels/csf_gemm/build_csf_gemm.py): the packing is off for this launch" >&2; SCALE_PACK=; fi
fi
IMAGES=${IMAGES-1}; IMAGE_MAX_EDGE=${IMAGE_MAX_EDGE-}
[ -n "$IMAGES" ] && [ ! -f "$HERE/../patches/vision_sm12x.py" ] && { echo "note: patches/vision_sm12x.py is missing: picture input is off for this launch" >&2; IMAGES=; }
GATE_HEAD=${GATE_HEAD-gate-head-mlp-reg.pt}
if [ -n "$GATE_HEAD" ]; then
  if [ -z "$GATE" ]; then echo "note: GATE_HEAD needs the gate (GATE): no confidence head for this launch" >&2; GATE_HEAD=
  elif [ ! -f "$HERE/../patches/$GATE_HEAD" ]; then echo "note: patches/$GATE_HEAD is missing (bench/cc_export_head.py writes one): the gate runs its own rule for this launch" >&2; GATE_HEAD=; fi
fi
GATE_FUSED=${GATE_FUSED-1}; [ -n "$GATE_HEAD" ] || GATE_FUSED=
ROCE_LIVE=${ROCE_LIVE-1}; { [ "$ROCE_AR" = 1 ] && [ -n "$GATE" ]; } || ROCE_LIVE=
[ -n "$ROCE_LIVE" ] && [ "$ROCE_AR_MAX" = 1MB ] && ROCE_AR_MAX=1200KB
GATE_THR=${GATE_THR-}; [ -n "$GATE_HEAD" ] || GATE_THR=
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

snap_of() { if [ -n "$PRESHARDED" ]; then echo "rank$1"; else echo "$SNAP"; fi; }   # the model folder of rank $1
run_cmd() {
  local r=$1 gid repo=${HOMES[$r]}/$REPO_REL patches=${HOMES[$r]}/inkling2-patches snap
  snap=$(snap_of $r)
  gid=$(on_rank $r "$(gid_snippet $r)" 2>/dev/null | tail -1); case "$gid" in [0-9]|[0-9][0-9]) ;; *) echo "ABORT: rank $r: no RoCE v2 GID" >&2; return 1;; esac
  printf '%s' "docker run -d --name $NAME --restart no --network host --ipc host --cap-add IPC_LOCK --gpus all \
 --shm-size 32g --memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --ulimit stack=67108864 --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
 -v $repo:/models/repo:ro -v $patches:/patches:ro -v ${HOMES[$r]}/inkling2-cache:/root/.cache -e PYTHONPATH=/patches -e SPARK_SERVED_ALIASES=${SERVED_ALIASES:-} \
 -e SPARK_ROCE_AR=$ROCE_AR -e SPARK_FI_MOE=$FI_MOE -e SPARK_PAGED_DECODE=$PAGED_DECODE -e SPARK_ROCE_AR_MAX=$ROCE_AR_MAX -e B12X_ROCE_HCA=$RDMA_HCAS -e B12X_ROCE_GID_INDEX=$gid -e B12X_ROCE_SPIN_LIMIT=$ROCE_SPIN \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOST_IP=${NODES[$r]} \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$RDMA_HCAS -e NCCL_SOCKET_IFNAME=$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE \
 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=$FABRIC_SUBNET -e NCCL_IB_SUBNET_PREFIX_LEN=${FABRIC_SUBNET#*/} -e NCCL_IB_GID_INDEX=$gid \
 -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_DEBUG=WARN \
 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8 ${GATE:+-e SPARK_GATE=$GATE} ${GATE_HEAD:+-e SPARK_GATE_HEAD=/patches/$GATE_HEAD} ${GATE_FUSED:+-e SPARK_GATE_FUSED=1} ${ROCE_LIVE:+-e SPARK_ROCE_LIVE=1} ${GATE_THR:+-e SPARK_GATE_THR=$GATE_THR} ${SCALE_PACK:+-e SPARK_SCALE_PACK=1 -e SPARK_CSF_GEMM=/patches/$CSF_SO} ${DENSE_I8:+-e SPARK_DENSE_I8=$DENSE_I8} ${DENSE_P12:+-e SPARK_DENSE_P12=$DENSE_P12} $SHARD_ENV ${WMARK_BOOST:+-e SPARK_WMARK_BOOST=1} ${DRAFT_KV_FP8:+-e SPARK_DRAFT_KV_FP8=1} ${PREFILL_GLUE:+-e SPARK_PREFILL_GLUE=$PREFILL_GLUE} ${BIAS_FAR:+-e SPARK_FA4_BIAS_FAR=$BIAS_FAR} ${FI_MXFP8:+-e SPARK_FI_MXFP8=1} ${STABLE_KERNELS:+-e SPARK_STABLE_KERNELS=$STABLE_KERNELS} ${KERNEL_LOAD_GUARD:+-e SPARK_KERNEL_LOAD_GUARD=$KERNEL_LOAD_GUARD} ${IMAGES:+-e SPARK_VISION=1 -e TORCHINDUCTOR_COMPILE_THREADS=1} ${IMAGES:+${IMAGE_MAX_EDGE:+-e SPARK_VISION_MAX_EDGE=$IMAGE_MAX_EDGE}} ${PRESHARDED:+-e SPARK_PRESHARDED=1} ${KV_NVFP4:+-e SPARK_KV_NVFP4=$KV_NVFP4} ${NCCL_EXTRA:-} ${REASONING_CACHE:+-e SPARK_REASONING_CACHE=$REASONING_CACHE} ${TOOL_PARSE_ERROR:+-e SPARK_TOOL_PARSE_ERROR=1} ${TOOL_CHOICE_FIX:+-e SPARK_TOOL_CHOICE_FIX=$TOOL_CHOICE_FIX} \
 --entrypoint python3 $IMAGE -m sglang.launch_server --model-path /models/repo/$snap --served-model-name $SERVED --trust-remote-code \
 --tp 4 --ep-size 4 --nnodes 4 --node-rank $r --dist-init-addr $HEAD:$DIST_PORT --host 0.0.0.0 --port $PORT \
 --attention-backend fa4 --mem-fraction-static $MEMFRAC --chunked-prefill-size $CHUNK --context-length $CTX \
 --max-running-requests $MAXREQ --cuda-graph-max-bs-decode $MAXREQ --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-prefill $MAXREQ \
 --watchdog-timeout 1800 --enable-metrics --kv-cache-dtype $KV_DTYPE $SPEC_ARGS ${POOL:+--max-total-tokens $POOL} ${SWA_RATIO:+--swa-full-tokens-ratio $SWA_RATIO} ${PREFILL_DECODE_INTERVAL:+--prefill-decode-interval $PREFILL_DECODE_INTERVAL} ${IMAGES:+--enable-multimodal} ${EXTRA_ARGS:-}"
}
if [ $DRY = 1 ]; then for r in 0 1 2 3; do echo "== rank $r"; run_cmd $r || exit 1; echo; done; exit 0; fi
echo "== preflight"; bad=""
for r in 0 1 2 3; do
  # The GB10 hooks must be on every node (a rank without them runs stock code and fails at the first collective).
  # with them go the forked GEMM's module when the packing is on and the confidence head's file when it is on (patches/*.py only otherwise)
  if [ $r = 0 ]; then mkdir -p ${HOMES[0]}/inkling2-patches && cp $HERE/../patches/*.py ${SCALE_PACK:+$HERE/../patches/$CSF_SO} ${GATE_HEAD:+$HERE/../patches/$GATE_HEAD} ${HOMES[0]}/inkling2-patches/
  else on_rank $r "mkdir -p ${HOMES[$r]}/inkling2-patches" && scp -q -o BatchMode=yes $HERE/../patches/*.py ${SCALE_PACK:+$HERE/../patches/$CSF_SO} ${GATE_HEAD:+$HERE/../patches/$GATE_HEAD} ${USERS[$r]}@${NODES[$r]}:${HOMES[$r]}/inkling2-patches/; fi
  out=$(on_rank $r "mkdir -p ${HOMES[$r]}/inkling2-cache; docker image inspect $IMAGE >/dev/null 2>&1 && echo image-ok || echo image-MISSING; test -f ${HOMES[$r]}/$REPO_REL/$(snap_of $r)/model.safetensors.index.json && echo weights-ok || echo weights-MISSING; ${OTHER_CONTAINER:+docker ps -q -f name=^$OTHER_CONTAINER\$ | grep -q . && echo other-engine-RUNNING || echo other-engine-off}" 2>&1)
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
