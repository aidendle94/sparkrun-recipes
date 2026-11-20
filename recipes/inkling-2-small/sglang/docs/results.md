# Inkling-2-Small on four DGX Sparks: lab notes

The working notes behind this recipe (2026-11-12 to 2026-11-20). First what the launcher starts and what took the
fleet down: the defaults, two outages, the silent stall and its recorder. Then by subject: what the GB10 hooks do,
the bug that produced garbage under concurrency, every decode lever with its numbers, the packed block scales, the
confidence head for the gate, long contexts and multi-turn history, the reasoning loop on long lists, the KV
cache, the prefill switches, reasoning effort, and the first day's measurements. "Production" and "the fleet" mean
the author's four Sparks; times are UTC. Each number is one boot on that fleet unless the text says otherwise. The
launch scripts are in `launch/`, the tools in `bench/`, the hooks in `patches/`, the build script of the forked
GEMM in `kernels/`. The summary and the way to run it are in the [README](../README.md).

## Launcher defaults since 2026-11-17

The 500K configuration the fleet has served since 2026-11-16 17:20 UTC is what the launcher starts when nothing is
set, so the serving line is now

    bash launch/relaunch-inkling2.sh

The last part of the old line, the scheduling setting `--prefill-decode-interval 2`, became a default later the
same day (`PREFILL_DECODE_INTERVAL`). After a prefill batch the scheduler runs two decode rounds before it takes
the next one, so requests that are already decoding keep getting tokens through a long chunked prefill; it is
skipped while nothing is decoding, and a new request still waits for the whole prefill ahead of it. It has been on
the serving line since the 500K stress test of 2026-11-14 and was never measured against a boot without it.

| variable (default) | container setting | empty means |
|---|---|---|
| `CTX` (524288) | `--context-length` | not meant to be empty; `CTX=262144` is the default before 2026-11-17 |
| `MEMFRAC` (0.94) | `--mem-fraction-static` | not meant to be empty; 0.93 before 2026-11-17 |
| `SWA_RATIO` (0.05) | `--swa-full-tokens-ratio` | the engine's 0.1 |
| `WMARK_BOOST` (1) | `SPARK_WMARK_BOOST=1` | the pool sized from the memory the kernel reports, boost included |
| `DRAFT_KV_FP8` (1) | `SPARK_DRAFT_KV_FP8=1` | the draft's KV pools in BF16 |
| `POOL` (follows the three before it) | `--max-total-tokens` | no cap: the engine sizes the pool at boot |
| `PREFILL_GLUE` (epilogue,stream,gate,qkvr,relbias) | `SPARK_PREFILL_GLUE` | stock's glue |
| `BIAS_FAR` (f2) | `SPARK_FA4_BIAS_FAR` | the image's forward kernel |
| `FI_MXFP8` (1) | `SPARK_FI_MXFP8=1` | the MXFP8 expert layers on Marlin: the target's arithmetic as it was |
| `PREFILL_DECODE_INTERVAL` (2) | `--prefill-decode-interval` | the engine's scheduling: no decode rounds held between prefill batches |

`POOL`, when it is not set, is the cap that was measured for the pool settings in force, the same bytes each time:
793,000 with FP8 KV, `DRAFT_KV_FP8=1` and `SWA_RATIO=0.05`; 716,000 with `DRAFT_KV_FP8=`; 600,000 with `SWA_RATIO=`
(or 0.1) as well. For any other combination (`KV_DTYPE=auto`, another ratio) the launcher sets no cap and says so:
no cap was measured for it. All three caps were measured at `MEMFRAC=0.94`.

`NCCL_EXTRA` and `EXTRA_ARGS` come after the defaults on the command line, so a value given there still wins, and a
relaunch line of the sections below still starts a fleet. It no longer starts the configuration of its day,
because the defaults now add to it. The lines of those sections are kept as they were run; with the launcher of
2026-11-17 the same four `docker run` commands came from these lines (dry-runs compared byte for byte; for the
launcher of 2026-11-20 see the end of the next subsection):

| configuration | line with the launcher since 2026-11-17 |
|---|---|
| serving since 2026-11-16 17:20 UTC (793,000, three prefill switches) | `bash launch/relaunch-inkling2.sh` |
| 793,000 without the prefill switches (KV section) | `PREFILL_GLUE= BIAS_FAR= FI_MXFP8= bash launch/relaunch-inkling2.sh` |
| 716,000 (KV section) | `DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= bash launch/relaunch-inkling2.sh` |
| 600,000 (Long contexts) | `SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= bash launch/relaunch-inkling2.sh` |
| the launcher's defaults before 2026-11-17 (262,144 context, pool sized at boot, no interval) | `CTX=262144 MEMFRAC=0.93 POOL= SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= PREFILL_DECODE_INTERVAL= bash launch/relaunch-inkling2.sh` |

### Two more defaults since: packed block scales (2026-11-19) and the confidence head (2026-11-20)

Both are on when nothing is set, so the serving line is unchanged.

| variable (default) | container setting | empty means |
|---|---|---|
| `SCALE_PACK` (1, serving since 2026-11-19 18:33 UTC) | `SPARK_SCALE_PACK=1` and `SPARK_CSF_GEMM=/patches/spark_csf_gemm_sm120.so` | the block scales as loaded and FlashInfer's own grouped GEMM: the stack before 2026-11-19 |
| `GATE_HEAD` (gate-head-mlp-reg.pt, serving since 2026-11-20 11:47 UTC) | `SPARK_GATE_HEAD=/patches/gate-head-mlp-reg.pt` | the gate's own rule, the draft's token probability against `GATE`: the stack before 2026-11-20 |

The launcher copies the module and the head file to every node with the hooks. It turns either switch off by
itself, with a note on stderr, when what the switch needs is missing: the packing without `FI_MOE=1` and `TP_SHARD`
or while `patches/spark_csf_gemm_sm120.so` is not built, the head without `GATE` or without its file. A launch
without the built module runs with the block scales as loaded, as before 2026-11-19; it is not a failed boot.

`SCALE_PACK= GATE_HEAD=` gives the four `docker run` commands of the launcher before them, token for token (dry-runs
compared; the bytes differ by the spaces the two empty settings leave). The lines of the table above give the
commands of their day only with these two in front as well, and with `STABLE_KERNELS= KERNEL_LOAD_GUARD=` (the two
kernel switches of 2026-11-17, see "Two outages on 2026-11-17"), which those lines never carried: with all four in
front, token for token (dry-runs against the launcher the table was written with). The two sections are below:
"Packed block scales and the forked GEMM" and "A confidence head for the gate".

## Two outages on 2026-11-17 and the kernels compiled while serving

Two of the image's Triton kernels take a property of the batch as a compile-time constant: `_tau_kernel` the number
of positions, `_gate_sink_kernel` the strides of its inputs. Triton compiles and loads one CUDA module for every
value, so each prefill of a size the process had not seen ran ptxas and loaded a module on every rank in the middle
of serving. One node's cache held 816 `_tau_kernel` variants after five days; a 14.5-hour boot added 71.

- 07:52:16 UTC: the rank-0 node froze without a line in any log and was power-cycled at 08:16. The other ranks
  compiled a 533-row variant at 07:52:16.7 and a 652-row one at 07:52:18.6; on rank 0 the first entry exists with
  empty files and the second does not exist. Memory was flat on the 10-minute samples up to 07:50 (2.1 GB free, no
  reclaim, no swap-out in 14.5 hours).
- 09:09:20 UTC: one rank's scheduler died with `Triton Error [CUDA]: operation not permitted` in
  `CompiledKernel._init_handles`, loading a 474-row variant compiled in the same second. The watchdog relaunched
  (9.5 minutes down). No driver line in any kernel log; rank 0's memory flat on a 2-second record.

Why the driver fails there is not known, and that the freeze has the same cause is an inference from the timing.
What is fixed is the trigger:

| variable (default since 2026-11-17) | what |
|---|---|
| `STABLE_KERNELS` (tau,gate_sink) | `patches/stable_kernels.py`: the two kernels, statement for statement, with those values as arguments. Exact: 422 comparisons on a GPU, all equal (`bench/stable_kernels_test.py`); 12 compiled variants where the stock kernels needed 268 for the same inputs |
| `KERNEL_LOAD_GUARD` (5) | `patches/kernel_load_guard.py`: a load refused with that error is tried again after `torch.cuda.synchronize`, up to five times, and logged |

On the boot of 18:32 UTC: all four ranks log both hooks; `bench/concurrency_check.py` and
`bench/reasoning_cache_check.py` clean; `bench/shape_sweep.py` sent 57 prompts of 55 different lengths (29 to 5,370
tokens) and no rank compiled a `_tau_kernel` or `_gate_sink_kernel` variant (three stable modules a rank since the
boot, nothing else). `STABLE_KERNELS= KERNEL_LOAD_GUARD=` gives the launcher's commands of before (dry-run).

`bench/head_mon.py` writes a line of memory, pressure and temperature every 2 s to `~/inkling2-headmon.log` and
fsyncs it, so a freeze leaves its last seconds on disk. It does not start by itself after a reboot. After a crash,
save `docker logs inkling2` on every node before relaunching: the relaunch removes the containers.

## The silent stall, and what every rank records since 2026-11-18

Twice the fleet stopped answering with every container up and every server process alive: on 2026-11-13, under the
concurrent agent loops of the tool-call suite, and on 2026-11-18, on parallel requests of one conversation. Both
times a new request was being prefilled beside running decodes, and both times the watchdog relaunched the fleet.
The cause is not found. The load test written to provoke it (`bench/stall_hunt.py`: short prefills joining running
decodes, long contexts, forks of a shared prefix, cancelled requests) did not stall the fleet on 2026-11-18.

A stall of this kind leaves nothing in a log. Every rank schedules for itself from the same broadcast requests and
must run the same forwards in the same order; if one rank decides differently, the collectives wait for each other,
the schedulers sit in a GPU sync or a socket and the GPUs idle. The RDMA wait itself gives up after
`B12X_ROCE_SPIN_LIMIT` polls (300 million: 133 s measured) and then only fills the error words of the runtime's
control record: nothing is logged, nothing is raised. So the hooks record what they can see:

- `patches/roce_collectives.py`, `SPARK_STALL_DUMP` (seconds; default 60, 0 = off; not a launcher variable, pass
  another value through `NCCL_EXTRA`). A daemon thread on every rank logs one `[stall] rank N: ...` line when a
  model forward has not returned for that long, or when no new forward began while the scheduler still holds running
  or waiting requests (a decode replay returns at once and the scheduler blocks later, reading its result or in the
  next request broadcast: the stall induced on 2026-11-18 12:48 UTC), and again at twice and at three times the
  limit. The line holds what the forward was (mode, rows, tokens, a checksum of the request ids and, at a prefill,
  each request's cached prefix and new tokens), the collectives issued since it began, which CUDA streams still
  have work queued, the RDMA runtime's doorbell sequence, its error words and the arrival flags of every peer, the
  proxy's counters, and the last 24 forwards. It reads host memory only, so it works while the main thread is
  blocked.
- Every 4,096 forwards a rank logs `[sig] rank N forwards=K crc=...`, a running checksum of those forward notes.
  The four ranks must print the same value: a difference is a rank that scheduled differently, before anything
  stalls.
- At its first forward after a boot every rank logs `[stall] watch armed on rank N, limit 60 s; ...` with its state
  at that moment, so the readout is known to work before a stall needs it.
- The watchdog, before it relaunches, also runs `bench/stall_evidence.sh` into its postmortem directory:
  `stall-rank<N>.txt` with that rank's `[stall]` lines, its last log lines, the CPU time of every thread of the
  scheduler over 3 s, GPU power, the RDMA error counters and py-spy stacks with locals.

How to read the four ranks together. A rank whose arrival flag from peer p is behind its own doorbell is waiting
for p; equal doorbells with every flag in mean the stall is not in the RDMA runtime (NCCL, or a local kernel).
Collectives since the forward began: 0 on every rank means the forward waits for work queued before it, equal
non-zero counts that every rank reached the same collective, unequal counts that one rank is behind. Among the
scheduler's threads, the main thread busy together with NCCL's progress thread is a pending NCCL operation; the
main thread alone is a wait inside the RDMA runtime.

`bench/stall_dump_check.py` checks the recorder on a CPU against a stand-in runtime. `bench/stall_hunt.py` is a
load test for a planned window (other users wait behind it): when no request finishes for `--quiet` seconds and
`/health` no longer answers 200, it disarms the watchdog, so that the stalled processes stay for inspection, and
exits; relaunch by hand afterwards. Since the recorder went in, the fleet has not stalled (as of 2026-11-20): it
has been read on induced stalls only, and the problem is open.

## GB10 hooks (`patches/sitecustomize.py`)

The dev image targets Hopper and SM100. On SM12x:

1. **FA4 arguments** — SGLang's SM12x FA4 wrapper drops `rel_bias_scale_after_add` and `rel_bias_presheared`. The
   first needs nothing (every kernel computes qk * scale + bias; the flag names SM100's operation order), presheared is
   forwarded, and the fp32 output Inkling requests is filled from the bf16 result. Until 2026-11-13 07:20 this hook
   multiplied the bias by the softmax scale, so prefill ran with a relative-position bias 11x too weak (see Measured).
2. **Experts** — the NVFP4/MXFP8 CUTLASS kernels are SM100-only; the experts take SGLang's Hopper path (Marlin GEMMs with
   NVFP4/MXFP8 fake-quantized activations).
3. **Collectives** — Inkling's all-reduces require NVLink multicast; across four boxes they use the process group's
   NCCL all-reduce over RoCE.
4. **Decode/verify attention** — Inkling2Attention sees the platform as Hopper, so decode and DSpark verify take the
   compact path, and that path's kernel is the Triton `decode_partials` (the SM100 bias-decode kernel does not run here).
   SGLang's kernel reads one page-table entry per tile of 32 or 128 keys and takes the tile's other keys from the
   pool slots that follow, which is only right when the page size is a multiple of the tile (SM100 runs 128) or a
   request's slots are consecutive. GB10 runs page size 1, so `patches/paged_decode.py` (`PAGED_DECODE=1`, the default)
   rebuilds the kernel from the image's source at import with the page looked up per key: bit-identical on consecutive
   slots, same speed (0.95-1.02x per call). If the image's kernel changes under it, the import fails and so does the boot.
5. **Aliases** — `/v1/models` lists `SPARK_SERVED_ALIASES` (ported from the DeepSeek-V4.1 recipe's overlay).
6. **Routing** — Inkling's CUDA `route_metadata` kernel needs more than GB10's 99 KB of shared memory from ~3,072 routed
   ids (forwards of ~384-512 tokens) and killed the scheduler; above 2,048 ids its Triton kernel runs instead.
7. **NVFP4 experts on FP4 tensor cores** (`patches/fi_moe.py`, `FI_MOE=1`) — instead of Marlin (BF16 math), the 40
   NVFP4 expert layers keep the checkpoint weights and run FlashInfer's SM120 NVFP4 grouped GEMM, fed by custom Triton
   kernels: one-CTA slot assignment, gather into the grouped 128x4 scale layout, capped SwiGLU + four-over-six
   requantization straight to packed NVFP4, and the scaled combine. Bit-identical to the Marlin path at decode
   sizes; 2.3x faster at 4,096 tokens. The 8 MXFP8 layers stay on Marlin. From 512 tokens the per-rank sums are
   combined in BF16 instead of FP32 (`SPARK_MOE_BF16=1`, the default; half the all-reduce bytes, NLL unchanged).
8. **RoCE all-reduce** (`patches/roce_collectives.py`, `ROCE_AR=1`) — b12x one-shot RDMA collectives for the TP group.
9. **Timers** (`SPARK_TIMERS=1`, with `--disable-cuda-graph`) — per-module CUDA-event breakdown on every rank;
   `bench/measure.sh` sends the standard workload. `SPARK_ROUTE_STATS=1` adds, per verify forward and layer, how many of
   the rank's experts the first K rows of each request reach (`bench/route_stats.py` summarizes). Diagnosis windows
   only: eager decode recompiles the Triton attention kernel for new shapes and is several times slower.

10. **FP8 KV cache** (`patches/fp8_kv.py`, `KV_DTYPE=fp8_e4m3`, the default since 2026-11-13 18:07; `KV_DTYPE=auto` is
   the BF16 pool, which takes none of it) — SGLang's e4m3 pool stores K/V as a plain saturating cast, which suits Inkling (K is L2-clipped, V
   RMS-normed), but SM12x has no reader for it. The hooks keep queries BF16, run decode/verify on the page-per-key
   kernel with a cast after the K/V loads, and hand FA4 a BF16 copy of the rows a prefill reads. Validated on the
   fleet 2026-11-13 (see Measured); the pool holds 1.9x the tokens at the same speed.

11. **Attention shadow check** (`patches/attn_check.py`, `SPARK_ATTN_CHECK=1`, diagnosis windows only) — recomputes
   sampled query rows of every prefill attention call (and of decode/verify calls with `--disable-cuda-graph`) in
   plain torch from the kernel's own inputs and logs the worst output and LSE error per geometry. A failure inside
   the check turns it off; it never fails a forward.
12. **Reasoning cache** (`patches/reasoning_cache.py`, launcher `REASONING_CACHE`: 1, the default since 2026-11-14
   17:35 UTC, or the path of a flag file; empty boots without) — the API server remembers the reasoning of each answer and puts it back into assistant history turns
   that arrive without it. See Long contexts below for why.

Never run GPU work while the fleet serves, and never send it log-probability requests over long
spans (`return_logprob` with a small `logprob_start_len`, or completions `echo` + `logprobs`): the full-vocabulary
logits of every scored position took a rank down on 2026-11-13.

## Garbage under concurrency (found and fixed 2026-11-13)

From bring-up until 2026-11-13 09:02 this deployment generated loops and cross-talk ("Need answer. Need.", one
stream's code in another's answer) whenever two or more requests shared a decode batch, and kept damaging later
single-stream requests once the KV pool's free list had cycled, until the next relaunch. Cause: hook 4's tile lookup at
page size 1. A decode batch allocates each step's slots in batch order (interleaved between requests), a prefill that
lands during a decode leaves a hole, and the free lists are FIFO and never sorted, so a request's newest keys were read
from other requests' slots. Prefill (FA4, real page table) was never affected, and a single stream on a fresh boot was
always right, which is why chats and every benchmark looked fine. `MAXREQ=1` (08:50-09:01) was not enough: the
fragmented free list still corrupted single streams after ~60K tokens of churn.

What caught it and what did not: acceptance length and tok/s looked *better* while broken (a loop is accepted at full
length). Only reading the text shows it; `bench/concurrency_check.py` does that with the distinct-4-gram ratio of the
output ids (sane 0.85+, loops below 0.7) and exact single-stream comparisons with prefills landing mid-decode and
after pool churn. Every multi-stream number measured before the fix was taken on corrupted output.

## Decode levers (2026-11-13 to 15: the gate, the int8 draft, the 12-bit packing and the sharded experts are the defaults, the rest is off unless set)

A decode step at one request is ~85 ms: the draft 8 ms, the verify of its 8 rows 77 ms (experts ~2/3, the dense BF16
weights ~1/3, both at memory bandwidth). Three switches were built and measured against the stock step on the same
24 code questions, 400 tokens each, greedy (`bench/gate_run.py`, compared with `bench/gate_compare.py`; tok/s =
tokens / decode seconds, per stream):

| | 1 stream | 4 streams | accept | what the target computes |
|---|---|---|---|---|
| stock (8 verify rows) | 37.3 | 18.7 | 3.49 | |
| `GATE=0.4`, the default since 2026-11-14 (`patches/spark_gate.py`: verify only the draft rows the draft is confident of) | 42.9 (x1.15) | 21.6 (x1.16) | 3.25 | the same model on fewer rows |
| `SPARK_DENSE_I8=draft,dhead` (`patches/dense_i8.py`: the draft reads int8 copies of its weights) | 39.0 (x1.05) | 19.7 (x1.06) | 3.49 | unchanged: held-out NLL identical on 11,542 tokens |
| both, the default since 2026-11-14 04:59 UTC (`DENSE_I8=draft,dhead,dfree`) | 45.2 (x1.21) | 23.0 (x1.23) | 3.26 | as the two above |
| `SPARK_DENSE_I8=draft,dhead,attn,router,head` + gate (the target's dense weights rounded to int8 too) | 48.5 (x1.30) | 23.4 (x1.25) | 3.17 | changed: NLL +0.0025 +- 0.0016, see below |

Steps at one request: stock 84.9 ms (draft 8.0, verify 76.8); gate 69.0; int8 draft 81.8 (draft 5.1); gate + int8
draft 66.0; gate + all int8 60.6 (verify 55.4).

**How far each one moves the text.** Greedy streams are compared at the first token where they part, with the
target's top-2 logit margin there from a stock boot's step trace (`SPARK_TRACE`, `bench/trace_margins.py`). The
yardstick is stock against itself: the same prompt alone and beside three other requests parts on 24 of 24 prompts
at a median margin of 0.21 logits (max 0.90), because the batch size changes the rounding (see batch noise below).

| against the stock single stream | prompts that part | margin where they part | took stock's runner-up |
|---|---|---|---|
| stock beside 3 other requests | 24/24 | median 0.21, max 0.90 | 18/24 |
| gate 0.4 | 15/24 | median 0.023, max 0.21 | 15/15 |
| int8 draft | 9/24 | median 0.029, max 0.17 | 9/9 |
| int8 draft and target | 24/24 | median 0.36, max 2.08 | 23/24 |

The gate and the int8 draft leave the target's arithmetic alone; they only move where a step ends, so a token is
computed in another row of another batch, and near-ties (1.2% of committed tokens sit within 0.06 logits of the
runner-up, 4.7% within 0.25) fall the other way. That is less than what a second request in the batch already does.
Rounding the target's own weights moves it more than that, and is not a candidate for the default.

**The gate as the default.** `launch-inkling2.sh` sets `GATE=0.4` unless the environment sets `GATE` (empty = stock
8 rows; the watchdog relaunches with the environment it was started in). Checked on the gated production boot of 2026-11-14 02:07 UTC (pool
705,049 tokens): `bench/concurrency_check.py` passed twice with every exact step equal to the reference on the first try (in both
runs one prompt's first run alone differed from its second and third, with no other request counted by the server.
The probable cause is the watchdog's own chat probe, 21 prompt tokens and 3 new ones every 60 s, which shares a
decode batch for a few steps and is gone before the check's one-second sample of the running-request count: of 48
identical greedy runs sent one at a time, 5 parted from their siblings, about 60 s apart, the first beside a logged
21-token prefill. It is not the prefix cache: hits come in whole 64-token blocks and those prompts are shorter), and
`bench/toolcall_check.py` (283 cases over the five APIs, two trials) gave 252 ok / 19 soft / 12 FAIL against 253 /
17 / 13 on the stock boot the day before, 12 of 12 agent loops on both. Ten of the failures are `tool_choice: none`
on either boot (the model writes the call as text); the other two (three on stock) are slips on the sampled trial,
in different cases each run.

**Int8 dense path.** `bench/dense_i8_test.py` (one GPU, production stopped): the kernel is within 1e-5 of float32 on
the same codes and reads half the bytes in half the time, us per call cuBLAS BF16 -> int8: attention qkvr 155 -> 79
(8 rows), three-source qkvr 160-182 -> 79-85, wo 64 -> 34, router 19 -> 13, unembedding 2,009 -> 1,027, draft
gate_up 378 -> 189, draft down 190 -> 96. cuBLAS's own BF16 output moves by up to 2 ulp with the row count (24 and
64 rows against 8), one source of the batch noise.
Cost: the codes sit beside the BF16 weights (prefill and the target keep reading those), 0.64 GB per rank for
`draft,dhead` and 1.91 GB with the target's parts, and a GB is ~140K tokens of pool (per million tokens: 5.7 GB of
FP8 target KV, 1.3 GB of BF16 draft KV): 630,466 tokens on the `draft,dhead` boot (750K on the stock boot after it),
267,058 with everything, where the engine also captured only 5 requests per batch. The pool is sized from the host
memory the head has left after loading, which moves by ~1.6 GB (~230K tokens) from one boot to the next whatever the
switches, so one boot's pool is a weak measure of a lever's cost.
**`dfree`** (`SPARK_DENSE_I8=draft,dhead,dfree`, 2026-11-14) frees the BF16 copies of the draft linears that only
the draft's own forward reads (qkv_proj, o_proj, gate_up_proj, down_proj: 20 weights, 802 MB per rank, called with
requests x 7 rows and never by a prefill; `kv_proj_ctx` runs over prefill chunks and the unembedding is the
target's, so those keep theirs). Speed and NLL equal `draft,dhead` (66.1 ms steps with the gate, every span
+0.000). The memory does not go back to the host: on every rank the 20 weights sit in 5 allocator segments (1.7 GB)
that also hold 825 MB of tensors that stay, and the allocator returns whole segments only. Freeing all the copies
after all the codes were built (first version) therefore left 802 MB of holes beside 413 MB of new codes, and the
draft took 1.6-1.7 GB of host memory per rank against 1.0-1.4 GB on stock boots. The patch now frees each copy as
soon as its codes exist, smallest first, so the next weight's codes land in the hole (a weight alone in its segment
would go back to the host; `bench/dense_i8_test.py` checks both layouts). Two boots nine minutes apart on
2026-11-14, head rank, GB:

| | stock (gated production) | `draft,dhead,dfree` |
|---|---|---|
| free after the target loads | 15.73 | 15.57 (holds 0.22 of unembedding codes) |
| the draft's load takes | 1.04 | 1.11 (reserved by torch unchanged, 386 MB of holes left over) |
| free when the pool is sized | 14.43 | 14.07 |
| pool, tokens | 784,899 | 747,193 |

So the int8 draft now costs ~0.36 GB per rank, 5% of the pool, for its 5% of decode speed, where it cost 0.64 GB
before. `bench/concurrency_check.py` passed on that boot with every exact step on the first try. An earlier failure
on the first `dfree` boot (after the churn all 8 first tries left the reference and ran at half speed) was a second
request in the batch that the check did not send: the gate's counters show 1,535 request rows over 1,024 steps in
that interval, and the check now prints the server's running-request count beside such a stream.

**The int8 draft as the default** (2026-11-14 04:59 UTC). `launch-inkling2.sh` sets `DENSE_I8=draft,dhead,dfree`
unless the environment sets `DENSE_I8` (empty = BF16 draft). On that production boot: all four ranks armed with the
gate and the int8 draft, no refusals, `bench/concurrency_check.py` passed, `bench/toolcall_check.py` 252 ok / 19
soft / 12 FAIL of 283 (the same totals as the gate-only boot: ten `tool_choice: none`, two slips on the sampled
trial) and 12 of 12 agent loops. The pool on that boot is 492,686 tokens, which is the head's boot-to-boot memory
variation and not the int8 path: the head's target load took 95.1 GB where it usually takes 92.9-93.6, as it did on
a stock boot 36 minutes earlier (94.6 GB, pool 496,528) while the other three ranks never move (93.2-94.0). Ten to
thirty minutes after such a boot the head has ~1.2 GB more available than after a 712K boot, so most of that memory was only in use
while the pool was sized. `bench/boot_mem.sh` samples the head's memory and largest processes during a boot to find
what holds it.
The relaunch of 06:00 UTC (same settings, sampled) landed high again: 92.9 GB, pool 694,200. What the two kinds of
boot have in common and where they part: torch's allocated / reserved bytes on the head are identical on the 493K
and the 694K boot (100,137 / 100,431 MB when the draft is in), while the host's available memory at that line is
12.8 GB against 14.6 GB, so the 1.8 GB sits outside torch's allocator. The sample of the 694K boot shows why the
head is the rank that varies: ranks 1-3 have their weights in after ~75 s and then wait, the head needs ~200 s
with 2.4 GB free the whole time (its server process's host memory swings between 2.2 and 6.4 GB with every shard),
and in the four seconds around "Load weight end" it drops 8 GB of page cache and gets 4-5 GB back from the driver.
The pool is sized ten seconds after that. A sample of a low boot is still missing; `relaunch-inkling2.sh` now starts
the sampler on every boot (`$LOG_DIR/inkling2-bootmem-<date>.tsv`, 15 minutes, /proc only), so the next one is on
file.
The all-int8 boot failed `bench/concurrency_check.py` on one of 16 concurrent streams at d4 0.79 against the 0.80
limit (text not kept; the same prompt scores 0.87 alone and 0.87-0.93 in that step on stock boots), and passed the exact
single-stream steps.

**Fake quantization** (`SPARK_DENSE_FQ=8|6`, `patches/dense_fq.py`) rounds the same weights with stock kernels, for
quality windows: int8 per output row is a relative weight error of 0.9-1.5%, int6 3.8-6.0% (margins where streams
part: median 0.31 and 0.51).

**Exact 12-bit packing of the target's dense weights** (`DENSE_P12=attn,head,free`, `patches/dense_p12.py`,
2026-11-15; a default together with the sharded experts since 2026-11-15, `DENSE_P12=` turns it off). The target's BF16 dense weights carry 10.5 bits each; one byte of sign + mantissa and a 4-bit exponent
offset from the row's largest exponent hold all but 5-9 weights in 100,000 exactly, the rest go in per-row exception
slots (0.754 of the BF16 bytes). The kernel rebuilds the BF16 bits in integers and multiplies as the BF16 path does.
`bench/dense_p12_test.py` (window only, one GPU) checks that every weight comes back exactly and times it against
cuBLAS on the attention projections: x1.27-1.30 at up to 16 rows, x1.24-1.26 to 32, x1.15 to 64 (tiles of
16 x 32 and 32 x 64 outputs with 4 warps, 64 x 128 with 8, all 2 stages: with outputs at least twice the rows Triton
keeps its warps along the outputs; the first version's 32 x 32 and 64 x 32 tiles were x0.75); x1.29 on the
unembedding at 8 rows, x0.88 on the router. That is about 3 ms of a 66 ms step for one or two streams. Outputs
differ from cuBLAS by the order of the sum only. Above 64 rows the kernel loses (the weight is read once per 64
rows), so prefill goes through `unpack_into`: a kernel that rebuilds the BF16 weight bit for bit in a scratch buffer
at the speed of a copy (0.3 ms for 3792 x 4608), then cuBLAS: 15-20 ms more per prefill chunk over 48 layers.
`bench/compile_check.py` compiles every tile without a GPU and fails on any that needs more than a program's
101,376 bytes of shared memory ((stages - 1) x (512 x rows + 384 x outputs) for this kernel; a 3-stage 64 x 128 tile
cost a window); `bench/emu/` is a CPU stand-in for Triton that runs the whole window script at tiny shapes first.

In the engine (`SPARK_DENSE_P12`, hooks in `patches/sitecustomize.py`): `attn` packs the target's qkvr and wo_ud of
all 48 layers, `head` its unembedding (the BF16 copy stays for the prefill), `free` drops the BF16 copies of the
attention weights once packed. Every weight is checked at the load (the unpack kernel must return its bits, the
linear kernel its columns for one-hot rows) and keeps its BF16 path if a check fails. Forwards of up to 64 rows
without a bias go through the kernel, q, k and v from their three sources in one launch; every other forward
(prefill) rebuilds the weight in a shared scratch buffer and runs stock's code, bit for bit. The unembedding writes
what stock writes: float32 logits softcapped per shard for this checkpoint (BF16 otherwise). Each seam logs its
first packed forward and every reason it leaves a forward to stock (`[dense_p12]` in the rank logs). `DENSE_I8`
parts of the same name win. Only `attn,head,free` is meant to be run: without `free` the packed copies cost 1.8 GB a
rank beside the BF16 ones.

Measured on one test boot against plain production (2026-11-15, 8 questions x 400 tokens, greedy): held-out NLL
identical token by token (the prefill path is stock's arithmetic), decode against prefill logprobs 0.009-0.044,
concurrency check and 12 agent loops clean; one stream -2.0 ms a step at equal accept (-2.8%, se 0.6 ms), four and
eight streams and prefill unchanged within noise, 0.2 GB a rank less allocated. On that boot the unembedding was
still stock's (the seam only took BF16 logits then). A second boot the same day ran the float32 unembedding through
the packed kernel on every rank and measured 32 questions: -3.12 ms a step at equal accept (se 0.33) = -4.5% for one
stream, NLL again identical, concurrency check clean. Not a default.

`bench/step_time.py <label> [--ref a,b,..] [--prompts 32]` compares two boots whose text differs: under the gate a
step that commits fewer tokens is also cheaper (6.5 ms a token of accept alone), so it fits ms a step over tokens a
step with a common slope ("at equal accept") instead of comparing either number raw.

**Experts sharded by width** (`patches/tp_shard.py` + `patches/fi_moe.py`, 2026-11-15; the default since
2026-11-15 as the launcher's `TP_SHARD=half,fp8`, which sets `SPARK_TP_SHARD=1`, `SPARK_TP_SHARD_HALF=1` and
`SPARK_TP_SHARD_FP8=1` in the containers; `TP_SHARD= DENSE_P12=` is the stack before it). Stock
places the 384 experts 96 a rank; a decode step's routed experts fall unevenly on the ranks and every rank waits in
the all-reduce for the one that reads most (about 6.6 ms a step for one stream). With `SPARK_TP_SHARD=1` every rank
holds all 384 NVFP4 experts cut to a quarter of the intermediate width (576), the same bytes a rank, and reads the
same bytes whatever the routing. One small all-reduce a layer exchanges each row's largest intermediate value, so
the activation is quantized exactly as unsharded (stock refuses NVFP4 under tensor parallelism for want of it); the
first call checks the exchange on a known pattern. What differs from stock is rounding (four partial rows summed
instead of one whole row): never bit-identical, mean NLL unchanged. `=local` quantizes each shard with its own
maximum (no exchange, not the checkpoint's arithmetic; test only).

- `SPARK_TP_SHARD_HALF=1`: the partial rows of the down projection leave the GEMM as FP16 scaled by a power of two
  and are summed in FP32. Relative L2 error of a layer's output against the float64 product of the same operands,
  on the GPU: stock 2.35e-3, shards 2.35e-3, shards with FP16 rows 2.05e-4; the same time a layer.
- `SPARK_TP_SHARD_FP8=1`: the MXFP8 layers (40-47) load the same way and run stock's Marlin code with every expert
  local. Their activations are quantized 32 values at a time, so no exchange is needed. `tp_shard.route_blocks`
  hands Marlin the row block expert parallelism would pick and keeps its routing kernel at stock's size. The rank
  log says `MXFP8 shard repacked for Marlin` (or that it was not).

Measured (32 questions x 400 tokens, one stream, step time at equal accept against production): sharded alone
-7.54 ms = -10.8% (56.1 against 48.3 tok/s); with FP16 rows and `DENSE_P12=attn,head,free` -9.69 ms = -13.9%; with
the MXFP8 layers too -12.82 ms (se 0.27) = -18.4% (55.9 ms a step, 60.0 tok/s), four at a time 106 against 88 tok/s a
batch, eight 130 against 114, prefill 2,417-2,482 against 2,337-2,409 tok/s, NLL +0.0012 +- 0.0015, concurrency
check clean on every boot.

Sign-off on the serving boot (2026-11-15 17:18 UTC, 500K context, pool 544,644 tokens), every stage against the
same test on the stack before: step time -12.22 ms (se 0.52) = -17.6% at equal accept; tool-call suite 250 ok /
20 soft / 13 FAIL of 283 (before: 252 / 19 / 12; the failures are the 10 `tool_choice: none` cases and 3 sampled
slips), 12 of 12 agent loops; temperature 0.7: 56.6 tok/s for one stream, 98 and 92 tok/s a batch of four; contexts
of 130K / 260K / 500K prefilled in 61 / 142 / 348 s, 4 of 6 lookups exact and 2 recitals (the 130K one recites
before too); `bench/recital_rate.py` at 130K: 36 right, 4 out of tokens, 0 wrong of 40 (4 of 40 before); 12 minutes
of `bench/stress.py` with 12 workers and one long context: 57 requests, 1 soft (a recital), health never failed;
concurrency check clean before and after; no traceback in a rank log; hottest node 89 C.

Tools for it: `bench/fi_shard_test.py` (window, one GPU, 41 s: the sharded kernels against expert parallelism and the
float64 product, timings; `--cpu` and `--emulate` run without a GPU, `--only` picks sections),
`bench/fi_shard_marlin_emu.py` (the Marlin checks' bookkeeping against a plain-torch stand-in),
`bench/compile_check_fi.py` (every Triton kernel the sharded path launches against GB10's shared-memory limit; two
runs, `--no-stock` and `--only stock`, since one process does not fit a 500 MB container),
`patches/graph_timers.py` + `bench/graph_timers_report.py` (`SPARK_GRAPH_TIMERS=1`, windows only: region times
inside the decode graphs; both graph runners report one class name, so select with `--rows`),
`patches/route_diag.py` (`SPARK_ROUTE_BALANCE=1`, windows only, garbage text: forces even routing to size the wait
for the busiest rank). `SPARK_FI_SF_EMPTY=1` skips writing scale buffers on the expert-parallel path
(bit-identical, 24-45 us a layer in the kernel test; not booted).

**Block verification** (`EXTRA_ARGS=--speculative-use-block-verification`, a stock server flag, 2026-11-14) accepts
the drafts of a sampled request jointly instead of one by one (arXiv 2403.10444): the same output distribution, never
fewer tokens per step. It changes nothing for greedy requests. `bench/bv_exact.py` (pure Python, exact rational
arithmetic on small Markov toys) checks that both rules stay exact under the gate's data-dependent cap, also when the
kernel runs over all 7 drafts with garbage in the masked rows and the cap is applied afterwards, which is what the
engine does. `patches/accept_stats.py` (`SPARK_ACCEPT_STATS=1`, windows only) computes the expected tokens per step of
both rules from the acceptance ratios of the same steps, so one boot compares them without sampling noise. On the
fleet, gated, 24 questions (`bench/sampled_run.py`): x1.002 at temperature 0.7 (2.737 -> 2.743 over 7,060 steps; the
same x1.002 on a second boot without the flag) and x1.005 at 1.0. The flag runs cleanly on GB10 (texts read, no errors),
but the gate already drops the drafts where the joint rule would help, so there is nothing to gain: left off.
For reference, sampled requests on the gated default: 40.7 tok/s at temperature 0.7 (66.9 ms steps, 2.74 tokens per
step, 3.8 verify rows), 21.2 per stream at four streams. Chat requests without a `temperature` are sampled at 1.0
(the engine's default for chat completions and messages; the model ships no generation config). Until 2026-11-20
this paragraph said 0.7, which is the default of the Responses API: see the capture under "A confidence head for
the gate".

**Sampled decode at temperature 1.0** (the publisher evaluates at temperature 1.0, top_p 1; window of 2026-11-16).
On the serving configuration one stream runs 53.4 tok/s greedy, 50.3 at temperature 0.7 and 48.5 at 1.0 (24
questions of 400 tokens; 2.93, 2.66 and 2.47 tokens per step), four streams 25.3 against 23.3 per stream. Steps
are not slower when sampling (at four forced rows 55.1 ms sampled against 54.5 greedy): the draft's tokens are
accepted less often, 0.69 0.60 0.56 0.51 0.47 0.45 0.42 by position at 1.0. Three levers that leave the output
distribution alone were measured, and none of them gains:

| lever | simulated on the dump (tok/s against the serving rule) | on the clock, same boot, paired by prompt |
|---|---|---|
| another gate rule for sampled requests: the best of five features and their thresholds is `mix` (the mean of the drafted token's and the best token's log probability) | x1.009 +- 0.002 at 1.0 (tau 0.35; x1.006 on held-out steps), x1.006 at 0.7 (tau 0.45); a lower threshold on the serving rule x1.002 | x1.012 +- 0.023 at 1.0, x1.007 +- 0.020 at four streams, x1.027 +- 0.023 at 0.7 |
| the draft sampling sharper than the target (alpha x the request's temperature, the accept reading the draft's probabilities at that temperature) | alpha 0.85 / 0.7 / 0.5 / 0.3: x0.996 / 0.991 / 0.984 / 0.976 at 1.0; an argmax draft x0.965 (x0.978 at its best threshold) | alpha 0.7: x0.988 +- 0.019 |
| the engine's block-wise rejection kernel (`SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1`) | not simulated (the same rule in another kernel) | a boot of its own: 51.2 tok/s at 50.7 ms a step against 49.0 at 50.6 (the difference is tokens per step of other sampled texts), 23.3 against 23.6 per stream at four, 52.2 against 52.4 at 0.7; at four forced rows 54.6 ms sampled against 54.3 greedy |

A rule that could read the target's own acceptance probabilities would gain x1.16: the room is in how well the
draft's confidence predicts the target, not in the threshold. The stock argmax-draft switch
(`SGLANG_DSPARK_USE_GREEDY_DRAFT=1`) keeps no step logits, which turns the gate off, so it was judged by
re-weighting the dump (alpha 0) and not booted. Block-wise rejection runs cleanly on GB10 with the gate (texts
read, K hashes equal on the four ranks) and is left off. Tools: `patches/sampled_lab.py` (`SPARK_SAMPLED_LAB=1` with
`SPARK_GATE_RID=1`, windows only: a request id `lab-<feature>_<tau>_<alpha>-` chooses the gate rule and the draft
temperature of its request; `bench/sampled_lab_test.py` is its CPU test), the dump of `patches/accept_stats.py`
(`SPARK_ACCEPT_DUMP=<dir>`, written by the rank that has the directory, paused while `<dir>/pause` exists) with
`bench/accept_sim.py` (scores gate rules and draft temperatures on dumped 8-row steps) and
`bench/sampled_lab_check.py`, and `bench/sampled_arms.py` (arms of one boot side by side, paired by prompt).
Sampled tok/s over 24 questions moves about 5 % between boots because the texts differ: compare rules within a
boot or on the dump.

**Three more kernel ideas for the decode step, none adopted** (windows of 2026-11-16; one stream spends about 57 ms
a step on the serving configuration):

- A fused path for the 1 to 8 rows of a verify step through the sharded experts: 204.1 against 204.5 us a layer at
  one row, and it failed one graph-against-eager comparison. The two grouped GEMMs are 190 of those 204 us and read
  their weights at the rate the memory gives. The code is gone; `bench/fi_shard_test.py --only stages` keeps the
  split of a layer that shows it.
- The engine's own fast-path switches (`SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH`, `SGLANG_OPT_SMALL_FIELD_COPY_VECTORIZED`,
  `SGLANG_OPT_SPEC_VERIFY_REUSE_STAGED_INPUTS` and `SGLANG_OPT_DSV41_DSPARK_MID_FOLD`, `_POST_FOLD`,
  `_PREPARE_INPUTS_ONCE`, `_SKIP_PREFILLED_INPUTS`, all `=1`): they run on GB10 with the hooks and leave the text
  alone; -0.5 ms a step against a second stock boot, which is inside what two boots of one configuration differ
  by. Not set. The gate arms under the post-fold relay since this window (it refused before), and under
  `SGLANG_DSPARK_SHARDED_ACCEPT` it verifies all 8 rows, because the draft's logits stay on vocabulary shards.
- The prefill glue's fused attn_res + RMSNorm kernel (below) inside the decode graph: the same bits, 1.4 to 2.5
  times slower than stock's pair of launches at 1 to 8 rows. It stays a prefill part.

What is left for decode is tokens per step, that is how well the draft's confidence predicts the target (see the
sampled-decode paragraph above), not the step. The confidence head of 2026-11-19/20 (its own section, below) is
that work.

`bench/nll_texts.py` scores held-out text (58 spans of 199 tokens, windows only: it asks for input logprobs) and
pairs a boot against a reference boot token by token; stock is 0.9828.

## Packed block scales and the forked GEMM (2026-11-19; the launcher's default, `SCALE_PACK=` turns it off)

A rank holds 191 MB of e4m3 block scales for every NVFP4 expert layer (7.6 GB for the 40 layers; a boot's log
prints them in units of 2^20 bytes, 182.2 and 7,290.0), one byte for 16 weights, and they carry about 4 bits each.
`patches/scale_pack.py` keeps them packed in the 128x4 tile layout the kernels read: one base byte and 32 four-bit
offsets for a group of 32 scales, and the 2.4-3.6% of groups whose bytes span more than 15 stored raw (56.0% of the
bytes; every layer is expanded once at the load and compared with the checkpoint's bytes). The idea of keeping the
scales resident in a compressed layout is from NVIDIA's hyperflare-inference
(github.com/NVIDIA/hyperflare-inference, Apache-2.0; no code taken); the format, the decode routine and the code are
this recipe's.

Expanding the routed experts' scales into a scratch tensor before each GEMM (the fallback path of `scale_pack.py`)
costs memory traffic: +7.5% a decode step on this fleet, measured on that path and on no other project's code. So the
GEMM reads the packed form itself.
`kernels/csf_gemm/build_csf_gemm.py` derives a fork of CUTLASS's `sm120_blockscaled_mma_array_tma.hpp` and of
FlashInfer's grouped NVFP4 wrapper from the image's own sources (exact text replacements: a changed upstream line
fails the build) and compiles `patches/spark_csf_gemm_sm120.so` (about 40 s, no GPU, several GB of memory: never
beside the serving fleet). In the stock kernel TMA copies a CTA's scale tile (128 rows x 16 scales = 2,048 bytes a
k tile) into a shared-memory stage; in the fork the producer warp writes that stage instead, each lane expanding two
groups from global memory (one 16-byte vector load of a group's codes and one byte, both groups' loads issued before
any arithmetic: a load stalls the warp, and with scalar loads the kernel was 5% slower), then lane 0 issues the TMA
loads of A, B and the activation scales with the stage's byte count reduced by the scale tile. Consumers are
untouched. swap_ab layout only (every decode call and every prefill below 16,384 routed rows); the other layout
expands. At the load `scale_pack` runs the fork on the layer's real weights with packed and with plain scales and
uses it only if the bytes are equal (it logs `nothing is expanded` or `NOT used`).

The module is built once per image, on rank 0, in the image the fleet runs and with nothing serving on that node:

    mkdir -p "$HOME/inkling2-csf-build"
    docker run --rm --network none --memory 16g --cpus 8 \
      -v "$PWD/kernels/csf_gemm":/k:ro -v "$HOME/inkling2-csf-build":/out \
      --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small /k/build_csf_gemm.py /out
    cp "$HOME/inkling2-csf-build/spark_csf_gemm_sm120.so" patches/

The script's one argument is an output directory: it writes the derived sources under `gen/`, the compiler's work
files under `ws/` and `spark_csf_gemm_sm120.so` beside them (`--no-compile` stops after the sources). The launcher
copies `patches/spark_csf_gemm_sm120.so` to every node with the hooks and names it in `SPARK_CSF_GEMM`. The module
is not in the repository (`patches/*.so` is git-ignored): it belongs to the image it was built from, so build it
again when the image changes. On 2026-11-19 a compile of it on a node that was serving left that node too little
memory and took the fleet down: stop the fleet first.

What a boot logs, on every rank: `[scale_pack] forked GEMM on w13_scale: packed scales give the stock bytes: the
kernel reads them, nothing is expanded`, the same for `w2_scale`, and `[scale_pack] 40 expert layers: block scales
... packed to ... (56.0%), every expert checked against the checkpoint's bytes`. `NOT used (<why>)` in the first two
means the module loaded and failed its check on that rank. A rank that does not find the file logs neither of the
two. In both cases the scales are expanded before each GEMM, with the same results and the 7.5% slower step. A file
that is there and cannot be loaded (a module built for another image, say) is not caught: by the code it stops the
boot. That case was not observed.

Measured (2026-11-19, the GPUs of all four nodes capped at 2,176 MHz): `bench/csf_gemm_test.py` (one GPU, fleet
stopped): same bytes as the stock kernel for both projections, BF16 and FP16 output, 485 against 507 us a call (w13)
and 289 against 303 (w2). Fleet, one boot against the plain boot of that morning, 16 coding questions of 400 tokens:
same text on 16 of 16 one-stream prompts, NLL identical token for token (10,547 tokens), one stream 56.55 against
56.06 ms a step (+1.0%, se 0.5% by the tool's fit), four and eight streams 30.6 and 19.9 against 27.7 and 19.6
tok/s a stream (x1.07 and x1.01 request by request; two runs of one boot differ by up to 7% at these widths),
3.0-3.4 GB more MemAvailable a node (the pool is not raised: it is margin). A cold prefill of 49,843 tokens ran at
2,770 tok/s against 2,719 on the plain boot, once each, on the fork's first build (the one with scalar loads); the
build that ships was not timed on that prompt.

The one-stream difference is the packing's cost and not noise. The 16 prompts took the same number of steps on both
boots, and prompt by prompt the packed boot is 0.59 ms a step slower (se 0.09, slower on 16 of 16). The two serving
boots of 2026-11-20, which carry the packing, ran 56.69 and 56.55 ms on the same prompts: 0.68 and 0.47 ms slower
than that plain boot (se 0.19 and 0.02). Another plain boot of that morning had been within 0.04 ms of it (se 0.06;
that one ran before the rank-0 GPU was capped). So the packing costs about 0.5 to 0.7 ms a step, 1%, for one stream.
At four and eight streams none was seen, on the one boot measured there, where two runs differ by more than 1%.

`bench/scale_pack_test.py` checks the packing on real checkpoint scales and on edge cases without a GPU;
`bench/csf_gemm_test.py` needs one GPU and the fleet stopped.

## A confidence head for the gate (2026-11-19/20; the launcher's default since 2026-11-20 11:47 UTC, `GATE_HEAD=` turns it off)

The gate (`GATE=0.4`, under Decode levers) verifies K of a request's 8 rows and takes K from the probability the
draft gave its own tokens. That number predicts poorly what the target accepts (log loss 0.503 on the capture
below), and the sampled-decode window had found the room there: in how well the draft's confidence predicts the
target. A small trained head predicts it better. For each drafted position j it gives the probability p_j that the
target takes draft j given that it took the drafts before it, and the gate's row count becomes

    K = 1 + the number of leading positions j with p_1 x ... x p_j >= thr_j

where thr_j is the measured step time of one more verify row times a rate tuned on held-out requests (the "leading
rows" rule). The head that ships is a 256-unit MLP (`patches/gate-head-mlp-reg.pt`, 4.7 MB, sha256
`0bc7b3d52da6c1bd...`). Position j reads the draft's final hidden state of that position (4,608 numbers, from one
forward over mask embeddings: no drafted token reaches it) and 24 numbers of the draft's own step distributions:
the log probability of the drafted tokens up to j, the two largest log probabilities and the entropy up to j + 1,
the drafted token's probability at the request's temperature, the temperature, the greedy flag and the position.

**Exactness.** The head never reads draft j + 1 or a later draft, the target's logits or the coins of the accept,
and K is a stopping rule over j. Those are the conditions under which a gate leaves sampled decoding exact, by the
argument that holds for the gate's own rule: at the capped row the engine emits the token of the uncapped rejection
chain (the draft if it was accepted, else the residual sample), never a fresh sample. A second reading of the
engine's accept code on 2026-11-20 (automated reviewers, not a second person) confirmed that and noted that no test
pins it: a kernel that resampled at the capped row would bias sampled output under any gate. Greedy requests are
exact under any rule, since a verified token is the target's argmax whatever the row count; their text can still
move at near-ties with the shape of the batch, as it does without a gate.

**In the engine** (`patches/spark_gate_head.py`, `SPARK_GATE_HEAD=<file>`; `patches/spark_gate.py` takes its row
counts from it). The head runs in the tail of the draft's CUDA graph, right after the draft wrote its step logits
and tokens, and K lands in a buffer the gate only copies. The ranks must verify the same rows, and they do not
rely on computing the same bits:

- at arming every rank runs the head on one seeded input and the ranks vote: the file, the vocabulary width read
  and the bits of that result must be equal on all of them, or no rank arms and the gate keeps its own rule
  everywhere;
- every step the ranks exchange their K through one small all-reduce and all of them use rank 0's; a counter adds
  the request-steps on which a rank's K was not rank 0's, and the gate's log line prints it (`ranks differ=0`).

The head refuses to arm, and logs why, on anything it was not built for (a missing or unreadable file, another
hidden size or vocabulary, a greedy-only draft, vocabulary-sharded step logits). After the vote there is no
fallback: an error in the tail stops the boot. A boot with the head logs on every rank

    [gate head] armed on 4 rank(s): /patches/gate-head-mlp-reg.pt (sha 0bc7b3d52da6c1bd, mlp-reg), hidden 4608, vocabulary 201024 (200058 read), thresholds [0.4302, 0.309, 0.3132, 0.2208, 0.2172, 0.2328, 0.2334], self-test 7d92deca3985c582, K exchanged every step: True
    [gate] row counts from the confidence head

and then `[gate] steps=... mean K=... hash=... ranks differ=0` every 1,024 steps. `NOT armed: <why>` in the first
line means the gate runs its own rule.

**The capture** (2026-11-19 21:19 to 2026-11-20 00:06 UTC). `patches/conf_capture.py` (`SPARK_CONF_CAPTURE=<dir>`,
capture windows only) records, on the one rank that has the directory, every decode step of the requests whose id
starts with a given prefix: the draft's hidden state of the 7 drafted positions, nine numbers a position from the
draft's and the target's distributions, and the step's outcome. Requests of other clients are never recorded. The
load (`bench/conf_capture_run.py`) was 2,316 requests in 155 minutes, all finished, 334,158 completion tokens: 650
general questions, 500 coding tasks, 330 agent turns with tool definitions, 500 tasks on a source file of 1-8K
tokens, 336 questions on 14 shared contexts of 15-104K tokens; 1,377 without a temperature, 702 greedy, 237 at
temperature 0.7. It gave 93,958 request-steps (657,706 drafted positions, 6.1 GB of hidden states). Every request
ran with all 8 rows verified, so each position has its label: for a greedy request whether the draft was the
target's argmax, for a sampled one the acceptance probability min(1, p/q) itself, without the coin. A first capture
an hour earlier had been stopped by a temperature guard after 141 requests (6,068 request-steps); a small head
trained on it alone already simulated at x1.035, so more data made the estimate firm and did not raise it.

A chat or messages request without a `temperature` is sampled at 1.0 on this server: that is the engine's default
for those two endpoints (0.7 is its default for the Responses API), and the recorder wrote 1.0 for every such
request. Where older paragraphs of these notes say 0.7 for chat requests, they were wrong.

Checks on the capture (`bench/conf_capture_check.py`, `bench/conf_capture_rank_check.py`): a greedy request's
accepted drafts are the leading run of target-argmax drafts in 27,525 of 27,525 rows; for sampled requests the
share of steps that accept at least j drafts is within one standard error of what the recorded p/q predict, at
every position; a recorded hidden row puts its own drafted token in the top 1% of the vocabulary for 96.1-97.5% of
positions and another request's token for 36%. `bench/conf_capture_align_check.py` reports FAILED on this capture
only because its margin (+0.2 between a row's own token and its neighbours') is too strict for a block draft, whose
neighbouring positions score 90-95%: read the rank check. 1,680 rows (1.8%) were steps the scheduler ran after a
request's end and are left out.

**Training** (2026-11-20, CPU only, the fleet serving). 5-fold cross-validation over requests, shared long contexts
and identical prompts kept in one fold: 92,278 held-out request-steps. The gain is simulated for one stream: the
tokens a step and the step time of the rows a rule would have verified, with step times measured on the fleet at
1, 2, 4, 6 and 8 verify rows (3, 5 and 7 interpolated), against the gate's own rule on the same steps (57.75
tok/s). "Whole families held out": each of the five prompt families (agent, coding, file, general, long) is judged
by heads that never trained on it. Intervals resample whole shared contexts and sets of identical prompts.

| head | log loss | gain, requests held out | gain, whole families held out |
|---|---|---|---|
| the gate's own rule (token probability) | 0.503 | x1.000 | |
| logistic regression on the gate's own numbers, no hidden state (under-trained, a lower bound) | 0.3879 | x1.0124 (1.0113-1.0136) | x1.0059 (1.0047-1.0071) |
| 256-unit MLP on 256 principal directions of the hidden state and those numbers | 0.3421 | x1.0402 (1.0385-1.0418) | x1.0312 (1.0297-1.0326) |
| **the same MLP with a better training recipe: the configuration of the head that ships** | 0.3379 | x1.0426 (1.0409-1.0444) | x1.0341 (1.0327-1.0355) |
| a two-stream model over all 7 positions, the same recipe | 0.3314 | x1.0446 (1.0427-1.0467) | x1.0353 (1.0338-1.0370) |
| ceiling: the true outcomes as predictions | | x1.1466 | x1.1457 |

- The recipe is full shuffling of the training rows, input dropout 0.5 on the hidden-state directions, 24 epochs
  instead of 8, target-side numbers as extra training targets (never inputs) and a uniform extra loss weight on all
  7 positions: `exp_combo.py --cfg 'mlp-reg:idrop=0.5,ep=24,lam=0.3,aux=p4+a6+p5,auxw=0.3,drop=0.1'`. By sampling
  class that configuration gains x1.051 without a temperature, x1.040 at temperature 0.7 and x1.030 greedy.
- The two-stream model's edge over it is 0.16-0.19 point, for 54 tensor operations a step where the MLP has 4
  (`bench/cc_count_ops.py`). That edge was not expected to survive the model's own run time at one stream, so the
  MLP was wired in.
- The gate is limited by information, not by training or by the rule: with the true outcomes as predictions the
  rule reaches x1.1463 of the x1.1466 ceiling, and more data did not help (12.5 / 25 / 50 / 100% of the training
  requests give x1.0330 / 1.0373 / 1.0399 / 1.0406 in a quick run on two folds). Most of the remaining 10 points
  need the target's verdict, which a gate cannot read.
- The drafted token as an input helps only on traffic like the capture's and costs a little elsewhere: left out.
- Agent turns are the exception among the families: the recipe adds nothing there, and a head that also reads the
  other positions adds 0.4-0.7 point. That family is narrow (330 requests from 14 templates and one tool set), and
  such a head is the next one to test for an agent workload.

**The test boot** (2026-11-20, 10:18 to 11:06 UTC): the serving defaults, the head, and request-id overrides so
that the same prompts ran back to back under the gate's own rule, under the head and with all 8 rows; step times of
the plain serving boot were taken before and after. It works as built:

- the head armed on all four ranks from one file, and the seeded self-test gave the same bits on every rank;
- the engine's rows are the trained head's: on 1,874 recorded request-steps (greedy, temperature 1.0 and 0.7, alone
  and four at a time) K from the draft graph equals K recomputed on a CPU from the recorded hidden states in every
  one (`bench/gate_head_capture_check.py`);
- the ranks agree: over 27,552 steps (37,635 request-steps) no rank computed another K than rank 0;
- greedy text does not depend on the rule: 16 of 16 outputs of 400 tokens are identical under the head, under the
  gate's own rule and with all 8 rows, alone. Four at a time, greedy outputs leave the 8-row reference at near-ties
  under either rule (the gate's own rule 0 of 16 identical, the head 1 of 16): the known effect of batching;
- `bench/concurrency_check.py` passed; no traceback and no `[stall]` report in a rank's log.

The head against the gate's own rule inside that boot (`bench/gate_run.py --head`, `bench/gate_ab_report.py`; the
first 16 prompts of `bench/trace_run.py`, 8 coding and 8 general questions; 400 tokens a request, tok/s per stream;
the two arms of a pair ran back to back on the same prompt):

| phase | units | gate's own rule | head | ratio per prompt or batch (bootstrap interval) | over all tokens | tokens a step | ms a step |
|---|---|---|---|---|---|---|---|
| greedy, alone | 16 prompts | 54.10 | 55.34 | x1.025 (1.013-1.040) | x1.023 | x0.970 | x0.948 |
| temperature 1.0, alone | 16 prompts | 49.12 | 51.60 | x1.057 (1.013-1.098) | x1.050 | x1.081 | x1.029 |
| greedy, four at a time | 4 batches | 25.45 | 26.80 | x1.040 (0.994-1.086) | x1.053 | x1.005 | x0.955 |
| temperature 1.0, four at a time | 3 batches | 24.22 | 25.45 | x1.021 (1.010-1.033) | x1.051 | x1.071 | x1.020 |

The interval is a percentile bootstrap over the prompts or the batches. With 4 and 3 batches it is little more than
the range of the batch ratios (0.973 to 1.100, and 1.010 to 1.033) and is not a 95% interval: a t interval is 0.95
to 1.13 for the greedy batches and 0.99 to 1.05 for the sampled ones. The two four-at-a-time rows show a direction,
not a size.

In that boot both arms replay the head's tail in the draft graph and only the gate's own rule also runs its eager
computation, so these ratios are the head's choice of rows plus the eager rule it replaces (0.28 ms), not net of
the tail's own cost. The simulation had said x1.030 for greedy requests and x1.051 for requests without a
temperature: the measurement agrees. Across boots (greedy, alone, 16 coding prompts of 400 tokens,
`bench/step_time.py`; the same text in all four runs):

| boot and rule | ms a step | tokens a step | tok/s | at equal accept, against the two plain boots |
|---|---|---|---|---|
| plain serving boot, before | 56.69 | 3.42 | 60.41 | |
| plain serving boot, after | 56.55 | 3.42 | 60.81 | (-0.21 ms, se 0.36, against the one before) |
| test boot, the gate's own rule | 58.08 | 3.42 | 59.97 | +0.75 ms a step (se 0.32) = +1.3% |
| test boot, the head | 56.05 | 3.42 | 61.64 | -0.89 ms a step (se 0.33) = -1.5% |

The plain serving boots of this table carry the packed block scales (the default since the day before). The plain
boot of the packing section (56.06 ms a step, 61.31 tok/s) carries neither of the two. The ms, tokens and tok/s
columns are medians over the 16 prompts.

- The head's tail costs about 0.75 ms a step at one stream (the gate's own rule in the test boot against the plain
  boots; a little of it is the test boot's own switches). Most of it is eight passes with torch kernels over a
  request's 7 x 200,058 step logits, for the two largest log probabilities, the entropy and the tempered
  probability. One fused kernel would read them once; it is not built.
- Net over the plain serving boots, greedy, one stream: x1.017 (61.64 against 60.61 tok/s, the mean of the two).
- Net for the other phases was not measured across boots. Taking the tail's 1.3% off the ratio inside the boot
  gives about x1.036 at temperature 1.0 alone: an estimate, not a measurement. Four at a time there is none: the
  tail's cost at four streams was not measured, and those ratios show a direction only.
- The serving boot after the deployment (healthy 11:47 UTC; the same 16 coding prompts): 56.09 ms a step, 3.42
  tokens a step, 61.79 tok/s, the same text on 16 of 16; at equal accept -0.72 ms a step (se 0.30) = -1.2% against
  the two plain boots. By the median tok/s, the measure of the figure above, that boot is x1.019 net (61.79 against
  60.61), a second boot for it. Over all tokens the two boots are x1.015 and x1.012.
  `bench/concurrency_check.py` passed on it, and the four ranks printed equal row-count lines over 3,072 steps with
  `ranks differ=0`.
- For scale: two boots of one configuration have differed by up to about 0.5 ms a step (0.9%). The net gain at
  equal accept (0.9 and 0.7 ms a step) is one and a half to two times that, seen on two boots.
- Against the stack before both of the newest defaults: the two plain boots of 2026-11-19, without the packing and
  without the head, ran the same prompts at 56.06 and 56.12 ms a step and 61.31 and 60.95 tok/s, with the same text
  on 16 of 16. The serving boot with both runs 56.09 ms and 61.79 tok/s. For one greedy stream the head gives back
  what the packing costs, and the two together come out x1.008 and x1.014 by the median tok/s (x1.002 over all
  tokens): about level.

What this does not show:

- Short prompts only, 24 in all, 400 tokens each: 16 coding questions (the step times across boots) and a mixed
  set of 16 for the comparison inside the boot, 8 general questions and 8 of those coding questions. No agent turns
  with tools and no long contexts ran on the test boot; for them there is the simulation on the capture and nothing
  else.
- Temperature 0.7 was only in the recorded check (4 prompts a phase, the recorder running): x0.97 per prompt (the
  four prompts ran x0.88 to x1.03), x0.96 over all tokens. Not measured. The other phases of that check: x1.02
  greedy and x1.05 at temperature 1.0 alone (four prompts each, x0.94 to x1.19 and x0.81 to x1.24), and x0.91 for
  its one greedy batch of four.
- Four at a time: 4 and 3 batches (the last phase was cut when the rank-0 node reached 95 C), too few for an
  interval (see the note under the table). The thresholds come from one-stream step times and were used unchanged;
  79% of the captured steps ran four requests at a time, where an extra row costs less.
- The head is the training harness's fold-1 model, fit on three fifths of the capture, not a head fit on all rows.
  The capture is one load of templated prompts with mostly truncated outputs (77% of the requests hit their token
  limit, 89% of the completion tokens were reasoning).
- The step times behind the thresholds were measured on the capture boot with the rank-0 GPU capped at 1,958 MHz
  (2,176 on the test boot).
- About 45 minutes of serving on the test boot. As the default it serves since 2026-11-20 11:47 UTC: about two
  hours when this was written.

**How the head was made, file by file.** The recorder is `patches/conf_capture.py`; its load generator
`bench/conf_capture_run.py` (paced by the node's temperature; `--corpus` wants a directory of source files for its
file tasks); the checks of a capture `bench/conf_capture_check.py`, `conf_capture_rank_check.py` and
`conf_capture_align_check.py`; `conf_capture_test.py` runs the recorder on a CPU. The first trainer, on the raw
capture, is `bench/conf_head_train.py` (`conf_head_selftest.py` is its self-test). The harness of the training
program: `bench/cc_build_features.py` (a compact feature set from a capture: 256 principal directions of the hidden
state), `bench/cc_lab.py` (folds, gate rules, the simulation), `bench/exp_base.py` (baselines), `bench/exp_combo.py`
(every head of the table is one `--cfg` line), `bench/cc_count_ops.py` (operations a step) and `bench/cc-run.sh`
(one capped CPU container a node: 3 GB, 6 CPUs, refused under 7 GB of available memory). Export and checks:
`bench/cc_export_head.py` (trains the configuration with `exp_combo.py`, folds the projection and the
standardisations into the first layer, writes the head file and compares the engine's functions with the trainer's
predictions), `bench/gate_head_test.py` (CPU checks of the engine code), `bench/gate_head_capture_check.py` (a test
boot's recorded rows against a CPU recomputation). The A/B is `bench/gate_run.py --head` with
`bench/gate_ab_report.py`.

The head that ships is fold 1 of the configuration above, exported with `cc_export_head.py --name mlp-reg`. Its
`meta` entry names the configuration, the rate (60 tok/s), the thresholds and the sha256 of the `exp_combo.py` and
`cc_lab.py` that trained it; they are the files in `bench/`. The harness scripts expect to sit in one directory
with `spark_gate_head.py`, `spark_gate.py` and `conf_capture.py` (the container's `/c`), with a feature set in `/d`
and an output directory in `/o`: `cc-run.sh` mounts them that way. Not in the repository: the capture (6.1 GB of
hidden states), the feature sets, the scratch scripts of the experiment lanes that `exp_combo.py`'s docstring names,
`spark_gate_before.py`, which `gate_head_test.py` compares the gate against (it is `patches/spark_gate.py` as it
was before the head: in a clone of the GitHub repository,
`git show 3d7e798:recipes/inkling-2-small/sglang/patches/spark_gate.py > spark_gate_before.py` makes it), and the
author's lab record, which the docstring of `bench/cc_lab.py` cites as
`research/2026-11-19-confidence-head/RESULTS.md`: this section is its public form.

## Long contexts and multi-turn history (2026-11-14)

Serving 524,288 tokens of context since 2026-11-14 06:49 UTC through the environment of the relaunch (the launcher's
defaults were 262,144 and an automatic pool until 2026-11-17; the line as it was run):

    CTX=524288 MEMFRAC=0.94 EXTRA_ARGS="--max-total-tokens 600000 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

The pool is fixed at 600,000 tokens. `--prefill-decode-interval 2` is a scheduling setting added for the test boot
(a launcher default since 2026-11-17, `PREFILL_DECODE_INTERVAL`). Measured on that boot (`bench/longctx.py`, `bench/stress.py`, `bench/longctx_edges.py`):

| | |
|---|---|
| cold prefill | 130K tokens 62 s (2,096 tok/s), 260K 142 s, 390K 244 s, 498K 345 s (1,443 tok/s) |
| decode | 52-55 tok/s at any of these lengths |
| memory, temperature | flat with context (head 1.6-1.8 GB available); 88-89 C at the peak; no watchdog recovery |
| edges | a 528K prompt is refused at once (HTTP 400); two 318K prompts at once both exact; a client abort mid-prefill recovers; 392K cold plus 48 short lookups: none wrong |
| a short request behind a long prefill | waits for the whole prefill (338 s behind a 500K one); the interval flag only helps streams already decoding. Open; `--schedule-policy shortest-prefill-first` is untested |
| output limit | a lookup above ~300K needs more than 600 output tokens (the reasoning uses them); the tools use 4,000 |

**Send the reasoning back with the history.** Inkling's chat template renders the thinking of an earlier assistant
turn only if the client sends it: `reasoning_content` on the assistant message (`/v1/chat/completions`), or the
thinking blocks (`/v1/messages`). Most OpenAI-style clients send the answer text only. Over a long first message
the model then starts the next turn's reasoning from the first question again: it answers a later lookup with the
codes of an earlier one, or loops on "which question is the current one" until the tokens run out.
`bench/stale_answer.py` (60K-token record context, a lookup, a coding question, two more lookups; two contexts x six
conversations, greedy), lookups after a history that were not right:

| how the history is sent | not right |
|---|---|
| answer text only (what the stress test did) | 9 of 24 (4 stale, 4 empty, 1 wrong) |
| the same on a boot close to SGLang's recipe (page size 128, BF16 KV, stock decode kernel, no DSpark, gate or int8) | 8 of 24 |
| with each turn's reasoning (`--variant reasoning`) | 0 of 24; 0 of 48 at 16K (7 of 48 empty without); 0 of 24 three conversations at a time; 0 of 8 at 130K |
| with one fixed sentence as every turn's reasoning (`stub`) | 0 of 24: the block has to be there, its text matters little |
| no coding turn (`nocode`) / coding answer kept whole (`fullcode`) | 10 of 24 / 5 of 24: not the cause |
| records in a system message, each question its own user message (`system`) | 1 of 24 |
| answer text only, **reasoning cache on** | 0 of 24; 0 of 24 three at a time |

At 4K of context the failure does not appear, and short agent loops pass either way (12 of 12 with
`--drop-reasoning`), so it needs a long context to show. The tool-call suite's loops have always sent the
reasoning back, which is why they never saw it.

The attention kernels were checked on the fleet for this (`SPARK_ATTN_CHECK=1`, all four ranks, 262K boot): 13,700
query rows of the global layers and 68,632 of the windowed layers against the torch reference, 44 cold geometries and
340 continuations of a cached prefix with up to ~130K keys: worst output error 0.017 relative, worst LSE error 0.0003,
no row above the 0.05 tolerance. The server reuses a cached prefix in steps of 64 tokens (all 340 continuations, and
146 of 146 reused prefixes in production's log), so a prefix offset other than 0 or 64 within a 128-key block, which
the kernel's own tests do not cover either, does not occur here. `bench/bias_consistency.py --tokens 60000 --n 200`: decode NLL 0.1872, prefill
0.1871, chosen-token logprob apart by 0.0033 on average. Decode was not shadow-checked directly (that needs a boot
without CUDA graphs).

**The reasoning cache** does the client's part in the server, for clients that cannot send reasoning back. Every
answer's reasoning is remembered under the conversation up to it (and under its tool-call ids, and its text if that
is long enough to be unambiguous); an assistant history turn that arrives without reasoning gets it back before the
template renders. Chat completions and `/v1/messages`, streamed or not; the Responses API is not covered. It lives in
the API server's memory (`SPARK_REASONING_CACHE_MB`, 32): a restart forgets it and turns from before are rendered as
sent. The launcher's default since 2026-11-14 17:35 UTC (`REASONING_CACHE=1`); `REASONING_CACHE=` on a relaunch boots
without it. With a flag-file path as the value the hooks are installed at boot and work only while that file exists
in the head's container, which switches it without a restart:

    REASONING_CACHE=/tmp/spark-reasoning-cache.on ... bash launch/relaunch-inkling2.sh
    docker exec inkling2 bash -c 'touch /tmp/spark-reasoning-cache.on'     # on;  rm -f turns it off
    python3 bench/reasoning_cache_check.py --expect on                     # the server's prompt-token counts say which

The cost: prompts grow by the restored reasoning (as they would with a client that sends it).

### The reasoning loop at 130K ("recital"), investigated 2026-11-14

One exact 130K lookup, prefilled cold, ran out of 4,000 output tokens copying record lines in its reasoning (3 of 4
cold runs on production-type boots; 17 of 17 right warm). What it is, from full captures and token-level replays:

- The model has the answer first. In 18 of 19 captured recitals the reasoning had already named the right code of the
  item ("I recall seeing `item 593202 | code 281710`"). It then opens a quote of the surrounding records to verify it.
  A healthy quote is the line before, the line, the line after. When the quote opens one or two lines after the
  target (about half the cases) or at a wrongly recalled neighbour hundreds of lines away, the target line never comes
  and the copy never ends: after 12 copied lines the next line's first token has 91-97%, after 40 lines over 99%;
  the only other candidate is `...`, a closing quote is below 0.1%. The second way of running out is rarer: the
  first recall fails ("I recall seeing item 483041? Not yet") and the model starts to scan 13,000 lines from the top.
- The way in is a near-tie in the model's own distribution: which record preceded the target (23% for the wrong one
  against 22% for the right one) and at which line the quote opens (33% against 35%). The engine follows that
  distribution: the tokens of the failing run are the most likely ones at 317 of 322 replayed positions and never
  outside the top 10.
- It is not one of this deployment's changes. The recipe-like control boot (page size 128, BF16 KV, the image's decode
  kernel, no DSpark, gate, int8, FlashInfer experts or RoCE all-reduce) gives the same distributions on the failing
  run (chosen token's logprob apart by 0.026 on average, median 0.0001, at most 0.61) and the same rate: 2 of 20
  lookups ran out there, 2 of 20 on production for the same questions. The decode path against the prefill path on
  production (`bench/path_compare.py`): mean 0.024, at most 0.68. The largest differences of both comparisons are on
  the fuzzy "which record was next to it" tokens, where several candidates are nearly tied; any numeric path moves
  them, none removes them.

Rates (`bench/recital_rate.py`, 3 random items a lookup, greedy, 1,500-token limit, 40 lookups unless noted):

| | ran out of tokens |
|---|---|
| records, 30K and 60K | 0 of 40 each |
| records, 130K | 4 of 40 (3 recitals, 1 scan); generated in 120-token pieces: 4 of 40 on the same questions, 6 of 40 on 40 others |
| records, 130K, control boot | 2 of 20 (production on the same 20: 2) |
| temperature 0.7 | 11 of 40 |
| **the hosted model on OpenRouter, the same 40 questions over the same 129,592 prompt tokens** (11-15, `--hosted <provider>`, no fallbacks) | Together 4, DeepInfra 5, Modal 7 and 5 (run twice), PrimeIntellect 8 of 40; no wrong answer; this fleet the same day: 4 of 40 |
| the hosted model at 60K (Modal, Together) | 0 of 40 each |
| "cite only the single line, do not copy other records" added to the question | 4 of 40 |
| reasoning effort medium / low | 9 of 40 / 4 of 40 |
| a guard that breaks every copy after 10 lines (simulated with the API) | 5 of 80 against 10 of 80 without; it fired in 6 lookups and 3 of them then answered right, the others kept re-verifying to the limit (the rest of the difference is the run-to-run movement below) |
| **Python source, 127K (14 standard-library files), file and parameters of 3 functions** | **no recital in 40**; 35 right within 1,500 tokens, the other 5 were still weighing how to read the question (with a 4,000 limit 3 are right; the other 2 asked for a function that exists only in a docstring: one answered that after 3,300 tokens, one used all 4,000) |

Which lookups fail moves with anything that moves the numbers (another boot, another wording, other requests in
the batch): two runs of the same 40 questions shared 1 of their 4 failures. Sending the identical request
again on the same boot is not a reliable way out: of four failing questions sent three more times each, one failed
every time, one once, two never (their failures were on the boot before). A changed wording draws again with the
same odds (three of the four were right with the extra sentence above). What follows for a client: keep temperature 0; give long-context turns
an output limit of 4,000 or more (code questions at 127K used up to 3,300 tokens of reasoning); treat
`finish_reason: length` with an empty answer as no answer and ask again in other words. Lists of thousands of
near-identical lines are the bad case; a search tool is the right way to look things up in them. Nothing was changed
in the server.

The hosted model does the same (2026-11-15, `thinkingmachines/inkling-2-small` on OpenRouter, each run pinned to one
provider): 29 of 200 lookups at 130K ran out of tokens (4 to 8 of 40 a run, recitals of 65 to 136 copied lines and
empty answers in about equal parts) against 4 of 40 here, none of 80 at 60K against none here, and no wrong answer
anywhere. Which questions fail differs from provider to provider and between two runs on one provider (Modal: 7,
then 5, four of them shared), as it does between two boots here. Reasoning effort was left unset on both sides; the
answers that came were as long as this fleet's (median 400 to 500 output tokens against 464).

## KV cache: what a token costs and what grows the pool (2026-11-15/16)

Serving since 2026-11-16 00:34 UTC, through the environment of the relaunch (the launcher's defaults since
2026-11-17; the line as it was run):

    CTX=524288 MEMFRAC=0.94 NCCL_EXTRA="-e SPARK_WMARK_BOOST=1 -e SPARK_DRAFT_KV_FP8=1" \
      EXTRA_ARGS="--max-total-tokens 793000 --swa-full-tokens-ratio 0.05 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

Since 17:20 UTC the same day the line also carries the three prefill switches (see Prefill below); they do not
change the pool.

Two steps, each in the bytes of the one before: the windowed pool at half its share (716,000 tokens, 11-15 22:37
UTC), then the draft's pools as FP8 (793,000 tokens, below).

### The grown pool (716,000 tokens, 2026-11-15 22:37 UTC)

    CTX=524288 MEMFRAC=0.94 NCCL_EXTRA="-e SPARK_WMARK_BOOST=1" \
      EXTRA_ARGS="--max-total-tokens 716000 --swa-full-tokens-ratio 0.05 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

The pool is 716,000 tokens (windowed pool 35,800 slots) in the bytes the 600,000 cap took: half the windowed pool
(6,349 bytes a token against 7,578), the cap raised by the same factor, and `SPARK_WMARK_BOOST=1`
(`patches/wmark_boost.py`), which gives the sizing the host's `MemAvailable` without the kernel's watermark boost
(below), so that a boosted boot gets the same pool. Nothing in a request's arithmetic changes. Measured on that
boot against the boots before:

| | 600K cap, ratio 0.1 | 716K cap, ratio 0.05 |
|---|---|---|
| pool, windowed pool | 544,644 / 54,464 on the last boot (a boosted one; 600,000 / 60,000 on a full one) | 716,000 / 35,800 (one worker read 2.24 GB of boost at the sizing second and had it added back) |
| one stream, greedy code (`bench/step_time.py`, against the sign-off boot) | 61.9 tok/s | 61.8 tok/s, step time x1.001, same text on 8 of 8 |
| 4 / 8 streams | 106 / 130 tok/s | 103-105 / 143 tok/s |
| windowed pool run full (`bench/swa_fill.py`: 64 new conversations of 1.5K tokens, 8 at a time, then the oldest 8 and the newest 8 again) | 80 of 80 lookups exact; old ones 8.1 s (recomputed), recent ones 2.4 s | 80 of 80 exact; 8.2 s and 2.6 s; the same with 48 conversations of 6K tokens (64 of 64) |
| 130K context | | lookups exact cold and warm, prefill 2,116 tok/s, decode 69 tok/s |
| 500K context, twice | prefill 1,433 tok/s; 1 of 2 lookups answered | prefill 1,453 and 1,440 tok/s, decode 56-58 tok/s; 1 of 4 lookups answered, the other three ran out of 4,000 tokens reciting the list (the recital loop above; the 114 records in each of two recitals checked against the prompt are exact and in order) |
| head memory during a 500K prefill | 1.6 GB available at the lowest | 2.33 GB available, 1.81 GB free at the lowest |
| soak (`bench/stress.py`, 12 workers, one long context, 15 min) | passed | passed: 54 requests, one recital on a 158K lookup (right when asked alone), no failed health check, peak 387K tokens in use |

What the smaller windowed pool costs: cached prompts whose windows stay alive, about 35 conversation ends of 1,024
tokens instead of 53 to 58. One that lost its window is prefilled again when it comes back (8 s at 1.5K tokens, 26 s
at 6K in the fill test); a running request is not affected. The first text check after the boot flagged one prompt
(the watchdog's probe shared its decode batch, a near-tie at token 80, sane text); the second run and the one at
the end of the window passed.

Rollback: the relaunch line under "Long contexts" (600,000 cap, no ratio, no `NCCL_EXTRA`); with the launcher
since 2026-11-17, the 600,000 line of "Launcher defaults".

### The draft's pools as FP8 (793,000 tokens, 2026-11-16 00:34 UTC; `SPARK_DRAFT_KV_FP8=1`; the launcher sets it since 2026-11-17)

SGLang resets the draft's KV to BF16 because FA4, which the draft's block reads its KV with, wants K and Q of one
dtype. With `SPARK_DRAFT_KV_FP8=1` (and an e4m3 target pool) the draft's two pools are e4m3 too, and
`patches/draft_decode.py` is the reader: the image's Triton decode kernel rebuilt at import (the page looked up per
key, a cast after the K/V loads, the causal mask and the bias replaced by the mask of the valid keys; a second
small kernel reduces the splits), static shapes, so it runs inside the draft's CUDA graph. The draft's block is 8
queries a request over the whole prefix and itself, not causal, no bias; a windowed layer arrives as a table of its
window's slots and is the same call. Every other FA4 call on an e4m3 cache goes on as before (logged once a kind: the
target's prefill). A pool token costs 5,734 bytes a node instead of 6,349, so the same bytes hold 793,000 tokens
(windowed pool 39,650): +10.8% on the step before, +32% on a full boot at the 600,000 cap. The target's arithmetic is
untouched; the draft's K and V are rounded when stored, which can move what it proposes (and, through the
confidence gate, a near-tie in the text).

Window of 2026-11-16 00:20 to 01:27 UTC (production stopped for the kernel test, a boot with the statistics, then
the boot that serves):

| | 716K, draft pools BF16 | 793K, draft pools e4m3 |
|---|---|---|
| the reader alone (`bench/draft_decode_test.py`, idle head, 68 checks) | | within 2.3e-3 relative L2 of dense fp64 on the stored values, 3.0e-3 of FA4 on the same values as BF16; a captured graph replays what eager computes; stock's tile of 32 keys kept (64 and 128 no faster) |
| its time against FA4 on BF16, one layer, one call | | windowed layer 26-41 us against 36-37; global layer at 2K keys 61 against 48 us, at 30K 104 against 126, at 200K 525 against 908, at 500K 1,204 against 2,264; 8 requests at 200K 3.5 against 7.1 ms |
| the draft's K and V at the store (`SPARK_DRAFT_KV_STATS=1`, 89,594 context tokens, every rank) | | K rms 1.3-2.2, largest 19; V rms 0.5-5.0, largest 31 (e4m3 holds 448): nothing saturates, nothing non-finite; relative RMS error of the rounding 2.7% on every layer's K and V; V below 2^-10 (rounds to zero) 0.01-0.15% of the values, K up to 4% (layer 2), small values that carry no weight in the error |
| pools as allocated (rank 0) | | all four e4m3: target 793,000 (1.51 + 1.51 GB) and 39,650 (0.38 + 0.38), draft 793,000 (0.19 + 0.19) and 39,650 (0.04 + 0.04) |
| one stream, greedy code (`bench/step_time.py`, against the 716K boot, two boots) | 61.8 tok/s, 58.1 ms a step | 60.2 and 59.0 tok/s; step time x1.001 and x1.007, at equal accept +0.6 ms a step (+1.0%, se 0.5); accepted tokens a step x1.000 (medians, request by request); same text on 6 of 8 |
| 4 / 8 streams | 103-105 / 143 tok/s | 104 and 95 / 142.5 tok/s |
| the target's decode path against its prefill path (`bench/bias_consistency.py`, 4 files, 199 tokens each; the target is not meant to move) | | decode NLL 0.001 to 0.013 below the prefill NLL, no logprob off by more than 0.72 (the size these had in earlier windows) |
| text under concurrency, three runs; reasoning cache | | all passed |
| tool calls (`bench/toolcall_check.py`, 2 trials) | 13 failures: the 10 `tool_choice: none` cases and 3 sampled slips | the same count and kinds; 12 of 12 agent loops pass |
| `bench/prefix_fork.py`, `bench/swa_fill.py` | | cached prefixes as before; 80 of 80 lookups exact with the windowed pool run full |
| 130K context | lookups exact, prefill 2,116 tok/s, decode 69 tok/s | exact cold and warm, 1,984 tok/s, 68 tok/s |
| 500K context | prefill 1,453 tok/s, decode 56-58 tok/s, 1 of 4 lookups answered | 1,445 tok/s, decode 60.2 tok/s; neither lookup answered: both ran out of 4,000 tokens reciting the list (the recital loop; the cold one recites the same stretch of records as the cold one of the 716K boot) |
| soak (`bench/stress.py`, 12 workers, one long context, 12 min) | passed | passed: 50 requests, none with a problem, no failed health check, peak 368K tokens in use |
| head memory | 2.2-3.0 GB available serving, 2.33 GB at the lowest in a 500K prefill | 2.3-3.0 GB available at the stages of the window (2.3 after the 500K prefill and through the soak) |

So: 77,000 more tokens for about 1% of step time on one short stream (two boots agree, each within about one
standard error of zero), nothing at 4 and 8 streams, and a faster draft at long contexts. Hottest node 88 C; no
traceback in a rank log.

Rollback: the relaunch line of the 716,000 pool above (no `SPARK_DRAFT_KV_FP8`, cap 716000). The flag without the
cap is harmless (the same pool tokens in fewer bytes); the cap without the flag asks for more than the bytes hold
and the engine sizes the pool down.

The KV cache is not replicated: the model has 8 KV heads and under TP4 every node holds 2 of them. A pool token
costs 7,578 bytes a node: 4,096 for the target's 8 global layers (FP8), 2,048 for its 40 windowed layers (20,480 a
token, in a pool of 0.1 token slots a token: `--swa-full-tokens-ratio`, 0.1 is the model's default), 1,024 and 410
for the draft's two pools, which stay BF16 unless `SPARK_DRAFT_KV_FP8=1` (the engine resets the draft's KV to the
model dtype because its attention backend wants K and Q of one dtype; `--speculative-draft-kv-cache-dtype` is read
before that reset; with the flag they are 512 and 205, and with the windowed share halved the token costs 5,734).
The pool is sized from the host's `MemAvailable` (unified memory), the least over the ranks, minus
(1 - `--mem-fraction-static`) of the memory before the load: the head, which also runs the API server (1.3 GB),
the detokenizer (1.0 GB) and whatever else runs there, has about 3.8 GB less than the workers at that point and
sizes every rank's pool. One GB there is about 130K tokens.

- Why the pool differs from boot to boot (545K-600K at the 600K cap, 493K-805K without): the kernel's watermark
  boost. `vm.min_free_kbytes` is 1 GB here, the high watermark 1.49 GB, and after a fragmentation event the kernel
  raises the low watermark by up to 150% of that (2.24 GB) until kswapd has caught up; `MemAvailable` subtracts the
  low watermark. `bench/boot_mem.sh` samples of six boots: page cache not counted as available 3.1 GB on the full
  boots, 5.3 GB on the low ones, 4.6 GB on one between; driver memory the same.
- Pool use on the serving boot: 8 streams of `bench/stress.py` with a long context peak at 0.91 of the full pool,
  0.21 of the windowed pool and 32 of the 57 state slots (the conv state every resumable prefix point needs;
  `--max-mamba-cache-size`); a 500K prefill at 0.92 / 0.09 / 0.26. The windowed pool is the slack.
- `bench/prefix_fork.py` (live, about 12K tokens): a request that continues a cached prompt reuses it to the last
  64 tokens; one that parts from it midway reuses it up to the last 4,096-token chunk boundary before the fork
  (6,000 shared: 4,096 cached; 9,000: 8,192; 2,048: none), and up to its own earlier end the second time.
- Not tried: moving clients off the head. A disk
  tier (the engine's HiCache) needs its 10 GiB host reserve patched and a host staging arena out of the same
  memory, and writes one file a token at page size 1; whether it restores the windowed tail, the conv state and the
  draft's KV is not known. KV page sharding across nodes applies to replicated KV only (not this model); FP4 KV is
  not built for this pool.

## Prefill: the glue, the far-bias kernel and MXFP8 off Marlin (2026-11-16)

Three switches for prefill, each a variable of the containers and off in the code unless set. The fleet serves
with all three since 2026-11-16 17:20 UTC, at first through the relaunch environment
(`NCCL_EXTRA="-e NAME=value"`); since 2026-11-17 the launcher sets them (`PREFILL_GLUE`, `BIAS_FAR`, `FI_MXFP8`;
each is switched off by setting it empty). Cold prefill of fresh prompts (`bench/prefill_probe.py`, three prompts of 30,000 tokens, one of
120,000, one of 300,000; one boot a row, the pool settings of the KV section on all of them):

| boot | exact | 30,000 tokens | 120,000 tokens | 300,000 tokens |
|---|---|---|---|---|
| without the switches | | 2,474 tok/s | 2,241 tok/s (53.6 s) | not measured |
| `SPARK_PREFILL_GLUE=epilogue,stream,gate,qkvr,relbias` | yes | 2,756 (+11.4%) | 2,469 (+10.2%, 48.7 s) | 1,961 (153.2 s) |
| glue + `SPARK_FA4_BIAS_FAR=f2` | yes | 2,785 (+12.6%) | 2,519 (+12.4%, 47.7 s) | 2,035 (147.6 s) |
| `SPARK_FI_MXFP8=1` alone | no | 2,747 (+11.0%) | 2,456 (+9.6%, 48.9 s) | not measured |
| glue + MXFP8 | no | 3,095 (+25.1%) | 2,733 (+22.0%, 44.0 s) | not measured |
| all three (the serving boot) | no | 3,110 (+25.7%) | 2,786 (+24.3%, 43.1 s) | 2,223 (135.1 s) |

Prefill tok/s repeats to about 2% between boots of one configuration, so the last two rows are the same speed at
30,000 tokens. Decode is not touched by any of them: only plain prefill batches take the new paths.

The serving line as it was run on 2026-11-16 (any of the three left out of `NCCL_EXTRA` ran stock's path for that
part; without all three it was the line of the KV section; today's equivalents are under "Launcher defaults"):

    CTX=524288 MEMFRAC=0.94 \
      NCCL_EXTRA="-e SPARK_WMARK_BOOST=1 -e SPARK_DRAFT_KV_FP8=1 -e SPARK_PREFILL_GLUE=epilogue,stream,gate,qkvr,relbias -e SPARK_FA4_BIAS_FAR=f2 -e SPARK_FI_MXFP8=1" \
      EXTRA_ARGS="--max-total-tokens 793000 --swa-full-tokens-ratio 0.05 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

### The glue (`patches/prefill_glue.py`, exact)

A 4,096-token chunk spends about a tenth of its time between the big kernels: each of the 48 layers builds its
inputs, projects, biases and closes with launches that read and write rows x 4608 values (or the 100 MB
relative-bias tensor) once more. Each part removes one of those passes and keeps every IEEE operation and its
order. One GB10, ms a call at 4,096 rows, stock against fused (`bench/glue_check.py`, 0 differing values):

| part | what | stock | fused |
|---|---|---|---|
| `epilogue` | the layer's closing convolution and residual add in one tiled kernel (`patches/fused_epilogue.py`) | 1.032 | 0.715 |
| `epilogue`, block-boundary layers | the same without the running partial sum | 0.869 | 0.540 |
| `stream` | attn_res + RMSNorm of a layer's inputs in one kernel that never writes the FP32 stream (`patches/fused_stream.py`); 2 sources | 1.195 | 0.530 |
| `stream`, global layer | 3 sources, 3 queries | 3.070 | 1.057 |
| `gate` | the attention output stays BF16 (stock asks for an FP32 copy, which the SM12x forward makes from its BF16 result) | 0.403 | 0.192 |
| `qkvr` | a global layer projects three sources through the whole qkvr weight and keeps k of the second and v of the third; here those two take only their slice of the weight | 4.309 | 1.927 |
| `relbias` | the relative-position bias written once, already sheared, instead of an einsum and the interface's shear (`patches/fused_relbias.py`) | 1.576 | 0.655 |

- `qkvr` is the one part whose equality depends on a library: cuBLAS chooses its sums by the shape it is given.
  The split product equals the stacked one from 500 rows up and differs at 65 to 384 rows, so the part has a floor
  of 512 rows and compares the two products the first time it sees a row count; a count that differs in one bit is
  left to stock (logged). The other floors: `stream` 512, `epilogue` and `gate` 2,048 rows.
- A boot with `SPARK_PREFILL_GLUE_CHECK=1` (diagnosis windows only: every fused call also runs stock's path)
  counted 0 differing values on every rank over a 9,000, four 30,000 and a 120,000-token prefill. Held-out NLL
  (`bench/nll_texts.py`) equals the reference boot's in every token; one-stream decode gave the same text on 16 of
  16 questions.
- Only plain extend batches outside CUDA graph capture take it. Decode, verification and mixed batches run stock's
  code, and so does any call a kernel does not cover (image tokens, no KV cache); each refusal logs its reason once.

### The far key blocks of a biased attention call (`patches/fa4_bias_far.py`, exact)

A prefill call of a global layer walks every key block of the prefix, and only the last 8 blocks (1,024 keys)
carry a relative bias. Compiled with a bias, SGLang's SM120 forward kernel takes 0.536 ms per 1,000 prefix keys
per 4,096-row chunk for the other blocks, against 0.379 for the same kernel compiled without one, although all it
has to do there is multiply by the softmax scale. `f2` splits the loop over the key blocks: the blocks that carry
a bias run stock's body, the far ones a body compiled without the bias state. Same operations, same order of
blocks. One GPU (`bench/bias_far_window.py`): 0.482 ms, output and log-sum-exp equal to stock's bit for bit in
every case. `f1` (the scale as one whole-tensor store, no loop split) gains nothing (0.542); deleting the multiply
altogether gives 0.484, so the rest of the gap to the no-bias build is not the multiply. On four nodes, on top of
the glue: 120,000 tokens 48.67 -> 47.70 s, 300,000 tokens 153.16 -> 147.61 s, NLL equal per token.

The class is derived from the image's source by exact text replacements, so a changed image fails the boot and not
the first request. The forward kernel cache moves to a directory named after the derived source: under the stock
key a later stock boot would load the derived kernel.

### The MXFP8 expert layers on FlashInfer (`patches/fi_mxfp8.py`, not bit-identical)

Layers 40 to 47 are MXFP8. On SM12x the image runs them as Marlin GEMMs over fake-quantized BF16 activations: about
30 ms a layer a rank for a 4,096-token chunk. FlashInfer's MXFP8 grouped GEMM runs e4m3 x e4m3 on GB10's FP8
tensor cores in about 17 ms for the same two GEMMs, with the checkpoint's weights and the activations quantized
block by block with the model's own rule, so the operands are the same numbers. The GEMM adds the same products in
another order before the BF16 rounding, which is the whole difference: held-out NLL 0.9840 against 0.9840 (11,542
paired tokens, difference 0.0000 +- 0.0002), mean absolute difference per token 0.0064, median 0.0006, above 0.1
in 0.5% of the tokens, above 1 in none. Nothing in the engine calls that GEMM on this path, so
`bench/fi_mxfp8_probe.py` pins the operand layout and `bench/fi_mxfp8_test.py` compares the layer with the Marlin
one. Needs the sharded MXFP8 layers (`TP_SHARD=half,fp8`, the default). Decode steps are within boot-to-boot
noise (Marlin is bound by the weight reads there).

### The serving boot (all three, 2026-11-16 17:20 UTC)

- All four ranks log the five glue parts, the derived forward kernel and the FlashInfer MXFP8 layers; no part fell
  back to stock.
- NLL equals the glue + MXFP8 window boot's in every token (0.9840), so `f2` is exact on four nodes too.
- Tool-call suite: 252 right, 20 soft, 11 failed of 283 (the 10 `tool_choice: none` cases and one sampled slip).
  No case of the temperature-0 trial changed its verdict against two runs of the suite on the boot without the
  switches. Those two runs gave 250 / 20 / 13 and 252 / 17 / 14 with 9 verdicts moved on the sampled trial: at four
  requests in parallel the totals move by one to three between any two runs, so compare cases, not totals.
- `bench/stress.py`: 21 requests in 450 s, 8 workers, one 210,000-token context, no problem.
  `bench/concurrency_check.py` clean after the boot and after the load; `bench/reasoning_cache_check.py` clean.
  Hottest node 87 degrees C.

### Tried and dropped, and what is left

- A 128-row query tile for the global layers' forward kernel: exact, 1.74 to 1.80 times slower.
- Left in a chunk: the all-reduces, about 309 ms of a 4,096-token chunk (19% before the switches). The 40 amax
  exchanges of the sharded experts are among them; leaving them out changes the target's operands and was not
  done. Running the grouped GEMMs on the routed experts alone (no empty groups) is worth under 1% and needs a
  change inside FlashInfer.

## Reasoning effort: thinking tokens by level (2026-11-14)

The effort is a number in the prompt (`Thinking effort level: x`), not a budget. The server maps the names: `none`
0.0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9 (also what a request without an effort gets), `xhigh` and
`max` 0.99; chat completions also take a float from 0 to 0.99. `/v1/messages` takes `output_config.effort` with the
same names; `thinking.budget_tokens` is accepted and not enforced, so `max_tokens` is the only hard cap.

Thinking tokens per request, temperature 0 (`bench/effort_tokens.py`, production boot of 19:20; the 30K rows are
`bench/recital_rate.py --context code --tokens 30000 --effort x`; median / mean / largest, and answers right):

| level | 16 small tasks, 2 runs each | 6 hard tasks, 3 runs each | 30K of source, 6 lookups |
|---|---|---|---|
| `none` | 0 / 13 / 174, 26 of 32 | 0 / 366 / 2,333, 15 of 18 | 373 / 523 / 1,488, 6 of 6 |
| `minimal` | 92 / 98 / 313, 32 of 32 | 325 / 465 / 1,734, 18 of 18 | 287 / 340 / 676, 6 of 6 |
| `low` | 86 / 126 / 378, 32 of 32 | 410 / 603 / 1,987, 17 of 18 | 261 / 373 / 885, 6 of 6 |
| 0.5 | 85 / 150 / 752, 32 of 32 | 469 / 695 / 2,773, 18 of 18 | 399 / 512 / 1,245, 6 of 6 |
| `medium` | 89 / 164 / 1,066, 32 of 32 | 503 / 900 / 2,969, 18 of 18 | 403 / 574 / 1,483, 6 of 6 |
| `high` | 92 / 166 / 851, 32 of 32 | 734 / 917 / 2,448, 18 of 18 | 355 / 1,036 / 3,850, 6 of 6 |
| `max` | 93 / 228 / 1,657, 32 of 32 | 589 / 1,038 / 3,160, 18 of 18 | 432 / 790 / 2,667, 6 of 6 |

- The task sets the amount more than the level does. A one-word fact takes 20-53 tokens at every level above `none`, tracing a
  recursive program 1,400-3,200 at every level. Between `minimal` and `max` the mean on the hard tasks about doubles.
- Where the level shows most is writing code: the three small functions took 160 (median) at `minimal`, 288 at
  `low`, 388 at `medium`, 425 at `high`, 792 at `max`; an expression evaluator 484, 346, 1,437, 1,246, 1,630.
- Agent turns that only have to make the next call think very little at any level: 7 tokens (median) for the simple
  turns; finding a missing line in a 50-line module from a failing test took 19 up to 0.5, 56 at `medium`, 75 at
  `high`, 82 at `max`, right 3 of 3 at each.
- `none` is the model deciding for itself, not an off switch: no thinking on most tasks (and then 6 of 32 small and
  3 of 18 hard answers wrong: a least common multiple, a weekday, a program's output, a count), but 1,900-2,333 tokens on the program
  trace and a median of 373 over the 30K context.
- At 130K of records the right lookups used 372 (median) output tokens at `low`, 444 at `medium`, 482 at `high`
  (see the reasoning loop above for the ones that were not right).
- One sample of 6 to 32 requests a cell; a run of the same request differs by tens of percent (the three runs of
  the evaluator at `low`: 305, 346, 1,110).

## Measured (2026-11-13)

| | |
|---|---|
| weights per rank | 92-93 GB of ~108 GB visible; BF16 KV pool 247K-395K tokens depending on what the head has free at boot (it is the tightest rank; 247,356 on the 17:07 boot is below the 262,144 context) |
| decode, 1 stream | 41-47 tok/s with DSpark on code (accept length 3.7-4.0, ~12 steps/s); 30-35 while one node's GPU ran at 695 MHz |
| GPU clock | one node ran at 695 MHz under load until a reboot on 2026-11-13; since then 2,385-2,392 MHz (others 2,150-2,560). The slowest GPU paces the fleet: check its clock first |
| prefill | ~1,980 tok/s (25K-token prompt); ~1,160 with that node at 695 MHz; ~440 before FI_MOE |
| decode/verify attention kernel, per call | windowed layer ~19 us; global layer, 8 verify rows: 145 us at 30K context, ~900 us at 200K; SGLang's tile lookup was wrong on 40 of 40 non-consecutive layouts (dense fp64 rel-L2 0.06-0.96), the page-per-key kernel 1.3e-4 to 3.9e-4 |
| concurrency check, fixed kernel | 8 streams together and staggered: all sane (d4 0.85-0.99); single streams token-identical to the reference with prefills landing mid-decode and after 63K tokens of pool churn |
| decode against streams, fixed kernel, BF16 KV | code questions, 400 tokens each: 1 stream 44.7 tok/s, 2: 65.9 together, 4: 85.3, 8: 107.1 (13.4 each); with 30K context each: 1: 34.7, 4: 68.2; 120K context: 36.5. ~11-11.6 steps/s whatever the context, the rest is accept length (3.0-4.1) |
| FP8 KV (`--kv-cache-dtype fp8_e4m3`) | pool 712,673 tokens (BF16 on the boots before: 351K-368K). Streams 1/2/4/8: 43.7 / 66.5 / 84.9 / 105.0 tok/s; 30K context 39.9 (4 streams 73.4); 120K 40.1. Prefill 2,470 tok/s (15.6K words), 2,210 (21.6K code), 2,200 (100K). Concurrency check ALL OK; prefill NLL of four texts within 0.02 of BF16; decode vs prefill logprobs mean 0.013-0.058 (BF16 0.018-0.056). Unit: e4m3 kernel within 7.6e-7 of the BF16 kernel on the same values, global layers 0.5-0.7x the time, windowed layers at batch 8 35 us against 19. Served 7 h on 2026-11-13 without incident |
| batch noise | a greedy stream leaves its single-stream text after 8-100 tokens once another request shares its decode batch (the watchdog's probe is enough): chosen-token logprob moves by 0.01-0.06 on average, the same as decode against prefill. Rounding in the BF16 attention kernels amplified by 4-bit activations, not a wrong read |
| boot | ~5 min with a warm JIT cache (`~/inkling2-cache` per node), ~17 min cold |
