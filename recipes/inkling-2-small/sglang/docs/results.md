# Inkling-2-Small on four DGX Sparks: lab notes

The working notes behind this recipe, in the order things were built and measured (2026-11-12 to 2026-11-16): what
the GB10 hooks do, the bug that produced garbage under concurrency, every decode lever with its numbers, long
contexts and multi-turn history, the reasoning loop on long lists, the KV cache, reasoning effort, and the first
day's measurements. "Production" and "the fleet" mean the author's four Sparks; times are UTC. Each number is one
boot on that fleet unless the text says otherwise. The launch scripts are in `launch/`, the tools in `bench/`, the
hooks in `patches/`. The summary and the way to run it are in the [README](../README.md).

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
step, 3.8 verify rows), 21.2 per stream at four streams. Chat requests without a `temperature` are sampled at 0.7
(the server's default; the model ships no generation config).

`bench/nll_texts.py` scores held-out text (58 spans of 199 tokens, windows only: it asks for input logprobs) and
pairs a boot against a reference boot token by token; stock is 0.9828.

## Long contexts and multi-turn history (2026-11-14)

Serving 524,288 tokens of context since 2026-11-14 06:49 UTC through the environment of the relaunch (the launcher's
defaults are still 262,144 and an automatic pool):

    CTX=524288 MEMFRAC=0.94 EXTRA_ARGS="--max-total-tokens 600000 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

The pool is fixed at 600,000 tokens. `--prefill-decode-interval 2` is a scheduling setting added for the test boot,
not a decided default. Measured on that boot (`bench/longctx.py`, `bench/stress.py`, `bench/longctx_edges.py`):

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
answers that came were as long as ours (median 400 to 500 output tokens against 464).

## KV cache: what a token costs and what grows the pool (2026-11-15/16)

Serving since 2026-11-16 00:34 UTC, through the environment of the relaunch (not the launcher's defaults):

    CTX=524288 MEMFRAC=0.94 NCCL_EXTRA="-e SPARK_WMARK_BOOST=1 -e SPARK_DRAFT_KV_FP8=1" \
      EXTRA_ARGS="--max-total-tokens 793000 --swa-full-tokens-ratio 0.05 --prefill-decode-interval 2" bash launch/relaunch-inkling2.sh

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

Rollback: the relaunch line under "Long contexts" (600,000 cap, no ratio, no `NCCL_EXTRA`).

### The draft's pools as FP8 (793,000 tokens, 2026-11-16 00:34 UTC; `SPARK_DRAFT_KV_FP8=1`, off unless set)

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
