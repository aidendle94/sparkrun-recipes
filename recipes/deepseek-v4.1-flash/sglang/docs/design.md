# Design notes

What each piece of the overlay does, and why the stock SGLang image needs it on a DGX Spark. The measurements the
decisions rest on are in [results.md](results.md); the pinned base image and its hook points are in
[upstream.md](upstream.md).

## Vocabulary

- **Prefill** is the forward pass that ingests a prompt; **decode** produces one token (one DSpark block) per step.
  SGLang feeds a prompt in **chunks** of `--chunked-prefill-size` tokens (2,048 here) and calls the batch that carries
  a chunk an **extend** batch. "Extend" and "prefill chunk" mean the same thing in this document and in the code.
- **MLA** is DeepSeek's multi-head latent attention, whose KV cache holds a compressed latent per token. V4.1 makes it
  sparse: an **indexer** scores the query against the cached tokens and the attention kernel only reads the top-k.
  V4.1 has three indexer sources, named by their compression ratio: **ratio-4** (the classic "c4" source, one entry per
  four tokens) and the two low-ratio sources **ratio-2** and **ratio-1**. SGLang keeps a separate KV source for each.
- **Groups.** SGLang wraps every process group in a `GroupCoordinator` with its own NCCL communicator: **TP** (tensor
  parallel, the four ranks that split every weight and all-reduce after each layer), **EP** (expert parallel, the
  experts split over the same four ranks; SGLang's `moe_ep` group), **DCP** (decode context parallel, a group that can
  split the KV cache for decode) and **PP** (pipeline parallel, unused here).
- **DSpark** is DeepSeek's speculative decoding for V4: a draft of `block_size` tokens (5 here) is verified in one
  forward pass, so a decode step runs six query rows through the model. Decode and verify are captured CUDA graphs.
- **Engram** is the model's hash-table embedding: layers 1 and 14 each carry a table of N ≈ 384 million rows. A token's
  n-gram hashes are row ids; a row is 256 fp8 (e4m3) bytes plus 8 e8m0 block scales, 264 bytes, so a table is about
  101 GB (94 GiB). A lookup gathers rows, dequantizes them to bf16 and, when the rows are sharded over TP, sums the
  ranks' contributions with an all-reduce (a rank writes zeros for rows it does not own).

Units: GiB (2^30 bytes) for memory figures, GB (10^9 bytes) for files and disks.

## Why the stock image does not fit on a Spark

A DGX Spark has one 128 GB pool of memory (121.7 GiB visible) shared by the CPU and the GPU. At TP4 the weights
take about 77 GiB per rank. Stock SGLang gives each rank one quarter of each Engram table, rows
[N·r/4, N·(r+1)/4), and keeps it either in device memory (about 24 GiB per layer per rank, two layers) or, with
`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, pinned in host memory. On a Spark both live in the same pool, so either
choice costs the 47 GiB that the KV cache and the prefill transients need. A decode step touches of the order of a
hundred rows per layer per rank and a prefill chunk tens of thousands; the tables are read sparsely and rarely, which
is what NVMe is good at.

## Engram rows from NVMe (`overlay/engram_store.py`, `overlay/engram_rows.c`)

The hook replaces how `EngramEmbedding` obtains rows and nothing else: SGLang's hasher, its Triton dequantize
kernel, the TP all-reduce that reassembles unowned rows, the gate and the projections all stay upstream.

**Row source.** `SPARK_ENGRAM_DIR` names a directory holding `model.safetensors.index.json` and the shard files that
contain the two tables. It is either a node-local sparse copy made by `tools/engram_local.py`, whose manifest
`engram-local.json` gives this rank's global row range per layer, or the full checkpoint snapshot on the node that
holds it, with the ranges in `SPARK_ENGRAM_ROWS` (`1:lo:hi,14:lo:hi`). The module reads the safetensors header
itself (an 8-byte little-endian length, then JSON) to find each tensor's dtype, shape and byte offsets; weight and
scale must sit in the same shard.

**Validation.** At construction the module checks that the shard's row count equals the model's `num_embeddings`,
that `0 ≤ lo ≤ hi ≤ N`, and, by exchanging every rank's range over the TP group's CPU (gloo) communicator, that
the four ranges tile `[0, N)` with no gap and no overlap. A mismatch raises with every rank's range in the message,
because the all-reduce would silently produce wrong embeddings otherwise. The table parameters are created with the
right dtypes but zero rows, and the loader SGLang calls for them validates the checkpoint tensor's shape and copies
nothing: no path allocates a table.

**The gather, as graph nodes.** A lookup with `k` ids runs four stream operations, all of which CUDA-graph capture
records and replays without Python:

1. an asynchronous copy of the ids from the device into a pinned host buffer;
2. a host-function node (`cudaLaunchHostFunc`) that calls the C library's gather with a work descriptor pointing at
   that buffer and at two pinned output buffers; for each id in the rank's range the library copies the 256-byte row
   and its 8 scale bytes from a read-only memory map of the shard, and writes zeros for ids outside the range;
3. an asynchronous copy of the staged rows into device staging buffers;
4. SGLang's own Triton `engram_gather`, run by position over the staged rows, which dequantizes to bf16 into the
   output that `_lookup` then all-reduces.

Every buffer, the work descriptors and the device staging are allocated once per layer, sized for
`SPARK_ENGRAM_MAX_IDS` ids (262,144 by default; the launcher sets it from the chunk size and the request limit),
and kept alive for the life of the process, because a captured graph keeps pointing at them. A lookup larger than
the capacity raises; an empty lookup returns an empty result without touching the library. The device staging is
created on the first eager call, before any capture; a first call under capture is an error with a clear message.
Pinned buffers are created with an explicit CPU device, because the model is built under a CUDA default-device
context. The CUDA runtime library is loaded through ctypes for the host-function launch.

**The C library.** The shard is memory-mapped read-only and advised `MADV_RANDOM`, so a cold row faults in exactly
its 4 KiB page; readahead would drag in the holes of a sparse file. Before the copies start, a `posix_fadvise(WILLNEED)`
pass puts every owned row's page in flight, and a persistent pool of `SPARK_ENGRAM_THREADS` workers (64) copies the
rows so that faults overlap. The pool exists because spawning threads per call cost more than the faults for a
decode-sized gather. The library counts calls, owned and zeroed rows and gather nanoseconds; a daemon thread logs one
line per layer every `SPARK_ENGRAM_STATS_SECONDS` (60; 0 disables). The library makes no CUDA calls and allocates
nothing on the hot path.

**The sparse copy.** `tools/engram_local.py` copies the shard header and the rank's rows into a local file of the
same name and size, at the same byte offsets, and leaves the rest as holes, so the store opens the copy exactly as it
would open the checkpoint. Each rank copies its quarter of each of the two tables, about 24 GiB (25 GB) per table and
48 GiB (50 GB) per node, rate-limited and spot-checked at the end (the shard header plus 2,002 rows per range). The manifest carries whatever range was
used; any tiling of the table works, and `tools/engram_partition.py` prints SGLang's own split for a TP size.

**Prefetch interface.** The module exposes the loaded library and the list of `(layer, store)` handles, and after
SGLang builds its `EngramHasher` it records CPU copies of the hash tables so the scheduler-side prefetch below can
hash tokens without the GPU.

## Next-chunk prefetch (`overlay/engram_prefetch.py`)

A chunk's hash ids depend only on its tokens and their three predecessors, and the scheduler holds every request's
whole prompt. When the scheduler launches an extend batch, the hook hashes the *following* chunk of every request
on the CPU with SGLang's own hash arithmetic and hands the row ids to the store's prefetch entry point, which does a
`posix_fadvise(WILLNEED)` per owned row and no copy, from a background thread. By the time the next chunk's
host-function node runs, its rows are in the page cache and the gather only copies. The hook is best effort: an
error disables it with one warning. `SPARK_ENGRAM_PREFETCH=0` turns it off.

## MXFP8 dense projections (`overlay/mxfp8_kernel.py`)

V4.1's dense projections are FP8 with 32×32 block scales. With `--fp8-gemm-backend flashinfer_cutlass` SGLang sends
them through its `flashinfer_mxfp8_blockscaled_linear` wrapper to FlashInfer's `mm_mxfp8` with `backend="cutlass"`,
a kernel with a 128-row tile: a six-token decode step is padded to 128 rows and the whole weight streams through for
those six rows. FlashInfer also ships `b12x`, a warp-level SM12x kernel with small row tiles. The hook makes the call
use the backend named by `SPARK_MXFP8_BACKEND` (default `b12x`; `cutlass` or empty restores the stock behaviour;
other names are passed through). Not every weight shape is accepted by the chosen backend: `b12x` needs K divisible by
128 and refuses this checkpoint's K = 576 projections (the ones that read the 512 + 64 wide compressed KV latent).
FlashInfer raises before launching anything, so the first call of every (N, K) weight shape is a probe: on success
the shape is pinned to the wanted backend, on refusal it is pinned to the stock backend with one warning and the
stock kernel answers that same call. The first call therefore never fails the server, and the probe costs one Python
exception per shape over the server's life. Calls that arrive with a `trtllm` backend carry differently shuffled
scales that no other backend reads and are left alone.

## Indexer plan on SM12x (`overlay/indexer_schedule.py`)

SGLang builds a `PagedIndexerMetadata` per indexer source; its `__post_init__` decides whether to call DeepGEMM's
`get_paged_mqa_logits_metadata` planner, and honours a `force_deep_gemm_metadata` field. On SM120/121 SGLang routes
the classic ratio-4 indexer through a torch implementation and, unless the field is set, skips the planner and
leaves the plan empty. V4.1's ratio-1 and ratio-2 sources, however, call DeepGEMM's `fp8_fp4_paged_mqa_logits`,
which reads the plan; without it CUDA-graph capture of the decode step fails with a `schedule_meta` mismatch. The
hook wraps `__post_init__` so that, on SM120 only, metadata for `compress_ratio` 1 or 2 sets the field before the
stock code runs; the stock code then plans as it would for its own FP4 path (the field also selects DeepGEMM's
planner over SGLang's JIT one, whose encoding the SM120 kernel does not accept). Ratio-4 metadata and every other
architecture are untouched.

## Ratio-2 KV pages for the sparse prefill (`overlay/sm120_prefill_pages.py`)

FlashInfer's SM12x sparse-MLA prefill kernel takes 64-token pages. SGLang re-pages the primary KV source from the
pool's 256-token pages with a Triton page-split kernel, but passes the ratio-2 source through with its 128-token
pages, and the kernel rejects the configuration. The hook re-pages that source the same way, with SGLang's own
page-mark and page-split kernels, into its own persistent buffers so the two conversions never share scratch.
Token-level indices survive the split unchanged: token `o` of page `p` becomes token `o mod 64` of page
`2p + o div 64`, the same flat position.

## Allocator flush after long prefill chunks (`overlay/prefill_flush.py`)

Every prefill chunk scores its query rows against the whole prefix, so the indexer's transient buffers grow with the
prefix. PyTorch's caching allocator cannot reuse a smaller freed block for the next chunk's larger request, so the
memory it holds reserved grows roughly with the square of the prompt length. On a Spark that reserved memory is host
memory: a 128K-token prompt took all four nodes to zero free within a minute; the OOM guard killed two ranks and the
other two hung in the TP all-reduce. `expandable_segments` would let the allocator grow blocks in place, but it
breaks the sparse prefill kernels on this stack. The hook runs after each `ModelRunner.forward` and, when the batch
is a plain prefill batch (extend, mixed or split prefill), its longest sequence is at least
`SPARK_PREFILL_FLUSH_TOKENS` tokens (8,192; 0 disables) and no graph is being captured, calls
`torch.cuda.empty_cache()`. The peak is then one chunk's live set. A DSpark verify step is also an "extend" in
SGLang's terms but runs once per decode token and is left alone. The longest sequence is read from the batch's
host-side lengths, so the check adds no device synchronization on the common path; the flush itself does synchronize
(it frees device memory), which short prompts never pay. A failure inside the hook is logged and never propagates.
Decode graphs keep their private memory pools and are not affected.

## One-shot RDMA collectives (`overlay/roce_collectives.py`)

A decode step on a four-node TP group issues about eighty all-reduces of 48–400 KB plus the logits all-gather. Each
is a NCCL ring kernel with its own launch and rendezvous; on the author's fabric a 48 KB all-reduce costs about
100 µs in graph replay. The b12x `comm/roce` runtime (Apache-2.0, vendored at revision `b58f34ea`) does the same
collective as one RDMA write per peer into pinned host memory over both ConnectX-7 functions, replayable inside CUDA
graphs: about 21 µs. Above roughly 1.2 MB NCCL wins again, so the hook routes all-reduces up to `SPARK_ROCE_AR_MAX`
(1 MB) and all-gather shards up to `SPARK_ROCE_AG_MAX` (16 MB); prefill's 64 MB all-reduces and the DCP, EP and PP
groups stay on NCCL.

The hook attaches a runtime to the `tp` GroupCoordinator only. Construction first votes over the CPU (gloo) group,
so a rank that cannot take part (missing package, unsupported device, different limits) disables the route on every
rank instead of leaving its peers waiting in the runtime's setup exchange. `all_reduce` and `all_gather` go to the
runtime when it accepts the tensor (dtype, contiguity and size; the decision is the same on every rank), else the
stock path runs. `graph_capture` prepares the runtime and pins the capture stream. The runtime is fail-stop: a
stalled peer poisons it and `check_health()` raises; the check runs after every model forward, so a poisoned step
never reaches a client, and the watchdog relaunches the fleet. The runtime needs the RoCE-v2 GID index of the fabric
address, which differs between nodes; the launcher probes it on each node and passes it in `B12X_ROCE_GID_INDEX`.
The spin limit is 300 million polls, about five minutes.

## Served-model aliases (`overlay/served_aliases.py`)

SGLang serves one `--served-model-name`. Its chat and completion handlers ignore the requested model id, but
`/v1/models` lists one name and `/v1/models/{id}` returns 404 for any other, which breaks clients that check the
listing first. The hook rebuilds those two routes so every name in `SPARK_SERVED_ALIASES` is listed and retrievable,
with the real served name as root. Without aliases it does nothing.

## Installing the hooks (`overlay/sitecustomize.py`)

The image puts `/opt/dsv41-spark/overlay` on `PYTHONPATH`, so `sitecustomize` runs at the start of every SGLang
process: the HTTP server, the tokenizer manager, the scheduler and the TP workers. When `SPARK_ENGRAM_DIR` is set it
registers a post-import hook for each target module through importlib's finder machinery: the real module spec is
resolved, its loader's `exec_module` is wrapped, and the hook's `install(module)` runs exactly once right after the
module has executed, whether it was imported by name or through `from ... import`. A hook that raises is re-raised
as an overlay error chained to the original, so the full traceback reaches the log and the process stops, because a
half-installed overlay would serve wrong results (a stock MXFP8 kernel is merely slow; a missing Engram gather or an
unplanned indexer answers with wrong tokens or dies at capture). When the variable is unset nothing is registered and
the image runs stock SGLang; in both cases the interpreter's own `sitecustomize`, which this file shadows, still runs.

| SGLang module | hook |
|---|---|
| `sglang.srt.layers.engram` | `engram_store.install` |
| `sglang.srt.layers.quantization.fp8_utils` | `mxfp8_kernel.install` |
| `sglang.srt.layers.attention.dsv4.metadata` | `indexer_schedule.install` |
| `sglang.kernels.ops.attention.flash_mla_sm120` | `sm120_prefill_pages.install` |
| `sglang.srt.model_executor.model_runner` | `prefill_flush.install`, `roce_collectives.install_health` |
| `sglang.srt.distributed.parallel_state` | `roce_collectives.install` |
| `sglang.srt.managers.scheduler` | `engram_prefetch.install` |
| `sglang.srt.entrypoints.http_server` | `served_aliases.install` |

## What stays upstream

Attention (the `dsv4` backend and its indexers), the FlashInfer MXFP4 MoE runner, DSpark draft and verify, the
Engram hasher and gate, the tool-call and reasoning parsers and the OpenAI server are SGLang's, as shipped in the
base image. The overlay depends on a small set of hook points, listed in [upstream.md](upstream.md); a newer base
image can be dropped in by changing the pinned digest as long as those survive, and `tests/test_hooks_cpu.py`
checks them on the CPU.

## The staged alternative (`engram_staged.py`)

`SPARK_ENGRAM_MODE=staged` swaps the Engram implementation for a port of the author's vLLM design: a hook on
`ModelRunner.forward` hashes the batch's tokens with the model's own hash kernel (without committing the hasher's
history, which the model's own call does), copies the ids to the host once, gathers both layers' rows from the
memory-mapped shards with a thread pool into pinned staging, and issues the H2D copies; the embedding's `_owned_rows`
then reads the staged rows by position through SGLang's Triton gather. Nothing runs inside the graph, so no C
library is needed. The trade: every prefill chunk pays a sync and a Python gather with the GPU idle (14–20 % slower
prefill on unique text), while decode is 8–9 % faster than two in-graph host nodes. Measurements in `docs/results.md`.
