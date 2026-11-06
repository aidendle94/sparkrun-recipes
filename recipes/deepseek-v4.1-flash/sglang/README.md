# DeepSeek-V4.1-Flash on four DGX Sparks with SGLang

This repository serves DeepSeek-V4.1-Flash on four NVIDIA DGX Spark nodes with SGLang, behind one OpenAI-compatible
endpoint. Both checkpoints work unchanged: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
and DeepSeek's 2026-10-17 update [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(same config and tensor list, new weights), which is what the author's fleet serves. The engine is the SGLang team's
own nightly image of SGLang `main` (`lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3`, pinned by digest),
unmodified. On top of it sits an MIT-licensed overlay of Python import hooks and one C library that supply what a
Spark needs and the stock image does not have: the model's Engram tables read from node-local NVMe inside the CUDA
graph (they do not fit next to the weights in 128 GB of unified memory), kernels and schedules that work on the
GB10's SM12x architecture, and RDMA collectives that make a four-node tensor-parallel decode step fast. Nothing
model-specific is forked; the same image still runs stock SGLang when the overlay's switch is unset.

The published image is `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8` (Docker Hub, arm64). The previous
generation, `production-1.7.1`, was built on the SGLang team's `dev-dsv41` branch image; it stays on Docker Hub and is
what the prompt-lookup wide mode needs (see below). Use the launcher of the same generation as the image: the hooks
of one do not bind to the other's SGLang.

Glossary. **Engram**: the model's two hash-table embedding layers (layers 1 and 14), each an fp8 table of about
101 GB looked up by n-gram hash. **DSpark**: DeepSeek's speculative decoding, where a draft block of tokens is verified
in one forward pass (block size 5 here). **TP4/EP4**: tensor parallel over the four nodes, with the experts split
four ways. **MXFP4/MXFP8**: 4- and 8-bit floating point with shared block scales; the experts are MXFP4, the dense
projections 32×32-block FP8. **RoCE**: RDMA over Converged Ethernet, direct memory writes between nodes over the
Sparks' ConnectX-7 ports. **SM12x (GB10)**: the Spark's GPU architecture (compute capability 12.1), which needs
different kernels and schedules from the datacenter Blackwell parts.

## What you get

Measured on four Sparks over a switched RoCE fabric with the production profile (`launch/production.sh`), serving
DeepSeek-V4.1-Flash-1017. The production-1.8 column is from its test windows and deploy on 2026-10-18; the
production-1.7.1 column is the same fleet on the previous image. Rows marked * were last measured on production-1.2
(2026-09-23) and not re-run since. Conditions and boot-by-boot numbers are in [docs/results.md](docs/results.md).

| what | production-1.8 | production-1.7.1 |
|---|---|---|
| time to first token, 32K / 128K needle of unique text | 6.6–6.7 s / 34.1–39 s (~4,900 / ~3,400–3,850 tok/s) | 8.0 s / 45.0 s |
| prefill, repetitive text | 4,100–5,300 tok/s | ~3,400 tok/s |
| 12 long unique prefills back to back with decode streams (`bench/long_prefill_stress.py`) | 140 s | 167 s |
| decode, one stream, counting / code / prose (tok/s) | 114 / 96–100 / 35–36 | ~108 / 93 / 37 |
| decode, 10-turn coding conversation over ~62K tokens of source | 65.5 tok/s | 65.9 tok/s |
| aggregate at 1 / 8 / 16 concurrent streams, fixed prompt set (tok/s) * | 57 / 211 / 337 (64 / 31 / 26 per stream) | |
| KV pool, auto-sized at each boot | 6.30 M tokens (2026-10-21 boot) | 5.2–6.2 M tokens |
| context length | 524,288 tokens | 524,288 tokens |
| boot to healthy | about 9–10 min (542 s at the deploy) | about 10 min |
| request features | images, tool calls, thinking (on by default at maximum effort, off per request) | the same |

Against the author's previous vLLM deployment on the same nodes (2026-09-17, the original checkpoint), this stack
was already 30–45 % faster at prefill with decode level or ahead (110 / 94 / 37 against 93 / 67 / 31 tok/s) and a
23 % larger KV pool; production-1.8's gain over 1.7.1 is all prefill, because a decode step is set by the MoE's memory
bandwidth (see [docs/results.md](docs/results.md)).

Prefill tok/s is prompt tokens divided by time to first token on a needle prompt that is salted per run
(`NEEDLE_SALT`, which `window-sgl.sh` sets), so the prefix cache cannot hit. Decode is greedy with thinking off. Each cell is one run on one boot unless the results page
says otherwise.

## What you need

- Four DGX Spark nodes (GB10, 128 GB unified memory, arm64). One of them is rank 0: you run every script there.
- A RoCE fabric between the nodes, switched or direct, with the RDMA devices visible in `/sys/class/infiniband` on
  every node. The launcher uses every device you list (two ConnectX-7 functions per node on the author's fabric).
- The checkpoint, about 510 GB, on one node and exported read-only over NFS to the other three. Mount the HuggingFace
  repo directory (`models--deepseek-ai--DeepSeek-V4.1-Flash-1017`, or `...--DeepSeek-V4.1-Flash` for the original),
  not the snapshot inside it, so the blob symlinks resolve on the clients. A download on a node that is already
  serving should run under a memory limit (for example `systemd-run --user -p MemoryMax=4G hf download ...`): on GB10
  the page cache and the downloader compete with the GPU for the same memory.
- On every node: about 50 GB of free NVMe for that rank's Engram row copy (the node that holds the checkpoint reads
  its rows from the shards instead and needs none) and about 34 GB for the image.
- Docker with the NVIDIA container runtime; the launching user in the `docker` group on every node; key-based ssh
  from rank 0 to the other three; permission to pass `/dev/infiniband` into a container and to lift the memlock limit
  (the launcher passes `--device /dev/infiniband`, `--cap-add IPC_LOCK` and `--ulimit memlock=-1`).
- Not part of the stack: an out-of-memory guard. The author runs earlyoom on every node as site practice. The
  container starts with `--oom-score-adj 500`, so when the kernel has to kill something it takes the engine first.
- On rank 0: Python 3 (standard library only, for `tools/` and `bench/`), `ip`, `ssh`, `curl`, `flock` and `nvidia-smi`;
  Python 3 on every node that runs `tools/engram_local.py`.

## Run it

```bash
# 0. The checkpoint, on the node that will export it over NFS.
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017

# 1. The image, on every node (arm64). Pull the published build...
docker pull aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8
#    ...or build it on each node from this repository, then put IMAGE=sglang-dsv41-spark:local in launch/fleet.env
#    so the launcher, relaunch.sh and the watchdog all use it.
docker build -t sglang-dsv41-spark:local .

# 2. The site file: nodes, users, devices, paths. Every line of the example says how to find its value.
cp launch/fleet.env.example launch/fleet.env

# 3. Each rank's Engram row ranges (the split used by SGLang, for TP4).
python3 tools/engram_partition.py /path/to/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<revision>

# 4. On every node that does not hold the checkpoint: copy that rank's rows to local NVMe (about 50 GB,
#    rate-limited to 600 MB/s by default with --mbps=, then spot-checked against the source: the shard header and
#    2,002 rows per range, first, last and 2,000 random). The rows differ between the original checkpoint and
#    1017, so a copy belongs to one revision: rebuild it, under a new directory name, when you change weights.
python3 tools/engram_local.py /path/to/snapshot ~/dsv41-engram-local 1:<lo>:<hi> 14:<lo>:<hi>
#    Optional CPU check, on a node that now has a row copy: the hooks bind to this image's SGLang (no GPU).
#    It must end with HOOKS CPU TEST PASS.
docker run --rm -v ~/dsv41-engram-local:/engram-local:ro -e SPARK_ENGRAM_DIR=/engram-local \
  --entrypoint python3 aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8 /opt/dsv41-spark/tests/test_hooks_cpu.py

# 5. Print the four docker run commands without starting anything. This already needs ssh to every rank:
#    the RoCE-v2 GID index is probed on each node.
launch/production.sh --dry-run

# 6. Launch: disarms the watchdog, removes any previous sgldsv41 container on the four nodes (and runs
#    PREVIOUS_STACK_STOP from fleet.env if you set it), starts the four ranks, waits for /health, arms the watchdog.
launch/relaunch.sh

# 7. First request (port 8210 on rank 0).
curl -s http://<rank-0>:8210/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Is 91 prime?"}],"max_tokens":200}'
```

What to look for in `docker logs sgldsv41` on rank 0 (the container is named `sgldsv41` on every node):

| line | meaning |
|---|---|
| `overlay: 20 import hooks armed`, then `overlay hook <module>.install applied to <sglang module>` per hook | `sitecustomize` saw `SPARK_ENGRAM_DIR` and each hook patched its module; a hook that fails raises and the process dies here |
| `engram_store installed: Engram rows of layers [1, 14] come from <dir> through libengram_rows.so` at import, then, as the model is built, `engram layer 1: rank r/4 owns rows [lo, hi) of N, shard <file>, 64 gather threads, <capacity> ids per lookup` (and the same for layer 14) | the tables are served from NVMe; the four ranks' ranges were checked to tile the table |
| `engram layer 1: <n> lookups, <n> owned rows, <n> zeroed rows, <n> us mean gather per lookup` every 60 s | the gather statistics; the microseconds are the row store's own timer |
| `MXFP8 dense GEMMs routed to FlashInfer backend b12x (per-shape probe, cutlass fallback)`, then `MXFP8 dense GEMMs: FlashInfer b12x selected for N=... K=...` per shape on first use, and for the shared expert `N=5120 K=576 with K padded to 640 (first call max \|diff\| 0 against cutlass, ...)` | every dense shape runs on b12x; the padded one was checked against the stock kernel on its first call |
| `candidate indexer on SM12x: dense blocks for prefill and decode (...)`, later `candidate indexer decode on SM12x: paged scoring (...)` | V4.1's candidate indexer runs on a backend that exists on SM12x |
| `indexer schedule: DeepGEMM plan forced for compress ratios (1, 2) on SM120`, `prefill flush armed: ...`, `Engram prefetch: next-chunk rows warmed from the scheduler`, `inline system notes of /v1/messages requests are folded in place` | the remaining hooks are on |
| `SM12x: ratio-2 KV source re-paged to 64-token pages for the sparse prefill kernel` | logged at the first prefill that reaches the sparse kernel |
| `RoCE collectives on: world=4 hcas=... all-reduce<=... all-gather shard<=...`, later `RoCE all-reduce live: ...` | the RDMA route is up and carried its first tensor; a `RoCE collectives disabled` warning names the rank and the reason, and the fleet runs on NCCL |
| `Served-model aliases: ...` | the extra ids are listed (only with `SERVED_ALIASES` set) |
| `DSV4 memory calculation ...` | the KV pool this boot got (it is auto-sized) |
| `The server is fired up ...` | `/health` answers; `relaunch.sh` prints the last few of these for you |

The watchdog (`launch/watchdog.sh`) is started by `relaunch.sh` on rank 0 and logs to `~/sgl-watchdog.log`. Every
60 s it requests `/health` and a three-token chat completion. It arms only after the first healthy check, so a boot is
never interrupted. A completion that times out while `/health` is fine counts as busy (a very long prefill can hold
the scheduler for minutes) and becomes a failure only after 600 s of continuous busy. Three consecutive failures
make it save every rank's container log and `dmesg` under `$LOG_DIR/postmortem/<timestamp>/`, run
`launch/production.sh` (which tears every rank down before starting any), and wait for `/health` again. After three
relaunch attempts in a row without a healthy period in between it stops relaunching and waits for a human; the log
says so. All of these periods are environment knobs of the script (`CHECK_INTERVAL`, `FAIL_THRESHOLD`,
`PROBE_TIMEOUT`, `BUSY_GRACE`, `READY_TIMEOUT`, `RELAUNCH_LIMIT`).

To stop: `launch/production.sh --stop` removes the container on all four nodes. While the watchdog is armed, running
the launcher or `--stop` by hand makes the watchdog relaunch production within a few minutes. Disarm it first: run
`launch/relaunch.sh` (it disarms and re-arms around the boot), or kill the pid from the last
`watchdog started (pid N, ...)` line in `~/sgl-watchdog.log` and remove `~/.sgl-watchdog.lock` (the watchdog also
stands down by itself when its lock file disappears or names another pid). `--stop` prints a note when it sees an
armed watchdog.

Tuning knobs are environment variables read by `launch/launch-sgl-dsv41.sh` and listed in its header: `CHUNK`
(prefill chunk, 2,048 in the launcher, 4,096 in `launch/production.sh`), `SPEC_K` (DSpark block, 5), `MEMFRAC` (0.80), `KVTOK` (0 = let SGLang size the KV pool),
`MAXREQ` (16 running requests), `CTX` (524,288), `MXFP8`, `ROCE_AR`, `THINKING_DEFAULT`, `ENGRAM_MODE`, `ENGRAM_EARLY`,
`ENGRAM_PREFETCH`, `STEP_TIMERS`, `EXTRA_ARGS`. Thinking is
on by default at maximum reasoning effort; a request turns it off with `"chat_template_kwargs": {"thinking": false}`.
Named reasoning efforts follow the 1017 reference on this base (`low` 50, `high` 75, `max` 100; production-1.7.1's SGLang
mapped `low` 25, `high` 50, `xhigh` 75); an integer from 1 to 100 means the same on both.

`launch/production.sh` also starts the server with `--enable-metrics` (`METRICS=0` turns it off): Prometheus metrics
on `http://<rank-0>:8210/metrics`, SGLang's own names (`sglang:num_running_reqs`, `sglang:num_queue_reqs`,
`sglang:prompt_tokens_total`, `sglang:generation_tokens_total`, `sglang:spec_accept_length`, ...), labelled with the
served model name.

## How it works

`overlay/sitecustomize.py` is on the image's `PYTHONPATH`, so it runs at the start of every SGLang process. When
`SPARK_ENGRAM_DIR` is set it registers one post-import hook per SGLang module in the table below; each hook runs
once, right after that module has been executed for the first time, and patches it in place. A hook that fails
raises, because a half-installed overlay would serve wrong results. Without `SPARK_ENGRAM_DIR` nothing is installed
and the image runs stock SGLang.

| piece | SGLang module it patches | what it does | why a Spark needs it |
|---|---|---|---|
| `overlay/engram_store.py` + `overlay/engram_rows.c` (the row store) | `sglang.srt.layers.engram` | Each rank owns one contiguous quarter of each Engram table and gathers the rows a lookup needs from a memory-mapped shard on local NVMe. The gather runs as a host-function node inside the CUDA graph; SGLang's own Triton kernel dequantizes the rows and SGLang's all-reduce sums the ranks. The table parameters are zero-size; nothing loads them. | Stock SGLang keeps the rows in device memory or pinned host memory. On a Spark both are the same 128 GB pool the weights already fill. |
| `overlay/engram_prefetch.py` | `sglang.srt.managers.scheduler` | When the scheduler launches a prefill chunk, a background thread hashes the *next* chunk of every request with SGLang's hash arithmetic and asks the row store to page those rows in (`posix_fadvise`, no copy). | A cold row is a page fault on the model's critical path; a 2,048-token chunk needs tens of thousands of rows per layer. |
| `overlay/mxfp8_kernel.py` | `sglang.srt.layers.quantization.fp8_utils` | Routes the 32×32-block FP8 dense projections to FlashInfer's `b12x` SM12x kernel (`SPARK_MXFP8_BACKEND`). A shape the kernel refuses because K is not a multiple of 128 (the shared expert's 5120×576) runs with K zero-padded to the next multiple (`SPARK_MXFP8_PAD_K`, default 1; a padded weight copy is built once outside graph capture and the first call is checked against the stock kernel); any other refused shape stays on the stock kernel, with one warning. | The stock choice is a 128-row-tile kernel that pads a six-token decode step to 128 rows. |
| `overlay/indexer_schedule.py` | `sglang.srt.layers.attention.dsv4.metadata` | On SM120 makes the attention-indexer metadata for the ratio-1 and ratio-2 sources build DeepGEMM's plan. Inert elsewhere. | SGLang skips the planner on SM120 for the classic ratio-4 indexer, but V4.1's low-ratio sources call the DeepGEMM kernel that needs it; without the plan CUDA-graph capture fails. |
| `overlay/sm120_prefill_pages.py` | `sglang.kernels.ops.attention.flash_mla_sm120` | Re-pages V4.1's ratio-2 KV source to 64-token pages, with SGLang's own page-split kernels, into persistent buffers. | FlashInfer's SM12x sparse-MLA prefill kernel takes 64-token pages; SGLang converts the primary source but passes the ratio-2 source through with 128-token pages, which the kernel rejects. |
| `overlay/prefill_flush.py` | `sglang.srt.model_executor.model_runner` | After a prefill chunk whose longest sequence is at least `SPARK_PREFILL_FLUSH_TOKENS` (8,192), returns PyTorch's cached allocator blocks to the driver. | The indexer's transient buffers grow with the prefix and the caching allocator cannot reuse smaller freed blocks, so reserved memory grows with the square of the prompt. On a Spark that is host memory; a 128K prompt drove all four nodes to zero free. |
| `overlay/roce_collectives.py` | `sglang.srt.distributed.parallel_state` and `model_runner` | Gives the tensor-parallel group a one-shot RDMA runtime (b12x's `comm.roce`, Apache-2.0, the copy bundled in the base image's FlashInfer 0.7.2): each small all-reduce or all-gather is one RDMA write per peer, replayable in CUDA graphs. The runtime's prepared plan is published once at start and every rank votes that it got the same one. Tensors above 1 MB (16 MB per all-gather shard) and every other group stay on NCCL. A health check after every forward turns a stalled peer into an error instead of a hang. | A decode step issues about eighty all-reduces of 48–400 KB; as NCCL kernels each costs about 100 µs in graph replay on this fabric, as an RDMA write about 21 µs. |
| `overlay/served_aliases.py` | `sglang.srt.entrypoints.http_server` | Lists the names in `SPARK_SERVED_ALIASES` on `/v1/models` and answers `/v1/models/{id}` for them. | SGLang serves exactly one model name; clients that ask for another id get a 404 from the listing endpoints. |
| `overlay/page_cache_release.py` | `sglang.srt.model_executor.model_runner` | Right after each model runner has allocated its KV pool, drops the checkpoint files from the page cache (`posix_fadvise(DONTNEED)`, no privileges). | Every rank reads the checkpoint at boot and never again, but the kernel kept 12–17 GB of it cached per node. On a Spark the GPU driver allocates from the same memory and fails instead of evicting cache; when such a failure hit cuBLAS, a rank died mid-prefill and the fleet hung (twice, see `docs/results.md`). |
| `overlay/late_tail.py` | `sglang.srt.managers.scheduler`, the V4 attention backend | With `SPARK_LATE_TAIL_SKIP=1` (production profile) the layers after the last kv_source layer run on one token for prefill chunks that end before the prompt's last 128 tokens; chunks holding any of those run exactly as SGLang runs them. `SPARK_LATE_TAIL_SKIP_ALL=1` (skip them outright, production-1.5 to 1.7.1) is not ported to SGLang `main` and refuses to start. | V4.1 caches its long-range KV only at layers 2, 8, 14 and 20 (YOCO-style); decode reads the later layers' window state only for the last 128 prompt positions, so their pass over earlier chunks was 14 % of every chunk and unread. |
| `overlay/kernel_load_guard.py` | `triton.compiler.compiler` | Retries a Triton kernel load the CUDA driver refuses with "operation not permitted" (waits for the GPU, up to 5 more attempts, logs each refusal); every other error is raised as before. | A mid-serving load of a new kernel specialization was refused once in production and took the fleet down; the cause is not known yet (`CUDA_LOG=1` in the launcher makes the driver log its reason). A safety net, not a fix. |
| `overlay/inline_system.py` | `sglang.srt.entrypoints.anthropic.serving` | On `/v1/messages`, system notes a client sends in the middle of the conversation (Claude Code adds one per turn) stay where they were and are folded into the next user message, instead of being hoisted into the system prompt at the top (`SPARK_INLINE_SYSTEM_IN_PLACE`, default 1). | Hoisting changed the prompt at about token 4,400 on every turn, so the prefix cache matched ~4K of a 150K-token Claude Code prompt and each turn re-prefilled the rest (about a minute before the first token). |
| `overlay/candidate_indexer.py` | `sglang.srt.layers.attention.deepseek_v4_backend` | On SM12x builds V4.1's candidate-block indexer on SGLang's dense-blocks backend (`SPARK_CANDIDATE_DENSE`, default 1), and its decode scoring takes SGLang's paged branch, which reads only each row's visible positions (`SPARK_CANDIDATE_DECODE=paged`, default). | SGLang picks a sparse-table backend on every device from compute capability 10 up, but its DeepGEMM kernel is enabled on SM100 only, so on SM12x backend init raises. The dense-blocks backend's other decode branch scores the whole graph width: 10 ms per step at a 64K context, against 1.8 ms paged. |
| `overlay/lookup_draft.py` + `overlay/lookup_wide.py` | the DSpark draft/worker/planner modules, `ngram_embedding_manager`, `spec_info`, the decode graph runner, `layers.engram`, `dsv41_sparse`, the V4 attention backend | Off unless `SPARK_LOOKUP_DRAFT=1`; the wide mode is not ported to production-1.8 (`WIDE=1 launch/production.sh` refuses; run production-1.7.1 for it). The ragged Engram and token maps of `lookup_wide.py` also serve `verify_cap.py`. Prompt-lookup drafting on top of DSpark: a device table of every request's tokens and a Triton matcher find where the text being written already occurs in the context. In wide mode DSpark still drafts its 5 tokens and, when its block continues an earlier 8-token match, the next up to 10 tokens of that occurrence are appended; only those rows verify the longer block (SGLang's compact ragged verify, graphs captured at 6b, 6b+10, 6b+20 tokens). Sampling stays exact (one-hot draft distributions on lookup tokens). | Answers that quote, restate or edit their input copy long runs of it; DSpark alone takes at most 6 tokens a step there. See "Prompt-lookup wide mode" below. |
| `overlay/step_timers.py` | `sglang.srt.models.deepseek_v4`, `deepseek_v2`, `layers.engram`, the V4 attention backend | Off unless `SPARK_STEP_TIMERS` is set: capture-safe CUDA events around attention, MoE and Engram inside the decode graphs (mode 1), or the GPU-timeline gaps between graphs without any synchronisation (mode 2). | Diagnosis only; this is how the decode-step breakdown in `docs/results.md` was measured. |
| `overlay/verify_cap.py` | `speculative.dspark_components.dspark_draft` | Off unless `SPARK_VERIFY_CAP=1` (with `RAGGED_VERIFY_MODE=compact` and a step-cost table): SGLang's compact DSpark verify, where the 1017 checkpoint's confidence head gives each request a verify length, made to run on V4.1 TP4: verify graphs at every token count up to 12, and the draft confidence broadcast from TP rank 0 so every rank picks the same graph. | A decode step verifies 6 rows per request and the MoE's cost grows almost linearly with them (397 µs per layer at 1 row, 1,189 at 6); on prose about 2 survive. Measured: prose +18 %, chat +4 %, but code −13 % and the coding conversation −4 %, so it stays off (`docs/results.md`). |

Environment variables the hooks read (all prefixed `SPARK_`; the launcher sets them from its knobs):

| variable | default | meaning |
|---|---|---|
| `SPARK_ENGRAM_DIR` | unset = overlay off | directory with `model.safetensors.index.json` and the Engram shards: a node-local copy from `tools/engram_local.py` (its manifest gives the row ranges) or the checkpoint snapshot itself |
| `SPARK_ENGRAM_ROWS` | | `1:lo:hi,14:lo:hi`, the row ranges when the directory is the full snapshot and has no manifest |
| `SPARK_ENGRAM_THREADS` | 64 | gather threads of the row store |
| `SPARK_ENGRAM_MAX_IDS` | 262,144 | ids one lookup may carry; the launcher derives it from `CHUNK` and `MAXREQ` |
| `SPARK_ENGRAM_STATS_SECONDS` | 60 | period of the per-layer gather statistics line (0 = off) |
| `SPARK_ENGRAM_CACHE_GB`, `SPARK_ENGRAM_CACHE_SECONDS` | 4, 5 | page-cache budget of this rank's Engram rows and how often it is checked; above it the rows are dropped and read from NVMe again when needed (0 = unbounded) |
| `SPARK_ENGRAM_PREFETCH` | 1 | next-chunk row prefetch from the scheduler |
| `SPARK_ENGRAM_EARLY` | 1 | start each Engram layer's host gather on a side stream as soon as the step's hash ids exist, overlapping the layers before it |
| `SPARK_ENGRAM_EARLY_VERIFY` | 0 | on eager steps also gather inline and compare byte for byte (diagnosis) |
| `SPARK_ENGRAM_MODE` | `hostnode` | `staged` selects the pure-Python staged implementation (below) |
| `SPARK_MXFP8_BACKEND` | `b12x` | FlashInfer MXFP8 backend for the dense projections; `cutlass` or empty = stock |
| `SPARK_LATE_TAIL_SKIP`, `SPARK_LATE_TAIL_SKIP_ALL` | 0, 0 (1, 0 in `launch/production.sh`) | prefill: the late layers run on one token on chunks that end before the prompt's last 128 tokens (`late_tail.py`); `SKIP_ALL` is not ported to this base and refuses. Decode attention on the rank's real 16 heads, a hook until production-1.7.1 (`SPARK_SM120_REAL_HEADS`), is native in SGLang `main` |
| `SPARK_CANDIDATE_DENSE`, `SPARK_CANDIDATE_DECODE` | 1, `paged` | the SM12x candidate indexer backend and its decode scoring branch (`candidate_indexer.py`; `stock` = SGLang's choice) |
| `SPARK_MXFP8_PAD_K` | 1 | zero-pad K to a multiple of 128 for shapes b12x refuses (`mxfp8_kernel.py`) |
| `SPARK_VERIFY_CAP`, `SPARK_VERIFY_CAP_BUCKETS` | 0, `fine` | compact verify with per-request lengths (`verify_cap.py`, off); verify graph token counts (`fine`, `stock` or a comma list) |
| `SPARK_KERNEL_LOAD_RETRIES` | 5 | retries of a refused Triton kernel load (0 = hook off) |
| `SPARK_PAGE_CACHE_RELEASE` | 1 | drop the checkpoint's page cache after the KV pool is allocated (0 = keep it) |
| `SPARK_PREFILL_FLUSH_TOKENS` | 8,192 | longest sequence that triggers the allocator flush after a prefill chunk (0 = off) |
| `SPARK_ROCE_AR`, `SPARK_ROCE_AR_MAX` | 0, 1MB | RDMA all-reduce route on/off, largest tensor routed (the launcher turns it on) |
| `SPARK_ROCE_AG`, `SPARK_ROCE_AG_MAX` | 1, 16MB | RDMA all-gather route, largest per-rank shard routed |
| `SPARK_SERVED_ALIASES` | empty | extra model ids for `/v1/models` |
| `SPARK_STEP_TIMERS` | 0 | 1: per-block decode-step timing, sampled every `SPARK_STEP_TIMERS_EVERY` (100) replays; 2: gaps between graphs |
| `SPARK_STEP_TIMERS_ATTN` | 0 | with mode 1, also split attention into projections, compressor, indexer, kernel and output |
| `SPARK_LOOKUP_DRAFT`, `SPARK_LOOKUP_MODE` | 0, `replace` (1, `wide` in `launch/production.sh`) | prompt-lookup drafting on/off; `wide` is the mode to use (`replace`/`extend` are the measured-and-rejected variants) |
| `SPARK_LOOKUP_EXT_KEY`, `SPARK_LOOKUP_EXT_BUDGET`, `SPARK_LOOKUP_MIN_TOKENS`, `SPARK_LOOKUP_WINDOW` | 8, 20, 2, 0 | tokens that must match before extending; extension tokens per step over the batch; shortest extension; tokens searched back (0 = whole context) |
| `SPARK_LOOKUP_SYNC_GATE`, `SPARK_LOOKUP_NARROW_SLOTS`, `SPARK_LOOKUP_FUSED_C2` | 0, 0, 0 (1, 1, 1 in `launch/production.sh`) | wide-mode step-cost fixes: skip the per-step length copy while no extension is in demand; size graph slots for 6-token rows; production's fused ratio-2 compressor on uniform steps |

A second Engram implementation ships alongside: `SPARK_ENGRAM_MODE=staged` (launcher `ENGRAM_MODE=staged`) stages the
rows before every forward from a pre-forward hook instead of gathering them inside the graph. It is a port of the
author's vLLM implementation, needs no C library, decodes 8–9 % faster and prefills 14–20 % slower on unique text
(numbers in `docs/results.md`). The default stays host-node.

## Prompt-lookup wide mode (production-1.7.1 only, `WIDE=1`)

The wide mode is not ported to production-1.8: on SGLang `main`, `WIDE=1 launch/production.sh` stops with a message.
To run it, use the production-1.7.1 image with this repository at its `production-1.7.1` commit. On that image,
`WIDE=1 launch/production.sh` turns it on (off by default since 2026-09-29: a lookup continuation ran into image placeholder ids and a device-side assert stopped the fleet; 1.7.1 stops continuations at the first non-token id, and wide mode comes back after a verification window): `SPEC_K=15 LOOKUP_DRAFT=1 LOOKUP_MODE=wide RAGGED_VERIFY_MODE=compact
LOOKUP_SYNC_GATE=1 LOOKUP_NARROW_SLOTS=1 LOOKUP_FUSED_C2=1` (each overridable from the environment; `LOOKUP_DRAFT=0
SPEC_K=5 RAGGED_VERIFY_MODE=static` gives the 1.6 decode path on the same image). A 90-minute soak against
production-1.6 on the same cluster (`docs/results.md`, window 33): a 63K-token coding conversation 65.9 -> 74.4 tok/s,
copy-heavy answers 69.7 -> 85.3, plain chat 53.7 -> 54.2, ten concurrent requests 132.9 -> 161.0; news replay,
long-prefill stress, archive codes, images and batched-correctness checks all passed. `tests/test_lookup_draft_cpu.py`
checks the matcher, the table bookkeeping, the gate and the graph layouts on CPU (Triton interpreter), including
SGLang's own compact verify-id gather.

## Caveats

- Host-memory floor. With 2,048-token prefill chunks every node kept 11–14 GiB free through a 128K prefill; with
  1,024-token chunks 16–20 GiB. production-1.8 runs 4,096-token chunks (SGLang `main` bounds the prefill indexer's
  transient): they passed the needles to 128K, the long-prefill stress and three weeks of production without a GPU
  out-of-memory event, but the floor itself was not re-measured. The floor is what your OOM guard has to live with.
- Page cache is not free memory for the GPU. The driver fails an allocation rather than evict cached file data, and
  the kernel only reclaims cache when free memory is down to its watermarks (about 1.5 GB by default). The overlay
  keeps its own files out of the way (`page_cache_release`, `SPARK_ENGRAM_CACHE_GB`); anything else on the node that
  reads large files while the engine serves eats into the same headroom. Watch `MemFree`, not `MemAvailable`.
- The RoCE runtime is fail-stop. A stalled peer poisons it, the health check after the next forward raises, the engine
  dies, and the watchdog relaunches the fleet. That is the intended behaviour; there is no in-place recovery.
- The base is a nightly of SGLang `main`, and the hooks bind to its internals (each one raises if its target moved).
  The Dockerfile pins the base by digest; rebase deliberately with [docs/upstream.md](docs/upstream.md), which lists
  what the last rebase had to change.
- The RoCE-v2 GID index of the fabric address differs from node to node. The launcher probes it on each node before
  every launch; a wrong index fails every RDMA connect.
- Never start a rank while a previous engine still holds the GPU: the new rank dies with "device busy" and the other
  ranks hang in distributed init. The launcher waits for free GPUs, and the watchdog only ever launches through it.
- One served model name. Aliases only change what `/v1/models` lists; every request is served by the same model.
- The API listens on `0.0.0.0:8210` without authentication. Put it behind your own proxy.
- SGLang sizes the KV pool at boot from the memory it finds free, which depends on the page cache; two boots gave
  4.26 M and 5.43 M tokens. Pin it with `KVTOK` (`--max-total-tokens`) for identical boots.
- The slowest rank paces every step, for both NCCL and RDMA collectives. One hot or throttled node slows the fleet.
- GB10 has a slow clock state. When comparing numbers, take them on the same minute on the same boot.

## Repository layout

```
Dockerfile                    the overlay image: base image + overlay/ + tests/, RoCE proxy and C library built
overlay/sitecustomize.py      installs the hooks at interpreter start when SPARK_ENGRAM_DIR is set
overlay/engram_store.py       Engram rows from NVMe inside the CUDA graph (hook; default)
overlay/engram_staged.py      Engram rows staged before each forward (hook; SPARK_ENGRAM_MODE=staged, no C library)
overlay/engram_rows.c         the row store: mmap, fadvise prepass, thread pool, host-function entry point (C)
overlay/engram_prefetch.py    next-chunk row prefetch from the scheduler (hook)
overlay/mxfp8_kernel.py       FlashInfer MXFP8 backend selection with per-shape fallback (hook)
overlay/indexer_schedule.py   DeepGEMM plan for the ratio-1/2 indexers on SM120 (hook)
overlay/sm120_prefill_pages.py  64-token pages for the ratio-2 KV source in the sparse prefill (hook)
overlay/prefill_flush.py      allocator flush after long prefill chunks (hook)
overlay/kernel_load_guard.py  retry a Triton kernel load the driver refuses (hook)
overlay/late_tail.py          prefill: late layers only where decode reads them; prefill timers (hook)
overlay/page_cache_release.py drops the checkpoint's page cache once the KV pool is allocated (hook)
overlay/roce_collectives.py   one-shot RDMA collectives for the TP group + fail-stop health check (hook)
overlay/served_aliases.py     extra model ids on /v1/models (hook)
overlay/candidate_indexer.py  V4.1's candidate indexer on the dense-blocks backend with paged decode scoring on SM12x (hook)
overlay/verify_cap.py         compact DSpark verify with per-request lengths from the confidence head (hook, off unless SPARK_VERIFY_CAP=1)
overlay/inline_system.py      Anthropic-API inline system notes folded in place, for the prefix cache (hook)
overlay/step_timers.py        per-graph decode-step timing for diagnosis windows (hook, off unless SPARK_STEP_TIMERS=1)
overlay/lookup_draft.py       prompt-lookup drafting: token table, Triton matcher, proposal builder (hook, off unless SPARK_LOOKUP_DRAFT=1)
overlay/lookup_wide.py        wide mode: 5-token DSpark draft, per-row compact verify, graph buckets, ragged Engram/token maps (hooks)
overlay/lookup_index.py       pure-Python reference matcher the CPU test checks the Triton one against
launch/fleet.env.example      the site file, every line documented; copy to launch/fleet.env (git-ignored)
launch/launch-sgl-dsv41.sh    the four-node launcher: preflight, GID probe, free-GPU wait, docker run per rank, --dry-run, --stop
launch/production.sh          the production profile: published image, port 8210
launch/relaunch.sh            disarm watchdog, stop, launch production, wait for /health, print boot facts, re-arm
launch/watchdog.sh            liveness probe with evidence capture and orchestrated relaunch
launch/window-sgl.sh          deadline-guarded test window on a separate port, restores production afterwards
tools/engram_partition.py     prints each rank's Engram row ranges for a TP size
tools/engram_local.py         copies one rank's rows to a node-local sparse file and verifies them
tests/test_engram_rows.py     CPU test of the C library against numpy on a synthetic sparse shard
tests/test_hooks_cpu.py       CPU check, inside the image, that every hook binds to its SGLang module
tests/test_lookup_draft_cpu.py CPU test of the lookup drafter (Triton interpreter): matcher, table, gate, layouts, prefetch hash
../../../bench/needle.py  (shared) long-context needle: prefill tok/s from time to first token, answer checked
../../../bench/prefill_repetitive.py  (shared) the same on repetitive filler (the input behind most published prefill numbers)
../../../bench/decode_bench.py  (shared) single-stream decode: counting / code / prose, best of N runs
../../../bench/long_prefill_stress.py  (shared) long unique prefills back to back with decode streams: the load behind the 2026-09 rank crashes
../../../bench/multi_check.py  (shared) batched-decode correctness: the same greedy chats alone and 4 at a time
../../../bench/vision_check.py  (shared) image requests, including an image inside a chunked long prompt
vendor/                       b12x wheel and source snapshot (Apache-2.0) that production-1.7.1 was built with, with checksums;
                              copied into the image, not installed since 1.8 (b12x comes with the base's FlashInfer)
docs/design.md                why each hook exists and how it works
docs/results.md               measurements, bring-up history, production boot facts
docs/upstream.md              the pinned base image and how to rebase the overlay
NOTICE, LICENSE               licences of this repository and of the third-party material
```

## License

This repository is MIT (see `LICENSE`). SGLang and FlashInfer (Apache-2.0) and PyTorch (BSD-3-Clause) are used as
installed in the base image and are not redistributed here; production-1.8 uses the b12x bundled in that FlashInfer.
The b12x release production-1.7.1 was built with (Apache-2.0) is redistributed unmodified in `vendor/` with its licence. `tools/engram_local.py` and `bench/needle.py` are adapted from tonyd2wild's MIT-licensed
work and carry that copyright notice; `bench/prefill_repetitive.py` reuses its request loop. The model weights are not
included. The `NOTICE` file lists all of this precisely.
