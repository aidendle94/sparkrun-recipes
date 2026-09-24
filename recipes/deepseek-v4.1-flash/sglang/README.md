# DeepSeek-V4.1-Flash on four DGX Sparks with SGLang

This repository serves the [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
checkpoint on four NVIDIA DGX Spark nodes with SGLang, behind one OpenAI-compatible endpoint. The engine is the
SGLang team's own DeepSeek-V4.1 image, `lmsysorg/sglang:dev-dsv41`, unmodified. On top of it sits an MIT-licensed
overlay of eleven Python import hooks and one C library that supply what a Spark needs and the stock image does
not have: the model's Engram tables read from node-local NVMe inside the CUDA graph (they do not fit next to the
weights in 128 GB of unified memory), kernels and schedules that work on the GB10's SM12x architecture, and RDMA
collectives that make a four-node tensor-parallel decode step fast. Nothing model-specific is forked; the same
image still runs stock SGLang when the overlay's switch is unset.

Glossary. **Engram**: the model's two hash-table embedding layers (layers 1 and 14), each an fp8 table of about
101 GB looked up by n-gram hash. **DSpark**: DeepSeek's speculative decoding, where a draft block of tokens is verified
in one forward pass (block size 5 here). **TP4/EP4**: tensor parallel over the four nodes, with the experts split
four ways. **MXFP4/MXFP8**: 4- and 8-bit floating point with shared block scales; the experts are MXFP4, the dense
projections 32×32-block FP8. **RoCE**: RDMA over Converged Ethernet, direct memory writes between nodes over the
Sparks' ConnectX-7 ports. **SM12x (GB10)**: the Spark's GPU architecture (compute capability 12.1), which needs
different kernels and schedules from the datacenter Blackwell parts.

## What you get

Measured on four Sparks over a switched RoCE fabric with the launcher defaults: prefill and the vLLM comparison on
2026-09-17, decode and concurrency on the production-1.2 image on 2026-09-23. Conditions, boot-by-boot numbers and
the comparison baseline are in [docs/results.md](docs/results.md).

| what | this stack | the author's previous vLLM deployment, same checkpoint, same nodes |
|---|---|---|
| prefill, 32K / 128K prompt of unique text (tok/s) | 3,068 / 2,580 | 2,094–2,362 / 2,152 |
| prefill, 32K / 128K prompt of repetitive text (tok/s) | 4,063 / 3,435 | 2,652 / 2,550 |
| decode, one stream, counting / code / prose (tok/s) | 110 / 94 / 37 | 93 / 67 / 31 |
| decode, four prose streams, aggregate (tok/s) | 72 | 70 |
| aggregate at 1 / 8 / 16 concurrent streams, fixed prompt set (tok/s) | 57 / 211 / 337 (64 / 31 / 26 per stream) | 185 at 6 streams |
| KV pool | 4.3–5.4 M tokens on the two production boots, auto-sized at each boot (4.4–5.8 M over earlier boots) | 3.42 M tokens |
| context length | 524,288 tokens | |
| boot to healthy | about 10 min (582–604 s over five boots) | |
| request features | images, tool calls, thinking (on by default, off per request) | |

Prefill tok/s is prompt tokens divided by time to first token on a needle prompt that is salted per run
(`NEEDLE_SALT`, which `window-sgl.sh` sets), so the prefix cache cannot hit. Decode is greedy with thinking off. Each cell is one run on one boot unless the results page
says otherwise.

## What you need

- Four DGX Spark nodes (GB10, 128 GB unified memory, arm64). One of them is rank 0: you run every script there.
- A RoCE fabric between the nodes, switched or direct, with the RDMA devices visible in `/sys/class/infiniband` on
  every node. The launcher uses every device you list (two ConnectX-7 functions per node on the author's fabric).
- The checkpoint, about 510 GB, on one node and exported read-only over NFS to the other three. Mount the HuggingFace
  repo directory (`models--deepseek-ai--DeepSeek-V4.1-Flash`), not the snapshot inside it, so the blob symlinks
  resolve on the clients.
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
# 1. The image, on every node (arm64). Pull the published build...
docker pull aidendle94/sparkrun-sglang-dsv41-gb10:production-1.3
#    ...or build it on each node from this repository, then put IMAGE=sglang-dsv41-spark:local in launch/fleet.env
#    so the launcher, relaunch.sh and the watchdog all use it.
docker build -t sglang-dsv41-spark:local .

# 2. The site file: nodes, users, devices, paths. Every line of the example says how to find its value.
cp launch/fleet.env.example launch/fleet.env

# 3. Each rank's Engram row ranges (the split used by SGLang, for TP4).
python3 tools/engram_partition.py /path/to/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/<revision>

# 4. On every node that does not hold the checkpoint: copy that rank's rows to local NVMe (about 50 GB,
#    rate-limited to 600 MB/s by default with --mbps=, then spot-checked against the source: the shard header and
#    2,002 rows per range, first, last and 2,000 random).
python3 tools/engram_local.py /path/to/snapshot ~/dsv41-engram-local 1:<lo>:<hi> 14:<lo>:<hi>
#    Optional CPU check, on a node that now has a row copy: the hooks bind to this image's SGLang (no GPU).
#    It must end with HOOKS CPU TEST PASS.
docker run --rm -v ~/dsv41-engram-local:/engram-local:ro -e SPARK_ENGRAM_DIR=/engram-local \
  --entrypoint python3 aidendle94/sparkrun-sglang-dsv41-gb10:production-1.3 /opt/dsv41-spark/tests/test_hooks_cpu.py

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
| `overlay: 8 import hooks armed`, then `overlay hook <module>.install applied to <sglang module>` per hook | `sitecustomize` saw `SPARK_ENGRAM_DIR` and each hook patched its module; a hook that fails raises and the process dies here |
| `engram_store installed: Engram rows of layers [1, 14] come from <dir> through libengram_rows.so` at import, then, as the model is built, `engram layer 1: rank r/4 owns rows [lo, hi) of N, shard <file>, 64 gather threads, <capacity> ids per lookup` (and the same for layer 14) | the tables are served from NVMe; the four ranks' ranges were checked to tile the table |
| `engram layer 1: <n> lookups, <n> owned rows, <n> zeroed rows, <n> us mean gather per lookup` every 60 s | the gather statistics; the microseconds are the row store's own timer |
| `MXFP8 dense GEMMs routed to FlashInfer backend b12x ...`, later `MXFP8 backend b12x refused N=5120 K=576 (...); this shape stays on cutlass` | expected once per refused shape; every other shape logs `MXFP8 backend b12x serves N=... K=...` on first use |
| `indexer schedule: DeepGEMM plan forced for compress ratios (1, 2) on SM120`, `prefill flush armed: ...`, `Engram prefetch: next-chunk rows warmed from the scheduler` | the remaining hooks are on |
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
(prefill chunk, 2,048), `SPEC_K` (DSpark block, 5), `MEMFRAC` (0.80), `KVTOK` (0 = let SGLang size the KV pool),
`MAXREQ` (16 running requests), `CTX` (524,288), `MXFP8`, `ROCE_AR`, `THINKING_DEFAULT`, `ENGRAM_MODE`, `ENGRAM_EARLY`,
`ENGRAM_PREFETCH`, `STEP_TIMERS`, `EXTRA_ARGS`. Thinking is
on by default at maximum reasoning effort; a request turns it off with `"chat_template_kwargs": {"thinking": false}`.

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
| `overlay/mxfp8_kernel.py` | `sglang.srt.layers.quantization.fp8_utils` | Routes the 32×32-block FP8 dense projections to FlashInfer's `b12x` SM12x kernel (`SPARK_MXFP8_BACKEND`). A weight shape the kernel refuses (the K = 576 projections) stays on the stock kernel for good, with one warning. | The stock choice is a 128-row-tile kernel that pads a six-token decode step to 128 rows. |
| `overlay/indexer_schedule.py` | `sglang.srt.layers.attention.dsv4.metadata` | On SM120 makes the attention-indexer metadata for the ratio-1 and ratio-2 sources build DeepGEMM's plan. Inert elsewhere. | SGLang skips the planner on SM120 for the classic ratio-4 indexer, but V4.1's low-ratio sources call the DeepGEMM kernel that needs it; without the plan CUDA-graph capture fails. |
| `overlay/sm120_prefill_pages.py` | `sglang.kernels.ops.attention.flash_mla_sm120` | Re-pages V4.1's ratio-2 KV source to 64-token pages, with SGLang's own page-split kernels, into persistent buffers. | FlashInfer's SM12x sparse-MLA prefill kernel takes 64-token pages; SGLang converts the primary source but passes the ratio-2 source through with 128-token pages, which the kernel rejects. |
| `overlay/prefill_flush.py` | `sglang.srt.model_executor.model_runner` | After a prefill chunk whose longest sequence is at least `SPARK_PREFILL_FLUSH_TOKENS` (8,192), returns PyTorch's cached allocator blocks to the driver. | The indexer's transient buffers grow with the prefix and the caching allocator cannot reuse smaller freed blocks, so reserved memory grows with the square of the prompt. On a Spark that is host memory; a 128K prompt drove all four nodes to zero free. |
| `overlay/roce_collectives.py` | `sglang.srt.distributed.parallel_state` and `model_runner` | Gives the tensor-parallel group a one-shot RDMA runtime (b12x, Apache-2.0, vendored): each small all-reduce or all-gather is one RDMA write per peer, replayable in CUDA graphs. Tensors above 1 MB (16 MB per all-gather shard) and every other group stay on NCCL. A health check after every forward turns a stalled peer into an error instead of a hang. | A decode step issues about eighty all-reduces of 48–400 KB; as NCCL kernels each costs about 100 µs in graph replay on this fabric, as an RDMA write about 21 µs. |
| `overlay/served_aliases.py` | `sglang.srt.entrypoints.http_server` | Lists the names in `SPARK_SERVED_ALIASES` on `/v1/models` and answers `/v1/models/{id}` for them. | SGLang serves exactly one model name; clients that ask for another id get a 404 from the listing endpoints. |
| `overlay/page_cache_release.py` | `sglang.srt.model_executor.model_runner` | Right after each model runner has allocated its KV pool, drops the checkpoint files from the page cache (`posix_fadvise(DONTNEED)`, no privileges). | Every rank reads the checkpoint at boot and never again, but the kernel kept 12–17 GB of it cached per node. On a Spark the GPU driver allocates from the same memory and fails instead of evicting cache; when such a failure hit cuBLAS, a rank died mid-prefill and the fleet hung (twice, see `docs/results.md`). |
| `overlay/request_guard.py` | `sglang.srt.managers.tokenizer_manager` | Answers a request for prompt-token log-probabilities (for example `/v1/completions` with `echo` and `logprobs`) with HTTP 400. | Under V4.1's decoder sliding-window bounded replay such a request raises inside the model and stops the whole server. |
| `overlay/step_timers.py` | `sglang.srt.models.deepseek_v4`, `deepseek_v2`, `layers.engram`, the V4 attention backend | Off unless `SPARK_STEP_TIMERS` is set: capture-safe CUDA events around attention, MoE and Engram inside the decode graphs (mode 1), or the GPU-timeline gaps between graphs without any synchronisation (mode 2). | Diagnosis only; this is how the decode-step breakdown in `docs/results.md` was measured. |

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
| `SPARK_PAGE_CACHE_RELEASE` | 1 | drop the checkpoint's page cache after the KV pool is allocated (0 = keep it) |
| `SPARK_PREFILL_FLUSH_TOKENS` | 8,192 | longest sequence that triggers the allocator flush after a prefill chunk (0 = off) |
| `SPARK_ROCE_AR`, `SPARK_ROCE_AR_MAX` | 0, 1MB | RDMA all-reduce route on/off, largest tensor routed (the launcher turns it on) |
| `SPARK_ROCE_AG`, `SPARK_ROCE_AG_MAX` | 1, 16MB | RDMA all-gather route, largest per-rank shard routed |
| `SPARK_SERVED_ALIASES` | empty | extra model ids for `/v1/models` |
| `SPARK_STEP_TIMERS` | 0 | 1: per-block decode-step timing, sampled every `SPARK_STEP_TIMERS_EVERY` (100) replays; 2: gaps between graphs |
| `SPARK_STEP_TIMERS_ATTN` | 0 | with mode 1, also split attention into projections, compressor, indexer, kernel and output |

A second Engram implementation ships alongside: `SPARK_ENGRAM_MODE=staged` (launcher `ENGRAM_MODE=staged`) stages the
rows before every forward from a pre-forward hook instead of gathering them inside the graph. It is a port of the
author's vLLM implementation, needs no C library, decodes 8–9 % faster and prefills 14–20 % slower on unique text
(numbers in `docs/results.md`). The default stays host-node.

## Caveats

- Host-memory floor. With 2,048-token prefill chunks every node kept 11–14 GiB free through a 128K prefill; with
  1,024-token chunks 16–20 GiB. Larger chunks are untested. The floor is what your OOM guard has to live with.
- Page cache is not free memory for the GPU. The driver fails an allocation rather than evict cached file data, and
  the kernel only reclaims cache when free memory is down to its watermarks (about 1.5 GB by default). The overlay
  keeps its own files out of the way (`page_cache_release`, `SPARK_ENGRAM_CACHE_GB`); anything else on the node that
  reads large files while the engine serves eats into the same headroom. Watch `MemFree`, not `MemAvailable`.
- The RoCE runtime is fail-stop. A stalled peer poisons it, the health check after the next forward raises, the engine
  dies, and the watchdog relaunches the fleet. That is the intended behaviour; there is no in-place recovery.
- The base tag `lmsysorg/sglang:dev-dsv41` is a moving tag, and upstream's newer `dsv4.1` branch removed the
  `_owned_rows` method this overlay hooks. The Dockerfile pins the base by digest; rebase deliberately with
  [docs/upstream.md](docs/upstream.md).
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
Dockerfile                    the overlay image: base image + overlay/ + tests/ + vendor/, b12x installed, C library built
overlay/sitecustomize.py      installs the hooks at interpreter start when SPARK_ENGRAM_DIR is set
overlay/engram_store.py       Engram rows from NVMe inside the CUDA graph (hook; default)
overlay/engram_staged.py      Engram rows staged before each forward (hook; SPARK_ENGRAM_MODE=staged, no C library)
overlay/engram_rows.c         the row store: mmap, fadvise prepass, thread pool, host-function entry point (C)
overlay/engram_prefetch.py    next-chunk row prefetch from the scheduler (hook)
overlay/mxfp8_kernel.py       FlashInfer MXFP8 backend selection with per-shape fallback (hook)
overlay/indexer_schedule.py   DeepGEMM plan for the ratio-1/2 indexers on SM120 (hook)
overlay/sm120_prefill_pages.py  64-token pages for the ratio-2 KV source in the sparse prefill (hook)
overlay/prefill_flush.py      allocator flush after long prefill chunks (hook)
overlay/page_cache_release.py drops the checkpoint's page cache once the KV pool is allocated (hook)
overlay/roce_collectives.py   one-shot RDMA collectives for the TP group + fail-stop health check (hook)
overlay/served_aliases.py     extra model ids on /v1/models (hook)
overlay/request_guard.py      rejects prompt-logprob requests that would stop the server under bounded replay (hook)
overlay/step_timers.py        per-graph decode-step timing for diagnosis windows (hook, off unless SPARK_STEP_TIMERS=1)
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
../../../bench/needle.py      (shared) long-context needle: prefill tok/s from time to first token, answer checked
../../../bench/prefill_repetitive.py  (shared) the same on repetitive filler (the input behind most published prefill numbers)
../../../bench/decode_bench.py  (shared) single-stream decode: counting / code / prose, best of N runs
../../../bench/long_prefill_stress.py  (shared) long unique prefills back to back with decode streams: the load behind the 2026-09 rank crashes
vendor/                       b12x wheel and source snapshot (Apache-2.0) used at image build, with checksums
docs/design.md                why each hook exists and how it works
docs/results.md               measurements, bring-up history, production boot facts
docs/upstream.md              the pinned base image and how to rebase the overlay
NOTICE, LICENSE               licences of this repository and of the third-party material
```

## License

This repository is MIT (see `LICENSE`). SGLang and FlashInfer (Apache-2.0) and PyTorch (BSD-3-Clause) are used as
installed in the base image and are not redistributed here. b12x (Apache-2.0) is redistributed unmodified in
`vendor/` with its licence. `tools/engram_local.py` and `bench/needle.py` are adapted from tonyd2wild's MIT-licensed
work and carry that copyright notice; `bench/prefill_repetitive.py` reuses its request loop. The model weights are not
included. The `NOTICE` file lists all of this precisely.
