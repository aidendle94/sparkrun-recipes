# Inkling-2-Small on four DGX Sparks with SGLang (experimental)

This recipe serves [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B parameters, 17B active per token, the official NVFP4 checkpoint, 394 GB) on four NVIDIA DGX Spark nodes behind
one OpenAI-compatible endpoint. The engine is the SGLang team's own image for the model,
`lmsysorg/sglang:dev-inkling-2-small`, unmodified. That image targets Hopper and datacenter Blackwell. On a Spark's
GB10 (SM12x) it does not serve the model as it is. The MIT-licensed Python hooks in `patches/` are copied to every
node and put on `PYTHONPATH`. They supply what GB10 needs (attention arguments, expert kernels, collectives across
four nodes, a correct decode kernel at page size 1) and then the speed: about 61 tok/s for one stream, against 37
with the image's own decode step on the same hardware.

**Status: experimental.**

- In service on one four-node fleet since 2026-11-12.
- The base is a development image, not an SGLang release, and the hooks are tied to one build of it (digest under
  Run it). Three hooks rebuild an attention kernel from the image's own source at boot, and the forked GEMM of the
  packed block scales is compiled from it once per image. Each stops if that source has changed.
- Every number is a single boot on that fleet. The defaults are recent: the sharded experts since 2026-11-15, the
  500K configuration (524,288 tokens of context, the 793,000-token pool, the three prefill switches) since
  2026-11-17, the packed block scales since 2026-11-19, the confidence head since 2026-11-20.
- Open problems are listed under [Known problems](#known-problems). One is a silent stall of the whole fleet whose
  cause is not known.

For a recipe with more serving time behind it, use [`deepseek-v4.1-flash/sglang`](../../deepseek-v4.1-flash/sglang/).

Glossary. **TP4/EP4**: tensor parallelism over the four nodes, with the experts split four ways. **DSpark**: the
engine's built-in speculative decoding, in which a small draft model proposes up to 7 tokens and the target
verifies them in one forward. **NVFP4 / MXFP8**: 4-bit and 8-bit floating-point weight formats with shared block
scales; 40 of the model's expert layers are NVFP4 and 8 are MXFP8. **RoCE**: RDMA over Converged Ethernet, here
between the Sparks' ConnectX-7 ports. **Pool**: the KV cache, counted in tokens and shared by all running and
cached requests.

## What you get

Measured on four Sparks over a switched RoCE fabric between 2026-11-13 and 2026-11-20. Every number is one boot.
All rows but two were measured by 2026-11-16, before the four newest default switches. Most of them were measured
on the 600,000-token pool without the prefill switches; a row that was measured with the prefill switches or on
the 793,000-token pool says so. Two of the four newest switches are the kernel switches of 2026-11-17 against
kernels compiled and loaded while serving; they are exact and have no row. The other two are the packed block
scales (2026-11-19) and the confidence head (2026-11-20); they have a row each, measured against the configuration
without them. The conditions and the record of each number are in [docs/results.md](docs/results.md).

| what | measured |
|---|---|
| decode, one stream, greedy, coding questions of 400 tokens | 60–61 tok/s (37 with the image's own decode step, on another set of such questions) |
| decode, one stream, temperature 0.7 | 56.6 tok/s |
| decode, four / eight streams at a time, greedy | 106 / 130 tok/s for the batch (about 26 / 16 a stream) |
| decode, one stream, greedy, with the confidence head (the default since 2026-11-20; a test boot and the serving boot after it, 16 coding questions of 400 tokens) | 61.64 tok/s on the test boot and 61.79 on the serving boot, against 60.41 and 60.81 on the boots without it before and after (60.61 on average): x1.017 and x1.019, net of the head's own cost. These are medians over the 16 questions; the step time at equal accept is 1.5 % and 1.2 % lower. Those comparison boots carry the packed block scales, which cost about 1 % of a step. The boots of 2026-11-19 with neither default ran 61.31 and 60.95 tok/s, so for one greedy stream the two newest defaults together are about level with the stack before them (the serving boot against them: x1.008 and x1.014). Inside the test boot, on a mixed set of 16 short prompts (8 of those coding questions, 8 general ones) run back to back under the head and under the gate's own rule: x1.023 greedy and x1.050 at temperature 1.0, not net (both arms carried the head's cost); net at temperature 1.0 was not measured (about x1.036 estimated) |
| memory with the packed block scales (the default since 2026-11-19) | 3.0–3.4 GB more available on every node; the same text and the same NLL token for token. One stream's step is about 1 % longer (0.5 to 0.7 ms on three boots) |
| prefill, ~15K-token prompts | 2,400–2,500 tok/s (before the prefill switches; not measured again with them) |
| cold prefill of 30K / 120K tokens | 3,110 / 2,786 tok/s (2,474 / 2,241 without the three prefill switches) |
| cold prefill of 130K / 260K / 500K tokens | 61 / 142 / 348 s before the prefill switches; with them 120K takes 43 s (54 without) and 300K 135 s |
| decode behind a long context (130K–500K) | 52–55 tok/s on the configuration with the 600K pool cap; 68 at 130K and 60 at 500K on the one with the 793K cap |
| context length | 524,288 (the default since 2026-11-17; 262,144 before) |
| KV pool | 793,000 tokens with the defaults (a cap; see [Run it](#run-it)). Before these defaults: 545K–600K with the 600K cap that most rows here were measured on, and 493K–805K when the pool was sized at each boot without a cap |
| concurrent requests | 8 |
| tool calls, 283 cases over five wire formats (chat, Anthropic messages, each also streamed, and responses), two trials | 250 right, 20 soft, 13 failed (10 are `tool_choice: none`, 3 slips on the sampled trial); 12 of 12 agent loops. With the prefill switches (the defaults): 252 / 20 / 11, and no case of the temperature-0 trial changed its verdict |
| boot to healthy | about 5 min with a warm kernel cache, about 17 min the first time |
| request features | tool calls, reasoning with effort levels; OpenAI chat completions and responses, Anthropic `/v1/messages`; extra model names on `/v1/models` and the Ollama-style `/api/tags` and `/api/show` |

Decode tok/s is tokens divided by decode seconds, taken from the server's own timing. For comparison, the hosted
model on OpenRouter answered the same long lookups at 190–320 tok/s, depending on the provider. This recipe runs on
desktop machines with 128 GB of memory each, and memory bandwidth sets its decode step.

## What is changed against the stock image

Nothing in the image is modified. Every change is a hook in `patches/` behind an environment switch. The following
make it work on GB10 at all:

- **Attention**: SGLang's SM12x wrapper drops two arguments that Inkling passes. At page size 1, which is what GB10
  runs, the image's decode kernel reads keys from other requests' cache slots as soon as two requests share a
  batch. The result is loops and cross-talk that throughput numbers do not show. `patches/paged_decode.py` looks
  the page up per key. Read [docs/results.md](docs/results.md#garbage-under-concurrency-found-and-fixed-2026-11-13)
  before trusting any multi-stream number from a GB10 setup without it.
- **Experts**: the image's NVFP4 kernels are SM100-only. `patches/fi_moe.py` runs the 40 NVFP4 expert layers on
  FlashInfer's SM120 grouped GEMM with the checkpoint's weights (prefill goes from about 440 to about 2,000 tok/s).
- **Collectives**: the model's all-reduces assume NVLink. Here they go over RoCE (`patches/roce_collectives.py`).

The following make it faster or larger. All are launcher variables, all are on by default, and each has its own
off switch:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `KV_DTYPE=fp8_e4m3` | FP8 KV cache with a reader for SM12x: 1.9x the pool at the same speed | prefill NLL within 0.02 of BF16 |
| `GATE=0.4` | verifies only the draft rows the draft is confident of: +15 % decode | greedy text changes only at near-ties; sharing a batch with a second request changes it more |
| `DENSE_I8=draft,dhead,dfree` | the draft reads int8 copies of its own weights: +5 % | none for the target (held-out NLL identical); about 5 % of the pool |
| `DENSE_P12=attn,head,free` | the target's attention weights packed into 12 bits without rounding: −4.5 % step time for one stream | none: exact |
| `TP_SHARD=half,fp8` | every node holds all 384 experts at a quarter of their width instead of 96 whole ones, so no node waits for the busiest: −18 % step time | not bit-identical to the stock placement (rounding only; mean NLL unchanged) |
| `REASONING_CACHE=1` | the server puts an answer's reasoning back into history turns that arrive without it | prompts grow by that reasoning |

Three more grow the pool within the same bytes (defaults since 2026-11-17). None of them touches the target's
arithmetic:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `SWA_RATIO=0.05` (`--swa-full-tokens-ratio`) | halves the pool of the 40 windowed layers, which was the slack: a pool token costs 6,349 bytes a node instead of 7,578 | about 35 cached conversation ends keep their window instead of 53–58; one that lost it is prefilled again when it comes back |
| `WMARK_BOOST=1` (`SPARK_WMARK_BOOST`) | sizes the pool from the memory a node has without the kernel's temporary watermark boost, so that every boot gets the same pool (`patches/wmark_boost.py`) | none |
| `DRAFT_KV_FP8=1` (`SPARK_DRAFT_KV_FP8`) | the draft's KV pools as FP8 with a reader of their own (`patches/draft_decode.py`): 5,734 bytes a token, 793,000 tokens where 716,000 fitted | about 1 % of step time for one short stream, nothing at 4 and 8 streams; accepted tokens a step unchanged; the draft is faster behind a long context |

`POOL` is the cap on the pool (`--max-total-tokens`). When it is not set, it follows these three switches: 793,000
with all three on, 716,000 with `DRAFT_KV_FP8=`, and 600,000 with `SWA_RATIO=` as well. The three caps take the
same bytes. The numbers are in
[docs/results.md](docs/results.md#kv-cache-what-a-token-costs-and-what-grows-the-pool-2026-11-1516).

And three for prefill (defaults since 2026-11-17). Together they take a cold prefill from 2,474 to 3,110 tok/s at
30K tokens (+26 %) and from 2,241 to 2,786 tok/s at 120K (+24 %). Decode is unchanged. The first two keep every
value of the model bit for bit. The third does not:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `PREFILL_GLUE=epilogue,stream,gate,qkvr,relbias` (`SPARK_PREFILL_GLUE`) | the small steps between a layer's big kernels (residual adds, norms, the closing convolution, the projections of a global layer, the relative bias) in fewer passes over memory, same operations in the same order (`patches/prefill_glue.py`): +11 % at 30K, +10 % at 120K | none: exact (0 differing values against the stock path on every node, NLL equal token by token) |
| `BIAS_FAR=f2` (`SPARK_FA4_BIAS_FAR`) | the SM120 attention kernel rebuilt from the image's source so that key blocks beyond the reach of the relative bias skip the bias bookkeeping (`patches/fa4_bias_far.py`): 2 % of a 120K prefill, 3.6 % of a 300K one | none: exact (output and log-sum-exp equal to the stock kernel's bit for bit); refuses to boot if that source has changed |
| `FI_MXFP8=1` (`SPARK_FI_MXFP8`) | the 8 MXFP8 expert layers on FlashInfer's FP8 grouped GEMM instead of the BF16 Marlin one (`patches/fi_mxfp8.py`): +11 % at 30K, +10 % at 120K | not bit-identical: the same products summed in another order. Held-out NLL 0.9840 against 0.9840; per token the difference is 0.006 on average, above 0.1 for 0.5 % of the tokens |

Set `FI_MXFP8=` to keep the target's arithmetic as it was. The other two switches are worth 12 % on their own.
The numbers, and what was tried and dropped, are in
[docs/results.md](docs/results.md#prefill-the-glue-the-far-bias-kernel-and-mxfp8-off-marlin-2026-11-16).

And two that keep the fleet up (defaults since 2026-11-17). Two of the image's Triton kernels take the size of the
batch as a compile-time constant, so every prefill of a new size compiles and loads a CUDA module on every node
while serving; one node's cache held 816 variants of one of them after five days. On GB10 such a load can go
wrong: on the test fleet a node froze at one, and at another the driver refused the load with
`Triton Error [CUDA]: operation not permitted` and that node's server process died.

| launcher variable (default) | what it does | cost |
|---|---|---|
| `STABLE_KERNELS=tau,gate_sink` (`SPARK_STABLE_KERNELS`) | the same two kernels with the batch's row count and strides as arguments (`patches/stable_kernels.py`): one module a kernel, loaded at the first prefill after the boot | none: the same bits (422 comparisons on a GPU, `bench/stable_kernels_test.py`) |
| `KERNEL_LOAD_GUARD=5` (`SPARK_KERNEL_LOAD_GUARD`) | a kernel load the driver refuses with that error is tried again, up to five times, and logged (`patches/kernel_load_guard.py`) | none |

Why the driver fails on such a load is not known. The first switch removes the loads, and the second is for the
ones that remain. The details are in
[docs/results.md](docs/results.md#two-outages-on-2026-11-17-and-the-kernels-compiled-while-serving).

And the two newest (defaults since 2026-11-19 and 2026-11-20). The first gives memory back, and the second makes
the gate choose its rows better:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `SCALE_PACK=1` (`SPARK_SCALE_PACK`, `SPARK_CSF_GEMM`) | the block scales of the 40 NVFP4 expert layers stay packed in memory, 56 % of their bytes (`patches/scale_pack.py`), and a fork of FlashInfer's grouped GEMM, compiled from the image's own sources, reads them inside the kernel: 3.0–3.4 GB more available memory on every node | none in the results: the same bytes into the GEMM, the same text on 16 of 16 prompts, NLL equal token by token. About 1 % of step time for one stream: 0.5 to 0.7 ms a step longer on three boots than on the boots without it; none seen at four and eight streams (one boot; runs differ by up to 7 % there). The module has to be built once per image ([Run it](#run-it), step 2); without it the launcher leaves the packing off and says so |
| `GATE_HEAD=gate-head-mlp-reg.pt` (`SPARK_GATE_HEAD`) | the gate's row count comes from a small trained head that reads the draft's hidden state, computed inside the draft's CUDA graph (`patches/spark_gate_head.py`; the head file is in `patches/`). Net of its own cost, for one greedy stream: x1.017 on the test boot and x1.019 on the serving boot, against the boots without it. Inside the test boot, against the gate's own rule with both arms carrying the head's cost: x1.023 greedy and x1.050 at temperature 1.0 for one stream (net at temperature 1.0 not measured, about x1.036 estimated) | exact for sampled requests like the gate, and greedy text identical on 16 of 16 prompts one at a time; its tail in the draft graph costs about 0.75 ms a step (se 0.32), most of it the statistics; measured on short prompts only (see [Known problems](#known-problems)) |

The pool is not raised for the memory that the packing frees: that memory is margin. The format, the kernel and
the numbers of the packing are in
[docs/results.md](docs/results.md#packed-block-scales-and-the-forked-gemm-2026-11-19-the-launchers-default-scale_pack-turns-it-off).
How the head was trained, its test boot and what that boot did not measure are in
[the section after it](docs/results.md#a-confidence-head-for-the-gate-2026-11-1920-the-launchers-default-since-2026-11-20-1147-utc-gate_head-turns-it-off).

One more change is not a launcher switch: the stall recorder (since 2026-11-18). Every rank logs what it is
waiting for when a forward does not return or the scheduler stops making forwards (`[stall]` lines;
`SPARK_STALL_DUMP`, 60 s by default). Every rank also prints a running checksum of its forwards, which must be
equal on the four ranks (`[sig]` lines). Before the watchdog relaunches, it saves each rank's view
(`bench/stall_evidence.sh`). The recorder is there for a stall whose cause is not known: see
[Known problems](#known-problems) and
[docs/results.md](docs/results.md#the-silent-stall-and-what-every-rank-records-since-2026-11-18).

Every switch of these tables is turned off by setting it empty on the relaunch (`KV_DTYPE=auto` for the BF16
pool). The switches are independent of each other, with four exceptions: `FI_MXFP8` does nothing without `fp8` in
`TP_SHARD`, `DRAFT_KV_FP8` does nothing without the FP8 KV cache, `SCALE_PACK` needs `TP_SHARD`, and `GATE_HEAD`
needs `GATE`. In the last two cases the launcher turns the switch off and says so. Lever by lever, an empty
setting leaves you with: the gate's own rule (`GATE_HEAD=`); the block scales as loaded (`SCALE_PACK=`); prefill
without the three prefill switches (`PREFILL_GLUE= BIAS_FAR= FI_MXFP8=`); the experts in the stock placement and
the target's dense weights without the 12-bit packing, both as before 2026-11-15 (`TP_SHARD= DENSE_P12=`).

## What you need

- Four DGX Spark nodes (GB10, 128 GB unified memory, arm64). One of them is rank 0, and every script runs there.
- A RoCE fabric between the nodes, switched or direct. The RDMA devices must be visible in
  `/sys/class/infiniband` on every node.
- The checkpoint on the local disk of **every** node, 394 GB each. The launcher mounts each node's own copy. The
  default place is the HuggingFace cache of the launching user; `INKLING_REPO_REL` in `launch/fleet.env` moves it.
  On a node that is serving something else, run the download under a memory limit (for example
  `systemd-run --user -p MemoryMax=4G hf download ...`): on GB10 the page cache competes with the GPU for memory.
- About 37 GB on every node for the image.
- Docker with the NVIDIA container runtime; the launching user in the `docker` group on every node; key-based ssh
  from rank 0 to the other three nodes; permission to pass `/dev/infiniband` into a container and to lift the
  memlock limit.
- On rank 0: `bash`, `ssh`, `scp`, `curl`, `flock`, `nvidia-smi`, and Python 3 (standard library only) for `bench/`.
  The one-time build of step 2 of [Run it](#run-it) runs in the image and needs the node idle: several GB of
  memory, and eight cores for about 40 s on a Spark.
- Nothing else on the GPUs. The weights take 92–93 GB of each node, and the pool takes what is left.

## Run it

```bash
# 0. The checkpoint, on every node, at the revision this recipe was measured on.
hf download thinkingmachines/Inkling-2-Small --revision b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b

# 1. The image, on every node (arm64). The hooks are written against this build of the tag:
#    lmsysorg/sglang@sha256:424c1da97f6d6eef1612006c77b2ca791657a8184c1fbdbc682ebf9485d3b32c  (2026-11-12)
#    If the tag has moved since then and a boot fails in a hook, pull that digest and set INKLING_IMAGE to it.
docker pull lmsysorg/sglang:dev-inkling-2-small
#    Or that pinned build with this recipe's hooks already inside (one small layer on top, see the Dockerfile):
#      docker pull aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.0
#      export INKLING_IMAGE=aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.0     # before launching
#    The launcher runs either image. It mounts its own patches/ over the image's, so the two cannot drift apart.
#    That tag was built on 2026-11-17, before the packed block scales, the stall recorder and the confidence
#    head: its own /patches are the hooks of that day, and it holds neither the head file nor kernels/. Under
#    the launcher it runs the current hooks all the same, because of that mount. It has passed a file and
#    import check and has not been booted on a fleet; the measurements here used the upstream tag.

# 2. Once per image, on rank 0, with nothing serving on that node: build the forked grouped GEMM that reads the
#    packed block scales (SCALE_PACK, a default). The build takes about 40 s on a Spark and no GPU, but several
#    GB of memory. It compiles from the image's own CUTLASS and FlashInfer sources and stops if a line that it
#    patches has changed. Its one argument is the output directory. Use the image that you are going to launch
#    (INKLING_IMAGE).
mkdir -p "$HOME/inkling2-csf-build"
docker run --rm --network none --memory 16g --cpus 8 \
  -v "$PWD/kernels/csf_gemm":/k:ro -v "$HOME/inkling2-csf-build":/out \
  --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small /k/build_csf_gemm.py /out
cp "$HOME/inkling2-csf-build/spark_csf_gemm_sm120.so" patches/
#    Without that file the launcher prints "note: patches/spark_csf_gemm_sm120.so is not built
#    (kernels/csf_gemm/build_csf_gemm.py): the packing is off for this launch" and starts the stack without the
#    packing: the same text, 3.0-3.4 GB less available memory on every node, one stream's step about 1 % shorter.
#    Build the module again when the image changes: it belongs to the image it was built from.

# 3. The site file: nodes, users, devices. Each line of the example says how to find its value.
cp launch/fleet.env.example launch/fleet.env

# 4. Print the four docker run commands without starting anything. This already needs ssh to every rank:
#    the launcher probes the RoCE-v2 GID index on each node.
launch/launch-inkling2.sh --dry-run

# 5. Launch. The script disarms the watchdog, copies the hooks, the built module and the head file to every node,
#    removes a previous inkling2 container, starts ranks 3..0, waits for /health and arms the watchdog.
launch/relaunch-inkling2.sh

# 6. What the gate and the two newest defaults logged. Run this on each node; rank 0 is shown:
docker logs inkling2 2>&1 | grep -E 'scale_pack\] forked GEMM|gate head\]|\[gate\] (armed|SPARK_GATE|row counts)'
#    Expect six lines a rank: "nothing is expanded" twice (w13_scale, w2_scale), "[gate] armed: tau=0.4 ...",
#    "[gate head] armed on 4 rank(s): ...", "[gate] row counts from the confidence head" and
#    "[gate head] warm-ups: ranks differ=0". Without the module of step 2, the two "forked GEMM" lines are
#    missing, which leaves four lines a rank. The launcher's note about it is in the relaunch log
#    ($LOG_DIR/inkling2-relaunch-*.log); on the terminal it appears only in step 4. "NOT used" means that the
#    module failed its check and the scales are expanded per call (the same results, a 7.5 % slower step).
#    "[gate head] ... NOT armed: <why>" means that the gate runs its own rule.
#    "[gate] SPARK_GATE=... NOT armed: <why>" means that there is no gate at all, whatever the head's lines say.

# 7. A first request (port 8210 on rank 0).
curl -s http://<rank-0>:8210/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Is 91 prime?"}],"max_tokens":400}'

# 8. Read the text of overlapping requests. Do this after every boot of a changed configuration, because a
#    broken decode path looks fast and healthy in every counter.
mkdir -p logs && python3 bench/concurrency_check.py && python3 bench/reasoning_cache_check.py --expect on
```

With nothing set, the launcher starts the 500K configuration: 524,288 tokens of context, `MEMFRAC=0.94`, a pool
capped at 793,000 tokens, the three prefill switches and one scheduling setting, `--prefill-decode-interval 2`
(`PREFILL_DECODE_INTERVAL`). With that setting, the scheduler runs two decode rounds after a prefill batch before
it takes the next one, so requests that are already decoding keep getting tokens through a long prefill. A new
request still waits for the whole prefill ahead of it. The setting was not measured against a boot without it.
`PREFILL_DECODE_INTERVAL=` leaves the scheduling to the engine. On top of that configuration come the two kernel
switches of 2026-11-17 (`STABLE_KERNELS`, `KERNEL_LOAD_GUARD`), the packed block scales (since 2026-11-19, if the
module of step 2 is built) and the confidence head (since 2026-11-20).

Each of the two newest defaults has a rollback line. `GATE_HEAD=` takes exactly the head's variable out of the
four `docker run` commands, and `SCALE_PACK=` takes out exactly the packing's two. With both empty, the commands
are those of the launcher before 2026-11-19, token for token (dry-runs compared):

```bash
GATE_HEAD= launch/relaunch-inkling2.sh               # the gate's own rule: the stack before 2026-11-20
SCALE_PACK= GATE_HEAD= launch/relaunch-inkling2.sh   # and the block scales as loaded: the stack before 2026-11-19
```

The cap goes with the pool settings. The same bytes hold 716,000 tokens with `DRAFT_KV_FP8=` and 600,000 with
`SWA_RATIO=` as well, and the launcher picks the cap to match. The engine sizes down a cap that is above what the
bytes hold. The bytes are what the rank-0 node has free at boot, so the cap that fits is a property of the site.
`POOL=<tokens>` sets another cap. `POOL=` sets none, and the engine then sizes the pool at boot. For a
combination that no cap was measured for (`KV_DTYPE=auto`, another ratio), the launcher sets none and says so.

Earlier configurations with the current launcher:

```bash
# The 600,000-token pool without the prefill switches: what most rows of the table above were measured on.
SCALE_PACK= GATE_HEAD= SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= \
  launch/relaunch-inkling2.sh

# The defaults before 2026-11-17: 262,144 tokens of context, a pool sized at each boot, the engine's scheduling.
SCALE_PACK= GATE_HEAD= CTX=262144 MEMFRAC=0.93 POOL= SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= \
  BIAS_FAR= FI_MXFP8= PREFILL_DECODE_INTERVAL= launch/relaunch-inkling2.sh
```

Both lines switch the two newest defaults off (`SCALE_PACK= GATE_HEAD=`). Without that, the two would run on top
of the older configuration. The two kernel switches of 2026-11-17 (`STABLE_KERNELS`, `KERNEL_LOAD_GUARD`) stay on
in both lines on purpose: they are exact and they keep the fleet up. With `STABLE_KERNELS= KERNEL_LOAD_GUARD=` in
front as well, the lines give the commands of their day token for token (dry-runs compared).

`NCCL_EXTRA` (extra `-e NAME=value` arguments for the containers) and `EXTRA_ARGS` (extra server arguments) follow
the defaults on the command line, so a value given there wins.

To stop, kill the watchdog (its pid is in the last `watchdog started` line of `~/inkling2-watchdog.log`), then run
`launch/launch-inkling2.sh --stop`. The watchdog probes `/health` and a three-token chat completion every 60 s.
After three failures it relaunches the fleet with the environment it was started in. Before it relaunches, it
saves the state, logs and Python stacks of every rank under `$LOG_DIR/postmortem/`, and what each rank was waiting
for (`stall-rank<N>.txt`, written by `bench/stall_evidence.sh`).

The top of `launch/launch-inkling2.sh` documents every launcher variable: the ones above and the others (`PORT`,
`MAXREQ`, `CHUNK`, `SERVED`, `SERVED_ALIASES`, `MTP`, `ROCE_AR`, `FI_MOE`, `PAGED_DECODE`). Leave
`PAGED_DECODE=1`.

## Using it as a client

- **Temperature.** A chat or messages request without a `temperature` is sampled at 1.0 (the engine's default for
  those endpoints; the Responses API defaults to 0.7; the model ships no generation config). For lookups in long
  contexts and for agent work, send 0.
- **History.** The model's template renders the thinking of an earlier turn only if the client sends it back
  (`reasoning_content`, or the thinking blocks on `/v1/messages`). Without it, over a long context, the model
  answers a later question with an earlier answer. The reasoning cache covers clients that drop the thinking, for
  chat completions and `/v1/messages`. It does not cover the Responses API, and a restart empties it.
- **Output limit.** Give turns over a long context 4,000 output tokens or more, because the reasoning uses them.
  Treat `finish_reason: length` with an empty answer as no answer.
- **Reasoning effort** is a number in the prompt, not a budget. The levels are `none`, `minimal`, `low`, `medium`,
  `high` (the default), `xhigh` and `max`; chat completions also take a float. `thinking.budget_tokens` is accepted
  and not enforced, so `max_tokens` is the only hard cap. The task sets the amount of thinking more than the level
  does.
- **`tool_choice: none`** does not work: the model writes the call as text. Leave the tools out of the request
  instead.

## Known problems

- **A silent stall of the whole fleet (open).** The server can stop answering while every container is up and
  every rank is alive. It has happened twice, both times while a new request was being prefilled beside running
  decodes. The cause is not known, and a load test written to provoke it did not. The watchdog detects it and
  relaunches the fleet; every rank logs what it is waiting for (`[stall]` lines) and the watchdog saves that
  before it relaunches. There is no fix.
- **The confidence head is new and narrowly measured.** It was trained, tested and made the default on
  2026-11-20. Its test boot ran short prompts only, 24 in all, 400 tokens each: 16 coding questions for the step
  times across boots, and a mixed set of 16 (8 general questions and 8 of those coding questions) for the
  comparison inside the boot. Not measured on the test fleet: agent turns with tools, long contexts, and
  temperature 0.7 (4 prompts in a side check: x0.97 per prompt, the four between x0.88 and x1.03, x0.96 over all
  tokens). The four-stream samples are 4 and 3 batches, too few for an interval, and the one greedy batch of four
  in that side check ran x0.91. The ratios inside the test boot are not net of the head's own cost. Net was
  measured for one greedy stream only, on two boots (x1.017 and x1.019), where two boots of one configuration have
  differed by up to 0.9 % in step time. Against the boots of 2026-11-19 with neither default, the head and the
  packing together are about level for one greedy stream: the head gives back what the packing costs. The head
  that ships is fit on three fifths of one capture of templated prompts, not on all of it. Its tail costs about
  0.75 ms a step, most of it statistics that one fused kernel could read in a single pass; that kernel is not
  built and its gain is an estimate. That the head leaves sampled output exact rests, as for the gate itself, on
  how the engine's accept treats the capped row. A second reading of the engine's code confirmed that, and no test
  pins it. `GATE_HEAD=` goes back to the gate's own rule.
- **A short request waits behind a long prefill.** It waits for the whole prefill: 338 s behind a 500K one.
- **The reasoning loop on long lists.** At 130K tokens of near-identical record lines, about 1 lookup in 10 runs
  out of tokens while copying the list in its reasoning (0 at 60K, 0 on source code). This is the model's own
  behaviour: the hosted endpoints do the same on the same questions (29 of 200 across four providers, against 4 of
  40 on the test fleet). No server setting fixes it. A search tool is the right way to look things up in such
  lists.
- **The pool depends on what the rank-0 node has free at boot.** Sized without a cap, it differed from boot to
  boot (493K–805K), because the kernel's watermark boost moves the free memory by gigabytes. `WMARK_BOOST=1` (the
  default since 2026-11-17) corrects the sizing for that. The default cap of 793,000 is what the test fleet gets.
  A node with less memory free gets a smaller pool, and the boot log of rank 0 says how large.
- **Log-probability requests over long spans can take a node down.** The cause is the full-vocabulary logits of
  every scored position. Keep scored spans to a few hundred tokens.
- **No GPU work beside the serving fleet**, however small: there is no memory for it, and the driver fails
  allocations instead of evicting page cache. No compile either: build the forked GEMM (step 2 of
  [Run it](#run-it)) with nothing serving on that node. The build needs several GB of memory.
- **Heat.** Sustained all-node load runs the GPUs hot: 88–89 °C on the hottest node in most measurements, and
  95–98 °C on the rank-0 node under the heaviest test loads, although its GPU clock was capped at 2,176 MHz or
  below. Pace long load tests by temperature.

## Files

| file | what |
|---|---|
| `launch/launch-inkling2.sh` | one container per node (`inkling2`), RoCE/NCCL environment, the hooks on PYTHONPATH; `--dry-run`, `--stop` |
| `launch/relaunch-inkling2.sh` | disarm the watchdog, launch, wait for /health, re-arm the watchdog (use this to restart) |
| `launch/watchdog-inkling2.sh` | probes and relaunches (log `~/inkling2-watchdog.log`); before a relaunch it saves every rank's state, stacks and stall evidence |
| `Dockerfile` | the hooks image: the pinned base plus `patches/` on `PYTHONPATH`, published as `aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.0` (no hook is switched on in it: the launcher's variables do that). The published tag was built on 2026-11-17 and holds the hooks of that day |
| `launch/fleet.env.example` | the site file; copy to `launch/fleet.env` (git-ignored) |
| `patches/sitecustomize.py` | installs the hooks at interpreter start in every rank; the other `.py` files in `patches/` are the hooks |
| `patches/gate-head-mlp-reg.pt` | the confidence head the launcher's `GATE_HEAD` names: a 256-unit MLP, 4.7 MB, sha256 `0bc7b3d52da6c1bd...`, read by `patches/spark_gate_head.py`. Trained for this recipe on the draft's hidden states (see [docs/results.md](docs/results.md)); no weight of the model is in it |
| `kernels/csf_gemm/build_csf_gemm.py` | builds the forked grouped GEMM of the packed block scales from the image's sources into `spark_csf_gemm_sm120.so`. The module is not in the repository (`patches/*.so` is git-ignored): it belongs to the image it was built from |
| `bench/scale_pack_test.py`, `bench/csf_gemm_test.py`, `bench/dhead4_test.py` | the packing on real checkpoint scales, without a GPU; the forked GEMM against FlashInfer's, bit for bit (one GPU, the fleet stopped); the quantizer of an optional 4-bit draft head (`dhead4` in `DENSE_I8`, not a default and not measured to pay) |
| `bench/stall_evidence.sh`, `bench/stall_hunt.py`, `bench/stall_dump_check.py` | what every rank is doing right now (the watchdog runs it before a relaunch); a load test that tries to provoke the silent stall (planned windows only: it disarms the watchdog when it succeeds); the recorder checked on a CPU |
| `patches/conf_capture.py`, `bench/conf_capture_run.py`, `bench/conf_capture_check.py`, `bench/conf_capture_rank_check.py`, `bench/conf_capture_align_check.py`, `bench/conf_capture_test.py` | training data for a confidence head: the recorder (capture windows only), its load generator, the checks of a capture, the recorder's CPU test |
| `bench/conf_head_train.py`, `bench/conf_head_selftest.py`, `bench/cc_build_features.py`, `bench/cc_lab.py`, `bench/cc-run.sh`, `bench/exp_base.py`, `bench/exp_combo.py`, `bench/cc_count_ops.py` | training a head, CPU only: the first trainer and its self-test; the harness (features, folds, gate rules, the simulation, one capped container a node), the baselines, the script every head of the results comes from, operation counts |
| `bench/cc_export_head.py`, `bench/gate_head_test.py`, `bench/gate_head_capture_check.py`, `bench/gate_run.py`, `bench/gate_ab_report.py` | from a trained head to the engine: the export with its parity check, CPU checks of the engine code, a test boot's recorded rows against a recomputation, the A/B inside one boot (`gate_run.py --head`) and its summary |
| `bench/concurrency_check.py` | reads the text of overlapping requests against single-stream references. **Run it after any change to a decode path** |
| `bench/reasoning_cache_check.py` | whether the server restores a history turn's reasoning, over six wire formats |
| `bench/toolcall_check.py` | the tool-call suite (wire formats and agent loops) |
| `bench/streams.py`, `bench/step_time.py` | decode tok/s against the number of streams; step time of a boot, comparable between boots whose text differs |
| `bench/longctx.py`, `bench/longctx_edges.py`, `bench/stress.py` | long contexts end to end, their edges, mixed load |
| `bench/swa_fill.py`, `bench/prefix_fork.py` | the windowed pool run full, then old and recent conversations asked again; what a continued or a forked prompt reuses of the cache |
| `bench/stale_answer.py`, `bench/recital_rate.py`, `bench/path_compare.py` | multi-turn lookups over a long context; how often a lookup recites; decode path against prefill path |
| `bench/stable_kernels_test.py`, `bench/shape_sweep.py` | the stable kernels against the image's on one GPU, bit for bit; prompts of many lengths, after which the kernel cache shows whether a prefill size still compiles anything |
| `bench/head_mon.py` | a line of memory, pressure and temperature every 2 s, fsynced, so a freeze leaves its last seconds on disk |
| `bench/effort_tokens.py` | thinking tokens and right answers at each reasoning effort |
| `bench/glue_check.py`, `bench/bias_far_window.py`, `bench/fi_mxfp8_probe.py`, `bench/fi_mxfp8_test.py` | the prefill switches against the stock path on one idle GPU: differing values, bits, timings |
| `bench/accept_sim.py`, `bench/sampled_arms.py`, `bench/sampled_lab_check.py`, `bench/timers_report.py` | lab tools: gate rules and draft temperatures scored on dumped steps, arms of one boot side by side, the timers' log lines as a table |
| `bench/*_test.py`, `bench/compile_check*.py`, `bench/emu/` | kernel tests (idle GPU only, their docstrings say so) and the offline checks that run without a GPU |
| `docs/results.md` | the lab notes: every hook, lever and number, in the order they were found |

The tools that talk to the server send ordinary requests. Most say in their docstring whether they are safe on a
serving fleet, and most write their results to `logs/` in this directory (create it once). `bench/window.py` also
reads `logs/code_prompt.json`, which you supply: a chat request body with a source file as its first message. The
capture checks and the training tools of the confidence head need numpy or torch. They run with the image's Python
in a CPU-only container, and their docstrings give the command. No capture is in the repository, so training a
head of your own starts with a capture window on your own fleet.

## Licence

The licence is MIT (see `LICENSE` at the top of this repository). `NOTICE` lists third-party material and what is
not included. The image and the model weights are not part of this repository; they come under their own terms.
