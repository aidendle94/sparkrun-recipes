# Inkling-2-Small on four DGX Sparks with an NVFP4 KV cache (experimental)

This recipe serves [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) on
four NVIDIA DGX Spark nodes behind one OpenAI-compatible endpoint, with the KV cache stored in NVFP4 (4-bit). It is
the stack of [`../sglang`](../sglang/) with one switch on (`KV_NVFP4=target,draft`) and the pre-sharded weights,
plus a script that sets everything up.

Against the same stack with the FP8 KV cache, measured on one fleet on 2026-11-23:

| | FP8 KV cache (`../sglang`) | NVFP4 KV cache (this recipe) |
|---|---|---|
| KV cache, same memory | 793,000 tokens | **1,409,000 tokens** |
| decode, one stream, short prompts | 54.0 tok/s | 56.9 tok/s |
| decode, one stream, behind 130K tokens of context | 54.1 tok/s | 55.7 tok/s |
| decode, four streams behind 130K, per stream | 23.6 tok/s | 25.4 tok/s |
| decode, one stream, behind 500K tokens of context | 47.6 tok/s | **52.2 tok/s** |
| held-out NLL, 10,547 tokens | 1.4437 | 1.4437 (difference +0.0001 ± 0.0022) |
| lookups in a 130K-token context, 40 questions | 36 right | 35 right, no wrong answer in either |

Decode is at temperature 1.0, one boot each. The lookup misses on both caches are the model reciting the context
or running out of its 1,500 tokens before answering, about one lookup in ten; see Known problems in `../sglang`.

**Status: experimental.** The engine is the SGLang team's development image for this model, and the GB10 hooks
that make it run here are this repository's own. One fleet has served it. Known problems are listed in
[`../sglang`](../sglang/README.md#known-problems).

## Quick start

You need:

- **Four DGX Sparks** connected by their 200G ports (a switch, or a direct ring), with an IPv4 address on that port
  of every node. This is the "fabric address" below. `ip -br addr` on a node lists it next to an interface named
  like `enp1s0f0np0`.
- **ssh without a password** from the first node to the other three. On the first node, once for each other node:
  `ssh-copy-id <address>`.
- **Docker** usable without sudo on every node: `sudo usermod -aG docker $USER`, then log out and in again.
- **Disk**: about 130 GB free in your home directory on every node (93 GB of weights, 37 GB of image).
- **Nothing else running** on the GPUs. The model uses nearly all memory of every node.

Then, on the first node:

```bash
git clone https://github.com/aidendle94/sparkrun-recipes
cd sparkrun-recipes/recipes/inkling-2-small/sglang-nvfp4-kv
./quickstart.sh 192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14   # your four fabric addresses, this node first
```

Write `user@address` for a node whose login is not yours. The script prints what it does, step by step:

1. writes `launch/fleet.env` from the four addresses (it reads the interface, RDMA devices, subnet and home
   directories from the nodes);
2. checks every node (ssh, docker, GPU, memory, disk, RDMA) and stops with the fix if one is missing;
3. pulls the image on every node;
4. copies a prebuilt kernel module out of the image;
5. downloads each node's own folder of the weights, about 93 GB a node, to `~/inkling2-weights` (an hour or more,
   depending on the line; all four at once, progress every minute);
6. starts the fleet and waits until it answers: about 17 minutes the first time, while kernels compile, and about
   5 after that;
7. checks that every node runs the NVFP4 KV cache, sends one chat request, and prints the endpoint.

If a step fails, fix what it says and run the same command again. Finished steps are skipped and a stopped
download resumes. `./quickstart.sh --check` runs only steps 1 and 2 and changes nothing.

## Use it

```bash
curl -s http://<first-node>:8210/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Is 91 prime?"}],"max_tokens":400}'
```

Any OpenAI-compatible client works with base URL `http://<first-node>:8210/v1` and model `inkling-2-small`. The
endpoint also speaks the Anthropic `/v1/messages` API and takes pictures; see
[Using it as a client](../sglang/README.md#using-it-as-a-client).

## Run it

| what | command |
|---|---|
| status | `./quickstart.sh --status` |
| stop | `./quickstart.sh --stop` |
| start or restart | `launch/relaunch.sh` |
| the same stack with the FP8 KV cache | `KV_NVFP4= launch/relaunch.sh` |

A watchdog runs on the first node after every start. It sends a request every 60 s and restarts the fleet with the
same settings after three failures; `--stop` stops it first. Logs: `~/inkling2-logs/` and `~/inkling2-watchdog.log`
on the first node, `docker logs inkling2` on every node.

Every setting of the launcher in `../sglang` works here too, in front of `launch/relaunch.sh`. The top of
[`../sglang/launch/launch-inkling2.sh`](../sglang/launch/launch-inkling2.sh) lists them. This recipe sets four of
them in [`launch/settings.sh`](launch/settings.sh), and a value set on the command line wins:

| setting | here | |
|---|---|---|
| `KV_NVFP4` | `target,draft` | the NVFP4 KV cache, for the model and its draft |
| `PRESHARDED` | `1` | every node loads its own 93 GB folder instead of the 394 GB checkpoint |
| `INKLING_REPO_REL` | `inkling2-weights` | where that folder is, under each node's home |
| `INKLING_IMAGE` | `aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.5` | the engine image with this repository's hooks |

The pool cap follows the cache: 1,409,000 tokens hold the bytes that 793,000 FP8 tokens did, so the memory left on
every node is the same as with `../sglang`'s defaults.

## What the NVFP4 KV cache is

Each attention head stores 128 numbers per token for K and 128 for V. FP8 keeps each in a byte: 128 bytes. NVFP4
keeps 4 bits a number plus one 8-bit scale per 16 numbers: 72 bytes. The scale of a block is chosen by the rule
the checkpoint uses for its own 4-bit activations (the better of amax / 6 and amax / 4). Rounding K and V to 4 bits
moves single tokens' probabilities a little: 31% of tokens by more than 0.1 nat against FP8. The average does not
move.

The speed comes from the readers in [`../sglang/patches/nvfp4_kv.py`](../sglang/patches/nvfp4_kv.py): GB10 turns two
4-bit codes into two 16-bit values in one instruction, the attention runs in FP16, and the far part of a long
context skips the relative-position bias. A long context is read faster than from the FP8 cache, so decode behind
a long prompt gains the most. The SGLang image's own `--kv-cache-dtype nvfp4` runs on datacenter Blackwell only.
Measurements, tests and the kernel steps are in [`../sglang/README.md`](../sglang/README.md#nvfp4-kv-cache) and
[`../sglang/docs/results.md`](../sglang/docs/results.md).

## When something goes wrong

| what you see | what to do |
|---|---|
| `cannot log in to ... without a password` | `ssh-copy-id <user>@<address>` from the first node, then run the script again |
| `docker does not answer for user ...` | on that node: `sudo usermod -aG docker <user>`, log out and in |
| `no RDMA device` | the 200G port has no RoCE device: the fabric is not set up on that node (see [fleet/](../../../fleet/)) |
| `... GB free, ... GB needed` | free disk space in that node's home, or set `INKLING_REPO_REL` to a directory on a larger disk |
| `is not an address of this node` | run the script on the node you list first |
| the launch stops with a hook error | `docker logs inkling2 2>&1 \| grep -m5 -iE 'error\|abort'` on each node |
| it serves, but answers are garbage | `python3 ../sglang/bench/concurrency_check.py` (it reads the text of overlapping requests) |
| you want FP8 back | `KV_NVFP4= launch/relaunch.sh` |

## Licence

MIT, like the rest of this repository. The engine image is the SGLang team's (Apache-2.0), and the weights are
Thinking Machines' (Apache-2.0). See [`../sglang/NOTICE`](../sglang/NOTICE).
