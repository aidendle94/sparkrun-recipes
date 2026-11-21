"""Inkling-2's vision encoder on GB10 against the engine's own CPU path: one GPU, the fleet stopped.

    docker run --rm --gpus all --network none -v <snapshot dir>:/m:ro -v <recipe dir>:/r:ro \
      -e PYTHONPATH=/r/patches -e SPARK_VISION=1 --entrypoint /opt/sglang/bin/python3 <image> /r/bench/vision_check.py

Builds sglang.srt.models.inkling2_vision.Inkling2VisionEncoder from the checkpoint's vision_config, loads the 422
model.visual.* tensors, and encodes the same random images twice: on the CPU (the engine's reference branch: torch
rotary, einsum position embeddings, per-image SDPA) and on the GPU with patches/vision_sm12x.py (the engine's GPU
rotary and position kernels, per-image SDPA). Images of several sizes, one at a time and several in one call.
A third copy runs on the CPU in float32 as the reference for both: BF16 rounding alone moves a 23-layer encoder by a
few percent, so the GPU path passes when it is no further from float32 than the engine's own BF16 CPU path is
(mean relative difference at most 1.5 times the CPU's, plus 0.005) and its cosine to float32 is above 0.999.
Then, on the GPU: one frame (the hook's transport) against the engine's two identical frames, and the encoder in
groups of one picture against one pass (bit for bit, or within the same bound); that the kernel probe and an encode
leave the GPU generator's offset where it was (rank 0 encodes alone, and a moved generator makes it sample other
tokens than the other ranks at temperature > 0); and the time and peak GPU memory
of the largest picture at the default cap (2,048 px, 52 x 52 prompt tokens) and at 4,096 px.
With --cpu-checks DIR (no GPU needed): (a) the lean patch builder of patches/vision_sm12x.py against the engine's
own Inkling2ImageProcessor._encode on every picture in DIR, bit for bit; (b) the encoder in groups against one
pass, and on one frame against two identical frames, on the CPU, bit for bit.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

SNAP = os.environ.get("SNAP", "/m")
cfg = json.load(open(os.path.join(SNAP, "config.json")))
vc = cfg["vision_config"]
text_hidden = cfg["text_config"]["hidden_size"]

import sitecustomize  # noqa: F401,E402  (the hooks: SPARK_VISION=1 installs vision_sm12x on import)
from sglang.srt.models import inkling2_vision as V  # noqa: E402

idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
names = [k for k in idx if k.startswith("model.visual.")]
shards = sorted(set(idx[k] for k in names))


def build(device):
    enc = V.Inkling2VisionEncoder(vc, text_hidden)
    sd = {}
    for sh in shards:
        with safe_open(os.path.join(SNAP, sh), framework="pt") as f:
            for k in names:
                if idx[k] == sh:
                    sd[k.removeprefix("model.visual.")] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("_clip") for m in missing), missing
    return enc.to(device).eval()


if "--cpu-checks" in sys.argv:
    import vision_sm12x as H
    from sglang.srt.multimodal.inkling2 import image_processing as IP
    d = sys.argv[sys.argv.index("--cpu-checks") + 1]
    stock_encode = IP.Inkling2ImageProcessor._encode.__wrapped__ if hasattr(IP.Inkling2ImageProcessor._encode, "__wrapped__") else IP.Inkling2ImageProcessor._encode
    proc = IP.Inkling2ImageProcessor.__new__(IP.Inkling2ImageProcessor)
    ok = True
    for name in sorted(os.listdir(d)):
        if not name.endswith((".png", ".jpg", ".webp")) or name in ("truncated.png",):
            continue
        raw = open(os.path.join(d, name), "rb").read()
        a = stock_encode(proc, raw, 200005)            # the engine's code (with the size cap, same for both)
        b = IP.Inkling2ImageProcessor._encode(proc, raw, 200005)
        bp = b.vision_patches_bthwc.expand(-1, a.vision_patches_bthwc.shape[1], -1, -1, -1)  # one frame -> the engine's two
        same = (torch.equal(a.vision_patches_bthwc.contiguous().view(torch.int16), bp.contiguous().view(torch.int16))
                and torch.equal(a.vision_grid_thw, b.vision_grid_thw) and torch.equal(a.vision_original_thw, b.vision_original_thw)
                and a.int_tokens == b.int_tokens)
        ok &= same
        print(f"patches {name}: grid {b.vision_grid_thw.tolist()}, {'bit-identical' if same else 'DIFFERENT'}")
    enc = build("cpu")
    side = vc["native_patch_size"] * vc["adapter_merge_size"]
    grid = [(1, 3, 4), (1, 2, 2), (1, 5, 3), (1, 1, 1)]
    n = sum(h * w for _, h, w in grid)
    gen = torch.Generator().manual_seed(3)
    patches = (torch.rand((n, 2, side, side, 3), generator=gen) * 2 - 1).to(torch.bfloat16)
    thw = torch.tensor(grid, dtype=torch.int64)
    with torch.no_grad():
        H.TOKEN_BUDGET = 0
        one = enc(patches, thw)
        H.TOKEN_BUDGET = 70                              # forces groups of one or two pictures
        grouped = enc(patches, thw)
    same = torch.equal(one.view(torch.int16), grouped.view(torch.int16))
    ok &= same
    print(f"encoder in groups (budget 70, {H._S['groups']} groups) against one pass: {'bit-identical' if same else 'DIFFERENT'}")
    p1 = (torch.rand((n, 1, side, side, 3), generator=gen) * 2 - 1).to(torch.bfloat16)
    with torch.no_grad():
        H.TOKEN_BUDGET = 0
        two = enc(p1.expand(-1, 2, -1, -1, -1).contiguous(), thw)    # the engine's two identical frames
        single = enc(p1, thw)                                         # one frame, widened by the hook
    same = torch.equal(two.view(torch.int16), single.view(torch.int16))
    ok &= same
    print(f"encoder on one frame against two identical frames: {'bit-identical' if same else 'DIFFERENT'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

side = vc["native_patch_size"] * vc["adapter_merge_size"]
cases = [[(1, 2, 3)], [(1, 8, 8)], [(1, 16, 24)], [(1, 4, 6), (1, 10, 7), (1, 3, 3)], [(1, 26, 26)]]
g = torch.Generator().manual_seed(7)
cpu = build("cpu")
gpu = build("cuda")
ref32 = build("cpu").float()
print(f"vision encoder: {sum(p.numel() for p in cpu.parameters()) / 1e6:.1f}M parameters, {len(names)} tensors; "
      f"GPU capability {torch.cuda.get_device_capability()}")
ok = True
for grid in cases:
    n = sum(h * w for _, h, w in grid)
    patches = (torch.rand((n, vc["temporal_patch_size"], side, side, 3), generator=g) * 2 - 1).to(torch.bfloat16)
    thw = torch.tensor(grid, dtype=torch.int64)
    with torch.no_grad():
        f32 = ref32(patches, thw).float()
        ref = cpu(patches, thw).float()
        gpu(patches.cuda(), thw)  # warm-up (Triton and torch.compile)
        torch.cuda.synchronize(); t0 = time.time()
        out = gpu(patches.cuda(), thw).float().cpu()
        torch.cuda.synchronize(); ms = (time.time() - t0) * 1e3
    scale = f32.abs().mean().item()
    rel = lambda a: (a - f32).abs().mean().item() / scale
    cos = lambda a: torch.nn.functional.cosine_similarity(a.flatten(), f32.flatten(), dim=0).item()
    e_cpu, e_gpu = rel(ref), rel(out)
    good = out.shape == f32.shape and e_gpu <= 1.5 * e_cpu + 0.005 and cos(out) > 0.999
    ok &= good
    print(f"grid {grid}: {tuple(out.shape)} tokens out; against float32: CPU BF16 {e_cpu:.2e} (cos {cos(ref):.6f}), "
          f"GPU hook {e_gpu:.2e} (cos {cos(out):.6f}); GPU vs CPU BF16 {(out - ref).abs().mean().item() / scale:.2e}; "
          f"GPU {ms:.1f} ms -> {'ok' if good else 'FAIL'}")
import vision_sm12x as H
grid = cases[3]                                       # three pictures: grouped (small budget) against one pass
n = sum(h * w for _, h, w in grid)
p1 = (torch.rand((n, 1, side, side, 3), generator=g) * 2 - 1).to(torch.bfloat16)
thw = torch.tensor(grid, dtype=torch.int64)
with torch.no_grad():
    f32 = ref32(p1.expand(-1, vc["temporal_patch_size"], -1, -1, -1).contiguous(), thw).float()
    budget = H.TOKEN_BUDGET
    H.TOKEN_BUDGET = 0
    two = gpu(p1.expand(-1, vc["temporal_patch_size"], -1, -1, -1).contiguous().cuda(), thw).float().cpu()
    single = gpu(p1.cuda(), thw).float().cpu()        # one frame, widened by the hook
    H.TOKEN_BUDGET = 300                               # one picture a group
    grouped = gpu(p1.cuda(), thw).float().cpu()
    H.TOKEN_BUDGET = budget
scale = f32.abs().mean().item()
for name, a in (("one frame against two identical frames", single), ("in groups against one pass", grouped)):
    same = torch.equal(a, two)
    diff = (a - two).abs().mean().item() / scale
    good = same or ((a - f32).abs().mean().item() / scale <= 1.5 * (two - f32).abs().mean().item() / scale + 0.005)
    ok &= good
    print(f"GPU, {name}: {'bit-identical' if same else f'differs by {diff:.2e} (mean, relative)'} -> {'ok' if good else 'FAIL'}")
# Random numbers: an encode on rank 0 alone must not move the GPU generator, or rank 0 samples other tokens than the
# other ranks at temperature > 0 and the fleet deadlocks. The kernel probe runs again here, then one encode.
H._S["backends"] = None
gen = torch.cuda.default_generators[torch.cuda.current_device()]
before = gen.get_offset()
with torch.no_grad():
    gpu(p1.cuda(), torch.tensor(cases[3], dtype=torch.int64))
torch.cuda.synchronize()
good = gen.get_offset() == before and not H._S.get("rng_logged")
ok &= good
print(f"GPU generator offset across the kernel probe and an encode: {before} -> {gen.get_offset()}"
      f"{' (the hook had to put it back)' if H._S.get('rng_logged') else ''} -> {'ok' if good else 'FAIL'}")
for grid in ([(1, 52, 52)], [(1, 103, 103)]):          # the largest picture at the default cap, and at 4,096 px
    n = sum(h * w for _, h, w in grid)
    pt = ((torch.rand((n, 1, side, side, 3), generator=g) * 2 - 1).to(torch.bfloat16)).cuda()
    thw = torch.tensor(grid, dtype=torch.int64)
    with torch.no_grad():
        gpu(pt, thw)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        out = gpu(pt, thw)
        torch.cuda.synchronize(); ms = (time.time() - t0) * 1e3
    print(f"GPU timing, grid {grid} ({n} prompt tokens, {n * 4 + 1} encoder tokens): {ms:.0f} ms, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
