"""CPU check of patches/scale_pack.py's packing on real checkpoint scales (the Triton expansion kernel is checked on
the GPU by `adopt` at every load): pack -> expand_reference must give back the bytes, in the 128x4 layout of a shard.
    docker run --rm --memory 3g --network none -v $PWD:/w:ro -v <repo>:/repo:ro --entrypoint /opt/sglang/bin/python3 \
        <image> -I /w/bench/scale_pack_test.py /repo/snapshots/<snapshot>
MIT License, Copyright (c) 2026 Aiden Le."""
import json, os, struct, sys
import torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "patches"))
import scale_pack as sp


class Snapshot:
    """Rows of a checkpoint tensor's leading dimension, read by offset (no safetensors import, no whole file)."""

    def __init__(self, path):
        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]

    def read(self, name, first, count):
        with open(os.path.join(self.path, self.map[name]), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            t = json.loads(fh.read(n))[name]
            a, b = t["data_offsets"]
            per = (b - a) // t["shape"][0]
            fh.seek(8 + n + a + first * per)
            buf = bytearray(fh.read(count * per))
        return torch.frombuffer(buf, dtype=torch.uint8).view(count, *t["shape"][1:])

def swizzle(sf):  # fi_moe.swizzle_scales_128x4
    experts, n, k = sf.shape
    n_pad, k_pad = -(-n // 128) * 128, -(-k // 4) * 4
    x = torch.zeros(experts, n_pad, k_pad, dtype=torch.uint8); x[:, :n, :k] = sf
    return x.view(experts, n_pad // 128, 4, 32, k_pad // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().view(experts, n_pad, k_pad)

snap = Snapshot(sys.argv[1]); raw = packed = 0
for layer in (0, 21, 39):
    p = f"model.llm.layers.{layer}.mlp.experts."
    for which, rows, cols in (("w13", slice(1152, 2304), slice(0, 288)), ("w2", slice(0, 4608), slice(72, 108))):
        s = swizzle(snap.read(p + which + "_weight_scale", 100, 20)[:, rows, cols].contiguous())
        q = sp.pack(s, experts_at_a_time=7)
        assert torch.equal(sp.expand_reference(q), s), (layer, which)
        raw += s.numel(); packed += q.nbytes()
        print(f"layer {layer} {which}: {tuple(s.shape)} -> {q.nbytes() / s.numel():.1%}, raw groups {q.raw.shape[0] / (q.groups * s.shape[0]):.2%}")
# edge cases: constant groups, a span of exactly 15 and of 16, zeros
t = torch.zeros(2, 128, 4, dtype=torch.uint8); t[0, 0, 0] = 15; t[1, 5, 1] = 16; t[1, 64:, :] = 0x7E
q = sp.pack(t); assert torch.equal(sp.expand_reference(q), t) and q.raw.shape[0] == 1 + 0, q.raw.shape
print(f"PASS: {packed / raw:.1%} of the scale bytes")
