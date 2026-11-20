"""CPU check of the 4-bit draft head's quantizer in patches/dense_i8.py (dhead4): encode -> decode on a weight with the
unembedding's statistics; the kernel itself is checked on the GPU at every load (check_f4).
    docker run --rm --memory 3g --network none -e SPARK_DENSE_I8=dhead4 -v $PWD:/w:ro --entrypoint /opt/sglang/bin/python3 <image> -I /w/bench/dhead4_test.py
MIT License, Copyright (c) 2026 Aiden Le."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import dense_i8 as d

torch.manual_seed(0)
w = torch.randn(300, 4608) * torch.rand(300, 1) * 0.05
w[7] = 0  # an all-zero row
w[9, 16:32] = 0  # an all-zero block
q, err, ref = d.quantize_f4(w.clone(), rows=128)
back = d.decode_f4(q)
rel = float(((back - w).pow(2).sum() / w.pow(2).sum()).sqrt())
assert abs(rel - (err / ref) ** 0.5) < 1e-6, (rel, (err / ref) ** 0.5)
assert q.codes.shape == (300, 2304) and q.blocks.shape == (300, 288) and float(back[7].abs().max()) == 0 and float(back[9, 16:32].abs().max()) == 0
assert float((back.abs().amax(1) - w.abs().amax(1)).abs().max()) < 1e-6 * 1 + 0.07 * float(w.abs().max())  # the row maximum keeps its magnitude
q8, e8, r8 = d.quantize(w.clone(), round_weight=False)
again, e2, _ = d.quantize_f4(back.clone())
assert torch.equal(again.codes & 0x77 | again.codes & 0x88 & ((again.codes & 0x07) > 0) * 0x08 | again.codes & 0x80 & ((again.codes & 0x70) > 0) * 0x80, q.codes) or float((d.decode_f4(again) - back).abs().max()) < 1e-7, "not a fixed point"
print(f"PASS: 4-bit relative error {rel:.4f} (int8 {(e8 / r8) ** 0.5:.4f}); {(q.codes.numel() + q.blocks.numel() + 4 * q.scale.numel()) / w.numel() * 8:.2f} bits a weight")
