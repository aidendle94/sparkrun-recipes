#!/usr/bin/env python3
"""One MoE layer with DeepSeek-V4.1's dimensions: FlashInfer CUTLASS MXFP8xMXFP4 (SGLang's SM120 path) vs b12x
fused MoE (w4a8_mx), identical synthetic weights and inputs. Prints relative error and median latency per token count.
Run in the image with a GPU and nothing else loaded: docker run --rm --gpus all -v $PWD/tools:/tools --entrypoint python3 <image> /tools/moe_microbench.py
MIT License, Copyright (c) 2026 Aiden Le."""
import sys, time, torch
import b12x.moe.fused_moe as fm
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
torch.manual_seed(0); dev = torch.device("cuda", 0)
H, TOPK, LIMIT = 5120, 6, 10.0
COUNTS = [6, 24, 48, 96, 512, 2048]

def make(E, I):
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)      # [up; gate]
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)
    return w13, w2, s13, s2

def timed(fn, iters=40):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2] * 1000.0

def cutlass_prep(w13, w2, s13, s2, E):
    cw13, cw2, cs13, cs2 = w13.clone(), w2.clone(), s13.clone(), s2.clone()
    for sc in (cs13, cs2): sc.copy_(block_scale_interleave(sc).reshape_as(sc))
    return cw13, cw2, cs13, cs2, torch.ones(E, dtype=torch.float32, device=dev), torch.full((E,), LIMIT, dtype=torch.float32, device=dev)

def b12x_prep(w13, w2, s13, s2, E, I):
    bw13, bw2 = w13.clone(), w2.clone(); bs13, bs2 = s13.clone().view(torch.float8_e8m0fnu), s2.clone().view(torch.float8_e8m0fnu)
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    wplan = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu", params_dtype=torch.bfloat16,
                            num_experts=E, hidden_size=H, intermediate_size=I, w13_layout="up_gate")
    return fm.prepare_weights(plan=wplan, w1_fp4=bw13, w1_blockscale=bs13, w1_global_scale=ones, a1_gscale=ones,
                              w2_fp4=bw2, w2_blockscale=bs2, w2_global_scale=ones, a2_gscale=ones, params_dtype=torch.bfloat16)

def cutlass_fn(c, x, ids, wts, out, M, ep_size=1, ep_rank=0):
    cw13, cw2, cs13, cs2, g, lim = c
    def f():
        xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        cutlass_fused_moe(input=xq, token_selected_experts=ids, token_final_scales=wts, fc1_expert_weights=cw13.view(torch.int64),
                          fc2_expert_weights=cw2.view(torch.int64), output_dtype=torch.bfloat16,
                          quant_scales=[cs13.view(torch.int32), g, cs2.view(torch.int32), g], input_sf=xsf, swiglu_limit=lim,
                          ep_size=ep_size, ep_rank=ep_rank, use_mxfp8_act_scaling=True, activation_type=ActivationType.Swiglu,
                          tune_max_num_tokens=1 << (M - 1).bit_length(), output=out)
    return f

def b12x_fn(prep, x, ids, wts, out, M):
    plan = fm.plan(fm.Caps(max_tokens=M, num_topk=TOPK, device=dev, weight_plan=prep.plan, core_token_counts=(M,), route_num_experts=0,
                           quant_mode="w4a8_mx", apply_router_weight_on_input=False, swiglu_limit=LIMIT, swiglu_alpha=None,
                           swiglu_beta=None, frozen=True))
    scratch = torch.empty(int(plan.scratch_specs()[0].shape[0]), dtype=torch.uint8, device=dev)
    return lambda: fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=prep, topk_weights=wts, topk_ids=ids, output=out,
                                          input_scales_static=True, unit_scale_contract=False))

def inputs(M, route_E):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(route_E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    wts = (torch.softmax(torch.randn(M, TOPK, device=dev), -1) * 1.5).float().contiguous()
    return x, ids, wts

# 1. Same weights, same shape (production's per-rank EP4 experts: 96 x 2304): correctness and kernel speed
E, I = 96, 2304
w = make(E, I); c = cutlass_prep(*w, E); p = b12x_prep(*w, E, I); del w; torch.cuda.empty_cache()
print(f"== 1. same shape, 96 experts x 2304, routing over the 96: CUTLASS vs b12x", flush=True)
print(f"{'tokens':>7} {'cutlass us':>11} {'b12x us':>9} {'speedup':>8} {'rel err':>8}", flush=True)
for M in COUNTS:
    x, ids, wts = inputs(M, E); oc = torch.empty(M, H, dtype=torch.bfloat16, device=dev); ob = torch.empty_like(oc)
    fc, fb = cutlass_fn(c, x, ids, wts, oc, M), b12x_fn(p, x, ids, wts, ob, M)
    tc, tb = timed(fc), timed(fb)
    err = ((ob.float() - oc.float()).norm() / oc.float().norm().clamp_min(1e-9)).item()
    print(f"{M:>7} {tc:>11.1f} {tb:>9.1f} {tc/tb:>7.2f}x {err:>8.4f}", flush=True)
del c, p; torch.cuda.empty_cache()

# 2. Production-equivalent per rank: CUTLASS EP4 (96 local of 384 full experts) vs b12x TP4 slice (384 x 576)
w = make(96, 2304); c = cutlass_prep(*w, 96); del w
w = make(384, 576); p = b12x_prep(*w, 384, 576); del w; torch.cuda.empty_cache()
print(f"\n== 2. per-rank work: CUTLASS EP4 (96 of 384 experts x 2304) vs b12x TP4 slices (384 x 576), routing over 384", flush=True)
print(f"{'tokens':>7} {'cutlass EP4 us':>15} {'b12x TP4 us':>12} {'speedup':>8}", flush=True)
for M in COUNTS:
    x, ids, wts = inputs(M, 384); oc = torch.empty(M, H, dtype=torch.bfloat16, device=dev); ob = torch.empty_like(oc)
    tc = timed(cutlass_fn(c, x, ids, wts, oc, M, ep_size=4, ep_rank=0)); tb = timed(b12x_fn(p, x, ids, wts, ob, M))
    print(f"{M:>7} {tc:>15.1f} {tb:>12.1f} {tc/tb:>7.2f}x", flush=True)
print("MICROBENCH DONE", flush=True)
