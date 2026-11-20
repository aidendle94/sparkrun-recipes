"""How many tensor operations one forward of a confidence head takes (a proxy for GPU kernel launches per step).

    python3 -I cc_count_ops.py [--data /d --out /o]        (through cc-run.sh: CPU only, seconds)

Builds exp_combo.py's Net for a few configurations (the classes are taken from the script's source; its main body is
not run), runs one forward for one request-step (7 positions) under the CPU profiler, and counts the leaf aten
operations that compute something (views and shape bookkeeping left out). The Python path of the transformer layers
is profiled, not the fused inference fast path, so that every operation is visible. Not counted, because every head
needs them alike: the projection of the 4,608-dim hidden state on 256 directions (one GEMM), the scalar features
(top-2 log p and entropy of 7 distributions, running sums) and the row rule itself.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import ast
import collections
import json
import os
import sys

import torch
from torch import nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "exp_combo.py")).read()
tree = ast.parse(src)
ns = {"torch": torch, "nn": nn, "Fn": Fn, "F": Fn, "G": 7, "__name__": "cc_count_ops_model"}
keep = []
for node in tree.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("ResBlock", "encoder", "Net", "parse_cfg"):
        keep.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEF" for t in node.targets):
        keep.append(node)
exec(compile(ast.Module(body=keep, type_ignores=[]), "exp_combo.py (model classes)", "exec"), ns)

VIEWS = {"view", "reshape", "transpose", "permute", "expand", "select", "slice", "unsqueeze", "squeeze", "as_strided",
         "t", "contiguous", "chunk", "unflatten", "flatten", "alias", "detach", "to", "unbind", "split", "narrow",
         "expand_as", "view_as", "lift_fresh", "resolve_conj", "resolve_neg", "empty", "empty_like", "zeros",
         "zeros_like", "ones_like", "result_type", "item", "_local_scalar_dense", "size", "dim", "numpy_T", "mT",
         "unsafe_chunk", "unsafe_split", "_unsafe_view", "type_as", "clone", "dropout", "feature_dropout", "copy_",
         "_to_copy", "full", "fill_", "scalar_tensor", "broadcast_to", "movedim", "swapaxes", "diagonal", "split_with_sizes",
         "empty_strided", "_reshape_alias", "_unsafe_view", "conj", "_conj", "is_same_size", "set_", "resize_"}


def computes(e):
    return e.name.startswith("aten::") and e.name[6:] not in VIEWS


def below(e):
    """Does any operation nested under e compute something?"""
    return any(computes(ch) or below(ch) for ch in e.cpu_children)


REG = "idrop=0.5,ep=24,lam=0.3,aux=p4+a6+p5,auxw=0.3,drop=0.1"
CFGS = [
    ("mlp (the baseline head)", "mlp:"),
    ("mlp-tok", "mlp-tok:tok=1,tokin=trunk"),
    ("lite-tok (linear map of 7 hidden states + 2 lags)", f"lite:hctx=cat,hk=32,tctx=lag2,tok=1,{REG}"),
    ("two-stream (model lane, L2-ep24)", f"L2:hctx=tf,tctx=catt,tk=64,hk=32,L=2,{REG}"),
    ("two-stream + token (FINAL)", f"tok:hctx=tf,tctx=catt,tk=64,hk=32,L=2,tok=1,{REG}"),
]
P, NS, NV = 256, 24, 1900
out = []
for label, cfg in CFGS:
    _, c = ns["parse_cfg"](cfg)
    torch.manual_seed(0)
    net = ns["Net"](c, P, NS, 0, [8, 9], nv=NV if c["tok"] else 0).eval()       # no auxiliary heads at inference
    x = torch.randn(1, 7, P + NS)
    cls = torch.zeros(1, dtype=torch.long)
    ids = torch.randint(0, NV, (1, 7)) if c["tok"] else None
    net(x, cls, ids)                                                            # warm-up
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        net(x, cls, ids)                                                        # grad on: the layers' Python path
    hist = collections.Counter()
    for e in prof.events():
        if not computes(e) or below(e):
            continue                # the innermost computing operations only (aten::linear -> its addmm, and so on)
        hist[e.name[6:]] += 1
    n = sum(hist.values())
    params = sum(p.numel() for p in net.parameters())
    print(f"{label}: {n} operations, {params} parameters")
    print("   " + ", ".join(f"{k} x{v}" for k, v in hist.most_common()))
    out.append({"head": label, "cfg": cfg, "operations": n, "parameters": params, "histogram": dict(hist)})
target = next((sys.argv[i + 1] for i, a_ in enumerate(sys.argv[:-1]) if a_ == "--out"), None)
if target and os.path.isdir(target):
    json.dump(out, open(os.path.join(target, "cc-count-ops.json"), "w"), indent=1)
