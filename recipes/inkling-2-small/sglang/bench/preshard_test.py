"""patches/preshard.py on a toy model whose load_weights has the engine's loop shape (dtype checks, weight loaders that
slice, a post-step after the loop). CPU only, no server, no checkpoint:

    docker run --rm --network none -v "$PWD":/r:ro --entrypoint /opt/sglang/bin/python3 \
      lmsysorg/sglang:dev-inkling-2-small /r/bench/preshard_test.py

Checks: an export boot writes the rank folder; a reload from it gives the plain boot's parameters bit for bit and the
same post-step, with every weight loader restored; a wrong rank, another TP_SHARD setting and a changed byte in a file
are each refused; the files are standard safetensors (the library reads them back to the same tensors). Prints PASS.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import os, sys, json, types, shutil, importlib, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
OUT = "/tmp/ps"; SRC = "/tmp/src"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(SRC, exist_ok=True)
for f in ("config.json", "hf_quant_config.json"):
    open(os.path.join(SRC, f), "w").write("{}")
RANK = {"r": 1, "w": 4}
rc = types.ModuleType("sglang.srt.runtime_context")
rc.get_parallel = lambda: types.SimpleNamespace(tp_rank=RANK["r"], tp_size=RANK["w"])
for n in ("sglang", "sglang.srt"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["sglang.srt.runtime_context"] = rc

def default_weight_loader(param, w):
    param.data.copy_(w)

mod = types.SimpleNamespace(default_weight_loader=default_weight_loader)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False)
        self.b = torch.nn.Parameter(torch.zeros(16, dtype=torch.uint8), requires_grad=False)
        # a sliced loader, like the experts' (this rank's quarter of the rows)
        self.b.weight_loader = lambda p, w: p.data.copy_(w.view(4, 16)[RANK["r"]])
        self.scaled = None
    def load_weights(self, weights):
        params = dict(self.named_parameters()); loaded = set()
        for name, w in weights:
            if name.startswith("mtp."):
                continue
            expected = {"a": torch.bfloat16, "b": torch.uint8}[name]
            if w.dtype != expected:
                raise ValueError("dtype")
            p = params[name]
            getattr(p, "weight_loader", mod.default_weight_loader)(p, w)
            loaded.add(name)
        if set(params) - loaded:
            raise ValueError("missing")
        self.scaled = self.a.float().sum().item() * 2   # a post-step after the loop
        return loaded

g = torch.Generator().manual_seed(0)
ckpt = [("a", (torch.randn(4, 8, generator=g)).to(torch.bfloat16)), ("b", torch.randint(0, 255, (64,), generator=g, dtype=torch.uint8)), ("mtp.x", torch.zeros(1))]

def boot(env):
    for k in ("SPARK_PRESHARD_EXPORT", "SPARK_PRESHARDED", "SPARK_TP_SHARD"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("preshard", None)
    ps = importlib.import_module("preshard")
    Cls = type("Toy", (Toy,), {}); ps._wrap(Cls, mod, draft=False)
    ps._S["model_path"] = env.get("_PATH", SRC)
    m = Cls(); out = m.load_weights(iter(ckpt)); return m, ps

m0, _ = boot({"SPARK_TP_SHARD": "half,fp8"})                                  # plain boot
m1, ps = boot({"SPARK_PRESHARD_EXPORT": OUT, "SPARK_TP_SHARD": "half,fp8"})  # export boot
folder = f"{OUT}/rank1"; print("files:", sorted(os.listdir(folder)))
assert torch.equal(m0.a, m1.a) and torch.equal(m0.b, m1.b)
m2, ps = boot({"SPARK_PRESHARDED": "1", "SPARK_TP_SHARD": "half,fp8", "_PATH": folder})
same = torch.equal(m0.a.view(torch.int16), m2.a.view(torch.int16)) and torch.equal(m0.b, m2.b) and m0.scaled == m2.scaled
print("reload equals the plain boot (parameters and post-step):", same)
print("weight loaders restored:", m2.b.weight_loader.__name__ == "<lambda>" and mod.default_weight_loader is default_weight_loader)
fails = 0
for label, env, change in (("wrong rank", {"SPARK_PRESHARDED": "1", "SPARK_TP_SHARD": "half,fp8", "_PATH": folder}, lambda: RANK.update(r=2)),
                           ("other switch", {"SPARK_PRESHARDED": "1", "SPARK_TP_SHARD": "", "_PATH": folder}, lambda: RANK.update(r=1))):
    change()
    try:
        boot(env); print(label, "NOT refused")
    except RuntimeError as e:
        fails += 1; print(label, "refused:", str(e)[:90])
RANK.update(r=1)
f = [x for x in os.listdir(folder) if x.startswith("model-spark")][0]
from safetensors.torch import load_file, save_file
d = load_file(f"{folder}/{f}"); d["a"] = d["a"].clone(); d["a"][0, 0] += 1; save_file(d, f"{folder}/{f}")
try:
    boot({"SPARK_PRESHARDED": "1", "SPARK_TP_SHARD": "half,fp8", "_PATH": folder}); print("corrupt file NOT refused")
except RuntimeError as e:
    fails += 1; print("corrupt file refused:", str(e)[:90])
# the files must be standard safetensors: the library reads them back to the same tensors
shutil.rmtree(OUT + "2", ignore_errors=True)
m3, ps = boot({"SPARK_PRESHARD_EXPORT": OUT + "2", "SPARK_TP_SHARD": "half,fp8"})
f2 = [x for x in os.listdir(OUT + "2/rank1") if x.startswith("model-spark")][0]
d2 = load_file(f"{OUT}2/rank1/{f2}")
standard = torch.equal(d2["a"].view(torch.int16), m0.a.view(torch.int16)) and torch.equal(d2["b"], m0.b)
print("safetensors library reads the files back exactly:", standard)
ok = same and fails == 3 and standard
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
