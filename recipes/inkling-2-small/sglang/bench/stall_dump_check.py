"""CPU-only check of the stall recorder in patches/roce_collectives.py (SPARK_STALL_DUMP): a fake runtime, no torch,
no GPU. Needs numpy: run it with the image's python, e.g.
  docker run --rm --memory 1g --network none -v $PWD:/w:ro --entrypoint /opt/sglang/bin/python3 <image> -I /w/bench/stall_dump_check.py /w/patches
MIT License, Copyright (c) 2026 Aiden Le.
"""
import sys, types, time, logging, os
import numpy as np
t = types.ModuleType("torch"); t.Tensor = object; t.device = object
d = types.ModuleType("torch.distributed"); t.distributed = d
sys.modules["torch"] = t; sys.modules["torch.distributed"] = d
os.environ["SPARK_ROCE_AR"] = "1"; os.environ["SPARK_STALL_DUMP"] = "0.4"
sys.path.insert(0, sys.argv[1])
import roce_collectives as rc
logging.basicConfig(level=logging.INFO, format="%(message)s")

class Lay: flag_off = 64; flag_stride = 64; slots = 2
class Proxy:
    def failed(self): return False
    def error(self): return ""
    def stats(self): return {"ops_posted": 10, "writes_completed": 60, "last_seq": 10}
class Region:
    def __init__(self): self.a = np.zeros(4096, np.uint8)
    def numpy(self): return self.a
class RT:
    world_size = 4; rank = 1; rail_count = 2; _layout = Lay(); _proxy = Proxy()
    def __init__(self):
        self._region = Region(); self._ctrl_np = np.array([10, 4096, 0, 0, 4096, 9216, 0], np.int32)
        for peer, vals in {0: (10, 10, 9, 9), 2: (10, 10, 9, 9), 3: (8, 8, 9, 9)}.items():
            for i, v in enumerate(vals):
                off = 64 + (peer * 4 + i) * 64
                self._region.a[off:off + 4] = np.array([v], np.uint32).view(np.uint8)
    def check_health(self): pass
tp = rc._Runtime.__new__(rc._Runtime)
tp.rt = RT(); tp.rank = 1; tp.seen = {"ar": 3, "ag": 1, "nccl_ar": 2, "nccl_ag": 0, "last": ("ar", 4096), "last_nccl": ("ar", 2 << 20)}
rc._TP = tp
out = tp.describe(); print(out)
assert "doorbell seq 10" in out and "3: [8, 8, 9, 9]" in out and "0: [10, 10, 9, 9]" in out and "1:" not in out.split("flags")[1].split("proxy")[0]

class MR:
    forward_stream = types.SimpleNamespace(query=lambda: False)
    def forward(self, batch, hang=0.0):
        time.sleep(hang); return "ok"
m = types.SimpleNamespace(ModelRunner=MR); rc.install_health(m)
class B: forward_mode = types.SimpleNamespace(name="EXTEND"); batch_size = 1; input_ids = types.SimpleNamespace(numel=lambda: 21); rids = ["a", "b"]; seq_lens_sum = 1045; extend_prefix_lens_cpu = [1024]; extend_seq_lens_cpu = [21]
recs = []
h = logging.Handler(); h.emit = lambda r: recs.append(r.getMessage()); rc.logger.addHandler(h)
mr = MR()
for _ in range(50): assert mr.forward(B()) == "ok"
time.sleep(0.6); assert all("watch armed" in r for r in recs) and len(recs) == 1, recs          # fast forwards: silent
assert mr.forward(B(), 1.1) == "ok"            # one stalled forward: exactly two reports
time.sleep(0.6)
recs[:] = [r for r in recs if "watch armed" not in r]
assert len(recs) in (2, 3) and "EXTEND bs 1 tok 21 rids" in recs[0] and "pre [1024] new [21]" in recs[0] and "last forwards: " in recs[0] and "51: EXTEND" in recs[0], recs
assert rc._FWD[0] is None
assert "collectives since its start: ar 0, ag 0, nccl_ar 0, nccl_ag 0" in recs[0] and "streams: forward0 BUSY" in recs[0], recs[0][:300]
# a forward that returned, then nothing: reported only while the scheduler holds requests (three times)
class Scheduler:
    running_batch = types.SimpleNamespace(reqs=[]); waiting_queue = []
    schedule_stream = types.SimpleNamespace(query=lambda: True)
sched = Scheduler(); rc._SCHED["obj"] = sched
recs.clear(); assert mr.forward(B()) == "ok"; time.sleep(0.9); assert not recs, recs   # idle: silent
sched.running_batch.reqs = [1, 2]; assert mr.forward(B()) == "ok"; time.sleep(1.9)
assert len(recs) == 3 and "the scheduler holds 2 running and 0 waiting requests, no new forward for" in recs[0] and "schedule_stream idle" in recs[0], (len(recs), recs[:1])
recs.clear(); time.sleep(0.6); assert not recs   # an episode is reported three times, no more
sched.running_batch.reqs = []
tp.rt = None; print(rc.stall_report(1, "x")); rc._TP = None; print(rc.stall_report(1, "x"))
assert rc._SIG["n"] == 53 and len(rc._RING) == 24
n0 = rc._SIG["crc"]; rc._batch_note(types.SimpleNamespace()); assert rc._SIG["crc"] != n0 and "rids -" in rc._RING[-1][1]
print("PASS", len(recs))
