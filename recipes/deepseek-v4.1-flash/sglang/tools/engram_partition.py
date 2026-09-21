#!/usr/bin/env python3
"""Print each rank's Engram row ranges for a tensor-parallel size, from the checkpoint's config.json.

    tools/engram_partition.py <snapshot-dir> [--tp 4]

The overlay gives every rank one contiguous range of each Engram table (the ranges must partition the table;
the all-reduce after the lookup reassembles the rows). The split printed here is SGLang's own,
rows [N*r/tp, N*(r+1)/tp) with N = engram_num_embeddings of the layer. Use the output as the row arguments of
tools/engram_local.py for the ranks that get a node-local copy, and as ENGRAM_ROWS_HOST in launch/fleet.env for
the rank that reads the checkpoint shards directly. Any other partition works too; the manifest of each copy carries whatever range was used.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    snap = sys.argv[1]
    tp = int(sys.argv[sys.argv.index("--tp") + 1]) if "--tp" in sys.argv else 4
    cfg = json.load(open(os.path.join(snap, "config.json")))
    text = cfg.get("text_config", cfg)
    layers = text["engram_layer_ids"]
    sizes = text["engram_num_embeddings"]
    for r in range(tp):
        spec = " ".join(f"{layer}:{n * r // tp}:{n * (r + 1) // tp}" for layer, n in zip(layers, sizes))
        print(f"rank {r}: {spec}")
    print("(engram_local.py takes the ranges space-separated; ENGRAM_ROWS_HOST takes them comma-separated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
