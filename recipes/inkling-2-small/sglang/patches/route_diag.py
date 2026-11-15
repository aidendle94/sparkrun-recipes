"""SPARK_ROUTE_BALANCE=1 (diagnosis windows only: the text is garbage. Never in production).

What a decode step would take if every rank read the same number of experts. Under expert parallelism the ranks own
contiguous blocks of experts and a step's routed experts fall unevenly on them; the expert all-reduce then holds the
time every rank waits for the busiest one. `balance` renames the expert ids of a forward so the distinct experts it
routes to are dealt to the ranks in turn (the i-th distinct expert goes to rank i % ranks): the same number of
distinct experts, rows and pairs as the real routing, and two pairs share an expert exactly when they did before,
but the weights read are other experts' (hence garbage). With patches/graph_timers.py on the same boot, the expert
all-reduce region is then the collective alone, and the difference to an unbalanced boot at the same verify row
count is the wait.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch


def balance(ids, num_experts, ranks):
    flat = ids.reshape(-1).long()
    present = torch.zeros(num_experts, dtype=torch.int64, device=ids.device).index_fill_(0, flat, 1)
    order = torch.cumsum(present, 0) - 1  # an expert's position among the forward's distinct experts
    renamed = (order % ranks) * (num_experts // ranks) + order // ranks
    return renamed[flat].to(ids.dtype).view_as(ids)
