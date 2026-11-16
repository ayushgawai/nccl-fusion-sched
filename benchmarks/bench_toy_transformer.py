#!/usr/bin/env python3
"""Toy transformer: no-overlap vs scheduled-overlap step time, 2..8 ranks.

CPU/gloo by default (GPU/NCCL if CUDA is present):
    python benchmarks/bench_toy_transformer.py --ranks 2 --steps 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402

from ncclfuse.runtime import measure_overlap  # noqa: E402


class ToyTransformer(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, 4, 256, batch_first=True)
        self.net = nn.TransformerEncoder(layer, 2)

    def forward(self, x):
        return self.net(x)


def worker(rank, world, steps, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world)
    torch.manual_seed(0)
    model = ToyTransformer()
    if torch.cuda.is_available():
        model = model.cuda()
    x = torch.randn(4, 16, 128, device=next(model.parameters()).device)
    results = {}
    for overlap in (False, True):
        times = []
        for _ in range(steps):
            for p in model.parameters():
                p.grad = None
            t0 = time.perf_counter()
            model(x).sum().backward()
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

            def comm(gs=grads):
                for g in gs:
                    dist.all_reduce(g)

            def compute():
                y = x
                for _ in range(4):
                    y = torch.tanh(y @ torch.eye(x.shape[-1], device=x.device))

            timing = measure_overlap(compute, comm, overlap=overlap)
            times.append((time.perf_counter() - t0, timing.hidden_pct))
            if dist.is_initialized() and not overlap:
                pass
        results["overlap" if overlap else "no_overlap"] = (
            sum(t for t, _ in times) / len(times), sum(h for _, h in times) / len(times))
    if rank == 0:
        q.put(results)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--port", type=int, default=29517)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(worker, (a.ranks, a.steps, a.port, q), nprocs=a.ranks, join=True)
    r = q.get()
    dev = "GPU/NCCL" if torch.cuda.is_available() else "CPU/gloo"
    print("| ranks | mode | step (ms) | comm hidden % | device |")
    print("|---|---|---|---|---|")
    for mode in ("no_overlap", "overlap"):
        step, hidden = r[mode]
        print(f"| {a.ranks} | {mode} | {step * 1e3:.2f} | {hidden:.1f} | {dev} |")


if __name__ == "__main__":
    main()
