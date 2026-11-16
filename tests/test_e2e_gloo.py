"""2-rank gloo end-to-end (CPU). GPU/NCCL path is the same code, device-gated."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from ncclfuse.graph_pass import schedule_module
from ncclfuse.runtime import init, measure_overlap


def _worker(rank, world, port, q):
    os.environ["MASTER_PORT"] = str(port)
    init(rank, world)
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    x = torch.randn(4, 8)
    model(x).sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]

    def comm():
        for g in grads:
            dist.all_reduce(g)

    timing = measure_overlap(lambda: torch.tanh(x @ torch.eye(8)), comm, overlap=True)
    sched = schedule_module(model, world=world)
    if rank == 0:
        q.put((timing.total_s, timing.comm_s, len(sched.ops),
               float(grads[0].sum()), float(x.sum())))
    dist.barrier()
    dist.destroy_process_group()


def test_two_rank_gloo_e2e():
    if not dist.is_gloo_available():
        pytest.skip("gloo unavailable")
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, (2, 29519, q), nprocs=2, join=True)
    total, comm, n_ops, grad_sum, _ = q.get()
    assert total > 0 and comm >= 0 and n_ops > 0 and grad_sum != 0
