"""Runtime: init process groups and execute / time scheduled work."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .schedule import Schedule


def init(rank: int, world: int, backend: str | None = None) -> str:
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group(backend, rank=rank, world_size=world)
    return backend


@dataclass
class Timing:
    total_s: float
    comm_s: float
    exposed_s: float

    @property
    def hidden_s(self) -> float:
        return max(0.0, self.comm_s - self.exposed_s)

    @property
    def hidden_pct(self) -> float:
        return 100.0 * self.hidden_s / self.comm_s if self.comm_s else 0.0


def measure_overlap(compute, comm, overlap: bool = True) -> Timing:
    """Run comm (a collective callable) under compute; report exposed vs hidden comm.

    CUDA: separate comm stream. CPU/gloo: a thread — gloo collectives release
    the GIL while waiting on the network, so this genuinely overlaps.
    """
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        c0 = time.perf_counter()
        with torch.cuda.stream(stream):
            comm()
        compute()
        torch.cuda.current_stream().synchronize()
        stream.synchronize()
        comm_s = time.perf_counter() - c0  # includes overlap; exposed measured below
        total = time.perf_counter() - t0
        return Timing(total, comm_s, exposed_s=max(0.0, total - _time(compute, dry=True)))
    if overlap:
        box = {}

        def run_comm():
            c0 = time.perf_counter()
            comm()
            box["comm_s"] = time.perf_counter() - c0

        th = threading.Thread(target=run_comm)
        th.start()
        c0 = time.perf_counter()
        compute()
        compute_s = time.perf_counter() - c0
        th.join()
        total = time.perf_counter() - t0
        comm_s = box["comm_s"]
        return Timing(total, comm_s, exposed_s=max(0.0, comm_s - compute_s))
    c0 = time.perf_counter()
    comm()
    comm_s = time.perf_counter() - c0
    compute()
    return Timing(time.perf_counter() - t0, comm_s, exposed_s=comm_s)


def _time(fn, dry: bool = False) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def execute_schedule(sched: Schedule, tensors: dict[str, torch.Tensor]) -> Timing:
    """Execute a schedule's ops against named tensors (matmul = mm on 'a','b').

    Stream-1 ops are launched in a thread and joined when a dependent op on
    stream 0 needs them — that join gap is the exposed communication.
    """
    done: dict[int, threading.Thread] = {}
    comm_s = exposed_s = 0.0
    t0 = time.perf_counter()

    def do(op):
        nonlocal comm_s
        x = tensors.get("x")
        if op.kind in ("bucket_all_reduce", "all_reduce") and dist.is_initialized() \
                and x is not None:
            c0 = time.perf_counter()
            dist.all_reduce(x)
            comm_s += time.perf_counter() - c0
        elif op.kind == "matmul" and "a" in tensors and "b" in tensors:
            tensors["x"] = tensors["a"] @ tensors["b"]

    for op in sched.ops:
        for d in op.deps:
            if d in done:
                j0 = time.perf_counter()
                done.pop(d).join()
                exposed_s += time.perf_counter() - j0
        if op.stream == 1 and torch.cuda.is_available():
            do(op)  # ponytail: CUDA stream path lives in measure_overlap; CPU uses threads
        elif op.stream == 1:
            th = threading.Thread(target=do, args=(op,))
            th.start()
            done[op.id] = th
        else:
            do(op)
    for th in done.values():
        j0 = time.perf_counter()
        th.join()
        exposed_s += time.perf_counter() - j0
    return Timing(time.perf_counter() - t0, comm_s, min(exposed_s, comm_s) if comm_s else exposed_s)


def distributed_step(model: torch.nn.Module, x: torch.Tensor, overlap: bool = True) -> Timing:
    """One toy training step: forward/backward, then bucketed grad all-reduce.

    Overlap mode launches each bucket's all-reduce on a thread while the next
    bucket is being copied/processed — the standard DDP trick, by hand.
    """
    grads = [p.grad for p in model.parameters()]
    out = model(x).sum()
    out.backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    def comm():
        for g in grads:
            if dist.is_initialized():
                dist.all_reduce(g)

    def compute():  # stand-in for independent work (next micro-batch / optimizer math)
        y = x
        for _ in range(3):
            y = torch.tanh(y @ torch.eye(x.shape[-1]))
        return y

    return measure_overlap(compute, comm, overlap=overlap)
