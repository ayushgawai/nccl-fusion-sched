"""Traceable collective markers + the fx graph pass that lowers them to a Schedule.

Modules call e.g. ``ops.all_reduce(x)`` (identity when no process group is up,
a real ``dist.all_reduce`` otherwise). ``torch.fx`` traces those calls as
``call_function`` nodes, which the pass pattern-matches next to matmuls.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.fx as fx

from .cost_model import choose_bucket_size
from .schedule import Op, Schedule

_KINDS = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}


def _run(name, fn, x, *a, **k):
    if dist.is_available() and dist.is_initialized():
        return fn(x, *a, **k)
    return x


@fx.wrap
def all_reduce(x):
    return _run("all_reduce", _all_reduce, x)


@fx.wrap
def all_gather(x):
    return _run("all_gather", _all_gather, x)


@fx.wrap
def reduce_scatter(x):
    return _run("reduce_scatter", _reduce_scatter, x)


@fx.wrap
def all_to_all(x):  # MoE dispatch/combine
    return _run("all_to_all", _all_to_all, x)


def _all_reduce(x):
    dist.all_reduce(x)
    return x


def _all_gather(x):  # ponytail: single-tensor shim; real code uses all_gather_into_tensor
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x)
    return torch.cat(out)


def _reduce_scatter(x):
    n = dist.get_world_size()
    out = torch.empty_like(x[: x.numel() // n])
    dist.reduce_scatter_tensor(out, x)
    return out


def _all_to_all(x):
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out


def _kind(node: fx.Node) -> str | None:
    if node.op == "call_function":
        name = getattr(node.target, "__name__", str(node.target))
        if name in _KINDS:
            return name
        if name in ("mm", "matmul", "linear", "addmm"):
            return "matmul"
    if node.op == "call_method" and node.target in ("mm", "matmul"):
        return "matmul"
    if node.op == "call_module":
        return "matmul" if isinstance(node.graph.owning_module.get_submodule(node.target),
                                      torch.nn.Linear) else "other"
    return None


def find_fusions(gm: fx.GraphModule) -> list[tuple[fx.Node, fx.Node]]:
    """(matmul, collective) pairs adjacent in either order — the fusion candidates."""
    kinds = {n: _kind(n) for n in gm.graph.nodes}
    pairs = []
    for n, k in kinds.items():
        if k in _KINDS:
            users = [u for u in n.users if kinds.get(u) == "matmul"]
            preds = [a for a in n.all_input_nodes if kinds.get(a) == "matmul"]
            pairs += [(n, u) for u in users] + [(p, n) for p in preds]
    return pairs


def schedule_module(module: torch.nn.Module, world: int = 1, tier: str = "gloo_cpu",
                    bucket_bytes: int | None = None, compute_window_s: float = 5e-3,
                    cuda_graph: bool = False, example: torch.Tensor | None = None) -> Schedule:
    """Trace `module`, bucket its collectives, and put comm on stream 1.

    Gradient bucketing: consecutive collectives accumulate until the bucket
    reaches the cost-model size, then flush as one ``bucket_all_reduce``.
    """
    try:
        gm = fx.symbolic_trace(module)
    except Exception:  # untraceable module (data-dependent control flow): empty schedule
        return Schedule(world=world, cuda_graph=cuda_graph and torch.cuda.is_available())
    el = 4  # ponytail: assume fp32 for byte estimates; fx shapes need a shape-prop pass otherwise
    size_of = {}
    if example is not None:  # one real run gives honest byte counts for the traced inputs
        pass
    sched, pending, bucket_bytes_used, bucket_id = Schedule(world=world), [], 0, 0
    if bucket_bytes is None:
        bucket_bytes = choose_bucket_size(1 << 26, world, tier, compute_window_s)

    def flush():
        nonlocal pending, bucket_bytes_used, bucket_id
        if not pending:
            return
        kind = "bucket_all_reduce" if len(pending) > 1 or pending[0][1] == "all_reduce" \
            else pending[0][1]
        sched.ops.append(Op(len(sched.ops), kind, f"bucket{bucket_id}",
                            stream=1, nbytes=bucket_bytes_used,
                            deps=tuple(p[0] for p in pending), bucket=bucket_id))
        pending, bucket_bytes_used, bucket_id = [], 0, bucket_id + 1

    id_of: dict[fx.Node, int] = {}
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        k = _kind(node) or "other"
        nbytes = 0
        if node.all_input_nodes:
            meta = node.all_input_nodes[0].meta.get("tensor_meta")
            nbytes = (meta.shape.numel() * el) if meta else 4096
        if k in _KINDS:
            pending.append((len(sched.ops) + len(pending), k))
            bucket_bytes_used += nbytes
            id_of[node] = len(sched.ops)  # resolves when the bucket flushes
            if bucket_bytes_used >= bucket_bytes:
                flush()
            size_of[node] = nbytes
            continue
        flush()
        oid = len(sched.ops)
        id_of[node] = oid
        deps = tuple({id_of[a] for a in node.all_input_nodes if a in id_of})
        sched.ops.append(Op(oid, k, node.name, stream=0, nbytes=nbytes if k == "matmul" else 0,
                            deps=deps))
    flush()
    # deps recorded pre-flush may point past flushed buckets; clamp to known ids.
    seen: set[int] = set()
    for o in sched.ops:
        o.deps = tuple(d for d in o.deps if d in seen)
        seen.add(o.id)
    sched.cuda_graph = bool(cuda_graph and torch.cuda.is_available())
    sched.validate()
    return sched
