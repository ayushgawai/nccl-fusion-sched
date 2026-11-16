"""Alpha-beta cost model: T = steps*alpha + bytes_moved/beta."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    alpha_s: float  # per-step latency
    beta_Bps: float  # bytes / s


TIERS = {
    "nvlink": Tier("nvlink", 1.5e-6, 300e9),
    "pcie": Tier("pcie", 5e-6, 32e9),
    "infiniband": Tier("infiniband", 3e-6, 25e9),  # 200 Gb/s HDR-ish
    "ethernet": Tier("ethernet", 20e-6, 3e9),
    "gloo_cpu": Tier("gloo_cpu", 50e-6, 1e9),
}
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def _tier(tier: str | Tier) -> Tier:
    return TIERS[tier] if isinstance(tier, str) else tier


def collective_time(op: str, nbytes: int, world: int, tier: str | Tier,
                    algo: str = "ring") -> float:
    """Seconds for one collective. Ring moves (n-1)/n of data per phase; tree is log-step."""
    if world <= 1 or nbytes <= 0:
        return 0.0
    t = _tier(tier)
    if algo == "tree" and op == "all_reduce":
        steps = 2 * math.ceil(math.log2(world))
        return steps * (t.alpha_s + nbytes / t.beta_Bps / world)
    phases = {"all_reduce": 2, "all_gather": 1, "reduce_scatter": 1, "all_to_all": 1}[op]
    steps_per_phase = world - 1 if op != "all_to_all" else world - 1
    moved = phases * nbytes * (world - 1) / world
    return phases * steps_per_phase * t.alpha_s + moved / t.beta_Bps


def choose_bucket_size(total_bytes: int, world: int, tier: str | Tier,
                       compute_window_s: float, op: str = "all_reduce") -> int:
    """Largest power-of-two bucket whose predicted time fits the compute window.

    ponytail: 1 MiB floor / 256 MiB cap; tune from real NCCL busBW when profiling.
    """
    if total_bytes <= 0:
        return 0
    size = 1 << 20
    while size * 2 <= min(total_bytes, 1 << 28):
        if collective_time(op, size * 2, world, tier) > compute_window_s:
            break
        size *= 2
    return min(size, total_bytes)


def overlap_window(compute_s: float, comm_s: float) -> tuple[float, float]:
    """(hidden_s, exposed_s) if comm is launched under `compute_s` of independent work."""
    hidden = min(compute_s, comm_s)
    return hidden, comm_s - hidden
