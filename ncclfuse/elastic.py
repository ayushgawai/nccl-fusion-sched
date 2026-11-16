"""Elasticity: topology discovery, checkpointing, ring re-planning on resize."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .cost_model import choose_bucket_size

# nvidia-smi topo -m link codes, best tier first
_LINK_TIER = {"NV": "nvlink", "PIX": "pcie", "PXB": "pcie", "PHB": "pcie", "SYS": "infiniband"}


@dataclass
class Topology:
    ranks: list[int]
    tiers: dict[tuple[int, int], str]  # pair -> tier name

    def tier_between(self, a: int, b: int) -> str:
        if a == b:
            return "nvlink"
        return self.tiers.get((a, b), self.tiers.get((b, a), "infiniband"))


def parse_nvidia_smi_topo(text: str) -> Topology:
    """Parse `nvidia-smi topo -m` output (or a saved fixture of it)."""
    ranks, tiers = [], {}
    for line in text.splitlines():
        m = re.match(r"\s*GPU(\d+)\s+(.*)", line)
        if not m:
            continue
        i, cells = int(m.group(1)), m.group(2).split()
        ranks.append(i)
        for j, cell in enumerate(cells[: len(ranks) + 8]):
            tier = next((t for k, t in _LINK_TIER.items() if cell.startswith(k)), None)
            if tier and j != i:
                tiers[(i, j)] = tier
    return Topology(sorted(set(ranks)), tiers)


def load_topology(config: str | Path | dict | None = None) -> Topology:
    """From a JSON config / topo fixture; defaults to a single-node all-NVLink guess."""
    if config is None:
        return Topology([0], {})
    if isinstance(config, dict):
        ranks = [int(r) for r in config.get("ranks", [0])]
        tiers = {}
        for k, v in config.get("tiers", {}).items():
            a, b = (int(x) for x in str(k).split(","))
            tiers[(a, b)] = v
        return Topology(ranks, tiers)
    text = Path(config).read_text()
    if text.lstrip().startswith("{"):
        cfg = json.loads(text)
        ranks = cfg.get("ranks", [0])
        tiers = {}
        for k, v in cfg.get("tiers", {}).items():
            a, b = (int(x) for x in str(k).split(","))
            tiers[(a, b)] = v
        return Topology(ranks, tiers)
    return parse_nvidia_smi_topo(text)


def plan_ring(ranks: list[int], topo: Topology | None = None) -> list[int]:
    """Greedy nearest-tier ring: always extend to the best-connected unvisited rank.

    ponytail: greedy, not optimal TSP — rings are near-cliques on real nodes anyway.
    """
    if not ranks:
        return []
    order_rank = {"nvlink": 0, "pcie": 1, "infiniband": 2, "ethernet": 3}
    remaining, ring = set(ranks), [min(ranks)]
    remaining.discard(ring[0])
    while remaining:
        cur = ring[-1]
        nxt = min(remaining, key=lambda r: (order_rank.get(topo.tier_between(cur, r), 2)
                                            if topo else 0, r))
        ring.append(nxt)
        remaining.discard(nxt)
    return ring


@dataclass
class ElasticPlan:
    ring: list[int]
    world: int
    bucket_bytes: int
    tier: str


def replan(old_world: int, new_world: int, topo: Topology | None = None,
           total_grad_bytes: int = 1 << 26, tier: str = "nvlink") -> ElasticPlan:
    """Re-plan after ranks join/leave: fresh ring + cost-model bucket for the new world."""
    assert new_world >= 1
    ranks = [r for r in (topo.ranks if topo else list(range(old_world)))][:new_world]
    ranks = (ranks + [r for r in range(new_world) if r not in ranks])[:new_world]
    return ElasticPlan(plan_ring(sorted(ranks), topo), new_world,
                       choose_bucket_size(total_grad_bytes, new_world, tier, 5e-3), tier)


def save_checkpoint(path: str | Path, state: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, weights_only=False)


def simulate_failure(state: dict, dead_rank: int, new_world: int,
                     topo: Topology | None = None) -> tuple[dict, ElasticPlan]:
    """Drop a rank, restore from checkpointed `state`, and return the re-plan.

    The state dict itself is rank-agnostic (model + optimizer), so recovery is
    just: verify it loads, hand back a ring/bucket plan for the survivors.
    """
    assert state, "empty checkpoint — nothing to recover"
    plan = replan(max(new_world, dead_rank + 1) if topo is None else len(topo.ranks),
                  new_world, topo)
    return state, plan
