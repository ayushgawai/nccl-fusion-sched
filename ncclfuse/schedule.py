"""Schedule IR: a printable, JSON-able list of stream-assigned ops."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Op:
    id: int
    kind: str  # matmul | all_reduce | all_gather | reduce_scatter | all_to_all | bucket | other
    name: str
    stream: int = 0  # 0=compute, 1=communication
    nbytes: int = 0
    deps: tuple[int, ...] = ()
    bucket: int = -1

    def __str__(self) -> str:
        return (f"#{self.id:<3} stream={self.stream} {self.kind:<18} {self.name} "
                f"bytes={self.nbytes} deps={list(self.deps)} bucket={self.bucket}")


@dataclass
class Schedule:
    ops: list[Op] = field(default_factory=list)
    world: int = 1
    cuda_graph: bool = False

    def validate(self) -> None:
        ids = {o.id for o in self.ops}
        assert len(ids) == len(self.ops), "duplicate op id"
        for o in self.ops:
            assert set(o.deps) <= ids and o.id not in o.deps, f"bad deps on {o}"
        # ponytail: order is the topo order, so a dep after its user is a cycle here.
        seen: set[int] = set()
        for o in self.ops:
            assert set(o.deps) <= seen, f"op {o.id} depends on later op"
            seen.add(o.id)

    @property
    def comm_bytes(self) -> int:
        return sum(o.nbytes for o in self.ops if o.stream == 1)

    def __str__(self) -> str:
        head = (f"Schedule(world={self.world}, cuda_graph={self.cuda_graph}, "
                f"comm_bytes={self.comm_bytes})")
        return "\n".join([head, *(str(o) for o in self.ops)])

    def to_dict(self) -> dict:
        return {"world": self.world, "cuda_graph": self.cuda_graph,
                "ops": [{**asdict(o), "deps": list(o.deps)} for o in self.ops]}

    @classmethod
    def from_dict(cls, d: dict) -> Schedule:
        ops = [Op(**{**o, "deps": tuple(o["deps"])}) for o in d["ops"]]
        s = cls(ops, d["world"], d["cuda_graph"])
        s.validate()
        return s

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
