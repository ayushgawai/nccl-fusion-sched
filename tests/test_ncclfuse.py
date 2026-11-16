import json

import torch
import torch.nn as nn

import ncclfuse as nf
from ncclfuse import graph_pass


class Block(nn.Module):
    def forward(self, x):
        x = torch.mm(x, torch.eye(8))
        x = graph_pass.all_reduce(x)  # collective adjacent to matmul
        return torch.mm(x, torch.eye(8))


class TwoColl(nn.Module):
    def forward(self, x):
        x = graph_pass.all_reduce(torch.mm(x, torch.eye(8)))
        return graph_pass.all_gather(torch.mm(x, torch.eye(8)))


def test_pattern_matcher_finds_matmul_collective():
    import torch.fx as fx
    gm = fx.symbolic_trace(Block())
    pairs = nf.find_fusions(gm)
    assert len(pairs) == 2  # mm->all_reduce and all_reduce->mm


def test_schedule_ir_bucketing_and_streams():
    s = nf.schedule_module(Block(), world=2, tier="gloo_cpu", bucket_bytes=1 << 30)
    s.validate()
    comm = [o for o in s.ops if o.stream == 1]
    assert comm and comm[0].kind == "bucket_all_reduce"
    assert any(o.kind == "matmul" and o.stream == 0 for o in s.ops)
    assert "Schedule(world=2" in str(s)
    assert nf.Schedule.from_dict(json.loads(s.to_json())).to_dict() == s.to_dict()


def test_cost_model():
    assert nf.collective_time("all_reduce", 1 << 20, 1, "nvlink") == 0
    ring = nf.collective_time("all_reduce", 1 << 24, 8, "nvlink")
    assert 0 < ring < nf.collective_time("all_reduce", 1 << 24, 8, "ethernet")
    # more ranks -> more ring data movement for a fixed size
    assert nf.collective_time("all_gather", 1 << 24, 4, "nvlink") > \
        nf.collective_time("all_gather", 1 << 24, 2, "nvlink")
    b = nf.choose_bucket_size(1 << 26, 4, "nvlink", compute_window_s=1e-3)
    assert 1 << 20 <= b <= 1 << 26
    hidden, exposed = nf.overlap_window(10.0, 4.0)
    assert (hidden, exposed) == (4.0, 0.0)


TOPO = """GPU0 GPU1 GPU2
GPU0  X   NV4  SYS
GPU1 NV4   X   PIX
GPU2 SYS  PIX   X
"""


def test_topology_and_replan(tmp_path):
    topo = nf.load_topology(_write(tmp_path, TOPO))
    assert topo.tier_between(0, 1) == "nvlink"
    assert topo.tier_between(1, 2) == "pcie"
    ring = nf.plan_ring([0, 1, 2], topo)
    assert sorted(ring) == [0, 1, 2] and ring[1] == 1  # NVLink neighbour first
    plan = nf.replan(4, 2, topo)
    assert plan.world == 2 and len(plan.ring) == 2 and plan.bucket_bytes > 0


def _write(tmp_path, text):
    p = tmp_path / "topo.txt"
    p.write_text(text)
    return p


def test_elastic_checkpoint_and_failure(tmp_path):
    from ncclfuse.elastic import load_checkpoint, save_checkpoint
    state = {"model": {"w": torch.ones(3)}, "step": 7}
    ckpt = tmp_path / "ckpt.pt"
    assert save_checkpoint(ckpt, state)
    assert load_checkpoint(ckpt)["step"] == 7
    recovered, plan = nf.simulate_failure(state, dead_rank=3, new_world=2)
    assert recovered["step"] == 7 and plan.world == 2 and len(plan.ring) == 2


def test_cpu_kernels_skip_cleanly():
    from ncclfuse.kernels_triton import fused_matmul_reduce
    try:
        fused_matmul_reduce(torch.randn(4, 4), torch.randn(4, 4))
        raise AssertionError("should have raised on CPU")
    except RuntimeError as e:
        assert "CUDA" in str(e)


def test_chrome_trace(tmp_path):
    p = nf.chrome_trace([{"name": "all_reduce", "start_s": 0.0, "dur_s": 0.1, "stream": 1}],
                        tmp_path / "t.json")
    assert json.loads(p.read_text())["traceEvents"][0]["name"] == "all_reduce"
