"""Profiling: Chrome trace export + optional torch.profiler capture."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


def chrome_trace(events: list[dict], path: str | Path) -> Path:
    """events: [{name, start_s, dur_s, stream}] -> Chrome trace JSON."""
    out = [{"name": e["name"], "ph": "X", "ts": e["start_s"] * 1e6, "dur": e["dur_s"] * 1e6,
            "pid": 0, "tid": e.get("stream", 0)} for e in events]
    Path(path).write_text(json.dumps({"traceEvents": out}))
    return Path(path)


@contextmanager
def profile_step(path: str | Path | None = None, activities=("CPU",)):
    """Yield an event list; on exit write a Chrome trace (and a torch.profiler
    trace too, when a path ending in .json is given and CUDA/CPU profiling is on)."""
    events, t0 = [], time.perf_counter()
    try:
        from torch.profiler import ProfilerActivity, profile

        acts = [getattr(ProfilerActivity, a) for a in activities if hasattr(ProfilerActivity, a)]
        with profile(activities=acts or [ProfilerActivity.CPU]) as prof:
            yield events
        if path:
            prof.export_chrome_trace(str(path))
            return
    except Exception:
        yield events
    if path:
        chrome_trace(events or [{"name": "step", "start_s": 0.0,
                                 "dur_s": time.perf_counter() - t0, "stream": 0}], path)
