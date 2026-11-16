"""Model-agnostic benchmarking helpers (timing, memory, energy, result rows) used by every part of the project.

Everything a Pareto point needs besides accuracy:
  * latency distribution (warm-up excluded, percentiles, not just the mean)
  * throughput / real-time factor
  * energy (NVIDIA NVML counter on GPU, Intel RAPL on bare-metal CPU, else None)
  * peak memory, parameter count, hardware description

Teaching points embedded in the code:
  * CUDA is asynchronous: without torch.cuda.synchronize() you time the kernel launch.
  * The first calls include lazy init / autotuning: always warm up.
  * Report p50 and p90/p99: real-time constraints are about the tail, not the mean.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    import resource  # Unix only: Windows has no such module
except ImportError:
    resource = None


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError:
        return None


def default_device() -> str:
    """'cuda' when torch sees a GPU, else 'cpu': the default of every --device option (a laptop has no CUDA)."""
    torch = _torch()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def cuda_sync():
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "n_cpus": os.cpu_count(),
    }
    torch = _torch()
    if torch is not None:
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def count_params(model) -> int | None:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Energy meters
# --------------------------------------------------------------------------- #
class _NVMLMeter:
    """Uses the cumulative energy counter (mJ) available on Volta+ GPUs (A10G, A100, L4...)."""

    def __init__(self, index: int = 0):
        import pynvml  # pip install nvidia-ml-py
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)  # raises if unsupported

    def read_j(self) -> float:
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0


class _RAPLMeter:
    """Intel RAPL package counter. Usually NOT readable inside cloud containers."""

    PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")

    def __init__(self):
        if not self.PATH.exists():
            raise RuntimeError("RAPL not available")
        self.read_j()

    def read_j(self) -> float:
        return int(self.PATH.read_text()) / 1e6


def make_energy_meter():
    for cls in (_NVMLMeter, _RAPLMeter):
        try:
            return cls()
        except Exception:
            continue
    return None


@contextmanager
def energy_scope(result: dict, key: str = "energy_j"):
    """with energy_scope(res): run(); -> res['energy_j'] (None if no meter)."""
    meter = make_energy_meter()
    e0 = meter.read_j() if meter else None
    try:
        yield
    finally:
        cuda_sync()
        result[key] = (meter.read_j() - e0) if meter else None
        result["energy_meter"] = type(meter).__name__ if meter else None


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def summarize_latencies(lat_s: Iterable[float]) -> dict:
    a = np.asarray(list(lat_s), dtype=float) * 1000.0
    if a.size == 0:
        return {}
    return {
        "lat_mean_ms": float(a.mean()),
        "lat_p50_ms": float(np.percentile(a, 50)),
        "lat_p90_ms": float(np.percentile(a, 90)),
        "lat_p99_ms": float(np.percentile(a, 99)),
        "lat_std_ms": float(a.std()),
        "n_timed": int(a.size),
    }


def time_calls(fn: Callable[[Any], Any], inputs: list, warmup: int = 3) -> tuple[list, list[float]]:
    """Run fn on every input; return (outputs, latencies_s). Warm-up runs are discarded."""
    for x in inputs[:warmup]:
        fn(x)
    cuda_sync()
    outs, lats = [], []
    for x in inputs:
        t0 = time.perf_counter()
        y = fn(x)
        cuda_sync()
        lats.append(time.perf_counter() - t0)
        outs.append(y)
    return outs, lats


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB, on Linux, macOS and Windows."""
    if resource is not None:  # ru_maxrss is in kB on Linux but in bytes on macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (2**20 if sys.platform == "darwin" else 1024.0)
    import psutil  # Windows (a dependency of accelerate): peak working set
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def peak_memory_mb() -> dict:
    out = {"cpu_maxrss_mb": _peak_rss_mb()}
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        out["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def reset_peak_memory():
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Cost model (cloud credits / € per hour -> cost per 1k inferences)
# --------------------------------------------------------------------------- #
def cost_per_1k(lat_mean_ms: float, price_per_hour: float, batch: int = 1) -> float:
    """Cost of 1000 inferences if the machine is fully utilised at this latency."""
    return price_per_hour * (lat_mean_ms / 1000.0) * 1000.0 / 3600.0 / batch


# --------------------------------------------------------------------------- #
# Results logging (one JSON line per configuration -> easy to load in pandas
# or to PUT on a Snowflake stage and COPY INTO a table)
# --------------------------------------------------------------------------- #
@dataclass
class BenchRecord:
    task: str
    model: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=hardware_info)
    timestamp: float = field(default_factory=time.time)

    def flat(self) -> dict:
        d = {"task": self.task, "model": self.model, "timestamp": self.timestamp}
        # every row is tagged with the milestone it was produced for: `export DSP_MILESTONE=d2`
        self.config.setdefault("milestone", os.environ.get("DSP_MILESTONE", "untagged"))
        d.update({f"cfg_{k}": v for k, v in self.config.items()})
        d.update(self.metrics)
        d.update({f"hw_{k}": v for k, v in self.hardware.items()})
        return d


def log_record(rec: BenchRecord, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.flat(), default=str) + "\n")


def load_records(path: str | Path):
    import pandas as pd
    return pd.read_json(path, lines=True)
