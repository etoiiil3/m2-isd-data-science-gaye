"""First cell of every student notebook on Snowflake Container Runtime (UNTESTED here: needs an account).

    %run snowflake/notebook_bootstrap.py      (or copy-paste)
    fetch("fsdd")                              # @DATASETS/fsdd/ -> /tmp/data/fsdd/
    push_results("results/results.jsonl", group="team3")

Notebook settings: compute pool DS_PROJECT_GPU_POOL, external access DS_PROJECT_PYPI_EAI +
DS_PROJECT_HF_EAI (+ GITHUB for the reference streaming model; kenlm is built from its PyPI source).
"""
import json
import subprocess
import sys
from pathlib import Path

PIP = ["jiwer", "hmmlearn", "librosa", "nvidia-ml-py", "soundfile", "pygtrie", "hypothesis"]


def install(extra=()):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP, *extra], check=True)
    # pyctcdecode's metadata pins numpy<2 and would downgrade numpy: install it without deps
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pyctcdecode==0.5.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kenlm"], check=False)


def session():
    from snowflake.snowpark.context import get_active_session
    return get_active_session()


def fetch(name: str, stage: str = "@DS_PROJECT.COURSE.DATASETS", dest: str = "/tmp/data"):
    """Copy a dataset folder from the shared stage to local disk (models read local files fast)."""
    out = Path(dest) / name
    out.mkdir(parents=True, exist_ok=True)
    session().file.get(f"{stage}/{name}/", str(out))
    return out


def push_results(jsonl: str, group: str, table: str = "DS_PROJECT.COURSE.BENCH_RESULTS"):
    """Append benchmark JSON lines to the class-wide table (one VARIANT per configuration)."""
    s = session()
    rows = [json.loads(line) for line in Path(jsonl).read_text().splitlines() if line.strip()]
    for r in rows:
        s.sql(f"INSERT INTO {table}(GROUP_NAME, RECORD) SELECT ?, PARSE_JSON(?)", params=[group, json.dumps(r)]).collect()
    return len(rows)


def gpu_check():
    import torch
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("NVML energy counter (mJ):", pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    except Exception as e:  # if this fails, energy must be estimated from power draw x time
        print("NVML energy counter unavailable:", e)
