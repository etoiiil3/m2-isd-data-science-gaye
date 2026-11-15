import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" if (_ROOT / "src").is_dir() else _ROOT))  # teacher repo: src/; bundles: root


def pytest_configure(config):
    for m in ("d1", "d2", "d3", "final", "stretch"):
        config.addinivalue_line("markers", f"{m}: milestone {m} contract")


@pytest.fixture
def fake_torch(monkeypatch):
    """Minimal torch stand-in so CPU-only machines can test code paths that only need dtypes/sync."""
    if "torch" in sys.modules and not getattr(sys.modules["torch"], "_is_fake", False):
        return sys.modules["torch"]
    t = types.ModuleType("torch")
    t._is_fake = True
    t.float16, t.float32, t.bfloat16 = "float16", "float32", "bfloat16"
    t.__version__ = "fake"
    t.get_num_threads = lambda: 1
    t.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)

    class _NoGrad:
        def __call__(self, f=None):
            return f if f else self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    t.inference_mode = _NoGrad()
    t.no_grad = _NoGrad()
    monkeypatch.setitem(sys.modules, "torch", t)
    return t
