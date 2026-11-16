#!/usr/bin/env python3
"""Data for the speech project (~500 MB, a few minutes): FSDD, LibriSpeech test-clean, a pruned 3-gram LM and the
reference streaming model of the live demo. Pure standard library (urllib, zipfile, tarfile, gzip): the same
command works on macOS, Linux and Windows, and needs no wget, tar, gunzip or git.

    python scripts/fetch_data.py                     # everything: data/ and models/ at the root of the repository
    python scripts/fetch_data.py D:\\asr_data         # another data folder (models/ stays at the repository root)
    python scripts/fetch_data.py --only fsdd         # day 1 only (10 MB); also: librispeech, lm, model, lm4

What lands where (a step whose target exists is skipped, so the command can be rerun after a failure):
    fsdd          <DATA>/fsdd/recordings/*.wav              day 1, spoken digits (GitHub archive of the dataset)
    librispeech   <DATA>/LibriSpeech/test-clean/            days 2-3, 346 MB
    lm            <DATA>/lm/3-gram.pruned.1e-7.arpa         days 2-3, 3-gram LM, LOWER-CASED (wav2vec2 labels are
                                                            lower-cased in speech/ctc_lm.py), about 100 MB unpacked
    model         <MODELS>/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/    the demo's reference recogniser, ~130 MB
    lm4           <DATA>/lm/4-gram.arpa                     OPTIONAL, only with --only lm4: day-2 stretch, 4-gram LM,
                                                            lower-cased; 1.3 GB download, ~3 GB unpacked, several GB
                                                            of RAM to load (a GPU machine, or a laptop with 16 GB)

Each step is independent: a failed download only skips the steps that need it, and the exit status is 1.
Downloads go to a temporary name and are renamed when complete, so an interrupted run never leaves a folder that
looks finished.
"""
from __future__ import annotations

import argparse
import bz2  # noqa: F401  (tarfile "r:bz2" needs it: fail early, with a clear message, on a Python built without it)
import gzip
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # scripts/ (student bundle) and dryrun/ (teacher repository) are both one level down

FSDD_URL = ["https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"]
# OpenSLR has mirrors; the first one that answers wins.
_SLR = ["https://www.openslr.org/resources", "https://openslr.elda.org/resources", "https://openslr.trmal.net/resources"]
LIBRISPEECH_URL = [f"{m}/12/test-clean.tar.gz" for m in _SLR]
LM_URL = [f"{m}/11/3-gram.pruned.1e-7.arpa.gz" for m in _SLR]
MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
MODEL_URL = [f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"]
LM_FILE = "3-gram.pruned.1e-7.arpa"
LM4_FILE = "4-gram.arpa"
LM4_URL = [f"{m}/11/{LM4_FILE}.gz" for m in _SLR]


# ------------------------------------------------------------------------------------------------ downloading
def _ssl_context():
    """The system certificates, or certifi's when it is installed (python.org builds on macOS ship without)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_one(url: str, dest: Path, ctx) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "project-s-fetch-data/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown, t0 = 0, -1, time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if total and pct // 10 != shown // 10:      # one line every 10 %: readable in a log as well as a terminal
                shown = pct
                print(f"    {pct:3d} %  {done / 1e6:6.0f} / {total / 1e6:.0f} MB  ({done / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
    if total and done != total:
        raise OSError(f"incomplete download: {done} of {total} bytes")


def download(urls: list[str], dest: Path, retries: int = 3) -> None:
    """Download the first URL that works to `dest` (written as dest.part, then renamed). Raises if all fail."""
    ctx, part, last = _ssl_context(), dest.with_name(dest.name + ".part"), None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                print(f"    GET {url}" + (f" (attempt {attempt})" if attempt > 1 else ""), flush=True)
                _download_one(url, part, ctx)
                os.replace(part, dest)
                return
            except ssl.SSLError as e:
                last = e
                print(f"    certificate error: {e}\n    (macOS python.org build: run 'Install Certificates.command', or "
                      "'pip install certifi')", flush=True)
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = e
                print(f"    failed: {e}", flush=True)
                time.sleep(2 * attempt)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed ({last})")


def _safe_member(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    if base.resolve() != target and base.resolve() not in target.parents:
        raise RuntimeError(f"unsafe path in archive: {name}")
    return target


def _extract_tar(archive: Path, dest: Path, mode: str) -> None:
    with tarfile.open(archive, mode) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
        else:                                            # Python < 3.12 without the security backport
            for m in t.getmembers():
                _safe_member(dest, m.name)
            t.extractall(dest)


def _move_into_place(src: Path, dst: Path) -> None:
    """Rename src to dst, replacing an incomplete dst (the step only runs when dst is not complete)."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


# ------------------------------------------------------------------------------------------------ the four steps
def fetch_fsdd(data: Path, models: Path) -> None:
    rec = data / "fsdd/recordings"
    if any(rec.glob("*.wav")):
        print("  already there, skipped")
        return
    with tempfile.TemporaryDirectory(dir=data) as tmp:
        tmp = Path(tmp)
        zp = tmp / "fsdd.zip"
        download(FSDD_URL, zp)
        out = tmp / "fsdd"
        with zipfile.ZipFile(zp) as z:                   # the archive has one top folder (free-spoken-digit-dataset-master/)
            for info in z.infolist():
                parts = Path(info.filename).parts[1:]
                if not parts or info.is_dir():
                    continue
                target = _safe_member(out, str(Path(*parts)))
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        if not any((out / "recordings").glob("*.wav")):
            raise RuntimeError("the archive has no recordings/ folder of wav files")
        _move_into_place(out, data / "fsdd")


def fetch_librispeech(data: Path, models: Path) -> None:
    final = data / "LibriSpeech/test-clean"
    if any(final.rglob("*.flac")):
        print("  already there, skipped")
        return
    with tempfile.TemporaryDirectory(dir=data) as tmp:
        tmp = Path(tmp)
        tg = tmp / "test-clean.tar.gz"
        download(LIBRISPEECH_URL, tg)
        print("    extracting ...", flush=True)
        _extract_tar(tg, tmp / "x", "r:gz")
        tg.unlink()
        _move_into_place(tmp / "x/LibriSpeech/test-clean", final)


def fetch_lm(data: Path, models: Path, urls: list[str] = LM_URL, name: str = LM_FILE) -> None:
    final = data / "lm" / name
    if final.exists() and final.stat().st_size > 0:
        print("  already there, skipped")
        return
    (data / "lm").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data) as tmp:
        tmp = Path(tmp)
        gz = tmp / "lm.arpa.gz"
        download(urls, gz)
        print("    unpacking and lower-casing ...", flush=True)
        out = tmp / name
        with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                dst.write(chunk.lower())                 # bytes: LF stay LF on Windows, ARPA words are ASCII
        os.replace(out, final)


def fetch_model(data: Path, models: Path) -> None:
    final = models / MODEL_NAME
    if (final / "tokens.txt").exists():
        print("  already there, skipped")
        return
    models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models) as tmp:
        tmp = Path(tmp)
        tb = tmp / "model.tar.bz2"
        download(MODEL_URL, tb)
        _extract_tar(tb, tmp / "x", "r:bz2")
        tb.unlink()
        if not (tmp / "x" / MODEL_NAME / "tokens.txt").exists():
            raise RuntimeError(f"{MODEL_NAME}/tokens.txt not found in the archive")
        _move_into_place(tmp / "x" / MODEL_NAME, final)


STEPS = {
    "fsdd": ("FSDD (day 1, spoken digits, 10 MB)", fetch_fsdd),
    "model": ("Reference on-device streaming model (sherpa-onnx zipformer, 20M params, ~130 MB) -> models/", fetch_model),
    "librispeech": ("LibriSpeech test-clean (days 2-3, 346 MB)", fetch_librispeech),
    "lm": ("Pruned 3-gram LM, lower-cased (days 2-3, ~34 MB download)", fetch_lm),
    "lm4": ("4-gram LM, lower-cased (optional day-2 stretch, 1.3 GB download)",
            lambda data, models: fetch_lm(data, models, LM4_URL, LM4_FILE)),
}
DEFAULT_STEPS = ["fsdd", "model", "librispeech", "lm"]   # lm4 only on request


def folder_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", type=Path, default=Path("data"), help="data folder (default: data)")
    ap.add_argument("--models", type=Path, default=ROOT / "models", help="folder of the demo model (default: <repository>/models)")
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run only these steps (default: all but lm4)")
    a = ap.parse_args()
    data, models = a.data.resolve(), a.models.resolve()
    data.mkdir(parents=True, exist_ok=True)
    failed = []
    for key in a.only or DEFAULT_STEPS:
        title, fn = STEPS[key]
        print(f"== {title}", flush=True)
        try:
            fn(data, models)
            print(f"  [ok] {key}", flush=True)
        except Exception as e:  # keep going: the other steps are independent
            failed.append(key)
            print(f"  [FAILED] {key}: {e}\n           -- the steps that need it will be skipped; rerun the command to retry", flush=True)
    print(f"data: {data} ({folder_size(data):.0f} MB), models: {models} ({folder_size(models):.0f} MB)")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
