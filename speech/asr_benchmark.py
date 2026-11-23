"""Offline ASR Pareto front on LibriSpeech: WER vs real-time factor, latency, energy, size.

    export DSP_MILESTONE=d2
    python -m speech.asr_benchmark                      # CPU_MODELS x batch 1 and 16 on 200 test-clean utterances
    python -m speech.asr_benchmark --models openai/whisper-small.en --batch 1 --n 50   # one more point, cheaper
    python -m speech.asr_benchmark --plot               # fronts -> results/figures/asr_pareto.png

Defaults: --librispeech data/LibriSpeech (where scripts/fetch_data.py puts it), --device cuda if torch sees a GPU
else cpu, rows appended to results/results.jsonl (task "asr"). On a laptop CPU the default run (3 models x 2 batch
sizes, 22.5 min of audio) takes about 11 minutes once the models are downloaded (~0.5 GB); the first call of each
model downloads it. GPU_MODELS are slow on a CPU (whisper-small.en about 1.2 s and medium.en about 2.3 s per
utterance): run them with --n 50, or on a GPU.

batch=1 measures LATENCY (interactive use); batch=16 measures THROUGHPUT (offline transcription).
The two give different fronts -- one of the key messages of the project.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from common.bench import (BenchRecord, cuda_sync, default_device, energy_scope, log_record, peak_memory_mb,
                          reset_peak_memory, summarize_latencies)
from speech.data import load_librispeech, wer

# three families, small enough for a laptop CPU: >= 6 rows with the two batch sizes
CPU_MODELS = ["openai/whisper-tiny.en", "facebook/wav2vec2-base-960h", "UsefulSensors/moonshine-tiny"]
GPU_MODELS = [
    "openai/whisper-base.en", "openai/whisper-small.en", "openai/whisper-medium.en",
    "openai/whisper-large-v3-turbo", "distil-whisper/distil-large-v3",
    "facebook/wav2vec2-large-960h-lv60-self", "UsefulSensors/moonshine-base",
]


def bench_model(model_id, data, batch, device, fp16):
    """Benchmark one Hugging Face ASR model at one batch size.

    data: list of dicts with "audio" (float32 array), "sr" and "text" (from speech.data.load_librispeech).
    Returns (metrics, hypotheses): metrics has wer, rtf, n_params_M, energy_j_per_audio_min (None when not
    measurable), the keys of summarize_latencies (lat_p50_ms, lat_p90_ms, ...) and of peak_memory_mb.
    - fp16 only on a GPU (float32 on CPU).
    - The pipeline takes a list of {"raw": audio, "sampling_rate": sr} dicts and CONSUMES them (it pops the keys):
      pass fresh copies at every call, e.g. [dict(x) for x in inputs[i:i + batch]], or the second call gets {}.
    - Batching pads sequences: a tokenizer without a pad token (Moonshine) refuses to batch; set
      asr.tokenizer.pad_token_id = asr.model.config.eos_token_id when it is None.
    - At batch 1 there is one latency per utterance; at batch > 1, one per batch (what a request in that batch
      waits). Compare throughput with rtf.
    """
    import torch
    from transformers import pipeline

    # 1. Load the model in a Hugging Face ASR pipeline (fp16 only on a GPU)
    dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device, dtype=dtype)
    # Moonshine has no pad token: needed to batch several utterances
    if batch > 1 and getattr(asr, "tokenizer", None) is not None and asr.tokenizer.pad_token_id is None:
        asr.tokenizer.pad_token_id = asr.model.config.eos_token_id

    # 2. Model size
    n_params = sum(p.numel() for p in asr.model.parameters())

    # 3. Inputs in the format the pipeline expects; warm-up on 2 utterances (not timed)
    inputs = [{"raw": d["audio"], "sampling_rate": d["sr"]} for d in data]
    asr([dict(x) for x in inputs[:2]], batch_size=min(batch, 2))

    # 4. Timed loop: one call per batch, fresh copies because the pipeline empties its inputs
    reset_peak_memory()
    lats, hyps, res = [], [], {}
    with energy_scope(res):
        t_all = time.perf_counter()
        for i in range(0, len(inputs), batch):
            chunk = [dict(x) for x in inputs[i:i + batch]]
            t0 = time.perf_counter()
            out = asr(chunk, batch_size=batch)
            cuda_sync()  # GPU calls are asynchronous: wait before reading the clock
            lats.append(time.perf_counter() - t0)
            hyps.extend(o["text"] for o in out)
        total = time.perf_counter() - t_all

    # 5. Metrics
    audio_s = sum(len(d["audio"]) / d["sr"] for d in data)
    wer_value = wer([d["text"] for d in data], hyps)
    rtf = total / audio_s
    m = dict(wer=wer_value, rtf=rtf, n_params_M=n_params / 1e6,
             energy_j_per_audio_min=(res["energy_j"] / audio_s * 60) if res.get("energy_j") is not None else None)
    m.update(summarize_latencies(lats))  # lat_p50_ms, lat_p90_ms, ...
    m.update(peak_memory_mb())           # cpu_maxrss_mb (+ gpu_peak_mb)
    return m, hyps

def plot(path, out_png=Path("results/figures/asr_pareto.png")):
    """One front per batch size (WER vs RTF) from the "asr" rows of a results file; the last run of a
    configuration wins. Saved to results/figures/, where the grader looks for figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.bench import load_records
    from common.pareto import plot_pareto
    df = load_records(path)
    df = df[df.task == "asr"].copy()
    if df.empty:
        sys.exit(f"no 'asr' rows in {path}: run the benchmark first")
    for col, default in (("cfg_device", "?"), ("cfg_fp16", False)):
        if col not in df:
            df[col] = default
    df["cfg_device"] = df.cfg_device.fillna("?")
    df["cfg_fp16"] = df.cfg_fp16.fillna(False).astype(bool)
    df = df.drop_duplicates(["model", "cfg_batch", "cfg_device", "cfg_fp16"], keep="last")
    df["label"] = df.model.str.split("/").str[-1]
    several = df.cfg_device.nunique() > 1 or df.cfg_fp16.nunique() > 1
    if several:  # tell CPU / GPU / fp16 points apart
        df["label"] += " (" + df.cfg_device.astype(str) + np.where(df.cfg_fp16, ", fp16", "") + ")"
    batches = sorted(df.cfg_batch.unique())[:2]
    fig, axes = plt.subplots(1, len(batches), figsize=(7 * len(batches), 5.5), squeeze=False)
    devices = ", ".join(sorted(df.cfg_device.astype(str).unique()))
    for ax, b in zip(axes[0], batches):
        plot_pareto(df[df.cfg_batch == b].reset_index(drop=True), "rtf", "wer", "min", "min", label="label",
                    ax=ax, logx=True, title=f"WER vs real-time factor, batch={b} ({devices})")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"figure: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="ASR model zoo on LibriSpeech: WER vs cost.")
    ap.add_argument("--librispeech", type=Path, default=Path("data/LibriSpeech"))
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=CPU_MODELS, help="Hugging Face ids (default: CPU_MODELS)")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--fp16", action="store_true", help="half precision (GPU only; ignored on CPU)")
    ap.add_argument("--device", default=default_device(), help="cuda, cpu, ... (default: cuda if available)")
    ap.add_argument("--out", type=Path, default=Path("results/results.jsonl"))
    ap.add_argument("--plot", type=Path, nargs="?", const=Path("results/results.jsonl"),
                    help="draw the fronts from this results file (default: results/results.jsonl) and exit")
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    fp16 = args.fp16 and args.device.startswith("cuda")
    if args.fp16 and not fp16:
        print(f"--fp16 ignored on {args.device}: float32")
    data = load_librispeech(args.librispeech, args.subset, args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data) / 60:.1f} min of audio, "
          f"device {args.device}")
    logged = 0
    for mid in args.models:
        for b in args.batch:
            try:
                m, _ = bench_model(mid, data, b, args.device, fp16)
            except NotImplementedError:
                raise  # your TODO: not a model to skip
            except Exception as e:  # a model that does not load or run here: keep going with the others
                print(f"[skip] {mid} batch={b}: {type(e).__name__}: {e}")
                continue
            log_record(BenchRecord("asr", mid, dict(subset=args.subset, batch=b, fp16=fp16, device=args.device,
                                                    n=len(data)), m), args.out)
            logged += 1
            print(mid, b, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    if not logged:
        sys.exit("no configuration ran (see the [skip] lines above): nothing written")
    print(f"{logged} rows appended to {args.out}")


if __name__ == "__main__":
    main()
