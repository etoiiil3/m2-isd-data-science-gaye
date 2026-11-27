"""Day-3 Pareto figure: streaming systems on the axes of results/pareto.yaml (WER vs wait after release), with the
budget, plus WER vs commit lag. Rows task=asr_streaming, cfg_milestone=d3 of results/results.jsonl.

    python plot_streaming_pareto.py        -> results/figures/streaming_pareto.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

rows = [json.loads(l) for l in open("results/results.jsonl") if l.strip()]
df = pd.DataFrame([r for r in rows if r.get("task") == "asr_streaming" and r.get("cfg_milestone") == "d3"])
df = df.drop_duplicates(["model", "cfg_policy", "cfg_step_s"], keep="last")
names = {"openai/whisper-tiny.en": "whisper-tiny + LocalAgreement",
         "facebook/wav2vec2-base-960h": "wav2vec2-960h + LocalAgreement (ours)",
         "reference/fastconformer-114M-int8": "reference transducer (native streaming)"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for model, g in df.groupby("model", sort=False):
    g = g.sort_values("cfg_step_s")
    lab = names.get(model, model)
    axes[0].plot(1000 * g.finish_p50, 100 * g.wer_stream, "o-", label=lab)
    axes[1].plot(g.lag_p50, 100 * g.wer_stream, "o-", label=lab)
    close = len(g) > 1 and 1000 * (g.finish_p50.max() - g.finish_p50.min()) < 40   # steps with almost the same wait
    if close:
        axes[0].annotate("steps " + ", ".join(f"{v:g}" for v in g.cfg_step_s) + " s",
                         (1000 * g.finish_p50.max(), 100 * g.wer_stream.mean()), fontsize=7, xytext=(6, 3),
                         textcoords="offset points")
    for _, r in g.iterrows():
        txt = "native" if r.cfg_policy == "native" else f"step {r.cfg_step_s:g} s"
        if not close:
            axes[0].annotate(txt, (1000 * r.finish_p50, 100 * r.wer_stream), fontsize=7, xytext=(4, 3),
                             textcoords="offset points")
        axes[1].annotate(txt, (r.lag_p50, 100 * r.wer_stream), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax = axes[0]
ax.axvline(200, color="grey", ls="--", lw=1)
ax.axhline(5, color="grey", ls=":", lw=1)
ax.fill_between([0, 200], 0, 5, color="tab:green", alpha=0.08, label="budget: wait ≤ 0.2 s, WER ≤ 5%")
ax.set(xlabel="median wait after release (ms)  [finish_p50]", ylabel="WER of the committed text (%)",
       title="Streaming front on our laptop (20 LibriSpeech utterances)", xlim=(0, None), ylim=(0, None))
ax.legend(fontsize=7, loc="upper left")
axes[1].set(xlabel="median commit lag (s)  [lag_p50]", ylabel="WER of the committed text (%)",
            title="Words appear later with a larger step", ylim=(0, None))
axes[1].grid(alpha=0.3); ax.grid(alpha=0.3)
fig.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/figures/streaming_pareto.png", dpi=130)
print("results/figures/streaming_pareto.png")