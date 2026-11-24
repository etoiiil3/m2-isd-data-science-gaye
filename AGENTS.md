# AGENTS.md — Project S: speech recognition under real-time constraints

This file is read by AI coding assistants (Claude Code, Codex, Cursor, Copilot, ...) when they work in this
repository. It is also written for you, the student: it is a one-page map of the project and the rules your
assistant is asked to follow. Keep it in your repository; add your own group conventions at the end.

## What this project is
A 3-day course project (M2 ISD, Data science). Groups build speech recognisers, from a classical GMM-HMM to
neural models, add language models, and measure everything as **Pareto fronts**: accuracy (WER) against
latency, real-time factor, size or memory. The core skill is choosing and configuring components under a
budget, and defending each choice at the oral exam. Groups do not train neural models.

| Day | Lecture | Students implement (`TODO(dX)` stubs) | Benchmarks (`results/results.jsonl` rows) |
|---|---|---|---|
| d1 | L1, L2 | reference solution on branch `solutions` since 17 Nov (never on `main`): `common/pareto.py::pareto_mask_array`; `speech/ngram.py::NGramLM`; `speech/gmm_hmm.py`: `recognise_isolated`, `WordLoopDecoder._build`, `decode_emissions`, `train_bigram` | `gmm_isolated`, `ngram_lm`, `gmm_connected`, `gmm_beam` |
| d2 | L3, L4 | `speech/asr_benchmark.py::bench_model`; `speech/ctc_lm.py`: `greedy_ctc`, `sweep` | `asr`, `ctc_decoding` |
| d3 | L5, L6 | `speech/streaming_sim.py`: `local_agreement`, `simulate_streaming`, `stream_native`; stretch: `speech/streaming_live.py::LocalAgreementSession` | `asr_streaming` |

`SUBJECT.md` is the authority on what is required each day; `SUBMISSION_SPEC.md` on what is handed in.

## Map
| Path | What | Who edits it |
|---|---|---|
| `common/` | measurement harness (`bench.py`: timing, memory, result rows), Pareto tools (`pareto.py`) | provided, one stub |
| `speech/` | pipeline stages; `run_classical_baseline.py` runs the day-1 benchmarks | provided, with stubs |
| `asr_demo/` | live demo (server, page, backends, replay); `digits.py` = day-1 entries | provided |
| `my_backends.py` | your systems registered in the demo (examples inside: CTC + n-gram, streaming wrapper) | the group |
| `notebooks/` | workbenches: they call the package and plot the results; no code to write in them | provided |
| `tests/public/` | the contract of each stub, marked `d1`, `d2`, `d3`, `stretch` | provided, **do not edit** |
| `lectures/` | slides; each stub has a slide with its equation or pseudocode, and the lab slide lists them | provided |
| `results/` | `results.jsonl` (written by the scripts), `figures/`, `pareto.yaml` | the scripts and the group |
| `DECISIONS.md` | one entry per scientific choice, with evidence | the group |
| `scripts/fetch_data.py` | downloads data to `data/` and the demo models to `models/` (neither is committed); cross-platform | provided |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | the environment: `uv sync` (pinned by `uv.lock`) or `pip install -r requirements.txt` | provided; add a package to both |

## Commands
The student works on macOS, Linux or Windows (PowerShell): never suggest bash-only syntax without its PowerShell
equivalent, use `python -m ...` and `/` in paths (they work everywhere), and read `README.md` ("Setup") before
touching the environment.
```bash
uv sync                                        # create .venv (Python 3.12) from uv.lock; day 2: uv sync --extra lm
uv run python scripts/fetch_data.py            # data/ and models/ (finished steps are skipped on a rerun)
source .venv/bin/activate                      # macOS/Linux, or  uv run <command>  without activating
                                               # Windows PowerShell: .venv\Scripts\Activate.ps1
pytest -m d1                                   # public tests of a day (d2, d3, stretch)
export DSP_MILESTONE=d1                        # tag every result row with the current day (PowerShell: $env:DSP_MILESTONE = "d1")
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part isolated   # also: lm, connected, beam, plot
python -m speech.asr_benchmark                 # day 2 model zoo (CPU by default without a GPU); --plot draws the fronts
python -m speech.ctc_lm --arpa data/lm/3-gram.pruned.1e-7.arpa                        # day 2 CTC + n-gram, tuned on dev
python -m speech.streaming_sim                 # day 3 streaming front (~6 min, 7 rows); then --models <name> appends extra systems
python -m asr_demo.replay --librispeech data/LibriSpeech --n 8 --backends ...      # day 3: backends on the same recordings
jupyter lab notebooks/d1_afternoon.ipynb       # see what the code does
python -m asr_demo.server                      # live demo, http://localhost:8000
bash run_smoke.sh                              # < 10 min end-to-end check; uv run bash run_smoke.sh; Windows: Git Bash/WSL
```
Without uv: `python -m venv .venv`, activate it, `pip install -r requirements.txt` (README.md lists the extra
day-2 steps for `pyctcdecode` and `kenlm`).

## Conventions in the code
- A stub is `raise NotImplementedError("TODO(dX): <what to do>")`. On days 2 and 3 the functions are written
  except a few lines: each `raise` sits where its lines go, and its message says what to write (replace it, keep
  the indentation; a function can have several). On day 1 the `raise` stands for the whole body. Keep the
  signature and the docstring's contract. Everything that is not a stub is provided and tested.
- The day-1 reference solution is on branch `solutions` of the starter repository, not on `main`. Do not copy it
  into the group's code unless the students ask for it; then copy only the files they name
  (`git checkout course/solutions -- <file>`), never merge the branch.
- The docstring and the lecture slide named on the lab slide give the equation. The public tests define
  "implemented", not "good": the benchmarks decide what is good.
- Symbols follow the lectures' notation pages. Code names: `lm_weight` = λ, `word_penalty` = β (negative = a
  penalty), `beam` = Ω (score beam), `n_beam` = N_beam (top-N beam, 1 = greedy); day 2's pyctcdecode uses
  `alpha` = λ, `beta` = β (a bonus there), `beam_width` = N_beam, `beam_prune_logp` = −Ω.
- Probabilities are combined in the log domain: add log-probabilities, use `logsumexp` to add probabilities
  (lecture L1, "the log-sum-exp trick"); never multiply long chains of probabilities.
- Result rows are written only by the provided `log_record` (`common/bench.py`), never by hand.

## Rules for the assistant
1. **The students must understand and defend every line.** When implementing a stub, first state the idea in
   two or three sentences and point to the lecture slide; after implementing, summarise what changed and why,
   then run `pytest -m dX`. Prefer a small, readable implementation over a clever one.
2. **When the student asks for an explanation or a hint, give that, not the full solution.** Ask before writing
   a whole stub if the request is ambiguous.
3. **Never edit `tests/`, never weaken a test, never special-case test inputs.** If a provided function or test
   looks wrong, say so and suggest the group tells the instructor; do not patch provided code silently.
4. **Never invent or edit measurements.** Numbers in `DECISIONS.md`, `pareto.yaml`, reports and slides must come
   from rows in `results/results.jsonl` or from a figure produced by the scripts. If a number is missing, run the
   experiment or say it is missing.
5. **Experiments:** change one knob at a time; tune on development data and report on test data (the scripts
   already split them); report offline latency with a tail percentile (`lat_p90_ms`), and day 3 with `lag_p50`/`lag_p90` and `finish_p50` (in seconds), not only the mean; set
   `DSP_MILESTONE` before running.
6. **Every scientific choice gets a `DECISIONS.md` entry** (template inside): question, alternatives, protocol,
   evidence (rows or figure path), decision, what would change the decision. Offer to draft it from the
   group's own results; the group checks and owns it.
7. Do not commit `data/`, `models/`, `.venv/`, model weights or large caches (`.gitignore` covers the usual ones).
   Keep the repository installable (`pyproject.toml` + `uv.lock`, and `requirements.txt`: a new package goes in both,
   then `uv lock`) and `run_smoke.sh` under 10 minutes.
8. Write in English (code, comments, decisions, report).

## When stuck
Read, in this order: the stub's docstring, its public test, the lecture slide (lab slide → slide number), the
matching notebook section. Then run the smallest example that fails and look at shapes and values.

## Group conventions
<!-- Add here who works on what, naming rules, the machine used for the reported latencies, ... -->
