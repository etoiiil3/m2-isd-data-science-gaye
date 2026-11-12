# M2 ISD 26-27 — Data science — Project S

This repository is the student material for M2 ISD 26-27, Data science, Project S (speech recognition).

- Course page: https://ecampus.paris-saclay.fr/course/section.php?id=785660 (section Data science of https://ecampus.paris-saclay.fr/course/view.php?id=256459).
- Lectures: `lectures/L1_*.pdf` and `lectures/L2_*.pdf` (PDF, notation summarised in the last slides); L3–L6 are still HTML decks, to open in a browser (arrows move, F fullscreen, N speaker notes).
- Project: `SUBJECT.md`, `SUBMISSION_SPEC.md`, `DECISIONS.md`, then the setup below.
- Datasets are downloaded by `scripts/fetch_data.sh` and are not in the repository.

# Project S — Speech recognition under real-time constraints (student bundle)

Start here: `SUBJECT.md` (what to do), `SUBMISSION_SPEC.md` (what to hand in), `DECISIONS.md` (template).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_data.sh data            # FSDD digits, LibriSpeech test-clean, 3-gram LM (speech data only)
pytest -m d1                               # failing tests = what you must implement on day 1
export DSP_MILESTONE=d1
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part isolated --states 2 3 5 --mixes 1 2
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part plot
```
Live demo (from day 1, on your laptop): `python -m asr_demo.server` then open http://localhost:8000.
The `digits/*` entries follow the day-1 lab: `digits/0-dtw-baseline` works at once; every other entry starts
working when you implement the function it names (details: `asr_demo/digits.py`).
On Snowflake: see `snowflake/notebook_bootstrap.py` (first cell of your notebook).

| Folder | Content |
|---|---|
| `common/` | measurement harness (`bench.py`) and Pareto tools (`pareto.py`, one TODO) |
| `speech/` | pipeline stages; `TODO(mX)` stubs are yours, everything else is provided |
| `asr_demo/` | live demo: server, browser page, backend registry, replay tool |
| `tests/public/` | daily contracts (`pytest -m d2`; demo: `pytest -m stretch`) |
| `scripts/` | data download |
| `results/` | your measurements, figures, `pareto.yaml` |
