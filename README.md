# M2 ISD 26-27 — Data science — Project S: speech recognition

This repository is the student material for M2 ISD 26-27, Data science, Project S (speech recognition under
real-time constraints). Each group starts its own Git repository from it.

- Course page: https://ecampus.paris-saclay.fr/course/section.php?id=785660 (section Data science of
  https://ecampus.paris-saclay.fr/course/view.php?id=256459).
- Start here: `SUBJECT.md` (what to do, day by day), `SUBMISSION_SPEC.md` (what to hand in), `DECISIONS.md`
  (your decision log). `ANNOUNCEMENT.md` and `WELCOME_SESSION.md` describe the course.
- Lectures: `lectures/` — PDFs (notation summarised in the last slides); the lectures not yet converted are HTML
  decks to open in a browser (arrows move, F fullscreen, N speaker notes).
- Working with an AI assistant: `AGENTS.md` is written for it (and for you): a map of the repository, the
  commands, and the rules it should follow.
- Datasets are downloaded by `scripts/fetch_data.py` (same command on macOS, Linux and Windows) and are not in the
  repository.

## Setup
Everything runs on a laptop CPU, on macOS, Linux and Windows. You need **Python 3.12** and a recent `git`;
with uv (below) you do not even need to install Python: uv downloads 3.12 for you. Run every command from the
root of this repository.

### With uv (recommended)
Install uv once ([documentation](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Open a new terminal, then create the environment (`pyproject.toml` and `uv.lock` pin every package; the first
run downloads about 1 GB) and the data:
```bash
uv sync                                    # creates .venv with Python 3.12 and everything below
uv run python scripts/fetch_data.py        # FSDD digits, LibriSpeech test-clean, 3-gram LM, demo model -> data/, models/
uv run pytest -m d1                        # failing tests = what you must implement on day 1
```
`uv run <command>` runs a command inside `.venv` without activating it. If you prefer to activate it, then drop
`uv run` from every command of this README:
```bash
source .venv/bin/activate                  # macOS / Linux
```
```powershell
.venv\Scripts\Activate.ps1                 # Windows PowerShell
# "running scripts is disabled on this system": run  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned  once, or use uv run
# cmd.exe instead of PowerShell:  .venv\Scripts\activate.bat
```
**Day 2** (CTC decoding with an n-gram LM) needs two more packages, `pyctcdecode` and `kenlm`:
```bash
uv sync --extra lm                         # always add --extra lm from now on: a plain `uv sync` removes them again
```
`kenlm` has no ready-made package: it is **compiled** when you install it, so you need a C++ compiler. macOS: `xcode-select --install`.
Ubuntu/Debian: `sudo apt install build-essential`. Windows: install the
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop
development with C++" workload, open a new PowerShell, and rerun the command. If it still fails on Windows
(this is the most fragile step), do not lose time: use **WSL2** (`wsl --install` in an administrator PowerShell
installs Ubuntu; clone the repository inside the Linux home folder, not under `/mnt/c`, and follow the Linux
steps there), or run the n-gram part of day 2 on Snowflake or Kaggle where it installs, and tell the instructor.
Everything that does not use an n-gram LM (all of day 1, the model zoo, greedy and plain beam decoding) works
without `kenlm`: `uv sync --extra beam` installs `pyctcdecode` alone.

### With pip (if you cannot use uv)
```bash
python -m venv .venv                       # Windows: py -3.12 -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt  # Linux / Windows: first  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/fetch_data.py
pytest -m d1
```
Day 2, in the activated environment (`--no-deps`: the metadata of `pyctcdecode` would downgrade numpy):
```bash
python -m pip install --no-deps pyctcdecode==0.5.0
python -m pip install kenlm                # compiler needed, see above (skip it for beam search without an n-gram LM)
```

### Good to know
- **Windows:** set `PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8 = "1"` for this window, or
  `setx PYTHONUTF8 1` once, then open a new window) so that text files and output with Greek letters never fail
  on the legacy code page. Keep the repository path short and without spaces if you can (long paths break some tools).
- **Environment variables** differ by shell. bash/zsh: `export DSP_MILESTONE=d1`; PowerShell:
  `$env:DSP_MILESTONE = "d1"`; cmd.exe: `set DSP_MILESTONE=d1`. Paths with `/` work in PowerShell too.
- **Intel Macs:** current PyTorch, and numba (needed by librosa, hence by day 1 too), no longer ship macOS Intel
  wheels, so `uv sync` fails there. Work on another machine of the group, or on Snowflake / Kaggle.
- **NVIDIA GPU:** the lock file installs the small CPU build of torch. A GPU only speeds up the large models of
  day 2. To use it, pick your CUDA index on https://pytorch.org/get-started/locally/ and run
  `uv pip install --reinstall torch --index-url <that index>`, then use `uv run --no-sync ...` so that uv keeps it.
- **Downloads** can be rerun: finished steps are skipped. `python scripts/fetch_data.py --only fsdd` fetches the
  10 MB of day 1 alone (also `librispeech`, `lm`, `model`, and the optional 1.3 GB 4-gram `lm4`).
- **`bash run_smoke.sh`** is a bash script. macOS and Linux: `uv run bash run_smoke.sh` (or activate the
  environment first). Windows: the same command in Git Bash (installed with Git for Windows) or WSL; or paste its
  lines into PowerShell one by one. Keep the lines of `run_smoke.sh` plain `python -m ...` commands, with no
  bash-only syntax, so that they work in every shell.

## Day 1
The commands assume an activated environment; otherwise prefix each with `uv run`.
```bash
jupyter lab notebooks/d1_morning.ipynb     # morning workbench: figures of what your code does
jupyter lab notebooks/d1_afternoon.ipynb   # afternoon: decoding graph, trellis, lambda/beta, beam search
export DSP_MILESTONE=d1                    # PowerShell: $env:DSP_MILESTONE = "d1"
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part isolated --states 2 3 5 8 --mixes 1 2 4
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part lm
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part connected
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part beam
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part plot
```
Live demo (from day 1, on your laptop): `python -m asr_demo.server` then open http://localhost:8000 (same on
macOS, Linux and Windows; allow the microphone in the browser; a firewall prompt on Windows can be refused,
the demo only listens on this computer).
The `digits/*` entries follow the day-1 lab: `digits/0-dtw-baseline` works at once; every other entry starts
working when you implement the function it names (details: `asr_demo/digits.py`).
On Snowflake: see `snowflake/notebook_bootstrap.py` (first cell of your notebook).

## Layout
| Folder | Content |
|---|---|
| `common/` | measurement harness (`bench.py`) and Pareto tools (`pareto.py`, one TODO) |
| `speech/` | pipeline stages; `TODO(dX)` stubs are yours, everything else is provided |
| `asr_demo/` | live demo: server, browser page, backend registry, replay tool |
| `notebooks/` | workbenches that draw what your code does (no code to write in them) |
| `tests/public/` | daily contracts: `pytest -m d1` (d2, d3; demo stretch: `pytest -m stretch`) |
| `lectures/` | the course slides |
| `scripts/` | data download |
| `results/` | your measurements (`results.jsonl`), figures, `pareto.yaml` |
