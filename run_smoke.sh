#!/usr/bin/env bash
# Must finish in < 10 min and rerun a SMALL version of your pipeline end to end. Adapt it at every milestone.
# Run it with the project environment: `uv run bash run_smoke.sh`, or activate .venv first (README.md, Setup).
# Windows: Git Bash (comes with Git for Windows) or WSL, or paste the lines below into PowerShell one by one.
# So that it works everywhere, write plain `python -m ...` lines here: no bash-only syntax (export, $(...), &&-chains).
set -euo pipefail
python -m pytest -q tests/public
# add your own commands below, e.g. 20 digit strings through your decoder, or 5 LibriSpeech utterances
