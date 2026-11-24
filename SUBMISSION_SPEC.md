# What to hand in — Project S

One Git repository per group (this bundle + your work), tagged `d1`, `d2`, `d3` at the end of each
day, and `final` before the defence. The instructor pre-checks each tag automatically, then reads it.

```
README.md            how to run; one line per member: who did what
DECISIONS.md         decision log (≥ 1 entry per day, format below)
my_backends.py       your systems registered in the live demo (day 3, not graded)
REPORT.md            final only; level-2 headings: Task and constraints · Method · Benchmark protocol ·
                     Pareto analysis · Pipeline · Limits · Reproducibility · Contributions
requirements.txt     keep it installable
run_smoke.sh         < 10 min: reruns a SMALL version of your pipeline end to end (add -m "not stretch" if skipping stretch)
results/results.jsonl   one JSON line per measured configuration (written by the provided scripts)
results/pareto.yaml     columns of your main front, budget, claimed front, chosen configuration
results/figures/*.png   at least one Pareto plot per day
slides.pdf           final only
```

Rows in `results.jsonl` must carry `task`, `model`, `cfg_milestone` (set `export DSP_MILESTONE=d1` before
running) and the metric columns listed in `SUBJECT.md` for that milestone; report offline latency with a tail
percentile (`lat_p90_ms`), and day-3 streaming latency with `lag_p50`, `lag_p90` and `finish_p50` (median wait
after release, in seconds).

## DECISIONS.md entry
```markdown
## D3 — <decision in a few words>   (milestone: d2)
- Question: what did you need to decide?
- Alternatives: what did you compare?
- Protocol: data, hardware, metrics, number of runs
- Evidence: figure path and/or rows (e.g. cfg_milestone=d2)
- Decision: what you chose
- Would change our mind if: which result or constraint would make you choose differently
```

## Automatic pre-checks (per tag)
Public tests of the milestones so far (`pytest -m "d1 or d2"`), plus instructor tests on the same
functions · required rows and columns · decision log complete with resolvable evidence · Pareto front in
`pareto.yaml` consistent with your own data · budget declared · figures present · tags present.
These checks inform the instructor; they do not replace reading your work or the oral exam.
