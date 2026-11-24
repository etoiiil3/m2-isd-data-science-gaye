# Pull the day-3 update

Read this before you pull. The commands are the same in a macOS/Linux terminal and in Windows PowerShell.
They use `course` as the name of the starter repository. If `git remote -v` shows
`https://github.com/stephane-rivaud/m2-isd-data-science` under the name `origin`, use `origin` instead of
`course` below.

Day-1 and day-2 code, notebooks and tests are unchanged. The update does not modify `DECISIONS.md`,
`results/pareto.yaml`, or `run_smoke.sh`. If you have not edited `my_backends.py`, the old day-3 files
(`speech/streaming_sim.py`, `speech/streaming_live.py`), or the L5/L6 HTML slides, the pull merges with no
conflict.

## 1. Commit first

Git refuses to pull over uncommitted changes to a file the update modifies.

```
git add -A
git commit -m "work before the day-3 update"
```

If git says "nothing to commit", your work is already committed: go on.

If `git remote -v` shows no remote pointing at `https://github.com/stephane-rivaud/m2-isd-data-science`, add it:

```
git remote add course https://github.com/stephane-rivaud/m2-isd-data-science.git
```

## 2. Pull

```
git pull --no-rebase --no-edit course main
```

If git reports no conflict, skip to [the new reference model](#6-new-reference-model).
If git says "refusing to merge unrelated histories", use [the eCampus zip](#5-started-from-the-ecampus-zip) instead.

## 3. Conflicts

Only the files git lists. For each one:

- **`my_backends.py`**, and a group document such as **`README.md`**: keep yours.
  ```
  git checkout --ours my_backends.py
  ```
  The same command with the document's name (`git checkout --ours README.md`). New `my_backends.py` examples:
  `git show course/main:my_backends.py`.
- **`speech/streaming_sim.py` and `speech/streaming_live.py`**: take the new files. They add functions the tests
  and notebooks need.
  ```
  git checkout --theirs speech/streaming_sim.py speech/streaming_live.py
  ```
  Your previous body stays in history: `git show HEAD~1:speech/streaming_sim.py` (and the same for
  `speech/streaming_live.py`). Move that logic into the new `TODO(d3)` holes.
- **`lectures/L5_Deciding_what_to_deploy.html` and `lectures/L6_Streaming_and_the_live_demo.html`**: this is
  `CONFLICT (modify/delete)`. `git checkout --theirs` fails. `git rm` only the HTML files git lists. The PDFs
  are already staged. Your notes stay at `git show HEAD:lectures/L5_Deciding_what_to_deploy.html` (and the same
  for the L6 file).
  ```
  git rm lectures/L5_Deciding_what_to_deploy.html lectures/L6_Streaming_and_the_live_demo.html
  ```
  `git rm` already stages the deletion.
- **Other provided code** (`asr_demo/`, `scripts/`, `tests/`): take the new version.
  ```
  git checkout --theirs <file>
  ```

## 4. Finish the merge

`git add` every resolved file (`git rm` already staged the HTML), then commit:

```
git add my_backends.py speech/streaming_sim.py speech/streaming_live.py
git commit --no-edit
```

Use the files git actually listed. If `git commit` says unmerged files remain, run `git status` and resolve
those. To stop and go back to where you were: `git merge --abort`.

## 5. Started from the eCampus zip

Only if `git pull` says "refusing to merge unrelated histories". These three lines pull the day-3 files and
leave `DECISIONS.md`, `README.md`, `AGENTS.md`, `results/`, `run_smoke.sh`, `my_backends.py`, and your
day-1/day-2 files untouched.

If you already edited `speech/streaming_sim.py` or `speech/streaming_live.py`, copy them out first with
`git show HEAD:speech/streaming_sim.py` (and the same for `speech/streaming_live.py`) and move that logic into
the new `TODO(d3)` holes after the checkout. If you edited the L5 or L6 HTML slides, copy them out first with
`git show HEAD:lectures/L5_Deciding_what_to_deploy.html`.

```
git fetch course
git checkout course/main -- speech/streaming_sim.py speech/streaming_live.py asr_demo scripts/fetch_data.py notebooks/d3_morning.ipynb notebooks/d3_afternoon.ipynb lectures tests/public/test_d3_streaming.py tests/public/test_d3_demo_contract.py SUBJECT.md SUBMISSION_SPEC.md
git commit -m "day-3 course update"
```

## 6. New reference model

The demo's reference streaming model changed. The script skips what you already have:

```
python scripts/fetch_data.py
```

or `uv run python scripts/fetch_data.py`.

Delivery rules (7 rows, `pareto.yaml` budget in seconds, legacy rows) are in `SUBJECT.md`, "Day 3".
