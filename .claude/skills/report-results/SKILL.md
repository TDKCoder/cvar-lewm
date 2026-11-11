---
name: report-results
description: Turn finished runs into verified results - aggregate per-episode logs into success rate / worst-10% cost with bootstrap CIs, independently re-verify every number, make figures, and write the docs/EXPERIMENTS.md entry. Also writes the weekly status report. Use after any eval or training job finishes, when the user asks "how did it go", or for "/report-results weekly".
argument-hint: "[runs glob | weekly]"
---

# Reporting results

Two modes: **experiment** (default; argument is a runs glob like `runs/eval-*-p015-*`) and
**weekly** (argument `weekly`).

## Data contract (what every eval writes)

`runs/<RUN_ID>/episodes.jsonl`, one JSON object per finished episode:

```json
{"run_id": "eval-flow-cvar-p015-s0", "env": "pusht", "p": 0.15, "head": "flow",
 "cost": "cvar", "alpha": 0.25, "beta": null, "S": 8, "n_euler": 4, "seed": 0,
 "episode": 17, "success": true, "final_cost": 0.0123, "plan_time_s": 0.41}
```

Required keys: `env, p, head, cost, seed, episode, success, final_cost`. Optional: `alpha, beta, S,
n_euler, plan_time_s`. A cell is everything except `seed` and `episode` (pooled over seeds; pass
`--by-seed` to keep seeds separate). `(cell, seed, episode)` must be unique: resumed runs must not
duplicate episodes. The aggregator refuses duplicates.

Training runs write `runs/<RUN_ID>/metrics.jsonl` (`step, wall_s, train_loss, val_loss, ...`).

## Experiment mode: steps (show each step's output to the user)

0. **Pull.** `scripts/sync.sh pull "<glob>"` copies the small files (jsonl, logs, env.txt) from the
   cluster into `runs/` here. Everything below runs locally with `.venv-analysis/bin/python`.
1. **Inventory.** List the run dirs matched, episodes per run, and any run whose `env.txt` lacks a
   `done` line (incomplete). Say which runs you're excluding and why.
2. **Aggregate.**
   `.venv-analysis/bin/python .claude/skills/report-results/scripts/aggregate.py "runs/<glob>" --out results/<exp-name>`
   Produces `summary.csv`, `summary.md`, `manifest.json` (input files, line counts, sha256).
   Metrics: success rate with 95% percentile-bootstrap CI (10,000 resamples, seed 0); worst-10%
   final cost = mean of the top `ceil(0.1 n)` final costs, with bootstrap CI; failure rate; mean plan time.
3. **Verify independently.**
   `.venv-analysis/bin/python .claude/skills/report-results/scripts/verify_summary.py "runs/<glob>" results/<exp-name>/summary.csv`
   It recomputes counts, success rates and worst-10% from the raw files with a separate pure-Python
   path and compares Wilson intervals to the bootstrap CIs. **If it exits non-zero, stop and report
   the mismatch; don't publish numbers.**
4. **Figures** in `notebooks/<exp-name>.ipynb` (the user prefers notebooks), saved to
   `results/<exp-name>/*.png` and `.pdf`. House style: success rate vs `p` with CI error bars, one
   line per head, panels per cost; worst-10% cost vs `p` the same way. Label axes with units, put n
   per point in the caption, and use the same colors per head across all figures
   (deterministic = gray, mixture = blue, flow = orange).
5. **Interpret against the hypotheses** (H1-H3 in `docs/project-spec.md`). For each: supported /
   not supported / inconclusive, with the specific numbers and whether CIs overlap. Overlapping
   95% CIs means "no evidence of a difference at this n", not "equal". Give the equation for any
   derived quantity. Say what would change your mind.
6. **Write the entry** at the top of `docs/EXPERIMENTS.md` using `templates/experiment-entry.md`.
   Commit `results/<exp-name>/` (small files only) and the entry.

## Weekly mode

Fill `templates/weekly-status.md` from: `git log --since="1 week ago" --oneline`,
`docs/EXPERIMENTS.md` entries from the week, `docs/gpu-ledger.md` (SU used this week vs 10),
`docs/decisions.md`, and the schedule in `docs/project-spec.md` section 9. Be blunt about slippage
and name the cut you'd make. Save to `docs/status/YYYY-MM-DD.md` and show it to the user.

## Rules

- Never type a number into a report that didn't come from `summary.csv` or a script output in this
  session. Quote file paths next to numbers.
- Report n with every rate. Report negative and null results in the same detail as positive ones.
- Don't drop episodes or seeds after seeing results. If you must exclude something, say why before
  looking at its numbers, and log it in `docs/decisions.md`.
