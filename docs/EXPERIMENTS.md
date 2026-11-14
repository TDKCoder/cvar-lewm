# Experiment log

Newest entry on top. One entry per experiment (a group of runs answering one question).
Template: `.claude/skills/report-results/templates/experiment-entry.md`.

<!-- entries below -->
## 2026-11-13: Phase 0 (week A): infrastructure, pretrained LeWM on CPU, stochastic PushT spread

**Question / hypothesis:** none of H1-H3 yet. Phase 0 checks that the pieces work and sets the
stochastic-PushT magnitudes: does the pretrained checkpoint load and plan on CPU, does p = 0
reproduce upstream PushT exactly, how far does a fixed plan's outcome spread under kicks, and what
do data and planning cost.

**Setup:** CPU worker pod (4 CPUs), ephemeral venv from `requirements.lock.txt`. Pretrained
`quentinll/lewm-pusht`. Spread study: upstream's 50 eval pairs x 32 kick seeds, expert's 25 actions
open loop, kicks small / medium / large x p in {0, 0.05, 0.15} (+ medium with physics on).
Code at commits up to `35caf5b`. GPU smoke job `jobs/submitted/smoke-eval-pusht.sh` submitted
2026-11-13 18:46 (peak), est 1.0 h; results pending.

**Commands:**
```
scripts/remote.sh "scripts/withvenv.sh python -m pytest -q tests/"                       # 51 passed
scripts/remote.sh "scripts/withvenv.sh python -m cvar.planning.bench_cpu --out runs/phase0-cpu-bench"
scripts/remote.sh "bash scripts/snapshot_run.sh spread python -m cvar.envs.spread_study --out runs/phase0-spread --n-plans 50 --seeds 32"
.venv-analysis/bin/python cvar/envs/plot_spread.py runs/phase0-spread --out results/phase0/spread
scripts/remote.sh "scripts/withvenv.sh python -m cvar.data.render_check --out runs/phase0-render-check"
```

**Results** (all from files in `results/phase0/`):

| what | number | file |
|---|---|---|
| params (encoder / predictor / action enc / projector / pred_proj / total) | 5,501,376 / 10,791,360 / 156,206 / 792,768 / 792,768 / 18,034,478 | `pretrained_cpu.json` |
| default plan call (300 x 30 CEM), CPU, 4 threads | 23.9 s median of 3 (0.80 s per get_cost) | `pretrained_cpu.json` |
| env steps/s, CPU: no render / 224 px render | 3,373 / 416 | `data_probe.json` |
| frozen encoder on CPU, batch 64 | 41.7 frames/s | `data_probe.json` |
| eval states + scalers re-derived from the real h5 by independent code | 20/20 checks pass | `expert_eval_states_verify.json` |
| expert columns kept (no pixels) + 50 eval episodes' frames | 172 MB, 17/17 checks pass | `expert_columns_summary.json` |
| render vs dataset frame, latent squared distance (median) | 0.38 (one env step: 1.37; start-goal: 332) | `render_check.json` |

Spread study, open-loop success [95% CI over pairs] (verifier: separate recomputation, 0 mismatches):

| kicks | p = 0 | p = 0.05 | p = 0.15 |
|---|---|---|---|
| small (2-10 px, +-5 deg) | 1.000 | 0.974 [0.961, 0.986] | 0.904 [0.878, 0.930] |
| medium (5-20 px, +-10 deg) | 1.000 | 0.854 [0.825, 0.884] | 0.604 [0.559, 0.644] |
| large (10-40 px, +-20 deg) | 1.000 | 0.621 [0.587, 0.656] | 0.289 [0.253, 0.325] |

Realized kicks per step at p = 0.05 / 0.15: fired 0.049 / 0.149, full-size 0.035 / 0.107 (medium).

Figures: `results/phase0/spread/spread_scatter.png`, `results/phase0/spread/spread_ecdf.png`

**Verdict:** infrastructure works on CPU; medium kicks give a graded open-loop effect (1.00 / 0.85 /
0.60) and are the provisional choice. The final kick size waits for the closed-loop check in the GPU
smoke job, decided by the confirmed rule (decisions.md, "Kick ranges").

**Caveats:** open loop only; one CPU type; GPU timings not measured yet (the SU estimates assume
2 s per plan call). Rendered start/goal frames stand in for dataset frames on our seed-0 pairs.

**Open items (before week B work starts):**
1. GPU smoke job results -> kick size by the rule; SU estimate for the spec section-7 grid from the
   get_cost timings; GPU vs CPU collect+encode for the 3 main-grid conditions (Rohan picks).
2. `~/private` quota (Rohan) -> whether the two `randomize_physics` conditions get data.
3. **Eviction test, before the first multi-hour GPU job:** `jobs/queue/evict-test-eval-pusht.sh`
   (queue this weekend; the week's SU expire Monday 00:00 Pacific). Kick check on our pairs 50-99 (extra
   paired data), own RUN_ID. Once >= 10 episodes are logged, Rohan deletes the pod; pass criteria and
   the resubmit step are in the job header. `kubectl delete pod` gives the default grace period
   (30 s unless the pod spec sets more), so the phase must stop within it; one GPU episode
   (<= 2 plan calls) should fit, and the test checks it. Every GPU job now writes PID 1's command
   line and its own pid / ppid / pgid / sid to env.txt (gpu-job template).
4. ~~Commit the job move / ledger line~~ done (`c0b0463`); from now on after every submission.

**Next:** week B: collect + encode the 3 main-grid conditions (chunked, resumable,
`snapshot_run.sh`), then train the deterministic, mixture and flow heads.

