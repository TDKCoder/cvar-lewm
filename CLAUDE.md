# CVaR-LeWM (ECE 285 course project)

Flow-matching latent dynamics + risk-sensitive (CVaR) CEM planning for LeWorldModel, evaluated under
*stochastic dynamics*. Full spec, math and schedule: `docs/project-spec.md` (read it before any
non-trivial change). Proposal source: `docs/proposal/proposal.tex`.

## Where you are running (laptop mode)

- **You run on Rohan's laptop, in WSL2.** This repo is the source of truth: read and edit files here
  with your normal tools. DSMLP pods can't reach the Anthropic API, which is why you're not on the cluster.
- **Code executes on UCSD's DSMLP cluster**, in a free CPU-only *worker* pod. Anything that needs the
  project's Python deps runs through:
  - `scripts/remote.sh "<cmd>"`: pushes this repo to the cluster mirror, then runs `<cmd>` in the
    worker from the mirror root with `scripts/env.sh` sourced. Exit code is the command's.
  - `scripts/remote.sh --bg <name> "<cmd>"` for anything over ~5 min; it logs to
    `runs/_bg/<name>.log`. Check with `scripts/remote.sh --no-sync "tail -n 30 runs/_bg/<name>.log"`
    every few minutes, not in a tight loop.
  - `scripts/worker.sh status | start`: the worker ends after 12 h. If `remote.sh` says it's
    missing, start it (free) and rerun.
  - `scripts/sync.sh pull "<run glob>"`: copies small result files (jsonl, logs, csv, figures; never
    checkpoints) from the cluster into `runs/` here. Aggregate and plot locally.
  - `scripts/sync.sh put <local> [dest]`: copies a file into the cluster store (`$STABLEWM_HOME`).
    Ask first if it's over 100 MB.
- **SSH.** All scripts reuse one connection Rohan opens himself. If a script says there's no open
  SSH connection, stop and ask him to run `ssh dsmlp true`. Never try to log in or handle passwords,
  keys or Duo.
- **Never run commands on the DSMLP login node** yourself, and never call `ssh`, `scp`, `rsync` or
  `kubectl` directly. The scripts above are the only door. Don't edit them; ask Rohan.
- **GPU jobs:** you prepare them (`gpu-job` skill); Rohan submits with `scripts/submit_job.sh`.
  You never run that script.
- **Local tools**: `.venv-analysis/` has numpy, pandas, matplotlib, jupyter for aggregation and
  notebooks. Project deps (torch, stable-worldmodel) are not installed locally; don't install them.
- Cluster facts (limits, GPU classes, Service Units, storage, pod network): `dsmlp` skill.
- **Never touch Nautilus/NRP.** Rohan's laptop may hold a Nautilus kubeconfig: never read
  `~/.kube` or any kubeconfig, and never run `kubectl` locally.

## Hard rules

1. **Never spend GPU budget yourself.** Write the job to `jobs/queue/` with an SU estimate, commit,
   and stop. Rohan decides.
2. **Smoke-test on CPU first.** Every GPU job script must have run end-to-end in the worker pod with
   a tiny config (`--smoke`) before it goes into `jobs/queue/`.
3. **Every long run must survive interruption**: wall-clock checkpoints every 15-20 min, atomic
   saves, auto-resume, SIGTERM handler, termination-warning polling. See the `gpu-job` skill.
4. **Numbers come from files, never from memory.** Every number you report must come from a script
   run on raw per-episode logs and be cross-checked by `report-results`' independent verifier.
   If the two disagree, say so; don't pick one.
5. **Don't edit upstream LeWM files** (`jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py`,
   `config/`) unless there's no other way. Prefer subclassing/new configs under `cvar/` and
   `config/cvar/`. If you must, mark the line `# CVAR-LEWM:` and log it in `docs/decisions.md`.
6. **Storage is small.** Course home dirs are ~5-10 GB; everything big goes in `~/private` on the
   cluster. Check `du -sh` (via remote.sh) before downloading anything over 1 GB and ask first.
   Never download `pusht_expert_train` (13.1 GB compressed) without explicit approval.
7. **Pod network.** The worker can't reach api.anthropic.com. What else it can reach is recorded in
   `docs/setup-report-pod.txt` (Phase 0). If the pod can't reach a download, get it on the laptop
   and `scripts/sync.sh put` it. Never try to tunnel or proxy around the firewall.
8. Ask before deleting anything outside `runs/smoke-*` and `/tmp`.
9. **CPU parallelism uses `CVL_NCPU`.** DataLoader `num_workers`, multiprocessing/joblib/concurrent
   pools and any thread count read `int(os.environ["CVL_NCPU"])`, never `os.cpu_count()` or
   `nproc`: those report every core on the node (128-256), not the pod's limit (4). `nproc` only
   looks right because env.sh sets `OMP_NUM_THREADS`, which GNU nproc honors.

## Environment on the cluster

`scripts/env.sh` (sourced by remote.sh and job scripts) sets `PROJECT_ROOT`, `STABLEWM_HOME`,
`RUNS_DIR` (always the mirror's `runs/`), `VENV_DIR` (`~/private/.venvs/cvar-lewm`, shared by all
code snapshots), caches under `~/private/.cache`, `WANDB_MODE=offline`, `PYTHONUNBUFFERED=1`,
`PYTHONPATH=$PROJECT_ROOT`, `CVL_NCPU` (the pod's cgroup CPU limit, rounded up) with
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` defaulting to it, and activates the venv if it exists.
Target Python 3.10 (LeWM).

## Layout

```
jepa.py module.py train.py eval.py utils.py config/   # upstream LeWM (avoid editing)
cvar/                 # our code
  envs/               # stochastic wrappers: impulses, friction/mass resampling, TwoRoom slip
  predictors/         # deterministic (baseline), K-head mixture (WTA), flow-matching head
  planning/           # CEM with mean / mean+beta*std / CVaR_alpha costs, common random numbers
  calibration/        # ground-truth rollouts, energy score, coverage, spread correlation
  data/               # data collection + latent pre-encoding
  utils/              # resilience.py (ckpt/SIGTERM/podinfo), logging, seeding
config/cvar/          # Hydra configs for our runs
scripts/              # env.sh + pod_audit.sh (run in pods); remote/sync/worker/submit_job (laptop)
jobs/queue/ jobs/submitted/ jobs/logs/   # GPU job scripts; logs pulled back from the cluster
runs/                 # run outputs (git-ignored): full on the cluster, small files pulled here
results/              # small committed summaries: CSV, JSON, figures
notebooks/            # analysis notebooks (Rohan prefers notebooks for analysis)
docs/                 # project-spec, EXPERIMENTS.md, decisions.md, gpu-ledger.md, codebase-map.md
tests/                # pytest (run on the cluster)
```

## Commands

- Tests (cluster): `scripts/remote.sh "python -m pytest -q tests/"`
- Smoke run of anything: add `--smoke` (tiny data, 2 episodes, CPU, < 5 min)
- Pull results: `scripts/sync.sh pull "<run glob>"`
- Aggregate (local): `.venv-analysis/bin/python .claude/skills/report-results/scripts/aggregate.py "runs/<glob>" --out results/<name>`
- Verify (local): `.venv-analysis/bin/python .claude/skills/report-results/scripts/verify_summary.py "runs/<glob>" results/<name>/summary.csv`

## Method in one screen (details in docs/project-spec.md)

- Frozen LeWM encoder `E_phi`; `z_t = E_phi(o_t)`. Pre-encode datasets to latents once and train
  heads on cached latents.
- Flow head: `z^tau = (1-tau) eps + tau z_{t+1}`, loss `|| v_theta(z^tau, tau; z_t, a_t) - (z_{t+1} - eps) ||^2`,
  `eps ~ N(0,I)`, `tau ~ U[0,1]`. Few-step Euler sampling; autoregressive rollouts give S futures.
- Candidate cost samples `c^(s)(a) = || z_H^(s) - z_g ||^2`. CVaR_alpha = mean of the
  `ceil(alpha*S)` largest. `alpha = 1` is risk-neutral with the same model. All candidates in one
  CEM iteration share the noise seeds `eps^(1:S)` (common random numbers).
- Baselines, matched parameter count: deterministic LeWM head; K-head mixture (winner-take-all).
- Stochastic PushT: per-step impulse with prob `p in {0, 0.05, 0.15}`, friction and mass resampled
  per episode. `p = 0` must reproduce the original env exactly for a fixed seed (unit-tested).

## How to work

- Plan before editing anything that spans more than one file; keep a short plan in the chat.
- Show intermediate steps and the equations behind any metric you compute (Rohan wants
  equation-level reasoning paired with intuition).
- Small, reviewable commits: message `area: what and why`. Commit before preparing a GPU job: the
  job snapshot records the commit.
- After each experiment: `report-results` skill -> entry in `docs/EXPERIMENTS.md`
  (hypothesis, exact command, git SHA, result with CI, verdict). Negative results get written up too.
- Record design choices and deviations from the proposal in `docs/decisions.md` (date, choice, why).
- When unsure about cluster behavior, say so and point to the `dsmlp` skill's sources rather than
  inventing a number.
