# Decisions and deviations

Newest on top. Format: `## YYYY-MM-DD: <choice>` then context, options considered, why, and
whether it changes anything promised in the proposal.

## 2026-11-13: Training data for the heads: noisy-expert replay + weak policy, CPU collection and encoding (recommended; awaiting OK)

- Needs: per condition (main grid p in {0, 0.05, 0.15}; secondary `randomize_physics` at p in {0, 0.15}),
  trajectories in the stochastic env whose states look like eval states (eval start/goal pairs are
  expert states), and whose actions cover what CEM proposes (300 Gaussian candidates around the mean,
  so well off the expert action). LeWM trains on windows of history 3 + 1 prediction at frameskip 5
  starting at any row (`config/train/data/pusht.yaml`), so we store **one latent per env step**.
- Measured (`results/phase0/data_probe.json`, CPU worker, 1 process, 4 threads): env 3,373 steps/s
  without rendering, **416 steps/s with the 224 px render**; frozen encoder on CPU **41.7 frames/s**
  (batch 64). Bytes per frame: raw 224 px 150,528; **224 px zlib-compressed in stacks 2,138**; 64 px
  zlib 407; PNG / JPEG-90 at 224 px 6,207 / 4,937; a latent (192 x fp16) 384. PushT frames are mostly
  flat colour.
- Policy options:

  | Option | Matches eval states | Covers CEM's off-expert actions | Cost |
  |---|---|---|---|
  | A. expert replay: expert start state + its action sequence, open loop, kicks on | yes | no | needs expert actions (below) |
  | B. A + Gaussian action noise | near | partly | same |
  | C. WeakPolicy-style random actions near the block | no (contact-heavy) | yes | none |
  | D. CEM with pretrained LeWM, closed loop | yes (on-policy for planners) | yes | ~24 s per plan call on CPU (`pretrained_cpu.json`); thousands of episodes = days of CPU or tens of SU |

- **Recommended: B + C at 3:1**, i.e. 3,000 noisy-expert replays (half with action noise 0, half with
  sigma = 0.2 of the normalized action std) + 1,000 weak-policy episodes per condition, **4,000
  episodes ≈ 492,000 env steps** (median expert length 123). The same start states and env seeds in
  every condition, so the datasets are paired across p (coupled kick draws). Exclude every expert
  episode that holds one of our 1,000 eval pairs or upstream's 50, so training never sees an eval
  episode. Log `info["cvl_kick"]` and `cvl_physics` per step. D is rejected on cost.
- Expert actions: A/B need the expert action sequences and start states, which the eval npz doesn't
  hold. Getting them takes **one more transient download** into the worker's `/tmp` (13.1 GB, ~8 min,
  same script pattern), keeping every non-pixel column (action, state, proprio, episode/step index:
  about 0.14 GB raw) in `$STABLEWM_HOME`. **Needs Rohan's OK.**
- Where: **collect and encode in the free CPU worker**, chunked by 100 episodes (atomic, resumable),
  run through `snapshot_run.sh`. Full frames live only in the worker's `/tmp` until encoded.
  Collection ≈ 0.33 h per condition in one process. CPU encoding ≈ 3.3 h per condition (9.8 h for the
  3 main-grid conditions, 16.4 h for all 5), within the week-B schedule and 0 SU. Switch to a GPU
  encode job only if the block-8 smoke job shows it costs well under 0.1 SU and the schedule is tight.
  The smoke job will time GPU encoding too.
- What we keep per condition (≈ 0.44 GB on the quota): latents for every step, fp16 (0.19 GB); **224 px
  frames, zlib, for a 20% episode subset** (0.21 GB), for the SIGReg encoder fine-tune fallback; we
  keep full resolution instead of low-res because the encoder takes 224 px input and compressed 224 px
  frames cost about as much as raw 64 px ones; states/actions/kick logs (0.04 GB). Main grid ≈ 1.3 GB,
  all 5 conditions ≈ 2.2 GB. **Depends on the `~/private` quota (unconfirmed).**
- Normalization: the pretrained encoder and action encoder use the expert scaler stats (from the npz),
  so every head uses those too. No stats from our data.
- Size knob: if a head's validation loss is still falling with data, double to 8,000 episodes per
  condition. Cost is linear (CPU encode +3.3 h, disk +0.44 GB per condition).
- Effect on proposal: none in substance. The proposal didn't specify the data policy.

## 2026-11-13: Host-specific details stay out of the public repo

- Context: pushing Phase 0 published the cluster username (already in `SETUP.md` /
  `dsmlp.conf.example`), UID, home and NFS export paths and node hostnames in reports and results.
- Choice (Rohan): no force-push. A new commit replaces them with `<user>`, `<uid>`, `<home>`,
  `<node>` (`scripts/redact.py`), so they remain in earlier commits' history. From now on raw audits
  and env dumps go only to git-ignored `docs/local/` or `runs/`. Committed copies are made with
  `redact.py --copy`, and `redact.py --check` runs before every commit (CLAUDE.md rule 10). The
  literal values live in git-ignored `docs/local/redact.txt`, so the checker itself publishes nothing.
  The unredacted pod audit is kept locally as `docs/local/setup-report-pod.raw.txt`.
- Effect on proposal: none.

## 2026-11-13: Realized kick rate is a reported metric, per cell, next to nominal p

- Context: kicks that would push the block into the agent or a wall are scaled back or vetoed
  (2026-11-12 entry), so the noise a policy actually faces depends on how much it stays in
  contact. If one head pushes more, it may face fewer full-size kicks than another at the same p.
- Choice (Rohan): every PushT episode logs `n_steps`, `kicks_fired`, `kicks_full`
  (`episodes.jsonl` contract in the `report-results` skill). Every cell reports
  `fired_rate = sum(kicks_fired)/sum(n_steps)` (should match p) and
  `full_rate = sum(kicks_full)/sum(n_steps)` with an episode-bootstrap CI, next to p.
  `aggregate.py` checks that the heads' `full_rate` CIs overlap within each cell group
  (`kick_check.md`); a mismatch is reported as a confound, not hidden. `verify_summary.py`
  recomputes both rates independently and warns if `fired_rate` is more than 3 binomial SEs from p.
  Spec section 5 updated. The spread study reports the same rates.
- Effect on proposal: none (an added diagnostic).

## 2026-11-13: Pretrained checkpoint: retargeted HF config; transformers < 5

- `quentinll/lewm-pusht` (`config.json` 1,313 B; `weights.pt` 72,290,721 B, sha256 `48938400…5b607`)
  is in `$STABLEWM_HOME/checkpoints/pusht/lewm/`. `cvar/utils/checkpoint.py` rewrites the config's
  Hydra targets (`stable_worldmodel.wm.lewm.*`, missing in swm 0.1.1) to `jepa.JEPA`,
  `module.{ARPredictor,Embedder,MLP}` and keeps the original as `config.hf.json`. Then the stock
  `swm.wm.utils.load_pretrained("pusht/lewm")`, which is what `eval.py policy=pusht/lewm` calls,
  loads it with a strict state-dict match. No upstream file edited.
- **transformers pinned < 5** (`requirements.in`): the weights use transformers-4 ViT parameter
  names. transformers 5.20.2 (first resolution) renamed them, so the strict load failed on all 198
  encoder keys. **datasets >= 3**: with transformers < 5 the resolver fell back to datasets 2.14,
  which breaks `import stable_pretraining.backbone` with current pyarrow. Both are now caught by the
  venv build's import check.
- CPU numbers (`results/phase0/pretrained_cpu.json`, 4 threads, torch 2.11): one plan call with the
  eval config (CEM 300 samples x 30 iterations, top 30, horizon 5) takes a median of 23.9 s (0.80 s per
  `get_cost`). The predictor has 10,791,360 parameters; the predictor path (with action encoder and
  pred_proj) has 11,740,334, which is the budget our heads match.
- Effect on proposal: none.

## 2026-11-13: Ephemeral project venv in each pod's /tmp, built from a lock file

- Context: `~/private` is near its quota (2026-11-12). A CUDA venv is 8.3 GB (mostly the nvidia
  wheels).
- Choice (Rohan): no persistent venv. `scripts/ensure_venv.sh` builds `/tmp/cvl-venv` from
  `requirements.lock.txt` when the pod lacks it, and is a no-op otherwise. uv, its cache and Python
  3.10 live in `/tmp` too. `scripts/withvenv.sh` wraps remote commands; GPU job scripts call
  `ensure_venv.sh` first and log its duration to `env.txt`. A build stamp (lock hash) marks a
  finished venv, and `env.sh` only activates a stamped one. Concurrent callers serialize on a flock.
- Measured (`runs/_bg/venv-builds.tsv`, CPU worker <node>): **cold build from the lock
  (no uv, cache or Python in /tmp, as in a fresh GPU pod) 143 s and 146 s** (packages 118-124 s,
  import check 16-25 s). Warm-cache rebuild 21-41 s; up-to-date check 0.05 s. Disk in /tmp: venv
  8.3 GB + uv cache 8.3 GB + Python 87 MB. Per-job overhead at 146 s = 0.041 GPU-h, i.e. 0.041 SU
  peak / 0.020 SU off-peak on `medium` (spec section 8 rates). GPU nodes may download faster or
  slower; every job logs its own build time in `env.txt`.
- Also learned: `remote.sh` re-syncs the NFS mirror, which replaced `ensure_venv.sh` under a running
  bash ("Stale file handle", build killed). Long-running scripts (`ensure_venv.sh`,
  `fetch_expert_stats.sh`) now re-exec from a private copy in `/tmp`.
- Effect on proposal: none. Adds a fixed per-job overhead to every GPU job's SU cost.

## 2026-11-12: PushT stochasticity = pose kicks; "friction" = ground drag; "mass" = mass at fixed inertia

- Context: what the PushT physics actually does (swm 0.1.1 `envs/pusht/env.py`, measured with
  `cvar/envs/physics_probe.py`, output in `results/phase0/physics_probe.txt`):
  - `space.damping = 0`. pymunk integrates position and then scales velocity by `damping^dt`, so
    a dynamic body keeps velocity for one 0.01 s substep. A 200 px/s velocity kick moves the block
    1.348 px, `apply_impulse(200)` 3.350 px. A velocity impulse is not a usable perturbation.
  - A temporary `velocity_func` held at 200 px/s for 10 substeps moved it 4.785 px instead of
    20 px, so "force during the substeps" isn't clean either.
  - `body.friction = 1` is set on Bodies (a stray Python attribute). All Shape frictions are 0.
    There is **no pivot/gear constraint**, so no `max_force`. Ground friction isn't modelled; it
    is effectively infinite viscous drag via `damping = 0`.
  - The agent is a KINEMATIC body. Block mass x0.25 / x4 with inertia fixed changes the 60-step
    final block position by 84.38 / 174.28 px. **Mass and inertia scaled together: 0.00 px**, an
    exact no-op.
  - Knobs that matter (same probe, final-pose gap vs the unperturbed run): contact friction
    0.3 / 1.0 gives 121.88 / 53.22 px; `damping` 1e-6 / 1e-3 / 0.1 gives 153.48 / 286.35 /
    68.05 px; centre of gravity (10, 0) gives 173.41 px. Scale: two different action seeds end
    142.68-439.02 px apart.
- **Choice: perturbation (`cvar/envs/stochastic_pusht.py`).** Every env step draws, always and in
  this order, `u_t ~ U[0,1)`, `theta_t ~ U[0, 2pi)`, `d_t ~ U[d_lo, d_hi]` px and
  `rho_t ~ U[-rho_max, rho_max]`. The kick fires iff `u_t < p` and moves the T-block **before**
  the inner step:
  `x <- x + s d_t (cos theta_t, sin theta_t)`, `phi <- phi + s rho_t`, where
  `s = max{1, 1/2, 1/4, 1/8 : pen(after) <= max(pen(before), 1 px)}`, else `s = 0`.
  `pen` = deepest overlap of the block with the agent or a wall (pymunk 7.3 reports penetration
  as a positive contact distance). Draws are independent of p, so fired(p=0.05) ⊆ fired(p=0.15)
  with identical `(theta, d, rho)`, and cross-p comparisons are paired (unit-tested).
  Provisional ranges (`config/cvar/env/pusht_stoch.yaml`): `d in [5, 20]` px, `rho_max = 10 deg`.
  They are picked by the spread study.
- **Measured side effect:** with p = 1 and near-block random actions (10 episodes x 200 steps),
  kicks applied at s = 1 / 1/2 / 1/4 / 1/8 / vetoed: 0.561 / 0.0835 / 0.0585 / 0.052 / 0.245.
  So in contact-heavy play the full-size kick rate is about 0.56 p. Logged per step in
  `info["cvl_kick"] = [fired, u, theta, d, rho, s]`. The spread study reports the applied rate
  for expert-like plans.
- **Choice: physics knobs** (per episode, drawn always, applied only if `randomize_physics`):
  - "friction" = **ground drag**, `dv/dt = -c v` through `space.damping = exp(-c)`. Velocity kept
    per substep is `exp(-c dt)`; upstream is `c = inf`. Provisional `c ~ U[10, 50]` 1/s
    (time constant 20-100 ms). It acts on the block only; the kinematic agent ignores damping.
  - "mass" = block mass `m ~ U[0.5, 2]` with the moment of inertia held at 3000. This changes the
    radius of gyration `k = sqrt(I/m)`, which is the only mass change that does anything here.
  - agent-block contact friction `mu` (block shapes `mu`, agent 1.0; pymunk multiplies them) is
    available but off (`[0, 0]`).
- Rejected: velocity impulses and forces during substeps (shown above); a pivot/gear
  ground-friction constraint. That would replace upstream's quasi-static contact model even at
  p = 0, so it can't be bit-for-bit.
- Effect on proposal: "impulse" now means a pose kick, and "friction/mass" are concrete pymunk
  parameters. The spirit (random external pushes + per-episode physics) is unchanged.

## 2026-11-12: Physics resampling is a separate flag, OFF for the main p-sweep

- Choice (Rohan): `randomize_physics` is independent of `p` and **off for the whole main sweep**
  `p in {0, 0.05, 0.15}`, so p is the only knob and p = 0 is exactly upstream.
- Why: impulses are i.i.d. per-step noise, which the flow head models (fresh eps every rollout
  step). Friction/mass is a per-episode hidden parameter whose errors are correlated across steps,
  which independent per-step sampling under-represents. Mixing them would confound the
  p = 0 -> 0.05 step.
- Secondary condition: `randomize_physics` ON at p = 0 (physics noise alone) and p = 0.15 (both),
  flow and deterministic heads only, CVaR vs mean cost. It's on the cut list right after TwoRoom.
- Effect on proposal: **deviation**. The proposal resampled friction and mass at every p.
  Spec sections 2, 7 and 9 updated.

## 2026-11-12: Eval start/goal states and normalization stats from a transient expert copy

- Context: `eval.py` needs `pusht_expert_train` for StandardScaler stats and start/goal states.
  The HF model repo has no stats. `~/private` is at its quota (~25 GB by `du`; unconfirmed).
- Choice: download `pusht_expert_train.h5.zst` (13,136,247,974 B; 46,300,921,856 B decompressed)
  into the worker's node-local `/tmp`, extract into `$STABLEWM_HOME/pusht/expert_eval_states.npz`
  (292,753 B), and delete the temp files (`scripts/fetch_expert_stats.sh`,
  `cvar/data/extract_eval_states.py`). The npz holds the scaler stats (n = 2,336,736 rows),
  upstream's exact 50 pairs (seed 42), 1,000 pairs for our evals (seed 0), and 8 init/goal frames.
- Verified: goal rows = init + 25 in the same episode (`step_idx`, `episode_idx` stored). The
  sampler matched `eval.py`'s literal loop on synthetic data. The scaler formula matched sklearn
  (max abs diff 0.0) on synthetic data. **Not** cross-checked by sklearn on the real columns
  (numpy-2 shadowing in that run).
- **2026-11-13 re-check on the real file** (second transient download, `fetch_expert_stats.sh --verify`
  -> `cvar/data/verify_eval_states.py`, which shares no code with the extractor and runs eval.py's
  code verbatim; `results/phase0/expert_eval_states_verify.json`): sklearn StandardScaler on the real
  columns matches the stored stats exactly (max abs diff 0.0 for mean, scale, var; n = 2,336,736);
  eval.py's literal per-episode length loop + sampling reproduces both sets' global rows, episodes
  and starts (50 at seed 42, 1,000 at seed 0); episode lengths match the stored `ep_len`; every
  stored init/goal column and the 8 frames equal the h5 rows. 20/20 checks pass. Each pair's global
  row index is stored (`<set>_row_init`, `<set>_row_goal`), so later checks don't need the dataset.
- Effect on proposal: none.

## 2026-11-11: Repo is a public fork of upstream LeWM

- Choice: the project repo is `TDKCoder/cvar-lewm` on GitHub (public), a fork of
  `lucas-maes/le-wm`. Remotes: `origin` = git@github.com:TDKCoder/cvar-lewm.git,
  `upstream` = https://github.com/lucas-maes/le-wm.git.
- Base: at setup, `main` == `upstream/main` == `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
  Our work is layered on top (agent kit, `cvar/` package) without editing upstream files.
- Public means anything committed is visible: no credentials, cluster usernames beyond what's
  already in the kit, or large data in git.
- Laptop checkout lives at `~/projects/ECE285/Project/cvar-lewm` (SETUP.md's commands updated
  2026-11-12; the scripts are path-independent).
- Effect on proposal: none.

## 2026-11-10: Claude Code runs on the laptop; commands run in a CPU worker pod

- Context: `curl https://api.anthropic.com` from a DSMLP pod returns 000 (blocked), so Claude
  Code can't run inside a pod.
- Choice: Claude runs in WSL2 on Rohan's laptop. The laptop repo is the source of truth;
  `scripts/sync.sh` mirrors it to the cluster and `scripts/remote.sh` runs commands in a free
  CPU-only worker pod via one shared SSH connection (ControlMaster). GPU jobs are still prepared by
  Claude and submitted by Rohan (`scripts/submit_job.sh`), now from the laptop, from a per-job code
  snapshot on the cluster.
- Rejected: tunnelling the pod's API traffic through the laptop (routes around a campus firewall).
- Verified with a probe pod: login `~` is the pod's `~/private`, so the mirror at login
  `~/cvar-lewm` is `~/private/cvar-lewm` in pods. kubectl needs a login shell over SSH; the scripts
  wrap remote commands in `bash -lc`.
- Effect on proposal: none.

## 2026-11-09: GPU jobs are submitted by hand

- Context: DSMLP charges Service Units for every GPU launch and the default budget is about 10 SU
  per week. CPU-only pods are free and not idle-culled.
- Choice: Claude only prepares GPU job scripts; Rohan submits them after seeing the SU estimate.
- Effect on proposal: none.

## 2026-11-09: Schedule re-based to start Nov 9

- The proposal's 8-week plan started Oct 26. Weeks are compressed to six (A-F) in
  `docs/project-spec.md`; cut order is listed there.
