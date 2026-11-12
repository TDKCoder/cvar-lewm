# Decisions and deviations

Newest on top. Format: `## YYYY-MM-DD: <choice>` then context, options considered, why, and
whether it changes anything promised in the proposal.

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
