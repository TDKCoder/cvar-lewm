# CVaR-LeWM project spec

Source of truth for *what* we build and *how we judge it*. The proposal (`docs/proposal/proposal.tex`)
is the promise to the course; this file is the working version. Log deviations in `docs/decisions.md`.

## 1. Question

Does a stochastic latent predictor improve JEPA planning when the **dynamics** are stochastic, and
does it help more when the planner optimizes **tail** cost instead of mean cost?

Why the deterministic head fails: LeWM's predictor minimizes `E || f(z_t, a_t) - z_{t+1} ||^2`, whose
minimizer is `E[z_{t+1} | z_t, a_t]`. If a push sends the block left or right with similar
probability, the prediction lands between the two outcomes, where no real outcome is.

## 2. Environments

Built on `stable-worldmodel` (the platform LeWM uses). Implement as wrappers in `cvar/envs/`.

| Env | Stochasticity | Levels |
|---|---|---|
| PushT | per step, with prob `p`, a random pose kick to the T-block (random direction and length, small random rotation; applied before the step). Separately, flag `randomize_physics`: per episode, resample ground drag and block mass. Definitions in `docs/decisions.md` (2026-11-12) | `p in {0, 0.05, 0.15}`, `randomize_physics` off; secondary: on at `p in {0, 0.15}` |
| TwoRoom | a slip region near the door adds action noise (Gaussian, fixed std) inside the region | off / on |

Requirements:
- `p = 0` (and slip off, `randomize_physics` off) reproduces the original env **bit-for-bit** for a fixed seed. Unit test it.
- Kick draws `(u, theta, d, rho)` happen every step whatever p is, and fire iff `u < p`, so the
  kicked steps at p = 0.05 are a subset of those at p = 0.15 (paired comparisons). Unit test it.
- All randomness flows from one seeded `np.random.Generator` per env instance, so the same
  `(env_seed, actions)` gives the same trajectory. Unit test it.
- Ranges (impulse magnitude, friction, mass, slip std) live in a config file, not in code. Pick them
  in Phase 0 so that at `p = 0.15` an expert-ish open-loop plan visibly diverges, and record the
  choice with a figure in `docs/decisions.md`.

## 3. Models

All heads share the frozen LeWM encoder `E_phi` from the pretrained checkpoint
(`quentinll/lewm-pusht`, `quentinll/lewm-tworooms`). Pre-encode every dataset to latents once
(`cvar/data/encode.py`), then train heads on cached `(z_t, a_t, z_{t+1})` tuples.

| Head | Training objective | Sampling |
|---|---|---|
| Deterministic (baseline) | `|| f(z_t,a_t) - z_{t+1} ||^2` (same as LeWM, retrained on our data) | one future |
| K-head mixture (Branch-JEPA style) | winner-take-all: `min_k || f_k(z_t,a_t) - z_{t+1} ||^2` plus a small all-heads term to avoid dead heads; learn mixture weights by classifying the winning head | sample head by weight |
| Flow matching | `z^tau = (1-tau) eps + tau z_{t+1}`; `L = E || v(z^tau, tau; z_t, a_t) - (z_{t+1} - eps) ||^2`, `eps ~ N(0,I)`, `tau ~ U[0,1]` | Euler, `n_steps in {1, 2, 4, 8}` |

Parameter counts matched within 10% across heads (report them). Also train/evaluate the original
pretrained LeWM predictor as a reference row.

Rollout: autoregressive in latent space for horizon `H`; `S` samples per candidate.

## 4. Planner

CEM over action sequences `a_{1:H}`; keep the LeWM defaults for population, elites, iterations
unless Phase 0 timing forces a change (log it).

Costs per candidate `a`, from `S` sampled futures, `c^(s)(a) = || z_H^(s) - z_g ||^2`:
- mean: `(1/S) sum_s c^(s)`
- mean+std: `mean + beta * std`, `beta in {0.5, 1, 2}`
- CVaR_alpha: Rockafellar-Uryasev `min_eta eta + (1/alpha) E[(c - eta)_+]`; empirical estimator =
  mean of the `ceil(alpha * S)` largest samples. `alpha = 1` gives the mean.

Common random numbers: within one CEM iteration, every candidate uses the same noise
`eps^(1:S)` (and the same head indices for the mixture). Resample between iterations.
Edge case: with `S = 8, alpha = 0.1`, `ceil(0.8) = 1`, so CVaR is the max. Report that.

## 5. Metrics (definitions are binding)

Per evaluation cell = (env, p, head, cost, alpha/beta, S, seed). Each cell has N episodes
(target 100), each logged as one JSON line (schema in the `report-results` skill).

- **Success rate** = mean of the env's own success flag. 95% CI by percentile bootstrap over episodes
  (10,000 resamples, fixed seed), cross-checked with a Wilson interval.
- **Worst-10% final cost** = mean of the top `ceil(0.1 N)` final costs in the cell (tail metric).
  Bootstrap CI.
- **Failure rate** = 1 - success rate (reported separately because the hypotheses are about tails).
- **Plan time** = mean wall-clock seconds per replanning call (GPU, after warm-up).
- **Calibration** (from fixed states and action sequences; 32 true simulator rollouts vs S predicted):
  - Energy score in latent space: `ES = (1/S) sum_s ||X_s - y|| - (1/(2 S^2)) sum_{s,s'} ||X_s - X_s'||`,
    averaged over true outcomes `y`. Lower is better.
  - Coverage of the central 80% predicted cost interval by realized costs (target 0.8).
  - Spearman correlation between predicted cost std and realized cost std across (state, plan) pairs.

## 6. Hypotheses

- **H1**: stochastic heads match the deterministic head at `p = 0` and pull ahead as `p` grows.
- **H2**: for a fixed head, CVaR lowers worst-10% cost and failure rate at a small cost in mean success.
- **H3**: predicted spread tracks realized spread (coverage near nominal, positive Spearman).

Fallbacks: if the frozen encoder misses outcome differences, fine-tune it with SIGReg; if CVaR
doesn't beat the mean cost, H3 still answers whether JEPA predictors capture dynamics uncertainty.

## 7. Experiment grid

Main grid (PushT): 3 heads x {mean, mean+std(beta=1), CVaR(alpha=0.25)} x p in {0, 0.05, 0.15}
x 2 seeds x 100 episodes. Ablations on the flow head at p = 0.15: alpha in {0.1, 0.25, 0.5, 1},
S in {4, 8, 16}, Euler steps in {1, 2, 4, 8}. TwoRoom: 3 heads x {mean, CVaR} x slip on/off.
The main grid runs with `randomize_physics` off. Secondary condition (physics noise): `randomize_physics`
on at p in {0, 0.15}, flow and deterministic heads, {mean, CVaR(alpha=0.25)}.

## 8. Compute budget

Proposal estimate: about 18 GPU-h training + 40 GPU-h evaluation (estimate, not measured).
DSMLP default: 10 SU per member per week per workspace (confirm on the Dashboard). The ECE 285
workspace grants only the `medium` class: 1 SU per GPU-hour peak, 0.5 off-peak (1 AM-4 PM Pacific).
So 10 SU buys 20 GPU-h a week off-peak or 10 at peak. The 40-60 GPU-h estimate costs 20-30 SU
off-peak (2-3 weeks of budget) or 40-60 SU at peak (most of the six weeks' 60 SU). Submit off-peak.
Plan:
- The CPU worker pod (free) for data collection and smoke tests; the laptop for aggregation and plotting.
- GPU only for latent pre-encoding, head training if it doesn't fit CPU, and planning evals.
- Measure seconds per episode in the Phase 0 smoke job, then size every eval job from that number.
- Pack several eval cells into one job (one pod start, one CUDA warm-up) when each cell is short.
- Keep the running total in `docs/gpu-ledger.md`.

## 9. Schedule (re-based to Nov 9; proposal's week 1 was Oct 26)

Confirm the final deadline with the course. Assuming Dec 20:

| Week | Dates | Deliverable |
|---|---|---|
| A | Nov 9-15 | Phase 0: env audit, install, codebase map, CPU forward/plan on pretrained ckpt, stochastic PushT wrapper + tests, GPU smoke job queued |
| B | Nov 16-22 | Collect stochastic PushT data (CPU pods); pre-encode latents (GPU); train deterministic, mixture, flow heads |
| C | Nov 23-29 | Thanksgiving, light: CVaR / mean+std planner; first success-vs-p curves run unattended |
| D | Nov 30-Dec 6 | Full PushT grid, 2 seeds; ground-truth calibration study |
| E | Dec 7-13 | Ablations; TwoRoom slip variant if on schedule |
| F | Dec 14-20 | Final report, figures, code release |

Cut order if behind: TwoRoom, then the `randomize_physics` condition, then the beta sweep, then the S ablation. Never cut: p sweep with
CIs, CVaR vs mean on the flow head, calibration (H3).
