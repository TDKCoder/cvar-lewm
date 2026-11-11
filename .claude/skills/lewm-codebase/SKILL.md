---
name: lewm-codebase
description: Map and work inside the upstream LeWorldModel (le-wm) and stable-worldmodel code - where the encoder, predictor, cost model, CEM solver, env wrappers and configs live, and where CVaR-LeWM plugs in without editing upstream files. Use before changing anything that touches LeWM internals, and to (re)build docs/codebase-map.md.
---

# LeWM codebase map

## What we know from the README (lucas-maes/le-wm, MIT license)

- Python 3.10, `uv venv --python=3.10`, `uv pip install stable-worldmodel[train,env]` (unpinned).
- Root files: `jepa.py` (LeWM model), `module.py` (`ARPredictor`, `Embedder`, `MLP`, ...),
  `train.py`, `eval.py`, `utils.py`, `config/train/`, `config/eval/` (Hydra).
- Train: `python train.py data=pusht` (set `wandb.config.entity/project` in `config/train/lewm.yaml`;
  we run with `WANDB_MODE=offline`).
- Eval/plan: `python eval.py --config-name=pusht.yaml policy=pusht/lewm`. The policy path is relative
  to `$STABLEWM_HOME`, without the `_object.ckpt` suffix.
- Data: `.h5` files in `$STABLEWM_HOME`; datasets referenced by name without `.h5`
  (e.g. `pusht_expert_train`). HF dataset repos: `quentinll/lewm-pusht` (13.1 GB .zst),
  `quentinll/lewm-tworooms` (3.43 GB), `-cube`, `-reacher`.
- Checkpoints: HF model repos `quentinll/lewm-{pusht,tworooms,cube,reacher}` ship `weights.pt` +
  `config.json`; the README has a conversion snippet that rebuilds `JEPA` and saves
  `<env>/lewm_object.ckpt`. `swm.policy.AutoCostModel('pusht/lewm')` loads an object checkpoint.

Everything below this line must be filled from the actual code. Don't assume.

Where to read it: the le-wm files are in this repo on the laptop. `stable-worldmodel` is only
installed on the cluster, so read its source there, e.g.
`scripts/remote.sh --no-sync "python -c 'import stable_worldmodel as s; print(s.__file__)'"` and then
`scripts/remote.sh --no-sync "sed -n '1,200p' <path>"` or `grep -rn` over that directory.

## Mapping procedure (output: docs/codebase-map.md)

Use `grep -rn` / reading, not guessing. For each item record file:line, the signature, tensor
shapes with names (B, T, D, action dim), and a 1-2 line note.

1. **Model**: the JEPA class in `jepa.py`; encoder forward; predictor forward (`ARPredictor`):
   inputs (history length, action conditioning), output, loss. Where SIGReg is applied.
   Latent dim D and what `z_t` is exactly (CLS token? pooled patches?).
2. **Cost model / planning**: how `AutoCostModel` computes cost from a goal; the CEM (or other)
   solver in stable-worldmodel: population, elites, iterations, horizon, action repeat/frameskip,
   replanning frequency, and where the predictor is called inside the solver loop
   (batched over candidates?). This is where `cvar/planning/` hooks in.
3. **Eval loop**: what `eval.py` does per episode; how start and goal states are chosen; **whether it
   reads the training `.h5` dataset** (decides if we need the 13 GB download); how success is
   computed; what gets logged.
4. **Environments**: how stable-worldmodel builds PushT and TwoRoom (gymnasium id? pymunk params?);
   where friction/mass live; the env step function where an impulse can be injected; seeding.
5. **Data**: h5 schema (keys, shapes, dtypes), how the dataloader slices `(o_t, a_t, o_{t+1})`
   windows, frameskip.
6. **Checkpoint formats**: `_object.ckpt` vs `_weight.ckpt` vs HF `weights.pt`; how to load just
   the encoder; parameter count of the predictor (our heads must match within 10%).

## Integration plan (fill in after mapping)

| Need | Preferred hook (no upstream edits) | Fallback |
|---|---|---|
| stochastic env | gymnasium wrapper or subclass in `cvar/envs/` registered under a new id | monkeypatch the step fn in our entry point |
| new predictor heads | new modules in `cvar/predictors/` that take `(z_t, a_t)` and return S samples | subclass `ARPredictor` |
| CVaR cost | wrap the cost model so it returns S costs per candidate, reduce in our solver | subclass the solver |
| eval with our heads | `cvar/eval.py` that reuses upstream eval utilities | copy-and-modify `eval.py` into `cvar/` |

## Sanity checks to keep passing (tests/)

- Loading the pretrained PushT checkpoint and encoding one frame works on CPU.
- Our deterministic head, when initialized from the pretrained predictor, reproduces upstream
  predictions to 1e-5 on a fixed batch.
- `cvar` eval with the pretrained predictor and `p = 0` matches upstream `eval.py` success on the
  same seeds (small n, CPU smoke).
