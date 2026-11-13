# Codebase map: LeWM (this repo) + stable-worldmodel 0.1.1

Built 2026-11-12 from the code (lewm-codebase skill procedure). LeWM files are the repo root at
upstream commit `8edfeb3` (2026-05-22). stable-worldmodel ("swm") paths are relative to the
installed package (`stable_worldmodel/`), read from the `stable_worldmodel-0.1.1` wheel. Items
marked **[verify]** needed the venv; all were checked in blocks 4 and 6 (2026-11-13).

Shapes: `B` envs being planned, `N` CEM candidates (`num_samples`), `S` our noise samples,
`T` time, `D = 192` latent dim, `A = 2` env action dim, `action_block = 5`, so `A_blk = 10`.

## 0. Versions and drift

- le-wm predates swm 0.1.0 (2026-05-26) / 0.1.1 (2026-06-06); its README pins nothing.
  swm 0.1.1 still has every API `eval.py` calls: `World.evaluate(dataset=..., start_steps=...,
  goal_offset=..., eval_budget=..., callables=..., video=...)`, `swm.wm.utils.load_pretrained`,
  `swm.policy.WorldModelPolicy`, `swm.PlanConfig`, `swm.data.HDF5Dataset`.
- `HDF5Dataset` lives in `data/formats/hdf5.py` and is only exported if `h5py` + `hdf5plugin`
  import, which needs the `[format]` extra (the README's `[train,env]` lacks it).
- The HF model repo `quentinll/lewm-pusht` has `config.json` (1,313 B) and `weights.pt`
  (72,290,721 B) only. **No normalization stats.** Its config targets
  `stable_worldmodel.wm.lewm.LeWM` / `...wm.lewm.module.{Predictor,Embedder,MLP}`, **which do not
  exist in swm 0.1.1** (`wm/` = `loss.py, probes.py, utils.py`). So `load_pretrained("quentinll/lewm-pusht")`
  fails, and the README conversion snippet fails too: it splats `**cfg["predictor"]`, including
  `_target_`, into `ARPredictor`. Fix: rewrite the targets to `jepa.JEPA`, `module.ARPredictor`,
  `module.Embedder`, `module.MLP` (block 4). **Done:** `cvar/utils/checkpoint.py` rewrites them in place
  (original kept as `config.hf.json`), then `load_pretrained("pusht/lewm")` loads strictly. Needs
  **transformers < 5**: the weights use transformers-4 ViT names (`encoder.encoder.layer.N.attention.attention.query`),
  which transformers 5 renamed (`encoder.layers.N.attention.q_proj`).

## 1. Model (`jepa.py`, `module.py`)

| What | Where | Notes |
|---|---|---|
| `JEPA(encoder, predictor, action_encoder, projector, pred_proj)` | `jepa.py:11` | |
| `encode(info)` | `jepa.py:29` | `pixels (B,T,3,224,224)` -> ViT-tiny (patch 14) -> **CLS token** `last_hidden_state[:,0]` (`:38`) -> `projector` MLP (192->2048->192, BatchNorm1d) -> `info["emb"] (B,T,192)`. Also `act_emb = action_encoder(action)`. |
| `predict(emb, act_emb)` | `jepa.py:47` | `ARPredictor(emb, act_emb)` -> `pred_proj` MLP -> `(B,T,192)`; causal, so the output at t predicts `z_{t+1}`. |
| `ARPredictor` | `module.py:244` | learned pos-emb (`num_frames=3`) + 6 `ConditionalBlock`s (dim 192, 16 heads x 64, mlp 2048, dropout 0.1), **AdaLN-zero conditioned on the action embedding** (`module.py:88`). Causal attention (`module.py:75`). |
| `Embedder` (action encoder) | `module.py:189` | Conv1d(k=1) 10->10, then MLP 10->40->192. Input is the 10-dim action block (5 env steps x 2). |
| `SIGReg` | `module.py:10` | Epps-Pulley test on 1024 random projections; training only (`config/train/lewm.yaml`: weight 0.09). |
| latent | | **`D = 192`** (`embed_dim`), **history 3** (`history_size`); `z_t` = projected CLS token. |
| parameter counts | `tests/test_pretrained_cpu.py` | encoder 5,501,376; **predictor 10,791,360**; action_encoder 156,206; projector 792,768; pred_proj 792,768; total 18,034,478 (`results/phase0/pretrained_cpu.json`). Predictor path (predictor + action_encoder + pred_proj) = 11,740,334: the budget our heads match within 10%. |

## 2. Cost model and planning (where CVaR plugs in)

| What | Where | Notes |
|---|---|---|
| `JEPA.get_cost(info, candidates)` | `jepa.py:128` | Encodes the goal pixels (`info["goal"]`) -> `goal_emb`; `rollout`; `criterion`. Returns `(B, N)`. |
| `JEPA.rollout(info, actions, history_size=3)` | `jepa.py:61` | `actions (B,N,T,10)`; the first `H = info["pixels"].size(2)` blocks are the context. The start frame is encoded **once per env** (`[:,0]`) then expanded over N. Autoregressive loop (`:89`) over `T - H` steps on `(B·N, ≤3, 192)` + one final predict -> `predicted_emb (B,N,T+1,192)`. **This is the predictor call site.** |
| `JEPA.criterion` | `jepa.py:112` | `c = Σ_d (ẑ_last - z_goal)²` (sum of squared error over D, last step only). Same `c^(s)` as the spec (Euclidean²). |
| `CEMSolver` | `solver/cem.py:17` | `__init__(model, batch_size=1, num_samples=300, var_scale=1, n_steps=30, topk=30, device='cpu', seed=1234)`. Own `torch.Generator(device).manual_seed(seed)` (`:53`). |
| CEM loop | `solver/cem.py:189-263` | per iteration: `candidates = randn(B,N,H,A_blk)·σ + μ`, `candidates[:,0] = μ`; **`costs = self.model.get_cost(expanded_infos, candidates)` once per iteration, all N candidates batched (`:210`)**; must be `(B,N)` (asserted); top-k (lowest) -> new `μ = mean`, `σ = std` of elites (the variable named `var` is a std). Envs are planned in chunks of `batch_size` (=1 in config) **sequentially** (`:152`). |
| solve output | `solver/cem.py:272` | `actions = μ (B,H,A_blk)`; prints `CEM solve time` each call. |
| `WorldModelPolicy.get_action` | `policy.py:349` | `_prepare_info` (`:86`: `process[k].transform` = StandardScaler on numpy keys, `transform[k]` = ImageNet-normalize+resize on pixels/goal); replans only when an env's action buffer is empty (receding horizon 5 blocks = 25 env steps); warm-starts from the unused tail; `inverse_transform` on the action (`:436`). |
| `PlanConfig` | `policy.py:17` | `horizon=5, receding_horizon=5, action_block=5`, `history_len=1` (default; `eval.py` doesn't set it), `warm_start=True`. With H=1 context, rollout does 4 loop steps + 1 final = 5 predictions -> the cost is at **25 env steps**, matching `goal_offset_steps=25`. |
| config | `config/eval/pusht.yaml`, `config/eval/solver/cem.yaml` | 300 samples, 30 iterations, top-30, `device: "cuda"`; `eval.py:89` also hardcodes `.to("cuda")`. |

**Plan-call cost:** one replanning call = 30 `get_cost` calls x (1 goal encode + 1 start encode + 5
predictor passes on 300 sequences). With `S` samples the predictor batch becomes `300·S`.

## 3. Eval loop (`eval.py` + `world/world.py`)

1. `swm.World("swm/PushT-v1", num_envs=num_eval=50, image_shape=(224,224), max_episode_steps=100)` (`eval.py:57`).
2. Loads **`pusht_expert_train.h5`** (`get_dataset`) for two things:
   - **normalization**: a `StandardScaler` per key in `dataset.keys_to_cache = [action, proprio, state]`, fitted on the whole raw column minus NaN rows (`eval.py:75`; `get_col_data == f[col][:]`, `hdf5.py:134`);
   - **start/goal states**: valid starts are rows with `step_idx <= ep_len - 25 - 1`; `default_rng(seed=42).choice(len(valid)-1, 50, replace=False)`, sorted (`eval.py:124`).
3. `world.evaluate(dataset=..., episodes_idx, start_steps, goal_offset=25, eval_budget=50, callables=[_set_state(state), _set_goal_state(goal_state)])` -> `_evaluate_from_dataset` (`world.py:492`):
   `load_chunk` rows `[start, start+25]` (`dataset.py:74`; episode id used as a *position* in `ep_offset`) -> init = first row, goal = last row (`world.py:567`); `reset(seed=init_state.get('seed'))`; callables set the physical start state and the goal state on the unwrapped env; **the goal image is the dataset's frame**, re-injected every step (`world.py:542`).
4. All 50 envs step together for at most 50 env steps (`mode='wait'`); **success = env `terminated` at any step** (`world.py:543`), OR-accumulated.
5. Logs `metrics` (success_rate %, per-episode bools) and wall time to `<ckpt dir>/pusht_results.txt`. **No per-episode cost or plan time**, so our eval must log those itself.

**Does `eval.py` need the 13 GB file? Yes, as written** (stats + start/goal + goal frames).
Block 5 replaces it with `$STABLEWM_HOME/pusht/expert_eval_states.npz` (stats + 50 upstream
pairs + 1,000 of ours), extracted from a transient copy.

## 4. PushT environment (`envs/pusht/env.py`, id `swm/PushT-v1`, `envs/__init__.py`)

| What | Where | Notes |
|---|---|---|
| registration | `envs/__init__.py` | lazy entry-point strings; only pygame + pymunk + shapely + opencv are needed for PushT/TwoRoom. |
| `__init__` | `env.py:29` | `relative=True`, `action_scale=100`, `k_p,k_v = 100,20`, `dt = 0.01`, `control_hz = 10` -> **10 physics substeps per env step**. Optional `damping`, `block_cog` kwargs. |
| `reset(seed, options)` | `env.py:237` | `self.rng = default_rng(seed)`; `reset_variation_space(seed, options)` (colors/shapes/start pose); `_setup()` builds a fresh `pymunk.Space`; samples a **random goal pose** (unless `options['goal_state']`) and renders it as the goal image; start from the variation space (or `options['state']`). |
| `step(action)` | `env.py:312` | target = agent pos + 100·action; 10 x (PD acceleration on the agent; `space.step(0.01)`); `terminated, dist = eval_state(goal, state)`; `reward = -dist`; never truncates itself. |
| success | `env.py:347` | `‖goal[:4] - state[:4]‖ < 20` (agent xy + block xy, pixels) **and** block angle error `< π/9`. |
| state | `env.py:360` | `(agent x, y, block x, y, block θ mod 2π, agent vx, vy)`, float64 (7); `proprio = (agent x, y, vx, vy)`. |
| `_set_state` | `env.py:514` | sets agent pos/vel, block pos/angle, then one `space.step(dt)`. |
| physics | `env.py:532` | gravity 0, **`space.damping = 0`**: pymunk keeps a fraction `damping^dt` of velocity each step, so dynamic bodies lose all velocity every substep. The block is quasi-static and only moves through contact resolution with the agent. Verified: a 200 px/s velocity kick moves it 1.35 px (`results/phase0/physics_probe.txt`). |
| friction / mass | `env.py:626` (`add_tee`) | `mass = 1`; inertia from `moment_for_poly`. **`body.friction = 1` is set on the Body**, but pymunk friction is a Shape attribute, so contact friction is the Shape default. Verified: all Shape frictions are 0; no pivot/gear constraint (probe). |
| agent body | `add_circle` (`env.py:598`) | **KINEMATIC** (verified). Block mass alone matters (x4: 174 px final-pose change); mass and inertia scaled together is an exact no-op (probe). |
| expert / collection policy | `envs/pusht/expert_policy.py` | `WeakPolicy`: uniform random relative actions clipped to a ±100 px box around the block. A random-near-block policy, not an expert. |

**Consequences for the stochastic wrapper (block 6):**
- A velocity impulse via `apply_impulse_*` would be erased by `damping = 0` in the next substep.
  Options: a temporary `velocity_func` on the block for the impulse substeps (moves it physically,
  collisions resolved), or a direct pose kick.
- "Friction" needs a concrete meaning: agent-block shape friction, and/or `space.damping > 0` as floor drag.

## 5. Data (h5 schema)

`HDF5Dataset` (`data/formats/hdf5.py:27`): flat per-row columns plus `ep_len`, `ep_offset`; episode
`e` is rows `ep_offset[e] : ep_offset[e]+ep_len[e]`; frameskip is applied on read (not to
`action`, which gets reshaped to blocks).

`pusht_expert_train.h5` (read 2026-11-12 from a transient copy; `runs/phase0-expert-extract/summary.json`):
46,300,921,856 B, **2,336,736 rows, 18,685 episodes**, episode length min 49 / median 123 / max 246
env steps (recomputed from `step_idx`; matches the stored `ep_len`). No file attrs.

| key | shape | dtype | chunks | filter |
|---|---|---|---|---|
| `pixels` | (2336736, 224, 224, 3) | uint8 | (100, 224, 224, 3) | 32001 (Blosc), so it needs `hdf5plugin` |
| `action` | (2336736, 2) | float32 | (1000, 2) | none |
| `proprio` | (2336736, 4) | float32 | (1000, 4) | none |
| `state` | (2336736, 7) | float32 | (1000, 7) | none |
| `episode_idx`, `step_idx` | (2336736,) | int64 | (1000,) | none |
| `ep_len` / `ep_offset` | (18685,) | int32 / int64 | (1024,) | none |

Episode ids equal their position in `ep_offset` (asserted during extraction), which is what `load_chunk` assumes.
Train config (`config/train/data/pusht.yaml`): `frameskip 5`, `num_steps = num_preds + history = 4`,
name `pusht_expert_train.lance` (swm 0.1.x also has a Lance format).

## 6. Checkpoint formats

| Format | Load with | Notes |
|---|---|---|
| HF `weights.pt` + `config.json` | `swm.wm.utils.load_pretrained(name)` (`wm/utils.py:47`): resolves `name` under **`$STABLEWM_HOME/checkpoints/`** as a `.pt` file, a folder with exactly one `.pt` + `config.json`, or an HF repo id; then `hydra.utils.instantiate(config)` + `load_state_dict` | Targets must be importable (see §0). |
| `<name>_object.ckpt` | `swm.policy.AutoCostModel(name)` (`policy.py:519`): `torch.load` of a whole pickled module, from `$STABLEWM_HOME/checkpoints/<name>_object.ckpt` | README layout `$STABLEWM_HOME/pusht/lewm_object.ckpt` is stale for 0.1.1 (it now looks under `checkpoints/`). |
| our plan | `$STABLEWM_HOME/checkpoints/pusht/lewm/{weights.pt, config.json}` with retargeted config | `eval.py ... policy=pusht/lewm` then loads it unmodified. |

Encoder only: `JEPA.encoder` + `JEPA.projector` (`z = projector(encoder(x).last_hidden_state[:,0])`).

## Integration plan

| Need | Hook (no upstream edits) | Why it works |
|---|---|---|
| stochastic env | `gymnasium.Wrapper` in `cvar/envs/`, passed via `World(..., pre_wrappers=[...])` (`world.py:114`) or registered as a new id | `World` builds envs through gym; `pre_wrappers` sit directly on the base env |
| new predictor heads | modules in `cvar/predictors/` taking `(z_hist, a_emb)` and returning `S` samples | `JEPA.rollout` is the only predictor caller |
| CVaR / mean+std cost | a cost-model wrapper exposing `get_cost(info, candidates) -> (B,N)`: encode start/goal once, roll out `N·S` futures with **one shared ε^(1:S) per `get_cost` call** (one call = one CEM iteration, so this gives common random numbers), reduce the S costs per candidate | `CEMSolver` only needs `get_cost` returning `(B,N)` |
| eval with our heads / CPU | `cvar/eval*.py` reusing `World.evaluate` with start/goal from the `.npz`, `device` configurable, per-episode `episodes.jsonl` | `eval.py` hardcodes cuda, the dataset and the logging format |
