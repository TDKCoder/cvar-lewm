"""Closed-loop PushT eval of the pretrained LeWM with upstream's eval config, one episode at a time.

    python -m cvar.eval.closed_loop --phase upstream  --run-dir R [--pairs 10]  [--device cuda] [--smoke]
    python -m cvar.eval.closed_loop --phase kickcheck --run-dir R [--pairs 50]  [--device cuda] [--smoke]

Faithful to eval.py (codebase-map section 3) except where marked DEVIATION:
  - upstream's own swm.World._evaluate_from_dataset runs each episode (reset, the _set_state /
    _set_goal_state callables, start/goal info injection, goal frame re-injected every step,
    success = env `terminated` at any step, at most eval_budget = 50 env steps, reset_mode 'wait');
  - WorldModelPolicy with PlanConfig and CEM settings read from config/eval/pusht.yaml and
    config/eval/solver/cem.yaml (300 samples, 30 iterations, top 30, horizon 5, receding horizon 5,
    action_block 5), the expert StandardScaler stats, eval.py's image transform,
    model.interpolate_pos_encoding = True;
  - DEVIATION: one env per episode instead of 50 envs in one batch (so episodes are resumable and
    each has its own seeds). Planning is per env anyway (solver batch_size 1).
  - DEVIATION: CEM noise from CRNCEMSolver (cvar/planning/crn.py): reseeded per (pair, plan call)
    from --base-seed, identical in every condition, instead of one generator seeded with 42.
  - DEVIATION (kickcheck only): our seed-0 pairs have no stored frames, so the start and goal frames
    are rendered from the stored states via env._set_state (render_check: median latent distance
    to the dataset frame 0.38, i.e. 0.29 of one env step's latent change). The upstream phase uses
    the dataset's own frames (expert_columns.npz keeps them for upstream's 50 eval episodes).
  - The env is cvar/PushTStoch-v0 (p = 0: bit-for-bit upstream PushT, unit-tested).

Phases (each appends one JSON line per finished episode to <run-dir>/<phase>/episodes.jsonl via
resilience.EpisodeLog; a rerun skips finished (condition, pair) keys):
  upstream:  upstream's seed-42 pairs [0, --pairs), p = 0, dataset frames.
  kickcheck: our seed-0 pairs [0, --pairs), rendered frames, conditions p0, p0.05-medium,
             p0.15-medium, p0.05-large, p0.15-large; pair-major order, so a partial run stays paired.
             Same kick seed and CEM seeds for a pair in every condition. Afterwards, drift.json:
             pair 0 at p = 0 rerun once (identical actions expected on CPU; reported, not
             enforced, on GPU) and, per pair, agreement of the first plan's 25 actions across
             conditions (made from identical observations with identical seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CALLABLES = [{"method": "_set_state", "args": {"state": {"value": "state"}}},
             {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}]
COLS = ("action", "episode_idx", "proprio", "state", "step_idx")  # + pixels: HDF5Dataset's columns
KICK_SIZES = {"medium": ((5.0, 20.0), 10.0), "large": ((10.0, 40.0), 20.0)}  # = spread_study configs
KICKCHECK_CONDITIONS = [("p0", 0.0, None), ("p0.05-medium", 0.05, "medium"), ("p0.15-medium", 0.15, "medium"),
                        ("p0.05-large", 0.05, "large"), ("p0.15-large", 0.15, "large")]
SMOKE_CEM = dict(num_samples=30, n_steps=3, topk=10)


def store() -> Path:
    return Path(os.environ["STABLEWM_HOME"]) / "pusht"


# ----------------------------------------------------------------------------- pairs

@dataclass
class Pair:
    set: str
    idx: int
    uid: int            # global row of the start state (unique across sets)
    episode: int
    start: int
    init: dict
    goal: dict
    frames: np.ndarray  # (2, H, W, 3) uint8: start, goal
    frame_source: str


def load_pairs(set_name: str, n: int, frames: str) -> list[Pair]:
    ev = np.load(store() / "expert_eval_states.npz")
    rows_init, rows_goal = ev[f"{set_name}_row_init"][:n], ev[f"{set_name}_row_goal"][:n]
    init = {c: ev[f"{set_name}_init_{c}"][:n] for c in COLS}
    goal = {c: ev[f"{set_name}_goal_{c}"][:n] for c in COLS}
    if frames == "dataset":
        z = np.load(store() / "expert_columns.npz")
        pos = {int(r): i for i, r in enumerate(z["eval_episode_rows"])}
        missing = [int(r) for r in np.concatenate([rows_init, rows_goal]) if int(r) not in pos]
        assert not missing, f"no stored dataset frame for rows {missing[:5]} (only upstream's eval episodes)"
        px = z["eval_pixels"]
        fr = [np.stack([px[pos[int(a)]], px[pos[int(b)]]]) for a, b in zip(rows_init, rows_goal)]
    else:
        from cvar.data.render_check import render_states
        both = render_states(np.concatenate([init["state"], goal["state"]]), "set_state")
        fr = [np.stack([both[i], both[len(rows_init) + i]]) for i in range(len(rows_init))]
    return [Pair(set_name, i, int(rows_init[i]), int(init["episode_idx"][i]), int(init["step_idx"][i]),
                 {c: init[c][i] for c in COLS}, {c: goal[c][i] for c in COLS}, fr[i], frames)
            for i in range(len(rows_init))]


class PairDataset:
    """The two rows of one eval pair in swm HDF5Dataset's load_chunk format (only the first and
    last row of a chunk are used by World._evaluate_from_dataset, plus the rest for videos)."""

    column_names = ["action", "episode_idx", "pixels", "proprio", "state", "step_idx"]

    def __init__(self, pair: Pair, goal_offset: int):
        self.pair, self.goal_offset = pair, goal_offset

    def load_chunk(self, episodes_idx, start, end):
        p = self.pair
        assert list(episodes_idx) == [p.episode] and list(start) == [p.start], (episodes_idx, start)
        assert int(end[0]) - int(start[0]) == self.goal_offset + 1
        d = {c: torch.from_numpy(np.stack([p.init[c], p.goal[c]])) for c in COLS}
        d["action"] = d["action"].reshape(2, -1)
        d["pixels"] = torch.from_numpy(p.frames).permute(0, 3, 1, 2)  # (T, C, H, W) as HDF5Dataset
        return [d]


# ----------------------------------------------------------------------------- model / policy

def scalers():
    from sklearn.preprocessing import StandardScaler

    ev = np.load(store() / "expert_eval_states.npz")
    out = {}
    for col in ("action", "proprio", "state"):
        s = StandardScaler()
        s.mean_, s.scale_, s.var_ = (ev[f"scaler_{col}_{k}"].astype(np.float64) for k in ("mean", "scale", "var"))
        s.n_features_in_, s.n_samples_seen_ = len(s.mean_), int(ev[f"scaler_{col}_n"])
        out[col] = s
        if col != "action":
            out[f"goal_{col}"] = s  # as eval.py
    return out


def eval_config(smoke: bool) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "config/eval/pusht.yaml")
    sol = OmegaConf.load(ROOT / "config/eval/solver/cem.yaml")
    cem = {k: sol[k] for k in ("batch_size", "num_samples", "var_scale", "n_steps", "topk")}
    if smoke:
        cem.update(SMOKE_CEM)
    return {"plan_config": OmegaConf.to_container(cfg.plan_config), "cem": cem,
            "goal_offset": int(cfg.eval.goal_offset_steps), "eval_budget": int(cfg.eval.eval_budget),
            "img_size": int(cfg.eval.img_size)}


def build_policy(device: str, ecfg: dict, base_seed: int):
    import stable_worldmodel as swm
    from cvar.planning.bench_cpu import eval_transform
    from cvar.planning.crn import CRNCEMSolver
    from cvar.utils.checkpoint import load_lewm

    model = load_lewm("pusht/lewm", device)
    model.interpolate_pos_encoding = True
    solver = CRNCEMSolver(model=model, device=device, seed=0, base_seed=base_seed, **ecfg["cem"])
    tf = {"pixels": eval_transform(), "goal": eval_transform()}
    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**ecfg["plan_config"]),
                                         process=scalers(), transform=tf)
    return model, solver, policy


def make_world(p: float, kick: str | None, ecfg: dict):
    import stable_worldmodel as swm
    import cvar.envs  # noqa: F401  (registers cvar/PushTStoch-v0)

    kw = {}
    if kick is not None:
        kw = {"kick_dist_px": KICK_SIZES[kick][0], "kick_rot_deg": KICK_SIZES[kick][1]}
    return swm.World(env_name="cvar/PushTStoch-v0", num_envs=1, image_shape=(ecfg["img_size"],) * 2,
                     max_episode_steps=2 * ecfg["eval_budget"], p=p, **kw)


def find_wrapper(world):
    from cvar.envs.stochastic_pusht import StochasticPushT

    env = world.envs.envs[0]
    while not isinstance(env, StochasticPushT):
        env = env.env
    return env


# ----------------------------------------------------------------------------- one episode

@torch.inference_mode()
def latent_cost(model, frame: np.ndarray, goal: np.ndarray, device: str) -> float:
    from cvar.planning.bench_cpu import eval_transform

    tf = eval_transform()
    x = torch.stack([tf(frame), tf(goal)])[:, None].to(device)
    e = model.encode({"pixels": x})["emb"][:, -1].float()
    return float(((e[0] - e[1]) ** 2).sum())


def run_episode(world, policy, solver, model, pair: Pair, kseed: int, ecfg: dict, device: str) -> dict:
    wrap = find_wrapper(world)
    wrap.seed_next_reset(kseed)
    for buf in policy._action_buffer:  # episodes run back to back on one env: drop any leftover plan
        buf.clear()
    policy._next_init = None
    solver.begin_episode(pair.uid, step_fn=lambda: wrap.episode_kicks["n_steps"])
    t0 = time.perf_counter()
    res = world.evaluate(dataset=PairDataset(pair, ecfg["goal_offset"]), episodes_idx=[pair.episode],
                         start_steps=[pair.start], goal_offset=ecfg["goal_offset"],
                         eval_budget=ecfg["eval_budget"], callables=CALLABLES)
    wall = time.perf_counter() - t0
    base = wrap.unwrapped
    s, g = np.asarray(base._get_obs()), np.asarray(base.goal_state, dtype=np.float64)
    ang = abs(g[4] - s[4]) % (2 * np.pi)
    calls = solver.calls
    steps = [c["env_step"] for c in calls]
    return {
        "success": bool(res["episode_successes"][0]), **wrap.episode_kicks,
        "final_cost": latent_cost(model, base.render(), pair.frames[1], device),
        "final_block_pos_err": float(np.linalg.norm(g[2:4] - s[2:4])),
        "final_block_ang_err": float(min(ang, 2 * np.pi - ang)), "final_state": s.tolist(),
        "plan_calls": len(calls), "plan_seeds": [c["seed"] for c in calls], "replan_env_steps": steps,
        "env_steps_between_replans": np.diff(steps).tolist(),
        "plan_time_s": float(np.mean([c["time_s"] for c in calls])) if calls else None,
        "plan_times_s": [c["time_s"] for c in calls], "plan_costs": [c["cost"] for c in calls],
        "episode_time_s": wall, "actions": np.array(wrap.episode_actions).round(6).tolist(),
    }


# ----------------------------------------------------------------------------- phases

def base_record(pair: Pair, cond: str, p: float, kick: str | None, args, kseed: int, ecfg: dict) -> dict:
    return {"env": "pusht", "phase": args.phase, "condition": cond, "p": p, "kick": kick,
            "head": "lewm-pretrained", "cost": "mean", "alpha": None, "beta": None, "S": None, "n_euler": None,
            "seed": args.base_seed, "episode": pair.idx, "pair_set": pair.set, "pair": pair.idx,
            "pair_uid": pair.uid, "dataset_episode": pair.episode, "start_step": pair.start,
            "frames": pair.frame_source, "kick_seed": kseed, "device": args.device, "smoke": args.smoke,
            "cem": ecfg["cem"], "plan_config": ecfg["plan_config"], "eval_budget": ecfg["eval_budget"]}


def drift_report(run: Path, world0, policy, solver, model, pairs, ecfg, args) -> dict:
    from cvar.planning.crn import kick_seed

    rows = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines() if x.strip()]
    by_pair: dict[int, dict[str, list]] = {}
    for r in rows:
        if r["plan_calls"] > 0:
            k = min(len(r["actions"]), 25)  # the first plan covers env steps 0..24
            by_pair.setdefault(r["pair"], {})[r["condition"]] = np.array(r["actions"][:k])
    first_plan = []
    for pair, conds in sorted(by_pair.items()):
        acts = list(conds.values())
        k = min(len(a) for a in acts)
        first_plan.append(max(float(np.abs(a[:k] - acts[0][:k]).max()) for a in acts) if k else 0.0)
    p0 = next((r for r in rows if r["condition"] == "p0" and r["pair"] == pairs[0].idx), None)
    rep = run_episode(world0, policy, solver, model, pairs[0], kick_seed(args.base_seed, pairs[0].uid), ecfg, args.device)
    a, b = np.array(p0["actions"]) if p0 else None, np.array(rep["actions"])
    out = {"device": args.device,
           "first_plan_max_abs_diff_across_conditions": {"max": max(first_plan) if first_plan else None,
                                                          "n_pairs": len(first_plan),
                                                          "n_pairs_exact": int(sum(d == 0.0 for d in first_plan))},
           "repeat_pair0_p0": None if p0 is None else {
               "same_length": len(a) == len(b),
               "max_abs_diff": float(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))]).max()),
               "same_success": p0["success"] == rep["success"]}}
    (run / "drift.json").write_text(json.dumps(out, indent=1))
    print("drift:", json.dumps(out), flush=True)
    return out


def main(argv=None) -> int:
    from cvar.planning.crn import kick_seed
    from cvar.utils.resilience import EpisodeLog, InterruptGuard

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["upstream", "kickcheck"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", type=int, default=None, help="default 10 (upstream) / 50 (kickcheck)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--resume", default="auto", choices=["auto"], help="always resumes from episodes.jsonl")
    ap.add_argument("--smoke", action="store_true", help=f"2 pairs, CEM {SMOKE_CEM}")
    args = ap.parse_args(argv)
    if "CVL_NCPU" in os.environ:
        torch.set_num_threads(int(os.environ["CVL_NCPU"]))
    n = 2 if args.smoke else (args.pairs or (10 if args.phase == "upstream" else 50))
    ecfg = eval_config(args.smoke)
    run = Path(args.run_dir) / args.phase
    run.mkdir(parents=True, exist_ok=True)
    log = EpisodeLog(run / "episodes.jsonl", key=("condition", "pair"))
    guard = InterruptGuard()

    if args.phase == "upstream":
        pairs, conds = load_pairs("upstream", n, "dataset"), [("p0", 0.0, None)]
    else:
        pairs, conds = load_pairs("ours", n, "render"), KICKCHECK_CONDITIONS
    todo = [(pr, c) for pr in pairs for c in conds if (c[0], pr.idx) not in log.done]
    print(f"[{args.phase}] {len(pairs)} pairs x {len(conds)} conditions; {len(todo)} episodes to run "
          f"({len(log.done)} already logged); device {args.device}; cem {ecfg['cem']}", flush=True)
    model, solver, policy = build_policy(args.device, ecfg, args.base_seed)
    worlds = {}
    for name, p, kick in conds:
        worlds[name] = make_world(p, kick, ecfg)
    current = None
    for pair, (name, p, kick) in todo:
        if current != name:
            worlds[name].set_policy(policy)  # (re)binds the policy's buffers and the solver to this world
            current = name
        kseed = kick_seed(args.base_seed, pair.uid)
        rec = base_record(pair, name, p, kick, args, kseed, ecfg)
        rec.update(run_episode(worlds[name], policy, solver, model, pair, kseed, ecfg, args.device))
        log.append(rec)
        print(f"[{args.phase}] {name:13s} pair {pair.idx:3d}: success={rec['success']} steps={rec['n_steps']} "
              f"kicks {rec['kicks_fired']}/{rec['kicks_full']} plans {rec['plan_calls']} "
              f"{rec['env_steps_between_replans']} plan {rec['plan_time_s']:.2f}s", flush=True)
        if guard.should_stop():
            print("[resilience] stopping between episodes; rerun resumes", flush=True)
            return 75
    if args.phase == "kickcheck" and not (run / "drift.json").exists():
        worlds["p0"].set_policy(policy)
        drift_report(run, worlds["p0"], policy, solver, model, pairs, ecfg, args)
    for w in worlds.values():
        w.close()
    print(f"[{args.phase}] done: {len(log.done)} episodes in {run / 'episodes.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
