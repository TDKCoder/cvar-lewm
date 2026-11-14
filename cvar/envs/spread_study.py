"""Phase 0 block 6: how far a fixed open-loop plan's outcome spreads under stochastic PushT.

    python -m cvar.envs.spread_study --out $RUNS_DIR/phase0-spread [--smoke]

The plan: for each of the first --n-plans upstream eval pairs (expert_eval_states.npz, seed 42), the
expert's own 25 actions from the start row (expert_columns.npz rows row_init .. row_init+24), replayed
open-loop from the start state with the pair's goal state, exactly as eval sets them
(env.reset(options={"state", "goal_state"})). At p = 0 this is a deterministic replay; with kicks
it is the same action sequence under --seeds kick seeds (seed j uses reset seed 1000*plan + j).

Grid: kick configs (named below) x p in {0, 0.05, 0.15}, physics off; plus the "medium" kick config
with randomize_physics on at p in {0, 0.15}. The kick draws are coupled across p AND across configs
(same u_t, theta_t; d_t and rho_t are the same uniforms mapped to each config's range).

One JSON line per rollout -> rollouts.jsonl:
  config, physics, p, plan, seed, n_steps, kicks_fired, kicks_full (s = 1), kicks_partial (0 < s < 1),
  kicks_vetoed (s = 0), success_any (env `terminated` at any step: eval.py's OR-accumulated success),
  success_final, final_state (7), goal_state (7), block_pos_err (px, block xy to goal),
  block_ang_err (rad, wrapped), eval_pos_diff (||goal[:4]-state[:4]||, the env's success distance),
  replay_block_err_max (px, max over steps of block xy vs the expert's own state; meaningful at p=0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

KICK_CONFIGS = {  # name: (dist_px [lo, hi], rot_deg)
    "small": ((2.0, 10.0), 5.0),
    "medium": ((5.0, 20.0), 10.0),   # the provisional default in config/cvar/env/pusht_stoch.yaml
    "large": ((10.0, 40.0), 20.0),
}


def store() -> Path:
    return Path(os.environ["STABLEWM_HOME"]) / "pusht"


def load_plans(n_plans: int, goal_offset: int = 25):
    ev = np.load(store() / "expert_eval_states.npz")
    cols = np.load(store() / "expert_columns.npz")
    rows = ev["upstream_row_init"][:n_plans]
    plans = []
    for i, r in enumerate(rows):
        seg = slice(int(r), int(r) + goal_offset)
        assert np.all(cols["episode_idx"][seg] == ev["upstream_episode"][i])
        plans.append({
            "plan": i, "row_init": int(r), "episode": int(ev["upstream_episode"][i]),
            "state0": cols["state"][int(r)].astype(np.float64),
            "goal": cols["state"][int(r) + goal_offset].astype(np.float64),
            "actions": cols["action"][seg].astype(np.float32),
            "expert_states": cols["state"][int(r) + 1: int(r) + goal_offset + 1].astype(np.float64),
        })
        assert np.allclose(plans[-1]["goal"], ev["upstream_goal_state"][i]), "goal row mismatch"
    return plans


def rollout(env, plan: dict, seed: int) -> dict:
    base = env.unwrapped
    env.reset(seed=seed, options={"state": plan["state0"].copy(), "goal_state": plan["goal"].copy()})
    fired = full = partial = vetoed = 0
    success_any = False
    rep_err = 0.0
    for t, a in enumerate(plan["actions"]):
        obs, _, term, _, info = env.step(a)
        k = info["cvl_kick"]  # [fired, u, theta, dist, rot, scale]
        if k[0]:
            fired += 1
            full += k[5] == 1.0
            partial += 0.0 < k[5] < 1.0
            vetoed += k[5] == 0.0
        success_any |= bool(term)
        rep_err = max(rep_err, float(np.linalg.norm(obs["state"][2:4] - plan["expert_states"][t][2:4])))
    s, g = obs["state"], plan["goal"]
    ang = abs(g[4] - s[4]) % (2 * math.pi)
    ang = min(ang, 2 * math.pi - ang)
    success_final, _ = base.eval_state(g, s)
    return {"n_steps": len(plan["actions"]), "kicks_fired": fired, "kicks_full": int(full),
            "kicks_partial": int(partial), "kicks_vetoed": int(vetoed),
            "success_any": success_any, "success_final": bool(success_final),
            "final_state": s.tolist(), "goal_state": g.tolist(),
            "block_pos_err": float(np.linalg.norm(g[2:4] - s[2:4])), "block_ang_err": float(ang),
            "eval_pos_diff": float(np.linalg.norm(g[:4] - s[:4])), "replay_block_err_max": rep_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-plans", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="2 plans x 3 seeds")
    a = ap.parse_args()
    if a.smoke:
        a.n_plans, a.seeds = 2, 3

    import stable_worldmodel.envs.pusht.env as pusht
    from cvar.envs.stochastic_pusht import StochasticPushT, load_config

    cfg = load_config()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = load_plans(a.n_plans)
    cells = [(name, False, p) for name in KICK_CONFIGS for p in (0.0, 0.05, 0.15)]
    cells += [("medium", True, p) for p in (0.0, 0.15)]
    t0 = time.time()
    tmp = out / "rollouts.jsonl.tmp"
    with tmp.open("w") as fh:
        for name, phys, p in cells:
            (lo, hi), rot = KICK_CONFIGS[name]
            env = StochasticPushT(pusht.PushT(), p=p, kick_dist_px=(lo, hi), kick_rot_deg=rot,
                                  backoff_halvings=cfg["kick"]["backoff_halvings"], randomize_physics=phys,
                                  drag_rate=cfg["physics"]["drag_rate"], mass=cfg["physics"]["mass"],
                                  contact_friction=cfg["physics"]["contact_friction"])
            n_succ = 0
            for pl in plans:
                for j in range(a.seeds):
                    r = rollout(env, pl, seed=1000 * pl["plan"] + j)
                    n_succ += r["success_any"]
                    fh.write(json.dumps({"config": name, "physics": phys, "p": p, "plan": pl["plan"],
                                         "episode": pl["episode"], "row_init": pl["row_init"],
                                         "seed": 1000 * pl["plan"] + j, **r}) + "\n")
            env.close()
            print(f"{name:6s} physics={int(phys)} p={p:.2f}: success_any {n_succ}/{len(plans) * a.seeds} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    tmp.replace(out / "rollouts.jsonl")
    (out / "meta.json").write_text(json.dumps({
        "n_plans": len(plans), "seeds": a.seeds, "kick_configs": KICK_CONFIGS, "cells": cells,
        "physics": cfg["physics"], "backoff_halvings": cfg["kick"]["backoff_halvings"],
        "goal_offset": 25, "elapsed_s": round(time.time() - t0, 1), "smoke": a.smoke}, indent=1))
    print("SPREAD_OK", out)


if __name__ == "__main__":
    main()
