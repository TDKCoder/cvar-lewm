"""Phase 0 block 4: pretrained LeWM on CPU - encode, one predictor step, CEM plan latency.

    python -m cvar.planning.bench_cpu --out $RUNS_DIR/phase0-cpu-bench [--smoke]

Uses the first upstream (seed-42) start/goal frame pair from expert_eval_states.npz, transformed
exactly like eval.py's img_transform (ToImage -> float [0,1] -> ImageNet normalize -> resize 224).
A "plan call" = one CEMSolver.solve on one env = n_steps x JEPA.get_cost over num_samples
candidates of shape (horizon=5, action_block 5 x 2 = 10), i.e. one replanning step of eval.py.
Latency = wall-clock per solve after one warm-up call; median and all repeats are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from cvar.utils.checkpoint import load_lewm, param_counts

HORIZON, ACTION_BLOCK, ACTION_DIM = 5, 5, 2


def eval_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2 as T

    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                      T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=224)])


def load_frames():
    npz = Path(os.environ["STABLEWM_HOME"]) / "pusht" / "expert_eval_states.npz"
    if npz.is_file():
        z = np.load(npz)
        return z["frames_init"][0], z["frames_goal"][0], str(npz)
    rng = np.random.default_rng(0)
    return (rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
            rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), "random")


def make_solver(model, num_samples, n_steps, topk, seed=1234):
    import gymnasium as gym
    import stable_worldmodel as swm
    from stable_worldmodel.solver import CEMSolver

    solver = CEMSolver(model, batch_size=1, num_samples=num_samples, n_steps=n_steps, topk=topk,
                       device="cpu", seed=seed)
    space = gym.spaces.Box(-1, 1, shape=(1, ACTION_DIM), dtype=np.float32)  # vectorized: (n_envs, 2)
    solver.configure(action_space=space, n_envs=1,
                     config=swm.PlanConfig(horizon=HORIZON, receding_horizon=HORIZON, action_block=ACTION_BLOCK))
    return solver


def time_solve(solver, info, repeats):
    out = solver.solve({k: v.clone() for k, v in info.items()})  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = solver.solve({k: v.clone() for k, v in info.items()})
        times.append(time.perf_counter() - t0)
    return out, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny CEM only (30 samples x 3 iterations)")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    ncpu = int(os.environ["CVL_NCPU"])
    torch.set_num_threads(ncpu)
    res: dict = {"torch": torch.__version__, "threads": ncpu, "cpu": platform.processor() or platform.machine()}

    t0 = time.perf_counter()
    model = load_lewm("pusht/lewm", "cpu")
    res["load_s"] = time.perf_counter() - t0
    res["params"] = param_counts(model)
    print("params:", res["params"], flush=True)

    f_init, f_goal, res["frames_source"] = load_frames()
    tf = eval_transform()
    pix = tf(f_init)[None, None]   # (B=1, T=1, 3, 224, 224)
    goal = tf(f_goal)[None, None]

    with torch.inference_mode():
        t0 = time.perf_counter()
        e = model.encode({"pixels": pix})["emb"]
        res["encode_one_frame_s"] = time.perf_counter() - t0
        eg = model.encode({"pixels": goal})["emb"]
        res["emb_shape"] = list(e.shape)
        res["start_goal_cost"] = float(((e - eg) ** 2).sum())  # same criterion as JEPA.criterion
        hist = e.expand(1, 3, -1).contiguous()                  # history of 3 (num_frames)
        act = torch.zeros(1, 3, ACTION_BLOCK * ACTION_DIM)
        t0 = time.perf_counter()
        pred = model.predict(hist, model.action_encoder(act))
        res["predict_one_step_s"] = time.perf_counter() - t0
        res["pred_shape"] = list(pred.shape)
    print({k: res[k] for k in ("emb_shape", "pred_shape", "encode_one_frame_s", "predict_one_step_s",
                               "start_goal_cost")}, flush=True)

    info = {"pixels": pix, "goal": goal, "action": torch.zeros(1, 1, ACTION_BLOCK * ACTION_DIM)}
    configs = [("tiny", 30, 3, 5)] + ([] if a.smoke else [("default", 300, 30, 30)])
    res["plans"] = {}
    for name, n, it, k in configs:
        solver = make_solver(model, n, it, k)
        out, times = time_solve(solver, info, a.repeats)
        acts = out["actions"]
        costs = out["costs"]
        last = costs[-1] if isinstance(costs, list) and costs else costs
        res["plans"][name] = {
            "num_samples": n, "n_steps": it, "topk": k, "repeats": a.repeats,
            "solve_s": times, "solve_s_median": statistics.median(times),
            "per_get_cost_s": statistics.median(times) / it,
            "actions_shape": list(acts.shape), "actions_finite": bool(torch.isfinite(acts).all()),
            "final_cost": float(np.asarray(last.cpu() if torch.is_tensor(last) else last).min()),
        }
        print(name, res["plans"][name], flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "bench.json").write_text(json.dumps(res, indent=1))
    print("BENCH_OK", Path(a.out) / "bench.json")


if __name__ == "__main__":
    main()
