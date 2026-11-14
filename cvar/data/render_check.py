"""Can a PushT frame rendered from a stored state stand in for the dataset's own frame?

Upstream eval feeds the policy the dataset's start and goal frames. For our seed-0 eval pairs we
keep states only, so the closed-loop eval renders those frames from the state. This checks that
substitution on the 6,812 kept frames of upstream's 50 eval episodes (expert_columns.npz).

    python -m cvar.data.render_check --out $RUNS_DIR/phase0-render-check [--n-encode 256]

Two ways to put the env in a stored state before render():
  set_state: env._set_state(state), upstream's callable (sets poses + agent velocity, then one
             0.01 s physics substep, so the agent can move by |v| * 0.01 px);
  pose_only: set agent/block poses, reindex shapes, no physics step.
Pixel metrics per frame: exact match, mean |diff| (0-255), fraction of differing pixels.
Latent metric (frozen LeWM encoder, CPU): d = ||E(render) - E(dataset)||^2 (the planner's cost is
the same squared distance to the goal embedding), compared with the distance between the dataset
frames of consecutive env steps (t, t+1) and the median start->goal distance of the eval pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def render_states(states: np.ndarray, mode: str) -> np.ndarray:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)

    env = gym.make("swm/PushT-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    out = []
    for s in states.astype(np.float64):
        if mode == "set_state":
            env._set_state(s)
        else:
            env.agent.position = tuple(s[:2])
            env.agent.velocity = tuple(s[5:7])
            env.block.position = tuple(s[2:4])
            env.block.angle = float(s[4])
            env.space.reindex_shapes_for_body(env.agent)
            env.space.reindex_shapes_for_body(env.block)
        out.append(env.render().copy())
    env.close()
    return np.stack(out)


def pixel_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per = d.reshape(len(d), -1)
    frac = (d.max(axis=-1) > 0).reshape(len(d), -1).mean(1)
    return {"exact_frac": float((per.max(1) == 0).mean()), "mean_abs_median": float(np.median(per.mean(1))),
            "mean_abs_max": float(per.mean(1).max()), "diff_pixel_frac_median": float(np.median(frac)),
            "diff_pixel_frac_p99": float(np.percentile(frac, 99))}


def encode(frames: np.ndarray, model, tf, bs: int = 64) -> torch.Tensor:
    embs = []
    with torch.inference_mode():
        for i in range(0, len(frames), bs):
            x = torch.stack([tf(f) for f in frames[i:i + bs]])[:, None]
            embs.append(model.encode({"pixels": x})["emb"][:, -1].float())
    return torch.cat(embs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-encode", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ["CVL_NCPU"]))
    t0 = time.time()
    store = Path(os.environ["STABLEWM_HOME"]) / "pusht"
    z = np.load(store / "expert_columns.npz")
    rows, pix = z["eval_episode_rows"], z["eval_pixels"]
    states = z["state"][rows]
    res = {"n_frames": int(len(rows)), "frame_shape": list(pix.shape[1:])}
    renders = {}
    for mode in ("set_state", "pose_only"):
        renders[mode] = render_states(states, mode)
        res[mode] = pixel_stats(renders[mode], pix)
        print(mode, res[mode], flush=True)

    from cvar.planning.bench_cpu import eval_transform
    from cvar.utils.checkpoint import load_lewm

    model, tf = load_lewm("pusht/lewm", "cpu"), eval_transform()
    rng = np.random.default_rng(0)
    ep = z["episode_idx"][rows]
    # sample frames whose next row is in the same episode, so (t, t+1) is a real env step
    ok = np.nonzero(ep[:-1] == ep[1:])[0]
    idx = np.sort(rng.choice(ok, size=min(a.n_encode, len(ok)), replace=False))
    e_data, e_next = encode(pix[idx], model, tf), encode(pix[idx + 1], model, tf)
    step = ((e_next - e_data) ** 2).sum(-1)
    lat = {"n": int(len(idx)), "data_step_t_t1": {"median": float(step.median()), "p10": float(step.quantile(0.1))}}
    for mode in renders:
        d = ((encode(renders[mode][idx], model, tf) - e_data) ** 2).sum(-1)
        lat[mode] = {"median": float(d.median()), "p90": float(d.quantile(0.9)), "max": float(d.max()),
                     "median_ratio_to_step": float((d / step.clamp_min(1e-12)).median())}
    ev = np.load(store / "expert_eval_states.npz")
    k = min(50, len(ev["frames_init"]))
    sg = ((encode(ev["frames_goal"][:k], model, tf) - encode(ev["frames_init"][:k], model, tf)) ** 2).sum(-1)
    lat["start_goal_median"] = float(sg.median())
    res["latent_sq_dist"] = lat
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(lat, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "render_check.json").write_text(json.dumps(res, indent=1))
    print("RENDER_CHECK_OK")


if __name__ == "__main__":
    main()
