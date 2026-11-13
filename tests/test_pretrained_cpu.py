"""Pretrained LeWM (quentinll/lewm-pusht) on CPU: load, encode, one predictor step, a tiny CEM plan.

Skips when the checkpoint isn't in $STABLEWM_HOME/checkpoints/pusht/lewm. Latency of the default
CEM config is measured by `python -m cvar.planning.bench_cpu` (results/phase0/pretrained_cpu.json).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
from cvar.utils.checkpoint import TARGET_MAP, checkpoint_dir, load_lewm, param_counts, retarget  # noqa: E402

CKPT = checkpoint_dir("pusht/lewm")
needs_ckpt = pytest.mark.skipif(not (CKPT / "weights.pt").is_file(), reason=f"no checkpoint in {CKPT}")

# Measured 2026-11-13 (bench_cpu.py); a change means a different architecture/config was loaded.
EXPECTED_PARAMS = {"encoder": 5_501_376, "predictor": 10_791_360, "action_encoder": 156_206,
                   "projector": 792_768, "pred_proj": 792_768, "total": 18_034_478}


def test_retarget_maps_every_hf_target():
    cfg = {"_target_": "stable_worldmodel.wm.lewm.LeWM",
           "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor", "depth": 6},
           "proj": {"_target_": "stable_worldmodel.wm.lewm.module.MLP",
                    "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}}}
    out = retarget(cfg)
    assert out["_target_"] == "jepa.JEPA" and out["predictor"]["_target_"] == "module.ARPredictor"
    assert out["proj"]["_target_"] == "module.MLP" and out["proj"]["norm_fn"]["_target_"] == "torch.nn.BatchNorm1d"
    assert retarget(out) == out and cfg["_target_"] == "stable_worldmodel.wm.lewm.LeWM"  # idempotent, no mutation


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(int(os.environ.get("CVL_NCPU", "4")))
    return load_lewm("pusht/lewm", "cpu")


@pytest.fixture(scope="module")
def frames():
    from cvar.planning.bench_cpu import eval_transform, load_frames

    f_init, f_goal, _ = load_frames()
    tf = eval_transform()
    return tf(f_init)[None, None], tf(f_goal)[None, None]


@needs_ckpt
def test_checkpoint_layout_after_retarget(model):
    cfg = json.loads((CKPT / "config.json").read_text())
    assert not any(t in json.dumps(cfg) for t in TARGET_MAP)  # every old target rewritten
    assert (CKPT / "config.hf.json").is_file()  # original kept


@needs_ckpt
def test_param_counts(model):
    pc = param_counts(model)
    print("param counts:", pc)
    for k, v in EXPECTED_PARAMS.items():
        assert pc[k] == v, (k, pc[k], v)


@needs_ckpt
def test_encode_and_predict_shapes(model, frames):
    pix, goal = frames
    with torch.inference_mode():
        e1 = model.encode({"pixels": pix.clone()})["emb"]
        e2 = model.encode({"pixels": pix.clone()})["emb"]
        eg = model.encode({"pixels": goal.clone()})["emb"]
        assert e1.shape == (1, 1, 192) and torch.isfinite(e1).all()
        assert torch.equal(e1, e2), "eval-mode encoding must be deterministic"
        assert ((e1 - eg) ** 2).sum() > 0
        hist = e1.expand(1, 3, -1).contiguous()
        pred = model.predict(hist, model.action_encoder(torch.zeros(1, 3, 10)))
        assert pred.shape == (1, 3, 192) and torch.isfinite(pred).all()


@needs_ckpt
def test_tiny_cem_plan_improves_on_the_start_cost(model, frames):
    from cvar.planning.bench_cpu import make_solver

    pix, goal = frames
    info = {"pixels": pix, "goal": goal, "action": torch.zeros(1, 1, 10)}
    solver = make_solver(model, num_samples=30, n_steps=3, topk=5, seed=0)
    t0 = time.perf_counter()
    out = solver.solve({k: v.clone() for k, v in info.items()})
    dt = time.perf_counter() - t0
    acts = out["actions"]
    assert acts.shape == (1, 5, 10) and torch.isfinite(acts).all()
    # cost of the plan vs cost of doing nothing (all-zero normalized actions), same model
    with torch.inference_mode():
        def cost(a):
            d = {k: v.clone().unsqueeze(1) for k, v in info.items()}
            return float(model.get_cost(d, a.unsqueeze(1)).squeeze())
        c_plan, c_zero = cost(acts.float()), cost(torch.zeros_like(acts))
    print(f"tiny CEM: {dt:.2f} s, cost plan {c_plan:.2f} vs zero actions {c_zero:.2f}")
    assert c_plan < c_zero
    # same seed -> same plan
    again = make_solver(model, num_samples=30, n_steps=3, topk=5, seed=0).solve({k: v.clone() for k, v in info.items()})
    assert torch.equal(again["actions"], acts)
