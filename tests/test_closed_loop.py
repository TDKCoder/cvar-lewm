"""Closed-loop eval with planner common random numbers (cvar/eval/closed_loop.py, cvar/planning/crn.py).

CPU only: the same pair with the same kick and CEM seeds must give identical actions on two runs
(on GPU the job reports drift instead of failing). Uses the real pretrained checkpoint and eval
npz; skipped where they are absent.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_worldmodel")
torch = pytest.importorskip("torch")

STORE = Path(os.environ.get("STABLEWM_HOME", "/nonexistent"))
need = pytest.mark.skipif(
    not (STORE / "pusht" / "expert_eval_states.npz").is_file()
    or not (STORE / "checkpoints" / "pusht" / "lewm" / "weights.pt").is_file(),
    reason="needs the eval npz and the pretrained checkpoint in $STABLEWM_HOME")


@pytest.fixture(scope="module")
def setup():
    from cvar.eval import closed_loop as CL

    torch.set_num_threads(int(os.environ.get("CVL_NCPU", "1")))
    ecfg = CL.eval_config(smoke=True)
    model, solver, policy = CL.build_policy("cpu", ecfg, base_seed=0)
    pair = CL.load_pairs("ours", 1, "render")[0]
    worlds = {name: CL.make_world(p, kick, ecfg) for name, p, kick in CL.KICKCHECK_CONDITIONS[::2]}
    yield CL, ecfg, model, solver, policy, pair, worlds
    for w in worlds.values():
        w.close()


def run(CL, ecfg, model, solver, policy, pair, world, kseed):
    world.set_policy(policy)
    return CL.run_episode(world, policy, solver, model, pair, kseed, ecfg, "cpu")


@need
def test_same_pair_and_seeds_give_identical_actions(setup):
    CL, ecfg, model, solver, policy, pair, worlds = setup
    a = run(CL, ecfg, model, solver, policy, pair, worlds["p0.15-medium"], kseed=123)
    b = run(CL, ecfg, model, solver, policy, pair, worlds["p0.15-medium"], kseed=123)
    assert a["plan_seeds"] == b["plan_seeds"] and a["plan_calls"] == b["plan_calls"] >= 1
    assert np.array_equal(np.array(a["actions"]), np.array(b["actions"]))
    assert (a["success"], a["n_steps"], a["kicks_fired"], a["kicks_full"]) == \
           (b["success"], b["n_steps"], b["kicks_fired"], b["kicks_full"])


@need
def test_first_plan_is_shared_across_conditions(setup):
    """Plan call 0 sees the same observation and the same CEM seed in every condition."""
    CL, ecfg, model, solver, policy, pair, worlds = setup
    r0 = run(CL, ecfg, model, solver, policy, pair, worlds["p0"], kseed=7)
    r1 = run(CL, ecfg, model, solver, policy, pair, worlds["p0.15-large"], kseed=7)
    assert r0["plan_seeds"][0] == r1["plan_seeds"][0]
    k = min(25, len(r0["actions"]), len(r1["actions"]))
    assert np.array_equal(np.array(r0["actions"][:k]), np.array(r1["actions"][:k]))
    assert r0["kicks_fired"] == 0
    assert r0["env_steps_between_replans"] in ([], [25])  # receding horizon 5 blocks x action_block 5


@need
def test_plan_seed_changes_the_plan(setup):
    from cvar.planning.crn import plan_seed

    CL, ecfg, model, solver, policy, pair, worlds = setup
    a = run(CL, ecfg, model, solver, policy, pair, worlds["p0"], kseed=7)
    solver.base_seed = 1
    try:
        b = run(CL, ecfg, model, solver, policy, pair, worlds["p0"], kseed=7)
    finally:
        solver.base_seed = 0
    assert b["plan_seeds"][0] == plan_seed(1, pair.uid, 0) != a["plan_seeds"][0]
    assert not np.array_equal(np.array(a["actions"][:5]), np.array(b["actions"][:5]))


def test_seed_derivation_is_stable():
    from cvar.planning.crn import kick_seed, plan_seed

    assert plan_seed(0, 5, 0) == plan_seed(0, 5, 0)
    assert len({plan_seed(0, 5, c) for c in range(4)} | {plan_seed(0, 6, 0), plan_seed(1, 5, 0)}) == 6
    assert kick_seed(0, 5) != plan_seed(0, 5, 0)
