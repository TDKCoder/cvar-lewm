"""Stochastic PushT wrapper: exactness at p = 0, determinism, paired coupling across p, physics."""
import math

import numpy as np
import pytest

pusht = pytest.importorskip("stable_worldmodel.envs.pusht.env")
from cvar.envs.stochastic_pusht import StochasticPushT, load_config  # noqa: E402

N_STEPS = 100


def near_block_actions(env, rng):
    """WeakPolicy-style action: random target within 60 px of the block, as a relative action."""
    base = env.unwrapped
    a = rng.uniform(-1, 1, 2) * base.action_scale + np.array(base.agent.position)
    bp = np.array(base.block.position)
    a = np.clip(a, bp - 60, bp + 60)
    return np.clip((a - np.array(base.agent.position)) / base.action_scale, -1, 1).astype(np.float32)


def rollout(env, seed, action_seed, n=N_STEPS, frames_every=10, actions=None):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(action_seed)
    out = {"state": [obs["state"].copy()], "proprio": [obs["proprio"].copy()], "reward": [], "term": [],
           "frames": [env.render().copy()], "kick": [], "actions": []}
    for t in range(n):
        a = actions[t] if actions is not None else near_block_actions(env, rng)
        obs, r, term, trunc, info = env.step(a)
        out["state"].append(obs["state"].copy()); out["proprio"].append(obs["proprio"].copy())
        out["reward"].append(r); out["term"].append(term); out["actions"].append(a)
        if "cvl_kick" in info:
            out["kick"].append(info["cvl_kick"])
        if (t + 1) % frames_every == 0:
            out["frames"].append(env.render().copy())
    return {k: np.array(v) for k, v in out.items()}


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_p0_is_bit_for_bit_upstream(seed):
    raw = rollout(pusht.PushT(), seed, action_seed=100 + seed)
    wrapped = rollout(StochasticPushT(pusht.PushT(), p=0.0), seed, action_seed=100 + seed)
    for k in ("state", "proprio", "reward", "term", "frames", "actions"):
        assert np.array_equal(raw[k], wrapped[k]), f"{k} differs at p=0"
    assert not wrapped["kick"][:, 0].any()
    assert np.ptp(raw["state"][:, 2:5], axis=0).max() > 1.0, "block never moved: test would be vacuous"


def test_same_seed_same_trajectory_at_p015():
    a = rollout(StochasticPushT(pusht.PushT(), p=0.15), 3, action_seed=5)
    b = rollout(StochasticPushT(pusht.PushT(), p=0.15), 3, action_seed=5)
    for k in ("state", "frames", "kick"):
        assert np.array_equal(a[k], b[k]), k
    assert a["kick"][:, 0].sum() > 0, "no kick fired in 100 steps at p=0.15"


def test_different_seeds_differ_at_p015():
    a = rollout(StochasticPushT(pusht.PushT(), p=0.15), 3, action_seed=5)
    env = StochasticPushT(pusht.PushT(), p=0.15)
    b = rollout(env, 3, action_seed=5)  # same env seed -> same as a
    env._rng = np.random.default_rng(np.random.SeedSequence([4, 1]))  # different kick stream only
    obs, _ = env.env.reset(seed=3)
    states = [obs["state"].copy()]
    for act in a["actions"]:
        obs, *_ = env.step(act)
        states.append(obs["state"].copy())
    assert np.array_equal(a["state"], b["state"])
    assert not np.array_equal(a["state"], np.array(states)), "different kick streams gave the same trajectory"


def test_kick_draws_are_coupled_across_p():
    # Same env seed + same action sequence: draws (u, theta, d, rho) are identical for every p and
    # the fired set grows with p.
    lo = rollout(StochasticPushT(pusht.PushT(), p=0.05), 11, action_seed=2)
    hi = rollout(StochasticPushT(pusht.PushT(), p=0.15), 11, action_seed=2, actions=lo["actions"])
    zero = rollout(StochasticPushT(pusht.PushT(), p=0.0), 11, action_seed=2, actions=lo["actions"])
    assert np.array_equal(lo["kick"][:, 1:5], hi["kick"][:, 1:5])
    assert np.array_equal(zero["kick"][:, 1:5], hi["kick"][:, 1:5])
    f_lo, f_hi = lo["kick"][:, 0].astype(bool), hi["kick"][:, 0].astype(bool)
    assert np.all(f_hi[f_lo]), "a step that fired at p=0.05 did not fire at p=0.15"
    assert np.array_equal(f_lo, lo["kick"][:, 1] < 0.05) and np.array_equal(f_hi, hi["kick"][:, 1] < 0.15)


def test_kick_moves_block_by_the_drawn_amount():
    env = StochasticPushT(pusht.PushT(), p=1.0, kick_dist_px=(15.0, 15.0), kick_rot_deg=0.0)
    env.reset(seed=0)
    base = env.unwrapped
    base.agent.position = (30, 30)  # park the agent away from the block
    base.space.reindex_shapes_for_body(base.agent)
    p0 = np.array(base.block.position)
    _, _, _, _, info = env.step(np.zeros(2, np.float32))
    fired, u, theta, d, rot, s = info["cvl_kick"]
    assert fired == 1.0 and s > 0
    moved = np.linalg.norm(np.array(base.block.position) - p0)
    # pose kick (s*d) plus at most a pixel or two of contact resolution / one-substep drift
    assert abs(moved - s * d) < 2.0, (moved, s * d)


def test_randomize_physics_flag():
    cfg = load_config()
    on = StochasticPushT.from_config(pusht.PushT(), cfg, randomize_physics=True)
    off = StochasticPushT.from_config(pusht.PushT(), cfg, randomize_physics=False)
    _, info_on = on.reset(seed=5)
    _, info_off = off.reset(seed=5)
    applied, c, m, mu = info_on["cvl_physics"]
    ph = cfg["physics"]
    assert applied == 1.0 and ph["drag_rate"][0] <= c <= ph["drag_rate"][1] and ph["mass"][0] <= m <= ph["mass"][1]
    assert math.isclose(on.unwrapped.space.damping, math.exp(-c)) and math.isclose(on.unwrapped.block.mass, m)
    assert np.array_equal(info_on["cvl_physics"][1:], info_off["cvl_physics"][1:]), "draws must not depend on the flag"
    assert off.unwrapped.space.damping == 0.0 and off.unwrapped.block.mass == 1.0


def test_penetration_detects_agent_and_wall_overlap():
    env = StochasticPushT(pusht.PushT(), p=0.0)
    env.reset(seed=0)
    b = env.unwrapped
    b.agent.position = (30, 30)
    b.block.position = (250, 250)
    assert env._penetration(b) == 0.0
    b.agent.position = b.block.local_to_world((0, -14))  # agent radius 15: ~1 px into the bar
    assert 0.5 < env._penetration(b) < 1.5
    b.agent.position = (30, 30)
    b.block.position = (8, 250)  # bar pushed into the left wall
    assert env._penetration(b) > 5.0


def test_kick_into_agent_is_scaled_back():
    env = StochasticPushT(pusht.PushT(), p=1.0, kick_dist_px=(20.0, 20.0), kick_rot_deg=0.0)
    env.reset(seed=0)
    b = env.unwrapped
    b.block.position, b.block.angle = (250, 250), 0.0
    b.agent.position = b.block.local_to_world((0, -16))  # 1 px below the bar, i.e. touching side
    b.space.reindex_shapes_for_body(b.agent)
    toward_agent = math.atan2(*(np.array(b.agent.position) - np.array(b.block.position))[::-1])
    s = env._kick(toward_agent, 20.0, 0.0)
    assert s < 1.0, "a 20 px kick straight into the agent should be scaled back or vetoed"
