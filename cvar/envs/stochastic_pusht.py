"""Stochastic PushT: random T-block pose kicks (prob p per step) + optional per-episode physics.

Definitions (also in docs/decisions.md, 2026-11-12):

Per env step t, ALWAYS drawn in this order from the wrapper's own Generator, whatever p is:
    u_t ~ U[0, 1),  theta_t ~ U[0, 2 pi),  d_t ~ U[d_lo, d_hi] px,  rho_t ~ U[-rho_max, rho_max] rad
The kick fires iff u_t < p. So for a fixed seed the steps that fire at p = 0.05 are a subset of
those at p = 0.15, with identical (theta, d, rho): cross-p comparisons are paired.
A fired kick moves the block BEFORE the inner step:
    x_block <- x_block + s d_t (cos theta_t, sin theta_t),   phi_block <- phi_block + s rho_t
with s the largest of {1, 1/2, 1/4, ...} (backoff_halvings times) that leaves the block not
overlapping the agent or a wall, else s = 0 (logged). A velocity impulse would not work: the env
runs with space.damping = 0, which wipes a dynamic body's velocity after one 0.01 s substep
(measured: a 200 px/s kick moves the block 1.35 px; results/phase0/physics_probe.txt).

Per episode (reset), ALWAYS drawn: c ~ U[c_lo, c_hi] 1/s, m ~ U[m_lo, m_hi], mu ~ U[mu_lo, mu_hi].
Applied only if randomize_physics:
    ground drag  dv/dt = -c v       via space.damping = exp(-c)  (velocity kept per substep: exp(-c dt))
    block mass   m, moment of inertia unchanged (scaling both is an exact no-op under the kinematic pusher)
    contact friction mu on the block's shapes, 1.0 on the agent's (pymunk multiplies the two)
With p = 0 and randomize_physics off, the wrapper never touches the env: bit-for-bit upstream.

RNG: seeded from np.random.SeedSequence([reset_seed, STREAM]) on every seeded reset, so the same
(seed, actions) gives the same trajectory; an unseeded reset keeps the current stream.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

STREAM = 0xC7A2  # separates our stream from anything else seeded with the same env seed
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "cvar" / "env" / "pusht_stoch.yaml"


def load_config(path: str | Path | None = None, **overrides: Any) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


class StochasticPushT(gym.Wrapper):
    """Wrap a stable-worldmodel PushT env (`swm/PushT-v1`). Must sit directly on the base env."""

    def __init__(
        self,
        env: gym.Env,
        p: float = 0.0,
        kick_dist_px: tuple[float, float] = (5.0, 20.0),
        kick_rot_deg: float = 10.0,
        backoff_halvings: int = 3,
        randomize_physics: bool = False,
        drag_rate: tuple[float, float] = (10.0, 50.0),
        mass: tuple[float, float] = (0.5, 2.0),
        contact_friction: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.kick_dist_px = tuple(map(float, kick_dist_px))
        self.kick_rot = math.radians(float(kick_rot_deg))
        self.backoff_halvings = int(backoff_halvings)
        self.randomize_physics = bool(randomize_physics)
        self.drag_rate = tuple(map(float, drag_rate))
        self.mass = tuple(map(float, mass))
        self.contact_friction = tuple(map(float, contact_friction))
        self._rng = np.random.default_rng(np.random.SeedSequence())
        self.physics = np.full(4, np.nan)  # [applied, c, m, mu] for the current episode

    @classmethod
    def from_config(cls, env: gym.Env, cfg: dict | str | Path | None = None, **overrides: Any) -> "StochasticPushT":
        cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        cfg = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}
        k, ph = cfg.get("kick", {}), cfg.get("physics", {})
        return cls(
            env,
            p=cfg.get("p", 0.0),
            kick_dist_px=k.get("dist_px", (5.0, 20.0)),
            kick_rot_deg=k.get("rot_deg", 10.0),
            backoff_halvings=k.get("backoff_halvings", 3),
            randomize_physics=cfg.get("randomize_physics", False),
            drag_rate=ph.get("drag_rate", (10.0, 50.0)),
            mass=ph.get("mass", (0.5, 2.0)),
            contact_friction=ph.get("contact_friction", (0.0, 0.0)),
        )

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence([int(seed), STREAM]))
        c = self._rng.uniform(*self.drag_rate)
        m = self._rng.uniform(*self.mass)
        mu = self._rng.uniform(*self.contact_friction)
        if self.randomize_physics:
            self._apply_physics(c, m, mu)
        self.physics = np.array([float(self.randomize_physics), c, m, mu])
        info["cvl_physics"] = self.physics.copy()
        return obs, info

    def step(self, action):
        u = self._rng.random()
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        dist = self._rng.uniform(*self.kick_dist_px)
        rot = self._rng.uniform(-self.kick_rot, self.kick_rot)
        fired = u < self.p
        scale = self._kick(theta, dist, rot) if fired else 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["cvl_kick"] = np.array([float(fired), u, theta, dist, rot, scale])
        info["cvl_physics"] = self.physics.copy()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def _apply_physics(self, c: float, m: float, mu: float) -> None:
        base = self.env.unwrapped
        base.space.damping = math.exp(-c)
        base.block.mass = m  # moment left at its default on purpose (see module docstring)
        for s in base.block.shapes:
            s.friction = mu
        for s in base.agent.shapes:
            s.friction = 1.0

    @staticmethod
    def _penetration(base) -> float:
        """Deepest overlap (px, >= 0) of the block's shapes with any other body (agent, walls)."""
        block = base.block
        base.space.reindex_shapes_for_body(block)
        base.space.reindex_shapes_for_body(base.agent)  # cheap; guards against a teleported agent
        depth = 0.0
        for shape in block.shapes:
            for hit in base.space.shape_query(shape):
                if hit.shape is None or hit.shape.body is block:
                    continue
                # Only touching shapes are returned. pymunk 7.3 reports penetration as a POSITIVE
                # distance (older docs: negative), so use |distance| to be convention-proof.
                for pt in hit.contact_point_set.points:
                    depth = max(depth, abs(pt.distance))
        return depth

    def _kick(self, theta: float, dist: float, rot: float) -> float:
        # Accept the largest scale whose penetration is no worse than before the kick (the agent
        # is often in resting contact while pushing) or than a 1 px slop.
        base = self.env.unwrapped
        block = base.block
        pos0, ang0 = block.position, block.angle
        allowed = max(self._penetration(base), 1.0)
        scale = 1.0
        for _ in range(self.backoff_halvings + 1):
            block.position = (pos0.x + scale * dist * math.cos(theta), pos0.y + scale * dist * math.sin(theta))
            block.angle = ang0 + scale * rot
            if self._penetration(base) <= allowed:
                return scale
            scale *= 0.5
        block.position, block.angle = pos0, ang0
        base.space.reindex_shapes_for_body(block)
        return 0.0


def make_stochastic_pusht(config: dict | str | None = None, p: float | None = None,
                          randomize_physics: bool | None = None, **env_kwargs: Any) -> gym.Env:
    """gym entry point for `cvar/PushTStoch-v0`: swm PushT + StochasticPushT from the yaml config."""
    import stable_worldmodel.envs.pusht.env as pusht  # noqa: F401  (swm registers swm/PushT-v1)

    base = pusht.PushT(**env_kwargs)
    return StochasticPushT.from_config(base, config, p=p, randomize_physics=randomize_physics)
