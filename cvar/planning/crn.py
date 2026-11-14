"""Common random numbers for the CEM planner across eval conditions (decision 2026-11-13).

stable_worldmodel's CEMSolver draws every candidate from one torch.Generator seeded once at
construction, so the noise a plan call sees depends on how many calls came before it. In a paired
comparison (same pair, different kick condition) that adds CEM sampling noise to the difference.
CRNCEMSolver instead reseeds the generator before every solve:

    seed(pair_uid, call) = SeedSequence([base_seed, pair_uid, call]).generate_state(1)[0]

with `call` the plan-call index within the episode (0, 1, ...), identically in every condition.
Two conditions then see identical CEM noise at plan call k; as long as their observations at that
call are identical (always true for call 0), they produce identical plans.

Each call is logged in `self.calls`: seed, the env step it happened at (from `step_fn`), wall time
and the elites' final cost.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from stable_worldmodel.solver import CEMSolver


def plan_seed(base_seed: int, pair_uid: int, call: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(pair_uid), int(call)]).generate_state(1)[0])


def kick_seed(base_seed: int, pair_uid: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(pair_uid), 0xC1C]).generate_state(1)[0])


class CRNCEMSolver(CEMSolver):
    def __init__(self, *args, base_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = int(base_seed)
        self.pair_uid: int | None = None
        self.calls: list[dict] = []
        self.step_fn: Callable[[], int] = lambda: -1

    def begin_episode(self, pair_uid: int, step_fn: Callable[[], int] | None = None) -> None:
        self.pair_uid = int(pair_uid)
        self.calls = []
        if step_fn is not None:
            self.step_fn = step_fn

    def solve(self, info_dict: dict, init_action=None) -> dict:
        assert self.pair_uid is not None, "call begin_episode(pair_uid) before planning"
        seed = plan_seed(self.base_seed, self.pair_uid, len(self.calls))
        self.torch_gen.manual_seed(seed)
        env_step = int(self.step_fn())
        t0 = time.perf_counter()
        out = super().solve(info_dict, init_action=init_action)
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        self.calls.append({"seed": seed, "env_step": env_step, "time_s": time.perf_counter() - t0,
                           "cost": float(np.mean(out["costs"])) if out.get("costs") else None})
        return out
