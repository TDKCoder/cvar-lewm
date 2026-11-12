"""Stochastic environment wrappers (PushT impulses / physics resampling; TwoRoom slip later)."""
from gymnasium.envs.registration import register, registry

if "cvar/PushTStoch-v0" not in registry:
    register(id="cvar/PushTStoch-v0", entry_point="cvar.envs.stochastic_pusht:make_stochastic_pusht")
