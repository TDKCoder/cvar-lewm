"""Pretrained LeWM checkpoints from the HF mirror, loadable by stable-worldmodel 0.1.1 unmodified.

The HF model repos (e.g. quentinll/lewm-pusht) ship `weights.pt` + `config.json`, but the config's
Hydra targets (`stable_worldmodel.wm.lewm.*`) don't exist in swm 0.1.1. `prepare_checkpoint`
rewrites them to the upstream LeWM classes in this repo (jepa.py / module.py, importable because
env.sh puts the project root on PYTHONPATH), keeping the original as `config.hf.json`. After that,
`swm.wm.utils.load_pretrained("pusht/lewm")` (what `eval.py policy=pusht/lewm` calls) works, with a
strict state-dict load.

Layout: $STABLEWM_HOME/checkpoints/<env>/lewm/{weights.pt, config.json, config.hf.json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

TARGET_MAP = {
    "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
}


def retarget(cfg: Any) -> Any:
    """Return a copy of a Hydra config with the HF-mirror targets mapped to upstream LeWM classes."""
    if isinstance(cfg, dict):
        out = {k: retarget(v) for k, v in cfg.items()}
        if out.get("_target_") in TARGET_MAP:
            out["_target_"] = TARGET_MAP[out["_target_"]]
        return out
    if isinstance(cfg, list):
        return [retarget(v) for v in cfg]
    return cfg


def checkpoint_dir(name: str = "pusht/lewm") -> Path:
    return Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")) / "checkpoints" / name


def prepare_checkpoint(name: str = "pusht/lewm") -> Path:
    """Idempotent: write the retargeted config.json next to weights.pt (original -> config.hf.json)."""
    d = checkpoint_dir(name)
    cfg_path, orig_path = d / "config.json", d / "config.hf.json"
    if not (d / "weights.pt").is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"expected weights.pt and config.json in {d}")
    cfg = json.loads(cfg_path.read_text())
    new = retarget(cfg)
    if new == cfg:
        return d  # already retargeted (or nothing to do)
    if not orig_path.exists():
        orig_path.write_text(cfg_path.read_text())
    tmp = cfg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new, indent=4))
    tmp.replace(cfg_path)
    return d


def load_lewm(name: str = "pusht/lewm", device: str = "cpu"):
    """Load the pretrained JEPA in eval mode (BatchNorm in the projectors uses running stats)."""
    import stable_worldmodel as swm

    prepare_checkpoint(name)
    model = swm.wm.utils.load_pretrained(name)
    return model.to(device).eval().requires_grad_(False)


def param_counts(model) -> dict[str, int]:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    parts = {k: count(getattr(model, k)) for k in ("encoder", "predictor", "action_encoder", "projector", "pred_proj")}
    parts["total"] = count(model)
    parts["predictor_path"] = parts["predictor"] + parts["action_encoder"] + parts["pred_proj"]
    return parts
