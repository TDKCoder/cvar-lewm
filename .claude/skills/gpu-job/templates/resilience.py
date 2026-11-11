"""Interruption-safe training/eval helpers for DSMLP GPU pods.

Copy to cvar/utils/resilience.py. Covers what the DSMLP docs say can end a pod:
runtime limit (no warning), preemption (~15 min warning via /etc/podinfo files),
eviction (SIGTERM, 600 s grace), maintenance.

Usage:
    ckpt = CheckpointManager(run_dir, every_s=900, keep=2)
    guard = InterruptGuard()
    state = ckpt.load_latest()            # None on a fresh run
    if state: model.load_state_dict(state["model"]); ...; restore_rng_state(state["rng"])
    for step in range(start, total):
        ...
        ckpt.maybe_save(lambda: make_state(step), step)
        if guard.should_stop():
            ckpt.save_now(make_state(step), step)
            sys.exit(75)                  # 75 = interrupted, resumable
"""
from __future__ import annotations

import glob
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
except ImportError:  # lets the CPU-only unit tests run without torch
    torch = None

PODINFO = os.environ.get("PODINFO", "/etc/podinfo")


# ----------------------------------------------------------------------------- saving

def _save(obj: Any, fh) -> None:
    if torch is not None:
        torch.save(obj, fh)
    else:
        import pickle
        pickle.dump(obj, fh)


def _load(path: str) -> Any:
    if torch is not None:
        # weights_only=False is needed for optimizer/RNG state; only load our own files.
        return torch.load(path, map_location="cpu", weights_only=False)
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_atomic(obj: Any, path: str) -> None:
    """Write to path.tmp, fsync, then rename: a crash never leaves a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        _save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        s["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def restore_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    if torch is not None and "torch" in s:
        torch.set_rng_state(s["torch"])
        if "cuda" in s and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["cuda"])


class CheckpointManager:
    """Wall-clock checkpoints, keep the newest `keep`, fall back if the newest is corrupt."""

    def __init__(self, run_dir: str, every_s: float = 900, keep: int = 2):
        self.dir = Path(run_dir) / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every_s = every_s
        self.keep = keep
        self._last = time.monotonic()

    def _paths(self) -> list[Path]:
        return sorted(self.dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

    def save_now(self, state: dict, step: int) -> Path:
        path = self.dir / f"step_{step:09d}.pt"
        state = dict(state, _step=step, _saved_at=time.time())
        save_atomic(state, str(path))
        for old in self._paths()[: -self.keep]:
            old.unlink(missing_ok=True)
        self._last = time.monotonic()
        return path

    def maybe_save(self, make_state: Callable[[], dict], step: int) -> Optional[Path]:
        if time.monotonic() - self._last >= self.every_s:
            return self.save_now(make_state(), step)
        return None

    def load_latest(self) -> Optional[dict]:
        for path in reversed(self._paths()):
            try:
                return _load(str(path))
            except Exception as exc:  # corrupt or partial: try the previous one
                print(f"[resilience] could not load {path}: {exc}; trying older checkpoint")
        return None


# ------------------------------------------------------------------------ interruptions

def termination_warned(podinfo: str = PODINFO) -> bool:
    """True if the reservation controller has posted a termination warning.

    Per the DSMLP checkpointing docs, annotations appear as files under /etc/podinfo and an
    absent annotation is an EMPTY file, so test for non-empty. Match by suffix because the
    file may be named with or without the 'galends/' prefix.
    """
    for f in glob.glob(os.path.join(podinfo, "**", "*termination-warning-at"), recursive=True):
        try:
            if os.path.getsize(f) > 0:
                return True
        except OSError:
            pass
    return False


def termination_message(podinfo: str = PODINFO) -> str:
    for f in glob.glob(os.path.join(podinfo, "**", "*termination-warning-message"), recursive=True):
        try:
            return Path(f).read_text().strip()
        except OSError:
            pass
    return ""


class InterruptGuard:
    """SIGTERM flag + rate-limited polling of the podinfo warning files."""

    def __init__(self, poll_every_s: float = 20.0, podinfo: str = PODINFO):
        self.poll_every_s = poll_every_s
        self.podinfo = podinfo
        self._sigterm = False
        self._last_poll = 0.0
        self._warned = False
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame) -> None:
        print("[resilience] SIGTERM received; will checkpoint and exit", flush=True)
        self._sigterm = True

    def should_stop(self) -> bool:
        if self._sigterm:
            return True
        now = time.monotonic()
        if now - self._last_poll >= self.poll_every_s:
            self._last_poll = now
            self._warned = termination_warned(self.podinfo)  # re-read: warnings can be withdrawn
            if self._warned:
                print(f"[resilience] termination warning: {termination_message(self.podinfo)}", flush=True)
        return self._warned
