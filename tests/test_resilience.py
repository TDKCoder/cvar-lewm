"""cvar/utils/resilience.py: checkpoints, interruption handling, RNG state, resumable episode log."""
import json
import os
import random
import signal

import numpy as np
import pytest

from cvar.utils import resilience as R


# ----------------------------------------------------------------------------- checkpoints

def test_rotation_keeps_newest(tmp_path):
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=2)
    for step in (1, 5, 12, 40):
        ck.save_now({"w": step}, step)
    names = sorted(p.name for p in (tmp_path / "ckpt").iterdir())
    assert names == ["step_000000012.pt", "step_000000040.pt"]
    assert ck.load_latest()["w"] == 40


def test_step_order_is_numeric_not_lexical(tmp_path):
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=3)
    for step in (9, 10, 100):
        ck.save_now({"w": step}, step)
    assert ck.load_latest()["_step"] == 100


def test_truncated_newest_falls_back(tmp_path, capsys):
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=2)
    ck.save_now({"w": 1}, 1)
    p2 = ck.save_now({"w": np.arange(10000)}, 2)
    p2.write_bytes(p2.read_bytes()[:50])  # the pod died while a non-atomic writer was mid-file
    state = ck.load_latest()
    assert state["w"] == 1 and "could not load" in capsys.readouterr().out


def test_all_corrupt_returns_none(tmp_path):
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=2)
    ck.save_now({"w": 1}, 1).write_bytes(b"garbage")
    assert ck.load_latest() is None


def test_fresh_run_returns_none(tmp_path):
    assert R.CheckpointManager(str(tmp_path)).load_latest() is None


def test_atomic_save_leaves_no_tmp_and_tmp_is_ignored(tmp_path):
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=2)
    ck.save_now({"w": 1}, 1)
    assert not list((tmp_path / "ckpt").glob("*.tmp"))
    # a leftover .tmp from a crash during save_atomic is not a checkpoint
    (tmp_path / "ckpt" / "step_000000099.pt.tmp").write_bytes(b"partial")
    assert ck.load_latest()["w"] == 1


def test_atomic_save_failure_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "x.pt"
    R.save_atomic({"v": 1}, str(path))

    def boom(obj, fh):
        fh.write(b"half")
        raise RuntimeError("disk full")

    monkeypatch.setattr(R, "_save", boom)
    with pytest.raises(RuntimeError):
        R.save_atomic({"v": 2}, str(path))
    monkeypatch.undo()
    assert R._load(str(path))["v"] == 1
    assert not (tmp_path / "x.pt.tmp").exists()


def test_maybe_save_respects_every_s(tmp_path, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(R.time, "monotonic", lambda: t[0])
    ck = R.CheckpointManager(str(tmp_path), every_s=900, keep=5)
    calls = []
    make = lambda: calls.append(1) or {"w": len(calls)}  # noqa: E731
    t[0] += 899
    assert ck.maybe_save(make, 1) is None and not calls  # state not even built before it's due
    t[0] += 1
    assert ck.maybe_save(make, 2) is not None and len(calls) == 1
    t[0] += 10
    assert ck.maybe_save(make, 3) is None  # timer restarted by the save
    t[0] += 890
    assert ck.maybe_save(make, 4) is not None


def test_save_now_resets_timer(tmp_path, monkeypatch):
    t = [0.0]
    monkeypatch.setattr(R.time, "monotonic", lambda: t[0])
    ck = R.CheckpointManager(str(tmp_path), every_s=100, keep=5)
    t[0] = 99
    ck.save_now({"w": 0}, 0)  # e.g. the SIGTERM path
    t[0] = 150
    assert ck.maybe_save(lambda: {"w": 1}, 1) is None


def test_pickle_path_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "torch", None)
    ck = R.CheckpointManager(str(tmp_path), every_s=0, keep=2)
    ck.save_now({"w": [1, 2]}, 3)
    assert ck.load_latest()["w"] == [1, 2]


# ---------------------------------------------------------------------------- RNG state

def test_rng_state_round_trip():
    torch = pytest.importorskip("torch")
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    s = R.rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    random.random(); np.random.rand(3); torch.rand(5)  # advance everything
    R.restore_rng_state(s)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == a


def test_rng_state_survives_checkpoint(tmp_path):
    pytest.importorskip("torch")
    np.random.seed(7)
    ck = R.CheckpointManager(str(tmp_path), every_s=0)
    ck.save_now({"rng": R.rng_state()}, 1)
    want = np.random.rand(4)
    np.random.seed(0)
    R.restore_rng_state(ck.load_latest()["rng"])
    assert np.array_equal(np.random.rand(4), want)


# --------------------------------------------------------------------- interruptions

@pytest.fixture
def restore_sigterm():
    old = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, old)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_sigterm_sets_flag(tmp_path, restore_sigterm):
    g = R.InterruptGuard(podinfo=str(tmp_path))
    assert not g.should_stop()
    os.kill(os.getpid(), signal.SIGTERM)
    assert g.should_stop()


def test_empty_warning_file_means_no_warning(tmp_path):
    write(tmp_path / "termination-warning-at", "")  # absent annotation = empty file (DSMLP docs)
    assert not R.termination_warned(str(tmp_path))


@pytest.mark.parametrize("name", ["termination-warning-at", "galends.termination-warning-at",
                                  "sub/termination-warning-at", "galends/termination-warning-at"])
def test_non_empty_warning_file_is_a_warning(tmp_path, name):
    write(tmp_path / name, "2026-11-13T20:00:00Z")
    assert R.termination_warned(str(tmp_path))


def test_missing_podinfo_dir_is_no_warning(tmp_path):
    assert not R.termination_warned(str(tmp_path / "nope"))
    assert R.termination_message(str(tmp_path / "nope")) == ""


def test_warning_message_is_read(tmp_path):
    write(tmp_path / "galends" / "termination-warning-message", "  preempted by reservation  \n")
    assert R.termination_message(str(tmp_path)) == "preempted by reservation"


def test_polling_is_rate_limited_and_rereads(tmp_path, monkeypatch, restore_sigterm):
    t = [1000.0]
    monkeypatch.setattr(R.time, "monotonic", lambda: t[0])
    reads = []
    real = R.termination_warned
    monkeypatch.setattr(R, "termination_warned", lambda pi: reads.append(1) or real(pi))
    g = R.InterruptGuard(poll_every_s=20, podinfo=str(tmp_path))
    assert not g.should_stop() and len(reads) == 1  # first call polls
    write(tmp_path / "termination-warning-at", "soon")
    t[0] += 19
    assert not g.should_stop() and len(reads) == 1  # within 20 s: no read, warning not seen yet
    t[0] += 1
    assert g.should_stop() and len(reads) == 2
    (tmp_path / "termination-warning-at").write_text("")  # warning withdrawn
    t[0] += 20
    assert not g.should_stop()


# ------------------------------------------------------------------- episode log (evals)

def test_episode_log_resume_skips_done(tmp_path):
    p = tmp_path / "run" / "episodes.jsonl"
    log = R.EpisodeLog(p, key=("condition", "pair"))
    for pair in range(3):
        log.append({"condition": "p0", "pair": pair, "success": True})
    again = R.EpisodeLog(p, key=("condition", "pair"))
    assert again.done == {("p0", 0), ("p0", 1), ("p0", 2)}
    todo = [("p0", i) for i in range(5)]
    assert [k for k in todo if k not in again.done] == [("p0", 3), ("p0", 4)]


def test_episode_log_drops_torn_last_line(tmp_path):
    p = tmp_path / "episodes.jsonl"
    log = R.EpisodeLog(p, key=("pair",))
    log.append({"pair": 0})
    log.append({"pair": 1})
    with open(p, "a") as fh:
        fh.write('{"pair": 2, "succ')  # pod killed mid-write
    again = R.EpisodeLog(p, key=("pair",))
    assert again.dropped_partial and again.done == {(0,), (1,)}
    again.append({"pair": 2})  # the rerun episode appends cleanly after the cut
    lines = p.read_text().splitlines()
    assert [json.loads(x)["pair"] for x in lines] == [0, 1, 2]


def test_episode_log_only_line_torn(tmp_path):
    p = tmp_path / "episodes.jsonl"
    p.write_text('{"pair": 0')
    log = R.EpisodeLog(p, key=("pair",))
    assert log.done == set() and p.read_bytes() == b""


def test_episode_log_refuses_duplicates(tmp_path):
    p = tmp_path / "episodes.jsonl"
    log = R.EpisodeLog(p, key=("pair",))
    log.append({"pair": 0})
    with pytest.raises(ValueError):
        log.append({"pair": 0})
    with open(p, "a") as fh:
        fh.write('{"pair": 0}\n')
    with pytest.raises(ValueError, match="duplicate"):
        R.EpisodeLog(p, key=("pair",))


def test_episode_log_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "episodes.jsonl"
    p.write_text('{"pair": 0}\n{"pair": 1, oops}\n{"pair": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        R.EpisodeLog(p, key=("pair",))
