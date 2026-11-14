"""extract_eval_states -> verify_eval_states round trip on a synthetic expert-style h5."""
import json
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("sklearn")


def make_h5(path, n_eps=40, seed=0, pixels=False):
    rng = np.random.default_rng(seed)
    lens = rng.integers(20, 120, n_eps)  # some shorter than goal_offset + 1: no valid start
    ep = np.repeat(np.arange(n_eps), lens)
    step = np.concatenate([np.arange(n) for n in lens])
    n = len(ep)
    with h5py.File(path, "w") as f:
        f["episode_idx"] = ep
        f["step_idx"] = step
        f["ep_len"] = lens
        f["ep_offset"] = np.concatenate([[0], np.cumsum(lens)[:-1]])
        f["action"] = rng.normal(size=(n, 2)).astype(np.float32)
        f["action"][5] = np.nan  # eval.py drops NaN rows before fitting
        f["proprio"] = rng.normal(3, 2, size=(n, 4)).astype(np.float32)
        f["state"] = rng.normal(100, 50, size=(n, 7)).astype(np.float32)
        if pixels:
            f["pixels"] = rng.integers(0, 256, size=(n, 4, 4, 3), dtype=np.uint8)
    return n


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)


def test_extract_then_verify_roundtrip(tmp_path):
    h5 = tmp_path / "d.h5"
    make_h5(h5)
    npz, summ, rep = tmp_path / "s.npz", tmp_path / "s.json", tmp_path / "v.json"
    # defaults on both sides: upstream (seed 42, n 50), ours (seed 0, n 1000); ~1.8k valid starts
    r = run("cvar.data.extract_eval_states", "--h5", str(h5), "--out", str(npz), "--summary", str(summ),
            "--n-frames", "0")
    assert r.returncode == 0, r.stderr
    r = run("cvar.data.verify_eval_states", "--h5", str(h5), "--npz", str(npz), "--out", str(rep))
    out = json.loads(rep.read_text())
    assert r.returncode == 0 and out["all_ok"], json.dumps(out["checks"], indent=1)


def test_verify_catches_a_wrong_row(tmp_path):
    h5 = tmp_path / "d.h5"
    make_h5(h5, n_eps=60)
    npz, summ, rep = tmp_path / "s.npz", tmp_path / "s.json", tmp_path / "v.json"
    assert run("cvar.data.extract_eval_states", "--h5", str(h5), "--out", str(npz), "--summary", str(summ),
               "--n-frames", "0").returncode == 0
    z = dict(np.load(npz))
    z["upstream_row_init"] = z["upstream_row_init"].copy()
    z["upstream_row_init"][3] += 1
    np.savez(npz, **z)
    r = run("cvar.data.verify_eval_states", "--h5", str(h5), "--npz", str(npz), "--out", str(rep))
    out = json.loads(rep.read_text())
    assert r.returncode == 1 and not out["checks"]["upstream_rows_equal"]


def test_expert_columns_keep_everything_but_pixels(tmp_path):
    h5 = tmp_path / "d.h5"
    make_h5(h5, pixels=True)
    npz, summ = tmp_path / "s.npz", tmp_path / "s.json"
    assert run("cvar.data.extract_eval_states", "--h5", str(h5), "--out", str(npz), "--summary", str(summ),
               "--n-frames", "0").returncode == 0
    cols, csum = tmp_path / "c.npz", tmp_path / "c.json"
    r = run("cvar.data.extract_expert_columns", "--h5", str(h5), "--eval-npz", str(npz), "--out", str(cols),
            "--summary", str(csum))
    out = json.loads(csum.read_text())
    assert r.returncode == 0 and out["all_ok"], r.stderr + json.dumps(out["checks"], indent=1)
    z, ev = np.load(cols), np.load(npz)
    with h5py.File(h5) as f:
        for k in ("action", "proprio", "state", "episode_idx", "step_idx", "ep_len", "ep_offset"):
            assert np.array_equal(z[k], f[k][:], equal_nan=z[k].dtype.kind == "f"), k
        assert np.array_equal(z["eval_pixels"], f["pixels"][z["eval_episode_rows"]])
    assert "pixels" not in z.files
    assert set(np.unique(np.concatenate([ev["upstream_episode"], ev["ours_episode"]]))) == set(z["excluded_episodes"])
    assert set(np.unique(z["episode_idx"][z["eval_episode_rows"]])) == set(ev["upstream_episode"])
