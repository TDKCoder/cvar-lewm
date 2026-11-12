"""Extract normalization stats + eval start/goal pairs from the LeWM PushT expert dataset.

Run once on a transient copy of pusht_expert_train.h5 (see scripts/fetch_expert_stats.sh), so
the 46 GB file never has to stay on disk. Reproduces upstream eval.py exactly:

  - scalers: sklearn StandardScaler fitted on the full raw column (f[col][:], NaN rows dropped)
    for action / proprio / state, as eval.py does (HDF5Dataset.get_col_data == f[col][:]).
      mean_k = (1/n) sum_i x_ik,   scale_k = sqrt((1/n) sum_i (x_ik - mean_k)^2)   (ddof=0;
      sklearn replaces scale 0 by 1)
  - start/goal sampling: rows whose step_idx <= ep_len - goal_offset - 1 are valid starts;
    g = default_rng(seed); g.choice(len(valid) - 1, size=n, replace=False); sort; the goal is
    the row goal_offset steps later in the same episode (World.evaluate -> load_chunk, frameskip 1).

Output (.npz): scaler_<col>_{mean,scale,var,n}; for each set S in {upstream, ours}:
S_episode, S_start, S_row_init, S_row_goal, S_init_<col>, S_goal_<col> for every per-row
non-image column; frames_{init,goal} (uint8, a few pairs of the upstream set) for a
render-equivalence test; meta (JSON string). A JSON summary (schema, sizes, stats) goes to --summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

try:  # the expert file may use Blosc/LZ4 filters
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


def episode_lengths(ep_col: np.ndarray, step_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized eval.py get_episodes_length: len(ep) = max(step_idx[ep_col == ep]) + 1."""
    ep_ids, inv = np.unique(ep_col, return_inverse=True)
    max_step = np.full(len(ep_ids), -1, dtype=np.int64)
    np.maximum.at(max_step, inv, step_idx.astype(np.int64))
    return ep_ids, max_step + 1


def sample_pairs(ep_col, step_idx, goal_offset: int, n: int, seed: int) -> np.ndarray:
    """Same arithmetic and RNG calls as upstream eval.py; returns sorted global start rows."""
    ep_ids, lengths = episode_lengths(ep_col, step_idx)
    max_start = dict(zip(ep_ids.tolist(), (lengths - goal_offset - 1).tolist()))
    max_start_per_row = np.array([max_start[e] for e in ep_col.tolist()])
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]
    g = np.random.default_rng(seed)
    pick = g.choice(len(valid_indices) - 1, size=n, replace=False)
    return np.sort(valid_indices[pick]), len(valid_indices)


def fit_scaler(x: np.ndarray) -> dict:
    x = x[~np.isnan(x).any(axis=1)].astype(np.float64)
    mean = x.mean(axis=0)
    var = x.var(axis=0)  # ddof=0, as StandardScaler
    scale = np.sqrt(var)
    scale[scale == 0.0] = 1.0
    out = {"mean": mean, "scale": scale, "var": var, "n": np.int64(len(x))}
    try:
        from sklearn.preprocessing import StandardScaler
        sk = StandardScaler().fit(x)
        out["sklearn_max_abs_diff"] = float(max(np.abs(sk.mean_ - mean).max(),
                                                np.abs(sk.scale_ - scale).max()))
    except ImportError:
        out["sklearn_max_abs_diff"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--n-ours", type=int, default=1000)
    ap.add_argument("--seed-ours", type=int, default=0)
    ap.add_argument("--n-upstream", type=int, default=50)
    ap.add_argument("--seed-upstream", type=int, default=42)
    ap.add_argument("--n-frames", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    f = h5py.File(a.h5, "r")
    schema = {}
    for k in f.keys():
        d = f[k]
        schema[k] = {"shape": list(d.shape), "dtype": str(d.dtype), "chunks": d.chunks and list(d.chunks),
                     "compression": d.compression, "filters": sorted(d._filters.keys())}
    print("schema:", json.dumps(schema, indent=1))
    attrs = {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in f.attrs.items()}

    ep_key = "episode_idx" if "episode_idx" in f else "ep_idx"
    ep_col = f[ep_key][:]
    step_idx = f["step_idx"][:]
    offsets = f["ep_offset"][:]
    ep_len_stored = f["ep_len"][:]
    n_rows = len(step_idx)

    scalers = {}
    for col in ("action", "proprio", "state"):
        scalers[col] = fit_scaler(f[col][:])
        print(f"scaler {col}: n={scalers[col]['n']} mean={scalers[col]['mean']} scale={scalers[col]['scale']} "
              f"sklearn_diff={scalers[col]['sklearn_max_abs_diff']}")

    # Per-row columns small enough to copy for each pair (no images).
    row_cols = [k for k in f.keys() if k not in ("ep_len", "ep_offset") and f[k].shape[0] == n_rows
                and f[k].ndim <= 2 and f[k].dtype.kind in "fiub"]
    npz = {}
    for col, s in scalers.items():
        for k in ("mean", "scale", "var", "n"):
            npz[f"scaler_{col}_{k}"] = s[k]

    sets = {}
    for name, n, seed in (("upstream", a.n_upstream, a.seed_upstream), ("ours", a.n_ours, a.seed_ours)):
        rows, n_valid = sample_pairs(ep_col, step_idx, a.goal_offset, n, seed)
        eps, starts = ep_col[rows], step_idx[rows]
        row_init = offsets[eps] + starts            # what load_chunk reads (ep id used as position)
        row_goal = row_init + a.goal_offset
        assert np.array_equal(row_init, rows), "ep ids are not positions in ep_offset"
        assert np.all(ep_col[row_goal] == eps), "goal row left the episode"
        npz.update({f"{name}_episode": eps, f"{name}_start": starts,
                    f"{name}_row_init": row_init, f"{name}_row_goal": row_goal})
        for col in row_cols:  # h5py fancy indexing needs increasing indices: rows are sorted
            npz[f"{name}_init_{col}"] = f[col][row_init]
            npz[f"{name}_goal_{col}"] = f[col][row_goal]
        sets[name] = {"n": n, "seed": seed, "n_valid_starts": int(n_valid),
                      "n_distinct_episodes": int(len(np.unique(eps)))}
        print(f"set {name}: {sets[name]}")

    if "pixels" in f and a.n_frames > 0:
        k = min(a.n_frames, a.n_upstream)
        npz["frames_init"] = np.stack([f["pixels"][r] for r in npz["upstream_row_init"][:k]])
        npz["frames_goal"] = np.stack([f["pixels"][r] for r in npz["upstream_row_goal"][:k]])

    ep_ids, lengths = episode_lengths(ep_col, step_idx)
    summary = {
        "source": "quentinll/lewm-pusht (dataset) pusht_expert_train.h5.zst",
        "h5_bytes": Path(a.h5).stat().st_size, "n_rows": int(n_rows), "n_episodes": int(len(ep_ids)),
        "ep_len": {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(lengths.max()),
                   "matches_stored_ep_len": bool(np.array_equal(np.sort(lengths), np.sort(ep_len_stored)))},
        "episode_key": ep_key, "row_cols_copied": row_cols, "attrs": attrs, "schema": schema,
        "goal_offset": a.goal_offset, "sets": sets,
        "scalers": {c: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()}
                    for c, s in scalers.items()},
        "hdf5plugin": hdf5plugin is not None, "elapsed_s": round(time.time() - t0, 1),
    }
    npz["meta"] = np.array(json.dumps({k: summary[k] for k in ("source", "goal_offset", "sets", "episode_key")}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **npz)
    Path(tmp).replace(a.out)
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size} B) and {a.summary} in {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()
