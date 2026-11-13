"""Independent check of expert_eval_states.npz against the real pusht_expert_train.h5.

Shares no code with extract_eval_states.py. It runs upstream eval.py's own code paths, copied
verbatim (upstream 8edfeb3, eval.py:29-37, 68-79, 109-129), on a duck-typed dataset that serves
raw h5 columns, which is what swm's HDF5Dataset.get_col_data / get_row_data return:

  1. scalers: sklearn StandardScaler fitted exactly as eval.py:75-78 (raw dtype, NaN rows dropped)
     vs the npz's scaler_<col>_{mean,scale,var,n}.
  2. pairs: eval.py's literal per-episode loop + sampling for (seed 42, n 50) and (seed 0, n 1000)
     vs the npz's <set>_{row_init,episode,start}; also ep lengths vs the stored ep_len column.
  3. rows: the npz's global rows re-read from the h5 (episode_idx, step_idx, state, ...): the goal
     row is init + 25 in the same episode with step_idx + 25, and every stored init/goal column
     equals the h5 row (and frames_init/goal equal pixels at those rows if hdf5plugin is there).
Writes a JSON report; exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import h5py
import numpy as np
from sklearn import preprocessing

try:
    import hdf5plugin  # noqa: F401  (pixels use Blosc)
except ImportError:
    hdf5plugin = None


class H5Cols:
    """The slice of swm.data.HDF5Dataset that eval.py's sampling and scalers touch."""

    def __init__(self, f: h5py.File):
        self.f = f
        self.column_names = list(f.keys())
        self._cache: dict[str, np.ndarray] = {}

    def get_col_data(self, col: str) -> np.ndarray:
        if col not in self._cache:
            self._cache[col] = self.f[col][:]
        return self._cache[col]

    def get_row_data(self, rows) -> dict:
        return {c: self.get_col_data(c)[rows] for c in ("episode_idx", "ep_idx", "step_idx") if c in self.column_names}


# ---------------------------------------------------------------- verbatim from eval.py:29-37
def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def eval_py_scalers(stats_dataset, keys=("action", "proprio", "state")):
    # eval.py:71-79 (keys_to_cache minus pixels)
    process = {}
    for col in keys:
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
    return process


def eval_py_sampling(dataset, ep_indices, episode_len, seed, num_eval, goal_offset_steps):
    # eval.py:110-129 with cfg.seed / cfg.eval.num_eval / cfg.eval.goal_offset_steps as arguments
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    return random_episode_indices, eval_episodes, eval_start_idx, int(valid_mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal-offset", type=int, default=25)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.npz)
    f = h5py.File(a.h5, "r")
    ds = H5Cols(f)
    rep: dict = {"npz": a.npz, "checks": {}, "details": {}}
    chk, det = rep["checks"], rep["details"]

    # 1. scalers
    for col, sk in eval_py_scalers(ds).items():
        d = {
            "n": [int(sk.n_samples_seen_), int(z[f"scaler_{col}_n"])],
            "max_abs_diff_mean": float(np.abs(sk.mean_ - z[f"scaler_{col}_mean"]).max()),
            "max_abs_diff_scale": float(np.abs(sk.scale_ - z[f"scaler_{col}_scale"]).max()),
            "max_rel_diff_scale": float((np.abs(sk.scale_ - z[f"scaler_{col}_scale"]) / sk.scale_).max()),
            "max_abs_diff_var": float(np.abs(sk.var_ - z[f"scaler_{col}_var"]).max()),
            "input_dtype": str(ds.get_col_data(col).dtype),
            "sklearn_mean": sk.mean_.tolist(), "sklearn_scale": sk.scale_.tolist(),
        }
        det[f"scaler_{col}"] = d
        chk[f"scaler_{col}_n_equal"] = d["n"][0] == d["n"][1]
        chk[f"scaler_{col}_close"] = d["max_abs_diff_mean"] <= 1e-6 * max(1.0, np.abs(sk.mean_).max()) \
            and d["max_rel_diff_scale"] <= 1e-6
        print(f"scaler {col}: {d}", flush=True)

    # 2. pairs, with eval.py's literal per-episode length loop (slow but exact)
    col_name = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_indices, _ = np.unique(ds.get_col_data(col_name), return_index=True)
    t1 = time.time()
    episode_len = get_episodes_length(ds, ep_indices)
    det["episode_len_loop_s"] = round(time.time() - t1, 1)
    chk["ep_len_matches_stored_column"] = bool(np.array_equal(episode_len, f["ep_len"][:][ep_indices])) \
        if len(f["ep_len"]) == len(ep_indices) else False
    for name, seed, n in (("upstream", 42, 50), ("ours", 0, 1000)):
        rows, eps, starts, n_valid = eval_py_sampling(ds, ep_indices, episode_len, seed, n, a.goal_offset)
        det[f"{name}_n_valid"] = n_valid
        chk[f"{name}_rows_equal"] = bool(np.array_equal(rows, z[f"{name}_row_init"]))
        chk[f"{name}_episodes_equal"] = bool(np.array_equal(eps, z[f"{name}_episode"]))
        chk[f"{name}_starts_equal"] = bool(np.array_equal(starts, z[f"{name}_start"]))
        print(f"{name}: rows/eps/starts equal = {chk[f'{name}_rows_equal']}, {chk[f'{name}_episodes_equal']}, "
              f"{chk[f'{name}_starts_equal']}; first rows {rows[:5].tolist()}", flush=True)

        # 3. rows re-read from the h5
        ri, rg = z[f"{name}_row_init"], z[f"{name}_row_goal"]
        e, s = ds.get_col_data(col_name), ds.get_col_data("step_idx")
        chk[f"{name}_goal_same_episode"] = bool(np.all(e[rg] == e[ri]))
        chk[f"{name}_goal_step_plus_offset"] = bool(np.all(s[rg] == s[ri] + a.goal_offset))
        bad = []
        for key in z.files:
            for side, r in (("init", ri), ("goal", rg)):
                pre = f"{name}_{side}_"
                if key.startswith(pre) and key[len(pre):] in f:
                    if not np.array_equal(z[key], ds.get_col_data(key[len(pre):])[r], equal_nan=True):
                        bad.append(key)
        chk[f"{name}_stored_columns_equal_h5"] = not bad
        det[f"{name}_mismatched_columns"] = bad

    if hdf5plugin is not None and "frames_init" in z.files:
        k = len(z["frames_init"])
        chk["frames_equal_h5"] = bool(
            np.array_equal(z["frames_init"], np.stack([f["pixels"][r] for r in z["upstream_row_init"][:k]]))
            and np.array_equal(z["frames_goal"], np.stack([f["pixels"][r] for r in z["upstream_row_goal"][:k]])))

    rep["all_ok"] = all(chk.values())
    rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["versions"] = {"numpy": np.__version__, "h5py": h5py.__version__,
                       "sklearn": sys.modules["sklearn"].__version__, "hdf5plugin": hdf5plugin is not None}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(chk, indent=1))
    print("VERIFY_OK" if rep["all_ok"] else "VERIFY_FAILED")
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
