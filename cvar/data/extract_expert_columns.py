"""Keep every non-pixel column of the LeWM PushT expert dataset, so it never has to be downloaded again.

Run on the transient /tmp copy of pusht_expert_train.h5 (scripts/fetch_expert_stats.sh --columns,
decision 2026-11-13 "block 5 data plan"):

    python -m cvar.data.extract_expert_columns --h5 X.h5 --eval-npz expert_eval_states.npz \
        --out $STABLEWM_HOME/pusht/expert_columns.npz --summary $RUNS_DIR/phase0-expert-columns/summary.json

Output (.npz, np.savez_compressed, original dtypes):
  - every per-row column except pixels (action, proprio, state, episode_idx, step_idx) and the
    per-episode ep_offset / ep_len, copied whole;
  - excluded_episodes: sorted union of the episodes that hold an eval start/goal pair (upstream
    seed-42 n-50 set and our seed-0 n-1000 set). Training data (noisy-expert replays) never uses them;
  - eval_episode_rows / eval_pixels: the full pixel frames of the upstream-set episodes (already
    excluded from training), so a replay of expert actions in our env can be checked frame by frame
    against the expert's own renders;
  - meta (JSON string): source, h5 attrs, schema, sizes.
Checks before the atomic rename (the run fails if any is false):
  - each kept column equals the h5 column (np.array_equal on the reloaded npz);
  - scalers refit from the kept columns equal the stats in --eval-npz (max abs diff);
  - eval pairs re-sampled from the kept episode_idx/step_idx equal the rows in --eval-npz, i.e. the
    kept columns alone reproduce the eval states.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from cvar.data.extract_eval_states import fit_scaler, sample_pairs

try:  # pixels use the Blosc filter (32001)
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--eval-npz", required=True, help="expert_eval_states.npz (scalers + eval pairs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--pixel-set", default="upstream", help="eval set whose episodes keep full pixels ('' = none)")
    a = ap.parse_args()
    t0 = time.time()

    f = h5py.File(a.h5, "r")
    ev = np.load(a.eval_npz)
    n_rows = f["step_idx"].shape[0]
    keep = [k for k in f.keys() if k != "pixels"]
    schema = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in f.keys()}
    print("keeping:", {k: schema[k] for k in keep}, flush=True)
    npz = {k: f[k][:] for k in keep}
    ep_key = "episode_idx" if "episode_idx" in npz else "ep_idx"

    sets = sorted({k[: -len("_episode")] for k in ev.files if k.endswith("_episode")})
    excluded = np.unique(np.concatenate([ev[f"{s}_episode"] for s in sets]))
    npz["excluded_episodes"] = excluded
    print(f"excluded episodes: {len(excluded)} (union of eval sets {sets})", flush=True)

    pix_eps = np.unique(ev[f"{a.pixel_set}_episode"]) if a.pixel_set else np.array([], dtype=np.int64)
    rows = np.concatenate([np.arange(npz["ep_offset"][e], npz["ep_offset"][e] + npz["ep_len"][e]) for e in pix_eps]) \
        if len(pix_eps) else np.array([], dtype=np.int64)
    assert np.all(npz[ep_key][rows] == np.repeat(pix_eps, npz["ep_len"][pix_eps])), "ep_offset/ep_len disagree with episode_idx"
    npz["eval_episode_rows"] = rows
    npz["eval_pixels"] = f["pixels"][rows] if len(rows) else np.zeros((0, 224, 224, 3), np.uint8)
    print(f"pixels kept: {len(pix_eps)} episodes, {len(rows)} frames", flush=True)

    meta = {"source": "quentinll/lewm-pusht (dataset) pusht_expert_train.h5.zst", "n_rows": int(n_rows),
            "n_episodes": int(len(npz["ep_len"])), "schema": schema, "kept": keep,
            "attrs": {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in f.attrs.items()},
            "excluded_from_sets": sets, "pixel_set": a.pixel_set, "goal_offset": a.goal_offset}
    npz["meta"] = np.array(json.dumps(meta))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, **npz)

    # --- checks on the file as written ---
    z = np.load(tmp)
    checks = {f"equal_{k}": bool(np.array_equal(z[k], f[k][:], equal_nan=z[k].dtype.kind == "f")) for k in keep}
    checks["equal_eval_pixels"] = bool(np.array_equal(z["eval_pixels"], f["pixels"][rows])) if len(rows) else True
    scaler_diff = {}
    for col in ("action", "proprio", "state"):
        s = fit_scaler(z[col])
        scaler_diff[col] = float(max(np.abs(s["mean"] - ev[f"scaler_{col}_mean"]).max(),
                                     np.abs(s["scale"] - ev[f"scaler_{col}_scale"]).max()))
        checks[f"scaler_{col}_n_equal"] = int(s["n"]) == int(ev[f"scaler_{col}_n"])
        checks[f"scaler_{col}_close"] = scaler_diff[col] < 1e-9
    m = json.loads(str(ev["meta"]))
    for s in sets:
        rows_s, _ = sample_pairs(z[ep_key], z["step_idx"], a.goal_offset, m["sets"][s]["n"], m["sets"][s]["seed"])
        checks[f"resample_{s}_rows_equal"] = bool(np.array_equal(rows_s, ev[f"{s}_row_init"]))
    checks["excluded_cover_eval_rows"] = bool(np.isin(np.concatenate([z[ep_key][ev[f"{s}_row_init"]] for s in sets]),
                                                      z["excluded_episodes"]).all())
    ok = all(checks.values())
    summary = {**{k: v for k, v in meta.items() if k != "attrs"}, "attrs": meta["attrs"],
               "n_excluded_episodes": int(len(excluded)), "n_pixel_episodes": int(len(pix_eps)),
               "n_pixel_frames": int(len(rows)), "bytes": Path(tmp).stat().st_size,
               "array_bytes_uncompressed": {k: int(z[k].nbytes) for k in z.files},
               "scaler_max_abs_diff": scaler_diff, "checks": checks, "all_ok": ok,
               "hdf5plugin": hdf5plugin is not None, "elapsed_s": round(time.time() - t0, 1)}
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps(checks, indent=1))
    if not ok:
        Path(tmp).unlink()
        raise SystemExit(f"checks failed; nothing written to {a.out} (summary: {a.summary})")
    Path(tmp).replace(a.out)
    print(f"wrote {a.out} ({summary['bytes']} B) in {summary['elapsed_s']} s; COLUMNS_OK")


if __name__ == "__main__":
    main()
