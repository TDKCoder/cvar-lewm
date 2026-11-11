#!/usr/bin/env python3
"""Aggregate per-episode eval logs into per-cell metrics with bootstrap CIs.

Usage:
    python aggregate.py "runs/eval-*" [more globs] --out results/<exp-name> [--by-seed]

Input: each matched run dir must contain episodes.jsonl (or pass .jsonl files directly).
Output (in --out): summary.csv, summary.md, manifest.json.

Metrics per cell (cell = all config fields except seed/episode, unless --by-seed):
  success_rate   = mean(success)                       95% percentile bootstrap CI
  worst10_cost   = mean of the top ceil(tail * n) final costs   (tail=0.1 by default), bootstrap CI
  failure_rate   = 1 - success_rate
  mean_final_cost, plan_time_mean (if logged)
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

REQUIRED = ["env", "p", "head", "cost", "seed", "episode", "success", "final_cost"]
CELL_FIELDS = ["env", "p", "head", "cost", "alpha", "beta", "S", "n_euler"]


def find_inputs(patterns: list[str]) -> list[str]:
    files = []
    for pat in patterns:
        for m in sorted(glob.glob(pat)):
            if os.path.isdir(m):
                f = os.path.join(m, "episodes.jsonl")
                if os.path.isfile(f):
                    files.append(f)
            elif m.endswith(".jsonl"):
                files.append(m)
    return sorted(set(files))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(files: list[str]) -> tuple[list[dict], list[dict]]:
    rows, manifest = [], []
    for f in files:
        n = 0
        with open(f) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"ERROR {f}:{lineno}: bad JSON ({e})")
                missing = [k for k in REQUIRED if k not in r]
                if missing:
                    sys.exit(f"ERROR {f}:{lineno}: missing keys {missing}")
                r["_file"] = f
                rows.append(r)
                n += 1
        manifest.append({"path": f, "episodes": n, "sha256": sha256(f)})
    return rows, manifest


def bootstrap_ci(values: np.ndarray, stat, n_boot: int, rng: np.random.Generator, level=0.95):
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat(values[idx])
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def top_k_mean_rows(mat: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest entries in each row (works for 1-D too)."""
    return np.sort(mat, axis=-1)[..., -k:].mean(axis=-1)


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return format(v, ".10g")
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--by-seed", action="store_true", help="keep seeds as separate cells")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail", type=float, default=0.1)
    args = ap.parse_args()

    files = find_inputs(args.patterns)
    if not files:
        sys.exit("ERROR: no episodes.jsonl found for " + " ".join(args.patterns))
    rows, manifest = load(files)

    fields = CELL_FIELDS + (["seed"] if args.by_seed else [])

    # Duplicate (cell, seed, episode) = a resumed run re-logged an episode. Refuse to guess.
    seen, dups = {}, []
    for r in rows:
        key = tuple(r.get(k) for k in CELL_FIELDS) + (r["seed"], r["episode"])
        if key in seen:
            dups.append((key, seen[key], r["_file"]))
        seen[key] = r["_file"]
    if dups:
        for key, a, b in dups[:10]:
            print(f"DUPLICATE {key}: {a} and {b}", file=sys.stderr)
        sys.exit(f"ERROR: {len(dups)} duplicate episodes; fix the logs (resume bug?) before aggregating")

    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r.get(k) for k in fields)].append(r)

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for key in sorted(cells, key=lambda t: tuple(fmt(x) for x in t)):
        rs = cells[key]
        n = len(rs)
        succ = np.array([1.0 if r["success"] else 0.0 for r in rs])
        cost = np.array([float(r["final_cost"]) for r in rs])
        k = max(1, math.ceil(args.tail * n))
        s_lo, s_hi = bootstrap_ci(succ, lambda m: m.mean(axis=-1), args.n_boot, rng)
        w_lo, w_hi = bootstrap_ci(cost, lambda m: top_k_mean_rows(m, k), args.n_boot, rng)
        pts = [float(r["plan_time_s"]) for r in rs if r.get("plan_time_s") is not None]
        row = dict(zip(fields, key))
        row.update(
            n=n,
            n_seeds=len({r["seed"] for r in rs}),
            successes=int(succ.sum()),
            success_rate=float(succ.mean()),
            success_lo=s_lo,
            success_hi=s_hi,
            failure_rate=float(1 - succ.mean()),
            worst10_k=k,
            worst10_cost=float(top_k_mean_rows(cost, k)),
            worst10_lo=w_lo,
            worst10_hi=w_hi,
            mean_final_cost=float(cost.mean()),
            plan_time_mean=float(np.mean(pts)) if pts else None,
        )
        out_rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    cols = list(out_rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in out_rows:
            w.writerow([fmt(r[c]) for c in cols])

    def pct(x):
        return f"{100 * x:.1f}"

    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        head = [f for f in fields] + ["n", "success % [95% CI]", "worst-10% cost [95% CI]", "plan s"]
        fh.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in out_rows:
            cells_txt = [fmt(r[f]) for f in fields] + [
                str(r["n"]),
                f"{pct(r['success_rate'])} [{pct(r['success_lo'])}, {pct(r['success_hi'])}]",
                f"{r['worst10_cost']:.4g} [{r['worst10_lo']:.4g}, {r['worst10_hi']:.4g}]",
                "" if r["plan_time_mean"] is None else f"{r['plan_time_mean']:.3g}",
            ]
            fh.write("| " + " | ".join(cells_txt) + " |\n")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(
            {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "args": vars(args),
                "numpy": np.__version__,
                "inputs": manifest,
                "total_episodes": len(rows),
                "cells": len(out_rows),
            },
            fh,
            indent=2,
        )

    print(f"{len(rows)} episodes from {len(files)} files -> {len(out_rows)} cells")
    print(open(os.path.join(args.out, "summary.md")).read())
    print(f"wrote {args.out}/summary.csv, summary.md, manifest.json")


if __name__ == "__main__":
    main()
