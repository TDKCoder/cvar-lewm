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
  PushT kicks (keys n_steps, kicks_fired, kicks_full required when env == "pusht"):
    fired_rate = sum(kicks_fired) / sum(n_steps)   (u_t < p; policy-independent, should match p)
    full_rate  = sum(kicks_full)  / sum(n_steps)   (applied at full size; depends on the policy,
                 because kicks into the agent/walls are scaled back or vetoed). 95% CI: bootstrap
                 over episodes of the ratio of sums.
  Cross-head check -> kick_check.md: within each group of cells that differ only in `head`, the
  heads' full_rate CIs must overlap; otherwise the realized noise level differs between heads and
  is a confound for that comparison (printed as KICK-RATE MISMATCH, not a hard error).
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
KICK_KEYS = ["n_steps", "kicks_fired", "kicks_full"]  # required for env == "pusht"


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
                if r.get("env") == "pusht":
                    missing += [k for k in KICK_KEYS if k not in r]
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


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int, rng: np.random.Generator, level=0.95):
    """Bootstrap over episodes of sum(num) / sum(den)."""
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    boots = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
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
        if all(r.get(k) is not None for r in rs for k in KICK_KEYS):
            steps = np.array([float(r["n_steps"]) for r in rs])
            fired = np.array([float(r["kicks_fired"]) for r in rs])
            full = np.array([float(r["kicks_full"]) for r in rs])
            f_lo, f_hi = ratio_ci(full, steps, args.n_boot, rng)
            row.update(n_steps=int(steps.sum()), kicks_fired=int(fired.sum()), kicks_full=int(full.sum()),
                       fired_rate=float(fired.sum() / steps.sum()), full_rate=float(full.sum() / steps.sum()),
                       full_rate_lo=f_lo, full_rate_hi=f_hi)
        else:
            row.update(n_steps=None, kicks_fired=None, kicks_full=None, fired_rate=None, full_rate=None,
                       full_rate_lo=None, full_rate_hi=None)
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
        head = [f for f in fields] + ["n", "success % [95% CI]", "worst-10% cost [95% CI]", "plan s",
                                      "kicks/step: fired ; full [95% CI]"]
        fh.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in out_rows:
            cells_txt = [fmt(r[f]) for f in fields] + [
                str(r["n"]),
                f"{pct(r['success_rate'])} [{pct(r['success_lo'])}, {pct(r['success_hi'])}]",
                f"{r['worst10_cost']:.4g} [{r['worst10_lo']:.4g}, {r['worst10_hi']:.4g}]",
                "" if r["plan_time_mean"] is None else f"{r['plan_time_mean']:.3g}",
                "" if r["full_rate"] is None else
                f"{r['fired_rate']:.4f} ; {r['full_rate']:.4f} [{r['full_rate_lo']:.4f}, {r['full_rate_hi']:.4f}]",
            ]
            fh.write("| " + " | ".join(cells_txt) + " |\n")

    # Cross-head comparability of the realized (full-size) kick rate.
    head_i = fields.index("head")
    groups = defaultdict(list)
    for r in out_rows:
        if r["full_rate"] is not None:
            groups[tuple(r[f] for i, f in enumerate(fields) if i != head_i)].append(r)
    lines = ["| cell without head | head: full_rate [95% CI] | max/min | CIs overlap |", "|---|---|---|---|"]
    mismatches = 0
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(fmt(x) for x in kv[0])):
        if len({r["head"] for r in rs}) < 2:
            continue
        overlap = max(r["full_rate_lo"] for r in rs) <= min(r["full_rate_hi"] for r in rs)
        rates = [r["full_rate"] for r in rs]
        ratio = max(rates) / min(rates) if min(rates) > 0 else (1.0 if max(rates) == 0 else float("inf"))
        mismatches += not overlap
        lines.append("| " + "/".join(fmt(x) for x in g if x is not None) + " | "
                     + "; ".join(f"{r['head']}: {r['full_rate']:.4f} [{r['full_rate_lo']:.4f}, {r['full_rate_hi']:.4f}]"
                                 for r in rs) + f" | {ratio:.3g} | {'yes' if overlap else 'NO'} |")
    with open(os.path.join(args.out, "kick_check.md"), "w") as fh:
        fh.write("Realized full-size kick rate per step, compared across heads (cells differing only in head).\n\n")
        fh.write("\n".join(lines) + "\n")

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
    if len(lines) > 2:
        print(open(os.path.join(args.out, "kick_check.md")).read())
    if mismatches:
        print(f"KICK-RATE MISMATCH: {mismatches} group(s) where heads' full-size kick-rate CIs don't overlap "
              f"(see kick_check.md); the heads faced different realized noise there")
    print(f"wrote {args.out}/summary.csv, summary.md, kick_check.md, manifest.json")


if __name__ == "__main__":
    main()
