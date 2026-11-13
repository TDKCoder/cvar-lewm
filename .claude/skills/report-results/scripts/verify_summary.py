#!/usr/bin/env python3
"""Independently re-verify an aggregate summary.csv against the raw episode logs.

Deliberately shares no code with aggregate.py and uses only the standard library, so a bug in
one is unlikely to be repeated in the other.

Usage:
    python verify_summary.py "runs/eval-*" [more globs] results/<exp-name>/summary.csv

Hard checks (exit 1 on failure):
  - same set of cells; same n and success count per cell
  - success_rate and worst-10% cost recomputed exactly (rel. tol 1e-9)
  - bootstrap CI contains the point estimate
  - no duplicate (cell, seed, episode)
  - PushT kick rates: fired_rate and full_rate = sums of kicks_fired / kicks_full over sum(n_steps)
    recomputed exactly; full_rate CI contains its point estimate
Soft checks (warning only): bootstrap CI vs Wilson score interval differ by > 0.05; fired_rate more
than 3 binomial SEs from p (u_t ~ U[0,1) i.i.d. per step, so fired ~ Binomial(n_steps, p)).
"""
import csv
import glob
import json
import math
import os
import sys

CELL = ["env", "p", "head", "cost", "alpha", "beta", "S", "n_euler"]


def canon(v):
    """Canonical text for a key value, so 0.15 / '0.15' / 1.5e-1 all match."""
    if v is None or v == "":
        return ""
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    *patterns, summary_path = sys.argv[1:]

    with open(summary_path, newline="") as fh:
        summary = list(csv.DictReader(fh))
    by_seed = "seed" in (summary[0].keys() if summary else [])
    fields = CELL + (["seed"] if by_seed else [])

    files = []
    for pat in patterns:
        for m in glob.glob(pat):
            if os.path.isdir(m) and os.path.isfile(os.path.join(m, "episodes.jsonl")):
                files.append(os.path.join(m, "episodes.jsonl"))
            elif m.endswith(".jsonl"):
                files.append(m)
    files = sorted(set(files))

    raw = {}
    kicks = {}
    seen = set()
    problems = []
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            uniq = tuple(canon(r.get(k)) for k in CELL) + (canon(r["seed"]), canon(r["episode"]))
            if uniq in seen:
                problems.append(f"duplicate episode {uniq} in {f}")
            seen.add(uniq)
            key = tuple(canon(r.get(k)) for k in fields)
            raw.setdefault(key, []).append((bool(r["success"]), float(r["final_cost"])))
            if all(r.get(k) is not None for k in ("n_steps", "kicks_fired", "kicks_full")):
                kick = kicks.setdefault(key, [0, 0, 0, float(r["p"])])
                kick[0] += int(r["n_steps"]); kick[1] += int(r["kicks_fired"]); kick[2] += int(r["kicks_full"])

    rows = {tuple(canon(row.get(k)) for k in fields): row for row in summary}

    if set(raw) != set(rows):
        problems.append(f"cell sets differ: only in raw {sorted(set(raw) - set(rows))}, "
                        f"only in summary {sorted(set(rows) - set(raw))}")

    warnings = []
    print(f"{'cell':60s} {'n':>5s} {'succ%':>7s} {'wilson95':>17s} {'boot95':>17s} {'worst10':>10s}  check")
    for key in sorted(set(raw) & set(rows)):
        eps = raw[key]
        row = rows[key]
        n = len(eps)
        s = sum(1 for ok, _ in eps if ok)
        rate = s / n
        costs = sorted((c for _, c in eps), reverse=True)
        k = max(1, math.ceil(0.1 * n))
        worst = sum(costs[:k]) / k
        wl, wh = wilson(s, n)
        bl, bh = float(row["success_lo"]), float(row["success_hi"])
        ok = True
        if int(row["n"]) != n or int(row["successes"]) != s:
            problems.append(f"{key}: n/successes {row['n']}/{row['successes']} vs raw {n}/{s}"); ok = False
        if not close(float(row["success_rate"]), rate):
            problems.append(f"{key}: success_rate {row['success_rate']} vs raw {rate}"); ok = False
        if int(row["worst10_k"]) != k or not close(float(row["worst10_cost"]), worst):
            problems.append(f"{key}: worst10 {row['worst10_cost']} (k={row['worst10_k']}) vs raw {worst} (k={k})"); ok = False
        if not (bl - 1e-12 <= rate <= bh + 1e-12):
            problems.append(f"{key}: bootstrap CI [{bl}, {bh}] excludes the point estimate {rate}"); ok = False
        if abs(bl - wl) > 0.05 or abs(bh - wh) > 0.05:
            warnings.append(f"{key}: bootstrap [{bl:.3f},{bh:.3f}] vs Wilson [{wl:.3f},{wh:.3f}] (small n or rate near 0/1?)")
        if key in kicks:
            steps, fired, full, p = kicks[key]
            if not (close(float(row["fired_rate"]), fired / steps) and close(float(row["full_rate"]), full / steps)):
                problems.append(f"{key}: kick rates {row['fired_rate']}/{row['full_rate']} vs raw "
                                f"{fired / steps}/{full / steps}"); ok = False
            if not (float(row["full_rate_lo"]) - 1e-12 <= full / steps <= float(row["full_rate_hi"]) + 1e-12):
                problems.append(f"{key}: full_rate CI excludes its point estimate"); ok = False
            se = math.sqrt(p * (1 - p) / steps) if 0 < p < 1 else 0.0
            if abs(fired / steps - p) > 3 * se + 1e-12:
                warnings.append(f"{key}: fired_rate {fired / steps:.4f} is > 3 SE from p={p} (n_steps={steps})")
            print(f"    kicks: n_steps={steps} fired={fired} ({fired / steps:.4f}/step, p={p}) "
                  f"full={full} ({full / steps:.4f}/step, {full / max(fired, 1):.3f} of fired)")
        elif key and canon("pusht") == key[0]:
            problems.append(f"{key}: pusht cell without n_steps/kicks_fired/kicks_full"); ok = False
        label = "/".join(x for x in key if x)
        print(f"{label[:60]:60s} {n:5d} {100*rate:7.1f} [{wl:.3f}, {wh:.3f}] [{bl:.3f}, {bh:.3f}] {worst:10.4g}  {'ok' if ok else 'MISMATCH'}")

    for w in warnings:
        print("WARN", w)
    if problems:
        for p in problems:
            print("FAIL", p)
        print(f"VERIFY: FAIL ({len(problems)} problems) for {summary_path}")
        sys.exit(1)
    print(f"VERIFY: PASS - {len(raw)} cells, {sum(len(v) for v in raw.values())} episodes, {len(files)} files")


if __name__ == "__main__":
    main()
