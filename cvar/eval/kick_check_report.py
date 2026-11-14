"""Summarize the block-8 closed-loop kick check and apply the kick-size rule (numpy only; local).

    .venv-analysis/bin/python cvar/eval/kick_check_report.py runs/<RUN_ID>/kickcheck --out results/phase0/kickcheck

Per condition c (p0, p0.05-medium, p0.15-medium, p0.05-large, p0.15-large):
  success rate s_c = k/n with a 95% Wilson interval
      center = (s + z^2/(2n)) / (1 + z^2/n),  half = z sqrt(s(1-s)/n + z^2/(4n^2)) / (1 + z^2/n),  z = 1.96;
  fired_rate = sum kicks_fired / sum n_steps, full_rate = sum kicks_full / sum n_steps, with 95% CIs
      by bootstrap over episodes (ratio of sums).
Paired drop for kick size k, over the pairs i run in both conditions:
  d_k = mean_i [succ_i(p0) - succ_i(p0.15-k)],  95% CI by bootstrap over pairs (10,000 resamples).
Rule (confirmed by Rohan 2026-11-13): pick the smallest k in (medium, large) with d_k >= 0.10, the CI
of d_k excluding 0 (lower bound > 0) and s(p0.15-k) >= 0.30. If none qualifies: report and wait.
Also: env steps between replans (as logged), and drift.json (planner CRN check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIZES = ["medium", "large"]
Z = 1.959964
B = 10_000


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    s = k / n
    den = 1 + Z**2 / n
    c = (s + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(s * (1 - s) / n + Z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)


def ratio_ci(num: np.ndarray, den: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(num), (B, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != "p0", c))
    by = {c: {r["pair"]: r for r in rows if r["condition"] == c} for c in conds}
    cells = {}
    for c in conds:
        rs = list(by[c].values())
        k, n = sum(r["success"] for r in rs), len(rs)
        steps = np.array([r["n_steps"] for r in rs], float)
        fired = np.array([r["kicks_fired"] for r in rs], float)
        full = np.array([r["kicks_full"] for r in rs], float)
        between = sorted({x for r in rs for x in r["env_steps_between_replans"]})
        cells[c] = {"p": rs[0]["p"], "kick": rs[0]["kick"], "n": n, "successes": int(k), "success": k / n,
                    "success_wilson": wilson(k, n), "n_steps": int(steps.sum()),
                    "fired_rate": float(fired.sum() / steps.sum()), "fired_rate_ci": ratio_ci(fired, steps, rng),
                    "full_rate": float(full.sum() / steps.sum()), "full_rate_ci": ratio_ci(full, steps, rng),
                    "plan_calls_per_episode": sorted({r["plan_calls"] for r in rs}),
                    "env_steps_between_replans": between,
                    "plan_time_s_mean": float(np.mean([r["plan_time_s"] for r in rs if r["plan_time_s"] is not None]))}
    drops = {}
    for size in SIZES:
        for p in ("0.05", "0.15"):
            c = f"p{p}-{size}"
            if c not in by or "p0" not in by:
                continue
            common = sorted(set(by["p0"]) & set(by[c]))
            d = np.array([float(by["p0"][i]["success"]) - float(by[c][i]["success"]) for i in common])
            bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            drops[c] = {"n_pairs": len(common), "drop": float(d.mean()),
                        "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                        "pairs_lost": int((d > 0).sum()), "pairs_gained": int((d < 0).sum())}
    rule = []
    for size in SIZES:
        c = f"p0.15-{size}"
        if c not in drops:
            rule.append({"kick": size, "evaluated": False})
            continue
        dr, s = drops[c], cells[c]["success"]
        checks = {"drop_ge_0.10": dr["drop"] >= 0.10, "ci_excludes_0": dr["ci"][0] > 0, "success_p015_ge_0.30": s >= 0.30}
        rule.append({"kick": size, "evaluated": True, **checks, "qualifies": all(checks.values())})
    chosen = next((r["kick"] for r in rule if r.get("qualifies")), None)
    return {"cells": cells, "paired_drops": drops, "rule": rule, "chosen_kick": chosen,
            "verdict": f"use {chosen}" if chosen else "no kick size qualifies: report and wait"}


def table(s: dict) -> str:
    out = ["| condition | n | success [95% Wilson] | kicks/step: fired ; full [95% CI] | steps between replans |",
           "|---|---|---|---|---|"]
    for c, v in s["cells"].items():
        out.append(f"| {c} | {v['n']} | {v['success']:.3f} [{v['success_wilson'][0]:.3f}, {v['success_wilson'][1]:.3f}] "
                   f"| {v['fired_rate']:.3f} ; {v['full_rate']:.3f} [{v['full_rate_ci'][0]:.3f}, {v['full_rate_ci'][1]:.3f}] "
                   f"| {v['env_steps_between_replans']} |")
    out += ["", "| paired drop vs p0 | pairs | drop [95% CI over pairs] | pairs lost / gained |", "|---|---|---|---|"]
    for c, d in s["paired_drops"].items():
        out.append(f"| {c} | {d['n_pairs']} | {d['drop']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
                   f"| {d['pairs_lost']} / {d['pairs_gained']} |")
    out += ["", "Rule (drop >= 0.10, CI excludes 0, success at p0.15 >= 0.30): "
            + "; ".join(f"{r['kick']}: " + ("not run" if not r["evaluated"] else
                        ("QUALIFIES" if r["qualifies"] else "fails " + ", ".join(k for k in
                         ("drop_ge_0.10", "ci_excludes_0", "success_p015_ge_0.30") if not r[k]))) for r in s["rule"]),
            f"**Verdict: {s['verdict']}.**"]
    if s.get("drift"):
        out += ["", f"Planner CRN / drift ({s['drift']['device']}): " + json.dumps(s["drift"])]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kickcheck_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.kickcheck_dir)
    rows = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
    s = summarize(rows)
    if (d / "drift.json").exists():
        s["drift"] = json.loads((d / "drift.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(table(s))
    print(table(s))


if __name__ == "__main__":
    main()
