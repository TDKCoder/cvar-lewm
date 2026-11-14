"""Summarize + plot the block-6 spread study (runs locally, .venv-analysis: numpy + matplotlib).

    .venv-analysis/bin/python cvar/envs/plot_spread.py runs/phase0-spread --out results/phase0/spread
(a plain script: the cvar.envs package imports gymnasium, which the analysis venv doesn't have)

Per cell (kick config, physics, p), from rollouts.jsonl:
  success_any = mean over rollouts of eval.py's success (terminated at any step), 95% CI by bootstrap
      over PLANS (rollouts of one plan share its start/goal, so plans are the independent unit);
  block_pos_err median / p90 (px), block_ang_err median (deg);
  fired_rate = sum kicks_fired / sum n_steps,  full_rate = sum kicks_full / sum n_steps, with 95% CIs
      by bootstrap over rollouts (ratio of sums, as report-results' aggregate.py);
  replay_block_err_max at p = 0: how closely a plain replay tracks the expert's own states.
Figures: spread_scatter.png (final block xy minus goal block xy, all plans, rows = kick config,
cols = p) and spread_ecdf.png (ECDF of final block position error per p, one panel per config).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

P_COLORS = {0.0: "#86b6ef", 0.05: "#2a78d6", 0.15: "#104281"}  # ordinal blue ramp, steps 250/450/650
P_MARKERS = {0.0: "o", 0.05: "s", 0.15: "^"}                     # secondary encoding: not color alone
B = 2000


def boot(values: np.ndarray, groups: np.ndarray | None, rng, stat=np.mean):
    if groups is None:
        idx = rng.integers(0, len(values), (B, len(values)))
        bs = np.array([stat(values[i]) for i in idx])
    else:
        ug = np.unique(groups)
        by = [values[groups == g] for g in ug]
        bs = np.array([stat(np.concatenate([by[j] for j in rng.integers(0, len(ug), len(ug))])) for _ in range(B)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ratio_ci(num, den, rng):
    idx = rng.integers(0, len(num), (B, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summarize(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(0)
    cells = {}
    for r in rows:
        cells.setdefault((r["config"], r["physics"], r["p"]), []).append(r)
    out = []
    for (cfg, phys, p), rs in cells.items():
        g = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        succ, plan = g("success_any"), g("plan")
        n_steps, fired, full = g("n_steps"), g("kicks_fired"), g("kicks_full")
        out.append({
            "config": cfg, "physics": phys, "p": p, "n_rollouts": len(rs), "n_plans": int(len(np.unique(plan))),
            "success_any": float(succ.mean()), "success_any_ci": boot(succ, plan, rng),
            "success_final": float(g("success_final").mean()),
            "block_pos_err_median": float(np.median(g("block_pos_err"))),
            "block_pos_err_p90": float(np.percentile(g("block_pos_err"), 90)),
            "block_ang_err_median_deg": float(np.degrees(np.median(g("block_ang_err")))),
            "n_steps": int(n_steps.sum()), "kicks_fired": int(fired.sum()), "kicks_full": int(full.sum()),
            "kicks_partial": int(g("kicks_partial").sum()), "kicks_vetoed": int(g("kicks_vetoed").sum()),
            "fired_rate": float(fired.sum() / n_steps.sum()), "fired_rate_ci": ratio_ci(fired, n_steps, rng),
            "full_rate": float(full.sum() / n_steps.sum()), "full_rate_ci": ratio_ci(full, n_steps, rng),
            "replay_block_err_max_median": float(np.median(g("replay_block_err_max"))),
        })
    order = {"small": 0, "medium": 1, "large": 2}
    return sorted(out, key=lambda c: (c["physics"], order.get(c["config"], 9), c["p"]))


def table(cells: list[dict]) -> str:
    h = ("| kicks | physics | p | rollouts (plans) | open-loop success [95% CI] | block err median / p90 (px) "
         "| angle err median (deg) | kicks/step: fired ; full [95% CI] |\n|---|---|---|---|---|---|---|---|\n")
    for c in cells:
        h += (f"| {c['config']} | {'on' if c['physics'] else 'off'} | {c['p']:.2f} | {c['n_rollouts']} ({c['n_plans']}) "
              f"| {c['success_any']:.3f} [{c['success_any_ci'][0]:.3f}, {c['success_any_ci'][1]:.3f}] "
              f"| {c['block_pos_err_median']:.1f} / {c['block_pos_err_p90']:.1f} | {c['block_ang_err_median_deg']:.1f} "
              f"| {c['fired_rate']:.3f} ; {c['full_rate']:.3f} [{c['full_rate_ci'][0]:.3f}, {c['full_rate_ci'][1]:.3f}] |\n")
    return h


def plots(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    configs, ps = ["small", "medium", "large"], [0.0, 0.05, 0.15]
    sel = lambda cfg, phys, p: [r for r in rows if r["config"] == cfg and r["physics"] == phys and r["p"] == p]  # noqa: E731

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
    lim = 150.0  # a few large-kick outliers fall outside; counted in the summary table
    for i, cfg in enumerate(configs):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            rs = sel(cfg, False, p)
            d = np.array([[r["final_state"][2] - r["goal_state"][2], r["final_state"][3] - r["goal_state"][3]] for r in rs])
            ok = np.array([r["success_any"] for r in rs])
            ax.add_patch(plt.Circle((0, 0), 20, fill=False, ls="--", lw=1, color="#8a8984"))
            ax.scatter(d[ok, 0], d[ok, 1], s=8, marker=P_MARKERS[p], facecolor="none", edgecolor=P_COLORS[p], lw=0.5, alpha=0.5)
            ax.scatter(d[~ok, 0], d[~ok, 1], s=8, marker=P_MARKERS[p], color=P_COLORS[p], alpha=0.35, lw=0)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_title(f"{cfg} kicks, p = {p:.2f}  (success {ok.mean():.2f})", fontsize=9, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel("final block y - goal y (px)")
            if i == 2:
                ax.set_xlabel("final block x - goal x (px)")
    fig.suptitle("Expert's 25-step plan replayed open-loop: final block position vs goal\n"
                 f"{len(np.unique([r['plan'] for r in rows]))} upstream eval pairs x "
                 f"{len(rs) // max(1, len(np.unique([r['plan'] for r in rs])))} kick seeds; "
                 "dashed circle = 20 px; hollow = success, filled = failure; axes clipped at 150 px", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "spread_scatter.png", dpi=150)
    plt.close(fig)

    panels = [(c, False) for c in configs] + [("medium", True)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, (cfg, phys) in zip(axes, panels):
        for p in ps:
            rs = sel(cfg, phys, p)
            if not rs:
                continue
            e = np.sort([r["block_pos_err"] for r in rs])
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", lw=2, color=P_COLORS[p], label=f"p = {p:.2f}")
        ax.axvline(20, ls="--", lw=1, color="#8a8984")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, 300)
        ax.set_title(f"{cfg} kicks" + (", physics on" if phys else ""), fontsize=9, color="#0b0b0b")
        ax.set_xlabel("final block position error (px, symlog)")
    axes[0].set_ylabel("fraction of rollouts <= x")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "spread_ecdf.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines()]
    meta = json.loads((run / "meta.json").read_text())
    cells = summarize(rows)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "cells": cells}, indent=1))
    (out / "summary.md").write_text(table(cells))
    print(table(cells))
    plots(rows, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
