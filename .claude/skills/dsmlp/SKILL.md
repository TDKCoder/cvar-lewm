---
name: dsmlp
description: UCSD DSMLP cluster facts for this project - login node rules, launch.sh flags, GPU classes, Service Unit (SU) budget and rates, runtime limits, idle culling, preemption, storage quotas and paths. Use whenever a question or plan depends on how DSMLP behaves (GPU choice, job length, cost, disk space, why a pod died).
---

# DSMLP quick facts

Source: dsmlp-docs.ucsd.edu, read 2026-11-09. Those pages are marked **draft**, and the Fall 2026
GPU-reservation figures are "not yet published". Treat every number here as provisional and point
the user to the source page (`reference.md`) when a decision depends on it.

## Our course workspace (from the Datahub launcher, 2026-11-09)

- Workspace label: `ECE285_FA26_002 - Qi Cao [FA26]` (exact `-W` id: see `scripts/dsmlp.conf`).
- Presets: `scipy-ml-notebook:2025.1-stable` with 8 CPU / 16 GB and 0 GPU, 1 GPU, or 1 GPU `[medium]`;
  also `datascience-notebook` and `rstudio-notebook` (2 CPU / 8 GB, no GPU).
- **GPU class granted: medium (24 GB).** `small` isn't offered, so don't plan on it.
- We pin `IMAGE=ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable` for both the worker pod and GPU
  jobs so the venv works in both. Keep pods at <= 16 GB RAM to match the presets.

## Verified on our account (2026-11-10)

- Over a one-off `ssh dsmlp <cmd>`, `kubectl` and `workspace` fail (kubectl falls back to
  localhost:8080). They only work in a login shell, so our scripts run everything through
  `bash -lc`. `rsync` (`/usr/bin/rsync`) works either way.
- Login-node `~` = `/dsmlp/home-fs03/03/003/rnambimadom` = the pod's `~/private` (personal pool).
- Pod `$HOME` with `-W ECE285_FA26_002` = `/dsmlp/workspaces-fs05/ECE285_FA26_002/home/rnambimadom`.
- `launch-scipy-ml.sh` without `-i` uses `scipy-ml-notebook:stable`; our scripts pin
  `2025.1-stable` to match the course presets.

## Where code runs (laptop mode)

| Where | What it's for | Cost |
|---|---|---|
| Rohan's laptop (WSL2) | **you (Claude)**, editing, git, aggregation and plots in `.venv-analysis` | free |
| Login node `dsmlp-login.ucsd.edu` | only reached through our scripts: rsync, `kubectl exec/get/delete`, `launch.sh`, snapshot copies | free; **running Python, builds, training or analysis there is prohibited** |
| CPU worker pod (`cvl-worker`) | everything that needs project deps: installs, tests, smoke runs, data collection | free; not idle-culled; ends at 12 h |
| GPU pod | encoding, training, planning evals | Service Units, charged from launch |

**Pod network:** pods can't reach api.anthropic.com (tested 2026-11-10), which is why you run on the
laptop. What else pods can reach (PyPI, Hugging Face, GitHub) is in `docs/setup-report-pod.txt`.
Never try to tunnel, proxy or otherwise route around the firewall; download on the laptop and
`scripts/sync.sh put` instead.

You can't launch GPU pods. Rohan runs `scripts/submit_job.sh` on the laptop.

## Limits that shape our jobs

- **Runtime**: 6 h default, up to 12 h if `K8S_TIMEOUT_SECONDS` is set at launch. It can't be
  extended for a running pod. Exceeding it kills the pod with `DeadlineExceeded` and **no warning**.
  Design every GPU job to finish in well under 12 h or to resume.
- **Namespace total** (all your pods together): 8 CPU / 64 GB RAM / **1 GPU**. Per pod: 8 CPU / 32 GB / 1 GPU.
  So GPU jobs run one at a time. Worker pod (4 CPU / 16 GB) + GPU job (4 CPU / 16 GB) fits.
- **Memory flag is a limit; half is guaranteed.** `-m 32` guarantees 16 GB. Plan for the half.
- **GPU requests need a class label**: `-g 1 -l gpu-class=<class>`, or the pod stays Pending forever.

## GPU classes (a memory band, not a model)

| class | ~VRAM | typically | SU/GPU-h peak | off-peak |
|---|---|---|---|---|
| extra-small | 6 GB | A30 slice | 0.25 | 0.125 |
| small | 12 GB | RTX 2080 Ti, or A30/H100 slice | 0.5 | 0.25 |
| medium | 24 GB | A30, A5000, H100/RTX 6000 slice | 1 | 0.5 |
| large | 48 GB | L40S, H100/RTX 6000 slice | 2 | 1 |
| extra-large | 96 GB | full H100 / RTX PRO 6000 | 4 | 2 |

Off-peak = 1 AM to 4 PM Pacific (50% off). Our workspace only grants `medium`, so every GPU
hour costs 1 SU peak / 0.5 SU off-peak. Hardware within a class varies between sessions
(A30, A5000, or an H100/RTX 6000 slice), so always log `nvidia-smi` and normalize timing comparisons.

## Budget

- Default **10 SU per member per week per workspace** (Mon 00:00 to Mon 00:00 Pacific, no carry-over).
  Check the remaining balance on reserve.dsmlp.ucsd.edu, Dashboard -> SU Budget. No CLI for it.
- No booking? Launching creates an **on-demand lease** = declared runtime + 10 min (1 h 10 min if
  none declared). You pay only for time used; stopping the pod ends the lease.
- Budget exhausted -> pod stays Pending with event `OnDemandLeaseDenied` until Monday.
- Launch with `-W <course workspace>` so the course budget and class grants apply. Without `-W`,
  GPU time is charged to `ORG_ON_DEMAND`.
- SU estimate for a job = est_hours x rate(class, peak/off-peak). `scripts/submit_job.sh` prints it.

## Why a pod dies, and how a job should react

| Cause | Warning | What our code does |
|---|---|---|
| Runtime limit | none | wall-clock checkpoints every 15-20 min; auto-resume |
| Idle culling (GPU only) | 30 min, annotation `dsmlp/idle-gpu-status` | never sit idle on a GPU: no CPU-only phases (data collection, plotting) in GPU jobs |
| Preemption (only past the reservation guarantee) | ~15 min, files under `/etc/podinfo` | poll `termination-warning-at` every 15-30 s; checkpoint and exit 75 |
| Eviction (booking cancelled/reassigned) | none | SIGTERM handler saves a checkpoint (10 min grace) |
| Node maintenance | sometimes | same as runtime limit |

Idle means: GPU memory <= ~150 MiB on every GPU **and** CPU <= ~0.1 cores (max over the last hour),
pod at least 45 min old. Checked every 5 min. The `gpu-job` skill has the code for all of this.

## Storage

| Path (inside a pod) | Pool | Notes |
|---|---|---|
| `/home/<user>` | workspace | course homes are about 5-10 GB; separate per workspace |
| `~/private/` | personal | same in every workspace; **project root lives here** (`~/private/cvar-lewm`) |
| `~/public/` | workspace | readable by every workspace member; never put credentials or results here |
| `/datasets/...` | read-only | shared datasets; read in place, don't copy |

- Check usage: `du -h --max-depth 1` in the directory, and the disk-quota-service on Datahub.
- Hidden space hogs: `.local` (pip --user installs), `.cache` (uv/pip/HF/torch), `.vscode-server`,
  Jupyter trash in `.local/share/Trash`. Our `scripts/env.sh` points caches into `~/private/.cache`.
- PushT LeWM data is 13.1 GB compressed (`quentinll/lewm-pusht`); TwoRoom is 3.43 GB. Neither fits a
  course home comfortably. Ask before downloading; prefer collecting our own smaller stochastic
  datasets in CPU pods.
- Containers run as you, unprivileged: no `sudo`, no `apt`. Use the `.venv`.

## Monitoring (Rohan runs these from the laptop; you ask him to paste output)

```
ssh dsmlp kubectl get pods              # Running / Pending / Completed / OOMKilled / DeadlineExceeded
ssh dsmlp kubectl describe pod <pod>    # why Pending (OnDemandLeaseDenied, WaitingForReservation, ...)
ssh dsmlp kubectl get events            # Preempted, ReservationCancelled, OverstayRelinked, ...
ssh dsmlp kubectl delete pod <pod>      # stop it (and stop paying)
```

You can read job output yourself: `scripts/sync.sh pull "<run glob>"` brings back
`jobs/logs/<job>.out` and `runs/<run_id>/{log.txt,env.txt,episodes.jsonl}`. These persist after the
pod is gone; `kubectl logs` doesn't.

## Etiquette

- Stop pods you're not using (`kubectl delete pod`); closing the laptop doesn't stop them.
- Weeks 8-10 and finals week are crowded; queue big eval batches early and off-peak.
- Help: datahub@ucsd.edu; status page datahub.ucsd.edu/hub/status.

See `reference.md` for the exact flags and the source URL for each fact.
