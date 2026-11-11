---
name: gpu-job
description: Prepare a GPU job for DSMLP - write the job script into jobs/queue/ with a JOB-META header and SU estimate, make it checkpoint/resume-safe and preemption-aware, smoke-test it on CPU first, then hand it to the user to submit. Use whenever work needs a GPU (latent encoding, head training, planning evals) or the user asks to run something on a GPU.
argument-hint: "[short job name] [what it should run]"
---

# Preparing a GPU job

You cannot launch GPU pods. Your output is a **ready-to-submit job script** plus a short message to
Rohan. He runs `scripts/submit_job.sh jobs/queue/<name>.sh` on the laptop. It reads the header,
prints the SU cost, asks for confirmation, pushes the repo, **snapshots the code on the cluster**
(`cvl-snapshots/<name>/`, so your later edits never change a running job), launches a batch pod
that runs the snapshot's copy, and logs it in `docs/gpu-ledger.md`. Runs still write to the main
mirror's `runs/` (via `RUNS_DIR`), so a resubmission resumes the same run directory.

## Checklist (do these in order, and show the user each result)

1. **Is a GPU actually needed?** Data collection, plotting, aggregation, and small-MLP training on
   cached latents usually run fine in the free CPU worker (`scripts/remote.sh --bg ...`). Estimate CPU time first; only go to GPU if
   CPU would take > ~3x longer than the GPU job's SU-equivalent is worth, or doesn't fit in RAM.
2. **Make the entry point resilient** using `templates/resilience.py` (copy it to
   `cvar/utils/resilience.py` once; import from there):
   - `ckpt = CheckpointManager(run_dir, every_s=900, keep=2)`; `state = ckpt.load_latest()` at start.
   - `guard = InterruptGuard()` installs a SIGTERM handler and polls `/etc/podinfo` for termination
     warnings (rate-limited to once per 20 s).
   - In the main loop: `ckpt.maybe_save(make_state, step)`; `if guard.should_stop(): ckpt.save_now(make_state(), step); sys.exit(75)`.
   - Saved state must include model, optimizer, scheduler, AMP scaler, step/episode counters,
     data position, and RNG states (`resilience.rng_state()` / `restore_rng_state`).
   - Evaluation jobs: write one JSON line per finished episode to `episodes.jsonl` (append mode) and
     on resume skip episodes already present. That *is* the checkpoint for evals.
3. **Smoke-test on CPU** in the worker: `scripts/remote.sh "python -m cvar.<entry> --smoke ..."`
   (tiny data, 2 episodes or 50 steps). Then kill it mid-run
   (`scripts/remote.sh "timeout -s TERM 20 python -m cvar.<entry> --smoke ..."`), rerun, and confirm
   it resumes without duplicating episodes or restarting the loss curve. Also run the job script
   itself once in the worker with a smoke flag if it has one. Paste both outputs to Rohan.
4. **Measure and estimate.** From the smoke run, estimate GPU wall-clock: say how (e.g., "CPU plan
   took 4.1 s; LeWM paper reports ~48x faster planning than DINO-WM on GPU; assume 10x CPU->GPU
   speedup, so ~0.4 s/plan, 300 plans/episode -> 2 min/episode -> 100 episodes = 3.3 h"). Add a
   25% buffer. Keep every job under 6 h of estimated time; split bigger grids into several jobs.
5. **Class is `medium`** (24 GB): it's the only GPU class the ECE 285 workspace grants. Rate:
   1 SU/GPU-h peak, 0.5 off-peak (1 AM-4 PM Pacific). LeWM is small, so use the VRAM headroom to
   batch more CEM candidates or run several eval cells per job instead of paying pod overhead
   repeatedly. Show the SU arithmetic: `SU = est_hours x rate`.
6. **Write the job script** from `templates/job_template.sh` to `jobs/queue/<name>.sh`.
   Names: lowercase letters, digits, hyphens (`eval-flow-cvar-p015-s0`). The `RUN_ID` is fixed per
   job so a resubmission resumes the same run directory.
7. **Commit** the job script and code (`git add` + `git commit`). The snapshot copies the working
   tree at submission, and the commit SHA in the job header is how results trace back to code.
   Note the SHA.
8. **Tell the user**, in this shape:

```
GPU job ready: jobs/queue/<name>.sh   (commit <sha>)
Purpose: <one line, which hypothesis/cell>
Class: medium | est 2.5 h (+25% buffer) | ~2.5 SU peak / ~1.25 SU off-peak (1 AM-4 PM)
Smoke test: <command> -> <result>; resume test: <result>
Submit (laptop, your terminal): scripts/submit_job.sh jobs/queue/<name>.sh   [off-peak: 1 AM-4 PM Pacific]
Outputs (cluster): runs/<RUN_ID>/ (episodes.jsonl, log.txt, env.txt, ckpt/), stdout in jobs/logs/<name>.out
After it finishes: tell me "job <name> finished"; I'll sync.sh pull it and run /report-results
```

## Rules

- One GPU job at a time (namespace limit is 1 GPU). If several are queued, say which goes first.
- No CPU-only phases inside a GPU job (the idle culler can kill a GPU pod that isn't using the GPU,
  and you'd pay for nothing). Pre-collect data on CPU; the GPU job only loads it.
- Never read or print credentials. GPU jobs don't need W&B online: `WANDB_MODE=offline` is set.
- If a job fails: `scripts/sync.sh pull "<RUN_ID>"`, read `jobs/logs/<name>.out` and
  `runs/<RUN_ID>/log.txt`, and ask Rohan for `ssh dsmlp kubectl describe pod cvl-<name>` output if
  the pod never started. Diagnose before suggesting a resubmit. A resubmit should resume.
- To resubmit, move `jobs/submitted/<name>.sh` back to `jobs/queue/` (git mv) and tell Rohan.
- Exit code 75 means "interrupted, resumable" by our convention.

## Files

- `templates/job_template.sh`: job script skeleton with the JOB-META header.
- `templates/resilience.py`: CheckpointManager, InterruptGuard, atomic save, RNG state helpers.
