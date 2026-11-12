# Decisions and deviations

Newest on top. Format: `## YYYY-MM-DD: <choice>` then context, options considered, why, and
whether it changes anything promised in the proposal.

## 2026-11-11: Repo is a public fork of upstream LeWM

- Choice: the project repo is `TDKCoder/cvar-lewm` on GitHub (public), a fork of
  `lucas-maes/le-wm`. Remotes: `origin` = git@github.com:TDKCoder/cvar-lewm.git,
  `upstream` = https://github.com/lucas-maes/le-wm.git.
- Base: at setup, `main` == `upstream/main` == `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
  Our work is layered on top (agent kit, `cvar/` package) without editing upstream files.
- Public means anything committed is visible: no credentials, cluster usernames beyond what's
  already in the kit, or large data in git.
- Laptop checkout lives at `~/projects/ECE285/Project/cvar-lewm` (SETUP.md's commands updated
  2026-11-12; the scripts are path-independent).
- Effect on proposal: none.

## 2026-11-10: Claude Code runs on the laptop; commands run in a CPU worker pod

- Context: `curl https://api.anthropic.com` from a DSMLP pod returns 000 (blocked), so Claude
  Code can't run inside a pod.
- Choice: Claude runs in WSL2 on Rohan's laptop. The laptop repo is the source of truth;
  `scripts/sync.sh` mirrors it to the cluster and `scripts/remote.sh` runs commands in a free
  CPU-only worker pod via one shared SSH connection (ControlMaster). GPU jobs are still prepared by
  Claude and submitted by Rohan (`scripts/submit_job.sh`), now from the laptop, from a per-job code
  snapshot on the cluster.
- Rejected: tunnelling the pod's API traffic through the laptop (routes around a campus firewall).
- Verified with a probe pod: login `~` is the pod's `~/private`, so the mirror at login
  `~/cvar-lewm` is `~/private/cvar-lewm` in pods. kubectl needs a login shell over SSH; the scripts
  wrap remote commands in `bash -lc`.
- Effect on proposal: none.

## 2026-11-09: GPU jobs are submitted by hand

- Context: DSMLP charges Service Units for every GPU launch and the default budget is about 10 SU
  per week. CPU-only pods are free and not idle-culled.
- Choice: Claude only prepares GPU job scripts; Rohan submits them after seeing the SU estimate.
- Effect on proposal: none.

## 2026-11-09: Schedule re-based to start Nov 9

- The proposal's 8-week plan started Oct 26. Weeks are compressed to six (A-F) in
  `docs/project-spec.md`; cut order is listed there.
