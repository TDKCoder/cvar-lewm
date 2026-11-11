# Prompts for Claude Code (laptop mode)

## Repo setup (paste once, from ~ in WSL, before the kickoff)

Change the repo URL if your fork is still named `le-wm`.

```
Set up my CVaR-LeWM repo at ~/cvar-lewm on this laptop (WSL2). The agent kit is unzipped at
~/kit/cvar-lewm-agent-kit/; read its CLAUDE.md and SETUP.md first (skip steps that need my logins).
My GitHub repo is https://github.com/TDKCoder/cvar-lewm, a public fork of lucas-maes/le-wm.

Show me each command's output as you go.
1. git clone https://github.com/TDKCoder/cvar-lewm.git ~/cvar-lewm, add
   https://github.com/lucas-maes/le-wm.git as `upstream`, fetch it, show `git remote -v`, and
   confirm main matches upstream/main (give the SHA).
2. Copy the kit into the repo root (including dotfiles like .claude/ and .gitignore.cvar). First
   list any kit path that already exists upstream; if there's a collision, stop and ask me.
   Append .gitignore.cvar to .gitignore and delete .gitignore.cvar. Make scripts/*.sh executable.
3. Scaffold the layout in CLAUDE.md: cvar/__init__.py and cvar/{envs,predictors,planning,
   calibration,data,utils}/__init__.py; config/cvar/, jobs/{queue,submitted,logs}/, results/,
   notebooks/, tests/ with .gitkeep where empty. Copy .claude/skills/gpu-job/templates/resilience.py
   to cvar/utils/resilience.py.
4. Create the local analysis venv: python3 -m venv .venv-analysis, then install numpy pandas
   matplotlib jupyter into it (ask me first). Check it's git-ignored.
5. Add a dated entry to docs/decisions.md: repo is a public fork, upstream SHA.
6. Check git user.name and user.email are set (stop and ask if not). Commit as two commits:
   "Add CVaR-LeWM agent kit" and "Scaffold cvar package".
7. Don't push. Show `git log --oneline -5`, the tree two levels deep (skip .git and .venv-analysis),
   and the push command for me to run.
```

## Kickoff (paste once, in plan mode, from ~/cvar-lewm)

```
We're starting CVaR-LeWM, my ECE 285 project. Read CLAUDE.md, docs/project-spec.md and
docs/proposal/proposal.tex first. You're on my laptop; code runs on DSMLP through scripts/remote.sh
in a free CPU worker pod. You can't launch GPU jobs and must not try. Today is Phase 0 (schedule
week A). Show me a plan before editing anything.

Do these in order. After each block, show me what you ran and what came out (I want the
intermediate steps), commit, and keep going unless something blocks you.

1. Cluster audit. `scripts/worker.sh status` (start it if needed), then
   `scripts/remote.sh "bash scripts/pod_audit.sh"`; save the output to docs/setup-report-pod.txt.
   Write docs/setup-report.md: what the pod can reach (PyPI, Hugging Face, GitHub, astral.sh,
   download.pytorch.org), python/torch/CUDA in the image, free space in $HOME and ~/private, and a
   "risks" section (disk and network especially).
2. Install plan. From step 1, pick how to build the venv at $VENV_DIR (see scripts/env.sh), and ask
   me before installing:
   (a) PyPI + GitHub/astral reachable: uv with a managed Python 3.10, then
       `uv pip install "stable-worldmodel[train,env]"`;
   (b) only PyPI reachable: uv from PyPI (`pip install --user uv`), and say whether Python 3.10 is
       obtainable or we should try the image's Python;
   (c) no PyPI: a wheelhouse built on the laptop for linux x86_64 and sent with
       `scripts/sync.sh put`; estimate its size first.
   Then freeze to requirements.lock.txt (capture `uv pip freeze` from the pod) and record the
   install size.
3. Codebase map. Use the lewm-codebase skill and write docs/codebase-map.md. I especially need:
   where the predictor is called inside CEM, the latent dim, how eval picks start/goal states, and
   whether eval.py needs the 13 GB pusht_expert_train file.
4. Pretrained checkpoint on CPU. Get the quentinll/lewm-pusht *model* repo (check its size and ask
   first; download on the laptop and `sync.sh put` it if the pod can't reach Hugging Face), convert
   it as the README shows, then in the worker encode one frame, run one predictor step and one tiny
   CEM plan. Report latency per plan call and the predictor's parameter count. Add these as tests.
5. Data decision. Given step 3, lay out options for eval start/goal states and for collecting our
   stochastic training data (which policy generates it, how many transitions, disk size). Recommend
   one and log it in docs/decisions.md. Don't download the 13 GB dataset.
6. Stochastic PushT. Implement cvar/envs/ wrappers (impulse with prob p, per-episode friction and
   mass resampling, all from one seeded Generator, ranges in a config file). Tests, run in the
   worker: p=0 reproduces the original env bit-for-bit for fixed seeds; same seed gives the same
   trajectory; p=0.15 gives outcome spread on a fixed open-loop plan (save a small figure to
   results/phase0/ and pull it back).
7. Resilience. Add tests for cvar/utils/resilience.py (checkpoint rotation, corrupt-file fallback,
   SIGTERM flag, podinfo warning files).
8. GPU smoke job. Use the gpu-job skill to write jobs/queue/smoke-eval-pusht.sh: confirm CUDA,
   run upstream LeWM eval on deterministic PushT for 10 episodes, log episodes.jsonl in our
   schema, and time it. Smoke- and resume-test it in the worker first. Don't submit it; give me the
   SU estimate and the submit command.
9. Wrap up. Entry in docs/EXPERIMENTS.md (Phase 0), any deviations in docs/decisions.md, and a short
   summary for me: what works, blockers, what you need me to decide, and whether the six-week
   schedule in project-spec section 9 still holds.
```

## Resuming (new day, or after the worker expired)

First run `ssh dsmlp true` yourself, then `claude --continue` and paste:

```
New session. Run `scripts/worker.sh status` (start it if needed), `git log --oneline -15` and
`git status`, read the newest entries in docs/EXPERIMENTS.md and docs/decisions.md, check
jobs/queue/ and jobs/submitted/, and check on any `remote.sh --bg` work from last time (runs/_bg/).
Then tell me where we are against project-spec section 9 and what you'll do next. Don't redo
finished work.
```

## After a GPU job finishes

```
Job <name> finished (or: died; here's `ssh dsmlp kubectl describe pod cvl-<name>`: <paste>).
Pull it with scripts/sync.sh, check runs/<RUN_ID>/ is complete, then /report-results on it. If it
died, diagnose before suggesting a resubmit, and confirm the resubmit will resume rather than restart.
```

## Weekly

```
/report-results weekly
```

## Before writing the final report

```
List every number we plan to put in the report with the results/ file it comes from, and re-run
verify_summary.py on each experiment. Flag anything without a verified source.
```
