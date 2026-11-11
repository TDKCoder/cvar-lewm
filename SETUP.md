# CVaR-LeWM: setup runbook (laptop mode)

DSMLP pods can't reach the Anthropic API, so **Claude Code runs on your laptop (in WSL2)** and the
code **executes on DSMLP** in a free CPU-only worker pod. About an hour the first time.

```
 your laptop (WSL2)                         DSMLP login node              DSMLP pods
 ┌──────────────────────────┐   one SSH   ┌─────────────────────┐      ┌──────────────────────┐
 │ Claude Code               │──────────▶│ rsync, kubectl exec,│────▶│ cvl-worker (CPU, free)│
 │ ~/cvar-lewm  (git, truth) │  (Duo once │ launch.sh           │      │  installs, tests,     │
 │ .venv-analysis (plots)    │   per 8 h) │ (no Python here)    │      │  smoke runs, data     │
 └──────────────────────────┘            └─────────────────────┘      ├──────────────────────┤
   scripts/remote.sh  sync.sh  worker.sh                               │ cvl-<job> (GPU, SU)  │
   scripts/submit_job.sh  (you only)                                   │  runs a code snapshot │
                                                                        └──────────────────────┘
```

- **The laptop repo is the source of truth.** `scripts/sync.sh push` mirrors it to
  `~/private/cvar-lewm` on the cluster; `scripts/remote.sh "<cmd>"` pushes, then runs `<cmd>` in the
  worker. Results come back with `scripts/sync.sh pull`.
- **One SSH connection, opened by you.** You type your password and approve Duo once
  (`ssh dsmlp true`); every script reuses that connection for 8 h. Claude never logs in.
- **GPU jobs are submitted by you** with `scripts/submit_job.sh`, which shows the SU cost, asks y/N,
  and runs the job from a frozen copy of the code so later edits can't change it mid-run.
- **Nautilus is untouched.** Claude is denied `kubectl`, `ssh` and reading `~/.kube`.

## What's in the kit

```
CLAUDE.md                        project instructions Claude reads every session
.claude/settings.json            permissions: only the scripts below reach the cluster
.claude/skills/                  dsmlp (cluster facts), gpu-job, report-results, lewm-codebase
scripts/ssh_config.example       the "dsmlp" SSH host with connection sharing
scripts/dsmlp.conf.example       workspace id, image, paths, pod sizes
scripts/remote.sh                run a command in the worker pod (pushes code first)
scripts/sync.sh                  push code / pull results / put big files
scripts/worker.sh                start / stop / status of the free CPU worker
scripts/submit_job.sh            YOU submit a queued GPU job (SU estimate, y/N, snapshot, ledger)
scripts/env.sh, pod_audit.sh     used inside pods
docs/                            project-spec, EXPERIMENTS, decisions, gpu-ledger, proposal/
KICKOFF_PROMPT.md                prompts to paste into Claude
```

## Step 0. Clean up the Datahub session

You don't need the Datahub server any more: JupyterLab -> File -> Hub Control Panel -> Stop My
Server. Leave `~/private/kit` there for now; step 3 uses it to check paths.

## Step 1. WSL2 and tools (laptop)

In PowerShell **as administrator**: `wsl --install -d Ubuntu`, reboot, and create your Ubuntu user.
Then in the Ubuntu terminal:

```bash
sudo apt update && sudo apt install -y git rsync openssh-client unzip curl python3-venv
cp /mnt/c/Users/<WindowsUser>/Downloads/cvar-lewm-agent-kit.zip ~ && unzip -q ~/cvar-lewm-agent-kit.zip -d ~/kit
```

Work inside the Linux filesystem (`~`), not under `/mnt/c`; it's much faster and keeps file
permissions sane.

## Step 2. SSH with one shared connection

```bash
ssh-keygen -t ed25519                                   # Enter for defaults
ssh-copy-id -i ~/.ssh/id_ed25519.pub <ADuser>@dsmlp-login.ucsd.edu   # password + Duo
sed "s/<ADuser>/YOUR_AD_USERNAME/" ~/kit/cvar-lewm-agent-kit/scripts/ssh_config.example >> ~/.ssh/config
chmod 600 ~/.ssh/config
ssh dsmlp true            # opens the shared connection (Duo if asked)
ssh -O check dsmlp        # should print: Master running (pid=...)
ssh dsmlp 'bash -lc "command -v rsync kubectl; workspace -l"'
```

`kubectl` and `workspace` only work in a login shell over SSH (hence `bash -lc`); the scripts
handle that. Your workspace id is `ECE285_FA26_002` (already in `dsmlp.conf.example`).

## Step 3. Storage layout (done: verified 2026-11-10)

A probe pod showed that your login-node `~` (`/dsmlp/home-fs03/03/003/rnambimadom`) is the pod's
`~/private`, and the pod's `$HOME` is the small ECE 285 workspace home. The defaults in
`dsmlp.conf.example` already match: the mirror is `~/cvar-lewm` on the login node and
`~/private/cvar-lewm` in pods. Clean up on the login node:

```bash
ssh dsmlp 'rm -f ~/.cvl-marker ~/core'     # ~/core is a 3.9 GB crash dump from Aug 3
```

## Step 4. Claude Code and git identity (laptop)

```bash
curl -fsSL https://claude.ai/install.sh | bash
exec bash -l                                   # pick up ~/.local/bin
claude                                         # log in; in WSL, paste the code shown in the browser if asked
# (then /exit)
git config --global user.name  "Rohan Nambimadom"
git config --global user.email "<the email on your GitHub account>"
git config --global credential.helper store    # saves your GitHub token for pushes (Claude can't read it)
```

## Step 5. Let Claude assemble the repo

```bash
cd ~ && claude
```

Paste the **Repo setup** prompt from `~/kit/cvar-lewm-agent-kit/KICKOFF_PROMPT.md`. Claude clones
your fork to `~/cvar-lewm`, adds `upstream`, copies the kit in, scaffolds `cvar/`, makes the local
analysis venv, and commits. When it's done, exit Claude and run:

```bash
cd ~/cvar-lewm && git push origin main        # username TDKCoder, password = the GitHub token
cp scripts/dsmlp.conf.example scripts/dsmlp.conf
# WS and paths are already right for your account; no edits needed
scripts/worker.sh start                        # free CPU pod, ~1 min
scripts/remote.sh "hostname && nproc && python3 --version"
```

If that last line prints a pod hostname, everything is wired up.

## Step 6. Kick off

```bash
cd ~/cvar-lewm && claude
```

Switch to plan mode (Shift+Tab) and paste the **Kickoff** prompt.

## Daily loop

```bash
ssh dsmlp true                                 # once per 8 h: password + Duo
cd ~/cvar-lewm && claude --continue
```

- Claude starts the worker if it has expired (`scripts/worker.sh start` is free).
- When Claude says a GPU job is ready, in a **second** WSL terminal:
  `cd ~/cvar-lewm && scripts/submit_job.sh jobs/queue/<name>.sh` (shows SU, asks y/N).
  Submit big jobs off-peak: **1 AM-4 PM Pacific** (that's 3 AM-6 PM Central). The script uses
  Pacific time whatever your laptop's zone is.
- Watch a job: `ssh dsmlp kubectl get pods`, `ssh dsmlp tail -f cvar-lewm/jobs/logs/<name>.out`.
- When it's done: tell Claude "job `<name>` finished". Fridays: `/report-results weekly`.
- Done for the day: `scripts/worker.sh stop`.

## Things that will bite if ignored

1. **Laptop sleep.** Claude stops when the laptop sleeps, and so does the SSH connection. Work
   already running in the worker (`remote.sh --bg`) and GPU jobs keep going. Next time:
   `ssh dsmlp true`, then `claude --continue`.
2. **The 8 h SSH window.** When it lapses, scripts print "No open SSH connection". Run
   `ssh dsmlp true` again.
3. **Disk.** Course homes are about 5-10 GB; everything big goes in `~/private` on the cluster.
   The PushT expert dataset is 13.1 GB compressed; don't download it until Phase 0 says eval needs it.
4. **Budget.** Medium costs 1 SU/h at peak. The proposal's 40-60 GPU-h is 20-30 SU off-peak but
   40-60 SU at peak.
5. **Runtime limit** kills pods with no warning at 6 h (12 h max). Long jobs must resume.
6. **Schedule.** The spec re-bases to six weeks from Nov 9. Tell Claude if your report is due
   before Dec 20.
