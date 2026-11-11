# DSMLP reference (with sources)

All pages under https://dsmlp-docs.ucsd.edu/ were marked draft when read (2026-11-09).
Confirm anything decision-critical with datahub@ucsd.edu.

## launch.sh flags
Source: https://dsmlp-docs.ucsd.edu/running-jobs/launch-sh-reference/

Absolute path: `/opt/launch-sh/bin/launch.sh`. Wrappers: `launch-scipy-ml.sh` (GPU-capable image,
2 CPU / 8 GB default), `launch-datascience.sh` (CPU image). Default image:
`ghcr.io/ucsd-ets/scipy-ml-notebook:stable`.

| flag | meaning |
|---|---|
| `-c <n>` | CPU limit (half guaranteed) |
| `-m <n>` | RAM limit in GB (half guaranteed) |
| `-g <n>` | GPU count (lowercase; `-G` is the group flag) |
| `-l gpu-class=<class>` | GPU class label; required with `-g` |
| `-W <workspace>` | workspace (budget, class grants, `$HOME`) |
| `-i <image>` / `-P Always` | image / pull policy |
| `-N <name>` | name the pod |
| `-b` | background: create pod, return to login shell |
| `-B` | batch: queue the job, don't wait |
| `-f <script>` | run a script, then exit |
| `-s` | CLI shell only, no Jupyter |
| `-H` | SSH server in the container (for VS Code Remote-SSH ProxyCommand) |
| `-u` | email when the job starts |
| `-d` | print the pod spec JSON, don't run |
| `--` | everything after goes to the container |

Environment-variable equivalents: `K8S_NUM_CPU`, `K8S_GB_MEM`, `K8S_NUM_GPU`, `K8S_DOCKER_IMAGE`,
`K8S_IMAGE_PULL_POLICY`, `K8S_TIMEOUT_SECONDS` (no flag), `K8S_ENTRYPOINT`.

Examples from the docs:
```
launch-scipy-ml.sh -g 1 -B -- python train.py --epochs 50 --lr 0.01
launch-scipy-ml.sh -g 1 -l gpu-class=small
ssh <user>@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -c 8 -m 16 -g 1 -i <image> -f ${HOME}/myproject/run-commands.sh
```

## Job modes and runtime
Source: https://dsmlp-docs.ucsd.edu/running-jobs/job-modes-and-limits/

- Interactive (default) ends when the login connection drops.
- Background `-b`: survives disconnect; enter with `kubesh <pod>`.
- Batch `-B -- cmd`: runs `cmd`, exits; `kubectl logs <pod>` until deleted.
- Runtime limit 6 h default, max 12 h via `export K8S_TIMEOUT_SECONDS=$(( 3600 * 12 ))` before launch.

## GPU classes, reservations, SU
Sources: https://dsmlp-docs.ucsd.edu/gpu-access/gpu-classes/ ,
https://dsmlp-docs.ucsd.edu/gpu-access/reservations/ ,
https://dsmlp-docs.ucsd.edu/gpu-access/service-units-and-budgets/

- Classes and rates: see SKILL.md table. Off-peak 1 AM-4 PM Pacific.
- Bookings at reserve.dsmlp.ucsd.edu (whole hours, >= 15 min ahead, <= 48 h; must be claimed within
  15 min of start; cancelling < 24 h ahead is penalized).
- On-demand lease: declared runtime + 10 min, pay for time used.
- Default budget 10 SU / member / week / workspace.

## What ends a session
Source: https://dsmlp-docs.ucsd.edu/gpu-access/what-ends-a-session/

Idle culling (GPU sessions only), end of reservation guarantee + preemption, eviction, maintenance.
SIGTERM then 600 s grace for launch.sh pods.

## Checkpointing guidance
Source: https://dsmlp-docs.ucsd.edu/running-jobs/checkpointing/

- Checkpoint on wall-clock every 15-30 min; keep last two plus best; atomic writes.
- Termination-warning files under `/etc/podinfo` (downward API). Test with `[[ -s file ]]`
  (an absent annotation is an empty file). Poll every 15-30 s; re-read each time.
- `kubectl logs` disappears with the pod; log to files.

## Login node
Source: https://dsmlp-docs.ucsd.edu/access/the-login-node/

`ssh <ADuser>@dsmlp-login.ucsd.edu`; Duo once per 8 h; key auth via `ssh-keygen -t ed25519` and
`~/.ssh/authorized_keys` on the login node. No VPN needed for SSH. Permitted: launching, file
movement, kubectl get/delete/kubesh, small housekeeping. Prohibited: running Python, builds,
training, notebooks, editor servers.

## Storage
Source: https://dsmlp-docs.ucsd.edu/workspaces-and-storage/your-files-and-quotas/

Workspace home `/home/<user>` (course: ~5-10 GB), `private/` (personal pool, all workspaces),
`public/`, `teams/` (with `-G`). Login node: `workspace -l`, `workspace -c <WS>`.

## Policy
Source: https://dsmlp-docs.ucsd.edu/reference/policy/

Campus acceptable-use policy applies. Stop unused sessions. No computation on the login node.
