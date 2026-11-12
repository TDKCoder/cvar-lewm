# Setup report (Phase 0, 2026-11-12)

Source: `docs/setup-report-pod.txt` (raw `scripts/pod_audit.sh` output from `cvl-worker`) plus a second
read-only probe run the same day through `scripts/remote.sh`. Every number below comes from those two outputs.

## Pod

| Item | Value |
|---|---|
| Image | `scipy-ml-notebook:2025.1-stable` (pinned in `dsmlp.conf`) |
| CPU limit | cgroup `cpu.max = 400000 100000` → **`CVL_NCPU=4`** (`OMP_NUM_THREADS=4`) |
| RAM limit | cgroup `memory.max = 17179869184` (16 GiB). `free` shows the node's 251 GiB; ignore it. |
| GPU | none (worker is CPU-only, as intended) |
| Python | `/opt/conda/bin/python3` = **3.11.9**; the image also has **`/usr/bin/python3.10`** |
| torch in image | **2.2.1+cu121** (CUDA build 12.1), numpy 1.26.4 |
| Tools | git, pip, conda 24.3.0, mamba, curl, wget, rsync. **No uv.** |
| Proxy env | none |
| `/etc/podinfo` | absent in the worker (expected; batch GPU pods may differ) |
| `whoami` | fails (no passwd entry for UID 23003); harmless |

## Network (from the pod)

The audit's HTTP status at a host's root isn't enough: 404/403 at `/` is normal for file CDNs. So I also
fetched one real object from each.

| Host | Root status | Real object | Verdict |
|---|---|---|---|
| pypi.org | 200 | `/pypi/stable-worldmodel/json` 200 | reachable |
| files.pythonhosted.org | 404 | `six-1.16.0.tar.gz` 200 (34,041 B); `pip download six` works | reachable |
| download.pytorch.org | 403 | `/whl/cu121/torch/` index 200 | reachable |
| huggingface.co | 200 | API + `config.json` 200; `weights.pt` → 302 → `us.aws.cdn.hf.co` (Xet) 200, 72,290,721 B | reachable |
| cdn-lfs.huggingface.co | 000 | not used by HF any more (files redirect to the Xet CDN) | unreachable, irrelevant |
| github.com | 200 | release page 200 | reachable |
| objects / release-assets.githubusercontent.com | 404 at root | uv tarball → `release-assets.githubusercontent.com` 200, 16,671,812 B | reachable |
| astral.sh | 200 | `install.sh` → `releases.astral.sh` 200 | reachable |
| conda.anaconda.org | 302 | not tested further | probably reachable |
| api.anthropic.com | **404** | not tested further | **see Risks** |

So install option **(a)** works: uv from GitHub/astral, a uv-managed Python 3.10 from python-build-standalone,
and wheels from PyPI and download.pytorch.org.

## Disk

| Path | Filesystem | Used by us |
|---|---|---|
| `$HOME` = `/home/rnambimadom` (ECE285 workspace) | fs05, 69T total, 1% used | **13 G, all of it in `~/public`**; everything else is < 20 KB |
| `~/private` (personal pool) | fs03, 136T total, 42% used | **21 G**: `repro-project` 14 G, `vla-project` 6.1 G, `.vscode-server` 805 M, `.conda` 237 M, `cvar-lewm` 13 M |

`df` reports the whole NFS export, not a per-user quota, and the pod has no `quota` command. **Neither
the course-home nor the `~/private` quota is known.**

## Risks

1. **Unknown `~/private` quota.** Everything big (venv, uv cache, checkpoints, datasets, latents) goes
   there, and 21 G is already in use by other projects. If the quota is around 25-50 G, a CUDA torch
   venv (several GB) plus a 13 GB (compressed) dataset won't fit. Check the quota on Datahub's
   disk-quota-service before any multi-GB download.
2. **13 G in `~/public` on the course home.** Course homes are said to be about 5-10 GB, and
   `~/public` is readable by every workspace member. If those are your files, they may already exceed
   the home quota. Nothing of ours goes there.
3. **CUDA torch size.** A cu121 torch wheel plus the nvidia-* wheels is several GB in the venv, plus
   the same again in the uv cache. Reusing the image's torch 2.2.1 would avoid that, but only on
   Python 3.11 (see block 2).
4. **Hugging Face downloads depend on `us.aws.cdn.hf.co`** (Xet). It works today. If it breaks, the
   fallback is to download on the laptop and `sync.sh put`.
5. **api.anthropic.com returns 404 from the pod today**, not 000 as recorded on 2026-11-10. 404 means
   an HTTP server answered, so egress to it may now be open. Nothing changes: we stay in laptop mode
   and don't use it. The earlier decision entry may need a note.
6. **The worker pod ends after 12 h.** Background jobs (`remote.sh --bg`) die with it, so long data
   collection must be resumable.
7. **RAM is 16 GiB limit, half guaranteed.** Keep CPU jobs under about 8 GiB to be safe.
