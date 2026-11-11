#!/usr/bin/env bash
# Rohan runs this on the laptop (never Claude). Submits one GPU job prepared in jobs/queue/.
#
#   scripts/submit_job.sh jobs/queue/<name>.sh [--dry-run]
#
# 1. reads "# JOB-META class=<c> est_hours=<h> desc=..." and prints the SU estimate
# 2. asks y/N
# 3. moves the job to jobs/submitted/, pushes the repo, and snapshots it on the cluster
#    (cvl-snapshots/<name>/) so later edits never change the running job
# 4. launches a batch GPU pod that runs the snapshot's copy of the job script
# 5. appends a row to docs/gpu-ledger.md
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

JOB_REL="${1:?usage: submit_job.sh jobs/queue/<name>.sh [--dry-run]}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1

JOB="$REPO_ROOT/$JOB_REL"
[ -f "$JOB" ] || { echo "No such job file: $JOB"; exit 1; }
case "$JOB_REL" in jobs/queue/*.sh) ;; *) echo "Job must be jobs/queue/<name>.sh"; exit 1 ;; esac

META="$(grep -m1 '^# JOB-META' "$JOB" || true)"
[ -n "$META" ] || { echo "No '# JOB-META' header in $JOB"; exit 1; }
CLASS="$(sed -n 's/.*class=\([a-z-]*\).*/\1/p' <<<"$META")"
CLASS="${CLASS:-${DEFAULT_GPU_CLASS:-}}"
EST_H="$(sed -n 's/.*est_hours=\([0-9.]*\).*/\1/p' <<<"$META")"
DESC="$(sed -n 's/.*desc="\([^"]*\)".*/\1/p' <<<"$META")"

case "$CLASS" in
  extra-small) PEAK=0.25; OFF=0.125 ;;
  small)       PEAK=0.5;  OFF=0.25 ;;
  medium)      PEAK=1;    OFF=0.5 ;;
  large)       PEAK=2;    OFF=1 ;;
  extra-large) PEAK=4;    OFF=2 ;;
  *) echo "Bad or missing class '$CLASS' in JOB-META"; exit 1 ;;
esac
[[ "$EST_H" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "Bad est_hours '$EST_H' in JOB-META"; exit 1; }
awk -v h="$EST_H" 'BEGIN{exit !(h > 0 && h <= 11)}' || { echo "est_hours must be in (0, 11]; split the job"; exit 1; }

# Off-peak = 1 AM to 4 PM *Pacific*, whatever time zone the laptop is in.
HOUR=$(TZ=America/Los_Angeles date +%-H)
if [ "$HOUR" -ge 1 ] && [ "$HOUR" -lt 16 ]; then RATE=$OFF; PERIOD="off-peak"; else RATE=$PEAK; PERIOD="peak"; fi
su_for() { awk -v h="$EST_H" -v r="$1" 'BEGIN{printf "%.3f", h*r}'; }
EST_SU="$(su_for "$RATE")"
# Runtime limit = 1.25 x estimate + 10 min, capped at 12 h. You pay for time used, not the limit.
TIMEOUT_S=$(awk -v h="$EST_H" 'BEGIN{t=int(h*1.25*3600+600); if(t>43200)t=43200; print t}')

BASE="$(basename "$JOB_REL")"
NAME="${BASE%.sh}"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Job name may only use letters, digits, . _ -"; exit 1; }
POD_NAME="$(echo "cvl-$NAME" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-50 | sed 's/-*$//')"
SNAP_R="${SNAP_REMOTE:-cvl-snapshots}/$NAME"
SNAP_P="${SNAP_POD:-\$HOME/private/cvl-snapshots}/$NAME"
POD_JOB="$SNAP_P/jobs/submitted/$BASE"
POD_LOG="$POD_ROOT/jobs/logs/$NAME.out"

ARGS=(-g 1 -l "gpu-class=$CLASS" -c "${GPU_CPU:-4}" -m "${GPU_MEM_GB:-16}" -N "$POD_NAME" -B)
[ -n "${IMAGE:-}" ] && ARGS=(-i "$IMAGE" "${ARGS[@]}")
[ -n "${WS:-}" ] && ARGS=(-W "$WS" "${ARGS[@]}")
POD_CMD="mkdir -p $POD_ROOT/jobs/logs && bash $POD_JOB > $POD_LOG 2>&1"

cat <<EOF
Job:        $JOB_REL
Purpose:    ${DESC:-<none>}
Class:      $CLASS   est ${EST_H} h   rate now: $RATE SU/h ($PERIOD, Pacific)  =>  ~${EST_SU} SU
Peak/off:   $(su_for "$PEAK") SU peak / $(su_for "$OFF") SU off-peak (1 AM-4 PM Pacific)
Workspace:  ${WS:-<default; charged to ORG_ON_DEMAND>}
Pod:        $POD_NAME   runtime limit $((TIMEOUT_S/60)) min
Code:       snapshot at $SNAP_R (commit $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), $(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted files)
Budget:     check remaining SU at https://reserve.dsmlp.ucsd.edu (Dashboard -> SU Budget)
EOF

if [ "$DRY" = 1 ]; then
  echo "(dry run: nothing launched)"
  exit 0
fi
read -r -p "Submit? [y/N] " ans
[[ "$ans" =~ ^[Yy]$ ]] || { echo "Not submitted."; exit 0; }

require_connection
if pod_exists "$POD_NAME"; then
  echo "A pod named $POD_NAME exists. Delete it first: ssh $SSH_HOST kubectl delete pod $POD_NAME"; exit 1
fi

mkdir -p "$REPO_ROOT/jobs/submitted"
mv "$JOB" "$REPO_ROOT/jobs/submitted/$BASE"
undo() { mv "$REPO_ROOT/jobs/submitted/$BASE" "$JOB"; echo "Launch failed; job moved back to the queue." >&2; exit 1; }

"$REPO_ROOT/scripts/sync.sh" push || undo
# Snapshot on the login node (plain file copy; allowed there).
dsmlp_ssh "mkdir -p $(quote_args "$SNAP_R") && rsync -a --delete --exclude /runs/ --exclude /jobs/logs/ $(quote_args "$REMOTE_ROOT/" "$SNAP_R/")" || undo

dsmlp_ssh "export K8S_TIMEOUT_SECONDS=$TIMEOUT_S; $LAUNCH $(quote_args "${ARGS[@]}") -- bash -c $(printf '%q' "$POD_CMD")" || undo

LEDGER="$REPO_ROOT/docs/gpu-ledger.md"
if [ -f "$LEDGER" ]; then
  printf '| %s | %s | %s | %s | %s | %s | %s |  |  | %s |\n' \
    "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M')" "$NAME" "$CLASS" "$PERIOD" "$EST_H" "$EST_SU" "$POD_NAME" "${DESC:-}" >> "$LEDGER"
fi

cat <<EOF

Submitted. Watch it:
  ssh $SSH_HOST kubectl get pods
  ssh $SSH_HOST kubectl describe pod $POD_NAME    # Pending + OnDemandLeaseDenied = out of SU this week
  ssh $SSH_HOST tail -f $REMOTE_ROOT/jobs/logs/$NAME.out
Stop it early (stops billing):  ssh $SSH_HOST kubectl delete pod $POD_NAME
When it's done, tell Claude: "job $NAME finished".
EOF
