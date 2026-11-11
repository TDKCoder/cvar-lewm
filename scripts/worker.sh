#!/usr/bin/env bash
# Manage the CPU-only worker pod where remote.sh runs commands. CPU pods cost no SU and
# aren't idle-culled, but end at WORKER_HOURS (max 12 h).
#
#   scripts/worker.sh status | start | stop
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_connection

case "${1:-status}" in
  status)
    phase="$(worker_phase)"
    echo "worker '$WORKER_NAME': ${phase:-not present}"
    ;;
  start)
    phase="$(worker_phase)"
    if [ "$phase" = "Running" ] || [ "$phase" = "Pending" ]; then
      echo "worker '$WORKER_NAME' is already $phase"; exit 0
    fi
    if [ -n "$phase" ]; then
      echo "removing old worker ($phase)"
      dsmlp_ssh "kubectl delete pod $WORKER_NAME --wait=true" >/dev/null
    fi
    args=(-c "${WORKER_CPU:-4}" -m "${WORKER_MEM_GB:-16}" -s -b -N "$WORKER_NAME")
    [ -n "${IMAGE:-}" ] && args=(-i "$IMAGE" "${args[@]}")
    [ -n "${WS:-}" ] && args=(-W "$WS" "${args[@]}")
    timeout_s=$(( 3600 * ${WORKER_HOURS:-12} ))
    echo "starting CPU-only worker (no GPU, no SU): launch.sh ${args[*]}  [${WORKER_HOURS:-12} h]"
    dsmlp_ssh "export K8S_TIMEOUT_SECONDS=$timeout_s; $LAUNCH $(quote_args "${args[@]}")"
    for _ in $(seq 1 30); do
      [ "$(worker_phase)" = "Running" ] && { echo "worker is Running"; exit 0; }
      sleep 5
    done
    echo "worker not Running after 150 s; check: ssh $SSH_HOST kubectl describe pod $WORKER_NAME" >&2
    exit 1
    ;;
  stop)
    dsmlp_ssh "kubectl delete pod $WORKER_NAME"
    ;;
  *)
    echo "usage: worker.sh status | start | stop" >&2; exit 2 ;;
esac
