#!/bin/bash
# STOP ONE RUN, BY RUN ID, SAFELY.  Usage:  ./shell_scripts/stop_run.sh <rid> [--yes]
#
# WHY THIS EXISTS. Stopping a run by hand has gone wrong twice in this project in the same way:
# `pkill -f` and `pgrep ... | xargs kill` MATCH THE SHELL RUNNING THEM, so the command kills its
# own session. A careless `/proc/$P/stat` with an empty $P collapses to `/proc/stat` and dumps the
# system file instead of a process. Both are pattern matching; this script never pattern matches.
#
# HOW IT IDENTIFIES PROCESSES. Pids come from `nvidia-smi --query-compute-apps`, which lists only
# processes actually holding a GPU. Each candidate is then CONFIRMED by reading
# /proc/<pid>/cmdline -- a direct read of a known pid, not a scan -- and kept only if it carries
#     --out <abs path>/training/runs/<rid>
# so a run whose id is a prefix of another's cannot be caught by accident, and neither can any
# process belonging to someone else.
#
# THE GPU HOLDERS ARE REFUSED BY PID. 306042, 306045 and 306050 reserve the cards for this project
# and killing one hands the GPU to another user mid-run. They are checked against an explicit
# denylist before any signal is sent, not merely expected to fail the cmdline test.
#
# SIGTERM FIRST, and only then SIGKILL. The trainer writes resume_ep*.pt at each epoch boundary,
# so a SIGTERM between boundaries costs at most the epoch in flight; SIGKILL during a checkpoint
# write could leave a truncated .pt that a later --resume would load.
#
# EXIT CODES: 0 stopped (or nothing was running), 1 refused, 2 still alive after SIGKILL.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
HOLDERS=" $( [ -r training/runs/gpu_holder_pids ] && tr '\n' ' ' < training/runs/gpu_holder_pids || echo '306042 306045 306050' ) "
TERM_WAIT=60      # seconds to allow a clean exit; an epoch boundary is ~3.5 min of training
KILL_WAIT=20

RID="${1:-}"
YES="${2:-}"
case "$RID" in
    "" | -*) echo "usage: $0 <rid> [--yes]"; exit 1 ;;
esac
RUNDIR="$(pwd)/training/runs/$RID"
[ -d "$RUNDIR" ] || { echo "REFUSING: no such run directory $RUNDIR"; exit 1; }

# ---- collect, then confirm ---------------------------------------------------------------------
mapfile -t CAND < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u)
TARGETS=()
for P in "${CAND[@]:-}"; do
    # EVERY GUARD MATTERS: an empty or non-numeric pid is what turns /proc/$P/... into /proc/...
    [ -n "$P" ] || continue
    case "$P" in *[!0-9]*) continue ;; esac
    case "$HOLDERS" in *" $P "*)
        echo "  skip pid $P -- GPU HOLDER, never killed"; continue ;;
    esac
    [ -r "/proc/$P/cmdline" ] || continue
    CMD="$(tr '\0' ' ' < "/proc/$P/cmdline")"
    # TWO FLAGS, NOT ONE. The trainer ranks carry --out <rundir>, but the EVAL SHARD workers
    # (phase_k_eval.py, one per card) carry only --ckpt <rundir>/head_epN_stepM.pt. Matching on
    # --out alone left three eval workers holding all three cards after the trainer exited, and
    # the next launch refused because the cards looked busy. Both forms are anchored to RUNDIR
    # followed by a space or a slash, so a run id that prefixes another still cannot match.
    case "$CMD" in
        *"--out $RUNDIR "*|*"--out $RUNDIR/"*|*"--ckpt $RUNDIR/"*) TARGETS+=("$P") ;;
    esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "nothing to stop: no GPU process carries --out $RUNDIR"
    exit 0
fi

echo "run $RID -- ${#TARGETS[@]} process(es) confirmed by --out path:"
for P in "${TARGETS[@]}"; do
    printf '  pid %-8s %s\n' "$P" "$(tr '\0' ' ' < "/proc/$P/cmdline" | cut -c1-110)"
done
LATEST="$(ls -t "$RUNDIR"/resume_ep*.pt 2>/dev/null | head -1)"
[ -n "$LATEST" ] && echo "  newest checkpoint: $(basename "$LATEST")  ($(stat -c %y "$LATEST"))"

if [ "$YES" != "--yes" ]; then
    echo
    echo "DRY RUN. Re-run with --yes to send SIGTERM to the pids above."
    exit 0
fi

# ---- SIGTERM, wait, escalate -------------------------------------------------------------------
echo "sending SIGTERM to ${TARGETS[*]}"
for P in "${TARGETS[@]}"; do kill -TERM "$P" 2>/dev/null || echo "  pid $P already gone"; done
for _ in $(seq "$TERM_WAIT"); do
    ALIVE=0
    for P in "${TARGETS[@]}"; do [ -d "/proc/$P" ] && ALIVE=$((ALIVE + 1)); done
    [ "$ALIVE" -eq 0 ] && break
    sleep 1
done
STILL=()
for P in "${TARGETS[@]}"; do [ -d "/proc/$P" ] && STILL+=("$P"); done
if [ "${#STILL[@]}" -gt 0 ]; then
    echo "SIGTERM left ${#STILL[@]} alive after ${TERM_WAIT}s; sending SIGKILL to ${STILL[*]}"
    for P in "${STILL[@]}"; do kill -KILL "$P" 2>/dev/null || true; done
    for _ in $(seq "$KILL_WAIT"); do
        ALIVE=0
        for P in "${STILL[@]}"; do [ -d "/proc/$P" ] && ALIVE=$((ALIVE + 1)); done
        [ "$ALIVE" -eq 0 ] && break
        sleep 1
    done
    for P in "${STILL[@]}"; do
        [ -d "/proc/$P" ] && { echo "FAILED: pid $P survived SIGKILL"; exit 2; }
    done
fi

# ---- the cards must actually be free, checked rather than assumed ------------------------------
echo "all target pids gone; cards_busy reports:"
$PY -I run_logging/cards_busy.py
BUSY="$($PY -I run_logging/cards_busy.py 2>/dev/null)"
echo "run $RID stopped. busy count on cards 4,5,7 (excluding holders): ${BUSY:-?}"
exit 0
