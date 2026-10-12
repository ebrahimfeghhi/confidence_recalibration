#!/bin/bash
# Every 5 minutes: check that run_e_to_i is moving, and FIX what can be fixed rather than printing it.
#
# A WATCHER THAT ONLY PRINTS IS NOT A WATCHER. Each check below either remediates or escalates by
# name into watch_e_to_i.log, which is the file to read first in the morning.
#
# WHAT IT REMEDIATES
#   orchestrator died mid-stage        relaunch it; every phase resumes from its own outputs
#   a fold worker died, queue unfinished   phase I's own round loop handles it; this only escalates
#                                          if BOTH workers are gone while stage6_fold is current
#   disk above 88%                     free the env database if phase F has already passed
#   no progress for 45 min in a stage  escalate with the stage's own log tail
#
# WHAT IT NEVER DOES: it never kills by pattern (any pattern search matches this script too), never
# deletes a pose, and never starts a fold the orchestrator has not reached.
set -u
R=/home/ebrahim/tts_abag
P=/home/ebrahim/Protenix
PY=$P/.venv/bin/python
S=$R/extra_training/scripts
PH=$R/extra_training/phases
W=$R/extra_training/msa_work
DB=$R/tools/colabfold/db
E=$R/extra_training/val_fold
LOG=$R/extra_training/watch_e_to_i.log
STATE=$R/extra_training/run_e_to_i.state
INTERVAL="${INTERVAL:-300}"
STALL_TICKS=9                       # 9 x 5 min = 45 min without progress

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
esc() { echo "[$(date '+%F %T')] ESCALATE: $*" >> "$LOG"; }

# PROGRESS IS A NUMBER THAT MUST GROW, chosen per stage: the thing that stage produces. A stage whose
# own product is not growing is stalled whatever its process table looks like.
progress() {
    case "$(cat $STATE 2>/dev/null)" in
        stage0_envdb)  stat -c %s "$DB/colabfold_envdb_202108.tar.gz" 2>/dev/null || du -sb "$DB" 2>/dev/null | cut -f1 ;;
        stage1_phaseE) find $W/msa_initial_idx $W/msa_short_idx $W/msa_bench_short_idx -name 'bfd*.a3m' 2>/dev/null | wc -l ;;
        stage2_phaseF) find $W/mmcif_msa -name non_pairing.a3m 2>/dev/null | wc -l ;;
        stage3*)       ls $R/extra_training/val_jobs/*-final-updated.json 2>/dev/null | wc -l ;;
        stage4_phaseH) find $W/mmcif_msa -name hmmsearch.a3m 2>/dev/null | wc -l ;;
        stage5_smoke|stage6_fold) find $E/trunks -name '*.npz' 2>/dev/null | wc -l ;;
        *)             echo 0 ;;
    esac
}

LAST=""; STUCK=0
say "=== watcher starting (pid $$), every ${INTERVAL}s ==="
while true; do
    ST=$(cat $STATE 2>/dev/null || echo "none")
    PR=$(progress)

    # 1. is the orchestrator alive? Relaunch it if the state says work remains.
    if ! tmux has-session -t e2i 2>/dev/null; then
        case "$ST" in
            COMPLETE|INCOMPLETE|FAILED)
                say "orchestrator finished: $ST. Watcher idling (nothing left to drive)." ;;
            *)
                esc "orchestrator session 'e2i' is gone at stage $ST -- relaunching. Every phase
     resumes from its own outputs, so a repeat is a no-op where the stage already passed."
                tmux new-session -d -s e2i "bash $S/run_e_to_i.sh" \
                    && say "relaunched e2i" || esc "could not relaunch e2i" ;;
        esac
    fi

    # 2. progress, per stage
    if [ "$ST" = "$LAST" ] && [ "$PR" = "${LASTPR:-}" ]; then
        STUCK=$((STUCK+1))
        if [ "$STUCK" -ge "$STALL_TICKS" ] && [ $((STUCK % STALL_TICKS)) -eq 0 ]; then
            esc "no progress in stage $ST for $((STUCK*INTERVAL/60)) min (measure stuck at $PR)"
            case "$ST" in
                stage1_phaseE) tail -5 $W/phase_e.log 2>/dev/null | sed 's/^/     E: /' >> "$LOG" ;;
                stage2_phaseF) tail -5 $W/phase_f.log 2>/dev/null | sed 's/^/     F: /' >> "$LOG" ;;
                stage4_phaseH) tail -5 $W/phase_h.log 2>/dev/null | sed 's/^/     H: /' >> "$LOG" ;;
                stage6_fold)   tail -5 $E/w0.log 2>/dev/null | sed 's/^/     w0: /' >> "$LOG"
                               tail -5 $E/w1.log 2>/dev/null | sed 's/^/     w1: /' >> "$LOG" ;;
                *)             tail -5 $R/extra_training/run_e_to_i.log | sed 's/^/     : /' >> "$LOG" ;;
            esac
        fi
    else
        [ "$STUCK" -ge "$STALL_TICKS" ] && say "stage $ST moving again ($PR)"
        STUCK=0
    fi
    LAST="$ST"; LASTPR="$PR"

    # 3. disk. The env database is the only thing here worth 160 GB, and it is safe to free once
    # phase F has passed -- which is exactly what the state tells us.
    USED=$(quota -s 2>/dev/null | awk '/nvme/{gsub(/[A-Za-z]/,"",$2); gsub(/[A-Za-z]/,"",$4); if($4>0) printf "%d", ($2/$4)*100}')
    if [ -n "${USED:-}" ] && [ "$USED" -ge 88 ]; then
        esc "quota at ${USED}%"
        case "$ST" in
            stage3*|stage4_phaseH|stage5_smoke|stage6_fold|verify|COMPLETE|INCOMPLETE)
                if [ -s "$DB/colabfold_envdb_202108_db_seq" ]; then
                    rm -rf "${DB:?}"/colabfold_envdb_202108_db* "${DB:?}"/COLABDB_READY
                    say "freed the env database (phase F has passed, so it has no reader left)"
                fi
                rm -f "${DB:?}/colabfold_envdb_202108.tar.gz" && say "freed the env tarball" ;;
            *)  esc "cannot free space safely at stage $ST: phase F has not passed, so the env
     database is still needed. Manual attention required." ;;
        esac
    fi

    # 4. during the fold, are both GPUs actually ours?
    if [ "$ST" = "stage6_fold" ]; then
        up=0
        for g in 0 1; do tmux has-session -t vf_val_$g 2>/dev/null && up=$((up+1)); done
        [ "$up" -eq 0 ] && esc "stage6_fold is current but neither vf_val worker session is alive.
     Phase I's round loop relaunches up to 5 times; if this repeats, read $E/w0.log and $E/w1.log."
    fi

    say "$ST  measure=$PR  stuck_ticks=$STUCK  quota=${USED:-?}%  sessions=$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
    sleep "$INTERVAL"
done
