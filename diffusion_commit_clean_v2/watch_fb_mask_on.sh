#!/bin/bash
# Every 15 minutes: check the FoldBench mask-ON self-templating run is folding, remediate what can be
# fixed, score what is new. Modelled on watch_scatter50.sh.
#
#   folds done      run.json under poses/ (a fold that finished)
#   scored          distinct (seed, arm) pairs in results.csv
#   FAILED / OOM    counted from the worker logs
#
# REMEDIATE: a GPU in GPUS with no worker session while the queue is unfinished gets its worker
# relaunched. Claims make that safe -- folder.py releases a dead worker's claim at startup.
#
# ESCALATE (logged for the next human/Claude check, not fixed here):
#   - workers up but no fold finished in 15 minutes
#   - out-of-memory lines: a fold that does not fit needs a decision, not a blind retry
#   - SILENT NaN FOLDS, the check this run exists because of. 8xnh seeds 180 and 193 folded fine,
#     emitted 17,661/17,661 NaN coordinates, were correctly refused by OST ("No residues found"),
#     and so produced NO results.csv rows -- silently costing the complex 2 of its 50 matched seeds
#     and blocking build_varseed_queue.py weeks later. Detected generally and cheaply as
#     "finished folds that produced no scored rows", which catches NaN and every other scoring
#     failure without grepping thousands of cifs. Any gap is named by (pdb, seed) so it can be
#     re-folded rather than discovered by accident.
#   - disk: FoldBench poses are large (8xnh is 1.8 MB per pose)
#
# Then the report (score_experiment.py, CPU) runs so the numbers accumulate while the run proceeds.
# When the queue is done and no worker is left: final report, write DONE, exit.
set -u
# E / PREFIX / REPORT are env-overridable so ONE watcher serves every FoldBench mask-ON arm.
#   nc5  arm: E=...selftmpl_mask_on       PREFIX=fbmask  (defaults)
#   nc10 arm: E=...selftmpl_mask_on_nc10  PREFIX=nc10_  REPORT=...fb_mask_on_nc10.md
# PREFIX must match the tmux session names actually in use, or REMEDIATE relaunches a duplicate
# worker onto a GPU that already has one.
E="${E:-/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/selftmpl_mask_on}"
C=/home/ebrahim/tts_abag/diffusion_commit_clean_v2
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS="${GPUS:-0 1 2 3}"
PREFIX="${PREFIX:-fbmask}"
REPORT="${REPORT:-/home/ebrahim/tts_abag/experimental_results/fb_selftmpl_mask_on.md}"
# DATASET too: the pxm2025 twin of this arm uses the same watcher, and scoring the wrong dataset
# would silently produce an empty report rather than an error.
DATASET="${DATASET:-foldbench}"
LOG=$E/watch.log
INTERVAL="${INTERVAL:-900}"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

total=$(grep -c . "$E/queue.tsv")
prev_done=-1; prev_oom=0
while true; do
    # TWO POSE LAYOUTS. The arms store poses at <E>/poses/<pdb>/<seed>-<id>/ but a free pool stores
    # them flat at <E>/<pdb>/<seed>-<id>/. Counting only the first saw zero for the free arm and
    # escalated "no fold finished" every tick while it was folding normally. Count both.
    done_n=$(( $(ls "$E"/poses/*/*/run.json 2>/dev/null | wc -l) \
             + $(ls "$E"/*/*/run.json 2>/dev/null | wc -l) ))
    # the watcher's OWN session must not be counted: with PREFIX=free, "^free" also matches
    # "freewatch", so a single worker reported as two and the stall message named the wrong number.
    workers=$(tmux ls 2>/dev/null | grep -c "^${PREFIX}[0-9]\+:")
    scored=$(awk -F, 'NR>1 && $7!="" {print $2"/"$3}' "$E/results.csv" 2>/dev/null | sort -u | wc -l)
    # STRANDED CLAIMS: a worker killed mid-fold leaves a claim behind, and a claimed-but-unscored
    # unit is skipped by every worker FOREVER -- it does not show up as an error, a failed fold or a
    # missing score, only as a complex that never reaches its full count. On 2026-10-06 three such
    # claims on 8fbw held the comparison's decoy row frozen for an hour with all GPUs at 100%.
    # REMEDIATE by releasing them (safe: a live owner's claim is never touched), then ESCALATE,
    # because folder.py reads its queue once at startup and will not revisit a released unit until a
    # worker restarts.
    stale=$($PY "$C/reconcile_stale_claims.py" "$E" --dataset "$DATASET" 2>&1 | tail -1)
    case "$stale" in
        *RELEASED*) say "  REMEDIATE/ESCALATE: $stale -- needs a worker restart to be refolded" ;;
    esac
    oom=$(cat "$E"/w[0-9]*.log 2>/dev/null | grep -ci "out of memory")
    failed=$(cat "$E"/w[0-9]*.log 2>/dev/null | grep -c FAILED)
    used=$(quota -s 2>/dev/null | tail -1 | awk '{print $2}')
    say "folds done $done_n/$total  scored $scored  workers $workers  FAILED $failed  OOM $oom  disk $used of 1024G"

    if [ "$done_n" -lt "$total" ]; then
        for g in $GPUS; do
            if ! tmux has-session -t "${PREFIX}$g" 2>/dev/null; then
                say "  REMEDIATE: no worker on GPU $g with the queue unfinished -- relaunching"
                tmux new-session -d -s "${PREFIX}$g" "bash $E/launch.sh $g" \
                    && say "  relaunched ${PREFIX}$g" || say "  ESCALATE: relaunch of GPU $g FAILED"
            fi
        done
        [ "$done_n" -eq "$prev_done" ] && [ "$workers" -gt 0 ] \
            && say "  ESCALATE: no fold finished in $((INTERVAL / 60)) min with $workers workers up"
    fi

    # finished folds that produced no scored row -- the NaN / scoring-failure gap
    if [ "$done_n" -gt "$scored" ] && [ "$done_n" -gt 0 ]; then
        $PY - "$E" <<'PYGAP' >> "$LOG" 2>&1
import csv, glob, json, os, sys
E = sys.argv[1]
scored = set()
try:
    for r in csv.DictReader(open(f"{E}/results.csv")):
        if r.get("dockq"):
            scored.add((r["seed"], r["arm"]))
except FileNotFoundError:
    pass
gap = []
for rj in glob.glob(f"{E}/poses/*/*/run.json"):
    try:
        j = json.load(open(rj))
    except Exception:
        continue
    key = (str(j.get("seed")), str(j.get("arm")))
    if key not in scored:
        gap.append(f'{j.get("pdb")}:{key[0]}:{key[1]}')
if gap:
    print(f"  ESCALATE: {len(gap)} finished folds produced NO scored row "
          f"(NaN coordinates or a scoring failure) -- {' '.join(sorted(gap)[:12])}"
          f"{' ...' if len(gap) > 12 else ''}")
PYGAP
    fi

    [ "$oom" -gt "$prev_oom" ] && say "  ESCALATE: $oom out-of-memory lines"
    used_g=${used%G}
    [ "${used_g%%.*}" -ge 950 ] 2>/dev/null && say "  ESCALATE: disk at $used"

    (cd "$C" && $PY score_experiment.py --exp "$E" --dataset "$DATASET" 2>&1 | tail -n 8 >> "$LOG")
    prev_done=$done_n; prev_oom=$oom

    if [ "$done_n" -ge "$total" ] && [ "$(tmux ls 2>/dev/null | grep -c "^${PREFIX}")" -eq 0 ]; then
        (cd "$C" && $PY score_experiment.py --exp "$E" --dataset "$DATASET" \
            --out "$REPORT" >> "$LOG" 2>&1)
        say "DONE: queue finished ($done_n folds, $scored scored, $failed FAILED lines); final report written; watcher exiting"
        exit 0
    fi
    say "---"
    sleep "$INTERVAL"
done
