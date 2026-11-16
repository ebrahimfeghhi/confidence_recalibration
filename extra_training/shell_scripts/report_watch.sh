#!/bin/bash
# TURN THE TRAINER'S OWN EVAL SHARDS INTO THE REPORT TABLES, on CPU, as they land.
#
# WHAT IT REPLACES. While the arm ran as segments, shell_scripts/score_checkpoint.sh ran the
# evaluation AND printed the table. With the arm in one process the trainer runs the evaluation
# itself at every epoch boundary and writes training/runs/<rid>/<pool>_eval_shards_ep<N>/, so the
# only missing half is the table. run_logging/score_report.py reads those shard rows directly --
# score_checkpoint.sh's own header says so -- and reading them costs no GPU.
#
# A DIRECTORY IS NOT DONE WHEN IT EXISTS. Each shard writes rows_*.csv and then its
# rows_*.csv.meta.json, so the meta files are the completion marker; three of them means all three
# shards finished. Reporting on a partial directory is how a 141-item table becomes a 90-item one.
#
# REPORTS AND NEVER ACTS: no kills, no launches, no deletes.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
SD="${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
TAGP="${TAGP:-lr4}"
RID="${RID:?pass RID=<the run id in one process>}"
SLEEP="${SLEEP:-300}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
say "watching $RID for valfull83/slice280 shard dirs -> \$SD/report_${TAGP}_ep<N>.txt"

while :; do
    for D in training/runs/$RID/valfull83_eval_shards_ep*; do
        [ -d "$D" ] || continue
        E=$(basename "$D" | sed -E 's/.*_ep([0-9]+)$/\1/')
        OUT="$SD/report_${TAGP}_ep$E.txt"
        [ "$(ls "$D"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ] || continue
        # THE SLICE ONLY IF IT RAN. It is on a ten-epoch cadence, so most epochs are val only and
        # score_report.py takes the val side alone.
        S=training/runs/$RID/slice280_eval_shards_ep$E
        ARGS=(--val-work "$D" --label "${TAGP}_ep$E")
        SLICE_READY=no
        if [ -d "$S" ] && [ "$(ls "$S"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ]; then
            ARGS=(--slice-work "$S" "${ARGS[@]}"); SLICE_READY=yes
        fi
        # A WRITTEN REPORT IS NOT NECESSARILY A FINISHED ONE. The two passes of a slice epoch
        # finish minutes apart -- val first, then the slice -- so the first tick after val lands
        # writes a val-only table, and "skip if the file exists" then meant the slice NEVER
        # reached the report. That is six of sixty epochs silently losing the in-sample side.
        # Rewrite exactly when the slice has since completed and the table on disk predates it.
        if [ -s "$OUT" ]; then
            [ "$SLICE_READY" = yes ] || continue
            grep -q "^TRAIN SLICE" "$OUT" && continue
            say "epoch $E: slice finished after the val-only table, rewriting"
        fi
        # AND DO NOT WRITE A VAL-ONLY TABLE WHILE A SLICE PASS IS STILL RUNNING, so the common
        # case costs no rewrite at all: the directory exists as soon as the pass starts.
        if [ "$SLICE_READY" = no ] && [ -d "$S" ] && [ ! -s "$OUT" ]; then
            continue
        fi
        if $PY -I run_logging/score_report.py "${ARGS[@]}" > "$OUT.part" 2>"$OUT.err"; then
            mv "$OUT.part" "$OUT"; rm -f "$OUT.err"
            say "epoch $E -> $(basename "$OUT")"
            grep -E "^VAL \(141|^TRAIN SLICE" "$OUT" | sed 's/^/    /'
        else
            # LEAVE THE EVIDENCE. An empty report is the failure mode that cost ten scoring
            # stops earlier today, so a failed read keeps its stderr and does NOT write $OUT.
            say "epoch $E: score_report.py failed, see $OUT.err"; head -3 "$OUT.err" | sed 's/^/    /'
        fi
    done
    sleep "$SLEEP"
done
