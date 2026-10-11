#!/bin/bash
# Every 30 minutes: is the 10-seed generation alive and is it producing usable poses?
#
# WHAT IT WATCHES. This arm is pure generation -- no method, no comparison -- so health is the whole
# question: folds landing, rows scored, and no silent-failure class. The three ways this run can go
# wrong without crashing are all printed:
#   DEAD WORKER   a GPU's session gone while its queue is unfinished -> relaunched (claims make that
#                 safe; folder.py releases a dead worker's claim at startup)
#   NO DOCKQ      folds that finished but scored no DockQ -- the silent-NaN class, which looks like
#                 progress in the fold count and is worthless
#   STALL         workers up but no new fold in 30 minutes
#
# It also prints the per-interface pair_iptm table for whatever has been scored, because every
# scored interface gets its ipTM reported.
set -u
ROOT=/home/ebrahim/tts_abag/experimental_data/protenix
C=/home/ebrahim/tts_abag/diffusion_commit_clean_v2
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS="${GPUS:-0 1}"
LOG=$ROOT/free_extra10_watch.log
INTERVAL="${INTERVAL:-1800}"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
PREV=""

while :; do
  [ -f "$ROOT/free_extra10_DONE" ] && { say "chain reports ALL PHASES COMPLETE -- watcher stopping"; break; }
  CUR=""
  for DS in foldbench pxm2025; do
    E=$ROOT/$DS/free_extra10
    [ -f $E/queue.tsv ] || continue
    TOTAL=$(wc -l < $E/queue.tsv)
    DONE=$(find $E/poses -name run.json 2>/dev/null | wc -l)
    ROWS=$(( $(wc -l < $E/results.csv 2>/dev/null || echo 1) - 1 ))
    FAIL=$(grep -c "FAILED" $E/w*.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
    say "$DS: folds $DONE/$TOTAL   rows $ROWS   failed $FAIL"
    CUR="$CUR $DS:$DONE"

    # remediate: a worker session gone while this phase still has work AND the phase is the live one
    if [ "$DONE" -lt "$TOTAL" ]; then
      for g in $GPUS; do
        if tmux has-session -t fx_${DS}_$g 2>/dev/null; then :; else
          # only revive a phase that has actually started (its log exists), never start the next one early
          if [ -f $E/w$g.log ]; then
            tmux new-session -d -s fx_${DS}_$g "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$g \
              LAYERNORM_TYPE=openfold $PY $C/folder.py --queue $E/queue.tsv --out $E/poses \
              --csv $E/results.csv --claim-dir $E/claims --dataset $DS --ncycle 10 --nsample 5 \
              --dtype bf16 --overlap-scoring >> $E/w$g.log 2>&1"
            say "  REMEDIATE: relaunched $DS worker on GPU $g"
          fi
        fi
      done
      for g in $GPUS; do
        [ -f $E/w$g.log ] || continue
        if grep -qE "CUDA out of memory|Traceback" $E/w$g.log; then
          say "  ESCALATE: failure signature in $DS/w$g.log"
        fi
      done
    fi

    # silent-NaN class: finished folds that scored nothing
    if [ "$DONE" -gt 0 ] && [ "$ROWS" -eq 0 ]; then
      say "  ESCALATE: $DONE folds finished but 0 scored rows in $DS"
    fi
  done

  [ "$CUR" = "$PREV" ] && say "  ESCALATE: no new fold in ${INTERVAL}s (workers may be stalled)"
  PREV="$CUR"

  # the numbers: pair_iptm and DockQ per interface, which is what these folds exist to produce
  $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import csv, os, statistics as st, collections
ROOT="/home/ebrahim/tts_abag/experimental_data/protenix"
for DS in ("foldbench","pxm2025"):
    p=f"{ROOT}/{DS}/free_extra10/results.csv"
    if not os.path.exists(p): continue
    rows=[r for r in csv.DictReader(open(p)) if r.get("dockq") and r.get("iptm")]
    if not rows: continue
    nan=sum(1 for r in csv.DictReader(open(p)) if not r.get("dockq"))
    by=collections.defaultdict(list)
    for r in rows: by[r["iface"]].append((float(r["dockq"]), float(r["iptm"])))
    ip=[i for v in by.values() for _,i in v]; dq=[d for v in by.values() for d,_ in v]
    acc=sum(1 for v in by.values() if max(d for d,_ in v)>=0.23)
    sel=sum(1 for v in by.values() if max(v,key=lambda x:x[1])[0]>=0.23)
    print(f"  {DS}: interfaces {len(by)}  poses {len(rows)}  unscored rows {nan}")
    print(f"    pair_iptm median {st.median(ip):.3f}  DockQ median {st.median(dq):.3f}  "
          f"argmax-iptm acceptable {sel}/{len(by)}  oracle acceptable {acc}/{len(by)}")
PYEOF
  sleep "$INTERVAL"
done
