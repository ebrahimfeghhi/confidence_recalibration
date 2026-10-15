#!/bin/bash
# Watch the FoldBench top-off, remediate it, then hand the cards back to the TRIMMED training queue.
#
# WHY THIS EXISTS. The switch of 2026-10-14 stopped tcache_watch along with its workers, so the
# FoldBench workers it launched had nothing watching them: a worker killed mid-fold would have left
# its claim behind and one card idle for the rest of the night, with the queue looking busy.
#
# WHAT IT HANDS OVER TO. Not the refolds -- those need corrected MSAs that do not exist yet. The
# trimmed training queue still holds 679 folds of structures that are NOT being refolded, and those
# are ready now, so they are the next useful work for the cards.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B=$R/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds
E=$R/extra_training/training/fold
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$B/watcher.log
GPUS="0 1 3"
TOTAL=$(wc -l < $B/queue.tsv)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

launch_fb(){ tmux new-session -d -s fbtop_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold $PY $R/run_model/folder.py --queue $B/queue.tsv --out $B/poses \
      --csv $B/results.csv --claim-dir $B/claims --dataset foldbench --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $B/w$1.log 2>&1"; }
launch_tr(){ tmux new-session -d -s tcache_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$E/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $E/queue.tsv --out $E/poses \
      --csv $E/results.csv --claim-dir $E/claims --dataset sabdab_train --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $E/w$1.log 2>&1"; }

say "watching the FoldBench top-off: $TOTAL folds"
last=0; stalled=0
while :; do
    n=$(find $B/poses -name run.json 2>/dev/null | wc -l)
    alive=$(tmux ls 2>/dev/null | grep -cE '^fbtop_[013]:')
    q=$(quota -s 2>/dev/null | awk 'NR==3{print $2}')
    # COMPLETION IS THE COUNT, never the absence of workers: every worker exiting is what a crash
    # looks like, and handing over then would leave the top-off unfinished and looking done.
    if [ "$n" -ge "$TOTAL" ]; then say "FoldBench COMPLETE: $n/$TOTAL, quota $q"; break; fi
    if [ "$alive" -lt 3 ]; then
        say "only $alive/3 fbtop workers at $n/$TOTAL -- reconciling and relaunching"
        $PY $R/run_model/reconcile_stale_claims.py $B --dataset foldbench --nsample 5 >> "$LOG" 2>&1
        for g in $GPUS; do tmux has-session -t fbtop_$g 2>/dev/null || { launch_fb $g; say "  relaunched fbtop_$g"; }; done
    fi
    if [ "$n" -eq "$last" ]; then
        stalled=$((stalled+1))
        [ $((stalled % 9)) -eq 0 ] && say "NO PROGRESS for $((stalled*5)) min at $n/$TOTAL -- investigate"
    else
        stalled=0; say "$n/$TOTAL folds, $alive workers, quota $q"
    fi
    last=$n
    sleep 300
done

say "handing the cards back to the TRIMMED training queue"
for g in $GPUS; do tmux kill-session -t fbtop_$g 2>/dev/null; done
sleep 10
$PY $R/run_model/reconcile_stale_claims.py $E --dataset sabdab_train --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_tr $g; say "  launched tcache_$g on the trimmed queue"; done
say "NOTE: the refolds are NOT queued -- they need the corrected MSAs, which are not built yet."
say "done"
