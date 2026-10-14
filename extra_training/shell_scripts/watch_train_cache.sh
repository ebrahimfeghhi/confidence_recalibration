#!/bin/bash
# Watch the FT3 training-cache fold, remediate, and hand the cards to the FoldBench top-off at the end.
#
# REMEDIATES, does not just print: a dead worker is relaunched, and a stall is reported loudly. The
# fold is claim-based, so a relaunched worker picks up whatever is unclaimed; units claimed by a
# worker that died are released by reconcile_stale_claims.py, which runs before the relaunch.
#
# HANDS OFF rather than releasing the GPUs: when the queue is exhausted the FoldBench completion
# top-off (foldbench_baseline_remaining_folds, 612 folds) starts on the same three cards, so they are
# never idle. It is claim-based too, so restarting it costs nothing.
set -u
# the repo root, derived from this file so a rename of the checkout needs no edit here
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
E=$R/extra_training/training/fold
B=$R/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds
LOG=$E/watcher.log
GPUS="0 1 3"
TOTAL=$(wc -l < $E/queue.tsv)
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

launch_train() {   # $1 = gpu
    tmux new-session -d -s tcache_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$E/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 \
      $R/../Protenix/.venv/bin/python $R/run_model/folder.py --queue $E/queue.tsv \
      --out $E/poses --csv $E/results.csv --claim-dir $E/claims --dataset sabdab_train \
      --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $E/w$1.log 2>&1"
}
launch_fb() {      # $1 = gpu
    tmux new-session -d -s fbtop_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold \
      /home/ebrahim/Protenix/.venv/bin/python $R/run_model/folder.py \
      --queue $B/queue.tsv --out $B/poses --csv $B/results.csv --claim-dir $B/claims \
      --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $B/w$1.log 2>&1"
}

say "watching the training cache: $TOTAL folds, 501 structures x 10 seeds"
last=0; stalled=0
while :; do
    n=$(find $E/poses -name run.json 2>/dev/null | wc -l)
    # COUNT THE WORKERS, NOT THE WATCHER. '^tcache_' also matches tcache_watch, this script's own
    # session, so a dead worker still left 3 and the relaunch below never fired -- the remediation
    # would have been silently disabled for exactly the failure it exists for.
    alive=$(tmux ls 2>/dev/null | grep -cE '^tcache_[013]:')
    q=$(quota -s 2>/dev/null | awk 'NR==3{print $2}')
    # COMPLETION IS THE COUNT, NOT THE ABSENCE OF WORKERS. Treating "every worker exited" as done
    # handed the cards to FoldBench on an INCOMPLETE cache -- which is what a crash looks like. Only
    # reaching TOTAL ends the loop; everything else is remediated below.
    if [ "$n" -ge "$TOTAL" ]; then
        say "training cache COMPLETE: $n/$TOTAL folds, quota $q"
        break
    fi
    if [ "$alive" -eq 0 ]; then
        say "ALL WORKERS GONE at $n/$TOTAL -- reconciling and restarting, not handing over"
    fi
    # RELAUNCH a missing worker, after releasing whatever it had claimed.
    if [ "$alive" -lt 3 ]; then
        say "only $alive/3 workers alive at $n/$TOTAL -- reconciling and relaunching"
        /home/ebrahim/Protenix/.venv/bin/python $R/run_model/reconcile_stale_claims.py \
            $E --dataset sabdab_train >> $LOG 2>&1
        for g in $GPUS; do
            tmux has-session -t tcache_$g 2>/dev/null || { launch_train $g; say "  relaunched GPU $g"; }
        done
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

say "handing the cards to the FoldBench top-off ($(wc -l < $B/queue.tsv) folds queued)"
for g in $GPUS; do launch_fb $g; say "  launched fbtop_$g"; done
say "done"
