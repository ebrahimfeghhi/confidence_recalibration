#!/bin/bash
# Watch the FoldBench top-off, remediate it, then hand the cards back to the TRIMMED training queue.
#
# WHY THIS EXISTS. The switch of 2026-10-14 stopped tcache_watch along with its workers, so the
# FoldBench workers it launched had nothing watching them: a worker killed mid-fold would have left
# its claim behind and one card idle for the rest of the night, with the queue looking busy.
#
# WHAT IT HANDS OVER TO, in order: FoldBench -> the training queue -> the validation queue. None of
# these are the refolds. Those need corrected MSAs that do not exist yet, and both queues were
# rebuilt by phase_j_queue_now.py so that no row of a structure whose query is changing can appear
# in either one -- folding a stale query unattended is exactly what this ordering prevents. The
# runnable work is 733 training units over 74 structures and 300 validation units over 3, and the
# 2,480 units that are waiting on the MSA rebuild are recorded in each fold dir's
# deferred_refold.csv rather than sitting in a queue file where a worker could pick them up.
#
# THE VALIDATION DATASET IS sabdab_val2, NOT sabdab_val. paths.py has no "sabdab_val" entry at all;
# phase_i_fold_validation.sh still DEFAULTS to that dead name and only works because its caller
# overrides DATASET. Spelling it wrong here would abort every validation fold on a KeyError after
# the training queue had already drained and the cards had been handed over.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B=$R/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds
E=$R/extra_training/training/fold
PY=/home/ebrahim/Protenix/.venv/bin/python
V=$R/extra_training/validation/fold
LOG=$B/watcher.log
GPUS="0 1 3"
TR_TOTAL=$(wc -l < $E/queue.tsv)
VA_TOTAL=$(wc -l < $V/queue_now.tsv)
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

launch_va(){ tmux new-session -d -s vfold_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$V/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $V/queue_now.tsv --out $V/poses \
      --csv $V/results.csv --claim-dir $V/claims --dataset sabdab_val2 --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $V/w$1.log 2>&1"; }

# DONE IS THE TRUNK COUNT FOR THIS QUEUE'S OWN ROWS, not a pose count and not the absence of
# workers. The trunk npz is the artefact this phase exists to produce, folder.py refuses to
# overwrite one, and poses for these structures already exist from the folds whose trunks were
# deleted -- so counting poses would report the training queue finished before a single trunk had
# been written.

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

say "handing the cards to the training queue: $TR_TOTAL units over 74 structures"
for g in $GPUS; do tmux kill-session -t fbtop_$g 2>/dev/null; done
sleep 10
$PY $R/run_model/reconcile_stale_claims.py $E --dataset sabdab_train --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_tr $g; say "  launched tcache_$g"; done

# ---- stage 2: the training queue, then the validation queue --------------------------------------
watch_stage(){   # $1 label  $2 fold dir  $3 queue  $4 total  $5 session prefix  $6 dataset
    local last=0 stalled=0 n alive q
    say "watching $1: $4 units"
    while :; do
        n=$(awk -F'\t' -v d="$2" '{if (system("test -e " d "/trunks/" $2 "/" $1 "_" $5 ".npz")==0) c++} END{print c+0}' "$3")
        alive=$(tmux ls 2>/dev/null | grep -cE "^$5_[013]:")
        q=$(quota -s 2>/dev/null | awk 'NR==3{print $2}')
        if [ "$n" -ge "$4" ]; then say "$1 COMPLETE: $n/$4, quota $q"; return 0; fi
        if [ "$alive" -lt 3 ]; then
            say "only $alive/3 $5 workers at $n/$4 -- reconciling and relaunching"
            $PY $R/run_model/reconcile_stale_claims.py "$2" --dataset "$6" --nsample 5 >> "$LOG" 2>&1
            for g in $GPUS; do
                tmux has-session -t ${5}_$g 2>/dev/null || { [ "$5" = tcache ] && launch_tr $g || launch_va $g; say "  relaunched ${5}_$g"; }
            done
        fi
        if [ "$n" -eq "$last" ]; then
            stalled=$((stalled+1))
            [ $((stalled % 9)) -eq 0 ] && say "$1: NO PROGRESS for $((stalled*5)) min at $n/$4 -- investigate"
        else stalled=0; say "$1 $n/$4 units, $alive workers, quota $q"; fi
        last=$n; sleep 300
    done
}

watch_stage "training cache" "$E" "$E/queue.tsv" "$TR_TOTAL" tcache sabdab_train

say "handing the cards to the VALIDATION queue: $VA_TOTAL units over 3 structures"
for g in $GPUS; do tmux kill-session -t tcache_$g 2>/dev/null; done
sleep 10
$PY $R/run_model/reconcile_stale_claims.py $V --dataset sabdab_val2 --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_va $g; say "  launched vfold_$g"; done
watch_stage "validation" "$V" "$V/queue_now.tsv" "$VA_TOTAL" vfold sabdab_val2

say "ALL RUNNABLE WORK DONE. The refolds are NOT queued: 1,180 training and 1,300 validation units"
say "wait on the corrected MSAs (deferred_refold.csv in each fold dir). GPU holders are untouched."
say "done"
