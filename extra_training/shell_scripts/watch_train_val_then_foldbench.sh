#!/bin/bash
# Train/validate FIRST, FoldBench LAST. Supersedes watch_foldbench_then_training.sh.
#
# WHY THE ORDER CHANGED. Measured, the remaining work is ~26.5 h: FoldBench ~18.7 h, the training
# cache 5.2 h, validation 2.6 h. FoldBench is 70% of that and it is the one stage the confidence-head
# project is NOT waiting on -- it is the benchmark top-off. The training cache and the validation
# folds are what everything downstream needs, and together they are 7.8 h. Running them first puts
# them in hand the same morning instead of a day and a half later, and the total is unchanged
# because it is the same work in a different order.
#
# The earlier 56.6 h estimate for FoldBench was wrong, and wrong in an instructive way: the queue is
# not sorted and the three workers had happened to spend their whole first 3.5 h on the two LARGEST
# complexes in it, 7zau at 2,520 residues and 7z2m at 2,244. Extrapolating a rate measured there
# across a queue whose median is 1,230 residues overstated it threefold. A quadratic-in-residues
# model, anchored on those two and checked against them (11.5 GPU-h predicted vs 10.6 actual, 8%
# over), gives ~18.7 h for the 576 that remain.
#
# FOLDBENCH IS DRAINED, NOT KILLED. folder.py has no stop file and no graceful exit, so a worker
# killed mid-fold loses that fold outright -- its poses do not reproduce. The switch therefore waits
# for every claimed-but-unwritten unit to finish before stopping the workers. FoldBench is
# claim-based and resumes where it stopped, so nothing else is lost by pausing it here.
#
# NOTHING IS TORN DOWN WITHOUT CONFIRMED COMPLETION. Each stage's teardown is gated on
# phase_j_stage_complete.py, which requires the trunk AND exactly one finished pose directory AND
# every expected score row -- the trunk alone is written during inference, before the poses are
# finalised and scored, and under --overlap-scoring the scoring is deferred to a background thread.
# A stage that does not confirm completion STOPS THE WATCHER and leaves its workers running; it
# never kills them and never switches stages. Resuming is re-running this script.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B=$R/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds
EG=$R/extra_training/training/fold_regen
VG=$R/extra_training/validation/fold_regen
PY=/home/ebrahim/Protenix/.venv/bin/python
CHK=$R/extra_training/phases/phase_j_stage_complete.py
LOG=$B/watcher.log
GPUS="0 1 3"
STALL_POLLS=${STALL_POLLS:-9}            # 9 x 300 s = 45 min
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

launch_fb(){ tmux new-session -d -s fbtop_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold $PY $R/run_model/folder.py --queue $B/queue.tsv --out $B/poses \
      --csv $B/results.csv --claim-dir $B/claims --dataset foldbench --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $B/w$1.log 2>&1"; }
launch_tr(){ tmux new-session -d -s tcache_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$EG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $EG/queue.tsv --out $EG/poses \
      --csv $EG/results.csv --claim-dir $EG/claims --dataset sabdab_train --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $EG/w$1.log 2>&1"; }
launch_va(){ tmux new-session -d -s vfold_$1 "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$1 \
      LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$VG/trunks PROTENIX_DUMP_TRUNK_BF16=1 \
      PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/run_model/folder.py --queue $VG/queue.tsv --out $VG/poses \
      --csv $VG/results.csv --claim-dir $VG/claims --dataset sabdab_val2 --ncycle 10 --nsample 5 \
      --dtype bf16 --overlap-scoring >> $VG/w$1.log 2>&1"; }

# in-flight = a claim whose unit has no run.json yet. That is a fold on a GPU right now.
inflight(){ $PY - "$1" <<'PYX'
import glob, os, sys
B = sys.argv[1]
done = {(p.split("/")[-3], p.split("/")[-2].split("-")[0])
        for p in glob.glob(f"{B}/poses/*/*/run.json")}
n = 0
for c in glob.glob(f"{B}/claims/*"):
    f = os.path.basename(c).split("_")
    if (f[0], f[-1]) not in done:
        n += 1
print(n)
PYX
}

drain_foldbench(){
    local alive n i
    alive=$(tmux ls 2>/dev/null | grep -cE '^fbtop_[013]:')
    [ "$alive" -eq 0 ] && { say "FoldBench not running; nothing to drain"; return 0; }
    say "draining FoldBench: letting the in-flight folds finish before stopping the workers"
    for i in $(seq 1 60); do                       # up to 2 h; a 2,520-residue fold is ~20 min
        n=$(inflight "$B")
        [ -z "$n" ] && n=0
        if [ "$n" -eq 0 ]; then
            say "  no fold in flight; stopping the fbtop workers"
            for g in $GPUS; do tmux kill-session -t fbtop_$g 2>/dev/null; done
            sleep 10
            $PY $R/run_model/reconcile_stale_claims.py $B --dataset foldbench --nsample 5 >> "$LOG" 2>&1
            return 0
        fi
        say "  $n fold(s) still in flight (check $i/60)"
        sleep 120
    done
    say "DRAIN TIMED OUT with folds still in flight -- NOT killing them. Nothing switched."
    return 1
}

watch_stage(){   # $1 label  $2 fold dir  $3 queue  $4 total  $5 prefix  $6 dataset  $7 extra flags
    local last=-1 stalled=0 n tot alive q cc fresh lf
    say "watching $1: $4 units (complete = trunk + one finished pose dir + all score rows)"
    while :; do
        cc=$($PY "$CHK" --fold "$2" --queue "$3" --dataset "$6" --nsample 5 $7 2>/dev/null \
             | sed -n 's|.*: \([0-9]*\)/\([0-9]*\) units COMPLETE.*|\1 \2|p')
        n=${cc%% *}; tot=${cc##* }
        [ -z "$n" ] && { say "$1: completion check did not report -- investigate"; n=0; }
        alive=$(tmux ls 2>/dev/null | grep -cE "^$5_[013]:")
        q=$(quota -s 2>/dev/null | awk 'NR==3{print $2}')
        if [ "$n" -ge "$4" ]; then say "$1 COMPLETE: $n/$4 fully done, quota $q"; return 0; fi
        if [ "$alive" -lt 3 ]; then
            say "only $alive/3 $5 workers at $n/$4 -- reconciling and relaunching"
            $PY $R/run_model/reconcile_stale_claims.py "$2" --dataset "$6" --nsample 5 >> "$LOG" 2>&1
            for g in $GPUS; do
                if ! tmux has-session -t ${5}_$g 2>/dev/null; then
                    case "$5" in tcache) launch_tr $g;; vfold) launch_va $g;; fbtop) launch_fb $g;; esac
                    say "  relaunched ${5}_$g"
                fi
            done
            stalled=0
        elif [ "$n" -eq "$last" ]; then
            stalled=$((stalled+1))
            [ $((stalled % 3)) -eq 0 ] && say "$1: no newly completed unit for $((stalled*5)) min at $n/$4"
            # a worker log written inside the window means a fold is RUNNING, however flat the count
            fresh=0
            for lf in "$2"/w*.log; do
                [ -f "$lf" ] || continue
                [ $(( $(date +%s) - $(stat -c %Y "$lf") )) -lt $(( STALL_POLLS * 300 )) ] && fresh=1
            done
            if [ "$stalled" -ge "$STALL_POLLS" ] && [ "$fresh" -eq 0 ]; then
                say "$1: no completed unit AND no worker log for $((stalled*5)) min."
                say "$1: $n of $4 complete. NOT switching stages and NOT killing anything."
                $PY "$CHK" --fold "$2" --queue "$3" --dataset "$6" --nsample 5 $7 --verbose >> "$LOG" 2>&1
                return 2
            fi
        else
            stalled=0; say "$1 $n/$4 complete, $alive workers, quota $q"
        fi
        last=$n; sleep 300
    done
}

say "=== REORDERED: training cache -> validation -> FoldBench ==="
drain_foldbench || exit 1

say "stage 1: the training cache, $(wc -l < $EG/queue.tsv) units over 74 structures"
$PY $R/run_model/reconcile_stale_claims.py $EG --dataset sabdab_train --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_tr $g; say "  launched tcache_$g"; done
if ! watch_stage "training cache" "$EG" "$EG/queue.tsv" "$(wc -l < $EG/queue.tsv)" tcache sabdab_train ""; then
    say "training cache did NOT confirm completion. LEAVING THE tcache WORKERS RUNNING."
    say "No stage switch, nothing killed. Investigate, then re-run this script to resume."
    exit 1
fi
for g in $GPUS; do tmux kill-session -t tcache_$g 2>/dev/null; done
sleep 10

say "stage 2: validation, $(wc -l < $VG/queue.tsv) units over 3 structures"
$PY $R/run_model/reconcile_stale_claims.py $VG --dataset sabdab_val2 --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_va $g; say "  launched vfold_$g"; done
if ! watch_stage "validation" "$VG" "$VG/queue.tsv" "$(wc -l < $VG/queue.tsv)" vfold sabdab_val2 ""; then
    say "validation did NOT confirm completion. LEAVING THE vfold WORKERS RUNNING, nothing killed."
    exit 1
fi
for g in $GPUS; do tmux kill-session -t vfold_$g 2>/dev/null; done
sleep 10

say "stage 3: FoldBench, resuming where the drain left it"
$PY $R/run_model/reconcile_stale_claims.py $B --dataset foldbench --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_fb $g; say "  launched fbtop_$g"; done
# --no-trunk: FoldBench dumps no trunks, so complete there is poses + every score row
if ! watch_stage "FoldBench" "$B" "$B/queue.tsv" "$(wc -l < $B/queue.tsv)" fbtop foldbench "--no-trunk"; then
    say "FoldBench did NOT confirm completion. LEAVING THE fbtop WORKERS RUNNING, nothing killed."
    exit 1
fi

say "ALL RUNNABLE WORK DONE. The refolds are NOT queued: 1,180 training and 1,300 validation units"
say "wait on the corrected MSAs (deferred_refold.csv in each fold dir). GPU holders are untouched."
say "done"
