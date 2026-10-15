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
# TRUNK REGENERATION WRITES INTO A FRESH ROOT, NOT BACK INTO THE OLD ONE.
#
# folder.py never reaches the folding path for a unit that already has poses or scores: it returns
# "skipped" at folder.py:931 when every interface is scored, and "rescored" at :968 when
# saved_run_dir finds a finished pose directory. Neither path runs inference, so neither writes a
# trunk. Measured against the two queues: 73 of the 733 training units and ALL 300 validation units
# have poses on disk from the folds whose trunks were deleted, so a same-root run would have left
# the watcher waiting on 373 trunks that can never appear -- hanging at 660/733 and 0/300 with three
# idle cards.
#
# AND A NEW TRUNK MUST NOT BE PAIRED WITH AN OLD POSE. Repeating a seed does not reproduce its
# poses; the same seed refolded has gone from DockQ 0.022 to 0.644. A trunk is only meaningful
# alongside the pose it was computed from, so the regenerated poses and their trunks go in one fresh
# tree together. phase_j_build_cache.py:84 already refuses a (pdb, seed) with two finished pose
# directories rather than guessing, so a same-root refold would have failed the cache build for
# those 8 structures anyway. It takes --fold, so each root is built separately and each is
# internally consistent.
EG=$R/extra_training/training/fold_regen
VG=$R/extra_training/validation/fold_regen
LOG=$B/watcher.log
GPUS="0 1 3"
TR_TOTAL=$(wc -l < $EG/queue.tsv)
VA_TOTAL=$(wc -l < $VG/queue.tsv)
TOTAL=$(wc -l < $B/queue.tsv)
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

# COMPLETION IS NOT THE TRUNK COUNT. folder.py:1034 passes PROTENIX_DUMP_TRUNK into the fold, so the
# trunk is written DURING inference, while the poses are finalised and the interfaces scored
# afterwards -- and under --overlap-scoring the scoring runs on a BACKGROUND thread after the fold
# returns. Counting trunks therefore reports COMPLETE while the last units are still being written,
# and the stage switch then kills the workers mid-unit. Reproduced: a trunk present with no poses
# passed the old check, which is exactly the unit the cache builder cannot use, since it needs the
# poses that trunk was computed from.
#
# phase_j_stage_complete.py requires, per queued unit, the trunk AND exactly one finished pose
# directory AND every expected score row, reusing folder.py's own iface_scored and
# interfaces_by_pdb so the watcher cannot disagree with the worker about what "scored" means.
#
# AND IT CANNOT WAIT FOREVER. Some units never score completely: 10op_FN has zero samples across
# every seed and 7sk4 is incomplete on 8 consecutive seeds, and a partially scored unit KEEPS its
# claim by design so it is never retried. A strict AND with no escape would hang the handoff on
# those -- the same failure inverted. So a stage also ends when the completed count has not moved
# for STALL_POLLS polls while all three workers are alive, and then the stragglers are NAMED in the
# log rather than passed over silently. Claim count is not a liveness signal: claims persist after
# success (3,323 of them against 3,097 trunks), so progress is measured by the completed count.
STALL_POLLS=${STALL_POLLS:-9}            # 9 x 300 s = 45 min, the same threshold used above
complete_count(){                        # $1 fold dir  $2 queue  $3 dataset -> "<done> <total>"
    $PY $R/extra_training/phases/phase_j_stage_complete.py --fold "$1" --queue "$2" \
        --dataset "$3" --nsample 5 2>/dev/null \
        | sed -n 's|.*: \([0-9]*\)/\([0-9]*\) units COMPLETE.*|\1 \2|p'
}

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

# STAGE 1 HAS THE SAME ORDERING PROBLEM, so its teardown gets the same gate. The loop above breaks
# on a POSE count, and folder.py:1074 writes run.json before finish_unit scores the unit -- deferred
# to a background thread under --overlap-scoring -- so the count reaches 612 while the last folds
# are still scoring. Killing then loses those score rows. Less costly than the training case, where
# a killed fold loses a trunk that no rescore can rebuild, but the same bug and the same fix.
# FoldBench dumps no trunks, hence --no-trunk.
fb_done=0
for i in $(seq 1 12); do
    if $PY $R/extra_training/phases/phase_j_stage_complete.py --fold $B --queue $B/queue.tsv \
         --dataset foldbench --nsample 5 --no-trunk >> "$LOG" 2>&1; then fb_done=1; break; fi
    say "FoldBench: poses complete but scoring still finishing; waiting (check $i/12)"
    sleep 120
done
if [ "$fb_done" -ne 1 ]; then
    say "FoldBench did NOT confirm poses+scores complete. LEAVING THE fbtop WORKERS RUNNING."
    say "No stage switch, nothing killed. Investigate, then re-run this script to resume."
    exit 1
fi
say "handing the cards to the training queue: $TR_TOTAL units over 74 structures"
for g in $GPUS; do tmux kill-session -t fbtop_$g 2>/dev/null; done
sleep 10
$PY $R/run_model/reconcile_stale_claims.py $EG --dataset sabdab_train --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_tr $g; say "  launched tcache_$g"; done

# ---- stage 2: the training queue, then the validation queue --------------------------------------
watch_stage(){   # $1 label  $2 fold dir  $3 queue  $4 total  $5 session prefix  $6 dataset
    local last=-1 stalled=0 n tot alive q cc
    say "watching $1: $4 units (complete = trunk + one finished pose dir + all score rows)"
    while :; do
        cc=$(complete_count "$2" "$3" "$6"); n=${cc%% *}; tot=${cc##* }
        [ -z "$n" ] && { say "$1: completion check failed to report -- investigate"; n=0; }
        alive=$(tmux ls 2>/dev/null | grep -cE "^$5_[013]:")
        q=$(quota -s 2>/dev/null | awk 'NR==3{print $2}')
        if [ "$n" -ge "$4" ]; then say "$1 COMPLETE: $n/$4 fully done, quota $q"; return 0; fi
        if [ "$alive" -lt 3 ]; then
            say "only $alive/3 $5 workers at $n/$4 -- reconciling and relaunching"
            $PY $R/run_model/reconcile_stale_claims.py "$2" --dataset "$6" --nsample 5 >> "$LOG" 2>&1
            for g in $GPUS; do
                tmux has-session -t ${5}_$g 2>/dev/null || { [ "$5" = tcache ] && launch_tr $g || launch_va $g; say "  relaunched ${5}_$g"; }
            done
            stalled=0
        elif [ "$n" -eq "$last" ]; then
            stalled=$((stalled+1))
            [ $((stalled % 3)) -eq 0 ] && say "$1: no newly completed unit for $((stalled*5)) min at $n/$4"
            # A STALLED COUNT IS NOT AN IDLE WORKER. The completed count cannot move while three
            # workers are each part-way through a long fold, and the slowest observed here is 1,208 s
            # on FoldBench and 843 s on the training cache -- but nothing guarantees a bigger complex
            # will not exceed the window, and three workers starting long folds together stalls the
            # count for as long as the longest of them. A worker log written inside the window means
            # a fold is RUNNING, so the stage is not stalled however flat the count looks.
            local fresh=0 lf
            for lf in "$2"/w*.log; do
                [ -f "$lf" ] || continue
                [ $(( $(date +%s) - $(stat -c %Y "$lf") )) -lt $(( STALL_POLLS * 300 )) ] && fresh=1
            done
            if [ "$stalled" -ge "$STALL_POLLS" ] && [ "$fresh" -eq 0 ]; then
                say "$1: no completed unit AND no worker log written for $((stalled*5)) min."
                say "$1: $n of $4 complete. NOT switching stages and NOT killing anything."
                $PY $R/extra_training/phases/phase_j_stage_complete.py --fold "$2" --queue "$3" \
                    --dataset "$6" --nsample 5 --verbose >> "$LOG" 2>&1
                return 2
            fi
        else
            stalled=0; say "$1 $n/$4 complete, $alive workers, quota $q"
        fi
        last=$n; sleep 300
    done
}

# A STAGE THAT DID NOT COMPLETE MUST NOT BE TORN DOWN. The return code was previously discarded
# and `tmux kill-session -t tcache_$g` ran unconditionally, so a stage that gave up killed workers
# that might each have been mid-fold -- and a killed fold's poses do not reproduce, so that is
# unrecoverable work lost, strictly worse than waiting. Logging the unfinished units does not make
# terminating them safe. The watcher now STOPS and leaves the workers running; stages switch only
# after confirmed completion, and resuming is just re-running this script.
if ! watch_stage "training cache" "$EG" "$EG/queue.tsv" "$TR_TOTAL" tcache sabdab_train; then
    say "training cache did NOT confirm completion. LEAVING THE tcache WORKERS RUNNING."
    say "No stage switch, nothing killed. Investigate, then re-run this script to resume."
    exit 1
fi

say "handing the cards to the VALIDATION queue: $VA_TOTAL units over 3 structures"
for g in $GPUS; do tmux kill-session -t tcache_$g 2>/dev/null; done
sleep 10
$PY $R/run_model/reconcile_stale_claims.py $VG --dataset sabdab_val2 --nsample 5 >> "$LOG" 2>&1
for g in $GPUS; do launch_va $g; say "  launched vfold_$g"; done
if ! watch_stage "validation" "$VG" "$VG/queue.tsv" "$VA_TOTAL" vfold sabdab_val2; then
    say "validation did NOT confirm completion. LEAVING THE vfold WORKERS RUNNING, nothing killed."
    exit 1
fi

say "ALL RUNNABLE WORK DONE. The refolds are NOT queued: 1,180 training and 1,300 validation units"
say "wait on the corrected MSAs (deferred_refold.csv in each fold dir). GPU holders are untouched."
say "done"
