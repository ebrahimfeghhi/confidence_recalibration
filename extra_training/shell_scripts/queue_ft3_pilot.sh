#!/bin/bash
# WAIT FOR FOLDBENCH, VERIFY THE EVALUATOR, THEN RUN THE 3-EPOCH FT3 PILOT.
#
# WHY IT WAITS. FoldBench is using 70-83 GB of each card's 98 GB on large complexes. Training needs
# ~16 GB; starting it now risks an OOM that kills a fold worker, which is exactly how ~1,000 folds
# were lost on 2026-10-10. The GPU holders are never touched.
#
# WHY THE EVAL IS VERIFIED FIRST, with the SHIPPED head. The metric depends on three joins that can
# each be silently wrong -- interface name to chain pair, pose to sample, and (iface, seed, sample)
# to DockQ -- and the pose-to-sample one is REAL: 29tj/400000's pose 0 carries the stored ipTM of
# SAMPLE 4. phase_k_eval refuses to report a number unless the shipped head reproduces results.csv's
# own ipTM row by row. If that fails, a training curve would be uninterpretable, so the pilot does
# not start.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export R
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG="$R/ft3_pilot.log"
OUT="$R/training/ft3_pilot"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "waiting for FoldBench to finish (pgrep to WAIT, never to kill)"
while pgrep -f "[f]older.py.*foldbench_baseline" > /dev/null 2>&1; do sleep 120; done
say "FoldBench has exited"
sleep 60                      # let the cards settle before claiming memory

# ---- STAGE 0: REFOLD THE 300 VALIDATION UNITS THAT HAVE NO TRUNKS ----
# WHY. 7tzh, 8v8f and 9cn2 have poses and full scores but NO trunks, because trunk dumping stopped
# at 2026-10-12 19:00 PT and all three folded between 19:03 and 23:07 that evening. 53 validation
# structures have poses, exactly 50 have trunks, and the three without are precisely the ones folded
# after dumping stopped. They were on the EARLIER refold list but not on deferred_refold.csv, the
# 1,300-row list that actually ran, so nothing ever went back for them.
#
# The evaluation needs trunks to rescore with a changed head. Without these, the metric covers 86 of
# 92 interfaces and cannot be compared with the 92-interface baseline of 0.5783 / 0.7283 -- so the
# round-9 coverage gate refuses, which is why this stage exists rather than a weaker metric.
TF="$R/validation/fold_trunkfix"; export TF
if [ "$(find "$R/validation/fold/trunks" -name '*.npz' | wc -l)" -lt 5300 ]; then
    say "STAGE 0: refolding $(wc -l < "$TF/queue.tsv") validation unit(s) that have no trunk"
    say "  7tzh, 8v8f, 9cn2 x 100 seeds. TRUNK_DUMP_DIR is set, which is what was missing before."
    mkdir -p "$TF/poses" "$TF/trunks" "$TF/claims"
    for g in 0 1 3; do
        tmux new-session -d -s tfix_$g "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$g             LAYERNORM_TYPE=openfold TRUNK_DUMP_DIR=$TF/trunks PROTENIX_DUMP_TRUNK_BF16=1             PROTENIX_DUMP_CONF_INPUTS=1 $PY $R/../run_model/folder.py --queue $TF/queue.tsv             --out $TF/poses --csv $TF/results.csv --claim-dir $TF/claims --dataset sabdab_val2             --ncycle 10 --nsample 5 --dtype bf16 --overlap-scoring >> $TF/w$g.log 2>&1"
        say "  launched tfix_$g"
    done
    # wait for it, reporting progress; never kill a worker
    last=-1; quiet=0
    while true; do
        n=$(find "$TF/trunks" -name '*.npz' 2>/dev/null | wc -l)
        alive=$(tmux ls 2>/dev/null | grep -cE '^tfix_[013]:')
        [ "$n" -ge 300 ] && { say "STAGE 0: $n/300 trunks -- complete"; break; }
        if [ "$alive" -eq 0 ]; then
            say "STAGE 0: all workers exited at $n/300 trunks. NOT starting the pilot; the metric"
            say "  would cover 86 of 92 interfaces. Investigate $TF/w*.log."
            exit 1
        fi
        [ "$n" -eq "$last" ] && quiet=$((quiet+1)) || quiet=0
        last=$n
        [ $quiet -ge 18 ] && { say "STAGE 0: no progress for 45 min at $n/300. Leaving the $alive worker(s) alone."; exit 1; }
        sleep 150
    done
    for g in 0 1 3; do tmux kill-session -t tfix_$g 2>/dev/null; done
    sleep 15
    # merge into validation/fold so the evaluator sees one root
    $PY -I - <<'PYM' >> "$LOG" 2>&1
import os, glob
src=os.environ["TF"]; dst=os.environ["R"]+"/validation/fold"
moved={"trunks":0,"poses":0}
for kind,pat in (("trunks","trunks/*/*.npz"),("poses","poses/*/*")):
    for p in glob.glob(f"{src}/{pat}"):
        rel=os.path.relpath(p,src); d=os.path.join(dst,rel)
        if os.path.exists(d): raise SystemExit(f"COLLISION: {p}")
        os.makedirs(os.path.dirname(d), exist_ok=True); os.rename(p,d); moved[kind]+=1
print("  merged into validation/fold:", moved)
PYM
    say "STAGE 0: merged. validation/fold trunks now $(find "$R/validation/fold/trunks" -name '*.npz' | wc -l)/5300"
else
    say "STAGE 0: skipped, validation/fold already has 5,300 trunks"
fi

say "GPU state before starting:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a "$LOG"

# GATE 1: the evaluator must reproduce the fold's own ipTM with the shipped head.
say "GATE 1: evaluator self-check with the SHIPPED head (full 100-seed budget)"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_k_eval.py" \
    --budget 100 --out "$R/training/ft3_pilot_baseline.json" >> "$LOG" 2>&1
rc=$?
say "GATE 1 exit=$rc"
if [ $rc -ne 0 ]; then
    say "THE EVALUATOR DOES NOT REPRODUCE results.csv. One of its three joins is wrong, so a"
    say "training curve would be uninterpretable. NOT starting the pilot. See $LOG."
    exit 1
fi

# GATE 2: the replay, which is what makes cached training equal to FT3 at all.
say "GATE 2: replay gate"
CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_j_replay.py" \
    --unit 9g13/free_500000 > "$R/training/ft3_pilot_replay.log" 2>&1
grep -q "^PASS:" "$R/training/ft3_pilot_replay.log" || {
    say "THE REPLAY DID NOT PASS. Cached training would be optimising something other than FT3."
    say "NOT starting the pilot. See $R/training/ft3_pilot_replay.log"; exit 1; }
say "GATE 2: PASS"

# PRELOADING IS GATED ON FREE RAM AT LAUNCH, AND AGAIN PER ITEM INSIDE THE TRAINER.
# Holding all 500 structure items saves the ~57 min/epoch of rebuilds (measured: 3.6-10.4 s each)
# but costs ~253 GB (measured: 331-886 MB each, avg 518). The user's condition is at least 500 GB
# free before preloading, so it is measured here rather than assumed -- and the trainer re-checks
# MemAvailable before EVERY item it caches, because the cache grows over the first epoch and a
# startup check alone would let the box run out later and take the fold workers with it.
FREE_GB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
MIN_GB=500
if [ "$FREE_GB" -ge "$MIN_GB" ]; then
    PRELOAD=500
    say "RAM: ${FREE_GB} GB available >= ${MIN_GB} GB -- preloading all 500 structure items"
    say "  saves ~57 min/epoch of dataset-item rebuilds, costs ~253 GB"
else
    PRELOAD=1
    say "RAM: only ${FREE_GB} GB available, below ${MIN_GB} GB -- NOT preloading (--preload-items 1)"
    say "  each structure is rebuilt per epoch, ~57 min/epoch, rather than risking the fold workers"
fi

mkdir -p "$OUT"
# ---- THE SWEEP: FOUR LEARNING RATES, ONE AT A TIME ----
# SEQUENTIAL at the user's direction: no contention, each run gets a whole card for training and a
# second for its evaluation, and no run's timing is polluted by another's.
#
# ORDERED SO THE MOST USEFUL COMES FIRST. 12 epoch-times run back to back will NOT all finish by
# morning -- one LR is 3 epochs plus three 16.5-minute evaluations, which is 8 h at the optimistic
# rate and 22 h at the rate measured under contention. So 1e-3 goes first (the planned default),
# then the two brackets, then the low end. Whatever has finished by morning is complete and scored;
# nothing is half-reported.
#
# RAM: 300 GB CAP ON OUR OWN PROCESS, per the user. --preload-items 200 is ~104 GB of dataset items
# on top of roughly 110 GB of baseline, and --max-rss-gb 300 stops the cache growing past the cap
# regardless. Only one run exists at a time, so the cap applies to the whole sweep.
LRS="1e-3 3e-4 3e-3 1e-4"
SUMMARY="$R/training/ft3_sweep/summary.tsv"
mkdir -p "$R/training/ft3_sweep"
[ -f "$SUMMARY" ] || printf "lr\tepoch\tmean_dockq\tfrac_acc\toracle_mean\theadroom\tbaseline_mean\tbaseline_acc\n" > "$SUMMARY"

say "SWEEP: 4 learning rates, 3 epochs each, SEQUENTIAL. Order: $LRS"
say "  training on GPU 0, evaluation on GPU 1. Baseline to beat: mean_dockq 0.5783, frac_acc 0.7283"
say "  12 epoch-times run back to back will not all finish overnight; each completed LR is whole."

for LR in $LRS; do
    OUT="$R/training/ft3_sweep/lr$LR"
    if [ -f "$OUT/head_final.pt" ]; then say "SWEEP: lr $LR already complete, skipping"; continue; fi
    mkdir -p "$OUT"
    say "SWEEP: lr $LR starting -> $OUT"
    CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold $PY -I "$R/phases/phase_k_train.py" \
        --fold "$R/training/fold" \
        --labels "$R/training/fold/labels" \
        --epochs 3 --accum 256 --lr "$LR" --warmup-steps 20 \
        --eval-every-epoch --eval-budget 100 --eval-device 1 \
        --preload-items 200 --max-rss-gb 300 \
        --grad-check-every 10 --save-every 100 \
        --require-replay "$R/training/ft3_pilot_replay.log" \
        --wandb-project confidence-recalibration --wandb-run "ft3-lr$LR" \
        --out "$OUT" >> "$OUT/run.log" 2>&1
    rc=$?
    say "SWEEP: lr $LR exit=$rc"
    # COLLECT WHAT IT PRODUCED, whether or not it finished. A run that dies at epoch 2 still has
    # epoch 1's metric, and that is the comparison the sweep exists for.
    for mj in "$OUT"/metric_ep*.json; do
        [ -f "$mj" ] || continue
        $PY -I -c "
import json,os,sys
m=json.load(open('$mj'))
ep=os.path.basename('$mj').replace('metric_ep','').replace('.json','')
print('\t'.join(['$LR', ep, str(m['mean_dockq']), str(m['frac_acc']),
                 str(m['oracle_mean_dockq']), str(m['headroom']), '0.5783', '0.7283']))" >> "$SUMMARY"
    done
    say "SWEEP: lr $LR collected. $SUMMARY now has $(( $(wc -l < "$SUMMARY") - 1 )) row(s)"
done

say "SWEEP DONE (or out of learning rates). Summary: $SUMMARY"
say "  THE LOSS IS NOT THE RESULT. Compare mean_dockq against 0.5783 and frac_acc against 0.7283."
say "  For significance use a PAIRED test on the 92 interfaces -- Wilcoxon signed-rank for"
say "  mean_dockq, where an interface whose selection did not move contributes an exact zero, so the"
say "  effective n is the number that moved. frac_acc is structurally underpowered: only 13 of 92"
say "  interfaces sit below acceptable with a rescuable oracle, so its whole range is +0.141."
