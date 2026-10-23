#!/bin/bash
# GRADIENT ALIGNMENT ACROSS TRAINING, to separate CANCELLATION from LOST SENSITIVITY.
#
# Runs --grad-probe once per checkpoint of a finished-or-stopped run and collects
#
#     ||sum_i g_i|| / sum_i ||g_i||        per pose, summed over ranks
#
# for both the PAE and the ranking term. Read against 1/sqrt(N), which the probe prints beside
# it: N independent directions already give ~1/sqrt(N), so a value near that is not evidence of
# anything and only a FALL across checkpoints is.
#
#   ratio falls while sum_i ||g_i|| holds up  -> CANCELLATION: the poses still push hard, in
#                                               directions that destroy each other.
#   sum_i ||g_i|| itself collapses            -> SENSITIVITY: the trainable parameters stopped
#                                               being able to move ipTM.
#
# WHY THIS EXISTS. zfahpb's parameter gradient fell 8x (0.0280 -> 0.0033) over six epochs while
# 40.1% of cross-seed pairs were still inverted, the mean per-pair gradient weight had RISEN
# 0.2977 -> 0.3465, and sum|c_i| had risen 8,354 -> 9,669. I called that cancellation; review
# pointed out it is equally consistent with lost sensitivity and that the two need different
# fixes. This measures which.
#
# EACH PROBE RESUMES A CHECKPOINT AND TAKES ONE STEP. --grad-probe exits after a single
# accumulation window and writes no head_*.pt, so these runs cannot be mistaken for trained arms
# and the source run can be resumed afterwards from its own resume_ep*.pt untouched. The probe
# needs the SAME architecture flags as the run it is reading, or the bundle will not match the
# parameter set.
#
# --epochs IS EP+1, NOT 1. A resume bundle carries the epoch count it finished at, and the
# trainer refuses a resume with nothing left to run: "the bundle has 6 epoch(s) done and
# --epochs is 1.0". --grad-probe exits after a single accumulation window regardless, so EP+1
# gives it exactly the one epoch it needs and never reaches a second.
#
# Usage:  ./shell_scripts/probe_alignment_trend.sh <rid> <ep> [<ep> ...]
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
RID="${1:-}"; shift || true
[ -n "$RID" ] || { echo "usage: $0 <rid> <ep> [<ep> ...]"; exit 1; }
[ "$#" -gt 0 ] || { echo "usage: $0 <rid> <ep> [<ep> ...]"; exit 1; }
OUT=training/runs/alignment_trend_$RID.log
: > "$OUT"
echo "=== alignment trend for $RID, epochs $* ===" | tee -a "$OUT"

for EP in "$@"; do
    CK="$(pwd)/training/runs/$RID/resume_ep${EP}.pt"
    if [ ! -f "$CK" ]; then
        echo "SKIP ep$EP: no $CK" | tee -a "$OUT"; continue
    fi
    BUSY=$($PY -I run_logging/cards_busy.py)
    if [ "$BUSY" != "0" ]; then
        echo "ABORT at ep$EP: $BUSY process(es) occupy cards 4,5,7 besides the holders" | tee -a "$OUT"
        exit 1
    fi
    echo "--- probing $RID ep$EP ---" | tee -a "$OUT"
    LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
      --cards 4,5,7 --note "ALIGNMENT PROBE of $RID at resume_ep${EP}.pt, NOT A TRAINING RUN. Reports ||sum_i g_i|| / sum_i ||g_i|| per pose, summed over ranks, for the PAE and ranking terms, to separate CANCELLATION (per-pose gradients keep magnitude but oppose each other) from LOST SENSITIVITY (the poses individually stop producing gradient). Read against the 1/sqrt(N) baseline the probe prints beside it; only a FALL across checkpoints is evidence. Architecture matches the source run exactly -- --pose-perblock --pose-perblock-copy, --train-only 'linear_no_bias_d,pose_perblock', 8 tensors / 40,960 parameters -- because a mismatched parameter set would not load the bundle. Exits after one accumulation window and writes no head_*.pt." \
      -- --pose-perblock --pose-perblock-copy \
      --resume "$CK" --resume-allow-drift \
      --resume-note "alignment probe, not a continuation: alpha_pae 1 (the source run used 0) because measuring both terms costs nothing and |g_rank| does not depend on it, and NO --l2-sp (the source used 2.5e-4) because the anchor is added to .grad after the backward and omitting it makes the reported gradient the PURE TASK gradient, which is the quantity the alignment ratio needs. One accumulation window, no checkpoint written." \
      --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
      --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
      --train-only 'linear_no_bias_d,pose_perblock' \
      --fold training/fold \
      --crop-margin 10 --crop-contact 5 \
      --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
      --manifest-selection training/overfit30/structures.csv \
      --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
      --grad-probe --epochs $((EP + 1)) --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
      --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
      --save-every 100000 \
      --pae-scored-pairs training/overfit30/pool.csv --pae-only \
      --shuffle structure --seeds 500000-500009 \
      --prefetch-units 2 \
      --val-loss-fold validation/fold \
      --val-loss-selection training/val10/structures.csv \
      --val-loss-pool training/val10/pool.csv \
      --val-loss-preflight training/val10/alignment_preflight.json \
      --val-loss-seeds 400000-400029 \
      --val-loss-prep validation/prep \
      --train-eval-pool training/val10/pool.csv \
      --train-eval-fold validation/fold \
      --train-eval-seed-base 400000 \
      --train-eval-budget 30 \
      --eval-cards 4,5,7 \
      --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
      --max-rss-gb 80 --preload-min-free-gb 260 \
      --require-replay training/ft3_pilot_replay.log --allow-code-drift >>"$OUT" 2>&1
    echo "  ep$EP launcher returned $?" | tee -a "$OUT"
done
echo "=== done; grep the per-probe run.log for 'ALIGNMENT' ===" | tee -a "$OUT"
