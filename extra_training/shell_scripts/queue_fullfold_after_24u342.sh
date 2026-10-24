#!/bin/bash
# QUEUED AFTER 24u342: THE SAME ARM ON THE FULL TRAINING FOLD, SCORED ON THE FULL VALIDATION SET.
#
# WHY. Every ranking arm so far has trained on overfit30 -- 30 structures, 100 units, 15 optimiser
# steps per epoch -- and been scored on val10's 18 interfaces. 24u342 has now shown that pool is
# exhausted: training loss flat at 0.0409-0.0416 from epoch 9 onward, selected DockQ oscillating
# in a 0.28-0.30 band around shipped 0.2909, frac locked at 11/18 for fifteen consecutive epochs.
# Six arms have ended at parity and EVERY ONE of those nulls is ambiguous in the same way, because
# "it just needed more data" has never been ruled out. This rules it out, or stops being able to.
#
# TWO HYPOTHESES, AND THEY PREDICT OPPOSITE THINGS:
#   too little data   -- the reordering learned from 30 structures does not generalise, so which
#                        interfaces it helps is near a coin flip (24u342 sits at 4 better / 4
#                        worse on the eight hard interfaces, stably, for a dozen epochs). More
#                        data should break that symmetry.
#   wrong objective   -- the coefficients sum to zero within each group by construction, pairs
#                        involving the selected pose are 2.29% of the loss, and nzqhdf drove this
#                        loss BELOW every other arm while producing the worst selection on record.
#                        More data then optimises the same wrong thing better and the symmetry
#                        holds.
#
# SCALE. 488 structures against overfit30's 30, so roughly 16x the units and 16x the optimiser
# steps PER EPOCH. --limit-units cannot be used to keep epochs small: it truncates the population
# permanently after the shuffle, which would just be a different fixed subset and would reproduce
# the problem being fixed. So epochs are genuinely full, and the EPOCH COUNT IS SCALED DOWN to
# match: 12 epochs here is on the order of 4x the total optimisation 24u342 received in 50, while
# still giving twelve held-out evaluations and clearing the ten-epoch minimum.
#
# VAL10 AND VALFULL ARE NOT COMPARABLE, AND THIS IS THE TRAP TO AVOID. val10 is 18 interfaces
# chosen as the hard tail; valfull is 92. Shipped scores 0.2909 / 0.5556 on val10 and something
# entirely different on valfull. NO NUMBER FROM ANY PREVIOUS ARM TRANSFERS. The reporting is
# already safe here by construction: render_findings.py and plot_three.py both derive the shipped
# baseline from the run's OWN eval shards via their stored_iptm column, so the reference line
# follows the pool automatically. The thing to police is prose, not code.
#
# EVERYTHING ELSE IS HELD FIXED so data is the only variable: cross-seed softplus at tau 0.0305,
# --alpha-pae 0, distance AND orientation at all four pairformer blocks, --train-only matching 12
# tensors / 44,032 parameters, --l2-sp-exclude 'frame_dir', lr 1e-2, warmup 20, --blocks-per-ckpt
# 0 (load-bearing: the orientation copies read pose_direction's cache, which the outer forward
# clears). Same --manifest-skip, since those ten PDBs are excluded for label reasons that do not
# change with the pool.
#
# LAMBDA IS RE-MEASURED EVEN THOUGH THE PARAMETER COUNT IS IDENTICAL. The sqrt(n_new/n_old) factor
# is exactly 1 here, so lambda moves only if |g_rank| differs on the new data distribution -- which
# is unknown and cheap to measure. If the probe comes back near 1.0x, lambda stays 2.5e-4 and the
# drift numbers remain directly comparable to 24u342's.
#
# NOTHING LAUNCHES IF: 24u342 exits non-zero, the cards are not free, the probe fails, or lambda
# does not parse. CANCEL: touch training/runs/CANCEL_FULLFOLD.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
PARENT=24u342
LOG=training/runs/fullfold_queue.log
TRAIN_ONLY='linear_no_bias_d,pose_perblock,frame_dir'
# 9dzj IS EXCLUDED VIA THE SELECTION FILE, NOT --manifest-skip, and the reason is the preflight. Those have no
# ranking labels at all. 9dzj has labels for seeds 500002-500009 on both its interfaces but NOT
# for 500000 and 500001, so --rank-loss's coverage guard refused the first probe (oimynw):
#   "4 group(s) have NO labels, e.g. [('9dzj', 500000, (0, 2)), ...]"
# which is the guard doing its job -- those poses would pay a forward and contribute no ranking
# gradient. Checked exhaustively before changing anything: across all 977 full_train interfaces,
# 9dzj is the ONLY structure with any missing (iface, seed) label, 4 groups in total. Excluding it
# costs 2 interfaces of 977. The alternative, narrowing --seeds to 500002-500009, would change the
# data for every structure to accommodate one, which is the worse trade.
#
# WHY THE SELECTION FILE AND NOT THE SKIP LIST. Adding 9dzj to --manifest-skip changed the manifest
# digest, and the alignment preflight records the digest of the population it was run over, so the
# trainer then refused: "the preflight was run over manifest 77fd7d86b4079a45, this run is
# 258f939fd53397bf". Re-running the real preflight costs 22.4 s/structure. Instead
# phases/phase_k_derive_preflight_subset.py derived a child marker, which REPRODUCED the parent
# digest over its 488 structures before emitting the child's 258f939fd53397bf over 487 -- so the
# inherited evidence is verified rather than assumed, and the marker records `derived_from` so
# anything reading it can see the check was inherited. --manifest-skip therefore stays at the
# original ten.
SKIP='9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3'
exec >> "$LOG" 2>&1
echo "=== full-fold arm armed $(date '+%Y-%m-%d %H:%M:%S %Z'), waiting on $PARENT ==="

while :; do
    [ -f training/runs/CANCEL_FULLFOLD ] && { echo "CANCELLED by training/runs/CANCEL_FULLFOLD"; exit 0; }
    if [ -f "training/runs/$PARENT/status.json" ]; then
        EC=$($PY -I -c "import json;print(json.load(open('training/runs/$PARENT/status.json')).get('exit_code','?'))" 2>/dev/null)
        echo "$(date '+%H:%M:%S') $PARENT ended, exit_code=$EC"
        [ "$EC" = "0" ] || { echo "NOT PROCEEDING: $PARENT exited $EC; the cards may be wanted to diagnose it."; exit 1; }
        break
    fi
    sleep 120
done

# The bracket stops this grep matching its own cmdline in ps; a SIGTERMed rank also keeps running
# for a few seconds after releasing its GPU, so both checks poll instead of refusing instantly.
for _ in $(seq 120); do
    LEFT=$(ps -eo pid,cmd | grep -cE "python3? -u .*[p]hase_k_train\.py|[p]hase_k_eval" || true)
    BUSY=$($PY -I run_logging/cards_busy.py)
    [ "$LEFT" = "0" ] && [ "$BUSY" = "0" ] && break
    sleep 1
done
[ "$LEFT" = "0" ] || { echo "NOT PROCEEDING: $LEFT training/eval process(es) still alive"; exit 1; }
[ "$BUSY" = "0" ] || { echo "NOT PROCEEDING: $BUSY process(es) occupy cards 4,5,7"; exit 1; }

COMMON=(
  --pose-perblock --pose-perblock-copy --frame-direction
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2
  --train-only "$TRAIN_ONLY"
  --fold training/fold
  --crop-margin 10 --crop-contact 5
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_no9dzj.json
  --manifest-selection training/full_train/structures_no9dzj.csv
  --manifest-skip "$SKIP"
  --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5
  --alpha-confidence 1.0 --clip-norm 25.0 --crop-size 3400
  --pae-scored-pairs training/full_train/pool.csv --pae-only
  --shuffle structure --seeds 500000-500009
  --prefetch-units 2
  --val-loss-fold validation/fold
  --val-loss-selection training/valfull/structures.csv
  --val-loss-pool training/valfull/pool.csv
  --val-loss-preflight training/valfull/alignment_preflight.json
  --val-loss-seeds 400000-400029
  --val-loss-prep validation/prep
  --train-eval-pool training/valfull/pool.csv
  --train-eval-fold validation/fold
  --train-eval-seed-base 400000
  --train-eval-budget 30
  --eval-cards 4,5,7
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 480
  --max-rss-gb 80 --preload-min-free-gb 260
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
)

echo "$(date '+%H:%M:%S') probing |g_rank| on the FULL training fold"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "GRADIENT PROBE on the FULL training fold for the full-fold arm, NOT A TRAINING RUN. Same trainable set as 24u342 -- 12 tensors / 44,032 parameters -- so the sqrt(n_new/n_old) factor in run_logging/l2sp_lambda.py is exactly 1 and lambda moves only if |g_rank| differs on this data distribution. Measures that. Exits after one accumulation window and writes no head_*.pt." \
  -- "${COMMON[@]}" --grad-probe --epochs 1 --save-every 100000
PRC=$?
[ "$PRC" = "0" ] || { echo "NOT LAUNCHING: the probe exited $PRC"; exit 1; }
# `ls -t training/runs | head -1` is WRONG: this script's own log is the newest entry, and `latest`
# is a six-character symlink. Match directories, six-char ids, and exclude `latest` by name.
PROBE=$(ls -td training/runs/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null \
        | grep -xE '[a-z0-9]{6}' | grep -vx latest | head -1)
[ -n "$PROBE" ] || { echo "NOT LAUNCHING: could not identify the probe run directory"; exit 1; }
echo "$(date '+%H:%M:%S') probe run $PROBE"

LAM=$($PY -I run_logging/l2sp_lambda.py --from-run zfahpb --from-probe ygpn6t \
        --to-probe "$PROBE" --new-params 40960 2>/dev/null \
      | grep -oE "nearest value worth using: [0-9.e-]+" | awk '{print $NF}')
case "$LAM" in
    ""|*[!0-9.e-]*) echo "NOT LAUNCHING: could not derive lambda from probe $PROBE (got '$LAM')"; exit 1 ;;
esac
echo "$(date '+%H:%M:%S') lambda $LAM from probe $PROBE -- launching 12 full-fold epochs"

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "THE ORIENTATION ARM ON THE FULL TRAINING FOLD, SCORED ON THE FULL VALIDATION SET. Identical to 24u342 in every setting except the data: cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0, distance AND orientation injected at all four pairformer blocks with untied per-block copies, --train-only 'linear_no_bias_d,pose_perblock,frame_dir' = 12 tensors / 44,032 parameters, --l2-sp-exclude 'frame_dir' leaving the 4 zero-init orientation tensors trainable but out of the anchor, lr 1e-2, warmup 20, --blocks-per-ckpt 0. TRAINING DATA: training/full_train (488 structures) instead of overfit30 (30), roughly 16x the units and 16x the optimiser steps per epoch, which is why this is 12 epochs rather than 50 -- on the order of 4x the total optimisation 24u342 received, with twelve held-out evaluations. HELD OUT: training/valfull (92 interfaces) instead of val10 (18). THESE TWO POOLS ARE NOT COMPARABLE and no number from any previous arm transfers to this one; shipped 0.2909 / 0.5556 are VAL10 figures. The reporting derives shipped from this run's own eval shards via their stored_iptm column, so the reference follows the pool automatically. WHY: six arms have ended at parity on val10 and every one of those nulls is ambiguous, because 30 structures was never enough to distinguish 'the reordering does not generalise' from 'the objective is wrong'. 24u342 made the case concrete -- training loss flat from epoch 9, frac pinned at 11/18 for fifteen epochs, and a stable 4-better/4-worse split across the eight hard interfaces that looks exactly like an un-generalised signal. If more data breaks that symmetry, the objective is fine and the pool was the limit; if the symmetry holds at 1,028 interfaces, the objective is the problem and no amount of data fixes it. Either answer is worth more than another overfit30 arm. LAMBDA was re-measured by a fresh --grad-probe on this data even though the parameter count is unchanged and the sqrt factor is exactly 1, because |g_rank| can differ on a new distribution. CRITERION: selected DockQ and fraction above 0.23 on the 92 valfull interfaces against shipped DERIVED FROM THIS RUN'S SHARDS. Spearman is reported but is NOT the criterion. CONTROL: 24u342, this arm on 30 structures scored on 18 interfaces." \
  -- "${COMMON[@]}" \
  --l2-sp "$LAM" --l2-sp-exclude 'frame_dir' \
  --epochs 12 --save-every 250 \
  --val-loss-every 1 --train-eval-every 1
rc=$?
echo "$(date '+%H:%M:%S') full-fold arm launcher returned $rc"
exit $rc
