#!/bin/bash
# RESUME of the relative-iTM arm at epoch 32 of 60, after the matched-crop eval pause.
# Every setting is byte-identical to launch_rel_itm_beta75.sh -- this file is that script plus
# --resume -- so the guard accepts the bundle with NO --resume-override, which is the check
# that nothing drifted across the pause. The original header follows.
#
# RELATIVE (MEAN-CENTRED) iTM REGRESSION + PAE, 60 epochs, beta 75, cards 4,5,7.
#
# THE OBJECTIVE, per scored chain pair over all 50 poses of a structure:
#   L = alpha_pae * PAE + 75 * (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2
# s_i is the MEAN over anchor rows of the predicted ipTM; g_i the same from the TRUE error bins.
#
# WHY. Measured on d32kqp epoch 66 over the 738 winnable interfaces:
#   corr(interface MEAN predicted ipTM, interface MEAN DockQ) = +0.866   <- the head HAS this
#   median WITHIN-interface Spearman vs DockQ                 = +0.317   <- the head LACKS this
#   the same quantity for the TRUE iTM                        = +0.690   <- what is available
# The head has learned "is this complex easy" almost perfectly and barely learned "which pose of
# this complex is best", which is the only thing selection uses. Subtracting each interface's own
# mean from BOTH sides removes the across-complex signal from the objective entirely, so the only
# thing left to earn is the within-complex ordering.
#
# BETA 75 IS ~0.3x THE MEDIAN PARITY, at the user's direction, and the median is doing a lot of
# work: over seven accumulation windows at identical weights the parity alpha ranged 30 to 1316 --
# a 44x spread, median 247 -- because rel's coefficients sum to EXACTLY zero per interface, so its
# accumulated gradient is a residual after near-complete cancellation and both its magnitude and
# its sign are dominated by which structures land in the window. cos(PAE,rel) over the same seven
# ranged -0.505 to +0.702 and was NEGATIVE on three. So 75 is a starting ratio, not a derived
# value, and the per-epoch ratio is the thing to watch.
#
# MEAN POOLING, and it is not optional here. The ceiling from the per-pose-OST labels puts the
# true-iTM selector at 58.6% of oracle headroom under MAX pooling and 67.7% under MEAN, mean minus
# max +0.0140 +- 0.0034 paired (t=4.06). Training the mean while selecting by the max would train
# a quantity nobody reads, so --rank-pool mean goes in together with --select-by mean in the
# evaluation, which eval_checkpoint propagates automatically.
#
# WHAT IS MEASURED EVERY 5 EPOCHS: the HELD-OUT valfull pool -- 92 interfaces, 53 structures --
# at budget 30, scored under BOTH poolings from one pass because --rank-pool mean makes
# eval_checkpoint propagate --pool-mean --select-by mean.
#
# HELD OUT AND NOT IN-SAMPLE, AND THAT IS A REAL LIMITATION. --train-eval-* takes ONE pool and one
# fold, and the only other validation path, --eval-every-epoch, is the 100-seed pass at ~73 min --
# unusable every epoch over 60 epochs. So the in-sample number cannot be produced from inside this
# run and has to come from the checkpoints afterwards, which --save-every 250 and the epoch
# bundles make cheap: run_logging/compare_rowweight_insample.py scores any bundle on the 738
# winnable interfaces in ~35 min. Said plainly rather than left as an assumption, because the
# question "is it learning at all" is an in-sample question and this run cannot answer it live.
#
# BUDGET 30, not 100, at the user's direction. It is also the natural ceiling for anything that
# needs the valfull LABEL cache, which holds seeds 400000-400029 only -- the trunks go to 100 but
# the labels do not. Selection needs trunks and results.csv, both of which cover 100, so 30 is a
# cost choice here and not a coverage limit: ~25 min a pass against ~73.
#
# VAL10 AND VALFULL ARE NOT COMPARABLE, and neither are budget-30 and budget-100 valfull numbers:
# a larger candidate set lifts the oracle and the baseline together. The reference figures for
# THIS configuration are valfull at budget 30, not the 0.5778 / 0.7283 measured at budget 100.
#
# WHAT IS HELD from the row-weight arm so the loss is the only change: 368 winnable structures,
# crop 3400, margin 10, contact 5, accum 85, nsample 5, seeds 500000-500009, cosine lr 1e-2 flat
# to epoch 20 then down to 1e-4 by 50 and flat to 60, --pae-only, --blocks-per-ckpt 0, no L2,
# pose-copy injection, --frame-direction (258 marked trainable; the EFFECTIVE set is 100 -- the
# other 158 are pLDDT/PDE/resolved parameters that --pae-only never reaches, true of every arm in
# this project and now printed by --grad-probe).
#
# THE LAUNCH LOG MUST PRINT, or this is not the arm it claims to be:
#   - "RELATIVE iTM REGRESSION on chain_pair ipTM pooled by MEAN ... alpha 75"
#   - the label line: 368 structure(s), 36900 row(s), column g_mean, and a digest
#   - "PAE MASK OVERRIDE (--rank-form rel): applied on N pose(s), DIFFERED ... on M"
#   - 368 structure(s) / 738 interface(s) on the PAE mask line
#   - steps/epoch 184, and NO "anchor SNAPSHOT" line
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY_BIN=/home/ebrahim/Protenix/.venv/bin/python
BUSY=$($PY_BIN -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$BUSY" = "0" ] || { echo "REFUSING: cards 4,5,7 are occupied ($BUSY)"; exit 1; }
for g in test_rel_itm_loss test_pool_mean test_eval_pool_paths test_lr_schedule test_guard_fields; do
  $PY_BIN -I run_logging/$g.py >/dev/null 2>&1 || { echo "REFUSING: $g failed"; exit 1; }
done
echo "all five gates pass; RESUMING rel-iTM arm from t4jtjr resume_ep31.pt (epoch 32 of 60)"
LAYERNORM_TYPE=openfold $PY_BIN -I run_logging/launch.py \
  --cards 4,5,7 --numa-map 0:0,1:0,2:1 --note "RESUMED FROM t4jtjr resume_ep31.pt, picking up at epoch 32 of 60. SECOND deliberate pause, ~50 min, for the MATCHED-CROP evaluation: phase_k_eval gained a --crop-margin so this head and the shipped head could both be scored on the cropped inputs this arm trains on, in-sample and on valfull, four passes. The uncropped reading had the arm at -0.0087 in-sample while its own cropped training diagnostic read +0.5392 against +0.2819 uncropped for the same checkpoint, so the arm was being read off its training distribution and the shipped head on its own. The pause was deliberate and ~40 min long: --train-eval-* takes ONE pool, so this arm cannot measure in-sample selection while it runs, and the question 'is it learning at all' is an in-sample question. The cards were spent on compare_rowweight_insample.py scoring head_ep27 on the 738 winnable interfaces at budget 10 (= all 50 training poses per interface) before being handed back. Epoch 28 is re-run from the epoch-27 bundle, which restores AdamW's moments and the step counter, so the only cost is the partial epoch 28 that was in flight. RELATIVE (MEAN-CENTRED) iTM REGRESSION + PAE, alpha_rank 75, mean pooling, 60 epochs on the 368 winnable structures. L = PAE + 75 * (1/P) sum_i [(s_i - mean_s) - (g_i - mean_g)]^2 per scored chain pair over all 50 poses of a structure, with s the MEAN over anchor rows of predicted ipTM and g the same from the TRUE error bins (cached by run_logging/build_itm_labels.py from the per-pose-OST ceiling). WHY: on d32kqp epoch 66 the head scores corr(interface MEAN ipTM, interface MEAN DockQ) = +0.866 but median WITHIN-interface Spearman only +0.317, against +0.690 for the true iTM -- it has learned which complexes are easy and not which pose is best, and only the latter is used in selection. Centring both sides on the interface mean removes the across-complex signal from the objective so the within-complex ordering is the only thing left to earn. BETA 75 IS ~0.3x THE MEDIAN PARITY AND IS A STARTING RATIO, NOT A DERIVED VALUE: over seven accumulation windows at identical weights the parity alpha ranged 30 to 1316 (median 247, a 44x spread) and cos(PAE,rel) ranged -0.505 to +0.702, negative on three of seven, because rel's coefficients sum to EXACTLY zero per interface so its accumulated gradient is a residual after near-complete cancellation whose magnitude AND sign are set by batch composition. The per-epoch ratio is the thing to watch, not this number. MEAN POOLING IS LOAD-BEARING: the per-pose-OST ceiling puts the true-iTM selector at 58.6% of oracle headroom under max and 67.7% under mean (+0.0140 +- 0.0034, t=4.06), and training the mean while selecting the max would train a quantity nobody reads -- eval_checkpoint propagates --pool-mean --select-by mean automatically and reports BOTH poolings. PAE IS MASKED TO THE SAME PER-POSE INTERFACE rel RANKS: _scored_asym is static per structure while rel resolves the pair per pose through OST, and on 52 of 368 structures they disagree, which would have had PAE anchoring one interface while rel ranked another. THE PREDICTION TO CHECK, which is more sensitive than selected DockQ on 92 interfaces (t=7.6 against t=2.7): within-interface Spearman should move off 0.317 toward 0.690. FALSIFIER: the loss falls while within-interface Spearman does not, which is the margin-collapse shape every ranking arm here showed -- though this loss is a regression onto an absolute target and a collapsed prediction costs MORE than an honest one, so collapse is not an available escape. Everything else is held from the row-weight arm so the loss is the only change." \
  -- --resume training/runs/t4jtjr/resume_ep31.pt \
     --rank-loss --rank-form rel --rank-cross-seed --alpha-rank 75 \
  --rank-pool mean --rel-labels training/fold/itm_labels \
  --rank-structs-per-step 3 \
  --pose-perblock --pose-perblock-copy --frame-direction \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/full_train/alignment_preflight_winnable.json \
  --manifest-selection training/full_train/structures_winnable.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 60 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --lr-final 1e-4 --lr-decay-shape cosine \
  --lr-decay-start-epoch 20 --lr-decay-end-epoch 50 \
  --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 250 \
  --pae-scored-pairs training/full_train/pool_winnable.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --train-eval-pool training/valfull/pool.csv \
  --train-eval-fold validation/fold --train-eval-seed-base 400000 \
  --train-eval-budget 30 --train-eval-every 5 \
  --prefetch-units 2 --prefetch-workers 4 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "resume_rel_itm_ep32 returned $rc"
exit $rc
