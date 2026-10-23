#!/bin/bash
# FULL CONFIDENCE HEAD TRAINABLE, ANCHORED, under the ranking loss.
#
# THE ASK. 24u34d plateaued at a held-out ranking loss of ~0.040 with its TRAINING loss flat at
# the same value (0.04191 -> 0.04189 over epochs 6-7), which is the signature of UNDERFITTING,
# not overfitting: 44,032 parameters are not enough to express the correction. So drop
# --train-only entirely and let all 248 confidence-head tensors train, held near the shipped
# weights by L2-SP instead of by freezing.
#
# WHAT IS NEW HERE, AND WHAT IS NOT. Every previous full-head arm -- and there are ~44 of them --
# used the PAE loss; rank_form is None in all of them. NO FULL-HEAD ARM HAS EVER SEEN THE RANKING
# LOSS. That part is genuinely untried.
#
# But the full head HAS been run with an L2-SP anchor, twice, and it is a warning:
#   wpg274  full head, l2_sp 1e-5, 50 epochs, PAE loss
#           -> epoch 50 DockQ 0.1616, frac 0.2778, against shipped 0.2909 / 0.5556
#           -> drift reached 3,801 and the penalty was 0.0380 against a PAE loss near 2.0
#   jjvdhk  full head, l2_sp 1e-5, 100 epochs, PAE loss -> drift 279 by the epoch logged
# At 1e-5 the anchor was not binding on anything: 1.9% of the loss. The head moved freely and the
# metrics collapsed. So LAMBDA IS THE WHOLE EXPERIMENT, and it cannot be guessed.
#
# WHY THE sqrt RULE IS NOT ENOUGH BY ITSELF. l2sp_lambda.py transfers lambda as
#     lambda_new = lambda_old * (|g_new|/|g_old|) / sqrt(n_new/n_old)
# The anchored set goes 40,960 -> ~19.84M parameters, so the sqrt factor alone is 22.0 and would
# put lambda at 2.5e-4/22.0 = 1.1e-5 -- which is, to one digit, the value that already failed on
# this exact parameter set. That coincidence needs saying out loud rather than being walked into.
#
# It is NOT necessarily the same setting, for one measured reason: the ranking gradient is far
# smaller than the PAE gradient. The alignment probes put ||sum_i g_i|| at 1.4e-2 for the ranking
# term and 3.6e-1 for PAE at the same weights -- a factor of ~25. The same lambda is therefore
# ~25x more binding against this arm's objective than it was against wpg274's. Which is exactly
# why the |g_new|/|g_old| factor has to be MEASURED at this trainable set rather than assumed, and
# why this script probes before it launches.
#
# THREE STEPS, AND IT STOPS RATHER THAN GUESSING:
#   1. refuse unless the cards are free
#   2. --grad-probe with the full head trainable, to measure |g_rank| there
#   3. derive lambda from that probe, then launch 50 epochs. If the probe fails or lambda does not
#      parse, NOTHING LAUNCHES -- a wrong lambda would look like a clean experiment and silently
#      price weight motion wrong, and on this parameter set that is the difference between the
#      arm and wpg274.
#
# EVERYTHING ELSE MATCHES 24u34d, so 24u34d is the control and capacity is the only variable:
# cross-seed softplus ranking loss at tau 0.0305 with --alpha-pae 0, distance AND orientation at
# all four pairformer blocks, overfit30 train, val10 held out every epoch at budget 30, 100
# poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0 (load-bearing: the
# orientation copies read pose_direction's direction cache, which the outer forward clears).
#
# THE ORIENTATION TENSORS STAY OUT OF THE ANCHOR (--l2-sp-exclude 'frame_dir'). They are
# zero-init, so anchoring them to zero is plain weight decay on the term being added. Everything
# else in the head is PRETRAINED, so distance from the shipped weights is a prior worth holding.
#
# CANCEL: touch training/runs/CANCEL_FULLHEAD, or kill this script BY PID read off ps.
#
# --probe-only: run steps 1-3 up to and including the lambda derivation, PRINT IT, and exit 0
# WITHOUT launching. wpg274 is the reason this option exists: lambda is the whole experiment on
# this parameter set, and 1.1e-5 -- what the sqrt factor alone predicts -- is to one digit the
# value that already failed there. The number is worth reading before 50 epochs are spent on it.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
PROBE_ONLY=0
GIVEN_LAM=""
GIVEN_PROBE=""
case "${1:-}" in
    --probe-only)  PROBE_ONLY=1 ;;
    # --launch-with <lambda> <probe-rid>: skip step 2 because that probe ALREADY PASSED its gate.
    # Re-probing would cost 3 minutes of cards and tell us the same number; the rid is recorded so
    # the note still says which measurement set lambda.
    --launch-with) GIVEN_LAM="${2:?--launch-with needs a lambda}"
                   GIVEN_PROBE="${3:?--launch-with needs the probe run id}" ;;
esac
LOG=training/runs/fullhead_queue.log
# THE REACHABLE SET, DEFINED BY THE FIRST PROBE'S REFUSAL (run upnohf). Letting all 248 head
# tensors train was the original plan; --grad-probe refused it because 158 of them got NO non-zero
# ranking gradient -- 14,438,734 of 19,752,270 parameters, 73.1%. That is structural, confirmed in
# protenix/model/modules/confidence.py:
#     pae_pred = linear_no_bias_pae(pae_ln(z_pair))
#     plddt_pred, resolved_pred <- a <- s_single
# ipTM is computed from pae_pred alone, so it depends on z and never on s. Inside each pairformer
# block the triangle ops update z from z, while attention_pair_bias and single_transition update s
# using z as input -- z never depends on s. So the whole single-representation path, plus the
# plDDT/PDE/resolved output heads and their bin edges, CANNOT change ipTM no matter how long they
# train. Keeping them trainable would put 14.4M dead parameters in the optimiser and, worse, in the
# anchor's ||W - W0||^2, which would mis-price lambda against drift that cannot matter.
#
# What is left is 90 tensors / 5,313,536 parameters, still 120x what 24u34d had:
#     2,360,832  pair_transition          (blocks 0, 2, 3 at 262,144 each -- the largest live ones)
#     1,182,720  tri_mul_out
#     1,182,720  tri_mul_in
#       330,240  tri_att_start
#       114,944  linear_no_bias_s1        (s -> z, so these DO reach ipTM)
#       114,944  linear_no_bias_s2
#        16,384  linear_no_bias_pae       (the readout h7msge and zfahpb both left frozen)
#        10,240  linear_no_bias_d(+_wo_onehot)
#           512  pae_ln
#
# tri_att_end came back dead in all four blocks even though it updates z, which I cannot explain
# structurally; it is left OUT rather than guessed at, because the gate would refuse the run again
# if it is genuinely unreachable. One accumulation window cannot distinguish "structurally
# disconnected" from "numerically zero here", and the two families above are the ones where the
# source settles it.
# THE PATTERNS ARE PER-BLOCK AND IRREGULAR, AND THAT IS NOT A TYPO. A second probe (oqkgp9) with
# family-level patterns -- pair_transition, tri_mul_out, tri_mul_in, tri_att_start -- was REFUSED
# with 49 dead tensors, every one of them also dead in the first probe. The live/dead split is not
# per family, it is per BLOCK within a family:
#     block 0   pair_transition only                            (tri_mul_*, tri_att_start dead)
#     block 1   tri_mul_in, tri_mul_out only                     (pair_transition dead)
#     block 2   pair_transition, tri_mul_in, tri_mul_out
#     block 3   pair_transition, tri_att_start, tri_mul_in, tri_mul_out
# plus linear_no_bias_s1/s2 (s -> z), pae_ln, linear_no_bias_pae, and the pose pathway.
# These paths are DERIVED from upnohf's refusal rather than guessed, by taking every module whose
# tensors were ALL live, which covers 5,313,536 parameters with zero leftovers. The irregularity
# has no structural story I can give -- the s/z argument explains attention_pair_bias,
# single_transition and the plDDT/PDE/resolved heads, but not why block 0's tri_mul is dead while
# block 2's is live -- so the set is being handed back to the gate for independent confirmation
# instead of being trusted on my reading. If this probe passes, the split is reproduced twice.
TRAIN_ONLY='linear_no_bias_d,linear_no_bias_d_wo_onehot,linear_no_bias_pae,linear_no_bias_s1,linear_no_bias_s2,pae_ln,pairformer_stack\.blocks\.0\.pair_transition\.layernorm1,pairformer_stack\.blocks\.0\.pair_transition\.linear_no_bias,pairformer_stack\.blocks\.0\.pair_transition\.linear_no_bias_a,pairformer_stack\.blocks\.0\.pair_transition\.linear_no_bias_b,pairformer_stack\.blocks\.1\.tri_mul_in\.layer_norm_in,pairformer_stack\.blocks\.1\.tri_mul_in\.layer_norm_out,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_a_g,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_a_p,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_b_g,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_b_p,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_g,pairformer_stack\.blocks\.1\.tri_mul_in\.linear_z,pairformer_stack\.blocks\.1\.tri_mul_out\.layer_norm_in,pairformer_stack\.blocks\.1\.tri_mul_out\.layer_norm_out,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_a_g,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_a_p,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_b_g,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_b_p,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_g,pairformer_stack\.blocks\.1\.tri_mul_out\.linear_z,pairformer_stack\.blocks\.2\.pair_transition\.layernorm1,pairformer_stack\.blocks\.2\.pair_transition\.linear_no_bias,pairformer_stack\.blocks\.2\.pair_transition\.linear_no_bias_a,pairformer_stack\.blocks\.2\.pair_transition\.linear_no_bias_b,pairformer_stack\.blocks\.2\.tri_mul_in\.layer_norm_in,pairformer_stack\.blocks\.2\.tri_mul_in\.layer_norm_out,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_a_g,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_a_p,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_b_g,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_b_p,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_g,pairformer_stack\.blocks\.2\.tri_mul_in\.linear_z,pairformer_stack\.blocks\.2\.tri_mul_out\.layer_norm_in,pairformer_stack\.blocks\.2\.tri_mul_out\.layer_norm_out,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_a_g,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_a_p,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_b_g,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_b_p,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_g,pairformer_stack\.blocks\.2\.tri_mul_out\.linear_z,pairformer_stack\.blocks\.3\.pair_transition\.layernorm1,pairformer_stack\.blocks\.3\.pair_transition\.linear_no_bias,pairformer_stack\.blocks\.3\.pair_transition\.linear_no_bias_a,pairformer_stack\.blocks\.3\.pair_transition\.linear_no_bias_b,pairformer_stack\.blocks\.3\.tri_att_start\.layer_norm,pairformer_stack\.blocks\.3\.tri_att_start\.linear,pairformer_stack\.blocks\.3\.tri_att_start\.mha\.linear_g,pairformer_stack\.blocks\.3\.tri_att_start\.mha\.linear_k,pairformer_stack\.blocks\.3\.tri_att_start\.mha\.linear_o,pairformer_stack\.blocks\.3\.tri_att_start\.mha\.linear_q,pairformer_stack\.blocks\.3\.tri_att_start\.mha\.linear_v,pairformer_stack\.blocks\.3\.tri_mul_in\.layer_norm_in,pairformer_stack\.blocks\.3\.tri_mul_in\.layer_norm_out,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_a_g,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_a_p,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_b_g,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_b_p,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_g,pairformer_stack\.blocks\.3\.tri_mul_in\.linear_z,pairformer_stack\.blocks\.3\.tri_mul_out\.layer_norm_in,pairformer_stack\.blocks\.3\.tri_mul_out\.layer_norm_out,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_a_g,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_a_p,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_b_g,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_b_p,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_g,pairformer_stack\.blocks\.3\.tri_mul_out\.linear_z,pose_perblock,frame_dir'
exec >> "$LOG" 2>&1
echo "=== full-head arm armed $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

[ -f training/runs/CANCEL_FULLHEAD ] && { echo "CANCELLED by training/runs/CANCEL_FULLHEAD"; exit 0; }

# NOTHING MUST STILL BE READING THE CARDS.
#
# THE BRACKET IS LOAD-BEARING. `ps -eo pid,cmd` lists the grep that is reading it, and that grep's
# own cmdline CONTAINS the pattern, so a plain grep matches itself and the guard can never reach 0.
# `[p]hase_k_...` cannot match the literal text `[p]hase_k_...` in the grep's cmdline while still
# matching the real processes. This is the same class as `grep -v " grep "` not excluding ugrep:
# do not go back to filtering the matcher out by name, make it unmatchable.
#
# AND IT RETRIES RATHER THAN REFUSING ON A TRANSIENT. A SIGTERMed rank takes a few seconds to
# unwind its traceback and exit; the first attempt at this arm refused on exactly that, one second
# after stop_run.sh reported the cards free. 60 s of polling costs nothing against a 50-epoch run.
for _ in $(seq 60); do
    LEFT=$(ps -eo pid,cmd | grep -cE "python3? -u .*[p]hase_k_train\.py|[p]hase_k_eval" || true)
    BUSY=$($PY -I run_logging/cards_busy.py)
    [ "$LEFT" = "0" ] && [ "$BUSY" = "0" ] && break
    sleep 1
done
[ "$LEFT" = "0" ] || { echo "NOT PROCEEDING: $LEFT training/eval process(es) still alive after 60 s"; exit 1; }
[ "$BUSY" = "0" ] || { echo "NOT PROCEEDING: $BUSY process(es) occupy cards 4,5,7 after 60 s"; exit 1; }

# ---- step 2: probe |g_rank| with the full head trainable ---------------------------------------
if [ -n "$GIVEN_LAM" ]; then
    echo "$(date '+%H:%M:%S') skipping the probe: lambda $GIVEN_LAM already measured at probe $GIVEN_PROBE"
else
echo "$(date '+%H:%M:%S') probing |g_rank| at the FULL HEAD trainable set"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "GRADIENT PROBE for the full-head arm, NOT A TRAINING RUN. Measures |g_rank| with ALL 248 confidence-head tensors trainable (19.8M parameters) alongside the per-block distance and orientation pathways, so run_logging/l2sp_lambda.py can transfer lambda honestly. The anchored set goes 40,960 -> ~19.84M, a sqrt factor of 22.0 on its own, which would land lambda at 1.1e-5 -- to one digit the value that already failed on this exact parameter set in wpg274 (full head, l2_sp 1e-5, 50 epochs, PAE loss, ending at DockQ 0.1616 / frac 0.2778 with drift 3,801 and the penalty at 1.9% of the loss). It is not necessarily the same setting, because the ranking gradient is ~25x smaller than the PAE gradient that lambda was failing against there, so the same lambda is ~25x more binding here. That is precisely why |g_new| has to be measured rather than assumed. Exits after one accumulation window and writes no head_*.pt." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only "$TRAIN_ONLY" \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --grad-probe --epochs 1 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
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
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
PRC=$?
[ "$PRC" = "0" ] || { echo "NOT LAUNCHING: the probe exited $PRC"; exit 1; }
fi
# `ls -t training/runs | head -1` IS WRONG and cost a probe: this script's own log,
# training/runs/fullhead_queue.log, is the newest entry because the script is writing to it, so
# PROBE came back as "fullhead_queue.log" and lambda could not be derived from it. Match only
# DIRECTORIES, and only 6-character run ids, so no stray file or helper dir can win the race.
# AND `latest` IS ITSELF SIX LOWERCASE CHARACTERS. launch.py maintains training/runs/latest as a
# symlink to the newest run, so a bare [a-z0-9]{6} filter matches it and resolves to the wrong
# name. Exclude it by name; -xE already anchors the rest.
PROBE=$(ls -td training/runs/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null \
        | grep -xE '[a-z0-9]{6}' | grep -vx latest | head -1)
[ -n "$PROBE" ] || { echo "NOT LAUNCHING: could not identify the probe run directory"; exit 1; }
echo "$(date '+%H:%M:%S') probe run $PROBE"

# ---- step 3: lambda from the probe, then launch ------------------------------------------------
# --new-params is the ANCHORED count: the full head minus the 4 excluded zero-init orientation
# tensors (768 + 3*768 = 3,072 parameters).
# 5,313,536 live head + 33,024 per-block distance copies + 3,072 orientation, minus the 3,072
# orientation parameters that --l2-sp-exclude keeps OUT of the anchor.
ANCH=$($PY -I -c "print(5_313_536 + 33_024 + 3_072 - 3_072)")
if [ -n "$GIVEN_LAM" ]; then
    LAM="$GIVEN_LAM"; PROBE="$GIVEN_PROBE"
else
LAM=$($PY -I run_logging/l2sp_lambda.py --from-run zfahpb --from-probe ygpn6t \
        --to-probe "$PROBE" --new-params "$ANCH" 2>/dev/null \
      | grep -oE "nearest value worth using: [0-9.e-]+" | awk '{print $NF}')
fi
case "$LAM" in
    ""|*[!0-9.e-]*) echo "NOT LAUNCHING: could not derive lambda from probe $PROBE (got '$LAM')"; exit 1 ;;
esac
if [ "$PROBE_ONLY" = "1" ]; then
    echo "$(date '+%H:%M:%S') lambda from probe $PROBE: $LAM  -- PROBE ONLY, not launching"
    echo "LAMBDA=$LAM"
    echo "PROBE=$PROBE"
    exit 0
fi
echo "$(date '+%H:%M:%S') lambda from probe $PROBE: $LAM  -- launching 50 epochs"

LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py \
  --cards 4,5,7 --note "THE REACHABLE CONFIDENCE HEAD, TRAINABLE AND ANCHORED, under the cross-seed softplus ranking loss. 100 tensors / 5,349,632 parameters train -- every one verified by --grad-probe to actually receive ranking gradient -- held near the shipped weights by L2-SP instead of frozen. NOT the whole head: letting all 248 head tensors train was the original plan and --grad-probe REFUSED it, because 158 of them (14,438,734 parameters, 73.1%) get no ranking gradient at all. ipTM is computed from pae_pred = linear_no_bias_pae(pae_ln(z_pair)), so it depends on z and never on s; the triangle ops update z from z while attention_pair_bias and single_transition update s using z, so z never depends on s, and the entire single-representation path plus the plDDT/PDE/resolved heads cannot move ipTM however long they train. Training them would also put 14.4M dead parameters into the anchor's ||W-W0||^2 and mis-price lambda. See docs/confidence_head_reachability.md. WHY: 24u34d plateaued with its held-out ranking loss at ~0.040 AND its training loss flat at the same value (0.04191 -> 0.04189 over epochs 6-7), both curves flattening together -- the signature of UNDERFITTING, so 44,032 parameters cannot express the correction the loss asks for. This is also the first full-head arm ever run under the ranking loss: all ~44 earlier ones used PAE (rank_form None). LAMBDA WAS MEASURED, NOT GUESSED, and that matters more here than in any previous arm: wpg274 ran this exact parameter set at l2_sp 1e-5 for 50 epochs under the PAE loss and ended at DockQ 0.1616 / frac 0.2778 against shipped 0.2909 / 0.5556, with drift at 3,801 and the penalty at 1.9% of the loss -- the anchor was not binding and the head moved freely to a much worse place. The sqrt factor alone (40,960 -> 19.84M anchored, 22.0x) would put lambda at 1.1e-5, to one digit that same failed value; it is not necessarily the same setting, because the ranking gradient is ~25x smaller than the PAE gradient (alignment probes: ||sum_i g_i|| 1.4e-2 rank vs 3.6e-1 PAE at the same weights), so the same lambda is ~25x more binding against this objective -- but that factor was measured by a fresh --grad-probe at this trainable set rather than assumed. The 4 zero-init orientation tensors stay OUT of the anchor via --l2-sp-exclude 'frame_dir', since anchoring a zero-init tensor to zero is plain decay on the new term. EVERYTHING ELSE MATCHES 24u34d so capacity is the only variable: tau 0.0305, --alpha-pae 0, distance AND orientation at all four pairformer blocks, overfit30 train, val10 held out every epoch at budget 30, 50 epochs, 100 poses/step, 15 steps/epoch, lr 1e-2, warmup 20, --blocks-per-ckpt 0. CRITERION: selected DockQ and fraction above 0.23 on the 18 val10 interfaces against shipped (DockQ 0.2909, frac 0.5556, oracle 0.5968/1.0000). Spearman is reported but is NOT the criterion. THE RISK, STATED: 19.8M parameters against 30 training structures, and wpg274 is what this looks like when the anchor is too loose -- watch drift and the metrics together from epoch 1, not just the loss. CONTROLS: 24u34d (same arm, 44,032 parameters), wpg274 (same parameter set, loose anchor, PAE loss)." \
  -- --pose-perblock --pose-perblock-copy --frame-direction \
  --l2-sp "$LAM" --l2-sp-exclude 'frame_dir' \
  --rank-loss --alpha-rank 1.0 --rank-labels training/fold/results.csv \
  --rank-form softplus --rank-cross-seed --rank-structs-per-step 2 \
  --train-only "$TRAIN_ONLY" \
  --fold training/fold \
  --crop-margin 10 --crop-contact 5 \
  --minimal-items --alignment-preflight training/overfit30/alignment_preflight.json \
  --manifest-selection training/overfit30/structures.csv \
  --manifest-skip '9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3' \
  --epochs 50 --lr 0.01 --warmup-steps 20 --seed 0 --accum 85 --nsample 5 \
  --alpha-confidence 1.0 --alpha-pae 0.0 --clip-norm 25.0 --crop-size 3400 \
  --save-every 15 \
  --pae-scored-pairs training/overfit30/pool.csv --pae-only \
  --shuffle structure --seeds 500000-500009 \
  --prefetch-units 2 \
  --val-loss-every 1 \
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
  --train-eval-every 1 \
  --eval-cards 4,5,7 \
  --blocks-per-ckpt 0 --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 260 \
  --require-replay training/ft3_pilot_replay.log --allow-code-drift
rc=$?
echo "$(date '+%H:%M:%S') full-head arm launcher returned $rc"
exit $rc
