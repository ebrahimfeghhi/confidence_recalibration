#!/bin/bash
# FULL CONFIDENCE HEAD (minus certified dead ops) + POSE AT FIVE SITES. Owner-authorised
# 2026-11-09, after the memory probe.
#
# THE TOKEN CAP IS MEASURED, NOT GUESSED. The probe ran 7m7i (2,004 tokens) through this exact
# config on one card: pass A peaked at 50.32 GB and pass B OOMed, needing 7.66 GiB with 2.75 GiB
# free of 94.97 -- an implied peak near 99 GB. Memory goes as tokens^2, so a 1,850 cap predicts
# ~84 GB. That costs THREE structures beyond the five already skipped (7m7i 2004, 9b9j 1949,
# 9e7x 1879), leaving 845 of 853.
#
# SITE 5 COPIES THE PRETRAINED DISTANCE WEIGHTS, and the number says why it is safe. Measured
# on the shipped head: ||pae_ln.weight|| is 4.4236, and linear_no_bias_d's columns -- a one-hot
# selects exactly one -- have mean norm 0.0881 and max 0.3136. So the copy perturbs step 0 by
# 2.0% of the signal on average, 7.1% at the worst bin. An earlier estimate of "8.21" came from a
# TOY test head with random weights at c_z 16 and was not representative.
# The orientation arm stays ZERO at every site: no pretrained orientation matrix exists to copy.
#
# SITE 5 (the readout injection) IS REMOVED, at the owner's direction, after 73vv52.
# WHAT HAPPENED, recorded so it is not repeated: --pose-readout-copy-distance copies TWO arms, and
# only one of them is bounded. The one-hot arm's output is a single column of a 256x39 matrix,
# mean norm 0.0881 against ||pae_ln.weight|| 4.4236 -- 2.0% of the signal, which is what was
# measured and reported. The RAW-DISTANCE arm was not measured: its output is ||w|| * d with
# ||w|| = 4.1924 and d a distance in ANGSTROMS, so it is 474% of the signal at 5 A and 1,895% at
# 20 A. Those weights are fine inside the stack, where every later block re-normalises; at the
# readout nothing rescales them. 73vv52 epoch 1 opened at PAE 46.27 over its first 23 steps
# (first 50 poses: 61.67) against a normal 2.95, and spent the epoch recovering to 2.79.
# If this site is ever reinstated: copy the one-hot arm ONLY and leave the raw-distance arm zero.
# NO --prefix-cache-gb. It caches the FROZEN prefix, and with the full head trainable there is no
# frozen prefix -- it would either cache nothing or hold memory for no benefit.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
SD=/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# THE LIVE TRAINABLE SET: everything outside the pair stack, plus the pair modules the dead-op
# measurement did NOT certify dead. --skip-dead-ops refuses to stub a module left trainable, so
# the caller must drop them here rather than have the flag quietly leave them running.
TRAIN_ONLY="$(/home/ebrahim/Protenix/.venv/bin/python -I - <<'PYX'
DEAD = {(0,"tri_mul_out"),(0,"tri_mul_in"),(0,"tri_att_start"),(0,"tri_att_end"),
        (1,"tri_att_start"),(1,"tri_att_end"),(1,"pair_transition"),
        (2,"tri_att_start"),(2,"tri_att_end"),(3,"tri_att_end")}
M = ("tri_mul_out","tri_mul_in","tri_att_start","tri_att_end","pair_transition")
p = [r"^(?!pairformer_stack\.)"]
for b in range(4):
    for m in M:
        if (b,m) not in DEAD: p.append(rf"blocks\.{b}\.{m}\.")
print(",".join(p))
PYX
)"
say "train-only: $TRAIN_ONLY"

B=$(/home/ebrahim/Protenix/.venv/bin/python -I run_logging/cards_busy.py) || B=fail
[ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
for c in 4 5 7; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c 2>/dev/null)
  case "$F" in ''|*[!0-9]*) say "REFUSING: GPU $c unreadable"; exit 3;; esac
  [ "$F" -lt 80000 ] && { say "REFUSING: GPU $c only ${F} MiB free"; exit 3; }
done
say "cards 4,5,7 clear"

# CKPT_THRESH 1250: units at or below 1,250 tokens run UNCHECKPOINTED. Measured, not fitted --
# the tvqobh fit (peak = 2.02 + 2.896e-05*t^2) does NOT transfer to this architecture, it
# predicted 67 GB at 1,499 tokens where the real answer was an OOM. Measured here, one rank,
# this exact config, checkpointing OFF:
#     1,004 tok  pass A 13.95  pass B 56.81 GB   OK
#     1,254 tok  pass A 20.78  pass B 87.62 GB   OK   <- the worst case at this threshold
#     1,499 tok  pass A 28.93  pass B OOM
#     1,822 tok  pass A 41.91  pass B OOM
# Runtime token counts equal the manifest's `tokens` column (verified on 7v61: 1254 = 1254),
# so no unit can exceed the threshold and 87.62 GB is a true ceiling, not an estimate.
# It is 92% of a 94.97 GB card. tztukd and kfthfe both died at step 158 at 84%, so if this
# arm dies it will be deep in an epoch: CKPT_THRESH=1000 (pass B 56.81 GB) is the fallback.
CKPT_THRESH="${CKPT_THRESH:-1250}" \
  # CKPT_THRESH 1250: units at or below 1,250 tokens run UNCHECKPOINTED. Measured, not fitted --
# the tvqobh fit (peak = 2.02 + 2.896e-05*t^2) does NOT transfer to this architecture: it
# predicted 67 GB at 1,499 tokens where the real answer was an OOM. Measured here, one rank,
# this exact config, checkpointing OFF:
#     1,004 tok  pass A 13.95  pass B 56.81 GB   OK
#     1,254 tok  pass A 20.78  pass B 87.62 GB   OK   <- the worst case at this threshold
#     1,499 tok  pass A 28.93  pass B OOM
#     1,822 tok  pass A 41.91  pass B OOM
# Runtime token counts equal the manifest `tokens` column (verified on 7v61: 1254 = 1254), so
# no uncheckpointed unit can exceed the threshold and 87.62 GB is a ceiling, not an estimate.
# That is 92% of a 94.97 GB card. tztukd and kfthfe both died at step 158 at 84%, so if this
# arm dies it will be deep in an epoch: CKPT_THRESH=1000 (pass B 56.81 GB) is the fallback.
CKPT_THRESH="${CKPT_THRESH:-1250}" \
  SKIP="4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m" EPOCHS="${EPOCHS:-20}" NOEVAL=0 DRY="${DRY:-0}" \
  CARDS=4,5,7 INBLOCK=0 FREEZE_BLOCKS=0 LEAN_SINGLE=1 MEMLOG=1 \
  TRUNK_CACHE_GB=10 TRUNK_MEMMAP=1 EVAL_POOL=training/valfull83/pool.csv LR="${LR:-1e-3}" \
  EXTRA="--skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --eval-pool-every 5 --train-only $TRAIN_ONLY ${EXTRA_APPEND:-}" \
  ./shell_scripts/launch_default_arm.sh > $SD/arm_fullhead.log 2>&1
say "launcher exit $?"
NEW=$(grep -oE "^run [a-z0-9]{6}" $SD/arm_fullhead.log | head -1 | awk '{print $2}')
[ -z "$NEW" ] && { say "DID NOT START:"; grep -aE "REFUS|STOP|Traceback" $SD/arm_fullhead.log | tail -5; exit 4; }
echo "$NEW" > $SD/fullhead_rid
say "full-head five-site arm = run $NEW"
