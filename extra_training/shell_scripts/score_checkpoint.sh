#!/bin/bash
# SCORE ONE HEAD on the fixed 280-item training slice and on val, then print the table.
#
# DROPOUT IS OFF HERE and cannot be otherwise: phase_k_eval builds its own model and never takes
# --pair-dropout. Stated because "scored without dropout" is a claim worth being able to point at.
#
# ONE PASS PER SIDE. Each pass emits our score and the shipped score for the same candidate, so
# the shipped reference row costs no extra GPU.
set -u
# PIPEFAIL, BECAUSE THE REPORT IS PIPED INTO tee. score_report.py asserts its candidate count and
# exits non-zero on a short pass; without this the pipeline took tee's status, the assert went to
# stderr unnoticed and tee had already truncated the report to zero bytes. Ten reports were
# silently emptied that way once.
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
mkdir -p "${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}"
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
CKPT=${CKPT:?set CKPT to a head_ep*.pt}
TAG=${TAG:?set TAG, e.g. ep20}
SIDES=${SIDES:-both}          # both | val | slice
# RE-SCORING COSTS A FULL PASS PER SIDE. The run already wrote its own shards under
# training/runs/<rid>/{valfull83,slice280}_eval_shards_ep<N>/ at every scoring epoch; point
# run_logging/score_report.py at those instead of running this script, unless the checkpoint was
# never scored in-training (the epoch-2 head of a stage that ended before its trigger, say).
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# THE PATHWAY FLAGS ARE READ OFF THE RUN'S OWN args.json, NOT RETYPED. Arm B (w57hjo) trained with
# --pose-readout --pose-readout-prenorm --pose-readout-post-zero on top of the three flags COMMON
# hardcodes; its full-training-set pass was launched with POSE_FLAGS unset and all three shards
# refused the checkpoint 90 s in ("6 checkpoint tensor(s) the model does not have"). That is the
# lucky failure -- phase_k_eval says --pose-readout-prenorm and --pose-perblock-share add NO
# tensors, so when one of those is the flag left off, the load succeeds and the number is wrong.
# Setting POSE_FLAGS explicitly still overrides this; an empty derivation (no args.json beside the
# checkpoint) leaves the command line exactly as it was. COMMON's own three pathway flags are
# therefore repeated on most runs -- harmless and left visible rather than filtered: argparse
# takes a repeated store_true (and a repeated int of the same value) as one, verified, and a
# filter would be another place for the two lists to disagree.
if [ -z "${POSE_FLAGS:-}" ]; then
  POSE_FLAGS=$($PY -I run_logging/pathway_flags.py "$CKPT") || {
    echo "STOP: could not READ $(dirname "$CKPT")/args.json, so the pathway flags are unknown."
    echo "      Set POSE_FLAGS explicitly if you know them; an empty set would score the head"
    echo "      through a pathway it may not have been trained through."; exit 4; }
  if [ -n "$POSE_FLAGS" ]; then
    say "POSE_FLAGS derived from $(dirname "$CKPT")/args.json: $POSE_FLAGS"
  else
    # SAID OUT LOUD EVEN WHEN IT IS EMPTY. A silent empty derivation is indistinguishable in the
    # log from a checkpoint that genuinely needs no extra flag, and --pose-perblock-share and
    # --pose-readout-prenorm add NO tensors, so nothing downstream would catch the difference.
    say "POSE_FLAGS: nothing derived for $CKPT (no args.json beside it) -- COMMON's three pathway flags are all this pass installs"
  fi
fi

# NO --drop-unmapped-poses HERE. The in-training eval does not forward it, so with it the two
# harnesses measure different candidate sets: a pose OST cannot map is silently dropped in one and
# a hard stop in the other. Loud in both.
# --prefetch: read the next unit's trunk on a background thread while the GPU scores this
# one. Its own help measures load at 4.7 min of a 26.1 min pass, strictly serial with the
# forward -- which is what the idle gaps between units were on the epoch-1 and -2 passes.
# BOTH POOLINGS, ALWAYS. --pool-mean adds an iptm_mean column computed in the same forward, so
# the mean-pooled reading costs no extra GPU -- but it is default off, so every arm scored before
# this has rows that CANNOT answer "was the mean better" offline and would need a full re-score.
# The ranking arms train against --rank-pool mean while every baseline here selects by the max, so
# that question comes up for each of them. Computing both and choosing later is the cheap order.
#
# --pair-iptm-only IS REQUIRED BY --pool-mean (its path already holds the PAE probabilities). It
# does not move the candidate set, iptm, stored_iptm, the DockQ join or the rows schema, and the
# in-training eval has passed it on every pass -- phase_k_train.py calls it measured
# bit-identical. NOT independently re-measured here: the val30_zs_ep{8,10} rows already on disk
# were scored WITHOUT it, so the next pass over either epoch is a free check of that claim.
#
# SELECTION STAYS ON THE MAX. --select-by is not passed, so mean_dockq and every existing column
# keep the shipped convention and stay comparable to every earlier report; the mean-pooled numbers
# arrive as an extra column to be read offline, not as a changed headline.
COMMON="--pose-perblock --pose-perblock-copy --frame-direction --prefetch --pair-iptm-only --pool-mean ${POSE_FLAGS:-}"

# THE POOL IS OVERRIDABLE, everything else about this pass is not. SLICE_POOL lets the same
# argv score the FULL training set (training/combined/pool.csv, 1688 interfaces) instead of the
# 280-item slice, without hand-building a command line -- the one thing that has cost this
# project four failed launches in twenty minutes before. Its join cache is keyed to the pool,
# so it moves with it rather than being shared and silently stale.
#
# DEFINED OUTSIDE THE BRANCH, because the report at the bottom reads SLICE_NAME: it used to point
# at a hardcoded slice280_$TAG and so reported NOTHING for a renamed pool, and the ABORT handler
# tailed the same hardcoded path -- which is why today's failed full-training launch printed
# "tail: cannot open .../slice280_armB_ep20.log" instead of the shard error.
SLICE_POOL=${SLICE_POOL:-$R/training/slice280/pool.csv}
SLICE_CACHE=${SLICE_CACHE:-$R/training/slice280/join_cache.json}
SLICE_NAME=${SLICE_NAME:-slice280}
SLICE_MANIFEST=${SLICE_MANIFEST:-}
SLICE_BUDGET=${SLICE_BUDGET:-15}
SLICE_MAN_ARGS=""
SLICE_DEPTH_ARGS=""

# THE WHOLE TRAINING SET NEEDS A PER-STRUCTURE DEPTH, so the budget is overridable too and a
# manifest can be pointed at. 947 of training/combined/pool.csv's 1,688 interfaces have 15
# trunks and 741 have exactly one, so a uniform --budget 15 over it stops on 5,516 of 12,795
# intended units -- correctly, because that label would be a lie for 44% of the pool. With
# SLICE_MANIFEST set, phase_k_eval reads each structure's own depth from the manifest and
# reports one stratum per depth beside the pooled number.
#
# --mixed-budget IS DERIVED FROM THE MANIFEST, NEVER TYPED HERE. The flag and the manifest's
# own mixed:true must agree or phase_k_eval hard-stops either way, so reading it out of the
# file is the only way the two cannot drift -- and a uniform manifest (valfull83's, say) still
# goes through unchanged with the uniform assertion fully in force.
#
# RESOLVED OUTSIDE THE SIDES BRANCH, because the report at the bottom runs on EVERY invocation.
# Inside the branch, scoring the two sides as two invocations -- the natural way to schedule a
# 75-minute mixed pass around GPU availability -- reached the report from the SIDES=val run with
# SLICE_DEPTH_ARGS empty and EXPECT_CANDS "slice=75", pointed at the mixed slice dir, and STOPped
# on the 741 five-candidate items AFTER the val pass had been paid for. Found in review.
if [ -n "$SLICE_MANIFEST" ]; then
  [ -f "$SLICE_MANIFEST" ] || { say "ABORT: SLICE_MANIFEST $SLICE_MANIFEST is not a file"; exit 5; }
  SLICE_MAN_ARGS="--seed-manifest $SLICE_MANIFEST"
  if $PY -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('mixed') else 1)" \
       "$SLICE_MANIFEST"; then
    SLICE_MAN_ARGS="$SLICE_MAN_ARGS --mixed-budget"
    SLICE_DEPTH_ARGS="--slice-depths $SLICE_MANIFEST"
    say "$TAG: $(basename "$SLICE_MANIFEST") declares mixed:true -- MIXED-BUDGET pass, deepest $SLICE_BUDGET seed(s); the pooled top-1 is NOT comparable to a uniform-budget pass"
  fi
fi
# AND THE POOL IS CHECKED BEFORE THREE CARDS ARE SPENT. A typo'd SLICE_POOL printed
# "-1 interface(s)" -- $(( $(wc -l < missing) - 1 )) -- and then launched three shards to die
# inside phase_k_eval a minute later.
[ -f "$SLICE_POOL" ] || { say "ABORT: SLICE_POOL $SLICE_POOL is not a file"; exit 6; }

if [ "$SIDES" = both ] || [ "$SIDES" = slice ]; then
  say "$TAG: scoring $(basename "$(dirname "$SLICE_POOL")") -- $(( $(wc -l < "$SLICE_POOL") - 1 )) interface(s), $SLICE_BUDGET seeds"
  EXTRA="--fold $R/training/fold --results $R/training/fold/results.csv \
--pool $SLICE_POOL --roles $R/training/fold/chain_roles_all.csv \
--seed-base 500000 --join-cache $SLICE_CACHE $SLICE_MAN_ARGS $COMMON"
  # DRY=1 PRINTS THE ARGV AND LAUNCHES NOTHING. A wrong pathway flag or a wrong budget costs
  # three cards and however long it takes the shards to reach the refusal -- 90 s for the
  # checkpoint guard, 25 min for anything the coverage contract catches. This is how the argv for
  # a new pool gets read before it is run, and it is the only way to test this script without a GPU.
  if [ -n "${DRY:-}" ]; then
    echo "DRY slice: $PY -I phases/phase_k_eval_sharded.py --cards 4,5,7 --ckpt $CKPT --budget $SLICE_BUDGET --work $SD/${SLICE_NAME}_$TAG --out $SD/${SLICE_NAME}_$TAG.json --extra \"$EXTRA\""
  else
  LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
      --cards 4,5,7 --ckpt "$CKPT" --budget "$SLICE_BUDGET" \
      --work "$SD/${SLICE_NAME}_$TAG" --out "$SD/${SLICE_NAME}_$TAG.json" \
      --extra "$EXTRA" > "$SD/${SLICE_NAME}_$TAG.log" 2>&1 \
      || { say "ABORT: slice pass exit $?"; tail -3 "$SD/${SLICE_NAME}_$TAG.log"; exit 2; }
  fi
fi

if [ "$SIDES" = both ] || [ "$SIDES" = val ]; then
  say "$TAG: scoring val (141 items, 30 seeds)"
  EXTRA="--fold $R/validation/fold --results $R/validation/fold/results.csv \
--pool $R/training/valfull83/pool.csv --roles $R/validation/fold/chain_roles_all83.csv \
--seed-base 400000 --seed-manifest $R/training/valfull83/seed_manifest.json \
--join-cache $R/training/valfull83/join_cache.json $COMMON"
  if [ -n "${DRY:-}" ]; then
    echo "DRY val:   $PY -I phases/phase_k_eval_sharded.py --cards 4,5,7 --ckpt $CKPT --budget 30 --work $SD/val30_$TAG --out $SD/val30_$TAG.json --extra \"$EXTRA\""
  else
  LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
      --cards 4,5,7 --ckpt "$CKPT" --budget 30 \
      --work "$SD/val30_$TAG" --out "$SD/val30_$TAG.json" \
      --extra "$EXTRA" > "$SD/val30_$TAG.log" 2>&1 \
      || { say "ABORT: val pass exit $?"; tail -3 "$SD/val30_$TAG.log"; exit 3; }
  fi
fi

# THE REPORT'S EXPECTATIONS MOVE WITH THE POOL. score_report.py refuses a table over a subset by
# asserting the item count and the candidates per item, and both were hardcoded to the 280-item
# slice at 75 candidates -- so a 947- or 1,688-interface pass ended in a STOP after the GPU work
# was already done. Derived from SLICE_POOL and SLICE_BUDGET instead; under a mixed manifest the
# per-item candidate count is per structure, so --slice-depths supplies it and the table is split
# into one row per depth rather than pooling 75-candidate and 5-candidate items into one rate.
SLICE_N=$(( $(wc -l < "$SLICE_POOL") - 1 ))
# --slice-pool ALWAYS, and the bins from the pool's OWN directory. The table is named after the
# pool it covers, and only the real 280-item slice may be called "TRAIN SLICE": a 947-interface
# in-sample table under that name is read back into the slice panel by
# plot_fullset.report_points, which greps ^TRAIN SLICE out of these reports -- two different
# pools averaged into one curve. training/slice280/items.csv was likewise the hardcoded default
# for the per-bin split, and its bins describe ITS items only.
SLICE_ITEMS="$(dirname "$SLICE_POOL")/items.csv"
if [ -n "$SLICE_DEPTH_ARGS" ]; then
  EXPECT_CANDS="val=150"
else
  EXPECT_CANDS="slice=$((SLICE_BUDGET * 5)),val=150"
fi
if [ -n "${DRY:-}" ]; then
  echo "DRY report: $PY -I run_logging/score_report.py --slice-work $SD/${SLICE_NAME}_$TAG --val-work $SD/val30_$TAG --label $TAG --expect-slice $SLICE_N --expect-cands $EXPECT_CANDS --slice-pool $SLICE_POOL --slice-items $SLICE_ITEMS $SLICE_DEPTH_ARGS"
  exit 0
fi
$PY -I run_logging/score_report.py --slice-work "$SD/${SLICE_NAME}_$TAG" \
    --val-work "$SD/val30_$TAG" --label "$TAG" \
    --expect-slice "$SLICE_N" --expect-cands "$EXPECT_CANDS" \
    --slice-pool "$SLICE_POOL" --slice-items "$SLICE_ITEMS" $SLICE_DEPTH_ARGS \
    | tee "$SD/report_$TAG.txt"
