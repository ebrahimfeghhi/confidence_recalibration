#!/bin/bash
# THE DEFAULT ARM: --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED on both
# sides, over the 851-structure mixed-depth population.
#
# rel IS THE DEFAULT BY THE OWNER'S STANDING INSTRUCTION, 2026-11-06, and no other --rank-form is
# substituted without asking. It regresses each pose's ipTM on the TRUE iTM after subtracting the
# interface's own mean:
#
#     L = (1/P) sum_i [ (s_i - mean_s) - (g_i - mean_g) ]^2
#
# so it reads every pose of an interface regardless of seed. There is NO pair family, NO tau, and
# --rank-pairs and --rank-within-weight do not apply -- they are absent from the argv below rather
# than set to something that looks operative.
#
# WHY rel RATHER THAN A PAIRWISE FORM, measured not assumed: the head's failure is POSITIONAL, not
# collapse. On d32kqp epoch 66 the correlation between interface-MEAN ipTM and interface-MEAN DockQ
# was +0.866 while the median WITHIN-interface Spearman was +0.317, against +0.690 for the true
# iTM. Subtracting each interface's own mean from both sides removes the across-complex signal the
# head already has, leaving only the within-complex ordering that selection uses.
#
# ITS COEFFICIENTS SUM TO EXACTLY ZERO, so it cannot move an interface's level and its accumulated
# gradient cancels by construction. --alpha-rank must come from gradient NORMS via --grad-probe and
# never from loss values; 75 is the owner's figure, carried from the earlier rel arms.
#
# WHY UNCROPPED ON BOTH SIDES. Every previous arm trained on a crop and evaluated uncropped, and the
# gap shows: o2qczq scored 0.6305 cropped and 0.5387 uncropped, below shipped 0.5474. A cropped
# reading carries forced ground-truth contacts and is never a deployment number. Uncropped costs
# ~57 s/step and ~3.6 h/epoch at world=3, so quote that before proposing a long run.
#
# --ckpt-token-threshold IS 0 HERE, AND 855 IS WHAT KILLED tztukd. The flag means units AT OR BELOW
# that many tokens run with checkpointing OFF. 855 was MEASURED by run_logging/bench_ckpt_by_size.py
# on a population whose largest unit was 1,543 tokens AND which trained under token dropout, so its
# pair work was about a quarter of full. This population's largest unit is 2,289 tokens and there is
# no dropout, so an unchecked sub-855 unit no longer fits: tztukd died of CUDA OOM 2.6 h into epoch 1
# on 8g30, a 731-token structure, in the backward through pairformer.pair_transition, with 93.9 of
# 94.97 GiB in use on two ranks at once.
#
# The flag's own help says "SET IT FROM run_logging/bench_ckpt_by_size.py, NOT BY GUESSING ... A
# threshold chosen by arithmetic would be an OOM 40 minutes into an epoch." I carried a measured
# number to a population it was not measured on, which is the same mistake as guessing. 0 disables
# the size rule so every unit keeps --blocks-per-ckpt 1.
#
# IT COSTS COMPUTE AND NO MATHEMATICS. Checkpointing recomputes the head's four pairformer blocks in
# backward, about 30% more compute there, and gpu_backward was 70.3 min of a 143.9 min epoch at the
# old scale. It is gradient-exact: test_perblock_ckpt_equiv.py gates blocks_per_ckpt=1 against off
# at 0.000e+00 on every parameter. So this slows the arm and changes nothing about what it learns.
# Re-benchmark on THIS population to recover the speed; do not reuse 855.
#
# --indices all857_indices.csv IS REQUIRED. phase_k_train and the fingerprint gate both DEFAULT to
# job_indices.csv, which names 501 structures; this population's labels were built against
# all857_indices.csv and the indices path is part of the label fingerprint. Left on the default the
# run rejects every label it loads -- the gate reported 2,010 such labels over 223 structures, with
# `code` and `loss` matching exactly, which is the signature of a wrong indices file and not of a
# stale label.
#
# --alignment-preflight IS MANDATORY AND UNCONDITIONAL. The guard is `if True:` because the full
# dataset-item path was removed, so the minimal path is the only path and check_alignment can no
# longer run per structure. Dropping --minimal-items does NOT avoid it. The marker is built by
# phases/phase_k_alignment_preflight.py --workers 9 over this exact manifest and its digest must
# equal the trainer's or the launch refuses.
#
# THREE STRUCTURES ARE SKIPPED, EACH FOR A MEASURED REASON:
#   9mx1  never folded -- 10 GPU units queued at training/fold/queue_9mx1.tsv, never run.
#   4zto  its ost_json does not associate a pose with a sample ("2.json describes , not sample 2"),
#         because it was scored by the full-complex route.
#   9q5m  its E-H interface has no rows at seeds 500010-500014 and no route can produce them: chain
#         E is the copy with only 81 of 532 residues resolved, so it ties at 1.00 against all four
#         model antigen copies and only geometry can say which. Recorded with its evidence in
#         docs/unrecoverable_interfaces.csv. Skipping the whole structure also drops its five
#         HEALTHY interfaces; a per-interface exclusion would be better and does not exist yet.
# So the trained population is 851 of 853 declared structures and 6,941 of 6,971 declared units,
# packing into 231 cross-seed steps of 30 units. Those are the numbers this script's banner quotes;
# an earlier version quoted the pre-exclusion 6,971/232 and was wrong by the three skips.
#
# FIVE GATES RUN BEFORE ANY CARD IS TOUCHED: the pool agrees with the declared depth; every declared
# unit has a trunk and a label; the alignment marker passed and covers this manifest; the rel labels
# exist, report zero unmapped rows and cover every trained structure AND every expected
# (seed, sample, iface) key; and the cards are free with
# MemAvailable above the 250 GB floor other users need.

set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-10}"
DRY="${DRY:-1}"
MANIFEST=train_val_splits/cache_train_structures_extended.csv
MARKER="${MARKER:-training/alignment_preflight_sharded.json}"
RELLAB="${RELLAB:-training/fold/itm_labels_extended}"
# 0 MEANS CHECKPOINT EVERY UNIT, which is the only safe default for an UNCROPPED population.
# TEN STRUCTURES DO NOT FIT UNCROPPED ON A 95 GiB CARD, AND THAT IS MEASURED, NOT GUESSED. The
# --mem-log run tvqobh recorded max_memory_allocated per pose against the unit's token count, and
# the pass-B peak is clean n^2:
#
#     peak_GB = 2.02 + 2.896e-05 * tokens^2
#
# residual under 0.1 GB from 268 to 758 tokens, and a 1,364-token unit that arrived AFTER the fit
# landed at 56.19 GB against a predicted 55.90 -- a 2.4x extrapolation in n, 5.8x in n^2. So this
# is the pair tensor's own scaling and it extrapolates.
#
# The budget for one pose is 94.97 GiB card - 0.78 GPU holders - 3.80 post-pose baseline = 90.4 GB,
# which puts the ceiling at ~1,747 tokens. At a 10% margin (1,655 tokens) ten structures are over:
#
#     9dzf 2289 (154 GB)  9dzg 2289 (154)  7m7i 2004 (118)  9b9j 1949 (112)  9e7x 1879 (104)
#     9q5v 1822 (98)      9q7b 1821 (98)   8vba 1786 (94)   9bbn 1783 (94)   7cn2 1745 (90)
#
# THAT IS WHAT KILLED BOTH 10-EPOCH ATTEMPTS. tztukd and kfthfe died at the SAME step, 158 of 231,
# under --ckpt-token-threshold 855 and 0 -- so the threshold was never the cause. --seed 0 fixes
# the epoch shuffle, so the same oversized structure lands at the same step every run. The OOM
# reported 93.02 GiB allocated, which back-solves to ~1,745-1,790 tokens; of the ten, only 9e7x
# and 7cn2 carry 15 seeds, and a deep structure puts all 15 of its units in ONE cross-seed step,
# which is how one step takes down two ranks at once.
#
# IT COSTS 1.2% OF THE POPULATION: 851 -> 841 structures, 6,941 -> 6,903 units (0.55%), 230 steps
# instead of 231. The largest remaining unit is 1,650 tokens, predicted peak 80.9 GB, a 10.6%
# margin. Excluding them changes neither the objective nor the input distribution for the 841 that
# remain, which is less interference than --nsample or a crop would be. 30 of the 38 dropped units
# are 9e7x and 7cn2's seed groups and did cost GPU to fold; cropping just those two would keep them
# at the price of a train/eval mismatch, and that trade was declined in favour of excluding.
OVERSIZED="9dzf,9dzg,7m7i,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,7cn2"
SKIP="${SKIP:-9mx1,4zto,9q5m,$OVERSIZED}"
# WHETHER THE CALLER SET IT, captured before the default is applied, so INBLOCK can choose a
# different default without overriding an explicit choice.
CKPT_THRESH_SET="${CKPT_THRESH+yes}"
CKPT_THRESH="${CKPT_THRESH:-0}"
# ACTIVATION CHECKPOINTING IS LARGELY POINTLESS ONCE THE STACK IS FROZEN, so an INBLOCK arm takes
# a different default. DECIDED HERE, not at the INBLOCK block further down, because ARGS is built
# at :276 and would already hold the old value by then -- INBLOCK is read early for this reason
# alone and is re-defaulted to the same value below.
#
# --ckpt-token-threshold means "units AT OR BELOW this many tokens run with checkpointing OFF".
# The default 0 DISABLES the rule, so every unit uses --blocks-per-ckpt 1. mqird6 was launched with
# 855 and its epoch-6 log reads "3900 unit(s) UNCHECKPOINTED, 700 checkpointed" -- so the unfrozen
# arm already ran 85% of its units uncheckpointed, and an INBLOCK arm left at the default would
# checkpoint MORE than the arm it is meant to beat.
#
# Checkpointing buys not storing activations at the price of a second forward. With blocks 0-2 and
# block 3's two tri-muls frozen the backward begins after tri_mul_in, so the only activations
# needing storage are block 3's tri_att_start, tri_att_end and pair_transition plus the injections.
# Paying a whole extra forward to avoid storing that is the wrong trade.
#
# 99999 means every unit, the largest in the manifest being 1,650 tokens. UNMEASURED: mqird6 peaks
# at 81.30 GB of 95.6 GB WITH a full backward, and whether the frozen graph fits uncheckpointed at
# 1,650 tokens is exactly what gate 3 of inblock_gpu_smoke.sh measures on 7cn2. If it does not fit,
# pass CKPT_THRESH=855 and the size rule returns.
if [ "${INBLOCK:-0}" = "1" ] && [ -z "$CKPT_THRESH_SET" ]; then
  CKPT_THRESH=99999
fi
# NOEVAL=1 DROPS THE HELD-OUT EVALUATION. The owner asked for training only on 2026-11-06; the
# eval-free config existed only in resume_noeval_to_10.sh, which needs a bundle, so a fresh
# relaunch after an OOM silently brought eval back. A fresh launch must be able to honour it.
NOEVAL="${NOEVAL:-0}"
# MEMLOG=1 ADDS --mem-log, FOR TELLING A RETAINED REFERENCE FROM A CAPACITY CEILING. Both
# uncropped attempts died at the SAME step (158 of 231) with different --ckpt-token-threshold
# values, 855 and 0, which already rules the threshold out. The OOM reported 93.02 GiB ALLOCATED
# -- live tensors -- with 124 MiB of allocator slack, on a 731-token unit with checkpointing on.
# paper_results.md section 11 measured this regime with nvidia-smi, which reports RESERVED, so it
# could not separate the two. --mem-log prints allocated AND reserved per pose with the pass label
# at a consistent boundary, which can. Pair it with a fractional EPOCHS (0.15 is ~35 steps) so the
# question is answered in half an hour instead of three.
#
# DO NOT PAIR IT WITH A FRACTIONAL EPOCHS UNDER --rank-cross-seed. 0.15 was refused, correctly:
# "the cross-seed quotas sum to 350 unit(s) but the epoch holds 346". The per-structure quotas are
# whole seed GROUPS, so truncating the unit list mid-step would make the ranks step at different
# times. Use EPOCHS=1 and read the first few dozen MEM lines; the run then also reproduces the
# step-158 OOM with the instrumentation on, which is strictly more than a truncated run gives.
#
# DO NOT PAIR IT WITH A FRACTIONAL EPOCHS UNDER --rank-cross-seed. 0.15 was refused, correctly:
# "the cross-seed quotas sum to 350 unit(s) but the epoch holds 346". The per-structure quotas are
# whole seed GROUPS, so truncating the unit list mid-step makes the ranks step at different times.
# Use EPOCHS=1 and read the first few dozen MEM lines; the run then also reproduces the OOM at
# step 158 with the instrumentation on, which is strictly more than a truncated run would give.
MEMLOG="${MEMLOG:-0}"
# FREEZE_BLOCKS=1 TRAINS ONLY PAIRFORMER BLOCK 3 AND FREEZES BLOCKS 0-2. No trainer change: the
# existing --train-only takes a comma-separated list of REGEXES matched with re.search against each
# head tensor's name relative to `confidence_head.`, and it genuinely calls p.requires_grad_(False)
# on everything unmatched (phases/phase_k_train.py:2745) rather than merely filtering the optimiser.
# So with only blocks.3 trainable nothing below block 3 requires grad and AUTOGRAD BUILDS NO GRAPH
# THROUGH BLOCKS 0-2 AT ALL. The head's stack is 4 UNTIED blocks (nn.ModuleList, a distinct block
# per layer) of 4,749,184 parameters each, 24.0% of the head each, with 755,534 params (3.8%)
# outside it -- so a quarter of the backward is a fair division.
#
# EXPECTED 1.5-1.7x, FROM mqird6's OWN EPOCH-3 TIMERS and nothing else: gpu_backward 110.0 min of a
# 232.2 min epoch over 11,500 calls, gpu_forward 69.2 min over 23,000 (11,500 pass A + 11,500 pass
# B), overhead ~53 min. Backward 110 -> ~28 min, forward UNCHANGED at 69.2 because block 3's output
# still requires all four blocks to run: epoch 232 -> ~150 min, i.e. 1.55x, and up to ~1.67x if the
# ~4x smaller retained activations let --ckpt-token-threshold rise and remove the checkpoint
# re-forward (measured backward/forward is 3.2x per pose, 0.574 vs 0.181 s, where theory says ~2x).
# THE ESTIMATE IS ANALYTICAL AND UNVALIDATED. An earlier projection by this same method, made the
# same day, predicted 1.35x where the measured figure was 1.10x -- 25% optimistic. Falsify it in
# ~20 steps against mqird6's measured 62.1 s/step; a collapsed backward shows as ~40 s immediately.
#
# IT IS A CAPACITY CUT, NOT A FREE OPTIMISATION, AND NEEDS ITS OWN CONTROL. 4.75M trainable
# parameters instead of 19.75M. An arm run with this is NOT comparable against mqird6 or against any
# full-head arm; it needs its own held-out control. It also makes per-block injection untrainable in
# blocks 0-2 (--pose-perblock, --frame-direction at all four depths), so do not combine it with
# those without saying what that means.
#
# WHY ~20 PRIOR ARMS DID NOT ALREADY GET THIS SPEEDUP. ~20 arms already used --train-only, but every
# one of them targeted INPUT-SIDE parameters -- linear_no_bias_d, pose_perblock, frame_dir -- which
# sit at the BOTTOM of the stack. The gradient still had to traverse all four blocks to reach them,
# so those arms paid the FULL backward while updating almost nothing. This is the first
# configuration that puts the trainable set at the TOP, which is the only place the saving lives.
FREEZE_BLOCKS="${FREEZE_BLOCKS:-0}"
# OVERRIDABLE, so "freeze 0-1" or "freeze 0" can be tried without editing the script. It is a
# regex, not a prefix, and the backslashes must survive: inside DOUBLE quotes bash honours a
# backslash only before $ ` " \ and newline, so `\.` passes through verbatim here. An override
# given on the command line wants SINGLE quotes, since an unquoted \. would be eaten by the shell.
#
# IT ALSO FREEZES THE PAE READOUT, which is easy to miss and may not be what you want: everything
# outside blocks.3 includes linear_no_bias_pae and pae_ln, the two tensors the objective's own
# output passes through. Those sit ABOVE the stack (confidence.py applies them to z_pair after
# pairformer_stack returns), so unfreezing them adds no backward depth at all -- their gradient
# needs only their input and their output gradient. If the arm should keep an adaptable readout,
# use
#     FREEZE_PAT='pairformer_stack\.blocks\.3\.,^linear_no_bias_pae\.,^pae_ln\.'
# which is 4.75M + ~16.9k parameters (64x256 for the projection plus pae_ln's 256 weight and 256
# bias) and the same expected speedup. Left out of the default
# because the 1.5-1.7x figure above was computed for one block exactly.
FREEZE_PAT="${FREEZE_PAT:-pairformer_stack\.blocks\.3\.}"
POOL=training/combined/pool.csv
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  DRY=$DRY cards=$CARDS epochs=$EPOCHS"

[ -s "$MANIFEST" ] || { say "REFUSING: no $MANIFEST"; exit 1; }
say "gate 1: the pool must agree with the declared depth"
$PY -I run_logging/build_combined_train_pool.py --expect-manifest "$MANIFEST" \
  --allow-short docs/unrecoverable_interfaces.csv >/dev/null \
  || { say "REFUSING: the pool gate fails"; exit 1; }
say "  OK"
say "gate 2: every declared unit has a trunk and a label"
$PY -I run_logging/build_extended_train_selection.py \
  --reclaim-journal training/runs/reclaim_train.json \
  --reclaim-journal training/runs/reclaim_train_wave1.json --require-ready >/dev/null \
  || { say "REFUSING: a declared unit is missing a trunk or a label"; exit 1; }
say "  OK"
say "gate 3: the alignment marker exists, passed, and covers THIS manifest"
[ -s "$MARKER" ] || { say "REFUSING: no alignment marker at $MARKER"; exit 1; }
$PY -I - <<EOF || exit 1
import json, sys
j = json.load(open("$MARKER"))
if not j.get("ok") or not j.get("complete"):
    sys.exit(f"REFUSING: {'$MARKER'} is not a pass (ok={j.get('ok')}, complete={j.get('complete')})")
print(f"  marker ok, {j.get('n_structures')} of {j.get('n_manifest_structures')} structure(s), "
      f"digest {j.get('manifest_digest')}, sharded={j.get('sharded')}")
EOF
say "gate 4: the rel labels exist and cover the trained population"
[ -d "$RELLAB" ] || { say "REFUSING: no rel labels at $RELLAB"; exit 1; }
RELLAB="$RELLAB" MANIFEST="$MANIFEST" $PY -I - <<'EOF' || exit 1
import csv, json, os, sys
rl, mf = os.environ["RELLAB"], os.environ["MANIFEST"]
skip = {"9mx1", "4zto", "9q5m"}
want = [r["pdb"].strip().lower() for r in csv.DictReader(open(mf))
        if r["pdb"].strip().lower() not in skip]
miss = [p for p in want if not os.path.exists(os.path.join(rl, f"{p}.npz"))]
m = os.path.join(rl, "manifest.json")
j = json.load(open(m)) if os.path.exists(m) else {}
print(f"  rel labels: {j.get('structures')} structure(s), {j.get('rows')} row(s), "
      f"unmapped {j.get('rows_unmapped')}, pool {os.path.basename(str(j.get('pool')))}")
if miss:
    sys.exit(f"REFUSING: {len(miss)} trained structure(s) have no rel label, e.g. {miss[:5]}")
if j.get("rows_unmapped"):
    sys.exit(f"REFUSING: the rel labels report {j['rows_unmapped']} unmapped row(s)")
print(f"  every one of {len(want)} trained structure(s) has a rel label")
EOF
# AND THE EXACT KEY SET, not just the files. Raised in review: a directory-level check cannot see a
# structure whose label file exists but is short of a seed, a sample or an interface, and the
# trainer only discovers that when the epoch reaches the pose -- up to 3.6 h in on this population.
$PY -I run_logging/check_rel_label_coverage.py --labels "$RELLAB" --manifest "$MANIFEST" \
  --pool "$POOL" --skip '9mx1,4zto,9q5m' --exclude-ifaces docs/unrecoverable_interfaces.csv \
  || { say "REFUSING: the rel label key set is incomplete"; exit 1; }
# A DRY RUN MUST NOT REQUIRE FREE CARDS. This gate sat before the DRY check, so validating a
# config while anything was training refused at "cards_busy reports '5'" and never printed the
# command line -- which made a dry run useless exactly when it is most wanted, and made my own
# test of the NOEVAL path report a false pass.
B=$($PY -I run_logging/cards_busy.py 2>/dev/null | tail -1)
if [ "$DRY" = "1" ]; then
  say "DRY RUN: cards_busy reports '$B' and MemAvailable $(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo) GB -- not gated on a dry run"
  B=0
fi
# PER-CARD, NOT GLOBAL. This used to refuse unless cards_busy reported 0 anywhere on the box.
# That is the right rule for "do not start a long arm while something else is training" and the
# wrong one for independent jobs on separate cards: the three gates of the GPU smoke, one per
# card, deadlocked each other because the first to start made the gate false for the other two.
# The property that actually matters is whether THE CARDS THIS RUN ASKED FOR are free.
#
# The 3-card ceiling is kept, and enforced here rather than trusted to the caller.
#
# cards_busy's box-wide count is still printed: "something else is training" is worth knowing even
# when it is no longer a refusal.
CARD_GATE_MIB="${CARD_GATE_MIB:-80000}"
# A DRY RUN MUST NOT REQUIRE FREE CARDS -- the same rule the box-wide check above already had, and
# which I dropped when replacing it. Capturing an arm's argv with DRY=1 while another job holds a
# card is exactly when a dry run is most useful, and refusing there made it useless at that moment.
_n_cards=$(echo "$CARDS" | tr ',' '\n' | grep -c .)
if [ "$DRY" = "1" ]; then
  say "DRY RUN: per-card gate skipped ($_n_cards card(s) requested: $CARDS)"
  _n_cards=0
fi
if [ "$_n_cards" -gt 3 ]; then
  say "REFUSING: $_n_cards cards requested ($CARDS); the standing limit is 3."
  exit 1
fi
_busy_ours=""
for _c in $([ "$DRY" = "1" ] || echo "$CARDS" | tr ',' ' '); do
  # DIGITS, NOT MERELY NON-EMPTY. `nvidia-smi -i 99` prints "No devices were found" to STDOUT and
  # exits 6, so an emptiness test passes, the numeric comparison below errors, `[` returns
  # non-zero, and the card is treated as FREE -- a bad index would have sailed through the gate.
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$_c" 2>/dev/null)
  case "$_free" in
    ''|*[!0-9]*)
      say "REFUSING: GPU $_c did not report a numeric free-memory value (got '${_free:-<empty>}')."
      say "  Is $_c a valid index on this box?"
      exit 1;;
  esac
  if [ "$_free" -lt "$CARD_GATE_MIB" ]; then
    _busy_ours="$_busy_ours $_c(${_free}MiB)"
  fi
done
if [ -n "$_busy_ours" ]; then
  say "REFUSING: requested card(s) not free:$_busy_ours (need ${CARD_GATE_MIB} MiB each)."
  say "  cards_busy reports '$B' box-wide."
  exit 1
fi
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "gate 5: cards $CARDS each >= ${CARD_GATE_MIB} MiB free (box-wide cards_busy '$B'), MemAvailable ${M} GB"
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable under the 250 GB floor."; exit 1; }

ARGS=(--rank-loss --rank-form rel --rel-labels "$RELLAB"
      --rank-cross-seed --rank-structs-per-step 2
      --alpha-rank 75 --rank-pool mean
      --fold training/fold
      --indices training/prep/all857_indices.csv
      --manifest-selection "$MANIFEST"
      --manifest-skip "$SKIP"
      --pae-scored-pairs "$POOL" --pae-scored-roles training/fold/chain_roles_all.csv
      --pae-only --lean-readouts --minimal-items
      --alignment-preflight "$MARKER"
      --epochs "$EPOCHS" --lr "${LR:-0.01}" --warmup-steps 20 --seed 0 --accum 30 --nsample 5
      --alpha-confidence 1.0 --alpha-pae 1.0 --clip-norm 25.0
      --prefetch-units 2 --prefetch-workers 4
      --blocks-per-ckpt 1 --ckpt-token-threshold "$CKPT_THRESH"
      --dist-backend gloo --dist-timeout-min 240
      --max-rss-gb 80 --preload-min-free-gb 250
      --save-every 250
      --require-replay training/extended_mixed_replay.log
      --allow-code-drift)
if [ "$MEMLOG" = "1" ]; then
  ARGS+=(--mem-log)
  say "MEMLOG=1: --mem-log is on. This is a DIAGNOSTIC, not an arm -- it prints a line per pose."
fi
if [ "$FREEZE_BLOCKS" = "1" ]; then
  ARGS+=(--train-only "$FREEZE_PAT")
  say "FREEZE_BLOCKS=1: --train-only '$FREEZE_PAT'. Pairformer blocks 0-2 are FROZEN; 4.75M of"
  say "  19.75M head parameters train. This is a CAPACITY CUT and needs its own control arm -- it"
  say "  is NOT comparable against mqird6. Expected 1.5-1.7x, ANALYTICAL AND UNVALIDATED."
  say "  Check the trainer's own '--train-only ...: N of 248 head tensor(s)' line before walking"
  say "  away: a pattern that matches nothing is refused, but one that matches TOO MUCH is not."
fi
# TRUNK_MEMMAP=1 MAPS z_trunk INSTEAD OF READING IT INTO A HEAP BUFFER. ~6 min of a 232 min epoch,
# by removing one pass over ~243 MB per unit (~1.1 TB per rank per epoch). It changes NO number the
# run produces -- phases/test_trunk_memmap.py gates the two paths bit-identical on real trunks --
# but it is UNTESTED ON A GPU, so it is off by default and goes on with its own eyes on it. It also
# moves the disk read out of trunk_npz_read and into trunk_h2d, because a mapping is lazy.
if [ "${TRUNK_MEMMAP:-0}" = "1" ]; then
  ARGS+=(--trunk-memmap)
  say "TRUNK_MEMMAP=1: --trunk-memmap is on. Numerically identical, ~6 min/epoch, UNTESTED on GPU."
fi
# LEAN_SINGLE=1 STOPS COMPUTING THE PAIRFORMER'S SINGLE TRACK, which --lean-readouts left running.
# z never depends on s (PairformerBlock.forward updates z through five ops that are functions of z
# alone, then updates s), and under --pae-only the readout reads z only -- so the single track was
# computed in all 4 blocks for every pose and thrown away.
#
# IT IS SMALL, AND THE NUMBERS ARE MEASURED NOT ARGUED. 10,551,552 of the head's 19,752,270
# parameters (53.42%) stop being nominally trainable and leave AdamW and the gradient all-reduce --
# but the pair track is O(N^3) and the single track O(N^2), so saved-activation bytes measured 4.0%
# of a block at 128 tokens, 2.9% at 256 and 2.5% at 384, and the time difference was BELOW CPU
# measurement noise. Expect ~1-2% of the epoch. It is here because it is free and provably dead.
#
# NUMERICALLY IDENTICAL: run_logging/test_lean_single.py gates z bit-identical with the track on and
# off, and gates that the single track's parameters got NO gradient even when it ran -- which is why
# dropping them from the optimiser is a no-op and not a capacity cut. UNTESTED ON A GPU.
# INBLOCK=1 INJECTS THE POSE INSIDE THE LAST PAIRFORMER BLOCK and freezes everything the injection
# is supposed to make unnecessary. It sets --train-only itself, overriding FREEZE_PAT, because the
# freeze and the injection placement are ONE design and splitting them across two switches is how
# they would get combined wrongly.
#
# WHAT IS TRAINABLE UNDER IT: block 3's tri_att_start, tri_att_end and pair_transition, plus the two
# (or four) injection projections. NOT blocks 0-2 and NOT block 3's tri_mul_out / tri_mul_in.
#
# THE PLACEMENT IS LOAD-BEARING. The injection goes AFTER tri_mul_in and BEFORE tri_att_start.
# Autograd starts building at the first op whose parameters or inputs require grad, so injecting at
# the block's INPUT instead -- which is where --pose-perblock injects -- makes the z entering
# tri_mul_out require grad, brings both frozen tri-muls back into the backward, and evaporates the
# saving SILENTLY. run_logging/test_pose_inblock_graph.py asserts the boundary with a negative
# control that reproduces exactly that failure and checks the test can see it.
#
# BOTH INJECTIONS ARE ZERO-INIT AND VERIFIED EXACTLY ZERO, so step 0 is bit-identical to the
# un-injected head. This is still a DIFFERENT ARM -- a capacity change on top of a freeze -- and
# needs its own control. UNTESTED ON A GPU.
INBLOCK="${INBLOCK:-0}"
# INBLOCK_PAE=1 adds the second injection, after pae_ln in the readout. Separate because it adds an
# UNNORMALISED quantity to a normalised one and nothing downstream rescales it.
INBLOCK_PAE="${INBLOCK_PAE:-1}"
# INBLOCK_READOUT=1 also leaves linear_no_bias_pae and pae_ln TRAINABLE. They sit ABOVE the stack,
# so unfreezing them adds no backward depth at all -- their gradient needs only their input and
# their output gradient.
#
# ON BY DEFAULT, changed from 0 after review. The second injection is added AFTER pae_ln, so it
# hands an unnormalised quantity to linear_no_bias_pae; with the readout frozen, nothing
# downstream can adapt its scale and the injection's whole influence is fixed by its
# initialisation and the optimiser. The cost is 16,896 parameters -- linear_no_bias_pae's 64x256
# weight plus pae_ln's two 256-vectors -- which is a small price for letting the layer that
# consumes the injection learn to consume it. Set INBLOCK_READOUT=0 to run the frozen-readout
# ABLATION, which is a legitimate arm -- it just should not be the default reading of the
# two-injection design.
INBLOCK_READOUT="${INBLOCK_READOUT:-1}"
if [ "$INBLOCK" = "1" ]; then
  # REFUSED FIRST, BEFORE ANYTHING IS APPENDED. Both switches set --train-only, and argparse lets
  # the second win silently -- so the run would freeze whatever the loser's pattern said and the
  # banner would describe the winner's.
  if [ "$FREEZE_BLOCKS" = "1" ]; then
    say "REFUSING: INBLOCK=1 and FREEZE_BLOCKS=1 both set --train-only. The second occurrence wins"
    say "  silently, so the arm would not be the one either switch describes. Pick one."
    exit 1
  fi
  # tri_att_end IS DELIBERATELY ABSENT. It is in the shipped checkpoint as a constant bit pattern
  # that emits exactly zero, so it is a fixed point of gradient descent: zero output, zero
  # gradient, it can never move. Measured on GPU (run vhhpyf, --grad-probe over 150 poses): all
  # eight of its tensors accumulated exactly 0.000e+00 in BOTH the PAE and the rank terms, while
  # every other trainable tensor was healthy. Training it is 330,240 parameters of optimiser
  # state and all-reduce traffic buying nothing, and it made the arm's advertised capacity
  # (1,484,800) overstate the real one (1,154,560) by 22%.
  INBLOCK_PAT='pairformer_stack\.blocks\.3\.tri_att_start\.,pairformer_stack\.blocks\.3\.pair_transition\.,^pose_inblock_'
  if [ "$INBLOCK_READOUT" = "1" ]; then
    INBLOCK_PAT="$INBLOCK_PAT"',^linear_no_bias_pae\.,^pae_ln\.'
  fi
  ARGS+=(--pose-inblock --train-only "$INBLOCK_PAT")
  [ "$INBLOCK_PAE" = "1" ] && ARGS+=(--pose-inblock-pae)
  say "INBLOCK=1: the pose is injected inside the LAST pairformer block, after tri_mul_in and"
  say "  before tri_att_start$([ "$INBLOCK_PAE" = "1" ] && echo ", and after pae_ln in the readout")."
  say "  --train-only '$INBLOCK_PAT'"
  say "  Blocks 0-2 AND block 3's two tri-muls are FROZEN. Check the trainer's own '--train-only"
  say "  ...: N of 248 head tensor(s)' line and the 'POSE IN-BLOCK:' line before walking away."
  say "  This is a DIFFERENT ARM and needs its own control. Expected 1.5-1.7x and UNVALIDATED."
fi
if [ "${LEAN_SINGLE:-0}" = "1" ]; then
  ARGS+=(--lean-single)
  say "LEAN_SINGLE=1: the pairformer's single track is NOT COMPUTED and 10.55M of 19.75M head"
  say "  parameters leave the optimiser as provably gradient-free. Bit-identical on the pae path;"
  say "  expect ~1-2% of the epoch, not a large win. Check the 'LEAN SINGLE:' startup line."
fi
# TRUNK_CACHE_GB HOLDS A STEP'S TRUNKS AS HOST ARRAYS, so pass A and pass B read each unit ONCE
# instead of twice, and --prefetch-workers threads warm them while the GPU is busy. Estimated
# 10-20 min of a 232 min epoch. 0 is off; 4 covers a typical step (a deep step of the largest
# remaining structures is ~8.5 GB, and the cache refuses rather than evicts when full, so the cap
# bounds RSS and nothing else). It changes NO number the run produces. UNTESTED ON A GPU.
#
# DO NOT SET IT AND TRUNK_MEMMAP=1 EXPECTING BOTH SAVINGS: a mapping is lazy, so a warmed memmap
# does no read on the worker and hands the bytes back to the main thread's device copy. The warmer
# always reads into a heap buffer for that reason; --trunk-memmap then only affects the inline
# misses. Measure one at a time.
#
# GATED NUMERICALLY, NOT BY STRING. `[ "$TRUNK_CACHE_GB" != "0" ]` passed for 0.0, 00 and 0.0000:
# the flag went on, the banner and the run NOTE both claimed a step's trunks were being held, and
# the trainer computed int(0.0 * 2**30) == 0 and did nothing. A switch that lies in the run record
# is worse than a switch that is off.
TRUNK_CACHE_GB="${TRUNK_CACHE_GB:-0}"
if $PY -I -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$TRUNK_CACHE_GB" 2>/dev/null; then
  ARGS+=(--trunk-cache-gb "$TRUNK_CACHE_GB")
  say "TRUNK_CACHE_GB=$TRUNK_CACHE_GB GiB/rank: a step's trunks are held as host arrays and warmed"
  say "  on the --prefetch-workers threads. Watch the 'trunk cache ... hit' line: 50% means the"
  say "  warmers contributed nothing, 100% means every unit was warmed before pass A. At 4 GiB a"
  say "  30-unit step (~7.3 GB) fits about half its units, so ~75% is the target and 'refused for"
  say "  cap' says how many were turned away."
fi
# WHAT DROPPING EVAL COSTS, stated so it is a choice and not a surprise: no held-out valfull
# Spearman, no selected DockQ, no acceptable-pose fraction at 0.23. What survives is the trainer's
# own per-epoch IN-SAMPLE lines -- WITHIN-INTERFACE SPEARMAN (predicted mean ipTM vs DockQ) and
# SCORE SPREAD against std(true iTM) -- which need no eval and must never be read as generalisation.
if [ "$NOEVAL" != "1" ]; then
  # EVAL_POOL SELECTS THE HELD-OUT SET. training/valfull83 is the WHOLE validation fold -- both
  # waves, 83 structures / 141 interfaces -- and carries its own chain_roles.csv and
  # seed_manifest.json, which the trainer prefers over the fold's 53-structure ones.
  # training/valfull (53 / 92) is what every number in paper_results.md section 3-4 was measured
  # on, so it stays available for comparison.
  ARGS+=(--eval-pool "${EVAL_POOL:-training/valfull/pool.csv}" --eval-pool-fold validation/fold
         --eval-pool-seed-base 400000 --eval-pool-budget 30 --eval-pool-every 1)
else
  say "NOEVAL=1: no --eval-pool. IN-SAMPLE diagnostics only; no held-out number from this run."
fi

# THE POPULATION IS COUNTED, NOT ASSERTED. This line used to hardcode "841 ... less the TEN
# structures", which stopped being true the moment SKIP changed and would have written a false
# record of what the run trained on -- the one thing a run note exists to prevent.
NPOP=$($PY -I - <<EOF
import csv
skip = {x.strip().lower() for x in "$SKIP".split(",") if x.strip()}
rows = [r["pdb"].strip().lower() for r in csv.DictReader(open("$MANIFEST"))]
print(f"{len([p for p in rows if p not in skip])} of {len(rows)} declared, less {len(skip)} skipped: {','.join(sorted(skip))}")
EOF
)
NOTE="$EPOCHS EPOCH(S), --rank-form rel at --alpha-rank 75 plus the PAE term, FULLY UNCROPPED, over $NPOP. 6,903 units in 230 cross-seed steps of 30. THE EXCLUSION IS NOW MEASURED, NOT PREDICTED. peak_GB = 2.02 + 2.896e-05 * tokens^2 was fitted on the FULL-BACKWARD arm (tvqobh) and does not transfer to the frozen graph: it put 2,289 tokens at 154 GB, and the frozen graph peaks at 60. Measured on one card, checkpointing OFF, expandable_segments, full pass A and pass B: 1,650 tok (4rdq) peak 54.57 GB, 1,822 (9q5v) 66.15 GB, 2,004 (7m7i) 79.64 GB -- all fitting on a 94.97 GB card -- and 2,289 (9dzf/9dzg) OOM, short by about 10 GB. So the cut is between 2,004 and 2,289 and only 9dzf/9dzg are excluded. CAVEAT ON THAT: those are SINGLE-RANK, 2-seed probes. The arm runs 3 ranks over 30-unit steps with the prefix cache live, and tztukd and kfthfe both died at step 158 rather than step 1, so the failure mode is STEP COMPOSITION rather than single-pose capacity. 2,004 tokens at 79.64 GB is 84% of the card. This is what killed tztukd and kfthfe at the same step 158 under thresholds 855 and 0 -- the threshold was never the cause. THE RANKS ARE NOW BALANCED: a step is dealt round-robin across its structures via lr_schedule.deal_round_robin instead of shuffled globally, so each rank gets 5 seeds of each of the step's 2 structures rather than a random 10 of 30. Per-step tokens^2 imbalance falls from a median of 17.2% to 0.0% (90 of 230 steps over 20% to 13, those 13 being the all-shallow steps where no assignment can balance). --ckpt-token-threshold 855 turns checkpointing OFF at or below 855 tokens; 855 was measured on a population under token dropout where it meant ~427 effective tokens, so --mem-log is ON to read the unchecked peaks rather than assume them. rel regresses on the true iTM after mean-centring, so it uses every pose of an interface regardless of seed and has no pair family or tau."

# THE RUN RECORD MUST SAY WHAT THE RUN IS. Under MEMLOG this is a memory diagnostic of a few
# dozen steps, not the arm, and the arm's note ("THE FIRST REAL RUN ON THIS DATA") would be a
# false record of it -- the kind of mislabelling that makes a runs directory unreadable later.
if [ "$MEMLOG" = "1" ]; then
  NOTE="$NOTE INSTRUMENTATION: --mem-log prints allocated/peak/reserved per pose with the token count and the pass, which is how the n^2 fit above was obtained and how the 855 threshold's unchecked peaks are being checked. It costs a print per pose. NOTE the lines carry NO RANK LABEL, so three ranks interleave in one log -- read them as a population, not a sequence."
fi
# THE SAME RULE AS MEMLOG: a frozen-blocks arm is a DIFFERENT EXPERIMENT, not this one run faster,
# and recording it under the default note would be a false record of what was trained.
if [ "$FREEZE_BLOCKS" = "1" ]; then
  NOTE="$NOTE FROZEN STACK: --train-only '$FREEZE_PAT' trains pairformer block 3 only and freezes blocks 0-2, so 4.75M of 19.75M head parameters move. THIS IS A CAPACITY CUT AND A DIFFERENT ARM -- it is NOT comparable against mqird6 or any full-head arm and needs its own held-out control. It is here for epoch time: blocks 0-2 build no autograd graph, so gpu_backward should fall from 110 min to ~28 while gpu_forward stays at 69.2 (all four blocks still run), i.e. 232 min -> ~150, 1.5-1.7x. That estimate is ANALYTICAL AND UNVALIDATED; the same method ran 25% optimistic on 2026-11-07. Per-block injection in blocks 0-2 (--pose-perblock, --frame-direction) cannot train under it."
fi
if [ "${TRUNK_MEMMAP:-0}" = "1" ]; then
  NOTE="$NOTE --trunk-memmap maps z_trunk out of the npz in place instead of reading it into a heap buffer; gated bit-identical on CPU, so it changes epoch time and nothing else."
fi
if $PY -I -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$TRUNK_CACHE_GB" 2>/dev/null; then
  NOTE="$NOTE --trunk-cache-gb $TRUNK_CACHE_GB holds a step's uncropped trunks as host numpy arrays so pass A and pass B read each unit once instead of twice, warmed on the prefetch workers. It changes epoch time and nothing else."
fi
# THE SAME RULE AS MEMLOG AND FREEZE_BLOCKS: this is a different experiment, not this one run faster.
if [ "$INBLOCK" = "1" ]; then
  NOTE="$NOTE IN-BLOCK POSE INJECTION: the pose enters pairformer block 3 AFTER tri_mul_in and BEFORE tri_att_start, with both the 39-bin one-hot distance term and the raw-distance term$([ "$INBLOCK_PAE" = "1" ] && echo ", and again after pae_ln in the PAE readout"). Blocks 0-2 and block 3's tri_mul_out/tri_mul_in are FROZEN via --train-only '$INBLOCK_PAT'; trainable is block 3's tri_att_start, tri_att_end and pair_transition plus the injections$([ "$INBLOCK_READOUT" = "1" ] && echo " and the PAE readout"). THE PLACEMENT IS THE POINT: autograd starts at the first op whose parameters or inputs require grad, so injecting at the block INPUT -- where --pose-perblock injects -- would put both frozen tri-muls back into the backward and lose the saving silently; run_logging/test_pose_inblock_graph.py asserts the boundary with a negative control that reproduces exactly that failure. Both injections are ZERO-INIT and verified exactly zero, so step 0 is bit-identical to the un-injected head. THIS IS A CAPACITY CHANGE ON TOP OF A FREEZE AND A DIFFERENT ARM -- it is NOT comparable against mqird6 and needs its own held-out control. The readout injection is added AFTER the layer norm, so an unnormalised quantity is added to a normalised one and nothing downstream rescales it. UNTESTED on a GPU."
fi
# EXTRA PASSES RAW TRAINER ARGS THROUGH, word-split on purpose so EXTRA="--grad-probe --save-init
# /tmp/h.pt" works. It exists because the GPU smoke needs --save-init and --grad-probe, and the
# alternative was a second launcher that reconstructs this one's argv by hand -- which is how the
# --src-cifs mistake happened. Appended LAST, so an EXTRA that repeats a flag wins, and the banner
# prints it so the run.log records what was actually added.
if [ -n "${EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA)
  say "LR: ${LR:-0.01} (base, before --lr-steps)"
say "EXTRA: appended $EXTRA"
  NOTE="$NOTE EXTRA ARGS: $EXTRA."
fi
if [ "$DRY" = "1" ]; then
  say "DRY RUN"
  LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
    --numa-map 0:0,1:0,2:1 --note "$NOTE" --dry-run -- "${ARGS[@]}"
  say "dry run exit $?"
  exit 0
fi
say "LAUNCHING on cards $CARDS"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}"
say "launch exit $?"
