#!/bin/bash
# THE PAE-ONLY CROPPED ARM. Our own 256-token crop pool, PAE loss alone, no ranking loss,
# 48 distinct structures per optimiser step, 50 epochs, held-out eval every 5 epochs UNCROPPED.
#
# WHY THIS ARM EXISTS. Every cropped attempt so far carried the relative-iTM ranking loss, and
# that loss is what made cropping fragile: it scores ipTM on a NAMED chain pair, so a crop that
# removes that chain is fatal, and the pool has to be validated against whichever join the
# trainer will use. That is exactly what killed 3pdv7i (pool validated against the static
# chain_roles join, trainer scored the per-pose OST join, they disagree on 90 of 846 structures).
#
# WITH PAE ALONE THAT WHOLE CLASS OF FAILURE NARROWS TO ONE CHECKABLE CONDITION. The PAE term is
# a per token-pair cross-entropy, self-consistent under any crop. It is MASKED to scored chain
# pairs (--pae-scored-pairs), and that mask resolves chains through chain_roles -- the SAME
# STATIC JOIN build_crop_pool.py validated training/crop_final against (phase_k_train.py:2096 vs
# build_crop_pool.py's interface_chain_pairs). Pool and consumer therefore agree by construction,
# and check_crop_pairs.py --join static asserts it below rather than trusting the argument.
#
# WHAT IS DIFFERENT FROM THE RANKING ARMS, and all of it:
#   --pae-only                 pLDDT, PDE and resolved stubbed out; ipTM derives from PAE alone
#   no --rank-loss             no ranking objective, so no pass A: one forward per pose, not two
#   no --rank-cross-seed       frees the epoch builder to shuffle over ALL structures
#   --accum 240 --nsample 5    48 units per step, 16 per rank, ~47 distinct structures
#   --epoch-cover-all          every unit once per epoch, ranks holding DISJOINT structures
#   --epochs 50
#
# --nsample 5 IS THE ONLY LEGAL VALUE. Every cached unit holds 5 poses: slice_cached_unit crops
# ATOMS, not pose rows, and phase_k_train.py:5353 refuses n_pose != nsample when world > 1. An
# earlier version of this launcher passed 3 and would have STOPPED on the first unit of epoch 1.
# Caught by adversarial review and independently by external review of 216b0630.
#
# --epoch-cover-all IS NOT OPTIONAL HERE. Without --rank-cross-seed, build_epoch's unit of work is
# `world` SEEDS OF ONE STRUCTURE -- `for g in range(len(u) // world)` -- so each of the 412
# single-seed structures in this population yields 1//3 = 0 groups and NEVER TRAINS, in any epoch,
# with nothing in the log saying so. 412 of 846. The cover-all builder deals every unit instead,
# and run_logging/test_epoch_cover_all.py asserts the coverage against the real manifest.
#
# THE BATCH. structs_per_step = accum / (world * nsample) = 240 / (3 * 5) = 16 units per rank, so
# a step is 48 units spanning 43-48 DISTINCT structures (median 47; they differ only where a deep
# structure's seeds collide by chance) -- against 2 structures under --rank-cross-seed, which
# needed many poses of ONE structure to form a ranking group.
# WHOLE STRUCTURES GO TO ONE RANK, and the ranks hold disjoint units within a step. Splitting a
# structure's poses across ranks would make all three load the same ~100 MB trunk, and measured
# per-step rank imbalance here is already ~0.5% because the 256-token crop makes almost every
# unit the same size. Nothing left to balance, three times the bytes to pay for.
#
# HELD-OUT EVAL IS UNCROPPED, every 5 epochs, so every historical number stays comparable.
#
# --expect-trainable 113 IS A FAIL-CLOSED LAUNCH GATE, not decoration. 87ja6a and 9gsbdq both
# report "113 of 258 head tensor(s), 5,845,838 of 19,786,062 parameters" under this exact
# pattern. The pattern truncated to its bare negative lookahead gives 19 tensors instead, which
# leaves the optimiser non-empty so no existing guard fires -- that launch was caught only
# because a human read the startup line. Now the trainer refuses before the first step.
#
# DRY=1 (default) prints the argv and runs every gate without touching a card.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

POOL_DIR="${POOL_DIR:-training/crop_final}"
MANIFEST="${MANIFEST:-train_val_splits/cache_train_structures_extended.csv}"
# THE SCORED-PAIR POOL. Both the PAE mask and the ranking term are restricted to these
# interfaces, so narrowing it narrows the whole objective -- which is what an overfit test on a
# subset needs. It was hardcoded, so a subset arm could only be built by editing the launcher.
SCORED_PAIRS="${SCORED_PAIRS:-training/combined/pool.csv}"
SKIP="${SKIP:-4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m}"
CARDS="${CARDS:-4,5,7}"
EPOCHS="${EPOCHS:-50}"
ACCUM="${ACCUM:-240}"
NSAMPLE="${NSAMPLE:-5}"
EVAL_POOL="${EVAL_POOL:-training/valfull83/pool.csv}"
LR="${LR:-1e-3}"
DRY="${DRY:-1}"
# RESUME AND WORKER COUNT, so a mid-run parameter change does not mean losing the epochs already
# done. --prefetch-workers is NOT in the resume guard (checked), so changing it across a resume is
# accepted rather than refused as drift -- which is correct: it changes throughput, not the
# objective, the population or the batch.
RESUME="${RESUME:-}"
# PAIR DROPOUT, unset by default so an unset variable changes nothing. AF3 Algorithm 17 applies
# p=0.25 shared dropout to all four triangle updates of every Pairformer block, and the confidence
# head calls that same Algorithm 17 with Nblock=4 -- but Protenix's own config zeroes it for this
# head (configs/configs_base.py:347 "pairformer_dropout": 0.0), so every arm in this project has
# trained 19.8M parameters with no dropout of any kind. The trainer hard-stops if the assignment
# does not take (phase_k_train.py:2009), so a silent no-op is not possible.
PAIR_DROPOUT="${PAIR_DROPOUT:-}"
# L2-SP ANCHOR, unset by default. --l2-sp prices distance from the SHIPPED head rather than from
# zero; the anchor is written to <out>/l2sp_anchor.pt on a fresh run and read back on resume.
#
# THE EXCLUDE IS NOT OPTIONAL FOR THIS ARCHITECTURE. This arm runs --frame-direction and
# --pose-perblock --pose-perblock-copy, whose tensors are ZERO-INIT by construction. L2-SP on a
# zero-init tensor is plain weight decay toward zero -- it penalises the very term being added.
# The default below is copied from egw5wq/args.json, the previous arm with this same architecture;
# verified 2026-11-11 to match 10 of the 102 trainable tensors in rdi25k/resume_ep11.pt
# (frame_dir_proj.weight, pose_perblock_linear_no_bias_d{,_wo_onehot}.0-2,
# pose_perblock_frame_dir.0-2). An exclude that matched nothing would be the silent-no-op failure
# the flag's own help warns about, so the trainer names the excluded tensors at startup.
#
# SCALE, measured and recorded in paper_results.md section 17: the L2-SP gradient is
# 2*lambda*(W-W0), and against a median task grad_norm of 1.080 with ||W-W0|| = 43.2, lambda 1e-5
# is 0.08% of the task gradient and 1e-4 is 0.80% -- both effectively OFF, not "a little L2". The
# usable range starts around 1e-3 (7.99%). LAMBDA IS THE HYPERPARAMETER OF THE EXPERIMENT and is
# not derived; sweep it and read penalty-vs-loss.
L2_SP="${L2_SP:-}"
# LORA. Unset by default so nothing changes for an existing invocation.
#
# TARGETS DEFAULT TO $TRAIN_ONLY, AND THAT IS THE WHOLE POINT. apply_lora with an empty target
# list adapts every module with a 2-D weight, and phase_k_train then sets `trainable` to the
# adapter names ALONE -- so --train-only does not survive LoRA. Ten pairformer ops on this arm are
# DEAD (blocks.0 all four, blocks.1 tri_att_start/tri_att_end/pair_transition, blocks.2
# tri_att_start/tri_att_end, blocks.3 tri_att_end; measured "output exactly 0.0" in the run log),
# so an empty list spends rank-8 adapters on modules nothing reads. Reusing $TRAIN_ONLY adapts
# exactly the live set. run_logging/test_lora_targets.py is the gate and reads the pattern from
# THIS file rather than keeping its own copy.
#
# KEEP THE POSE PATHWAY DENSE. frame_dir_proj is (256, 3) and
# pose_perblock_linear_no_bias_d_wo_onehot is (256, 1), so at rank 8 apply_lora SKIPS them for
# rank-too-high and they would train not at all -- frozen at their zero init, which is the same as
# not having the feature. All ten pose tensors are 33,792 parameters, 0.6% of the head's effective
# capacity, so keeping them dense is free.
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
# LORA_TARGETS is derived from $TRAIN_ONLY and is therefore set BELOW, where TRAIN_ONLY exists.
# Setting it here expanded ${TRAIN_ONLY#*,} against an undefined variable and produced a target
# list beginning with a bare comma -- caught by an ordering assertion, not by bash, which was
# perfectly happy to carry on.
LORA_KEEP="${LORA_KEEP:-frame_dir_proj,pose_perblock_,^linear_no_bias_d}"
L2_SP_EXCLUDE="${L2_SP_EXCLUDE:-frame_dir_proj,pose_perblock_}"
# MEASURED, NOT ASSUMED. --trunk-memmap maps z_trunk instead of materialising it, so the crop
# reads the rows it keeps rather than the whole ~100 MB member. On a 173-unit slice: 1.2 min ->
# 0.5 min, cpucrop_slice 2.8m -> 0.1m, and the loss moved by 2.6e-07, i.e. the same bytes read
# lazily. --epoch-balance deals each step so the ranks get similar total uncropped size; measured
# per-step spread 16.93% -> 0.55%. Both default OFF so this launcher reproduces its own history.
TRUNK_MEMMAP="${TRUNK_MEMMAP:-}"
EPOCH_BALANCE="${EPOCH_BALANCE:-}"
# NAMED FIELDS ONLY, NEVER --resume-allow-drift. The guard's own text warns that the blanket flag
# "accepts everything at once, including what you did not intend"; naming the field means a second
# divergence still stops the run.
RESUME_OVERRIDE="${RESUME_OVERRIDE:-}"
RESUME_NOTE="${RESUME_NOTE:-}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-4}"

# A HOLD FILE DISARMS THIS LAUNCHER WITHOUT KILLING ANYTHING.
# chain_pae_crop_arm.sh polls cards_busy in the background and invokes this script BY PATH when
# the cards free, so the only safe way to stop a queued launch -- without pattern-killing a
# process, which has self-killed this session twice -- is to make the launcher itself refuse.
# The chain then reports a failed launch and exits, the cards stay as they are, and nothing on a
# GPU is touched. Remove the file to re-arm.
HOLD="${HOLD:-training/runs/chain_pae_crop/HOLD}"
if [ -f "$HOLD" ]; then
  say "REFUSING: $HOLD exists -- this arm is disarmed. Reason:"
  sed 's/^/    /' "$HOLD"
  exit 9
fi

say "PAE-only cropped arm: pool=$POOL_DIR cards=$CARDS epochs=$EPOCHS accum=$ACCUM nsample=$NSAMPLE"
say "  prefetch-workers=$PREFETCH_WORKERS  memmap=${TRUNK_MEMMAP:-off}  balance=${EPOCH_BALANCE:-off}  resume=${RESUME:-none}  pair-dropout=${PAIR_DROPOUT:-off}  l2-sp=${L2_SP:-off}  lora=${LORA_RANK:-off}  DRY=$DRY"

# THE CORE BUDGET IS 64 ACROSS EVERYTHING WE RUN, at the user's direction, and prefetch workers
# are the one knob here that multiplies by world size. 16 x 3 = 48 crop threads; measured core
# usage at 4 workers was 5.5, so this is a throughput change inside the budget, not a breach of
# it. Asserted rather than assumed because the arithmetic is easy to get wrong by a factor of 3.
_THREADS=$(( PREFETCH_WORKERS * 3 ))
[ "$_THREADS" -gt 60 ] && {
  say "REFUSING: $PREFETCH_WORKERS workers x 3 ranks = $_THREADS crop threads, over the 64-core budget"
  exit 8; }
say "  core budget: $_THREADS crop thread(s) across 3 ranks, ceiling 64"

[ -d "$POOL_DIR" ] || { say "REFUSING: $POOL_DIR is not a directory"; exit 2; }

# POPULATION, AS A SET NOT A COUNT.
$PY -I - "$POOL_DIR" "$MANIFEST" "$SKIP" "${SUBSET_OK:-}" <<'PYG' || exit 3
import csv, os, sys
pool_dir, manifest, skip_s, subset_ok = (sys.argv[1:5] + [""])[:4]
skip = set(x for x in skip_s.split(",") if x)
man = [r["pdb"] for r in csv.DictReader(open(manifest))]
want = set(p for p in man if p not in skip)
pool = set(f[:-4] for f in os.listdir(pool_dir) if f.endswith(".npz"))
print(f"  manifest {len(man)} rows, SKIP removes {len(skip & set(man))} -> trainable {len(want)}")
print(f"  crop pool {len(pool)}   want == pool: {'YES' if want == pool else 'NO'}")
# COVERAGE IS THE REAL REQUIREMENT -- every structure we intend to train on must have a crop.
# Crops the manifest does not name are harmless. But equality also caught the case this gate was
# written for: a manifest silently shrunk, training on a subset while the logs say otherwise. So a
# STRICT SUBSET is allowed only with SUBSET_OK=1, which makes a deliberate subset arm possible and
# an accidental one still refuse.
missing = want - pool
if missing:
    print(f"  REFUSING: {len(missing)} structure(s) absent from the pool: {sorted(missing)[:6]}")
    sys.exit(1)
if want != pool:
    extra = len(pool - want)
    if not subset_ok:
        print(f"  REFUSING: the manifest is a STRICT SUBSET of the crop pool -- {len(want)} of "
              f"{len(pool)} structures, {extra} crop(s) unused. If that is deliberate, pass "
              f"SUBSET_OK=1 and the run will say so in its banner; if it is not, the manifest is "
              f"not the one you think it is.")
        sys.exit(1)
    print(f"  SUBSET_OK: training on {len(want)} of {len(pool)} crop-pool structures by request "
          f"({extra} crop(s) unused). THIS ARM IS NOT COMPARABLE with a full-population arm.")
sys.exit(0)
PYG

say "CPU gate: wiring"
$PY -I run_logging/test_crop_pool_wiring.py || { say "REFUSING: wiring gates failed"; exit 4; }

# THE GATE 3pdv7i DID NOT HAVE. --join static because that is the join the PAE mask uses; the
# OST join is the ranking loss's and is NOT what this arm asks of the crop.
# THE POSE COUNT, FROM THE ACTUAL CACHE. phase_k_train.py:5353 refuses n_pose != --nsample when
# world > 1, and slice_cached_unit crops ATOMS, not pose rows -- so --nsample must equal what the
# trunks hold or the run STOPS on the first unit. An earlier version passed 3 against caches of 5.
say "CPU gate: cached units hold exactly --nsample $NSAMPLE pose(s)"
$PY -I - "$NSAMPLE" <<'PYG' || exit 7
import glob, random, sys
import numpy as np
want = int(sys.argv[1])
fs = glob.glob("training/fold/trunks/*/free_*.npz")
if not fs:
    print("  REFUSING: no trunks found under training/fold/trunks"); sys.exit(1)
random.seed(0)
bad, n = {}, 0
for f in random.sample(fs, min(40, len(fs))):
    z = np.load(f)
    k = [x for x in z.files if "ci_coordinate" in x]
    if not k:
        continue
    p = int(np.asarray(z[k[0]]).shape[0]); n += 1
    if p != want:
        bad.setdefault(p, []).append(f)
print(f"  sampled {n} trunk(s) of {len(fs)}: pose counts "
      f"{ {want: n - sum(len(v) for v in bad.values()), **{k: len(v) for k, v in bad.items()} } }")
if bad:
    print(f"  REFUSING: --nsample {want} but trunks hold {sorted(bad)}; the run would STOP at "
          f"unit 0 (phase_k_train.py:5353). e.g. {list(bad.values())[0][0]}")
sys.exit(1 if bad else 0)
PYG

say "CPU gate: the epoch plan covers every structure, singletons included"
$PY -I run_logging/test_epoch_cover_all.py || { say "REFUSING: epoch coverage gate failed"; exit 6; }

say "CPU gate: every crop keeps a scorable cross-chain pair under the STATIC join"
CROP_THREADS="${CROP_THREADS:-3}" $PY -I run_logging/check_crop_pairs.py \
    --pool "$POOL_DIR" --join static --roles training/fold/chain_roles_all.csv \
    --fold training/fold --scored-pairs training/combined/pool.csv \
    || { say "REFUSING: a selected crop leaves a scored pair unscorable"; exit 5; }

# ONLY WHEN LoRA IS ON, and it must run BEFORE the launch rather than be a file nobody calls. The
# first version of this gate was written, passed by hand, and never wired in -- so the arm it was
# written to protect launched without it, and the review that caught that also found 35 of the
# gate's 50 module names were invented and could never have matched anything.
if [ -n "$LORA_RANK" ]; then
    say "CPU gate: adapters avoid dead ops and dead readouts; all 12 pose tensors stay dense"
    LORA_RANK="$LORA_RANK" $PY -I run_logging/test_lora_targets.py \
        || { say "REFUSING: --lora-targets/--lora-keep-trainable do not place the adapters as intended"; exit 5; }
fi

# COPIED FROM 87ja6a's args.json, NOT RETYPED. The bare negative lookahead alone gives 19 tensors
# and 0.3M parameters; the comma-separated list re-includes the pairformer blocks. Launched
# truncated once -- 19 tensors against 102 -- and stopped before the first optimiser step.
TRAIN_ONLY='^(?!pairformer_stack\.),blocks\.0\.pair_transition\.,blocks\.1\.tri_mul_out\.,blocks\.1\.tri_mul_in\.,blocks\.2\.tri_mul_out\.,blocks\.2\.tri_mul_in\.,blocks\.2\.pair_transition\.,blocks\.3\.tri_mul_out\.,blocks\.3\.tri_mul_in\.,blocks\.3\.tri_att_start\.,blocks\.3\.pair_transition\.'

# THE POSE PATHWAY IS EXCLUDED FROM THE ADAPTERS AND KEPT DENSE, all twelve tensors of it. The
# first attempt reused $TRAIN_ONLY unchanged and got this wrong in two ways that an adversarial
# review measured on the live arm after one epoch:
#
#   1. $TRAIN_ONLY's `^(?!pairformer_stack\.)` admits every non-pairformer module, so the three
#      (256,39) pose_perblock distance terms were ALSO adapted. With their base kept dense they
#      trained dense + rank-8, and the two deltas were ~56% aligned and added: measured ||dW||
#      5.77/6.16/6.85 against 2.82 for the adapter-only depth-0 copy. A ~1.65x effective learning
#      rate, and a 2.1-2.4x asymmetry across copies that --pose-perblock-copy starts bit-identical
#      -- confounding the per-depth comparison with lr rather than capacity.
#   2. The keep pattern `frame_dir_proj,pose_perblock_` matched 10 of the 12 and MISSED BOTH
#      DEPTH-0 TERMS. linear_no_bias_d_wo_onehot is (256,1), so rank 8 exceeds its dimension,
#      apply_lora skips it, and it was frozen at its zero init -- in an arm whose premise is that
#      pose information may train, and after training in every previous arm of this architecture.
#
# So: exclude pose from --lora-targets entirely and name all twelve in --lora-keep-trainable.
# Uniformly dense, no adapters on pose, no per-depth asymmetry.
#
# THE DEAD READOUTS ARE EXCLUDED TOO. lean_readouts.DEAD_UNDER_PAE_ONLY names linear_no_bias_pde
# and friends and the run log says "pde, pLDDT and resolved are NOT COMPUTED", yet
# linear_no_bias_pde was adapted and its lora_B measured EXACTLY 0.0 after a full epoch -- it
# never received a gradient. --lean-single cannot strip it: it keys on names ending `.weight` and
# an adapter is `.lora_A`/`.lora_B`, so the "EFFECTIVE capacity" line was overstated by them.
LORA_EXCLUDE='^(?!pairformer_stack\.|frame_dir_proj|pose_perblock_|linear_no_bias_d|linear_no_bias_pde|plddt_|resolved_|pde_ln|input_strunk_ln)'
# DERIVED, never retyped: everything after $TRAIN_ONLY's first comma is its live-pairformer branch
# list, so the adapted set and the trainable set cannot drift apart.
LORA_TARGETS="${LORA_TARGETS:-${LORA_EXCLUDE},${TRAIN_ONLY#*,}}"
# ---- RANKING LOSS, OFF UNLESS ALPHA_RANK IS SET ----------------------------------------------
#
# rel IS THE ONLY FORM. The six pairwise ones were deleted, so there is no --rank-form to choose:
# it is passed as rel for the record and the trainer admits nothing else.
#
# --rank-cross-seed IS NOT OPTIONAL and the trainer refuses rel without it: the loss centres on an
# interface's poses across ALL its seeds, and within one seed it would centre on 5 poses and throw
# away most of the signal. RANK_S structures x all their seeds x 5 poses is then the step, packed
# to a constant 45 units (= 225 poses) by pack_cross_seed_steps: 3 deep structures fill one step,
# and 45 single-seed structures fill one of their own, so the two depth populations pack with
# nothing left over.
#
# IT OVERRIDES --epoch-cover-all, WHICH IS IN ARGS BELOW. phase_k_train:7981 takes the cross-seed
# builder first and the cover-all branch is never reached. That is harmless here -- the cross-seed
# builder uses every seed of every structure each epoch by construction, which is what cover-all
# exists to guarantee -- but it is stated because the flag is still being passed.
# HELD-OUT EVAL CADENCE. The trigger is (ep+1) % EVAL_EVERY == 0, so 2 evaluates the EVEN epochs.
# Each pass costs ~11 min on the same cards the epoch uses, against an ~11 min epoch, so halving
# the cadence from 5 to 2 roughly doubles the run's wall clock.
EVAL_EVERY="${EVAL_EVERY:-5}"
ALPHA_RANK="${ALPHA_RANK:-}"
CROP_ALLOW="${CROP_ALLOW:-training/crop_final/ost_usable_crops.json}"
RANK_S="${RANK_S:-3}"
REL_LABELS="${REL_LABELS:-training/fold/itm_labels}"
RANK_LABELS="${RANK_LABELS:-training/fold/results.csv}"
RANK_POOL_STAT="${RANK_POOL_STAT:-mean}"
# THE CORRELATION FORM. REL_STANDARDIZE divides both sides by their own std, which makes the loss
# exactly 2*(1 - corr) and so removes the reward for shrinking the predicted spread -- the plain
# rel loss is a squared error, and a squared error is minimised by hedging toward the mean
# whenever the target is only partly predictable, which is what compressed the val ipTM spread.
# REL_SD_WEIGHT then multiplies each interface by its TRUE std, restoring the importance that
# standardising equalises away (corr(within-interface std of true iTM, DockQ range) = +0.745).
# REL_STD_FLOOR near zero keeps the invariance all the way down: at the 0.02 default the clamp
# engages and a collapsed prediction scores 1.0, which BEATS an uncorrelated ordering at 2.0.
REL_STANDARDIZE="${REL_STANDARDIZE:-}"
REL_SD_WEIGHT="${REL_SD_WEIGHT:-}"
REL_STD_FLOOR="${REL_STD_FLOOR:-}"
# REL_G_WEIGHT weights each POSE's error by its raw true iTM to the power given, so misplacing a
# pose the truth rates highly costs more than misplacing a bad one. It is a weighted mean, not a
# weighted sum, so the loss keeps its scale; 1 is the linear form, 2 the sharper one.
REL_G_WEIGHT="${REL_G_WEIGHT:-}"
# REL_G_SIGMOID is the same idea as a SOFT THRESHOLD, MID:WIDTH, e.g. 0.23:0.1. The true iTM is
# bimodal (p25 0.220, p50 0.670), so a midpoint in the valley asks "is this pose good at all"
# where the raw score spends its contrast separating two poses that are both fine.
REL_G_SIGMOID="${REL_G_SIGMOID:-}"
# OFF BY DEFAULT. The gradient split costs one extra backward per pose on a measured step, and
# the real arm does not need it: beta is estimated once by the one-epoch diagnostic
# (shell_scripts/queue_rank_gradprobe_1ep.sh, which sets K=1) and then held fixed. A 50-epoch arm
# carrying the probe would pay for 7,750 measurements of a number it is not going to act on.
GRAD_PROBE_EVERY="${GRAD_PROBE_EVERY:-0}"
RANK_ARGS=()
if [ -n "$ALPHA_RANK" ]; then
  [ -d "$REL_LABELS" ] || { say "REFUSING: REL_LABELS $REL_LABELS is not a directory"; exit 4; }
  [ -f "$REL_LABELS/manifest.json" ] || {
      say "REFUSING: $REL_LABELS/manifest.json is missing, so the label set cannot be identified"
      exit 4; }
  # THE CROP ALLOWLIST IS PART OF THE RANKING ARM, not of the PAE one. The PAE term tolerates a
  # crop whose scored pair is absent (it counts the pose as unmasked); rel's pass A refuses,
  # which is what killed eqd2jx at step 117 of epoch 1 on 9o4o. 5 structures, 192 of 92,636
  # crops withheld.
  [ -f "$CROP_ALLOW" ] || { say "REFUSING: CROP_ALLOW $CROP_ALLOW is missing; build it with the"
                            say "  OST crop audit before launching a ranking arm"; exit 4; }
  RANK_ARGS=(--crop-allow "$CROP_ALLOW"
             --rank-loss --rank-form rel --alpha-rank "$ALPHA_RANK"
             --rank-cross-seed --rank-structs-per-step "$RANK_S"
             --rel-labels "$REL_LABELS" --rank-labels "$RANK_LABELS"
             --rank-pool "$RANK_POOL_STAT")
  [ -n "$REL_STANDARDIZE" ] && RANK_ARGS+=(--rel-standardize)
  [ -n "$REL_SD_WEIGHT" ] && RANK_ARGS+=(--rel-sd-weight)
  [ -n "$REL_STD_FLOOR" ] && RANK_ARGS+=(--rel-std-floor "$REL_STD_FLOOR")
  [ -n "$REL_G_WEIGHT" ] && RANK_ARGS+=(--rel-g-weight "$REL_G_WEIGHT")
  [ -n "$REL_G_SIGMOID" ] && RANK_ARGS+=(--rel-g-sigmoid "$REL_G_SIGMOID")
  # THE FLAG AND ITS VALUE TOGETHER, OR NEITHER. The first version of this line expanded to a
  # BARE NUMBER with the flag name dropped, so K=1 would have appended "1" as a positional
  # argument and argparse would have refused the whole launch. Appended conditionally instead,
  # which is the only form where "off" means the flag is absent rather than present with 0.
  if [ "${GRAD_PROBE_EVERY:-0}" != "0" ]; then
    RANK_ARGS+=(--grad-norm-every "$GRAD_PROBE_EVERY")
  fi
fi

ARGS=(--crop-pool "$POOL_DIR"
      --minimal-items --epoch-cover-all ${EPOCH_BALANCE:+--epoch-balance}
      --pae-only --alpha-pae 1.0 --alpha-confidence 1.0
      --accum "$ACCUM" --nsample "$NSAMPLE"
      --skip-dead-ops --pose-perblock --pose-perblock-copy --frame-direction --lean-single
      --train-eval-pool "$EVAL_POOL" --train-eval-fold validation/fold
      --train-eval-every "$EVAL_EVERY" --train-eval-budget 30 --train-eval-seed-base 400000
      --epochs "$EPOCHS" --lr "$LR"
      "${RANK_ARGS[@]+${RANK_ARGS[@]}}")

say "argv the trainer would receive:"
printf '    %s\n' "${ARGS[@]}"
if [ -n "$ALPHA_RANK" ]; then
  say "RANKING LOSS ON: rel at alpha $ALPHA_RANK, cross-seed, $RANK_S structure(s)/step,"
  say "  labels $REL_LABELS, ipTM pooled by $RANK_POOL_STAT, gradient split logged every"
  say "  $GRAD_PROBE_EVERY step(s). PAE stays on at --alpha-pae 1.0, so this is PAE + rel."
else
  say "NOT passed: --rank-loss, --rank-form, --rank-cross-seed, --rel-labels (set ALPHA_RANK to"
  say "  turn the ranking term on)"
fi
say "NOT passed, deliberately: --crop-margin, --crop-tokens, --eval-pool-crop-tokens"
say "  (eval stays UNCROPPED)"

if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing launched. Re-run with DRY=0 to go."
  exit 0
fi

B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "REFUSING: cards_busy reports '$B'"; exit 3; }
for c in $(echo "$CARDS" | tr ',' ' '); do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c" 2>/dev/null)
  case "$F" in ''|*[!0-9]*) say "REFUSING: GPU $c unreadable"; exit 3;; esac
  [ "$F" -lt 80000 ] && { say "REFUSING: GPU $c has only ${F} MiB free"; exit 3; }
done
M=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
[ "${M:-0}" -lt 250 ] && { say "REFUSING: MemAvailable ${M} GB is under the 250 GB floor"; exit 3; }
say "gates clear: cards $CARDS free, MemAvailable ${M} GB"

NOTE="PAE-ONLY CROPPED ARM. Our own 256-token crop pool (training/crop_final), PAE loss alone \
with pLDDT/PDE/resolved stubbed out, NO ranking loss and therefore no pass A -- one forward per \
pose instead of two. The epoch builder is free of --rank-cross-seed, so a step draws 48 DISTINCT \
structures (accum 144 / (world 3 x nsample 3) = 16 structures per rank x 3 poses) against 2 \
structures under the ranking arms. Whole structures go to one rank: splitting a structure's \
poses across ranks would make all three load the same ~100 MB trunk, and measured rank imbalance \
at this batch size is already ~0.5% because the 256-token crop makes nearly every unit the same \
size. The PAE term is masked to scored chain pairs via chain_roles -- the SAME static join the \
crop pool was validated against -- so pool and consumer agree by construction, and \
check_crop_pairs.py --join static asserts it before launch. This is the join question that \
killed 3pdv7i, where the pool used static and the RANKING loss used the per-pose OST join. \
Held-out eval every 5 epochs on valfull83, UNCROPPED, so every historical number stays \
comparable. 50 epochs."

say "LAUNCH"
LAYERNORM_TYPE=openfold $PY -I run_logging/launch.py --cards "$CARDS" \
  --numa-map 0:0,1:0,2:1 --note "$NOTE" -- "${ARGS[@]}" \
  --alignment-preflight training/alignment_preflight_sharded.json \
  --manifest-selection "$MANIFEST" --manifest-skip "$SKIP" \
  --fold training/fold --indices training/prep/all857_indices.csv \
  --pae-scored-pairs "$SCORED_PAIRS" \
  --pae-scored-roles training/fold/chain_roles_all.csv \
  --lean-readouts --train-only "$TRAIN_ONLY" --expect-trainable 113 \
  --warmup-steps 20 --seed 0 --clip-norm 25.0 \
  --blocks-per-ckpt 1 --ckpt-token-threshold 1250 --mem-log \
  --prefetch-units 2 --prefetch-workers "$PREFETCH_WORKERS" \
  ${TRUNK_MEMMAP:+--trunk-memmap} \
  ${PAIR_DROPOUT:+--pair-dropout "$PAIR_DROPOUT"} \
  ${L2_SP:+--l2-sp "$L2_SP" --l2-sp-exclude "$L2_SP_EXCLUDE"} \
  ${LORA_RANK:+--lora-rank "$LORA_RANK" --lora-targets "$LORA_TARGETS" --lora-keep-trainable "$LORA_KEEP"} \
  ${LORA_RANK:+${LORA_ALPHA:+--lora-alpha "$LORA_ALPHA"}} \
  ${RESUME:+--resume "$RESUME"} \
  ${RESUME_OVERRIDE:+--resume-override "$RESUME_OVERRIDE" --resume-note "$RESUME_NOTE"} \
  --dist-backend gloo --dist-timeout-min 240 \
  --max-rss-gb 80 --preload-min-free-gb 250 --save-every 250 \
  --require-replay training/extended_mixed_replay.log --allow-code-drift
say "launch.py returned $?"
