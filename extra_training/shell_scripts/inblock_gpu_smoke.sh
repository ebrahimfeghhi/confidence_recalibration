#!/usr/bin/env bash
# The bounded GPU validation Codex made a precondition for a long in-block arm.
#
# Gate 1  STEP-ZERO EVAL PARITY, through the real phase_k_eval CLI.
#         Both injections are zero-init, so an in-block head saved BEFORE any optimiser step is
#         bit-identical in function to the shipped head. Scoring it must therefore reproduce the
#         shipped head's numbers. That single comparison exercises the evaluator's new install
#         path, its exact head-key-set guard, the block index, and the zero-init claim -- and it
#         does it on the production CLI rather than in a test harness, which was the stated
#         limitation of run_logging/test_pose_inblock_integration.py.
#
#         THE NOISE FLOOR IS MEASURED, NOT ASSUMED. The head runs under autocast and trunk
#         tensors are stored bf16, so two runs of the SAME head need not agree bit for bit. So
#         the shipped head is scored TWICE first; whatever those two differ by is the floor, and
#         the in-block head has to land inside it. Asserting exact equality would be a test that
#         fails for reasons that have nothing to do with the injection.
#
# Gate 2  ONE OPTIMISER STEP at --blocks-per-ckpt 1, with --grad-probe, then resume.
#         Confirms on real geometry what the CPU test confirms on synthetic: both injection sites
#         take finite non-zero gradient, nothing frozen takes any, and the bundle round-trips.
#
# Gate 3  PER-SIZE ELAPSED TIME AND PEAK MEMORY, including one previously excluded structure.
#         Delegated to run_logging/bench_ckpt_by_size.py, which already does exactly this and
#         takes the arm's own argv so it cannot drift from the arm.
#
# ONE CARD. mqird6 holds 4, 5 and 7, and the standing rule is never more than three GPUs, so this
# runs on a card OUTSIDE that set and refuses if the one it is given is not genuinely idle.
#
#   DRY=1 CARD=0 ./shell_scripts/inblock_gpu_smoke.sh     # prints every command, runs nothing
#   DRY=0 CARD=0 ./shell_scripts/inblock_gpu_smoke.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CARD="${CARD:-0}"
DRY="${DRY:-1}"
NSTRUCT="${NSTRUCT:-2}"          # structures in the smoke pool
BUDGET="${BUDGET:-3}"            # seeds per interface
# WHOLE STEPS ONLY. EPOCHS=0.02 gave 138 units against cross-seed quotas summing to 150, and the
# trainer refused: "the step boundary would drift from the quota and the ranks would step at
# different times". That guard is right -- a partial step is not a smaller version of a step, it is
# a different objective. The arm's banner says "6,903 units in 230 cross-seed steps of 30", so a
# whole number of steps is STEPS/230 of an epoch. If the population ever changes, 230 changes with
# it and the same guard refuses again rather than drifting quietly.
# AND HALF A UNIT OF MARGIN, because STEPS/STEPS_PER_EPOCH is exact in arithmetic and not in
# floating point: 5/230 x 6903 came out as 149 units against a 150-unit quota and was refused for
# being one short. Expressing it in UNITS and adding 0.5 puts the value strictly inside the
# interval that floors to the number of units wanted, rather than exactly on its boundary.
STEPS="${STEPS:-5}"
UNITS_PER_STEP="${UNITS_PER_STEP:-30}"
UNITS_PER_EPOCH="${UNITS_PER_EPOCH:-6903}"
EPOCHS_FRAC=$(awk "BEGIN{printf \"%.10f\", ($STEPS*$UNITS_PER_STEP + 0.5)/$UNITS_PER_EPOCH}")
FREE_MIB="${FREE_MIB:-90000}"    # the card must have at least this much free
# WHICH GATES THIS INVOCATION RUNS. The three are independent, so with three cards free they go in
# parallel and the wall clock is the longest gate rather than their sum. GATES=123 is everything
# on one card, which is what a single-card run wants.
GATES="${GATES:-123}"
gate_on() { case "$GATES" in *"$1"*) return 0;; *) return 1;; esac; }
PY=/home/ebrahim/Protenix/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
W="training/runs/inblock_smoke_${TS}_g${GATES}"
say() { echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
run() { if [ "$DRY" = "1" ]; then echo "    WOULD RUN: $*"; else "$@"; fi; }

say "in-block GPU smoke, card $CARD, DRY=$DRY, work dir $W"
mkdir -p "$W"

# ---- GATE 0: the card, and the box -------------------------------------------------------------
# NOT cards_busy.py ALONE. That answers "is any card busy", which is always yes here -- mqird6
# holds three. What matters is whether THIS card is idle, which is a per-index question.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2>/dev/null)
MEMAV=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
say "GPU $CARD: ${USED} MiB used, ${FREE} MiB free. Host MemAvailable ${MEMAV} GB."
if [ "${FREE:-0}" -lt "$FREE_MIB" ]; then
  say "REFUSING: GPU $CARD has only ${FREE} MiB free, under FREE_MIB=$FREE_MIB."
  say "  Another tenant is on it. Pick an idle card or wait."
  exit 1
fi
if [ "${MEMAV:-0}" -lt 250 ]; then
  say "REFUSING: MemAvailable ${MEMAV} GB is under the 250 GB floor other users need."
  exit 1
fi
# NO HARDCODED CARD LIST. This used to refuse 4, 5 and 7 outright because the live arm held them
# -- and then the arm was stopped, the cards became the right ones to use, and the guard refused
# the very launch it was written to protect. A list of card numbers is a fact about one moment.
#
# The free-memory check above is the real guard and always was: a live training rank holds ~75 GB,
# so a card with 90 GB+ free demonstrably has no arm on it. That tests the property we care about
# instead of a name we have to remember to update.
if [ -n "${RESERVED_CARDS:-}" ]; then
  case ",${RESERVED_CARDS}," in *",$CARD,"*)
    say "REFUSING: card $CARD is in RESERVED_CARDS=$RESERVED_CARDS."
    exit 1;; esac
fi
say "GATE 0 PASS: card $CARD is idle and the host has room."

if gate_on 1; then
# ---- the smoke pool: the first NSTRUCT structures of the validation pool ------------------------
# DERIVED IN THE SCRIPT, not hand-made, so the pool this ran on is reproducible from the script
# and the source pool alone.
POOL_SRC="${POOL_SRC:-training/valfull/pool.csv}"
FOLD_DIR="${FOLD_DIR:-validation/fold}"
POOL="$W/pool_smoke.csv"
head -1 "$POOL_SRC" > "$POOL"
# CHOSEN BY WHAT IS ACTUALLY FOLDED, not by position in the pool file. The first attempt took the
# first N structures and drew 29tj, whose seeds are SPARSE -- free_400000, 400002, 400004... with
# no 400001 -- so the evaluator refused: "1 of 6 intended unit(s) have no trunk", which is the
# right refusal, because a metric labelled 'budget 3' that silently covered 2 seeds would be a
# number that does not mean what its label says. A structure only qualifies if every seed from
# --seed-base to --seed-base+BUDGET-1 exists as a trunk.
: > "$W/pdbs.txt"
for _p in $(awk -F, 'NR>1{print $1}' "$POOL_SRC" | awk '!seen[$0]++'); do
  [ "$(wc -l < "$W/pdbs.txt")" -ge "$NSTRUCT" ] && break
  _ok=1
  for _i in $(seq 0 $((BUDGET - 1))); do
    [ -f "$FOLD_DIR/trunks/$_p/free_$((400000 + _i)).npz" ] || { _ok=0; break; }
  done
  [ "$_ok" = "1" ] && echo "$_p" >> "$W/pdbs.txt"
done
if [ "$(wc -l < "$W/pdbs.txt")" -lt "$NSTRUCT" ]; then
  say "REFUSING: only $(wc -l < "$W/pdbs.txt") structure(s) in $POOL_SRC have all $BUDGET seeds"
  say "  folded under $FOLD_DIR/trunks. Lower BUDGET or NSTRUCT."
  exit 1
fi
awk -F, 'NR==FNR{k[$1];next} FNR==1{next} ($1 in k)' "$W/pdbs.txt" "$POOL_SRC" >> "$POOL"
say "smoke pool: $(( $(wc -l < "$POOL") - 1 )) interface row(s) over $NSTRUCT structure(s): $(paste -sd, "$W/pdbs.txt")"

EVAL_COMMON=(--fold validation/fold --results validation/fold/results.csv
             --pool "$POOL" --roles validation/fold/chain_roles.csv
             --seed-base 400000 --budget "$BUDGET" --pair-iptm-only
             --pool-mean --select-by mean)

# ---- GATE 1a: the head before any step ---------------------------------------------------------
HEAD="$W/head_init.pt"
say "GATE 1a: writing the in-block head BEFORE any optimiser step"
run env CARDS="$CARD" DRY=0 EPOCHS=1 NOEVAL=1 INBLOCK=1 LEAN_SINGLE=1 \
    TRUNK_CACHE_GB=0 TRUNK_MEMMAP=0 \
    EXTRA="--save-init $(pwd)/$HEAD" \
    ./shell_scripts/launch_default_arm.sh 2>&1 | tee "$W/save_init.log" | tail -20

# ---- GATE 1b: the shipped head, twice, to measure the floor -------------------------------------
for i in 1 2; do
  say "GATE 1b: shipped head, run $i of 2 (the noise floor)"
  run env LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="$CARD" \
      "$PY" -I phases/phase_k_eval.py "${EVAL_COMMON[@]}" \
      --out "$W/shipped_$i.json" 2>&1 | tee "$W/shipped_$i.log" | tail -6
done

# ---- GATE 1c: the in-block head at step zero ----------------------------------------------------
say "GATE 1c: the step-zero in-block head, through the SAME CLI"
run env LAYERNORM_TYPE=openfold CUDA_VISIBLE_DEVICES="$CARD" \
    "$PY" -I phases/phase_k_eval.py "${EVAL_COMMON[@]}" \
    --ckpt "$HEAD" --pose-inblock --pose-inblock-pae --pose-inblock-block -1 \
    --out "$W/inblock_init.json" 2>&1 | tee "$W/inblock_init.log" | tail -8

say "GATE 1 verdict"
G1=skipped
if [ "$DRY" = "1" ]; then
  echo "    WOULD RUN: $PY -I run_logging/cmp_eval_json.py --floor $W/shipped_{1,2}.json --test $W/inblock_init.json"
else
  # NOT THROUGH A PIPE. `cmd | tee` reports tee's status, so piping the comparator into tee would
  # discard the one thing it exists to return. It writes its own file and is then catted.
  "$PY" -I run_logging/cmp_eval_json.py \
      --floor "$W/shipped_1.json" "$W/shipped_2.json" \
      --test "$W/inblock_init.json" > "$W/gate1_verdict.txt" 2>&1
  G1_RC=$?
  cat "$W/gate1_verdict.txt"
  if [ "$G1_RC" -eq 0 ]; then G1=PASS; else G1=FAIL; fi
  say "GATE 1 $G1"
fi

fi   # gate 1

if gate_on 2; then
# ---- GATE 2: one optimiser step, the gradient probe, and a resume -------------------------------
say "GATE 2: $STEPS optimiser step(s) (--epochs $EPOCHS_FRAC) with --grad-probe"
run env CARDS="$CARD" DRY=0 EPOCHS="$EPOCHS_FRAC" NOEVAL=1 INBLOCK=1 LEAN_SINGLE=1 \
    TRUNK_CACHE_GB=0 TRUNK_MEMMAP=0 \
    EXTRA="--grad-probe --save-every 1" \
    ./shell_scripts/launch_default_arm.sh 2>&1 | tee "$W/one_step.log" | tail -30

say "GATE 2 verdict: the gradient probe, read off the run's own log"
if [ "$DRY" != "1" ]; then
  RID=$(grep -oE "^run [a-z0-9]{6}" "$W/one_step.log" | head -1 | awk '{print $2}')
  LOG="training/runs/$RID/run.log"
  echo "  run $RID -> $LOG" | tee "$W/gate2_verdict.txt"
  grep -E "GRAD PROBE|grad probe|pose_inblock|--lean-single:|trainable|non-finite" "$LOG" \
    | head -40 | tee -a "$W/gate2_verdict.txt"
  say "and the resume: the per-epoch head and the bundle"
  ls -la "training/runs/$RID"/*.pt 2>/dev/null | tee -a "$W/gate2_verdict.txt"
fi

fi   # gate 2

if gate_on 3; then
# ---- GATE 3: per-size elapsed time and peak memory, including an oversized complex --------------
# NOT A NEW HARNESS. run_logging/bench_ckpt_by_size.py already measures per-pose peak memory and
# per-pose device time on one card, and its --arm-argv exists so a benchmark cannot drift from the
# arm it benchmarks. For a live run that argv comes from /proc; this arm is not running, so it
# comes from the launcher's own dry run via run_logging/argv_from_dry.py -- still the launcher's
# output rather than a transcription of it.
#
# THE SKIP LIST IS DELIBERATELY SHORTER HERE. The arm skips the ten structures measured not to fit
# uncropped, which is exactly the set Codex wants a number for, so this argv is generated with
# SKIP holding only the three documented skips. That is why gate 3 captures its OWN argv instead
# of reusing gates 1-2'.
say "GATE 3: capturing the in-block argv, with the oversized structures NOT skipped"
run env DRY=1 CARDS="$CARD" INBLOCK=1 LEAN_SINGLE=1 TRUNK_CACHE_GB=0 TRUNK_MEMMAP=0 \
    SKIP="9mx1,4zto,9q5m" ./shell_scripts/launch_default_arm.sh > "$W/dry_gate3.log" 2>&1
# --alignment-preflight IS STRIPPED for the benchmark. The preflight marker is keyed to the arm's
# manifest, and bench_ckpt_by_size writes its OWN manifest subset (the ascending token walk), so
# the trainer refused: "the alignment preflight was run over manifest 568d3d30..., this run is
# 6791c723... It says nothing about the structures this run will train on". Correct, and fatal to
# a benchmark whose whole job is to run a different subset. The marker still gates the real arm;
# it is only meaningless for a probe, and stripping it here is narrower than weakening the check.
run "$PY" -I run_logging/argv_from_dry.py --log "$W/dry_gate3.log" \
    --strip="--alignment-preflight" \
    --out "$W/inblock_argv_oversized.txt"

# A SMALL, A MID, A LARGE AND ONE PREVIOUSLY EXCLUDED. 7cn2 is in the arm's OVERSIZED list, so it
# is the "prior oversized complex" the acceptance report asks for -- and the one most likely to
# OOM, which is the point: the new graph is shallower in backward than the arm this was fitted on,
# so whether it now fits is an open question and not a formality.
# ONE ARM ONLY: lean readouts on, checkpointing OFF. Codex's second comment on issue #8
# supersedes its first and says not to spend GPU time on a checkpointing-ON performance arm --
# ckpt OFF is the intended fast variant and the one INBLOCK now defaults to. The ckpt-ON fallback
# is one environment variable (CKPT_THRESH=855) if this does not fit.
#
# ASCENDING TO THE FIRST OOM, not descending from the largest. Codex asked to start at the biggest
# structure and walk down, but --probe already walks UP in token order with lean readouts on and
# checkpointing off and runs until it dies, which finds the SAME boundary while never beginning
# with a near-certain crash -- and its own instruction is "avoid repeatedly crashing... recover
# cleanly after OOM". The last structures to complete before the OOM are the top successful
# high-memory cases it wants; the first to fail is the documented failure.
#
# --start-tokens 1400 skips the hundreds of units that obviously fit. The largest unit in the
# TRAINED population is 1,650 (4rdq, observed peak 81.30 GB), and the ten excluded ones run
# 1,745 (7cn2) to 2,289 (9dzf), so the interesting band is 1,400-2,289 and the walk crosses the
# old boundary rather than stopping at it.
say "GATE 3: probing UP from 1,400 tokens, lean readouts on, checkpointing OFF, until it OOMs"
say "  the old predictor put the ceiling at ~1,747 tokens, but it was fitted on the FULL-backward"
say "  arm -- with the frozen graph it does not transfer, which is the point of measuring."
run env LAYERNORM_TYPE=openfold "$PY" -I run_logging/bench_ckpt_by_size.py \
    --arm-argv "$W/inblock_argv_oversized.txt" --cards "$CARD" \
    --probe --probe-seeds 2 --start-tokens 1400 --master-port 29571 \
    --out "$W/bench" 2>&1 | tee "$W/gate3_probe.log" | tail -25

fi   # gate 3

# ---- the summary, so the verdict is one line and not a reading exercise --------------------------
say "================ SMOKE SUMMARY ================"
if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing executed. Re-run with DRY=0 CARD=$CARD to actually do it."
  say "All three gates are wired. Gate 3 delegates to bench_ckpt_by_size.py."
  exit 0
fi
say "Gate 0 (card idle)            PASS"
say "gates requested: $GATES"
gate_on 1 && say "Gate 1 (step-zero parity)     ${G1:-ran, see $W}" || say "Gate 1                        not requested"
gate_on 2 && say "Gate 2 (one step + probe)     see $W/gate2_verdict.txt -- not auto-judged" || say "Gate 2                        not requested"
gate_on 3 && say "Gate 3 (time and VRAM)        see $W/gate3_probe.log -- not auto-judged" || say "Gate 3                        not requested"
say "Everything is under $W"
say "NOTHING was merged, nothing long was launched, and mqird6 was not touched."
# GATE 2 IS NOT GIVEN A PASS/FAIL BY THIS SCRIPT. Its evidence is the grad-probe block in the
# run's own log, and reducing "both sites took finite non-zero gradient and nothing frozen did"
# to a grep would be a check that passes when the log format changes. It is printed for a human.
if [ "${G1:-}" = "FAIL" ]; then
  say "OVERALL: FAIL -- gate 1 did not hold. Do not launch a long arm."
  exit 1
fi
if gate_on 1; then say "OVERALL: gate 1 held."; else say "OVERALL: gate 1 was not part of this invocation."; fi
say "  Gates 2 and 3 are not auto-judged -- read their logs."
