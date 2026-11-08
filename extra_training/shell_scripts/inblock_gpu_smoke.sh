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
EPOCHS_FRAC="${EPOCHS_FRAC:-0.02}"   # ~4 optimiser steps at 230 steps/epoch
FREE_MIB="${FREE_MIB:-90000}"    # the card must have at least this much free
PY=/home/ebrahim/Protenix/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
W="training/runs/inblock_smoke_$TS"
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
case ",4,5,7," in *",$CARD,"*)
  say "REFUSING: card $CARD is one of mqird6's (4,5,7). Use a card outside that set."
  exit 1;; esac
say "GATE 0 PASS: card $CARD is idle and the host has room."

# ---- the smoke pool: the first NSTRUCT structures of the validation pool ------------------------
# DERIVED IN THE SCRIPT, not hand-made, so the pool this ran on is reproducible from the script
# and the source pool alone.
POOL_SRC="${POOL_SRC:-training/valfull/pool.csv}"
POOL="$W/pool_smoke.csv"
head -1 "$POOL_SRC" > "$POOL"
awk -F, 'NR>1{print $1}' "$POOL_SRC" | awk '!seen[$0]++' | head -"$NSTRUCT" > "$W/pdbs.txt"
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

# ---- GATE 2: one optimiser step, the gradient probe, and a resume -------------------------------
say "GATE 2: ~$(awk "BEGIN{printf \"%.0f\", $EPOCHS_FRAC*230}") optimiser step(s) with --grad-probe"
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
run "$PY" -I run_logging/argv_from_dry.py --log "$W/dry_gate3.log" \
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

# ---- the summary, so the verdict is one line and not a reading exercise --------------------------
say "================ SMOKE SUMMARY ================"
if [ "$DRY" = "1" ]; then
  say "DRY RUN -- nothing executed. Re-run with DRY=0 CARD=$CARD to actually do it."
  say "All three gates are wired. Gate 3 delegates to bench_ckpt_by_size.py."
  exit 0
fi
say "Gate 0 (card idle)            PASS"
say "Gate 1 (step-zero parity)     ${G1:-skipped}"
say "Gate 2 (one step + probe)     see $W/gate2_verdict.txt -- READ IT, it is not auto-judged"
say "Gate 3 (time and VRAM)        see $W/gate3_probe.log -- also not auto-judged"
say "Everything is under $W"
say "NOTHING was merged, nothing long was launched, and mqird6 was not touched."
# GATE 2 IS NOT GIVEN A PASS/FAIL BY THIS SCRIPT. Its evidence is the grad-probe block in the
# run's own log, and reducing "both sites took finite non-zero gradient and nothing frozen did"
# to a grep would be a check that passes when the log format changes. It is printed for a human.
if [ "${G1:-}" = "FAIL" ]; then
  say "OVERALL: FAIL -- gate 1 did not hold. Do not launch a long arm."
  exit 1
fi
say "OVERALL: gate 1 held. Gate 2 needs a read, gate 3 needs running."
