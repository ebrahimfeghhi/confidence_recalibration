#!/bin/bash
# SCORE THE SHIPPED CONFIDENCE HEAD -- stock Protenix, no checkpoint, NO PATHWAY INSTALLED --
# on val and on an in-sample pool, under BOTH poolings, and print the table.
#
#   ./shell_scripts/score_shipped_head.sh                 # both sides
#   SIDES=val ./shell_scripts/score_shipped_head.sh       # val only (budget 30, seed base 400000)
#   SIDES=insample ./shell_scripts/score_shipped_head.sh  # in-sample (budget 15, seed base 500000)
#   DRY=1 ./shell_scripts/score_shipped_head.sh           # print the argv, launch nothing
#   REPORT_ONLY=1 ./shell_scripts/score_shipped_head.sh   # print the numbers already on disk
#   FORCE=1 ./shell_scripts/score_shipped_head.sh         # re-score into a NEW file, keeping the old
#
# WHY THIS EXISTS AS ITS OWN SCRIPT, AND WHAT IT COST TO LEARN. score_checkpoint.sh scores a
# TRAINED head, and its COMMON hardcodes --pose-perblock --pose-perblock-copy --frame-direction
# because every arm in this project trains with them. On 2026-11-18 the shipped head was scored
# through a copy of that script with only the `CKPT=${CKPT:?}` guard relaxed: the pass printed
# "using the SHIPPED head (no --ckpt)" and then installed all three injections on top of the
# shipped weights. --pose-perblock-copy copies block 0's pretrained injection into blocks 1-3,
# which is a DIFFERENT MODEL, so the resulting ipTM no longer reproduced the fold's stored values
# and the merge refused -- after twelve minutes on three cards. A shipped pass must therefore not
# be a trained-head pass with one guard removed: the hardcoded pathway flags are the whole
# difference, and they are not a thing to remember to delete.
#
# WHAT IS LIKE-FOR-LIKE HERE, which is the point of the pass. Every row in every eval already
# carries `stored_iptm`, the shipped score -- but it is MAX-pooled and has no mean-pooled twin,
# because mean pooling needs the per-anchor-row PAE distribution and that is only computed for the
# head under test. So "ours mean-pooled vs shipped stored_iptm" compares our MEAN against shipped's
# MAX and cannot say how much of a gain is the pooling rather than the training. --pool-mean on a
# shipped pass closes that: one pass, both poolings, same candidates.
#
# THE SELF-CHECK IS LEFT ON, AND IT IS THE REASON TO PREFER THIS SCRIPT. With no checkpoint and no
# pathway, phase_k_eval requires its recomputed ipTM to reproduce results.csv row by row (measured
# max |diff| 0.000050 over 21,150 rows on the val pass). That makes a clean shipped pass the one
# measurement in this project that verifies its own three joins while it runs.
#
# IT WILL NOT RE-SPEND GPU ON A NUMBER THAT CANNOT MOVE. The shipped head is fixed, so for a fixed
# pool, budget and seed manifest its metric is a constant: if the metric json is already there, this
# prints it and exits. FORCE=1 recomputes deliberately.
set -u
# PIPEFAIL, because the report below is piped into tee and a python assert on the way out must not
# be swallowed by tee's exit status -- the failure that once truncated ten reports to zero bytes.
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
R=/home/ebrahim/confidence_recalibration/extra_training
SIDES=${SIDES:-both}            # both | val | insample
CARDS=${CARDS:-4,5,7}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
# CHECKED FIRST, NOT LAST. This case lived at the BOTTOM of the file: SIDES=valfull ran neither
# side, printed nothing and then exited 2, which reads as "the pass produced nothing" rather than
# "you typed a side that does not exist".
case "$SIDES" in both|val|insample) ;;
  *) say "STOP: SIDES=$SIDES is not both|val|insample"; exit 2;; esac

# POSE_FLAGS IS REFUSED, NOT IGNORED. It is score_checkpoint.sh's override for a TRAINED head's
# derived pathway flags, and the one thing this script must never do is install a pathway. Silently
# dropping it would leave a caller believing a flag reached the pass; naming it is the only safe
# answer.
if [ -n "${POSE_FLAGS:-}" ]; then
  say "STOP: POSE_FLAGS is set (${POSE_FLAGS}). This script scores the SHIPPED head, which is stock"
  echo "      Protenix and carries no pose_perblock_proj, frame_dir_proj, pose_readout_* or"
  echo "      pose_inblock_* tensors -- any such flag would make this a different model reported"
  echo "      under the name 'shipped'. Use score_checkpoint.sh with a CKPT for a trained head."
  exit 7
fi
# AND NEITHER IS A CKPT. "Shipped" is defined by the absence of a checkpoint; a CKPT in the
# environment means the caller wanted the other script and would otherwise have it silently ignored.
if [ -n "${CKPT:-}" ]; then
  say "STOP: CKPT is set (${CKPT}). This script cannot score a trained head -- it passes no --ckpt"
  echo "      and installs no pathway. Run: CKPT=$CKPT TAG=<tag> ./shell_scripts/score_checkpoint.sh"
  exit 7
fi

# THE THREE OPTIMISATIONS AND BOTH POOLINGS, AND NOTHING THAT TOUCHES THE MODEL. This is COMMON
# from score_checkpoint.sh with the pathway flags REMOVED and nothing else changed:
#   --prefetch        reads the next unit's trunk on a background thread (4.7 min of a 26.1 min pass)
#   --pair-iptm-only  computes chain_pair_iptm straight from PAE; measured bit-identical
#   --pool-mean       adds the iptm_mean column from the SAME forward, so both poolings cost one pass
# --pair-iptm-only is REQUIRED by --pool-mean (its path already holds the PAE probabilities).
# --select-by is NOT passed: selection stays on the MAX, the shipped convention, so mean_dockq stays
# comparable to every earlier report and the mean-pooled numbers arrive as extra columns.
# run_logging/test_shipped_eval.py asserts this string is score_checkpoint.sh's COMMON minus the
# pathway flags, so the two cannot drift apart without a test failing.
COMMON="--prefetch --pair-iptm-only --pool-mean"

# THE SHELL'S OWN ASSERTION, BEFORE ANY CARD IS TOUCHED. phase_k_eval refuses a shipped pass that
# installs a pathway (refuse_shipped_with_pathways, at startup) and merge_rows refuses it again from
# the shards' own record -- this is the third, cheapest layer, and the one that reads at the point
# where someone would paste a flag in. A pattern match is enough: every pathway flag this evaluator
# has is spelled --pose-* or --frame-*.
assert_no_pathway(){
  case " $1 " in
    *" --pose-"*|*" --frame-"*)
      say "STOP: a pathway flag reached the shipped argv -- $1"
      echo "      The shipped head is stock Protenix. See the header of this file for the twelve"
      echo "      GPU-minutes this refusal is paying back."; exit 8;;
  esac
}

# ONE REPORT FUNCTION, SO A FRESH PASS AND AN EXISTING ONE PRINT THE SAME TABLE. The point of the
# pass is the comparison of poolings, and baseline_* is its built-in control: those two fields are
# computed from results.csv's own stored (max-pooled) ipTM over the same interfaces, so they MUST
# reproduce the max-pooled selection of this pass. If they do not, the pass is not scoring the
# candidates it claims and neither pooling can be trusted.
report(){ # $1 json  $2 label
  $PY -I - "$1" "$2" <<'PYG'
import json, sys
d = json.load(open(sys.argv[1]))
print()
print(f"  SHIPPED HEAD -- {sys.argv[2]}")
print(f"    {d['n_interfaces']} interface(s), budget {d['budget']}, selected by {d['select_by'].upper()}")
oth = "_select_mean" if d["select_by"] == "max" else "_select_max"
other = "mean" if d["select_by"] == "max" else "max"
print(f"    {d['select_by']:>4}-pooled ipTM : mean_dockq {d['mean_dockq']:.4f}   frac_acc {d['frac_acc']:.4f}")
print(f"    {other:>4}-pooled ipTM : mean_dockq {d['mean_dockq'+oth]:.4f}   frac_acc {d['frac_acc'+oth]:.4f}")
print(f"    oracle            : mean_dockq {d['oracle_mean_dockq']:.4f}   frac_acc {d['oracle_frac_acc']:.4f}"
      f"   headroom {d['headroom']:.4f}")
print(f"    CONTROL baseline_* (stored max-pooled, must equal the max row above): "
      f"{d['baseline_mean_dockq']:.4f} / {d['baseline_frac_acc']:.4f}")
if d["ckpt"] != "shipped":
    raise SystemExit(f"STOP: {sys.argv[1]} records ckpt {d['ckpt']!r}, not 'shipped'. This table "
                     f"would be labelling something else as the shipped baseline.")
PYG
}

# ONE PASS, SO THE TWO SIDES CANNOT DIVERGE IN ANYTHING BUT THEIR POOL. Everything that differs
# between val and in-sample is an argument here; the flags, the guards and the report are shared.
run_side(){ # $1 label  $2 out json  $3 work dir  $4 budget  $5 extra
  local LAB="$1" OUT="$2" WORK="$3" BUD="$4" EXTRA="$5"
  assert_no_pathway "$EXTRA"
  if [ -n "${DRY:-}" ]; then
    echo "DRY $LAB: $PY -I phases/phase_k_eval_sharded.py --cards $CARDS --budget $BUD --work $WORK --out $OUT --extra \"$EXTRA\""
    return 0
  fi
  # THE EXISTING NUMBER WINS, because the shipped head cannot have changed. The val pass of
  # 2026-11-11 is still on disk with both poolings; re-running it would be twelve minutes of three
  # cards to reproduce a constant, and those cards belong to a training arm.
  # REPORT_ONLY=1 PRINTS WHAT EXISTS AND LAUNCHES NOTHING, whatever the cards are doing. The
  # queue script leads with this: a queued pass that prints nothing for five hours and then prints
  # a number that was on disk at minute zero is a script nobody reads -- and the alternative,
  # calling this script with DRY unset to "just see what is there", would have LAUNCHED the pass
  # the moment the cards happened to be free.
  if [ -n "${REPORT_ONLY:-}" ]; then
    if [ -s "$OUT" ]; then report "$OUT" "$LAB" || return 5
    else say "$LAB: not scored yet ($OUT absent) -- needs a pass"; fi
    return 0
  fi
  if [ -s "$OUT" ] && [ -z "${FORCE:-}" ]; then
    say "$LAB: $OUT already holds this pass (shipped head, fixed pool, fixed manifest -- the number"
    echo "      cannot move). Printing it; set FORCE=1 to re-score on purpose."
    report "$OUT" "$LAB" || return 5
    return 0
  fi
  # THE CARDS ARE CHECKED BY OCCUPANCY, NOT BY PROCESS NAME. cards_busy.py fails CLOSED: anything
  # it cannot establish prints a non-numeric token, so an unknown state refuses the launch. This is
  # the check that keeps a 12-minute pass off a training arm's cards.
  # A FORCED RE-SCORE GOES SOMEWHERE NEW. The rows already in $WORK are the ones whose self-check
  # verified the three joins (val: max |diff| 0.000050 over 21,150 rows, 2026-11-11), and the shard
  # file names are a function of the budget alone -- rows_b30_shipped_s*.csv -- so a re-score into
  # the same directory would overwrite them with no way back. Nothing in this project may delete
  # measured data to make room for a repeat of itself.
  if [ -s "$OUT" ] && [ -n "${FORCE:-}" ]; then
    local STAMP; STAMP=$(date +%Y%m%d_%H%M%S)
    say "$LAB: FORCE=1 with $OUT present -- re-scoring into a NEW location, keeping the existing rows"
    OUT="${OUT%.json}_rescored_$STAMP.json"
    WORK="${WORK}_rescored_$STAMP"
    say "$LAB:   -> $OUT"
  fi
  local BUSY
  BUSY="$($PY -I run_logging/cards_busy.py "$CARDS" 2>/dev/null)"
  [ "${BUSY:-9}" = "0" ] || { say "REFUSING: cards_busy($CARDS) reports ${BUSY:-unknown} -- not launching onto busy cards"; return 1; }
  mkdir -p "$WORK" "$(dirname "$OUT")" || return 3
  say "$LAB: scoring the SHIPPED head, budget $BUD, both poolings, on cards $CARDS"
  LAYERNORM_TYPE=openfold $PY -I phases/phase_k_eval_sharded.py \
      --cards "$CARDS" --budget "$BUD" --work "$WORK" --out "$OUT" --extra "$EXTRA" \
      2>&1 | tee "$WORK/pass.log"
  local RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] || { say "ABORT: $LAB pass exit $RC -- see $WORK/pass.log"; return 4; }
  report "$OUT" "$LAB" || return 5
}

RC=0
if [ "$SIDES" = both ] || [ "$SIDES" = val ]; then
  # VAL: the 141 interfaces of training/valfull83/pool.csv at budget 30 from seed base 400000, with
  # the pool's own seed manifest -- the SAME pool, budget, roles file and manifest score_checkpoint.sh
  # scores a trained head on, which is the only way the two numbers are comparable. The roles file is
  # the one beside the fold (byte-identical to training/valfull83/chain_roles.csv, md5 94f7b46d...);
  # the fold's plain chain_roles.csv covers only the first wave and leaves 49 of the 141 interfaces
  # unmappable, which stopped all three shards once.
  VAL_OUT=${VAL_OUT:-$R/training/valfull83/metric_shipped_b30_141.json}
  VAL_WORK=${VAL_WORK:-$R/training/valfull83/shipped_rho_shards}
  run_side "val (valfull83, 141 interfaces, 30 seeds)" "$VAL_OUT" "$VAL_WORK" 30 \
    "--fold $R/validation/fold --results $R/validation/fold/results.csv \
--pool $R/training/valfull83/pool.csv --roles $R/validation/fold/chain_roles_all83.csv \
--seed-base 400000 --seed-manifest $R/training/valfull83/seed_manifest.json \
--join-cache $R/training/valfull83/join_cache.json $COMMON" || RC=$?
fi

if [ "$SIDES" = both ] || [ "$SIDES" = insample ]; then
  # IN-SAMPLE: the fixed 280-item training slice at budget 15 from seed base 500000, on the training
  # fold. POOL is overridable because the same argv must be able to score the whole training set
  # (training/combined/pool.csv) without anyone hand-building a command line -- retyping an argv has
  # cost this project four failed launches in twenty minutes. The join cache and the metric json live
  # BESIDE THE POOL, so they move with it instead of being shared and silently stale.
  POOL=${POOL:-$R/training/slice280/pool.csv}
  [ -f "$POOL" ] || { say "ABORT: POOL $POOL is not a file"; exit 6; }
  PD=$(dirname "$POOL")
  IN_OUT=${IN_OUT:-$PD/metric_shipped_b15.json}
  IN_WORK=${IN_WORK:-$PD/shipped_b15_shards}
  run_side "in-sample ($(basename "$PD"), $(( $(wc -l < "$POOL") - 1 )) interfaces, 15 seeds)" \
    "$IN_OUT" "$IN_WORK" 15 \
    "--fold $R/training/fold --results $R/training/fold/results.csv \
--pool $POOL --roles $R/training/fold/chain_roles_all.csv \
--seed-base 500000 --join-cache $PD/join_cache.json $COMMON" || RC=$?
fi

exit $RC
