#!/bin/bash
# Delete fold trees that a later, larger fold has entirely replaced, and finished benchmark output.
#
# WHY THESE AND NOT THE BIG ONES. After the checkpoints went, the repo's 2.3 TB is almost all
# training/fold (1.7 TB) and validation/fold (365 GB), and those are the substrate the trainer and
# the evaluator read: trunks the labels are built from, poses the scoring recovery needs, and the
# canonical results.csv. They are not clutter and nothing here touches them.
#
# WHAT IS ACTUALLY SUPERSEDED, measured rather than assumed:
#   validation/fold_regen   23 GB   4 structure(s), 3,001 rows, last written 2026-10-15
#   validation/fold        365 GB  84 structure(s), 70,501 rows, last written 2026-11-05
# All 4 of fold_regen's structures are present in validation/fold, so it holds nothing unique --
# `comm -23` over the two pose listings returns zero. It was a regeneration pass that the full fold
# then subsumed.
#
#   training/ckpt_probe     1.3 GB  the size-conditional checkpointing benchmark's output, with
#                                   zero references from phases/, run_logging/ or shell_scripts/.
# The benchmark's CONCLUSION lives in paper_results.md and EXPERIMENTS.md; this is its scratch.
#
# IT CHECKS THE SUPERSESSION RATHER THAN TRUSTING THIS COMMENT, because the comment was true when it
# was written and the tree is not immutable. If fold_regen ever holds a structure the live fold does
# not, this refuses and says which.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
DRY="${DRY:-1}"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }
say "script $(basename "$0") sha $(sha1sum "$0" | cut -c1-12)  DRY=$DRY"

# NOTHING MAY BE FOLDING OR RECOVERING. A sweep reads poses; a merge replaces results.csv.
S=$(tmux ls 2>/dev/null | grep -cE '^(w1fold|w1val|w2_)' || true)
if [ "$S" -gt 0 ]; then say "REFUSING: $S fold session(s) running."; exit 1; fi
if [ -e training/fold/.recover_gaps.lock ] && ! flock -n training/fold/.recover_gaps.lock true; then
  say "REFUSING: the gap recovery holds its lock; it is reading poses right now."
  exit 1
fi

TOTAL=0
plan(){ # plan <path> <reason>
  local p=$1 why=$2
  [ -e "$p" ] || { say "  $p: already gone"; return; }
  local sz; sz=$(du -sm "$p" 2>/dev/null | cut -f1)
  say "  $p  ${sz} MiB  -- $why"
  TOTAL=$((TOTAL + sz))
  TARGETS="${TARGETS:-} $p"
}

say "checking that validation/fold_regen is entirely contained in validation/fold"
UNIQ=$(comm -23 <(ls validation/fold_regen/poses 2>/dev/null | sort) \
                <(ls validation/fold/poses 2>/dev/null | sort) | tr '\n' ' ')
if [ -n "${UNIQ// /}" ]; then
  say "REFUSING: validation/fold_regen holds structure(s) the live fold does not: $UNIQ"
  exit 1
fi
say "  every fold_regen structure is present in validation/fold"

plan validation/fold_regen "4 structure(s) all present in the 84-structure live fold, stale since 2026-10-15"
plan training/ckpt_probe   "finished checkpointing-benchmark scratch, zero code references"

say "total ${TOTAL} MiB over:${TARGETS:-" nothing"}"
if [ "$DRY" = "1" ]; then
  say "DRY RUN: nothing deleted. Re-run with DRY=0."
  exit 0
fi
[ -z "${TARGETS:-}" ] && { say "nothing to do"; exit 0; }
Q=$(quota -u ebrahim 2>/dev/null | awk '/\/dev\//{print int($2/2^20)}')
for p in $TARGETS; do say "removing $p"; rm -rf -- "$p"; done
Q2=$(quota -u ebrahim 2>/dev/null | awk '/\/dev\//{print int($2/2^20)}')
say "quota used ${Q} GiB -> ${Q2} GiB"
say "RECLAIM COMPLETE"
