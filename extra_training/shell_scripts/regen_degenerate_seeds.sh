#!/bin/bash
# RE-FOLD SEEDS 500001-500014 FOR THE 22 DEGENERATE STRUCTURES that carry the 25 mixed interfaces.
#
# WHY. The DockQ-SD<0.05 prune kept one trunk per structure and all fifty DockQ labels. For 17 of
# these 22 the only acceptable pose sits at a seed whose trunk is gone, and a re-fold does not
# reproduce a seed's poses (phase_i_fold_validation.sh says so in its own header) -- so the labels
# were orphans. They are already archived to training/fold/archive/orphaned_degenerate_rows.csv;
# this generates fresh draws in their place, scored as they are folded.
#
# 14 SEEDS, NOT 15. Seed 500000's trunk is still on disk and its scores still match it. It is simply
# not queued; had it been, folder.py:959's csv to-do check would drop it, since its 240 rows are
# still there. 500000 + 500001-500014 = the 15 the user asked for.
#
# THE STALE POSE DIRS HAVE TO BE OUT OF THE WAY FIRST. folder.py:996 checks saved_run_dir BEFORE it
# folds and, finding poses on disk with no rows, RESCORES them and returns "rescored" -- no trunk is
# written and the summary line does not even count it. That path exists so a unit whose scoring died
# can be repaired without a re-fold, and our archived rows look exactly like that case. So seeds
# 500001-500009 of these 22 are only re-folded once training/fold/poses/<pdb>/<seed>-* is moved
# aside; archive_stale_pose_dirs.sh does that, and this script refuses to start until it has.
#
# COST, measured off seed 500010's own log (168 units on card 4 in 2.25 h = 48 s/unit): 22 x 14 =
# 308 units, ~1.4 h on three cards, ~49 GB of trunks at these structures' measured 159 MB mean. With
# the stale pose dirs left in place only 110 of those units would fold, which is the bug above.
#
# IT REUSES chain_extra_seeds.sh rather than copying its body: that script already carries the
# headroom gate, the per-seed completeness check and the OST recovery sweep for the interfaces the
# inline scorer cannot reach. LIST and WAIT_FOR are the two knobs this needs.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# WAIT FOR THE TRAINING ARM'S LAST EVAL. The arm stops itself at epoch 70; the cards are not free
# until its eval shards are down, so this gates on cards_busy rather than on the epoch count.
say "waiting for the cards (the hard-overfit arm's epoch-70 eval is still on 4,5,7)"
for i in $(seq 1 240); do
    B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
    [ "$B" = "0" ] && break
    sleep 30
done
B=$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)
[ "$B" = "0" ] || { say "ABORT: cards_busy still reports '$B' after 2 h"; exit 2; }
# REFUSE IF ANY STALE POSE DIR IS STILL THERE: its presence means 9 of the 14 seeds would be
# rescored off old poses instead of folded, silently and with no trunk.
STALE=$(ls -d training/fold/poses/*/50000[1-9]-* 2>/dev/null | awk -F/ '{print $4}' |
        grep -xFf <(grep -v '^#' docs/degenerate_regen_structures.txt) | wc -l)
if [ "$STALE" -gt 0 ]; then
    say "ABORT: $STALE stale pose dir(s) for seeds 500001-500009 of the 22 are still in place."
    say "  folder.py:996 would RESCORE those poses instead of folding, writing no trunk."
    say "  run ./shell_scripts/archive_stale_pose_dirs.sh first."
    exit 3
fi
say "no stale pose dirs; cards free; starting the fold"

LIST=docs/degenerate_regen_structures.txt \
SEEDS="500001 500002 500003 500004 500005 500006 500007 500008 500009 500010 500011 500012 500013 500014" \
WAIT_FOR=none TAG_RUN=_degen STRICT_IFACE=1 \
MIN_HEADROOM_GB=200 \
    ./shell_scripts/chain_extra_seeds.sh
RC=$?
say "chain exit $RC"
exit $RC
