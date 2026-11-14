#!/bin/bash
# THE OVERFIT TEST ON THE HARD LISTS. Train on nothing but the training interfaces that are hard
# by GROUND TRUTH ALONE -- at least one acceptable candidate and at most 10% acceptable -- and read
# in-sample top-1 on exactly those lists. Generalisation is explicitly not the question.
#
# 151 interfaces over 97 structures (95 after the standing SKIP list); median 75 candidates with
# median 2 acceptable. The split uses only DockQ, never any head's score, or the test would be
# "fit the lists this head already fails", which is circular.
#
# PAE LOSS ONLY, no ranking term.
#
# WHAT IS MEASURED, AND WHERE:
#   every epoch, on the CROPPED training batches -- the PAE loss split by group (acceptable vs
#     unacceptable at DockQ 0.23), via --pae-split-accept. It reuses the per-pose loss the step
#     has already computed, so it is free, and it gives a per-epoch trend between full passes.
#   every 5 epochs, on the UNCROPPED hard lists -- two --extra-eval passes, split by trunk depth
#     because a budget is a claim about poses that must exist: 130 interfaces at budget 15 and 21
#     at budget 1, 9,855 candidate poses per round, roughly 5-6 min on three cards, 12 rounds over
#     60 epochs. The other four reads (top-1, rank of the first acceptable pose, score gap,
#     headroom) are computed from those passes' own rows, where the poses are whole.
#
# THAT COSTS THE PER-EPOCH READOUT, and it is worth being explicit about: the trainer's IN-SAMPLE
# SELECTED DockQ line lives inside the cross-seed ranking block, gated on
# `a.rank_cross_seed and a.rank_loss`, so a PAE-only run prints no in-sample top-1 at all. The
# run still saves a head every epoch, so the answer comes from scoring those checkpoints over
# training/overfit_hard/pool.csv afterwards, at whatever cadence is chosen -- nothing is lost by
# deciding that later.
#
# NO HELD-OUT EVAL (EVAL_EVERY=0) and NO DROPOUT: both exist to help generalisation, and the
# question here is only whether the inputs can fit these lists at all.
#
# WHAT EACH OUTCOME MEANS. If top-1 on these lists stays near its starting value, the inputs do
# not contain what separates the rare good candidate from the dominant bad ones, and no loss or
# data change fixes that. If it climbs toward the oracle, the signal is present and the problem is
# generalisation -- i.e. how many hard lists there are, how they are weighted, and regularisation.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
PY=/home/ebrahim/Protenix/.venv/bin/python
say() { echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }
newest_rid() { ls -td training/runs/*/ | grep -vE "/latest/$" | head -1 | xargs basename; }

# THE FLAG MUST EXIST BEFORE THE CARDS MATTER. --pae-split-accept is being added in a parallel
# edit; launching with it before it lands would die on an unknown argument after waiting hours for
# the cards. Checked against the trainer's own --help, not by grepping the source, so a flag that
# is present but unparseable also fails here.
say "waiting for --pae-split-accept to exist in the trainer"
for i in $(seq 1 180); do
    $PY -I phases/phase_k_train.py --help 2>&1 | grep -q -- "--pae-split-accept" && break
    sleep 60
done
$PY -I phases/phase_k_train.py --help 2>&1 | grep -q -- "--pae-split-accept" || {
    say "ABORT: --pae-split-accept never appeared; not launching a run that cannot measure the split"
    exit 2; }
say "--pae-split-accept is present"

say "waiting for the cards (the in-sample pass is using them)"
for i in $(seq 1 180); do
    [ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] && break
    sleep 60
done
[ "$($PY -I run_logging/cards_busy.py 2>&1 | tail -1)" = "0" ] || { say "ABORT: cards still busy after 3 h"; exit 3; }

PRE=$(newest_rid)
say "=== overfit on 151 hard lists, 60 epochs, no eval, no dropout ==="
SUBSET_OK=1 \
MANIFEST=training/overfit_hard/structures.csv \
SCORED_PAIRS=training/overfit_hard/pool.csv \
CARDS=4,5,7 EPOCHS=60 EVAL_EVERY=0 LR=1e-3 \
PAE_SPLIT_ACCEPT=0.23 \
EXTRA_EVAL="name=hard15,pool=training/overfit_hard/pool_b15.csv,fold=training/fold,seed_base=500000,budget=15,every=5;name=hard1,pool=training/overfit_hard/pool_b1.csv,fold=training/fold,seed_base=500000,budget=1,every=5" \
PREFETCH_WORKERS=16 TRUNK_MEMMAP=1 EPOCH_BALANCE=1 DRY=0 \
    nohup ./shell_scripts/launch_pae_crop_arm.sh > training/runs/overfit_hard_launch.log 2>&1 &
LPID=$!
say "launcher backgrounded as pid $LPID"
RID=""
for i in $(seq 1 90); do
    N=$(newest_rid); [ "$N" != "$PRE" ] && { RID="$N"; break; }
    if ! [ -d "/proc/$LPID" ]; then
        say "ABORT: launcher exited before a run appeared"
        grep -E "REFUS|STOP:|command not found|Traceback" training/runs/overfit_hard_launch.log | head -5
        exit 4
    fi
    sleep 20
done
[ -n "$RID" ] || { say "ABORT: no run directory after 30 min"; exit 5; }
echo "$RID" > training/runs/overfit_hard_rid
echo 0 > "training/runs/${RID}_last_plotted"
say "overfit arm is $RID"
for i in $(seq 1 60); do [ -f "training/runs/$RID/args.json" ] && break; sleep 10; done
nohup ./shell_scripts/health_watch.sh "$RID" > "training/runs/$RID/health.log" 2>&1 &
nohup ./shell_scripts/plot_on_epoch.sh "$RID" > "training/runs/$RID/plot_on_epoch.log" 2>&1 &
say "watchers up on $RID -- PAE-only, so the in-sample read comes from scoring its"
say "  saved heads over training/overfit_hard/pool.csv once it has run"
