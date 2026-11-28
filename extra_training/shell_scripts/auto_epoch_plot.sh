#!/usr/bin/env bash
# THE PER-EPOCH PLOT, FOR WHATEVER IS RUNNING, WITHOUT BEING RE-ARMED.
#
# Every previous version of this was a cron pinned to one run id, so each new arm needed a new
# cron and every handover lost the epochs between the old job being deleted and the new one
# firing -- which happened at Arm A->B, at B->decay, and twice for Arm E. The user should not
# have to ask for the plot.
#
# THE LIVE RUN IS DISCOVERED, NOT CONFIGURED: the run directory whose run.log was written most
# recently, within STALE seconds. The reference arm (drawn grey) comes from a file that can be
# repointed once per experiment rather than per run.
#
# Prints one of:
#   SEND <rid> <epoch>   -- a new epoch landed and the figure has been redrawn
#   QUIET                -- nothing new, or nothing training
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/ebrahim/Protenix/.venv/bin/python
STALE=${STALE:-600}
FIG=${FIG:-figures/paeonly_group_loss.png}
REF_FILE=training/runs/plot_reference_rid
REF=$(cat "$REF_FILE" 2>/dev/null || echo "")

# THE COMPANION ARM WINS OVER plot_reference_rid. The L/M comparison is the experiment: the two
# differ only in the loss, so the figure should show those two and nothing else. The chain writes
# plot_reference_rid per round and once pointed Arm L at Arm I, which put a third, differently
# configured arm (injected layers, dropout on) on the same axes -- three curves where two were
# wanted, and the odd one out was the arm that had already been ruled out.
#
# SO: if the live run is one of the two arms, the reference is the OTHER one. In round 1 Arm M does
# not exist yet, so the reference resolves to empty and the figure shows Arm L alone -- which is
# correct, not a failure. Falls back to plot_reference_rid for any other arm, so nothing else
# changes.
_L=$(cat training/runs/armL_rid 2>/dev/null || echo "")
_M=$(cat training/runs/armM_rid 2>/dev/null || echo "")

now=$(date +%s); live=""; newest=0
for d in training/runs/*/; do
    # `latest` IS A SYMLINK TO THE LIVE RUN, not a run of its own. Including it made the newest
    # run.log appear twice, and whichever name won the tie became the figure's label -- the
    # watcher announced "latest" instead of the arm, and wrote its epoch marker under that name,
    # which would then let the real run's next epoch send twice or not at all.
    case "$(basename "$d")" in latest|archive) continue;; esac
    l="$d/run.log"; [ -f "$l" ] || continue
    m=$(stat -c %Y "$l"); age=$(( now - m ))
    [ "$age" -le "$STALE" ] || continue
    [ "$m" -gt "$newest" ] && { newest=$m; live=$(basename "$d"); }
done
[ -n "$live" ] || { echo "QUIET"; exit 0; }
# A DIAGNOSTIC IS NOT AN ARM. The live run with the newest log can be a one-epoch probe
# (gradient split, memory walk, smoke gate) whose loss curve means nothing next to a 20-epoch
# arm -- drawing it would relabel the figure mid-experiment and send a one-point line. Any run
# with a `_noplot` marker is skipped here and the watcher stays QUIET until a real arm starts.
[ -f "training/runs/${live}_noplot" ] && { echo "QUIET"; exit 0; }
[ "$live" = "$REF" ] && { echo "QUIET"; exit 0; }

# `|| true` IS LOAD-BEARING. This file runs under `set -e` with pipefail, and a brand-new arm has
# no head_ep*.pt yet, so `ls` exits 2, the pipeline inherits it, and the assignment killed the
# whole script -- the watcher exited 2 instead of printing QUIET for the first ~8 minutes of
# every new arm, which is exactly the handover it exists to cover.
ep=$(ls "training/runs/$live"/head_ep*.pt 2>/dev/null | sed -E 's/.*head_ep([0-9]+)_.*/\1/' \
     | sort -n | tail -1 || true)
ep=${ep:-0}
# EPOCH 0 IS NOT AN EPOCH. A run that has started but not finished its first epoch has nothing
# new to draw, and `prev` defaults to -1, so without this it would send an empty figure once.
[ "$ep" -gt 0 ] || { echo "QUIET"; exit 0; }
mark="training/runs/${live}_last_autoplot"
prev=$(cat "$mark" 2>/dev/null || echo -1)

# THE VAL PASS FINISHES AFTER THE HEAD, so it is NOT reportable on the epoch's own tick. The val
# check used to sit below the epoch gate, which exits when the epoch has not advanced -- so the
# val figure could only ever be drawn on a tick that ALSO had a new epoch, and a val pass landing
# two minutes later was silently skipped until the next boundary. The owner asked for the val
# plots when the val pass finishes, so EITHER a new epoch OR a new val pass now makes a tick
# report, and the two are tracked by separate markers.
VFIG=${VFIG:-figures/val6_live.png}
vmark="training/runs/${live}_last_valplot"
CHAIN_REV=$("$PY" -I -c "
import sys; sys.path.insert(0,'run_logging')
import plot_fullset as P
print(' '.join(list(d.split('/')[-1] for d,_ in P.chain('$live'))[::-1]))" 2>/dev/null)
vep=0
for r in $CHAIN_REV; do
  for f in training/runs/$r/metric_valfull83_ep*.json; do
    [ -s "$f" ] || continue
    e=$(echo "$f" | sed -E 's/.*_ep([0-9]+)\.json/\1/')
    [ "$e" -gt "$vep" ] && vep=$e
  done
done
vprev=$(cat "$vmark" 2>/dev/null || echo -1)
NEW_EP=0; NEW_VAL=0
[ "$ep"  -gt "$prev"  ] && NEW_EP=1
[ "$vep" -gt "$vprev" ] && NEW_VAL=1
[ "$NEW_EP" = 1 ] || [ "$NEW_VAL" = 1 ] || { echo "QUIET"; exit 0; }

# ARM L AND ARM M ON ONE FIGURE, both starting from shipped at epoch 0.
#
# THE PAIRING IS EXPLICIT, NOT "whatever ran last". The earlier version picked the reference by
# asking which of two rid files the live run was NOT, which silently drew a dead arm whenever a
# third arm started. Arm L's chain is named by training/runs/armL_rid and is the control for the
# TM-weighted arm: same native head, same pool, same lr, differing only in the pair weight. Drop
# ARM_L= in the environment to plot the live arm alone.
#
# ORDER MATTERS: plot_group_loss_arms colours predecessor-first, so the reference goes in first
# and gets grey, the live arm second and gets blue.
name(){ cat "training/runs/$1_name" 2>/dev/null || echo "ref"; }
args=()
ARM_L="${ARM_L-$(cat training/runs/armL_rid 2>/dev/null || echo "")}"
if [ -n "$ARM_L" ] && [ "$ARM_L" != "$live" ] && [ -d "training/runs/$ARM_L" ]; then
  args+=("$(name "$ARM_L") ($ARM_L)=$ARM_L")
fi
args+=("$(name "$live") ($live)=$live")
# THE SHIPPED BASELINE, DRAWN ON THE FIGURE, read from a file beside the arms rather than baked
# into this script. training/runs/shipped_baseline_good|bad hold the uncropped shipped means over
# the SAME units the live arm trains on; if either file is absent the lines are simply omitted and
# the figure is what it was before. Writing them to disk rather than hardcoding keeps the baseline
# tied to the pool: the hard 280 and val83 have different shipped levels, and drawing one against
# the other is exactly the mismatch that has to be avoided.
SG=$(cat training/runs/shipped_baseline_good 2>/dev/null || echo "")
SB=$(cat training/runs/shipped_baseline_bad 2>/dev/null || echo "")
SA=$(cat training/runs/shipped_baseline_all 2>/dev/null || echo "")
# AN ARM WITHOUT --pae-split-accept HAS NO GOOD/BAD SPLIT, and the group figure then cannot be
# drawn for it at all -- plot_group_loss_arms skips the arm and leaves whatever the PREVIOUS arm
# drew on disk, so the watcher reported "SEND qkicr2 22" and the figure beside it still showed Arm
# S and Arm N from an hour earlier. A stale figure announced as the live arm is worse than none.
#
# SO: if the live run has no group data, fall back to the STITCHED CHAIN TOTAL
# (plot_chain_loss.py), which is defined for every run because it reads trajectory.csv. The
# figure path is the same either way, so the cron's SEND line keeps meaning "this file is current
# for this arm".
HAS_GROUP=$("$PY" -I -c "
import sys; sys.path.insert(0,'run_logging')
import plot_fullset as P
print(len(P.group_loss(P.chain('$live'))))" 2>/dev/null || echo 0)
if [ "${HAS_GROUP:-0}" -gt 0 ]; then
  TITLE="PAE training loss on good vs bad poses" \
      SHIPPED_GOOD="$SG" SHIPPED_BAD="$SB" SHIPPED_ALL="$SA" \
      "$PY" -I run_logging/plot_group_loss_arms.py \
      "$FIG" "${args[@]}" >/dev/null 2>&1 || { echo "QUIET"; exit 0; }
else
  "$PY" -I run_logging/plot_chain_loss.py "$FIG" "$live" \
      --title "$(name "$live") ($live) -- stitched training PAE loss across the resume chain" \
      >/dev/null 2>&1 || { echo "QUIET"; exit 0; }
fi
echo "$ep" > "$mark"

# SIX PANELS: {mean DockQ, frac acceptable, mean rank of 1st acceptable} x {max, mean} ipTM.
# The chain goes in oldest-first so one continuous curve spans every run that contributed epochs.
VAL_TOKEN=""
if [ "$NEW_VAL" = 1 ]; then
  if "$PY" -I run_logging/plot_val_curve.py "$VFIG" $CHAIN_REV \
        --title "$(name "$live") chain ($CHAIN_REV) -- held out, val83 leak-free" >/dev/null 2>&1; then
    echo "$vep" > "$vmark"
    VAL_TOKEN=" VAL $vep"
  fi
fi
echo "SEND $live $ep$VAL_TOKEN"
