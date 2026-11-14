#!/bin/bash
# Renders BOTH canonical figures for an arm. The only command that should ever make them.
#
#   refresh_plots.sh <rid>[,<rid2>...]  [outdir]
#
# ATOMIC: each figure is written to a temporary name and moved into place, so a reader never
# opens a half-written PNG, and a failed render leaves the previous figure intact rather than a
# truncated one.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
RIDS="${1:?usage: refresh_plots.sh <rid>[,<rid2>] [outdir]}"
OUT="${2:-figures}"
FIRST="${RIDS%%,*}"
mkdir -p "$OUT"
rc=0
# WHICH PLOTTER FITS THIS ARM. plot_rel_arm draws four RANKING-loss panels; an arm with
# --pae-only and no --rank-loss leaves three of them permanently empty, which is not a degraded
# view but the wrong instrument. plot_pae_arm draws what a PAE-only arm actually has: the training
# PAE, both poolings of held-out DockQ and frac-acceptable against their OWN shipped baselines,
# and the within-interface ipTM/DockQ correlation. Decided from the run's own args.json so a
# caller never has to remember.
SPECS='plot_rel_arm.py:train plot_eval_series.py:validation'
_A="training/runs/$FIRST/args.json"
if [ -f "$_A" ] && $PY -I -c "
import json,sys
a=json.load(open('$_A'))
sys.exit(0 if (a.get('pae_only') and not a.get('rank_loss')) else 1)" 2>/dev/null; then
  # ONE FIGURE, NOT TWO. plot_pae_arm already carries the held-out panels, and
  # plot_eval_series assumes the `_select_max` alternate key that a MAX-selecting arm does not
  # write -- it has `_select_mean` -- so it dies on a None in its own axis-span helper.
  SPECS='plot_pae_arm.py:pae'
  echo "  PAE-only arm: using plot_pae_arm.py (plot_eval_series is for mean-selecting arms)"
fi
for spec in $SPECS; do
  script="${spec%%:*}"; name="${spec##*:}"
  tmp="$OUT/.${FIRST}_${name}.tmp.png"
  if $PY -I "run_logging/$script" "$RIDS" "$tmp" > "$OUT/.${FIRST}_${name}.log" 2>&1; then
    mv -f "$tmp" "$OUT/${FIRST}_${name}.png"
    echo "  ok  $OUT/${FIRST}_${name}.png"
  else
    rm -f "$tmp"
    echo "  FAILED $name -- previous figure left in place; see $OUT/.${FIRST}_${name}.log"
    tail -3 "$OUT/.${FIRST}_${name}.log" | sed 's/^/      /'
    rc=1
  fi
done
exit $rc
