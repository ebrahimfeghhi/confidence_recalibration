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
for spec in "plot_rel_arm.py:train" "plot_eval_series.py:validation"; do
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
