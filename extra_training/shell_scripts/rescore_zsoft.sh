#!/bin/bash
# RE-SCORE THE PAE + HELD-OUT RANKING LOSS ARM (zsoftmax, lr 1e-3) so it has rows, not just
# reports. Only report_*.txt and the metric jsons were preserved for that arm, and the
# ipTM-vs-DockQ correlation is a per-candidate quantity, so it cannot be recovered from a report.
#
# VAL ONLY. The slice side would double the GPU time and the question here is held-out ranking.
# Add SIDES=both if that changes.
#
# THE CHECKPOINTS LIVE ACROSS THE SEGMENT CHAIN -- the arm trained as segments, so epoch N's head
# is under whichever run id held that stretch. The map is built by globbing the chain rather than
# typed out, so a wrong path is a hard stop instead of a silently scored neighbour.
set -u
set -o pipefail
cd /home/ebrahim/confidence_recalibration/extra_training
SD=${SD:-/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad}
CHAIN="2tr3sj 82zwdx o4vn2i wir34q swopc6 o4rvah y565bm"
EPOCHS=${EPOCHS:-"1 2 4 7 8 10 12 14 20"}
SIDES=${SIDES:-val}
say(){ echo "### $(TZ=America/New_York date '+%-I:%M %p %Z') $*"; }

# REFUSE BEFORE SPENDING ANY GPU. Nine passes is ~2.5 hours; finding a missing head at pass seven
# wastes the first six.
declare -A CK
for E in $EPOCHS; do
  HIT=""
  for R in $CHAIN; do
    for F in training/runs/$R/head_ep${E}_step*.pt; do
      [ -f "$F" ] || continue
      [ -n "$HIT" ] && { say "STOP: epoch $E has two heads, $HIT and $F"; exit 2; }
      HIT="$F"
    done
  done
  [ -n "$HIT" ] || { say "STOP: no head for epoch $E anywhere in the chain"; exit 2; }
  CK[$E]="$HIT"
done
say "all $(echo $EPOCHS | wc -w) checkpoints located"
for E in $EPOCHS; do echo "    ep$E -> ${CK[$E]}"; done

for E in $EPOCHS; do
  T="zs_ep$E"
  if [ -s "$SD/report_$T.txt" ] && [ "$(ls "$SD/val30_$T"/rows_*.csv.meta.json 2>/dev/null | wc -l)" -eq 3 ]; then
    say "ep$E already has a complete pass, skipping"
    continue
  fi
  say "=== epoch $E: ${CK[$E]} ==="
  CKPT="${CK[$E]}" TAG="$T" SIDES="$SIDES" SD="$SD" ./shell_scripts/score_checkpoint.sh \
    || { say "ABORT at epoch $E"; exit 3; }
done
say "RESCORE COMPLETE"
