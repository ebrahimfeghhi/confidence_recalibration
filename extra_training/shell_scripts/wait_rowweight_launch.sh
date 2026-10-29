#!/bin/bash
# Report the new arm's startup: the four banner lines that say it is the arm it claims to be,
# plus the smoke test's verdict and its first read on the premise.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training
BEFORE=$(ls training/runs | grep -E '^[a-z0-9]{6}$' | sort | tr '\n' ' ')
DEADLINE=$(( $(date +%s) + 5400 ))
echo "waiting for a new run dir (known: $(echo "$BEFORE" | wc -w))"
RID=""
while [ -z "$RID" ]; do
  for d in $(ls training/runs | grep -E '^[a-z0-9]{6}$'); do
    case " $BEFORE " in *" $d "*) ;; *) RID="$d"; break;; esac
  done
  [ -n "$RID" ] && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: no new run dir"; exit 1; }
  sleep 15
done
echo "rid=$RID"
echo "$RID" > training/runs/rowweight_rid
echo 0 > training/runs/${RID}_last_plotted
while :; do
  [ -f "training/runs/$RID/run.log" ] && \
    grep -qE "ROW-WEIGHTED|steps/epoch|trainable:|STOP|Traceback" "training/runs/$RID/run.log" && break
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "GAVE UP: no startup lines"; exit 1; }
  sleep 10
done
echo "--- smoke verdict ---"
grep -E "PASS|FAIL|FIRST LOOK|rho\(" training/runs/_smoke_rowweight/../_smoke_rowweight/smoke.log 2>/dev/null | tail -4
tail -25 training/runs/rowweight_chain.log 2>/dev/null | grep -E "smoke|PASS|FAIL|rho|launching|HOLDING" | tail -8
echo "--- startup ---"
grep -E "ROW-WEIGHTED|steps/epoch|COSINE|trainable:|structure\(s\)|interface\(s\)|anchor SNAPSHOT|STOP|Traceback" \
  "training/runs/$RID/run.log" | head -14
