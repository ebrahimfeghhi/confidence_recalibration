#!/usr/bin/env bash
# THE watcher for a fold.py run. Health-checks, scores, and remediates every INTERVAL seconds.
#
# Replaces two overlapping watchers (diffusion_commit/watch_inmem.sh and the copy that lived beside the
# overnight run's data). Two watchers meant two places to fix a check, and they had already drifted:
# one knew how to detect which arm was live, the other had the structural checks.
#
# EVERY CHECK HERE CORRESPONDS TO A FAILURE THAT HAS ALREADY HAPPENED IN THIS PROJECT:
#   workers alive          a dead worker just stops adding rows; nothing else complains
#   progress advancing     alive-but-wedged looks identical to working from outside
#   claims vs folds        a claimed-but-unscored unit is skipped FOREVER on the next pass, and
#                          fold.py only reconciles at startup -- killing a worker mid-fold stranded
#                          9uzh seed 300011 for hours
#   splice == restore      in periodic mode a splice without its restore persists for the whole fold
#   splices per fold       a fold that silently skipped a splice is not the experiment
#   templates kept         "kept [4] of 5" is a near-duplicate pose deduped; "[0]" means the ids are
#                          broken and the fold measured nothing
#   decode sets per fold   a missing observation decode loses an entire recycle arm
#   confidence json        no json => no ipTM => only oracle DockQ is comparable, not SELECTED
#   score as we go         scoring drifted 442 pose sets behind when run by hand, which made two
#                          finished interfaces look missing and a third report on 10 of 25 seeds
#
# Usage:
#   ROOT=<run dir> ARM=<arm name> EXPECT=<folds> GPUS="1 5 6" [DECODES=1] [NDECODE=3] bash watch_folds.sh
# Stop with: touch <ROOT>/WATCH_STOP
set -u
ROOT=${ROOT:?set ROOT to the run directory (the --out passed to fold.py)}
ARM=${ARM:-}                      # empty => detect from the live worker's --queue
EXPECT=${EXPECT:-0}
GPUS=${GPUS:-}
INTERVAL=${INTERVAL:-300}
STALL_MIN=${STALL_MIN:-25}
NDECODE=${NDECODE:-0}             # expected mid-loop decode dirs per fold (0 = do not check)
DECODES=${DECODES:-0}             # 1 = score the mid-loop decodes as they appear
PY=${PY:-/home/ebrahim/Protenix/.venv/bin/python}
DC=/home/ebrahim/tts_abag/diffusion_commit
LOG=$ROOT/watch.log
rm -f "$ROOT/WATCH_STOP"
last=-1; last_change=$(date +%s)

live_arm() {
  # From the RUNNING process, never from log existence: an aborted launch left a stale at13 log and a
  # file-based check then reported the wrong arm while the real one folded unwatched.
  ps -u "$USER" -o cmd= | grep -oE "[f]old\.py --queue \S+" | head -1 \
    | sed 's#.*/queue_\?##; s#\.tsv##'
}

while :; do
  [ -f "$ROOT/WATCH_STOP" ] && { echo "[$(date +%H:%M:%S)] stopped" >> "$LOG"; exit 0; }
  now=$(date +%s)
  arm=${ARM:-$(live_arm)}; arm=${arm:-unknown}
  logs=$(ls "$ROOT"/*${arm}*gpu*.log "$ROOT"/${arm}_g*.log 2>/dev/null | sort -u)
  [ -z "$logs" ] && logs=$(ls "$ROOT"/*.log 2>/dev/null | grep -v watch | sort -u)
  cnt() { [ -z "$logs" ] && echo 0 || grep -hc "$1" $logs 2>/dev/null | paste -sd+ | bc; }
  f=$(cnt "interfaces in"); f=${f:-0}
  fail=$( { grep -hcE "FAILED|SCORING produced no" $logs 2>/dev/null | paste -sd+ | bc; } || echo 0); fail=${fail:-0}
  sp=$(cnt "fed own pose"); sp=${sp:-0}
  re=$(cnt "restored the original"); re=${re:-0}
  w=$(ps -u "$USER" -o cmd= | grep -cE "^[^ ]*python[0-9.]* .*[f]old\.py --queue")
  cdir=$(ls -d "$ROOT"/.claims* 2>/dev/null | head -1)
  cl=$([ -n "$cdir" ] && ls "$cdir" | wc -l || echo 0)
  [ "$f" -ne "$last" ] && { last=$f; last_change=$now; }
  stalled=$(( (now - last_change) / 60 ))

  m="[$(date '+%m-%d %H:%M:%S')] $arm ${f}/${EXPECT} | claims ${cl} | workers ${w} | fail ${fail}"
  [ "$sp" -gt 0 ] && m="$m | splice ${sp} restore ${re}"
  for g in $GPUS; do
    m="$m | gpu$g $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" 2>/dev/null)%"
  done
  [ "$sp" -gt 0 ] && [ "$sp" -ne "$re" ] && m="$m | !! splice/restore mismatch"
  # a failed fold contributes 1..3 splices and an in-flight one 0..3, so bound rather than predict
  if [ "$sp" -gt 0 ] && [ "$NDECODE" -gt 0 ]; then
    lo=$(( f * NDECODE )); hi=$(( (f + fail + w) * NDECODE ))
    { [ "$sp" -lt "$lo" ] || [ "$sp" -gt "$hi" ]; } && m="$m | !! ${sp} splices for ${f} folds"
  fi
  [ "$fail" -gt 0 ] && m="$m | !! ${fail} failures"
  [ "$w" -eq 0 ] && [ "$EXPECT" -gt 0 ] && [ "$f" -lt "$EXPECT" ] && m="$m | !! NO WORKERS"
  [ "$stalled" -ge "$STALL_MIN" ] && [ "$EXPECT" -gt 0 ] && [ "$f" -lt "$EXPECT" ] \
    && m="$m | !! no progress ${stalled}min"
  kept=$( { grep -hoE "kept \[[0-9, ]+\] of [0-9]+" $logs 2>/dev/null | sort | uniq -c | tr '\n' ' '; } || true)
  [ -n "$kept" ] && m="$m | templates: $kept"

  # structural check on FINISHED folds only: run.json is written after the fold returns, so its
  # absence means the fold is still going and its later decode dirs do not exist yet
  if [ "$NDECODE" -gt 0 ]; then
    bad=0
    for r in "$ROOT"/*/*/*/*/; do
      [ -f "$r/run.json" ] || continue
      [ -d "$r/_hook" ] || continue
      [ "$(ls -d "$r"/_hook/c* 2>/dev/null | wc -l)" -eq "$NDECODE" ] || bad=$((bad+1))
      for c in "$r"/_hook/c*; do
        [ "$(ls "$c"/poses/*summary_confidence*.json 2>/dev/null | wc -l)" -ge 1 ] || bad=$((bad+1))
      done
    done
    [ "$bad" -gt 0 ] && m="$m | !! ${bad} folds with missing decodes/confidence"
  fi

  # remediate leaked claims: fold.py reconciles only at startup
  if [ -n "$cdir" ] && [ "$cl" -gt $(( f + w )) ]; then
    "$PY" - "$ROOT" "$arm" "$w" "$cdir" <<'EOF' >> "$LOG" 2>&1
import csv, os, sys, collections
root, arm, live, cdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
scored = collections.defaultdict(set)
for cand in (f"{root}/{arm}.csv", f"{root}/results.csv"):
    if os.path.exists(cand):
        for r in csv.DictReader(open(cand)):
            scored[r["iface"].split("_")[0]].add(r["seed"])
stale = [c for c in os.listdir(cdir) if c.split("_")[-1] not in scored.get(c.split("_")[0], set())]
stale.sort(key=lambda c: os.path.getmtime(f"{cdir}/{c}"))
for c in stale[:max(0, len(stale) - live)]:
    os.rmdir(f"{cdir}/{c}"); print(f"  [watch] released leaked claim {c}")
EOF
  fi

  # score the mid-loop decodes, skipping if a scorer is already running so passes cannot pile up
  if [ "$DECODES" = 1 ] && ! ps -u "$USER" -o cmd= | grep -q "[s]core_hook_poses.py"; then
    nohup "$PY" "$DC/analysis/score_hook_poses.py" --root "$ROOT" \
      --csv "$ROOT/decodes_${arm}.csv" --jobs 6 >> "$ROOT/score.log" 2>&1 &
  fi

  echo "$m" >> "$LOG"
  [ "$EXPECT" -gt 0 ] && [ "$f" -ge "$EXPECT" ] && { echo "[$(date +%H:%M:%S)] $arm COMPLETE" >> "$LOG"; exit 0; }
  sleep "$INTERVAL"
done
