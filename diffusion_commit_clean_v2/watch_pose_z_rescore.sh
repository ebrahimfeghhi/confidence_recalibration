#!/bin/bash
# Every 30 minutes: is the per-pose rescoring sweep alive, and is the rescored ipTM ranking poses?
#
# THE QUESTION THIS WATCHES. Stock ipTM is a fold-level quantity: within one fold all five samples
# share a trunk z, measured within-fold pair_iptm spread 0.008 against DockQ ranges up to 0.68, so it
# cannot rank poses from one fold. The sweep gives each pose its own z (inject the pose, run one more
# recycle) so pair_iptm varies per pose. Two things then matter and both are printed:
#
#   SPREAD     within-fold pair_iptm range. Must be >> 0.008 or the mechanism did not fire.
#   SPEARMAN   within-fold rank correlation of rescored pair_iptm against DockQ, averaged over folds.
#              +1 = the scorer ranks poses correctly. -1 = it ranks them backwards, which is still a
#              usable signal (take the argmin) and is what the first smoke fold showed. 0 = useless.
#
#   SELECTION  what argmax-pair_iptm picks over the arm's poses, against the free-50 baseline pick,
#              per interface. This is the deliverable; the two above are diagnostics.
#
# REMEDIATE: a GPU in GPUS whose worker session is gone while the queue is unfinished gets its worker
# relaunched. Claims make that safe -- folder.py releases a dead worker's claim at startup.
#
# ESCALATE (printed as ESCALATE, left for the morning):
#   - workers up but no fold finished in 30 minutes
#   - out-of-memory or Traceback in a worker log
#   - finished folds that produced no scored rows (the silent-NaN class of failure)
#   - |mean Spearman| < 0.15 once 20+ folds are in: the scorer carries no ranking signal either way,
#     which is the result that would close this direction
set -u
E="${E:-/home/ebrahim/tts_abag/experimental_data/protenix/foldbench/pose_z_rescore}"
C=/home/ebrahim/tts_abag/diffusion_commit_clean_v2
PY=/home/ebrahim/Protenix/.venv/bin/python
GPUS="${GPUS:-0 1}"
PREFIX="${PREFIX:-pzr}"
LOG="$E/watch.log"
INTERVAL="${INTERVAL:-1800}"

say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

launch_worker() {
  local g=$1
  tmux new-session -d -s "${PREFIX}$g" \
    "cd /home/ebrahim/Protenix && CUDA_VISIBLE_DEVICES=$g LAYERNORM_TYPE=openfold \
     PROTENIX_POSE_Z_RESCORE=1 \
     $PY $C/folder.py --queue $E/queue.tsv --out $E/poses --csv $E/results.csv \
     --claim-dir $E/claims --dataset foldbench --ncycle 10 --nsample 5 --dtype bf16 \
     --overlap-scoring >> $E/w$g.log 2>&1"
  say "REMEDIATE: relaunched worker on GPU $g"
}

while true; do
  TOTAL=$(wc -l < "$E/queue.tsv")
  DONE=$(find "$E/poses" -name run.json 2>/dev/null | wc -l)
  ROWS=$(( $(wc -l < "$E/results.csv" 2>/dev/null || echo 1) - 1 ))
  say "folds $DONE/$TOTAL   scored rows $ROWS"

  # --- remediate dead workers while work remains ---
  if [ "$DONE" -lt "$TOTAL" ]; then
    for g in $GPUS; do
      tmux has-session -t "${PREFIX}$g" 2>/dev/null || launch_worker "$g"
    done
  fi

  # --- escalate hard failures ---
  for g in $GPUS; do
    [ -f "$E/w$g.log" ] || continue
    if grep -qE "CUDA out of memory|Traceback|AssertionError" "$E/w$g.log"; then
      say "ESCALATE: failure signature in w$g.log -> $(grep -oE 'CUDA out of memory|AssertionError|Traceback' "$E/w$g.log" | sort -u | tr '\n' ' ')"
    fi
  done

  # --- the science: spread, within-fold Spearman, and selection vs the free-50 baseline ---
  "$PY" - "$E" <<'PYEOF' 2>&1 | tee -a "$LOG"
import csv, os, statistics as st, sys
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean_v2")
E = sys.argv[1]
p = f"{E}/results.csv"
if not os.path.exists(p):
    print("  (no results.csv yet)"); raise SystemExit
rows = [r for r in csv.DictReader(open(p)) if r.get("dockq") and r.get("iptm")]
if not rows:
    print("  (no scored rows yet)"); raise SystemExit

def spearman(a, b):
    n = len(a)
    if n < 3: return None
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n
        i = 0
        while i < n:                                  # average ties
            j = i
            while j+1 < n and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2.0
            i = j+1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = (sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb)) ** 0.5
    return num/den if den else None

by = {}
for r in rows:
    by.setdefault((r["iface"], r["seed"]), []).append((float(r["dockq"]), float(r["iptm"])))
by = {k: v for k, v in by.items() if len(v) >= 5}
print(f"  complete folds with 5 scored poses: {len(by)}")
if not by: raise SystemExit

spreads, rhos = [], []
for v in by.values():
    ip = [x[1] for x in v]
    spreads.append(max(ip) - min(ip))
    rho = spearman([x[1] for x in v], [x[0] for x in v])
    if rho is not None: rhos.append(rho)
print(f"  within-fold iptm SPREAD: median {st.median(spreads):.4f}  (shared-z baseline 0.008)")
if rhos:
    m = st.mean(rhos)
    print(f"  within-fold SPEARMAN iptm vs DockQ: mean {m:+.3f}  median {st.median(rhos):+.3f}  n={len(rhos)} folds")
    pos = sum(1 for x in rhos if x > 0)
    print(f"    folds with positive rho: {pos}/{len(rhos)}")
    if len(rhos) >= 20 and abs(m) < 0.15:
        print(f"    ESCALATE: |mean rho| {abs(m):.3f} < 0.15 at n={len(rhos)} -- no ranking signal either way")

# selection, per interface, against the free-50 baseline
try:
    import paths
    from score_experiment import pool_by_seed, classes_at, ACCEPTABLE
    DS = paths.DATASETS["foldbench"]
    pool, first = pool_by_seed(50, str(DS.pool_csv)); cls = classes_at(pool, first)
    per = {}
    for r in rows:
        per.setdefault(r["iface"], []).append((float(r["dockq"]), float(r["iptm"])))
    print(f"  {'interface':14}{'cls':>7}{'poses':>7}{'argmax':>8}{'argmin':>8}{'oracle':>8}{'|':>3}{'free50':>8}")
    for iface in sorted(per):
        v = per[iface]; pdb = iface.split("_")[0]
        keep = [r for r in pool.get(iface, []) if r[0] in first.get(pdb, ())]
        fs = max(keep, key=lambda r: r[3])[2] if keep else float("nan")
        amax = max(v, key=lambda x: x[1])[0]
        amin = min(v, key=lambda x: x[1])[0]
        print(f"  {iface:14}{cls.get(iface,'?'):>7}{len(v):>7}{amax:>8.3f}{amin:>8.3f}"
              f"{max(x[0] for x in v):>8.3f}{'|':>3}{fs:>8.3f}")
except Exception as e:
    print(f"  (selection table unavailable: {e})")
PYEOF

  if [ "$DONE" -ge "$TOTAL" ]; then
    say "QUEUE COMPLETE: $DONE/$TOTAL -- stopping watcher"
    touch "$E/DONE"
    break
  fi
  sleep "$INTERVAL"
done
