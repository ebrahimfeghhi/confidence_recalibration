#!/bin/bash
# THE POOLING-MATCHED SHIPPED CONTROL, on the SAME 141 interfaces the arm is scored on.
#
# WHY IT IS NEEDED. eval_curve.rho_series takes the shipped reference from the `stored_iptm`
# column, which is the shipped head's MAX-pooled ipTM and nothing else. So quoting the arm's
# MEAN-pooled rho (+0.2324) against that +0.1515 is not a like-for-like comparison: part of the
# gap could be mean-vs-max pooling rather than anything training did. External review raised it;
# the MAX comparison (+0.1910 vs +0.1515) was already matched and is unaffected.
#
# This runs the SHIPPED head -- no --ckpt -- over the identical pool, budget and seed manifest,
# with --pool-mean so the same pass yields both poolings. Four cells then exist: shipped max,
# shipped mean, tuned max, tuned mean.
#
# TIMED FOR AN EVAL BOUNDARY, at the user's direction. It waits for the epoch-10 metric json, so
# it lands when the training ranks have just finished their own eval. It uses OUR cards only --
# every other card on this box belongs to someone else.
set -u
cd /home/ebrahim/confidence_recalibration/extra_training || exit 1
PY=/home/ebrahim/Protenix/.venv/bin/python
R=$(pwd)
RID="${RID:-rdi25k}"
EP="${EP:-10}"
CARDS="${CARDS:-4,5,7}"
OUT="training/valfull83/metric_shipped_b30_141.json"
WORK="training/valfull83/shipped_rho_shards"
say(){ echo "### $(TZ=America/New_York date '+%-I:%M:%S %p %Z') $*"; }

# THE ROLES FILE COMES FROM THE POOL DIRECTORY, NOT THE FOLD. phase_k_train resolves it that
# way (`_srol = <pool dir>/chain_roles.csv if it exists`), and the fold's own chain_roles.csv
# names only a subset: pointing at it made 49 of the 141 pool interfaces unmappable and all three
# shards refused rather than silently scoring a subset. Verified: 0 of 141 unmappable with the
# pool directory's file.
_ROLES=training/valfull83/chain_roles.csv
[ -f "$_ROLES" ] || { say "REFUSING: $_ROLES missing; the fold's file covers only a subset"; exit 6; }

say "waiting for $RID epoch $EP's own eval to land (ceiling 6 h)"
for _ in $(seq 1 2160); do
  [ -s "training/runs/$RID/metric_valfull83_ep${EP}.json" ] && break
  sleep 10
done
[ -s "training/runs/$RID/metric_valfull83_ep${EP}.json" ] \
  || { say "REFUSING: epoch $EP metric never appeared. Nothing run."; exit 2; }
say "epoch $EP eval is in; starting the shipped control"

mkdir -p "$WORK" || exit 3
$PY -I phases/phase_k_eval_sharded.py \
  --cards "$CARDS" --budget 30 --out "$R/$OUT" --work "$R/$WORK" \
  --extra "--fold $R/validation/fold --results $R/validation/fold/results.csv \
--pool $R/training/valfull83/pool.csv --roles $R/training/valfull83/chain_roles.csv \
--seed-base 400000 --seed-manifest $R/training/valfull83/seed_manifest.json \
--join-cache $R/training/valfull83/join_cache.json --pair-iptm-only --prefetch --pool-mean"
rc=$?
say "sharded eval exit $rc"
[ "$rc" -eq 0 ] || exit 4

$PY -I - "$OUT" "$RID" "$EP" <<'PYG'
import csv, glob, json, os, statistics, sys
sys.path.insert(0, "run_logging")
import eval_curve as EC
out, rid, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(out))
print(f"  shipped control: {d.get('n_interfaces')} interface(s), select_by {d.get('select_by')}")

def rho_from(paths, col):
    by = {}
    for f in paths:
        for r in csv.DictReader(open(f)):
            try:
                by.setdefault(r["iface"], []).append((float(r[col]), float(r["dockq"])))
            except (KeyError, TypeError, ValueError):
                pass
    v = [EC.spearman([a for a, _ in g], [b for _, b in g]) for g in by.values() if len(g) >= 3]
    v = [x for x in v if x == x]
    return (statistics.median(v) if v else float("nan")), len(v)

ship = sorted(glob.glob("training/valfull83/shipped_rho_shards/rows_*.csv"))
tuned = sorted(glob.glob(f"training/runs/{rid}/valfull83_eval_shards_ep{ep}/rows_*.csv"))
print(f"  shipped rows {len(ship)} file(s), tuned rows {len(tuned)} file(s)")
print()
print(f"  {'':10} {'max-pooled':>12} {'mean-pooled':>12}")
for lab, paths in (("shipped", ship), (f"ep{ep}", tuned)):
    if not paths:
        print(f"  {lab:10} {'no rows':>12} {'no rows':>12}")
        continue
    rmax, n1 = rho_from(paths, "iptm")
    rmean, n2 = rho_from(paths, "iptm_mean")
    print(f"  {lab:10} {rmax:+12.4f} {rmean:+12.4f}   ({n1}/{n2} interfaces)")
print()
print("  POOLING-MATCHED deltas are the columns, read top to bottom. Comparing the arm's")
print("  mean-pooled number against the shipped MAX-pooled one is what this control exists")
print("  to stop.")
PYG
say "done -> $OUT"
