#!/usr/bin/env python
"""Gate between the validation seed manifest and any destructive trim. Read-only.

THE FAILURE THIS PREVENTS. The manifest names 30 seeds per structure, chosen with DockQ. The
evaluator needs a LABEL npz per selected unit to score it, and labels currently exist only for
seeds 400000-400029 while the selection reaches outside that range on 50 of 53 structures. Trim
first and the selected pool cannot be scored, and the trunks needed to rebuild those labels are
gone -- so the selection cannot even be regenerated.

WHAT IS CHECKED, all of it a hard gate on trimming:
  1. the manifest parses, its digest matches its own content, and the algorithm is recorded
  2. every structure carries EXACTLY k distinct seeds, all inside the generated seed range
  3. every selected seed has a TRUNK unit on disk -- the thing the trim must keep
  4. every selected seed has a LABEL npz -- the thing the trim must not outrun
  5. every selected seed resolves in results.csv with the expected pose count, so the
     seed-to-pose mapping the scorer joins on is present and complete
  6. the per-interface statistics in the manifest still reproduce from results.csv, which catches
     a manifest written against a different results.csv than the one on disk now

Exit 0 only when all six pass. Nothing is deleted or modified.

Usage:
  python -I run_logging/verify_val_manifest.py --manifest training/valfull/seed_manifest.json
"""
import argparse, collections, csv, hashlib, json, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="training/valfull/seed_manifest.json")
ap.add_argument("--trunks", default="validation/fold/trunks")
ap.add_argument("--labels", default="validation/fold/labels")
ap.add_argument("--results", default="validation/fold/results.csv")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--show", type=int, default=6)
a = ap.parse_args()

man = json.load(open(os.path.join(ROOT, a.manifest)))
problems = collections.defaultdict(list)

# 1 -- digest over the manifest minus the digest field, exactly as the writer computed it
dig = man.pop("digest", None)
recomputed = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()[:16]
man["digest"] = dig
print(f"manifest {a.manifest}")
print(f"  algo {man.get('algo')}  k={man.get('k')}  rng={man.get('rng')}")
print(f"  digest recorded {dig}  recomputed {recomputed}  "
      f"{'MATCH' if dig == recomputed else 'MISMATCH'}")
if dig != recomputed:
    problems["digest"].append(("manifest", f"{dig} != {recomputed}"))
if not man.get("algo"):
    problems["digest"].append(("manifest", "no algo recorded"))

K = int(man["k"])
lo, hi = man["seed_range"]
sel = {p: v["seeds"] for p, v in man["structures"].items()}

# 2 -- exactly K distinct seeds, in range
for p, ss in sel.items():
    if len(ss) != K:
        problems["seed_count"].append((p, f"{len(ss)} seeds, expected {K}"))
    if len(set(ss)) != len(ss):
        problems["seed_count"].append((p, "duplicate seeds"))
    out = [s for s in ss if not (lo <= s <= hi)]
    if out:
        problems["seed_count"].append((p, f"{len(out)} seed(s) outside {lo}-{hi}"))

# 3 and 4 -- trunk and label present for every selected unit
for p, ss in sel.items():
    for s in ss:
        if not os.path.exists(os.path.join(ROOT, a.trunks, p, f"free_{s}.npz")):
            problems["trunk_present"].append((p, f"free_{s}.npz"))
        if not os.path.exists(os.path.join(ROOT, a.labels, p, f"free_{s}.npz")):
            problems["label_present"].append((p, f"free_{s}.npz"))

# 5 and 6 -- results.csv coverage, pose counts, and the recorded statistics
have = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(os.path.join(ROOT, a.results))):
    ifc = (r.get("iface") or "").strip()
    if not ifc:
        continue
    try:
        have[ifc][int(r["seed"])].append(float(r["dockq"]))
    except (ValueError, TypeError, KeyError):
        continue
for p, ss in sel.items():
    ifcs = [i for i in have if i.split("_")[0] == p]
    if not ifcs:
        problems["results_cover"].append((p, "no interface in results.csv"))
    for i in ifcs:
        for s in ss:
            n = len(have[i].get(s, []))
            if n != a.nsample:
                problems["results_cover"].append((f"{i}/{s}", f"{n} pose(s), expected {a.nsample}"))
for p, v in man["structures"].items():
    for ifc, rec in v.get("interfaces", {}).items():
        sub = [d for s in sel[p] for d in have.get(ifc, {}).get(s, [])]
        if not sub:
            continue
        if abs(max(sub) - rec["oracle_sub"]) > 1e-4:
            problems["stats_reproduce"].append((ifc, f"oracle {max(sub):.4f} vs recorded {rec['oracle_sub']:.4f}"))
        if abs(st.pstdev(sub) - rec["sd_sub"]) > 1e-4:
            problems["stats_reproduce"].append((ifc, f"sd {st.pstdev(sub):.4f} vs recorded {rec['sd_sub']:.4f}"))

CHECKS = ["digest", "seed_count", "trunk_present", "label_present",
          "results_cover", "stats_reproduce"]
print(f"\n{len(sel)} structure(s), {sum(len(v) for v in sel.values())} selected unit(s)\n")
for c in CHECKS:
    n = len(problems[c])
    print(f"  {'PASS' if not n else 'FAIL'}  {c:<16} {n} problem(s)")
    for k, why in problems[c][:a.show]:
        print(f"          {k}: {why}")
    if n > a.show:
        print(f"          ... and {n - a.show} more")

nlab = len(problems["label_present"])
if nlab:
    print(f"\n  {nlab} selected unit(s) have no label. BUILD THESE BEFORE ANY TRIM -- the trunks")
    print("  they are built from are exactly the ones the trim would keep, so the order is")
    print("  build labels, re-run this gate, then trim.")
bad = sum(len(v) for v in problems.values())
print(f"\n{'GATE PASSED' if not bad else 'GATE FAILED'}: {bad} problem(s)")
sys.exit(0 if not bad else 1)
