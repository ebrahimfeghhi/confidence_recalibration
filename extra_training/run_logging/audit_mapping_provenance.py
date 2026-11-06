#!/usr/bin/env python
"""SAME POSE, SAME REFERENCE, SCORED NOW -- against the row the csv already holds.

WHY THIS AND NOT THE CROSS-SEED COMPARISON. Raised in review and correct: comparing mean DockQ at
seeds 500000-500009 against 500010-500014 compares DIFFERENT POSES, so it is a distributional
sanity check and not a mapping-identity test. Two groups can differ under an identical scorer and a
valid mapping, and a systematic wrong-chain assignment can produce plausible, stable distributions
at both ends. The discriminating test is to re-score the RETAINED pose cif that produced a stored
row and compare per sample.

WHAT A DISAGREEMENT MEANS. The current scorer refuses maps it cannot verify by sequence
(eval/ost_dockq.py, 2eb1f2fd). So three outcomes are distinguishable, and only the third is a
data-integrity problem:
    REPRODUCED   same dockq to 1e-3 -- the historical row is confirmed on its own pose
    REFUSED      the current code declines to score this pair at all -- the row is UNVERIFIED,
                 neither confirmed nor contradicted, because the new code is more conservative
    DIFFERS      the current code scores it and gets a different number -- the historical row and
                 the current rules disagree about what this interface is

IT WRITES ITS OST JSON TO A SCRATCH DIRECTORY. scorer.py reuses an existing ost_json when it is
usable, so scoring in place would re-read the cached historical result and compare it with itself.
"""
import argparse, collections, csv, json, os, sys

sys.path.insert(0, "/home/ebrahim/confidence_recalibration")
import paths
sys.path.insert(0, str(paths.EVAL_DIR))
import ost_dockq as O


def stored(results, pdbs, seed):
    out = collections.defaultdict(dict)
    with open(results) as fh:
        for r in csv.DictReader(fh):
            if r["seed"] == str(seed) and r["iface"].split("_")[0] in pdbs:
                out[r["iface"]][str(r["sample"])] = r["dockq"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs", required=True, help="comma separated")
    ap.add_argument("--seed", default="500009", help="a seed whose rows predate the mapping change")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--refs", default="training/refs")
    ap.add_argument("--pool", default="training/combined/pool.csv")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--nsample", type=int, default=5)
    a = ap.parse_args()

    pdbs = [p.strip().lower() for p in a.pdbs.split(",") if p.strip()]
    pool = collections.defaultdict(set)
    for r in csv.DictReader(open(a.pool)):
        if r["pdb"] in pdbs:
            pool[r["pdb"]].add("".join(sorted((r["chain1"], r["chain2"]))))
    have = stored(os.path.join(a.fold, "results.csv"), set(pdbs), a.seed)
    os.makedirs(a.scratch, exist_ok=True)

    tally = collections.Counter()
    rows = []
    for pdb in pdbs:
        import glob
        rd = glob.glob(os.path.join(a.fold, "poses", pdb, f"{a.seed}-*"))
        if not rd:
            print(f"{pdb}: no pose directory at seed {a.seed}"); continue
        ref = os.path.join(a.refs, f"{pdb}.cif")
        for k in range(a.nsample):
            cif = os.path.join(rd[0], "predictions", f"{pdb}_sample_{k}.cif")
            if not os.path.exists(cif):
                print(f"  {pdb} sample {k}: no pose cif"); continue
            out = os.path.join(a.scratch, f"{pdb}_{a.seed}_{k}.json")
            res = O.score(cif, ref, out)
            if not res:
                print(f"  {pdb} sample {k}: the scorer produced nothing at all")
                for cc in sorted(pool[pdb]):
                    tally["REFUSED"] += 1
                    rows.append((f"{pdb}_{cc}", k, have.get(f"{pdb}_{cc}", {}).get(str(k)), None,
                                 "REFUSED"))
                continue
            for cc in sorted(pool[pdb]):
                ifc = f"{pdb}_{cc}"
                old = have.get(ifc, {}).get(str(k))
                try:
                    v = O.interface(res, cc[0], cc[1])
                except ValueError as e:
                    v = None
                now = None if not v or v.get("dockq") is None else float(v["dockq"])
                if old is None and now is None:
                    verdict = "absent both"
                elif now is None:
                    verdict = "REFUSED"
                elif old is None:
                    verdict = "NEW (absent historically)"
                elif abs(float(old) - now) <= 1e-3:
                    verdict = "REPRODUCED"
                else:
                    verdict = "DIFFERS"
                tally[verdict] += 1
                rows.append((ifc, k, old, now, verdict))

    print(f"\n{'interface':12s} {'s':>2s} {'stored':>8s} {'now':>8s}  verdict")
    for ifc, k, old, now, v in rows:
        print(f"{ifc:12s} {k:2d} {('-' if old is None else old):>8s} "
              f"{('-' if now is None else f'{now:.4f}'):>8s}  {v}")
    print("\ntally: " + ", ".join(f"{k}={n}" for k, n in sorted(tally.items())))
    return 1 if tally["DIFFERS"] else 0


if __name__ == "__main__":
    sys.exit(main())
