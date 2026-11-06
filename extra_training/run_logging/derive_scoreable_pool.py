#!/usr/bin/env python
"""Derive the SCOREABLE pool from a pool.csv, and a manifest of everything left out.

WHY NOT FILTER pool.csv IN PLACE. The pool defines which interfaces exist, and 84,400 rows in
training/fold/results.csv already name interfaces from it. Editing it would silently change the
denominator of every completeness figure ever quoted against those rows -- "97.4% scored" means
nothing if the total can move afterwards. So the original is immutable and this writes two derived
files beside it:

    pool_scoreable.csv   the interfaces a DockQ row can exist for, same columns as pool.csv
    pool_excluded.csv    every interface left out, with the measurement that excluded it

WHAT IS EXCLUDED, AND WHY EACH IS NOT A FAULT IN THE FOLD:

  no_contact -- the two reference chains do not touch, so there is no interface to score. These
  come from phase_g_build_val_jobs.py pairing every antibody chain with every antigen chain with
  no contact filter: in a structure with two Fabs or two antigen copies, one Fab's light chain is
  paired with the copy it does not bind. 3rhw_DK is 9.82 A apart, 4rdq_DF is 19.25 A. The absence
  of a row for these is correct, and counting them as missing work is what made the training fold
  look 4.3% incomplete when it was 2.6% incomplete.

  unrepresentable -- a chain name that is not a single character. An interface is written
  <pdb>_<c1><c2>, which cannot express 9dc3's chains ABID, GAID and VA: "9dc3_ABID" could be A+BID
  or AB+ID, and run_model/scorer.py refuses any pair that is not exactly two characters. No queued
  structure has one today (9dc3 is pooled but was never queued and has no rows), so this is listed
  rather than fixed -- changing the naming scheme would rewrite every iface string in every csv.
  Excluding it here means a future fold that includes such a structure gets a refusal with a
  reason, instead of a queue that silently produces nothing for it.

IT FAILS CLOSED. An interface whose contact cannot be measured -- absent reference, absent chain --
is neither scoreable nor excluded for a known reason, so it is reported and the tool exits 1 rather
than quietly dropping it into either file.

THE SAME RULE APPLIES TO TRAINING AND VALIDATION. The validation pool survives today only because
none of its 30 structures has more than one antigen chain; the builder is equally unfiltered there.
"""
import argparse, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contacts                                                        # noqa: E402

COLS = ["pdb", "chain1", "chain2"]


def classify(pool_rows, refs, cutoff):
    """out: (scoreable, excluded, uncheckable) lists of (row, nearest_A, reason)"""
    scoreable, excluded, uncheckable = [], [], []
    for r in pool_rows:
        pdb, c1, c2 = r["pdb"], r["chain1"], r["chain2"]
        if len(c1) != 1 or len(c2) != 1:
            excluded.append((r, None, "unrepresentable: chain name is not a single character"))
            continue
        ref = os.path.join(refs, f"{pdb}.cif")
        if not os.path.exists(ref):
            uncheckable.append((r, None, f"no reference at {ref}"))
            continue
        n, mn = contacts.contacts(ref, c1, c2, cutoff)
        if n is None:
            uncheckable.append((r, None, "a chain is absent from the reference"))
        elif n:
            scoreable.append((r, mn, ""))
        else:
            excluded.append((r, mn, f"no_contact: nearest heavy atoms {mn:.2f} A apart"))
    return scoreable, excluded, uncheckable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--cutoff", type=float, default=contacts.CUTOFF)
    ap.add_argument("--write", action="store_true",
                    help="write the derived files; without it this only reports")
    a = ap.parse_args()

    # PATHS ARE CHECKED BEFORE ANYTHING IS READ OR WRITTEN. Raised in review: the collision check
    # ran after the derived files had been written, so `--pool pool_scoreable.csv` would overwrite
    # its own input and only then refuse.
    d = os.path.dirname(os.path.abspath(a.pool))
    sp = os.path.join(d, "pool_scoreable.csv")
    xp = os.path.join(d, "pool_excluded.csv")
    if os.path.abspath(a.pool) in (sp, xp):
        raise SystemExit(f"STOP: --pool {a.pool} is one of this tool's own outputs; it would "
                         f"overwrite its input. Point --pool at the original pool.csv.")

    with open(a.pool) as fh:
        rows = list(csv.DictReader(fh))
    s, e, u = classify(rows, a.refs, a.cutoff)
    print(f"{a.pool}: {len(rows)} interface(s), cutoff {a.cutoff} A, references {a.refs}")
    print(f"  scoreable       : {len(s)}")
    print(f"  excluded        : {len(e)}")
    print(f"  UNCHECKABLE     : {len(u)}")
    by = {}
    for _r, _mn, why in e:
        by[why.split(":")[0]] = by.get(why.split(":")[0], 0) + 1
    for k, n in sorted(by.items()):
        print(f"      {k}: {n}")
    for r, _mn, why in u[:10]:
        print(f"      UNCHECKABLE {r['pdb']}_{r['chain1']}{r['chain2']}: {why}")

    if u:
        # NOTHING IS PUBLISHED FROM AN INCOMPLETE CLASSIFICATION. Raised in review: the derived
        # files used to be written and THEN the tool exited 1, so a reader that ignored the exit
        # code consumed a pool missing every interface that could not be measured.
        print("DERIVATION FAILED: the interfaces above are neither scoreable nor excluded for a "
              "known reason. Nothing written -- a partial pool is worse than no pool.")
        return 1

    if a.write:
        for path, items, extra in ((sp, s, False), (xp, e, True)):
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(COLS + (["nearest_atom_A", "cutoff_A", "reason"] if extra else []))
                for r, mn, why in items:
                    base = [r["pdb"], r["chain1"], r["chain2"]]
                    w.writerow(base + ([("" if mn is None else f"{mn:.3f}"),
                                        f"{a.cutoff}", why] if extra else []))
            os.replace(tmp, path)
            print(f"  wrote {path} ({len(items)} row(s))")

    print(f"DERIVATION OK: {len(s)} + {len(e)} = {len(rows)}, every interface accounted for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
