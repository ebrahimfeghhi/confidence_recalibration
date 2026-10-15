#!/usr/bin/env python
"""PHASE J -- compare two a3m files as ALIGNMENTS, not as statistics.

WHY AGGREGATES ARE NOT ENOUGH. Review constructed shifted hit alignments with identical hit ids,
depth, taxonomy AND per-column occupancy, and every aggregate check passed them. Occupancy counts how
many rows occupy a column; it says nothing about which residues are in it. So the comparison has to
reach the aligned strings themselves.

WHAT IS COMPARED, per hit shared between the two files:

  the MATCH-COLUMN STRING -- uppercase residues and gaps, lowercase stripped. This is the hit's
    alignment to the query, column for column. A shift changes it and changes nothing else.
  the INSERTION COUNT per locus -- lowercase runs carry real information in a3m, and a rebuild that
    drops or relocates them has changed the alignment even if the match columns agree.
  the TAXID, parsed from the accession suffix.

MATCHED BY ACCESSION *AND* SEGMENT. One accession can appear more than once with different aligned
segments of the same target, so the accession alone is not a key. The segment comes from the header's
target coordinates.

THREE CATEGORIES, REPORTED SEPARATELY, because they mean different things: a LOST hit may be a
prefilter difference, a GAINED hit may be the same difference in the other direction -- reordering
replaces hits, so gains and losses come together and neither alone establishes corruption -- while a
CHANGED alignment on a SHARED hit is the serious one: the same homolog aligned differently is the
signature of a shifted or rebuilt alignment.

NO TOLERANCES. This reports; it does not decide. Differences are to be investigated, not compared
against a threshold chosen to let them through.
"""
import argparse
import collections
import json
import os
import re
import sys


def read_a3m(path):
    rows, h, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if h is not None:
                    rows.append((h, "".join(buf)))
                h, buf = line, []
            else:
                buf.append(line)
    if h is not None:
        rows.append((h, "".join(buf)))
    return rows


def key_of(header):
    """(accession, segment) from an a3m header.

    The stored headers look like
        >UniRef100_UPI0009948DBD_9606/\t220\t0.816\t5.505E-61\t0\t210\t211\t0\t211\t214
    so the accession is the first field and the trailing numbers carry the aligned coordinates. The
    LAST THREE are the target's start, end and length -- that triple is the segment. A rebuild may
    reorder or reformat the middle fields, which is why only the accession and the segment are the
    key and everything else is cosmetic.
    """
    f = header.lstrip(">").split("\t")
    acc = f[0].rstrip("/")
    seg = tuple(f[-3:]) if len(f) >= 4 else ("", "", "")
    return acc, seg


def taxid(acc):
    m = re.search(r"_(\d+)$", acc)
    return m.group(1) if m else ""


def normalise(seq):
    """(match-column string, insertion profile). Lowercase is an a3m insertion, not a column."""
    match, ins, run = [], [], 0
    for c in seq:
        if c.islower():
            run += 1
        else:
            match.append(c.upper())
            ins.append(run)
            run = 0
    ins.append(run)                        # a trailing insertion after the last match column
    return "".join(match), tuple(ins)


def compare(a_path, b_path, label, out=sys.stdout):
    A, B = read_a3m(a_path), read_a3m(b_path)
    if not A or not B:
        print(f"{label}: EMPTY ({len(A)} vs {len(B)} rows)", file=out)
        return False
    qa = A[0][1].replace("-", "").upper()
    qb = B[0][1].replace("-", "").upper()
    res = {"label": label, "query_equal": qa == qb, "stored_rows": len(A) - 1,
           "rebuilt_rows": len(B) - 1}
    if qa != qb:
        print(f"{label}: QUERY DIFFERS -- {len(qa)} vs {len(qb)} residues. Nothing else is "
              "comparable.", file=out)
        res["verdict"] = "FAIL"
        return res

    # every row must carry exactly len(query) match columns, in BOTH files -- definitional
    bad_rows = []
    for nm, rows in (("stored", A), ("rebuilt", B)):
        for i, (h, s) in enumerate(rows):
            m, _ = normalise(s)
            if len(m) != len(qa):
                bad_rows.append(f"{nm} row {i}: {len(m)} match columns, query has {len(qa)}")
    res["malformed_rows"] = bad_rows

    da = {}
    for h, s in A[1:]:
        da[key_of(h)] = (h, s)
    db = {}
    for h, s in B[1:]:
        db[key_of(h)] = (h, s)

    shared = set(da) & set(db)
    lost = sorted(set(da) - set(db))
    gained = sorted(set(db) - set(da))
    changed_align, changed_ins, changed_tax = [], [], []
    for k in sorted(shared):
        ma, ia = normalise(da[k][1])
        mb, ib = normalise(db[k][1])
        if ma != mb:
            first = next((j for j in range(min(len(ma), len(mb))) if ma[j] != mb[j]), None)
            changed_align.append((k, first))
        elif ia != ib:
            changed_ins.append(k)
        if taxid(k[0]) != taxid(k[0]):     # same key, so same accession -- kept for symmetry
            changed_tax.append(k)

    res.update(shared=len(shared), lost=len(lost), gained=len(gained),
               changed_alignment=len(changed_align), changed_insertions=len(changed_ins))
    exact = (not lost and not gained and not changed_align and not changed_ins and not bad_rows)
    res["verdict"] = "EXACT" if exact else "DIFFERS"

    print(f"\n=== {label} ===", file=out)
    print(f"  query identical            : {res['query_equal']}", file=out)
    print(f"  rows  stored {res['stored_rows']} vs rebuilt "
          f"{res['rebuilt_rows']}", file=out)
    print(f"  malformed rows             : {len(bad_rows)}", file=out)
    for b in bad_rows[:3]:
        print(f"     {b}", file=out)
    print(f"  shared hits                : {len(shared)}", file=out)
    print(f"  LOST   (stored only)       : {len(lost)}", file=out)
    print(f"  GAINED (rebuilt only)      : {len(gained)}", file=out)
    print(f"  CHANGED ALIGNMENT on shared: {len(changed_align)}   <- the serious one", file=out)
    for k, pos in changed_align[:5]:
        print(f"     {k[0]} segment {k[1]}: first differing match column {pos}", file=out)
    print(f"  CHANGED INSERTIONS on shared: {len(changed_ins)}", file=out)
    print(f"  VERDICT: {res['verdict']}", file=out)
    if not exact:
        print("  Differences are to be INVESTIGATED, not compared against a tolerance. A changed"
              "\n  alignment on a shared hit is a shifted alignment; lost and gained together are"
              "\n  usually a prefilter/ordering difference, which still needs explaining.", file=out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True,
                    help="controls.json from phase_j_corrected_queries.py")
    ap.add_argument("--stored-root", default=None, help="the existing msa store")
    ap.add_argument("--rebuilt-root", required=True, help="where the control rebuild landed")
    ap.add_argument("--file", default="pairing.a3m", choices=("pairing.a3m", "non_pairing.a3m"))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    stored_root = a.stored_root or f"{ROOT}/building_msa/work/mmcif_msa"

    ctl = json.load(open(a.controls))
    results = []
    for name, meta in sorted(ctl.items()):
        sp = os.path.join(stored_root, meta["stored_msa_dir"], a.file)
        rp = os.path.join(a.rebuilt_root, name, a.file)
        if not os.path.exists(sp):
            print(f"{name}: no stored {a.file} at {sp}"); continue
        if not os.path.exists(rp):
            print(f"{name}: no rebuilt {a.file} at {rp} -- the control search has not run")
            continue
        results.append(compare(sp, rp, name))

    ok = [r for r in results if r and r.get("verdict") == "EXACT"]
    print(f"\n{len(ok)} of {len(results)} control(s) EXACT")
    if a.report:
        json.dump(results, open(a.report, "w"), indent=1)
        print(f"wrote {a.report}")
    if not results:
        raise SystemExit("STOP: no control could be compared")
    if len(ok) != len(results):
        raise SystemExit("STOP: not every control reproduced exactly. Investigate the categories "
                         "above before searching the corrected queries, and do NOT adjust search "
                         "parameters to make this pass.")


if __name__ == "__main__":
    main()
