#!/usr/bin/env python
"""Convert phase_j_fill_scores output into phase_l_merge_recovered's canonical header.

WHY A CONVERTER AND NOT A HAND EDIT. phase_j_fill_scores writes
    iface,seed,arm,sample,dockq,pair_iptm,model_chains
and phase_l_merge_recovered refuses anything whose header is not exactly
    iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm
That refusal is correct -- it is what stops a row with a blank iptm entering the canonical file,
where folder.open_csv would skip it while it sat there looking present. So the mapping is done
once, here, in a form that can be re-run and reviewed, rather than by editing a csv by hand.

THE MAPPING, each column justified against an EXISTING canonical row for the same interface
(9q5m_EH seed 500009, read 2026-11-11: `9q5m_EH,500009,free,-1,,0,0.1350,0.5366`):
    donor_pose  -> -1    the existing rows carry -1; phase_j hardcodes a free fold
    tmpl_chains -> ''    empty in the existing rows, because a free fold has no template
    iptm        <- pair_iptm, rounded to 4dp to match the existing rows' precision
    dockq       <- dockq, formatted to 4dp for the same reason

model_chains IS DROPPED, DELIBERATELY. It records which symmetry copy won for that pose, which is
real provenance -- but results.csv has no column for it and the earlier 10 seeds of 9q5m_EH do not
record it either, so inventing a column here would make this interface's rows differ in shape from
every other row in the file. It is kept in the phase_j output, which is not deleted.

Usage:
  python -I run_logging/fill_to_canonical.py <fill.csv> <out.csv>
"""
import csv
import sys

SRC = ["iface", "seed", "arm", "sample", "dockq", "pair_iptm", "model_chains"]
CANON = ["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"]

inp, out = sys.argv[1], sys.argv[2]
with open(inp) as fh:
    rd = csv.DictReader(fh)
    if rd.fieldnames != SRC:
        sys.exit(f"STOP: {inp} header is {rd.fieldnames}, expected {SRC}")
    rows = list(rd)

seen = set()
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=CANON)
    w.writeheader()
    for r in rows:
        key = (r["iface"], r["seed"], r["arm"], r["sample"])
        if key in seen:
            sys.exit(f"STOP: duplicate key {key} in {inp}")
        seen.add(key)
        for k in ("dockq", "pair_iptm"):
            v = float(r[k])
            if v != v or v in (float("inf"), float("-inf")):
                sys.exit(f"STOP: non-finite {k} for {key}; folder.open_csv would skip the row")
        w.writerow({"iface": r["iface"], "seed": r["seed"], "arm": r["arm"],
                    "donor_pose": "-1", "tmpl_chains": "",
                    "sample": r["sample"],
                    "dockq": f"{float(r['dockq']):.4f}",
                    "iptm": f"{float(r['pair_iptm']):.4f}"})
print(f"{len(rows)} row(s) -> {out}")
