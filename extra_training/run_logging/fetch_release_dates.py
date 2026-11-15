#!/usr/bin/env python
"""Initial release date per PDB id, from RCSB, cached to a csv.

WHY NOT THE LOCAL INDEX. training/prep/all857_indices.csv and job_indices.csv carry
release_date = 9999-12-31 for every row -- a placeholder, not a date -- and indices.csv covers only
the original 500 structures. Counting pre- vs post-cutoff off those files silently reports
"everything is post-cutoff", which is how the composition writeup came to say so.
"""
import argparse
import csv
import json
import os
import time
import urllib.request

API = "https://data.rcsb.org/rest/v1/core/entry/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="file with one pdb id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.05)
    a = ap.parse_args()

    want = [l.strip().lower() for l in open(a.ids) if l.strip() and l[0] != "#"]
    have = {}
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            have[r["pdb"]] = r["release_date"]
    todo = [p for p in want if p not in have]
    print(f"{len(want)} id(s), {len(have)} cached, {len(todo)} to fetch")

    new = 0
    for i, p in enumerate(todo, 1):
        try:
            with urllib.request.urlopen(API + p, timeout=20) as fh:
                d = json.load(fh)
            have[p] = d["rcsb_accession_info"]["initial_release_date"][:10]
            new += 1
        except Exception as e:                       # noqa: BLE001 -- recorded, not raised
            have[p] = ""
            print(f"  {p}: {type(e).__name__}")
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(a.sleep)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "release_date"])
        for p in sorted(have):
            w.writerow([p, have[p]])
    miss = sum(1 for v in have.values() if not v)
    print(f"wrote {a.out}: {len(have)} row(s), {new} new, {miss} without a date")


if __name__ == "__main__":
    main()
