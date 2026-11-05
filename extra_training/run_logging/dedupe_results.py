#!/usr/bin/env python
"""Report or snapshot a results.csv with duplicate (iface, seed, arm, sample) rows collapsed.

WHY NOT EDIT THE LIVE CSV. Fold workers append to it; rewriting it under them loses whatever lands
during the rewrite. So this never modifies the input. Default is a report; --out writes a separate
deduplicated snapshot for downstream analysis, and the canonical file should only be replaced
atomically once every writer has stopped.

WHAT COUNTS AS A CONFLICT. DockQ is the label; two rows for one pose that disagree on it are not
interchangeable and must not be silently collapsed to whichever came first. ipTM is read from the
pose's own summary_confidence json and carries small run-to-run numerical noise, so a difference
there within --iptm-tol is reported but not treated as a conflict. Measured on the 10 duplicates in
the training csv: DockQ identical in all 10, ipTM differing by at most 0.003 -- consistent with the
unit having been folded twice during the window when two generations of workers shared the cards.

Conflicting duplicates are REJECTED, not resolved: they are dropped from the snapshot and listed,
because choosing one would silently pick a label.
"""
import argparse, collections, csv, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="", help="write a deduplicated snapshot here; never the input")
ap.add_argument("--iptm-tol", type=float, default=0.01)
ap.add_argument("--show", type=int, default=20)
a = ap.parse_args()

if a.out and os.path.abspath(a.out) == os.path.abspath(a.csv):
    raise SystemExit("STOP: --out must not be the input; this never rewrites a live csv")

rows = list(csv.DictReader(open(a.csv)))
hdr = list(rows[0].keys()) if rows else []
KEY = ("iface", "seed", "arm", "sample")
g = collections.OrderedDict()
for r in rows:
    g.setdefault(tuple(r[k] for k in KEY), []).append(r)

dups = {k: v for k, v in g.items() if len(v) > 1}
collapsed, conflicts, noisy = [], [], []
for k, v in dups.items():
    dq = {r["dockq"] for r in v}
    if len(dq) > 1:
        conflicts.append((k, sorted(dq))); continue
    try:
        ips = [float(r["iptm"]) for r in v]
        spread = max(ips) - min(ips)
    except (TypeError, ValueError):
        spread = None
    if spread is not None and spread > a.iptm_tol:
        conflicts.append((k, [f"iptm spread {spread:.4f} > {a.iptm_tol}"])); continue
    if spread:
        noisy.append((k, spread))
    collapsed.append(k)

print(f"{a.csv}: {len(rows)} row(s), {len(g)} distinct (iface, seed, arm, sample)")
print(f"  duplicated keys:            {len(dups)}")
print(f"  collapsible (DockQ agrees): {len(collapsed)}")
print(f"  CONFLICTING (rejected):     {len(conflicts)}")
if noisy:
    worst = max(s for _k, s in noisy)
    print(f"  of the collapsible, {len(noisy)} differ on ipTM only, worst {worst:.4f}")
for k, why in conflicts[:a.show]:
    print(f"     CONFLICT {k}: {why}")

if not a.out:
    print("report only; pass --out to write a deduplicated snapshot")
    sys.exit(0)

rej = {k for k, _w in conflicts}
keep = []
for k, v in g.items():
    if k in rej:
        continue
    keep.append(v[0])          # DockQ agrees, so the first row carries the label
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader(); w.writerows(keep)
print(f"wrote {a.out}: {len(keep)} row(s) "
      f"({len(rows) - len(keep)} removed: {sum(len(v) - 1 for k, v in dups.items() if k not in rej)} "
      f"duplicate copies, {sum(len(g[k]) for k in rej)} rows of {len(rej)} conflicting key(s))")
