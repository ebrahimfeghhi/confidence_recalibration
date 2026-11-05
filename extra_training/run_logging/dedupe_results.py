#!/usr/bin/env python
"""Resolve duplicate (iface, seed, arm, sample) rows in a results.csv -- by evidence, not by order.

NEVER TOUCHES THE INPUT. Fold workers append to the live csv; rewriting it under them loses
whatever lands during the rewrite. This reports by default, writes a separate snapshot with --out,
and refuses if --out names the input. Replacing the canonical file is a separate, deliberate step
once every writer has exited (--replace, which takes a backup and swaps atomically).

MATCHING DockQ DOES NOT PROVE THE SAME POSE. Two folds of one seed can land on near-identical
coordinates and give DockQ equal to four decimals while differing in ipTM -- which is exactly what
the 10 duplicates here look like. So "the labels agree, keep either" is not a sound rule, and this
does not use it. Each duplicate is tested against the POSE STILL ON DISK:

  pose_sets.classify decides how many eligible saved runs the unit has.
    one complete run   -> there is a single retained pose set, and the row whose ipTM appears in
                          that run's own summary_confidence json is the one that describes it. If
                          exactly one row matches, it wins on evidence. If both match or neither
                          does, the identity is undecided and it is reported, not guessed.
    two complete runs  -> the rows may describe different folds and nothing here can tell which.
                          Unresolved.
    no complete run    -> the evidence has been deleted. Unresolved.

UNRESOLVED DUPLICATES ARE FATAL, not filtered. The exit status is nonzero and no snapshot is
written, because a snapshot that silently dropped or picked a contested label is worse than no
snapshot: everything downstream would treat it as settled.
"""
import argparse, collections, csv, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_sets                                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--poses", default="")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--out", default="", help="write a deduplicated snapshot here; never the input")
ap.add_argument("--replace", action="store_true",
                help="after a clean snapshot, back up the canonical csv and swap it in atomically. "
                     "Refuses while any fold session is writing.")
ap.add_argument("--tol", type=float, default=5e-5, help="float match tolerance for ipTM identity")
ap.add_argument("--show", type=int, default=20)
a = ap.parse_args()

if a.out and os.path.abspath(a.out) == os.path.abspath(a.csv):
    raise SystemExit("STOP: --out must not be the input; this never rewrites a live csv in place")
poses = a.poses or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "poses")

rows = list(csv.DictReader(open(a.csv)))
hdr = list(rows[0].keys()) if rows else []
KEY = ("iface", "seed", "arm", "sample")
g = collections.OrderedDict()
for r in rows:
    g.setdefault(tuple(r[k] for k in KEY), []).append(r)
dups = {k: v for k, v in g.items() if len(v) > 1}


def iptms_in_run(run_dir, sample):
    """every ipTM value the retained pose's own summary json carries, flattened"""
    out = set()
    for p in glob.glob(os.path.join(run_dir, "predictions",
                                    f"*_summary_confidence_sample_{sample}.json")):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        m = d.get(pose_sets.IPTM_KEY) or []
        for row in m:
            for v in (row if isinstance(row, list) else [row]):
                try:
                    out.add(float(v))
                except (TypeError, ValueError):
                    pass
        if "iptm" in d:
            try:
                out.add(float(d["iptm"]))
            except (TypeError, ValueError):
                pass
    return out


resolved, unresolved = {}, []
for k, v in dups.items():
    iface, seed, arm, sample = k
    pdb = iface.split("_")[0]
    st, run, detail = pose_sets.classify(poses, pdb, seed, a.nsample, arm=arm)
    if st != pose_sets.COMPLETE:
        unresolved.append((k, f"{st}: {detail}; cannot identify which fold these rows describe"))
        continue
    pool = iptms_in_run(run, sample)
    hits = []
    for r in v:
        try:
            ip = float(r["iptm"])
        except (TypeError, ValueError):
            continue
        # THE CSV STORES ipTM AT 4 DECIMALS while the json holds full precision (0.523106 against
        # 0.5231), so the comparison is made at the csv's own precision. Matching on raw floats
        # found 0 of 2 rows for every duplicate and looked like absent evidence rather than a
        # units mismatch on my side.
        if any(abs(ip - round(q, 4)) <= a.tol for q in pool):
            hits.append(r)
    if len(hits) == 1:
        resolved[k] = hits[0]
    else:
        dq = sorted({r["dockq"] for r in v})
        unresolved.append((k, f"{len(hits)} of {len(v)} row(s) match the retained run "
                              f"{os.path.basename(run)}; dockq {dq}, "
                              f"iptm {[r['iptm'] for r in v]}"))

print(f"{a.csv}: {len(rows)} row(s), {len(g)} distinct (iface, seed, arm, sample)")
print(f"  duplicated keys:                      {len(dups)}")
print(f"  resolved against the retained pose:   {len(resolved)}")
print(f"  UNRESOLVED:                           {len(unresolved)}")
for k, why in unresolved[:a.show]:
    print(f"     UNRESOLVED {k}: {why}")

if unresolved:
    print()
    print("No snapshot written. An unresolved duplicate means the evidence on disk cannot say "
          "which row describes the retained pose, and a snapshot that quietly picked one would be "
          "treated downstream as settled.")
    sys.exit(1)

if not a.out:
    print("report only; all duplicates are resolvable -- pass --out to write a snapshot")
    sys.exit(0)

keep = [resolved[k] if k in resolved else v[0] for k, v in g.items()]
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
tmp = a.out + ".part"
with open(tmp, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader(); w.writerows(keep)
os.replace(tmp, a.out)
print(f"wrote {a.out}: {len(keep)} row(s), {len(rows) - len(keep)} duplicate copy(ies) dropped")

if a.replace:
    live = [ln.split(":")[0] for ln in
            __import__("subprocess").run(["tmux", "ls"], capture_output=True,
                                         text=True).stdout.splitlines()
            if ln.startswith(("w1fold", "w1val", "w2_"))]
    if live:
        raise SystemExit(f"STOP: {live} are still writing. The snapshot is at {a.out}; "
                         f"re-run --replace once they have exited.")
    bak = a.csv + ".pre_dedupe"
    shutil.copy2(a.csv, bak)
    os.replace(a.out, a.csv)        # same filesystem, so this is atomic
    print(f"replaced {a.csv} atomically; previous content at {bak}")
