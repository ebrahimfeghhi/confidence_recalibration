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

sys.path.insert(0, "/home/ebrahim/confidence_recalibration/eval")
import ost_dockq                                                           # noqa: E402

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
ap.add_argument("--pool", default="",
                help="pool.csv, which knows each interface's reference chain pair")
ap.add_argument("--roles", nargs="*", default=[],
                help="chain_roles csv(s) mapping reference chains to model chains")
ap.add_argument("--tol", type=float, default=5e-5, help="float match tolerance for ipTM identity")
ap.add_argument("--show", type=int, default=20)
a = ap.parse_args()

if a.out and os.path.abspath(a.out) == os.path.abspath(a.csv):
    raise SystemExit("STOP: --out must not be the input; this never rewrites a live csv in place")
d0 = os.path.dirname(os.path.abspath(a.csv))
poses = a.poses or os.path.join(d0, "poses")
a.pool = a.pool or os.path.join(d0, "pool.csv")
a.roles = a.roles or sorted(glob.glob(os.path.join(d0, "chain_roles*.csv")))
if not a.roles:
    raise SystemExit(f"STOP: no chain_roles csv found beside {a.csv}; without the reference -> "
                     f"model chain mapping an interface's own pair ipTM cannot be identified")

rows = list(csv.DictReader(open(a.csv)))
hdr = list(rows[0].keys()) if rows else []
KEY = ("iface", "seed", "arm", "sample")
g = collections.OrderedDict()
for r in rows:
    g.setdefault(tuple(r[k] for k in KEY), []).append(r)
dups = {k: v for k, v in g.items() if len(v) > 1}


def load_pairs(pool_csv):
    """{iface -> (ref chain 1, ref chain 2)}, so an interface is never split out of its own name.

    iface is written as <pdb>_<c1><c2> and reference labels are not all single characters -- 8z7j
    has S1 and S2 -- so parsing the name back into two chains is ambiguous. The pool is the source
    that knows the pair.
    """
    out = {}
    for r in csv.DictReader(open(pool_csv)):
        out[f"{r['pdb']}_{r['chain1']}{r['chain2']}"] = (r["chain1"], r["chain2"])
    return out


def load_roles(paths):
    """{(pdb, ref chain) -> model chain}, from phase_g's own role tables"""
    out = {}
    for f in paths:
        for r in csv.DictReader(open(f)):
            if r.get("ref_chain"):
                out[(r["pdb"], r["ref_chain"])] = r["model_chain"]
    return out


def pair_iptm(run_dir, sample, pdb, refs, roles):
    """out: (value, "") -- the ipTM of THIS interface's chain pair in the retained pose, or (None, why)

    WHY NOT ANY VALUE IN THE MATRIX. Raised in review, and confirmed: matching a row's ipTM against
    every cell of chain_pair_iptm can match a DIFFERENT interface's pair and be mistaken for proof
    that the row describes the retained pose. 7t86 holds EI and FI at 0.5231 and 0.5250; 8w2e holds
    BC at 0.4209 and AC at 0.6320. A flattened search "resolved" all ten duplicates here, two of
    them against the wrong pair.

    AND WHY A MISMATCH IS STILL NOT PROOF OF A BAD ROW. ost_dockq.parse reads ipTM at
    chain_pair_iptm[i][j] where i and j come from OST's OWN model_chains for that interface -- its
    automatic chain mapper's answer, which can differ from the reference -> model mapping in the
    role table. That OST output json is not retained next to the poses, so when the role table's
    pair disagrees with every duplicate row, the cause cannot be distinguished from here: either a
    row carries another interface's value, or OST mapped the model differently when it scored.
    Both readings are reported as UNRESOLVED and neither row is chosen. That is why replacing the
    canonical csv stays blocked while any duplicate is unresolved -- the evidence on disk is not
    sufficient, and the honest output is a refusal rather than a snapshot.
    """
    mc = [roles.get((pdb, c)) for c in refs]
    if not all(mc):
        missing = [c for c, m in zip(refs, mc) if not m]
        return None, f"no model chain for reference {missing} in the role tables"
    idx = [ost_dockq._chain_index(m) for m in mc]
    paths = glob.glob(os.path.join(run_dir, "predictions",
                                   f"*_summary_confidence_sample_{sample}.json"))
    if not paths:
        return None, f"no summary json for sample {sample}"
    try:
        d = json.load(open(paths[0]))
    except (OSError, ValueError) as e:
        return None, f"unreadable summary json ({type(e).__name__})"
    M = d.get(pose_sets.IPTM_KEY)
    if not isinstance(M, list):
        return None, f"no {pose_sets.IPTM_KEY}"
    i, j = idx
    if not (0 <= i < len(M) and isinstance(M[i], list) and 0 <= j < len(M[i])):
        return None, f"model pair {mc} -> index ({i},{j}) outside a {len(M)}x" \
                     f"{len(M[0]) if M and isinstance(M[0], list) else '?'} matrix"
    try:
        return float(M[i][j]), ""
    except (TypeError, ValueError):
        return None, f"{pose_sets.IPTM_KEY}[{i}][{j}] is not a number"


pairs = load_pairs(a.pool)
roles = load_roles(a.roles)
print(f"  interface pairs from {a.pool}: {len(pairs)}; role mappings: {len(roles)}")

resolved, unresolved = {}, []
for k, v in dups.items():
    iface, seed, arm, sample = k
    pdb = iface.split("_")[0]
    refs = pairs.get(iface)
    if not refs:
        unresolved.append((k, f"{iface} is not in {a.pool}, so its chain pair is unknown"))
        continue
    st, run, detail = pose_sets.classify(poses, pdb, seed, a.nsample, arm=arm)
    if st != pose_sets.COMPLETE:
        unresolved.append((k, f"{st}: {detail}; cannot identify which fold these rows describe"))
        continue
    want, why = pair_iptm(run, sample, pdb, refs, roles)
    if want is None:
        unresolved.append((k, f"{why}; cannot check identity against {os.path.basename(run)}"))
        continue
    # THE CSV STORES ipTM AT 4 DECIMALS while the json holds full precision (0.523106 against
    # 0.5231), so the comparison is made at the csv's own precision.
    hits = []
    for r in v:
        try:
            ip = float(r["iptm"])
        except (TypeError, ValueError):
            continue
        if abs(ip - round(want, 4)) <= a.tol:
            hits.append(r)
    if len(hits) == 1:
        resolved[k] = hits[0]
    else:
        allv = set()
        try:
            d = json.load(open(glob.glob(os.path.join(
                run, "predictions", f"*_summary_confidence_sample_{sample}.json"))[0]))
            allv = {round(float(x), 4) for row_ in (d.get(pose_sets.IPTM_KEY) or [])
                    for x in (row_ if isinstance(row_, list) else [row_])}
        except Exception:
            pass
        elsewhere = [r["iptm"] for r in v if float(r["iptm"]) in allv]
        unresolved.append((k, f"{len(hits)} of {len(v)} row(s) match this interface's own pair "
                              f"ipTM {round(want, 4)} in {os.path.basename(run)}; rows carry "
                              f"{[r['iptm'] for r in v]}"
                              + (f" -- {elsewhere} match a DIFFERENT chain pair in the same pose, "
                                 f"so either a row holds another interface's value or OST mapped "
                                 f"the model differently when it scored" if elsewhere else "")))

print(f"{a.csv}: {len(rows)} row(s), {len(g)} distinct (iface, seed, arm, sample)")
print(f"  duplicated keys:                      {len(dups)}")
print(f"  resolved against the retained pose:   {len(resolved)}")
print(f"  UNRESOLVED:                           {len(unresolved)}")
for k, why in unresolved[:a.show]:
    print(f"     UNRESOLVED {k}:\n        {why}")

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
