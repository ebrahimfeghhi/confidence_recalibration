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
import argparse, collections, csv, fcntl, glob, json, os, shutil, sys

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

# --replace TAKES THE WRITER LOCK BEFORE IT READS, AND HOLDS IT THROUGH THE SWAP. Raised in
# review, and it is a real gap: the tmux scan further down only sees FOLD sessions, while
# phase_i_check's rescore is an ordinary process that announces itself by holding
# <csv>.rescore.lock. Without taking that lock, a rescore could append recovered rows after this
# snapshot was read and before os.replace -- which is atomic as a filesystem operation and does
# nothing whatever to prevent losing those rows. Reporting does not take the lock, because reading
# a csv a rescore is appending to costs nothing but a slightly stale report.
_LOCKS = []
if a.replace:
    _lp = a.csv + ".rescore.lock"
    _lk = open(_lp, "a+")
    try:
        fcntl.flock(_lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lk.seek(0)
        raise SystemExit(f"STOP: a rescore holds {_lp} (pid {_lk.read().strip() or 'unknown'}). "
                         f"It may append rows after this snapshot is read, and they would be lost "
                         f"by the swap. Wait for it to finish, then re-run --replace.")
    _LOCKS.append(_lk)

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


WHICH = {}   # (run, sample, refs) -> (model pair, which mapping said so)


def _ost_pair(run_dir, sample, refs):
    """out: ([model chain, model chain], "ost chain_mapping") -- the pair OST used, or (None, why)

    This is the mapping the scorer itself recorded when it produced the row, read from the json the
    fold keeps beside the poses. A .invalid sibling marks a failed run and carries no mapping.
    """
    p = os.path.join(run_dir, "ost_json", f"{sample}.json")
    try:
        cm = json.load(open(p)).get("chain_mapping")
    except (OSError, ValueError):
        return None, f"no readable {os.path.basename(p)}"
    if not isinstance(cm, dict):
        return None, f"{os.path.basename(p)} has no chain_mapping"
    mc = [cm.get(c) for c in refs]
    if not all(mc):
        absent = [c for c, m in zip(refs, mc) if not m]
        return None, f"ost left reference chain {absent} unmapped in this pose"
    return mc, "ost chain_mapping"


def pair_iptm(run_dir, sample, pdb, refs, roles):
    """out: (value, "") -- the ipTM of THIS interface's chain pair in the retained pose, or (None, why)

    WHY NOT ANY VALUE IN THE MATRIX. Raised in review, and confirmed: matching a row's ipTM against
    every cell of chain_pair_iptm can match a DIFFERENT interface's pair and be mistaken for proof
    that the row describes the retained pose. 7t86 holds EI and FI at 0.5231 and 0.5250; 8w2e holds
    BC at 0.4209 and AC at 0.6320. A flattened search "resolved" all ten duplicates here, two of
    them against the wrong pair.

    THE ROLE TABLE IS THE FALLBACK, NOT THE AUTHORITY. ost_dockq.parse reads ipTM at
    chain_pair_iptm[i][j] with i and j taken from OST's OWN mapping for that interface -- its
    automatic chain mapper's answer, which can differ from the reference -> model mapping in
    chain_roles. An earlier version of this file said that mapping was not retained and therefore
    reported such a disagreement as undecidable. It IS retained: the fold writes OST's output to
    <run>/ost_json/<sample>.json, chain_mapping included. Both duplicates that were unresolved
    under the role table resolve under it, and they resolve to the OTHER row:

        7uij_DF sample 3  roles -> (D,F)=0.1487   ost -> (B,C)=0.1481  (rows 0.1482 / 0.1481)
        8w2e_BC sample 1  roles -> (B,C)=0.4209   ost -> (C,B)=0.6320  (rows 0.6273 / 0.6320)

    So the mapping used to IDENTIFY a recorded row has to be the one the scorer used when it wrote
    it. chain_roles is consulted only when that json is missing, and the reported reason says which
    mapping was used so a disagreement is never silent.
    """
    mc, src = _ost_pair(run_dir, sample, refs)
    if mc is None:
        mc = [roles.get((pdb, c)) for c in refs]
        src = "chain_roles (no retained ost json for this sample)"
        if not all(mc):
            missing = [c for c, m in zip(refs, mc) if not m]
            return None, f"no model chain for reference {missing} in the role tables"
    idx = [ost_dockq._chain_index(m) for m in mc]
    WHICH[(run_dir, sample, tuple(refs))] = (tuple(mc), src)
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
        mcs, src = WHICH.get((run, sample, tuple(refs)), (refs, "?"))
        unresolved.append((k, f"{len(hits)} of {len(v)} row(s) match this interface's own pair "
                              f"{list(refs)} -> {list(mcs)} per {src}, "
                              f"ipTM {round(want, 4)} in {os.path.basename(run)}; rows carry "
                              f"{[r['iptm'] for r in v]}"
                              + (f" -- {elsewhere} match a DIFFERENT chain pair in the same pose, "
                                 f"so either a row holds another interface's value or OST mapped "
                                 f"the model differently when it scored" if elsewhere else "")))

print(f"{a.csv}: {len(rows)} row(s), {len(g)} distinct (iface, seed, arm, sample)")
print(f"  duplicated keys:                      {len(dups)}")
print(f"  resolved against the retained pose:   {len(resolved)}")
_by_src = collections.Counter(src for _k, (_mc, src) in WHICH.items())
for _src, _n in sorted(_by_src.items()):
    print(f"        identified by {_src}: {_n}")
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
    # THE SESSION CHECK ASKS WHETHER A FOLD COULD WRITE *THIS* CSV, not whether any fold exists.
    # It used to refuse on w1fold, w1val or w2_ whatever file was being replaced, which is a proxy
    # for "a fold is writing this csv" and too coarse to be right: the validation fold is launched
    # with --csv <...>/validation/fold/results.csv and cannot touch the training one, so a live
    # w1val blocked every repair of training/fold/results.csv for the hours it ran. Session names
    # are the only evidence available here -- reading another user's process cmdline is not
    # allowed on this box -- and the prefixes are the convention this guard already relied on, so
    # the fix is to match the prefix to the fold the csv belongs to and keep refusing on anything
    # whose fold cannot be determined.
    #
    # The two load-bearing protections are unchanged and below: the per-csv <csv>.rescore.lock,
    # and the row-count identity check that refuses if anything appended while the snapshot was
    # being built. This check is a third layer, and a third layer that fires on the wrong file is
    # not caution, it is a false alarm that invites someone to bypass all three.
    WRITERS = {"training": ("w1fold", "w2_"), "validation": ("w1val",)}
    parts = os.path.abspath(a.csv).split(os.sep)
    fold = next((p for p in parts if p in WRITERS), None)
    if fold is None:
        prefixes = tuple(p for v in WRITERS.values() for p in v)
        print(f"NOTE: {a.csv} is not under a known fold root, so every fold session counts as a "
              f"possible writer of it.")
    else:
        prefixes = WRITERS[fold]
    live = [ln.split(":")[0] for ln in
            __import__("subprocess").run(["tmux", "ls"], capture_output=True,
                                         text=True).stdout.splitlines()
            if ln.startswith(prefixes)]
    if live:
        raise SystemExit(f"STOP: {live} could still be writing {a.csv}. The snapshot is at "
                         f"{a.out}; re-run --replace once they have exited.")
    other = [ln.split(":")[0] for ln in
             __import__("subprocess").run(["tmux", "ls"], capture_output=True,
                                          text=True).stdout.splitlines()
             if ln.startswith(tuple(p for v in WRITERS.values() for p in v))
             and not ln.startswith(prefixes)]
    if other:
        print(f"NOTE: {other} are running but write a different fold's csv, so they do not block "
              f"this replace. The lock and the row-count check below still apply.")
    # AND THE FILE MUST STILL BE THE ONE THAT WAS READ. The lock keeps a rescore out, but it is
    # not proof on its own: a writer that predates the lock, or one that ignores it, would show up
    # here as extra rows. Counting them is cheap and turns a silent loss into a refusal.
    now = sum(1 for _ in csv.DictReader(open(a.csv)))
    if now != len(rows):
        raise SystemExit(f"STOP: {a.csv} held {len(rows)} data row(s) when this snapshot was built "
                         f"and holds {now} now, so something appended to it meanwhile. The "
                         f"snapshot is at {a.out} and would lose those rows; nothing replaced.")
    bak = a.csv + ".pre_dedupe"
    shutil.copy2(a.csv, bak)
    os.replace(a.out, a.csv)        # same filesystem, so this is atomic
    print(f"replaced {a.csv} atomically; previous content at {bak}")
