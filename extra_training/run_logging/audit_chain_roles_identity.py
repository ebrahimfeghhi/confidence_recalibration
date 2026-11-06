#!/usr/bin/env python
"""Does every pooled interface's REFERENCE chain actually exist in the MODEL?

WHY. chain_roles.csv records the chain correspondence INTENDED by job construction, one model chain
per reference chain in order. 9q5m shows that intent can be wrong: its reference holds three
372-residue chains (B, D, F) of one protein plus an 81-residue chain (E) of another, while the model
holds FOUR copies of E's protein at 532 residues and none of B/D/F. chain_roles still claims C<->B,
D<->D, F<->F. So five of its six pooled interfaces score the antibody against a protein that is not
in the model, and the only interface with a real counterpart -- E-H -- is the one the scorer refuses.

That is a DockQ filed under the wrong interface's name, which is the failure mode that matters: a
gap is visible, a mislabel trains.

WHAT IT MEASURES. For each structure, the total matching-block fraction of each reference chain
against each model chain, gaps allowed. A reference chain with no model chain above --min-id has no
counterpart, and every pooled interface naming it is unsupported. Reported per interface so the
output can be used directly as an exclusion list.

IT IS READ-ONLY and writes no exclusion itself: which interfaces to drop changes the training
population and is not this tool's decision.
"""
import argparse, collections, csv, difflib, glob, os, sys


def seqs(path):
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = {}
    for ch in st[0]:
        for res in ch:
            t = gemmi.find_tabulated_residue(res.name)
            c = (t.one_letter_code.upper() if t else "X")
            if c in "ACDEFGHIKLMNPQRSTVWY":
                out[res.subchain] = out.get(res.subchain, "") + c
    return {k: v for k, v in out.items() if v}


def frac(a, b):
    """total matching blocks over len(a) -- gaps allowed, so a partially resolved copy still scores."""
    sm = difflib.SequenceMatcher(None, a, b)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/combined/pool.csv")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--refs", default="training/refs")
    ap.add_argument("--min-id", type=float, default=0.80,
                    help="a reference chain needs some model chain at least this similar")
    ap.add_argument("--limit", type=int, default=0, help="first N structures, for a quick pass")
    ap.add_argument("--out", default="", help="csv of the unsupported interfaces")
    a = ap.parse_args()

    want = collections.defaultdict(set)
    for r in csv.DictReader(open(a.pool)):
        want[r["pdb"]].add("".join(sorted((r["chain1"], r["chain2"]))))
    pdbs = sorted(want)
    if a.limit:
        pdbs = pdbs[:a.limit]

    bad_if, no_pose, checked = [], [], 0
    for n, pdb in enumerate(pdbs, 1):
        rp = os.path.join(a.refs, f"{pdb}.cif")
        mp = sorted(glob.glob(os.path.join(a.fold, "poses", pdb, "*", "predictions",
                                           f"{pdb}_sample_0.cif")))
        if not (os.path.exists(rp) and mp):
            no_pose.append(pdb)
            continue
        try:
            r, m = seqs(rp), seqs(mp[0])
        except Exception as e:
            no_pose.append(f"{pdb} ({type(e).__name__})")
            continue
        checked += 1
        best = {}
        for rk, rs in r.items():
            best[rk] = max(((frac(rs, ms), mk) for mk, ms in m.items()), default=(0.0, None))
        unsupported = {rk for rk, (f, _mk) in best.items() if f < a.min_id}
        for cc in sorted(want[pdb]):
            touched = {cc[0], cc[1]} & unsupported
            if touched:
                bad_if.append((f"{pdb}_{cc}", sorted(touched),
                               {t: round(best[t][0], 2) for t in sorted(touched) if t in best}))
        if n % 100 == 0:
            print(f"  [{n}/{len(pdbs)}] {len(bad_if)} unsupported interface(s) so far", flush=True)

    print(f"\nchecked {checked} structure(s); {len(no_pose)} without a readable ref+pose")
    print(f"UNSUPPORTED interfaces (a reference chain with no model counterpart "
          f"at >= {a.min_id}): {len(bad_if)}")
    byp = collections.Counter(i.split("_")[0] for i, _t, _b in bad_if)
    print(f"  over {len(byp)} structure(s); worst: {byp.most_common(8)}")
    for i, t, b in bad_if[:25]:
        print(f"    {i}: reference chain(s) {t} best model match {b}")
    if len(bad_if) > 25:
        print(f"    ... and {len(bad_if) - 25} more")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["iface", "unsupported_ref_chains", "best_model_match"])
            for i, t, b in bad_if:
                w.writerow([i, "+".join(t), ";".join(f"{k}={v}" for k, v in b.items())])
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
