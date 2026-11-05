#!/usr/bin/env python
"""Does forcing OST's chain mapping recover the antibody-antigen interfaces it drops by itself?

THE QUESTION. 56 of the training pool's 1,711 interfaces have no DockQ row. For 20 of those
structures OST ran successfully and simply did not map the antigen chain: 5tl5's output carries
chain_mapping {'A':'B','B':'A'} with reference C absent, and chem_groups [['A','C'],['B']] -- two
reference chains in one chemical group against one model chain in that group, so the automatic
mapper can map only one of them. Review's point, and it is right: dockq_interfaces holds the
reference interfaces that COULD BE MAPPED, so a dropped interface is a statement about the mapper,
not about the interface. A wrong pose is supposed to score a low DockQ, not vanish.

WHAT THIS MEASURES. The same pose and the same reference, scored twice: once the way production
does it (OST picks the mapping) and once with the mapping forced from chain_roles.csv, which is
where this project's own reference -> model mapping lives. For every pool interface of that
structure it reports whether the interface appeared, its DockQ, the MODEL pair the json attributes
it to, and the ipTM read at that pair -- because a recovered interface scored against the wrong
model pair would be worse than a missing one.

IT CANNOT TOUCH PRODUCTION. --out-dir must be outside the poses tree. ost_dockq.run() reuses any
usable json already at its output path, so writing these probes into <run>/ost_json/ would make a
forced-mapping result the cached answer for a production rescore. Refused rather than documented.

Usage:
  python -I run_logging/compare_forced_chain_map.py --out-dir <scratch>/forcedmap \
      --case 5tl5:500000:0 --case 3rhw:500000:0 --case 4zto:500000:0 \
      --case 7uij:500007:3 --case 8w2e:500007:1
"""
import argparse, collections, csv, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "eval"))
import ost_dockq                                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--case", action="append", default=[], help="pdb:seed:sample")
ap.add_argument("--e", default=os.path.join(ROOT, "training/fold"))
ap.add_argument("--refs", default=os.path.join(ROOT, "training/refs"))
ap.add_argument("--timeout", type=int, default=600)
ap.add_argument("--rewrite-ref", action="store_true",
                help="also score against a gemmi-normalised copy of the reference. OST's chemical "
                     "grouping is computed from the reference, and for the 5tl5 class it claims "
                     "chains are equivalent whose sequences share nothing -- which points at the "
                     "staged instance cif's entity records rather than at OST. The copy keeps the "
                     "same label_asym_id set or the case is reported and skipped.")
a = ap.parse_args()

poses = os.path.join(a.e, "poses")
if os.path.abspath(a.out_dir).startswith(os.path.abspath(poses)):
    raise SystemExit(f"STOP: --out-dir is inside {poses}. ost_dockq.run() reuses a usable json at "
                     f"its output path, so a forced-mapping probe written there would become the "
                     f"cached answer for a production rescore.")
os.makedirs(a.out_dir, exist_ok=True)

pool = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(a.e, "pool.csv"))):
    pool[r["pdb"]].append((r["chain1"], r["chain2"]))
roles = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(a.e, "chain_roles*.csv"))):
    for r in csv.DictReader(open(f)):
        if r["ref_chain"]:
            roles[r["pdb"]][r["ref_chain"]] = r["model_chain"]


def _labels(path):
    """the label_asym_id set actually present in the atom records"""
    cols, out = [], set()
    for ln in open(path):
        if ln.startswith("_atom_site."):
            cols.append(ln.strip().split(".")[1])
        elif cols and (ln.startswith("ATOM") or ln.startswith("HETATM")):
            f = ln.split()
            try:
                out.add(f[cols.index("label_asym_id")])
            except (ValueError, IndexError):
                pass
    return out


def iptm_of(conf, model_pair):
    try:
        M = json.load(open(conf))["chain_pair_iptm"]
        i, j = (ost_dockq._chain_index(c) for c in model_pair)
        return round(float(M[i][j]), 4)
    except Exception:
        return None


def describe(out, conf, want):
    """out: {refpair -> (dockq, model pair, iptm)} for the pool pairs, plus what else was scored"""
    res = ost_dockq.parse(out, conf_json=None)
    got = {}
    for c1, c2 in want:
        v = ost_dockq.interface(res, c1, c2)
        if v is None:
            got[(c1, c2)] = None
            continue
        mp = tuple(v["model_chains"])
        got[(c1, c2)] = (v["dockq"], mp, iptm_of(conf, mp))
    extra = [k for k in res if tuple(k.split(",")) not in
             {tuple(sorted(p)) for p in want} | {tuple(p) for p in want} | {tuple(reversed(p)) for p in want}]
    return got, extra, res


print(f"out-dir {a.out_dir}")
for case in a.case:
    pdb, seed, sample = case.split(":")
    runs = sorted(glob.glob(os.path.join(poses, pdb, f"{seed}-*")))
    if len(runs) != 1:
        print(f"\n== {case}: {len(runs)} saved run(s) -- skipped"); continue
    run = runs[0]
    model = os.path.join(run, "predictions", f"{pdb}_sample_{sample}.cif")
    conf = os.path.join(run, "predictions", f"{pdb}_summary_confidence_sample_{sample}.json")
    ref = os.path.join(a.refs, f"{pdb}.cif")
    want = pool[pdb]
    cmap = [f"{rc}:{mc}" for rc, mc in sorted(roles[pdb].items())]
    print(f"\n== {pdb} seed {seed} sample {sample}")
    print(f"   pool interfaces: {want}")
    print(f"   forced map from chain_roles: {cmap}")
    if not (os.path.exists(model) and os.path.exists(ref)):
        print(f"   MISSING model or reference -- skipped"); continue

    modes = [("auto", None, ref), ("forced", cmap, ref)]
    if a.rewrite_ref:
        import gemmi
        st = gemmi.read_structure(ref)
        st.setup_entities()
        rr = os.path.join(a.out_dir, f"{pdb}_ref_rewritten.cif")
        st.make_mmcif_document().write_file(rr)
        before, after = _labels(ref), _labels(rr)
        if before != after:
            print(f"   rewritten reference changed the label chains {sorted(before)} -> "
                  f"{sorted(after)}; the pool names the old ones, so this probe is not comparable")
        else:
            modes.append(("rewritten", None, rr))

    rows = {}
    for mode, cm, refp in modes:
        out = os.path.join(a.out_dir, f"{pdb}_{seed}_{sample}_{mode}.json")
        ok = ost_dockq.run(model, refp, out, chain_map=cm, timeout=a.timeout)
        if not ok:
            print(f"   {mode:6}: OST produced no usable json")
            rows[mode] = ({}, [], {})
            continue
        rows[mode] = describe(out, conf, want)
        got, extra, res = rows[mode]
        print(f"   {mode:6}: {sum(1 for v in got.values() if v)}/{len(want)} pool interface(s), "
              f"{len(res)} interface(s) scored in total")
        for p, v in got.items():
            if v is None:
                print(f"            {p[0]},{p[1]}: ABSENT")
            else:
                print(f"            {p[0]},{p[1]}: dockq {v[0]}  model pair {v[1]}  iptm {v[2]}")
        if extra:
            print(f"            also scored (not in the pool): {extra}")

    # the retained production json, as a cross-check that 'auto' reproduces what was recorded
    kept = os.path.join(run, "ost_json", f"{sample}.json")
    if os.path.exists(kept):
        k = ost_dockq.parse(kept)
        same = set(k) == set(rows.get("auto", ({}, [], {}))[2])
        print(f"   retained production json: {sorted(k)}  (auto reproduces it: {same})")
