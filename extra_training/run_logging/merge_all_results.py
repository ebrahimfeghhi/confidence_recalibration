#!/usr/bin/env python
"""One master CSV of every antibody-to-antigen pose result we hold, across all three datasets.

WHAT IS IN IT. One row per pose, for ANTIBODY-TO-ANTIGEN chain pairs only. Pairs that score an
antibody against itself (heavy vs light) or an antigen against another antigen are excluded --
every structure in all three datasets is an antibody-antigen complex, but not every pair of
chains inside one is the binding site we care about.

THREE DATASETS:
    pxm          PXM-2025
    foldbench    FoldBench
    sabdab       our SAbDab pull, split into train / val83 / sab98

COLUMNS
    dataset      pxm | foldbench | sabdab
    split        train | val83 | sab98 for sabdab; empty for the other two
    structure    PDB id
    interface    the chain pair's name as its source store writes it, <pdb>_<chains>
    seed         the fold seed. THIS is what picks the trunk.
    pose         diffusion sample index inside that seed, 0..4. Arbitrary but stable: it is the
                 source store's own `sample` number, i.e. the order the sampler emitted them.
                 Pose 0 of one seed has nothing in common with pose 0 of the next.
    dockq        DockQ of that pose at that chain pair
    iptm         chain-pair ipTM, max-pooled, for that chain pair and pose
    has_trunk    1 if a cached pair rep (trunk) exists for (structure, seed), else 0
    role_source  how the pair was established to be antibody-to-antigen; see below
    source       the file the row came from

has_trunk IS KEYED ON (structure, seed). One trunk per structure per seed, shared by all of
that structure's chain pairs and all of its poses -- which is exactly why ranking poses inside
one seed is hard. Built by scanning the trunk roots for trunks/<pdb>/free_<seed>.npz, so it
reports what is on disk rather than trusting a stored has_trunk column.

HOW A PAIR IS ESTABLISHED TO BE ANTIBODY-TO-ANTIGEN. This is the whole difficulty of the join,
because the letters in an interface name are NOT a single consistent chain-naming system.

  role_source = ab_ag_table   (pxm, foldbench)
      Their chain_roles.csv names ab_chain and ag_chain outright, and the result stores carry
      explicit chain1/chain2 columns, so the join is exact. All 292 PXM and all 160 FoldBench
      pairs match a roles row and all of them are antibody-to-antigen.

  role_source = model_chains  (sabdab, where available)
      The DockQ store records the MODEL chains it actually scored, e.g. 10fj_AB -> "C+A". Those
      resolve against chain_roles' model_chain column with no guessing. This is authoritative
      and is preferred wherever it exists: 10fj_AB reads as heavy-vs-antigen0 through it, while
      guessing from the letters A and B gives heavy-vs-light, which is wrong.

  role_source = unanimous_roles  (sabdab, the rest)
      Without a model_chains record the interface letters have to be resolved against
      chain_roles, which holds THREE naming systems (auth_chain, model_chain, ref_chain) whose
      letters overlap. Resolving under each system separately gives conflicting answers for 475
      of sabdab's chain pairs, so no single system can be trusted. A pair is therefore kept only
      when EVERY system that resolves it agrees it is antibody-to-antigen. Pairs where the
      systems disagree about that are dropped as ambiguous and counted, never guessed.

EXCLUSIONS are counted and printed by reason, so nothing disappears silently:
    not_abag     resolved, but heavy-vs-light or antigen-vs-antigen
    ambiguous    naming systems disagree on whether it is antibody-to-antigen
    no_roles     no roles row for the structure at all

SOURCES ARE READ-ONLY. Nothing here writes to a result store.
"""
import argparse
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # extra_training/
REPO = os.path.dirname(HERE)                                          # repo root

AB_ROLES = {"H", "L"}
NAMING = ("auth_chain", "model_chain", "ref_chain")

# dataset, split, results path, schema, roles path, roles kind
SOURCES = [
    ("pxm", "", f"{REPO}/experimental_data/protenix/pxm2025/pxm2025_baseline_results.csv",
     "baseline", f"{REPO}/experimental_data/protenix/pxm2025/chain_roles.csv", "ab_ag"),
    ("foldbench", "", f"{REPO}/experimental_data/protenix/foldbench/foldbench_baseline_results.csv",
     "baseline", f"{REPO}/experimental_data/protenix/foldbench/chain_roles.csv", "ab_ag"),
    ("foldbench", "extra",
     f"{REPO}/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds/results.csv",
     "arm", f"{REPO}/experimental_data/protenix/foldbench/chain_roles.csv", "ab_ag"),
    ("sabdab", "train", f"{HERE}/training/fold/results_with_trunks.csv",
     "arm", f"{HERE}/training/fold/chain_roles_all.csv", "hlag"),
    ("sabdab", "val83", f"{HERE}/validation/fold/results.csv",
     "arm", f"{HERE}/validation/fold/chain_roles_all83.csv", "hlag"),
    ("sabdab", "sab98", f"{HERE}/validation/fold_sab98/results.csv",
     "arm", f"{HERE}/validation/fold_sab98/chain_roles.csv", "hlag"),
]

# Stores that record the MODEL chains a DockQ score was computed on. Authoritative.
MODEL_CHAIN_STORES = [
    f"{HERE}/training/fold/poses_extra/dockq_extra.csv",
    f"{HERE}/training/refill/refill_rows.csv",
]

TRUNK_ROOTS = [
    f"{HERE}/training/fold/trunks",
    f"{HERE}/training/fold_regen/trunks",
    f"{HERE}/training/fold_deferred/trunks",
    f"{HERE}/validation/fold/trunks",
    f"{HERE}/validation/fold_sab98/trunks",
    f"{HERE}/validation/fold_trunkfix/trunks",
    f"{HERE}/validation/fold_deferred/trunks",
]

COLS = ["dataset", "split", "structure", "interface", "seed", "pose", "pose_set",
        "dockq", "iptm", "iptm_mean", "has_trunk", "role_source", "source"]

# The regenerated poses: SAME trunks, same seeds, fresh diffusion noise. DockQ and ipTM live in
# two separate files that have to be joined on (interface, seed, sample), and the DockQ scorer
# runs behind the generator, so at any moment some poses have an ipTM and no DockQ yet. Only
# fully scored poses are written -- an inner join, not a left join.
REGEN_DOCKQ = f"{HERE}/training/fold/poses_extra/dockq_extra.csv"
REGEN_IPTM = f"{HERE}/training/fold/poses_extra/results_extra.csv"


def trunk_index():
    """{(pdb, seed)} for every cached pair rep on disk."""
    have, per_root = set(), {}
    for root in TRUNK_ROOTS:
        if not os.path.isdir(root):
            per_root[root] = None
            continue
        n = 0
        for pdb in os.listdir(root):
            d = os.path.join(root, pdb)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".npz") and fn.startswith("free_"):
                    try:
                        have.add((pdb, int(fn[5:-4])))
                        n += 1
                    except ValueError:
                        pass
        per_root[root] = n
    return have, per_root


def load_model_chains():
    """{iface: "C+A"} -- the model chains a DockQ score was actually computed on."""
    m = {}
    for path in MODEL_CHAIN_STORES:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                iface, mc = (r.get("iface") or "").strip(), (r.get("model_chains") or "").strip()
                if iface and mc:
                    m.setdefault(iface, mc)
    return m


def load_ab_ag(path):
    """pxm / foldbench: {(pdb, frozenset{c1, c2}): (ab_chain, ag_chain)}.

    Also returns a per-pdb list of the chain pairs, so a store that carries no chain1/chain2
    columns (the foldbench top-up uses the arm schema) can still be resolved by matching the
    letters in its interface name against the pairs this structure actually has. Without that
    fallback the top-up's 4,550 poses were dropped wholesale as "no roles".
    """
    out, by_pdb = {}, collections.defaultdict(list)
    if not os.path.exists(path):
        return out, by_pdb
    with open(path) as fh:
        for r in csv.DictReader(fh):
            c1, c2 = (r.get("chain_1") or "").strip(), (r.get("chain_2") or "").strip()
            if not (c1 and c2):
                continue
            pdb = r["pdb"].strip()
            val = ((r.get("ab_chain") or "").strip(), (r.get("ag_chain") or "").strip())
            out[(pdb, frozenset((c1, c2)))] = val
            by_pdb[pdb].append((c1, c2, val))
    return out, by_pdb


def chains_from_iface(by_pdb, pdb, iface):
    """(c1, c2, (ab, ag)) for the roles pair whose letters match this interface name.

    Tries every split of the chain string and accepts a split only when it names a pair this
    structure actually has, in either order. Returns None when nothing matches, or when two
    different pairs both match -- an ambiguous name is not resolved by guessing.
    """
    if "_" not in iface:
        return None
    ch = iface.split("_", 1)[1]
    hits = []
    for c1, c2, val in by_pdb.get(pdb, []):
        if ch in (c1 + c2, c2 + c1):
            hits.append((c1, c2, val))
    return hits[0] if len(hits) == 1 else None


def load_hlag(path):
    """sabdab: one {pdb: {chain_letter: role}} per naming system, kept SEPARATE."""
    maps = {s: collections.defaultdict(dict) for s in NAMING}
    if not os.path.exists(path):
        return maps
    with open(path) as fh:
        for r in csv.DictReader(fh):
            pdb, role = (r.get("pdb") or "").strip(), (r.get("role") or "").strip()
            if not (pdb and role):
                continue
            for s in NAMING:
                v = (r.get(s) or "").strip()
                if v:
                    maps[s][pdb][v] = role
    return maps


def _is_abag(ra, rb):
    ag = lambda r: bool(r) and r.startswith("ag")
    return (ra in AB_ROLES and ag(rb)) or (ag(ra) and rb in AB_ROLES)


def classify_hlag(maps, model_chains, pdb, iface):
    """-> (keep, role_source|reason). Prefers the recorded model chains; else unanimity."""
    mc = model_chains.get(iface)
    if mc and "+" in mc:
        a, b = mc.split("+", 1)
        known = maps["model_chain"].get(pdb, {})
        if a in known and b in known:
            return _is_abag(known[a], known[b]), "model_chains" if _is_abag(
                known[a], known[b]) else "not_abag"

    if "_" not in iface:
        return False, "no_roles"
    ch = iface.split("_", 1)[1]
    verdicts = []
    for s in NAMING:
        known = maps[s].get(pdb, {})
        for i in range(1, len(ch)):
            a, b = ch[:i], ch[i:]
            if a in known and b in known:
                verdicts.append(_is_abag(known[a], known[b]))
                break
    if not verdicts:
        return False, "no_roles"
    if all(verdicts):
        return True, "unanimous_roles"
    if not any(verdicts):
        return False, "not_abag"
    return False, "ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/training/fold/abag_all_results.csv")
    ap.add_argument("--keep-non-abag", action="store_true",
                    help="also write pairs that are not antibody-to-antigen (default: drop)")
    a = ap.parse_args()

    have, per_root = trunk_index()
    print(f"cached trunks on disk: {len(have)} (structure, seed) pair(s)")
    for root, n in per_root.items():
        print(f"    {os.path.relpath(root, REPO)}: "
              f"{'MISSING' if n is None else str(n) + ' npz'}")

    model_chains = load_model_chains()
    print(f"interfaces with a recorded model-chain pair: {len(model_chains)}")

    rows, seen, report = [], set(), []
    for dataset, split, path, schema, roles_path, kind in SOURCES:
        if not os.path.exists(path):
            report.append((dataset, split, "FILE MISSING", 0, collections.Counter()))
            continue
        ab_ag, ab_ag_by_pdb = (load_ab_ag(roles_path) if kind == "ab_ag"
                               else ({}, collections.defaultdict(list)))
        maps = load_hlag(roles_path) if kind == "hlag" else {}
        kept, why = 0, collections.Counter()
        with open(path) as fh:
            rd = csv.DictReader(fh)
            fields = rd.fieldnames or []
            ipcol = "pair_iptm" if "pair_iptm" in fields else "iptm"
            for r in rd:
                if schema == "arm" and "arm" in fields and \
                        (r.get("arm") or "").strip() not in ("", "free"):
                    why["non_free_arm"] += 1
                    continue
                iface = (r.get("iface") or "").strip()
                if not iface:
                    why["bad_row"] += 1
                    continue
                pdb = (r.get("pdb") or iface.split("_")[0]).strip()
                try:
                    seed, pose = int(r["seed"]), int(r["sample"])
                    dockq, iptm = float(r["dockq"]), float(r[ipcol])
                except (KeyError, TypeError, ValueError):
                    why["bad_row"] += 1
                    continue

                if kind == "ab_ag":
                    c1, c2 = (r.get("chain1") or "").strip(), (r.get("chain2") or "").strip()
                    hit = ab_ag.get((pdb, frozenset((c1, c2)))) if (c1 and c2) else None
                    if hit is None:
                        # No chain columns (arm schema): match the interface name instead.
                        m = chains_from_iface(ab_ag_by_pdb, pdb, iface)
                        if m is not None:
                            c1, c2, hit = m
                    if hit is None:
                        ok, tag = False, "no_roles"
                    elif hit[0] and hit[1] and {hit[0], hit[1]} == {c1, c2}:
                        ok, tag = True, "ab_ag_table"
                    else:
                        ok, tag = False, "not_abag"
                else:
                    ok, tag = classify_hlag(maps, model_chains, pdb, iface)

                if not ok and not a.keep_non_abag:
                    why[tag] += 1
                    continue

                key = (dataset, split, iface, seed, pose)
                if key in seen:
                    why["duplicate"] += 1
                    continue
                seen.add(key)
                rows.append({
                    "dataset": dataset, "split": split, "structure": pdb,
                    "interface": iface, "seed": seed, "pose": pose,
                    "pose_set": "orig",
                    "dockq": f"{dockq:.6f}", "iptm": f"{iptm:.6f}",
                    "iptm_mean": "",
                    "has_trunk": 1 if (pdb, seed) in have else 0,
                    "role_source": tag,
                    "source": os.path.relpath(path, REPO),
                })
                kept += 1
        report.append((dataset, split, os.path.relpath(path, REPO), kept, why))

    # ---- the regenerated poses, joined across their two files ----------------------------
    if os.path.exists(REGEN_DOCKQ) and os.path.exists(REGEN_IPTM):
        maps = load_hlag(f"{HERE}/training/fold/chain_roles_all.csv")
        ipr = {}
        with open(REGEN_IPTM) as fh:
            for r in csv.DictReader(fh):
                try:
                    ipr[(r["iface"], int(r["seed"]), int(r["sample"]))] = r
                except (KeyError, TypeError, ValueError):
                    continue
        kept, why = 0, collections.Counter()
        with open(REGEN_DOCKQ) as fh:
            for r in csv.DictReader(fh):
                iface = (r.get("iface") or "").strip()
                if not iface:
                    why["bad_row"] += 1
                    continue
                try:
                    seed, pose = int(r["seed"]), int(r["sample"])
                    dockq = float(r["dockq"])
                except (KeyError, TypeError, ValueError):
                    why["bad_row"] += 1
                    continue
                hit = ipr.get((iface, seed, pose))
                if hit is None:
                    why["no_iptm_yet"] += 1
                    continue
                try:
                    iptm = float(hit["iptm"])
                    iptm_mean = float(hit["iptm_mean"])
                except (KeyError, TypeError, ValueError):
                    why["bad_row"] += 1
                    continue
                pdb = (r.get("pdb") or iface.split("_")[0]).strip()
                ok, tag = classify_hlag(maps, model_chains, pdb, iface)
                if not ok and not a.keep_non_abag:
                    why[tag] += 1
                    continue
                key = ("sabdab", "train", iface, seed, pose)
                if key in seen:
                    why["duplicate"] += 1
                    continue
                seen.add(key)
                rows.append({
                    "dataset": "sabdab", "split": "train", "structure": pdb,
                    "interface": iface, "seed": seed, "pose": pose,
                    "pose_set": "regen",
                    "dockq": f"{dockq:.6f}", "iptm": f"{iptm:.6f}",
                    "iptm_mean": f"{iptm_mean:.6f}",
                    "has_trunk": 1 if (pdb, seed) in have else 0,
                    "role_source": tag,
                    "source": os.path.relpath(REGEN_DOCKQ, REPO),
                })
                kept += 1
        report.append(("sabdab", "train+", os.path.relpath(REGEN_DOCKQ, REPO), kept, why))

    rows.sort(key=lambda d: (d["dataset"], d["split"], d["pose_set"],
                             d["interface"], d["seed"], d["pose"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)

    print("\nPOSES KEPT, AND WHY ROWS WERE DROPPED")
    print(f"{'dataset':<10} {'split':<7} {'kept':>8}   dropped")
    for dataset, split, path, kept, why in report:
        drop = "  ".join(f"{k}={v}" for k, v in sorted(why.items())) or "none"
        print(f"{dataset:<10} {split:<7} {kept:>8}   {drop}")

    print("\nANTIBODY-TO-ANTIGEN CHAIN PAIRS KEPT")
    print(f"{'dataset':<10} {'split':<7} {'set':<6} {'poses':>8} {'pairs':>7} "
          f"{'structs':>8} {'folds':>7} {'w/ trunk':>9} {'%':>6}")
    agg = collections.defaultdict(lambda: {"n": 0, "if": set(), "st": set(),
                                           "sd": set(), "tr": 0})
    for d in rows:
        k = (d["dataset"], d["split"], d["pose_set"])
        g = agg[k]
        g["n"] += 1
        g["if"].add(d["interface"])
        g["st"].add(d["structure"])
        g["sd"].add((d["structure"], d["seed"]))
        g["tr"] += int(d["has_trunk"])
    for k in sorted(agg):
        g = agg[k]
        pct = 100.0 * g["tr"] / g["n"] if g["n"] else 0.0
        print(f"{k[0]:<10} {k[1]:<7} {k[2]:<6} {g['n']:>8} {len(g['if']):>7} "
              f"{len(g['st']):>8} {len(g['sd']):>7} {g['tr']:>9} {pct:>5.1f}%")
    tot = sum(g["n"] for g in agg.values())
    tot_tr = sum(g["tr"] for g in agg.values())
    print(f"{'TOTAL':<10} {'':<7} {'':<6} {tot:>8} {'':>7} {'':>8} {'':>7} {tot_tr:>9} "
          f"{100.0 * tot_tr / tot if tot else 0:>5.1f}%")

    rs = collections.Counter(d["role_source"] for d in rows)
    print("\nhow each kept pose was established to be antibody-to-antigen:",
          "  ".join(f"{k}={v}" for k, v in rs.most_common()))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
