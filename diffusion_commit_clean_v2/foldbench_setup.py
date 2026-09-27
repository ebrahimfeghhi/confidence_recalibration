#!/usr/bin/env python
"""Build FoldBench's inputs once, so the pipeline runs on it with --dataset foldbench and nothing else.

SUMMARY
    The pipeline was written on pxm2025, whose free pool, pool csv, crystal chain maps and role file sit
    in experimental_data/. FoldBench has the poses already -- a 50-seed Protenix run (seeds 101-150,
    paths.POSES_FOLDBENCH) -- and the benchmark's interfaces and roles (paths.TARGETS_CSV). This writes
    them in the pipeline's shapes under paths.FOLDBENCH_DATA, scoring every pool pose with OUR scorer:

      free/<pdb>/<seed>-legacy-no-provenance  a symlink to the run's <pdb>/seed_<seed>, whose inner layout
                                              (predictions/<pdb>_sample_N.cif + summary jsons) is already
                                              the pool's. EVERY seed on disk is linked, not only the pool's
                                              50: selection is capped at the pool seeds by the orchestrator,
                                              and seeds past them must stay visible so a free top-up never
                                              reuses a seed number
      ost_json/<pdb>/<seed>_<sample>.json     OST's output per pool pose (eval/ost_dockq.py, the scorer
                                              every FoldBench fold is scored with)
      pool.csv        pdb, chain1, chain2, seed, sample, dockq, pair_iptm -- from those, per interface
      chain_maps.csv  pdb, seed, sample, chain_mapping {reference chain: pose chain}, OST's own mapping
      chain_roles.csv interface, pdb, chain_1, chain_2, ab_chain, ag_chain -- the benchmark's roles

    WHY RESCORE rather than use the run's OST table (paths.FOLDBENCH_OST_CSV): for ~30 complexes that
    table's seeds 101-127 no longer match the poses on disk (those folds were re-run after it was made),
    so its DockQ and ipTM describe poses that no longer exist. It is kept only as a cross-check.

    REFERENCE COPIES. Some references hold the complex twice ("A", "A-2"); OST lays the model over the
    copy that fits best, and ost_dockq.interface() reads the interface from whichever copy that is. The
    chain map records a pose mapped onto copy N under the first copy's names ("A-2" -> "A") when the
    first copy is unmapped: the copies are the same molecules, and the map is read for roles only.

    EXCLUDED, and why, all written to setup_audit.md:
      a complex whose interface names a copy chain itself (8im0: A-B-2) -- interface ids are
        <pdb>_<C1><C2>, one letter per chain
      a complex any of whose interfaces is scored on no pose at all: 8f60 (its antigen never maps) and
        8ts0 (OST's chain mapper crashes on its predictions, which carry three one-residue chains; the
        run's own evaluation rewrote each model's entity block first, eval_ost_foldbench.postprocess_cif,
        which our scorer does not)
      a complex with no poses on disk (8evd)
      a complex with a protein entity of fewer than 4 standard residues (7yv1: a cyclic peptide with 9 of
        11 residues modified; 8b1a: a dipeptide) -- build_templates.pick_chains needs a pose chain of at
        least 4 standard residues for EVERY protein entity, so every donor of such a complex fails.
        Read from the job json, not the crystal. Neither entity is in a benchmark interface
      a complex whose role file makes two copies of one molecule the antibody and the antigen (8dtn: its
        only listed interface is nanobody A with nanobody B, identical chains) -- a role belongs to the
        molecule (anchor_relabel.every_copy), so such a labelling has no consistent reading
      a complex where a labelled chain lands on different molecules in different poses (8a67: its
        antigen is laid over a 76-residue ubiquitin in 166 poses and a 72-residue one in 84) -- which
        molecule is the antigen would then change from pose to pose. A pose chain's letter names the
        same molecule in every pose of a complex (Protenix names chains in input order), so one pose's
        sequences decide it

    THE AUDIT: every pool pose resolves through free/ with its json; every row's pair_iptm equals the
    pose's chain_pair_iptm read by an independent code path (pool_index.pair_iptm_at: gemmi chain order,
    not OST's chain names); and the rows are compared with the run's table, seeds 101-127 and 128-150
    separately.

INPUTS
    --jobs:        OST processes in parallel. Default 16. Each is pinned by ost_dockq (OST_TASKSET).
    --stop-after:  "link" or "score" to stop early. Default run everything.

OUTPUTS
    paths.FOLDBENCH_DATA/{free/, ost_json/, pool.csv, chain_maps.csv, chain_roles.csv, setup_audit.md}
    stdout: counts per step and the audit

EXAMPLE
    foldbench_setup.py --jobs 16

FUNCTIONS
    short_entity(): a complex -> the standard-residue count of its shortest protein entity
    interfaces():  the role file -> the benchmark's interfaces per complex, and complexes excluded
    link_free():   the free pool as symlinks into the 50-seed run
    score_pose():  one pool pose -> its OST result per interface and its chain mapping (resumable)
    first_copy():  a {reference: model} map with copy-N names folded onto the first copy's
    same_molecule(): one pose, two of its chains -> whether they are copies of one molecule
    molecules():   one pose -> {chain: its residue names}, which identifies the molecule
    write_csv():   write a csv, refusing to replace a different one
    iptm_check():  one pool row -> |pair_iptm - the pose's chain_pair_iptm|, read independently
    main():        link, score, write, audit
"""
import argparse
import collections
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import paths  # noqa: E402
import pool_index  # noqa: E402

sys.path.insert(0, str(paths.EVAL_DIR))
import ost_dockq  # noqa: E402

DS = paths.DATASETS["foldbench"]
OST_DIR = DS.data / "ost_json"
SAMPLES = range(5)
TOL_IPTM = 1e-4       # both paths read the same float from the same json


def short_entity(pdb):
    """in: a complex.  out: the standard-residue count of its shortest protein entity, from the job json"""
    job = json.load(open(f"{DS.jobs}/{pdb}-final-updated.json"))
    job = job[0] if isinstance(job, list) else job
    counts = []
    for e in job["sequences"]:
        if "proteinChain" in e:
            mod = {m["ptmPosition"] for m in e["proteinChain"].get("modifications") or []}
            seq = e["proteinChain"]["sequence"]
            counts.append(sum(1 for i, a in enumerate(seq, 1) if a != "X" and i not in mod))
    return min(counts)


def interfaces():
    """in: nothing.  out: ({pdb: [(chain1, chain2, ab chain, ag chain)]}, {pdb: reason} excluded)"""
    out, excluded = collections.defaultdict(list), {}
    for r in csv.DictReader(open(paths.TARGETS_CSV)):
        pdb = r["pdb_id"].split("-")[0]
        c = (r["interface_chain_id_1"], r["interface_chain_id_2"], r["antibody_chain"],
             r["antigen_chain"])
        if any(len(x) != 1 for x in c):
            excluded[pdb] = f"interface {c[0]}-{c[1]} names a multi-letter chain"
        out[pdb].append(c)
    for pdb in list(out):
        if pdb in excluded or not os.path.isdir(f"{paths.POSES_FOLDBENCH}/{pdb}"):
            excluded.setdefault(pdb, "no poses on disk")
            del out[pdb]
        elif short_entity(pdb) < 4:
            excluded[pdb] = (f"a protein entity has {short_entity(pdb)} standard residues; the template "
                             f"builder needs >= 4 for every entity")
            del out[pdb]
    return dict(out), excluded


def link_free(pdbs):
    """in: the complexes.  out: (links made, links already correct); raises on a wrong link"""
    made = same = 0
    for pdb in pdbs:
        src_dir = f"{paths.POSES_FOLDBENCH}/{pdb}"
        for d in sorted(os.listdir(src_dir)):
            if not d.startswith("seed_"):
                continue
            src = f"{src_dir}/{d}"
            dst = f"{DS.free}/{pdb}/{d[len('seed_'):]}-{paths.LEGACY_RUN}"
            if os.path.islink(dst):
                if os.readlink(dst) != src:
                    raise SystemExit(f"{dst} points at {os.readlink(dst)}, not {src}")
                same += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
            made += 1
    return made, same


def first_copy(cmap):
    """in: OST's {reference chain: model chain}.  out: the same, "<X>-<n>" renamed "<X>" where X itself
    is unmapped -- a pose laid over the reference's second copy, named as the first copy"""
    out = {}
    for ref, mdl in cmap.items():
        head, _, tail = ref.rpartition("-")
        key = head if head and tail.isdigit() and head not in cmap else ref
        if key in out:
            raise ValueError(f"two reference chains fold onto {key}: {cmap}")
        out[key] = mdl
    return out


def score_pose(task):
    """in: (pdb, seed, sample, [(chain1, chain2)]).
    out: (pdb, seed, sample, {(chain1, chain2): (dockq, pair_iptm, model chains)}, chain map, error)"""
    pdb, seed, sample, pairs = task
    try:
        cif = paths.pool_pose(pdb, seed, sample, str(DS.free))
        conf = cif.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json")
        out = f"{OST_DIR}/{pdb}/{seed}_{sample}.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        res = ost_dockq.score(cif, DS.ref(pdb), out, conf_json=conf)
        if not res:
            return pdb, seed, sample, {}, {}, "OST produced no result"
        got = {}
        for c1, c2 in pairs:
            v = ost_dockq.interface(res, c1, c2)
            if v and v.get("dockq") is not None and v.get("pair_iptm") is not None:
                got[(c1, c2)] = (float(v["dockq"]), float(v["pair_iptm"]), tuple(v["model_chains"]))
        return pdb, seed, sample, got, first_copy(json.load(open(out))["chain_mapping"]), ""
    except Exception as e:
        return pdb, seed, sample, {}, {}, f"{type(e).__name__}: {e}"


def molecules(pdb, seed, sample):
    """in: a pool pose.  out: {chain: tuple of residue names} -- equal tuples are copies of one molecule"""
    import gemmi
    model = gemmi.read_structure(paths.pool_pose(pdb, seed, sample, str(DS.free)))[0]
    return {c.name: tuple(r.name for r in c) for c in model}


def same_molecule(pdb, seed, sample, c1, c2):
    """in: a pool pose and two of its chains.  out: whether their residue sequences are identical"""
    m = molecules(pdb, seed, sample)
    return m[c1] == m[c2]


def write_csv(path, header, rows):
    """in: a path, the header, the rows.  out: "written" or "unchanged"; raises if a different file is
    already there -- these are inputs, and a silent change under a finished run is how results drift"""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    text = buf.getvalue()
    if os.path.exists(path):
        if open(path).read() == text:
            return "unchanged"
        raise SystemExit(f"{path} exists and differs; move it aside to rebuild")
    tmp = f"{path}.tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, path)
    return "written"


def iptm_check(task):
    """in: (pdb, seed, sample, model chain 1, model chain 2, pair_iptm).
    out: |difference| from pool_index.pair_iptm_at, or the error as a string"""
    pdb, seed, sample, m1, m2, want = task
    try:
        return abs(pool_index.pair_iptm_at(paths.pool_pose(pdb, seed, sample, str(DS.free)), m1, m2)
                   - want)
    except Exception as e:
        return f"{pdb} {seed} {sample} {m1}{m2}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--stop-after", choices=("link", "score"), default=None)
    A = ap.parse_args()
    os.makedirs(DS.data, exist_ok=True)
    import multiprocessing as mp

    ifs, excluded = interfaces()
    made, same = link_free(sorted(ifs))
    print(f"  free/: {made} seed links made, {same} already there -> {DS.free}", flush=True)
    if A.stop_after == "link":
        return 0

    # every pool pose scored once; OST reports all of its interfaces in one call
    tasks = [(p, s, x, [(c1, c2) for c1, c2, _, _ in ifs[p]])
             for p in sorted(ifs) for s in DS.pool_seeds for x in SAMPLES]
    scored, errors = {}, []
    with mp.Pool(A.jobs) as pool:
        for i, (p, s, x, got, cmap, err) in enumerate(pool.imap_unordered(score_pose, tasks, 4), 1):
            if err:
                errors.append(f"{p} {s} {x}: {err}")
            scored[(p, s, x)] = (got, cmap)
            if i % 1000 == 0 or i == len(tasks):
                print(f"    scored {i}/{len(tasks)}  errors {len(errors)}", flush=True)
    if A.stop_after == "score":
        return 0

    # a complex whose role file names two copies of one molecule as antibody and antigen
    for p in sorted(ifs):
        k = next((k for k in scored if k[0] == p and scored[k][1]), None)
        if k is None:
            continue                     # no pose mapped at all: excluded just below
        cmap = scored[k][1]
        both = [f"{c1}-{c2}" for c1, c2, ab, ag in ifs[p]
                if ab in cmap and ag in cmap and same_molecule(p, k[1], k[2], cmap[ab], cmap[ag])]
        if both:
            excluded[p] = (f"interface {', '.join(both)}: antibody and antigen chains are copies of one "
                           f"molecule")
            del ifs[p]

    # a complex where a labelled chain lands on different molecules in different poses
    for p in sorted(ifs):
        k = next((k for k in scored if k[0] == p and scored[k][1]), None)
        if k is None:
            continue
        mol = molecules(*k)
        split = sorted({c for c1, c2, ab, ag in ifs[p] for c in (ab, ag)
                        if len({mol[cmap[c]] for (q, _, _), (_, cmap) in scored.items()
                                if q == p and c in cmap}) > 1})
        if split:
            excluded[p] = f"labelled chain {', '.join(split)} lands on different molecules across poses"
            del ifs[p]

    # a complex with an interface no pose scores cannot be evaluated
    n_rows = collections.Counter((p, c1, c2) for (p, _, _), (got, _) in scored.items() for c1, c2 in got)
    for p in sorted(ifs):
        dead = [f"{c1}-{c2}" for c1, c2, _, _ in ifs[p] if not n_rows[(p, c1, c2)]]
        if dead:
            excluded[p] = f"interface {', '.join(dead)} scored on no pose"
            del ifs[p]
    pool_rows = [[p, c1, c2, s, x, f"{dq:.4f}", f"{ip:.6f}"]
                 for (p, s, x), (got, _) in sorted(scored.items()) if p in ifs
                 for (c1, c2), (dq, ip, _) in sorted(got.items())]
    map_rows = [[p, s, x, json.dumps(cmap, sort_keys=True)]
                for (p, s, x), (_, cmap) in sorted(scored.items()) if p in ifs and cmap]
    role_rows = [[f"{p}_{c1}{c2}", p, c1, c2, ab, ag] for p in sorted(ifs) for c1, c2, ab, ag in ifs[p]]
    header = ["pdb", "chain1", "chain2", "seed", "sample", "dockq", "pair_iptm"]
    print(f"  pool.csv: {write_csv(DS.pool_csv, header, pool_rows)} ({len(pool_rows)} rows)")
    print(f"  chain_maps.csv: {write_csv(DS.chain_maps, ['pdb', 'seed', 'sample', 'chain_mapping'], map_rows)}")
    print(f"  chain_roles.csv: {write_csv(DS.chain_roles, ['interface', 'pdb', 'chain_1', 'chain_2', 'ab_chain', 'ag_chain'], role_rows)}")

    # AUDIT 1: poses and maps
    n_pose = len(ifs) * len(DS.pool_seeds) * len(SAMPLES)
    unmapped = [k for k, (_, cmap) in scored.items() if k[0] in ifs and not cmap]
    roles_missing = [(p, s, x) for (p, s, x), (_, cmap) in scored.items() if p in ifs
                     and not {c for c1, c2, ab, ag in ifs[p] for c in (ab, ag)} <= set(cmap)]
    # AUDIT 2: ipTM through an independent code path
    with mp.Pool(A.jobs) as pool:
        diffs = pool.map(iptm_check, [(p, s, x, *m, ip) for (p, s, x), (got, _) in scored.items()
                                      if p in ifs for (dq, ip, m) in got.values()], 64)
    bad = [d for d in diffs if isinstance(d, str) or d > TOL_IPTM]
    # AUDIT 3: the run's table, where its poses still exist
    table = {(r["assembly_id"].split("-")[0], r["chain1"], r["chain2"], r["seed"], r["sample"]):
             (float(r["dockq"]), float(r["pair_iptm"])) for r in csv.DictReader(open(paths.FOLDBENCH_OST_CSV))
             if r["dockq_status"] == "ok"}
    agree = collections.Counter()
    for p, c1, c2, s, x, dq, ip in pool_rows:
        t = table.get((p, c1, c2, s, str(x)))
        if t:
            half = "seeds 101-127" if int(s) <= 127 else "seeds 128-150"
            same_ = abs(t[0] - float(dq)) <= 0.001 and abs(t[1] - float(ip)) <= TOL_IPTM
            agree[(half, same_)] += 1

    lines = ["# FoldBench setup audit", "",
             f"poses `{paths.POSES_FOLDBENCH}`, seeds {DS.pool_seeds[0]}-{DS.pool_seeds[-1]}, scored "
             f"with `eval/ost_dockq.py` into `ost_json/`", "",
             f"- kept: {len(ifs)} complexes, {len(role_rows)} interfaces, {len(pool_rows)} pool rows",
             *[f"- excluded {p}: {why}" for p, why in sorted(excluded.items())],
             f"- free/: {made} seed links made, {same} already present",
             f"- OST errors: {len(errors)}" + (f", e.g. {errors[:3]}" if errors else ""), "",
             "## Audit", "",
             f"- poses scored: {n_pose - len(unmapped)} of {n_pose} with a chain map; "
             f"{len(roles_missing)} whose map lacks a role chain",
             f"- ipTM: {len(diffs) - len(bad)} of {len(diffs)} rows equal the pose's chain_pair_iptm read "
             f"by pool_index.pair_iptm_at (tolerance {TOL_IPTM})" + (f"; e.g. {bad[:3]}" if bad else ""),
             "- against the run's table (`FOLDBENCH_OST_CSV`), rows present in both:", "",
             "| seeds | same DockQ and ipTM | different |", "|:---|---:|---:|",
             *[f"| {h} | {agree[(h, True)]} | {agree[(h, False)]} |" for h in ("seeds 101-127", "seeds 128-150")],
             ""]
    open(f"{DS.data}/setup_audit.md", "w").write("\n".join(lines))
    print("\n".join(lines[4:]))
    kept_errors = [e for e in errors if e.split()[0] in ifs]     # an excluded complex's are its reason
    return 0 if not (kept_errors or unmapped or roles_missing or bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
