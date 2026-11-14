"""PHASE K -- score a confidence head on THE validation metric, the one the project is judged on.

WHAT IS COMPUTED. For each validation interface, select the pose with the highest PER-INTERFACE
pair_iptm over a seed budget, then report

    mean_dockq  -- mean over interfaces of the SELECTED pose's DockQ
    frac_acc    -- fraction of interfaces whose selected pose clears DockQ 0.23 (CAPRI acceptable)

Baseline to beat, measured 2026-10-15 over 92 interfaces at 100 seeds with the SHIPPED head:
mean_dockq 0.5775, frac_acc 0.7283, against an oracle of 0.7171 / 0.8696. The 0.1396 gap is what a
better ranker wins at zero extra compute.

NO VALIDATION LABELS ARE NEEDED. The metric is built from ipTM and DockQ, not from the FT3 loss, so
nothing here reads a label npz. That is why validation label-building is off the critical path.

THREE JOINS, AND EVERY ONE OF THEM CAN BE WRONG SILENTLY:

  1. interface name -> chain pair. results.csv names interfaces by REFERENCE chain (29tj_AG), while
     chain_pair_iptm is indexed by the model's asym ids. chain_roles carries both, so the join is
     ref chain -> model chain -> asym index, model chains being lettered A, B, C... in asym order.
  2. pose -> sample. The cached poses are NOT in sample order -- measured here on 29tj/400000, whose
     pose 0 reproduces the stored ipTM of SAMPLE 4. Selecting a pose and then reading another
     sample's DockQ would silently score the wrong thing, so the join is by COORDINATE IDENTITY
     against the `_sample_<k>.cif` files, via build_cache's own sample_indices.
  3. (interface, seed, sample) -> DockQ, from results.csv.

SO THE SELF-CHECK IS NOT OPTIONAL. With the SHIPPED head and no checkpoint, the recomputed pair_iptm
must reproduce the `iptm` already in results.csv, row by row, to within --iptm-tol. If it does not,
one of the three joins is wrong and no number from this script means anything. --no-self-check
exists only for a checkpoint whose head has moved, where disagreement is the point.
"""
import argparse
import collections
import csv
import functools
import glob
import hashlib
import json
import math
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
print = functools.partial(print, flush=True)
ACCEPTABLE = 0.23


def interface_chain_pairs(pool_csv, roles_csv):
    """{pdb: [(iface_id, asym_i, asym_j)]} -- join 1, resolved once up front.

    Model chains are lettered A, B, C... in asym_id order by write_job_cif, so the letter's position
    in the alphabet IS its asym index.
    """
    roles = collections.defaultdict(dict)       # pdb -> ref_chain -> model_chain
    with open(roles_csv) as fh:
        for r in csv.DictReader(fh):
            roles[r["pdb"]][r["ref_chain"]] = r["model_chain"]
    out = collections.defaultdict(list)
    unresolved = []
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            pdb, c1, c2 = r["pdb"], r["chain1"], r["chain2"]
            m1, m2 = roles[pdb].get(c1), roles[pdb].get(c2)
            if m1 is None or m2 is None:
                unresolved.append(f"{pdb}:{c1}/{c2}")
                continue
            out[pdb].append((f"{pdb}_{c1}{c2}", ord(m1) - ord("A"), ord(m2) - ord("A")))
    return out, unresolved


def pool_ref_pairs(pool_csv):
    """{pdb: [(iface_id, ref_chain_1, ref_chain_2)]} -- the pool, in REFERENCE chain letters.

    The interface id is built from the reference letters because that is what results.csv's `iface`
    column holds; the MODEL chains those letters correspond to is a separate question, answered per
    pose by ost_chain_pairs.
    """
    out = collections.defaultdict(list)
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            pdb, c1, c2 = r["pdb"], r["chain1"], r["chain2"]
            out[pdb].append((f"{pdb}_{c1}{c2}", c1, c2))
    return out


def _cpi_index(asym_vec):
    """{original asym id: row in chain_pair_iptm}, replicating Protenix's compaction.

    sample_confidence.py remaps asym ids to 0..N-1 whenever `len(unique) != max+1`, which is
    exactly when a crop has removed a whole chain. Building the mapping from sorted unique ids
    unconditionally gives the identity in the ordinary contiguous case, so this is safe to apply
    always rather than only when a chain is missing -- and applying it always is what keeps the
    cropped and uncropped paths reading the same code.
    """
    import numpy as _np
    v = asym_vec.detach().cpu().numpy() if hasattr(asym_vec, "detach") else _np.asarray(asym_vec)
    return {int(o): i for i, o in enumerate(_np.unique(v.reshape(-1)))}


def ost_chain_pairs(unit_dir, pdb, k, refs):
    """Join 1 FOR ONE POSE: [(iface_id, asym_i, asym_j)] from OST's own chain mapping.

    WHY THIS IS PER POSE AND NOT A STATIC TABLE. chain_roles.csv assigns each model chain one
    reference chain for the whole structure. That is unanswerable when two reference chains are
    chemically identical: 9eg6 and 7zra -- the only two validation structures with more than one
    antigen chain -- each carry two antigen copies with IDENTICAL sequences, so which model chain is
    "the first antigen" is not determined by sequence. OST decides it per pose, by whichever
    assignment scores best, and writes the decision to ost_json/<k>.json as `chain_mapping`.

    Measured on 2026-10-16: chain_roles is consistently swapped against OST for 9eg6 (samples 0, 1
    and 2 alike) and swapped for only SOME poses of 7zra (sample 0 yes, samples 1 and 2 no). The
    static table therefore cannot be repaired by correcting it -- for 7zra there is no single correct
    answer. Using the static table made the evaluator read one interface's ipTM against the other
    interface's DockQ on 1,014 of 46,000 rows, which is what GATE 1 refused.

    DIRECTION: OST's chain_mapping is REFERENCE -> MODEL. Established on 7r58, whose three chains are
    chemically distinct so its mapping is unambiguous: OST's {'A':'C','B':'A','C':'B'} inverts to
    exactly chain_roles' model->ref {'A':'B','B':'C','C':'A'}.

    Model chains are lettered A, B, C... in asym_id order by write_job_cif, so a letter's position in
    the alphabet IS its asym index.
    """
    f = os.path.join(unit_dir, "ost_json", f"{k}.json")
    if not os.path.exists(f):
        return None, f"no {os.path.basename(f)}"
    with open(f) as fh:
        j = json.load(fh)
    # THE INDEX IS ASSERTED, NOT ASSUMED. ost_json/<k>.json is paired with _sample_<k>.cif by
    # filename only; reading the wrong pose's mapping would reintroduce exactly the defect this
    # function exists to fix, and silently.
    mdl = str(j.get("model", ""))
    if not mdl.endswith(f"_sample_{k}.cif"):
        return None, f"{os.path.basename(f)} describes {os.path.basename(mdl)}, not sample {k}"
    cm = j.get("chain_mapping")
    if not isinstance(cm, dict):
        return None, f"{os.path.basename(f)} has no chain_mapping"

    # A REFERENCE CHAIN OST DID NOT MAP IS RESOLVED BY ELIMINATION, OR NOT AT ALL.
    #
    # OST omits a chain it could not place -- 7shy's reference chain A is absent from chain_mapping
    # on every pose, which is the same OST behaviour that dropped 7shy_AH from the scores
    # (ost/mol/alg/scoring.py:1121). Falling back to chain_roles there is precisely the table this
    # function exists to distrust, and 7shy is NOT a safe case for it: OST's own chem_mapping puts
    # its model chains A and B in ONE group, so they are interchangeable and a static assignment of
    # them is a guess.
    #
    # It is still determined, though, without consulting that table. 7shy has three reference chains
    # and three model chains, and chain_mapping accounts for two of each, so exactly one of each is
    # left and the assignment is FORCED. That is the only inference allowed here: one unassigned
    # reference chain and one unassigned model chain. Two of either is genuinely ambiguous and is
    # refused, because a wrong chain pair reads one interface's ipTM against another's DockQ, and
    # with --ckpt the shipped-head self-check does not run, so nothing downstream would catch it.
    ref_all = [c for c in (j.get("reference_chains") or []) if c]
    mdl_all = [c for c in (j.get("model_chains") or []) if c]
    need = {c for _i, c1, c2 in refs for c in (c1, c2)}
    if not need <= set(cm):
        free_r = [c for c in ref_all if c not in cm]
        free_m = [c for c in mdl_all if c not in set(cm.values())]
        if len(free_r) == 1 and len(free_m) == 1:
            cm = {**cm, free_r[0]: free_m[0]}
        else:
            return None, (f"reference chain(s) {sorted(need - set(cm))} unmapped by OST and not "
                          f"forced: {len(free_r)} reference and {len(free_m)} model chain(s) free "
                          f"({free_r} / {free_m})")

    pairs = []
    for iface, c1, c2 in refs:
        m1, m2 = cm.get(c1), cm.get(c2)
        if not m1 or not m2 or len(m1) != 1 or len(m2) != 1:
            return None, f"{iface}: no single model chain for {c1}/{c2}"
        pairs.append((iface, ord(m1) - ord("A"), ord(m2) - ord("A")))
    return pairs, ""


_SEED_MANIFEST = {}          # path -> {pdb: (seed, ...)}
_SEED_MANIFEST_META = {}     # path -> {algo, digest, k}


def intended_seeds(a, pdb):
    """The seed ids a "budget N" label CLAIMS for this structure, as a tuple.

    Default is the contiguous window [seed_base, seed_base + budget), which is what an untrimmed
    fold wrote. A TRIMMED fold is the case that broke this guard: the valfull pool's 53 structures
    keep 30 seeds each, chosen out of the 100-wide sweep 400000-400099 by anchored-coverage-v2, so
    the kept ids are SPARSE -- 29tj keeps 400000, 400002, 400004, 400005, 400011, ... up to 400092.
    Against such a fold the contiguous window names seeds that were never meant to exist, and the
    guard below called them missing trunks: 321 of that pool's 1590 units, every one a seed the trim
    deliberately dropped rather than a gap in the data. 50 of the 53 keep a non-contiguous set, so
    the sparse case is the rule and the contiguous one the exception. (The fold holds 83 structures
    in all; the other 30 are a separate wave with its own manifest, so no single file covers it.)

    --seed-manifest makes the kept set authoritative WITHOUT weakening the assertion. The manifest
    must name this structure, must list exactly `budget` seeds for it, must agree with its own
    declared `k`, and every seed must fall inside its own declared seed_range. Each failure is a
    hard stop, because a partial or stale manifest would redefine what the metric covers while its
    label stayed the same -- the very substitution the original guard exists to prevent.

    NOTHING BUDGET-DEPENDENT IS CACHED. The file's shape and its own range are validated once per
    path; every claim that mentions --budget is re-checked on each call, so two budgets in one
    process cannot inherit each other's verdict.
    """
    if not getattr(a, "seed_manifest", ""):
        return tuple(range(a.seed_base, a.seed_base + a.budget))
    path = a.seed_manifest
    if path not in _SEED_MANIFEST:
        try:
            with open(path) as fh:
                man = json.load(fh)
        except (ValueError, OSError) as _e:
            raise SystemExit(f"STOP: --seed-manifest {path} is unreadable ({type(_e).__name__}: "
                             f"{_e}). Without it the intended seed set is undefined.")
        # EVERY SHAPE ERROR GETS ITS OWN STOP. Unguarded .get()/int()/[] on a hand-written file
        # raised AttributeError or IndexError instead, which on the sharded path surfaces as a
        # traceback buried in shard<i>.log rather than a sentence saying what is wrong.
        if not isinstance(man, dict):
            raise SystemExit(f"STOP: --seed-manifest {path} holds a {type(man).__name__} at the "
                             f"top level, not an object with a 'structures' key.")
        _st = man.get("structures")
        if not isinstance(_st, dict):
            raise SystemExit(f"STOP: --seed-manifest {path} has 'structures' as a "
                             f"{type(_st).__name__}, not an object keyed by structure id.")
        # THE DECLARED RANGE OR NO RANGE AT ALL. Inventing [seed_base, seed_base + budget - 1] as a
        # fallback rejected correctly-trimmed manifests for exactly the seeds the trim KEPT, and
        # blamed the manifest for a range it never declared.
        _rng, lo, hi = man.get("seed_range"), None, None
        if _rng is not None:
            try:
                lo, hi = int(_rng[0]), int(_rng[1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise SystemExit(f"STOP: --seed-manifest {path} declares seed_range {_rng!r}, "
                                 f"which is not a [lo, hi] pair of integers.")
            if lo > hi:
                raise SystemExit(f"STOP: --seed-manifest {path} declares seed_range [{lo}, {hi}], "
                                 f"which is empty.")
        got = {}
        for _p, _d in _st.items():
            if not isinstance(_d, dict) or not isinstance(_d.get("seeds"), (list, tuple)):
                raise SystemExit(f"STOP: --seed-manifest {path} entry for {_p} is a "
                                 f"{type(_d).__name__} with no 'seeds' list. The shape is "
                                 f"{{structures: {{<pdb>: {{seeds: [...]}}}}}}.")
            try:
                sds = tuple(int(s) for s in _d["seeds"])
            except (TypeError, ValueError):
                raise SystemExit(f"STOP: --seed-manifest {path} lists a non-integer seed for {_p}: "
                                 f"{list(_d['seeds'])[:4]!r}.")
            if not sds:
                raise SystemExit(f"STOP: --seed-manifest {path} lists no seeds for {_p}.")
            if len(set(sds)) != len(sds):
                raise SystemExit(f"STOP: --seed-manifest {path} lists a duplicate seed for {_p}. "
                                 f"A duplicated seed would be weighted twice in the selection.")
            if lo is not None:
                _bad = [s for s in sds if not lo <= s <= hi]
                if _bad:
                    raise SystemExit(f"STOP: --seed-manifest {path} lists seed(s) {_bad[:4]} for "
                                     f"{_p} outside its own seed_range [{lo}, {hi}].")
            got[_p] = sds
        if not got:
            raise SystemExit(f"STOP: --seed-manifest {path} names no structures, so it defines no "
                             f"seed set.")
        _SEED_MANIFEST[path] = got
        _SEED_MANIFEST_META[path] = {"algo": str(man.get("algo") or ""),
                                     "digest": str(man.get("digest") or ""),
                                     "k": man.get("k"),
                                     "seed_range": [lo, hi] if lo is not None else None}
        print(f"intended seed set: {path} (algo {man.get('algo')}, digest {man.get('digest')}, "
              f"k={man.get('k')}, range {lo}-{hi}) over {len(got)} structure(s) -- the TRIMMED "
              f"set, not the contiguous window")
    got, meta = _SEED_MANIFEST[path], _SEED_MANIFEST_META[path]
    # AND ITS OWN k MUST AGREE WITH THE LABEL. Checking only len(seeds) == budget let a manifest
    # declaring k=99 pass at --budget 1 while every structure carried one seed: self-inconsistent,
    # and the inconsistency is the signal that the file is not the one this budget means.
    if meta["k"] is not None:
        try:
            _k = int(meta["k"])
        except (TypeError, ValueError):
            _k = None
        if _k is not None and _k != a.budget:
            raise SystemExit(f"STOP: --seed-manifest {path} declares k={_k} seeds per structure "
                             f"but --budget is {a.budget}. One of the two is from a different "
                             f"pass; the label would not describe the file.")
    if pdb not in got:
        raise SystemExit(f"STOP: --seed-manifest {path} does not name structure {pdb}, so the "
                         f"seeds its 'budget {a.budget}' label claims are undefined. A manifest "
                         f"covering only part of the pool would silently shrink the metric.")
    if len(got[pdb]) != a.budget:
        raise SystemExit(f"STOP: --seed-manifest {path} keeps {len(got[pdb])} seed(s) for {pdb} "
                         f"but --budget is {a.budget}. The label would claim a depth the manifest "
                         f"does not carry.")
    return got[pdb]


def merge_rows(a):
    """The aggregate half of the evaluator, over the UNION of several shards' rows. No GPU, no model.

    A shard cannot judge coverage -- it holds a slice by construction -- so the self-check, the
    coverage contract and the metric all wait until the shards' rows are back together. Reading them
    here rather than recomputing anything means the merged result is exactly what an unsharded run
    would have produced, which is asserted by phase_k_eval_sharded.py comparing the two.
    """
    paths = [x.strip() for x in a.rows_in.split(",") if x.strip()]
    missing = [x for x in paths if not os.path.exists(x)]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} shard row file(s) absent: {missing}. A shard that "
                         f"died would otherwise silently shrink the metric's coverage.")
    rows, selfchk, metas = [], [], []
    for x in paths:
        mp = x + ".meta.json"
        if not os.path.exists(mp):
            raise SystemExit(f"STOP: {x} has no {os.path.basename(mp)}, so its shard cannot be "
                             f"confirmed to have finished rather than been truncated.")
        metas.append(json.load(open(mp)))
        with open(x) as fh:
            for r in csv.DictReader(fh):
                v = float(r["iptm"])
                dq = float(r["dockq"]) if r["dockq"] != "" else None
                vm = (float(r["iptm_mean"])
                      if r.get("iptm_mean") not in (None, "") else None)
                rows.append((r["iface"], int(r["seed"]), int(r["sample"]), v, dq, vm))
                if r["stored_iptm"] != "":
                    sv = float(r["stored_iptm"])
                    selfchk.append((abs(v - sv), r["iface"], int(r["seed"]), int(r["sample"]),
                                    v, sv))
    # EVERY SHARD OF THE SAME RUN, AND ALL OF THEM. A merge over 2 of 3 shards, or over shards from
    # two different budgets or checkpoints, would produce a number whose label is a lie.
    ns = {m["nshards"] for m in metas}
    if len(ns) != 1:
        raise SystemExit(f"STOP: the shards disagree on --nshards: {sorted(ns)}")
    n = ns.pop()
    have = sorted(m["shard"] for m in metas)
    if have != list(range(n)):
        raise SystemExit(f"STOP: shards {have} of {n} present; the metric would cover a subset.")
    bad = [(m["shard"], m["n_unmapped"]) for m in metas if m.get("n_unmapped")]
    if bad and not a.drop_incomplete_ifaces:
        raise SystemExit(f"STOP: shard(s) {bad} wrote rows but left pose(s) unmapped, so the merge "
                         f"would cover fewer poses than its budget claims. Resolve the chain "
                         f"mapping, exclude those structures from the pool, or pass "
                         f"--drop-incomplete-ifaces to report over the interfaces that ARE "
                         f"complete and have the dropped ones named in the metric.")
    for field in ("budget", "ckpt"):
        vals = {m[field] for m in metas}
        if len(vals) != 1:
            raise SystemExit(f"STOP: the shards disagree on {field}: {sorted(vals)}")
    if metas[0]["budget"] != a.budget:
        raise SystemExit(f"STOP: the shards were run at budget {metas[0]['budget']} but this merge "
                         f"says --budget {a.budget}; the label would not match the data.")
    a.ckpt = metas[0]["ckpt"]
    n_ost = sum(m["n_ost"] for m in metas)
    n_differ = sum(m["n_differ"] for m in metas)
    differ = collections.Counter()
    for m in metas:
        differ.update(m.get("differ_pdbs") or {})
    print(f"merged {len(paths)} shard(s): {sum(m['units'] for m in metas)} unit(s), "
          f"{len(rows)} row(s)")
    print(f"join 1: OST's per-pose chain mapping used on {n_ost} pose(s), unresolvable on 0")
    if n_differ:
        print(f"  it DISAGREED with chain_roles on {n_differ} pose(s), across {len(differ)} "
              f"structure(s): " + ", ".join(f"{k}:{v}" for k, v in differ.most_common(8)))
    want_ifaces = {i for v in pool_ref_pairs(a.pool).values() for i, _c1, _c2 in v}
    if a.drop_incomplete_ifaces:
        # COUNTED AGAINST THE INTENDED DEPTH, not against the modal one. The intended key set is
        # interfaces x seeds(budget from --seed-base) x range(--nsample), so budget*nsample is
        # what a complete interface has; inferring the expected depth from the data would let a
        # uniformly short run look complete.
        _exp = a.budget * a.nsample
        _cnt = collections.Counter(r[0] for r in rows)
        _drop = sorted(i for i in want_ifaces if _cnt.get(i, 0) != _exp)
        if _drop:
            _ds = set(_drop)
            rows = [r for r in rows if r[0] not in _ds]
            selfchk = [t for t in selfchk if t[1] not in _ds]
            want_ifaces = want_ifaces - _ds
            a.dropped_ifaces = _drop
            print(f"  DROPPED {len(_drop)} INCOMPLETE interface(s) (--drop-incomplete-ifaces), "
                  f"each short of {_exp} pose(s): "
                  + ", ".join(f"{i}:{_cnt.get(i, 0)}" for i in _drop[:8])
                  + (" ..." if len(_drop) > 8 else ""))
            print(f"  the metric below covers {len(want_ifaces)} interface(s), NOT the pool's "
                  f"{len(want_ifaces) + len(_drop)}. A baseline for it must be computed over the "
                  f"same {len(want_ifaces)}.")
    aggregate(a, rows, selfchk, want_ifaces)


def aggregate(a, rows, selfchk, want_ifaces):
    """The self-check, the coverage contract and the metric. Shared by the sharded and plain paths.

    ROW FORMAT IS (iface, seed, sample, iptm, dockq, iptm_mean-or-None), SIX FIELDS, ON BOTH
    PATHS. It was briefly five on the merge path and six on the single-process one, with aggregate
    unpacking five -- so the plain path raised ValueError and the merge path raised NameError on a
    mean_v that was local to merge_rows. Both were introduced by the --pool-mean change and caught
    in review. One format, normalised here at the top, is the only thing that keeps the two paths
    from drifting again: a helper that reads rows[i][5] would hide the mismatch instead.
    """
    rows = [(r[0], r[1], r[2], r[3], r[4], (r[5] if len(r) > 5 else None)) for r in rows]
    # ---- THE SELF-CHECK
    #
    # IT CANNOT APPLY UNDER A CROP, and this is not a loophole -- it is the premise failing. The
    # check compares the recomputed shipped ipTM against results.csv's stored value, and
    # results.csv was produced by the ordinary FULL-STRUCTURE pipeline. Scoring the shipped head
    # on cropped inputs deliberately computes a different quantity: measured on the 738 training
    # interfaces at 50.9% of tokens kept, 35,843 of 36,900 rows move by more than 0.002 and the
    # largest moves by 0.80. A check that fires on the experiment working is not protecting
    # anything.
    #
    # SO IT IS REPORTED RATHER THAN ENFORCED, and ONLY under --crop-margin. Uncropped it stays a
    # hard stop, which is the case it was written for. It is NOT given its own flag: a flag to
    # silence it would eventually be passed on an uncropped run, which is exactly the failure it
    # exists to catch. The condition is the crop itself.
    #
    # WHAT STILL GUARDS THE JOINS under a crop: the coverage check below, which enforces every
    # intended (iface, seed, sample) key with no duplicates and no non-finite values, and the
    # pose->sample join, which is done by coordinate identity against the UNCROPPED coordinates
    # and so is unaffected by cropping. What is genuinely weaker is join 1's independent
    # confirmation, and that is stated here rather than left for a reader to notice.
    # EITHER CROP MODE, NOT JUST --crop-margin. Raised in review of 33f24f87. The shipped pass
    # under --crop-tokens scores the shipped head on CROPPED inputs while results.csv holds the
    # UNCROPPED value, so every row is expected to differ -- exactly as under --crop-margin. With
    # the condition keyed to crop_margin alone the shipped baseline pass would have hit the hard
    # STOP below and the experiment would have had no baseline at all.
    # NO CROP RULE REACHES THE EVALUATOR ANY MORE. --crop-margin, --crop-tokens and
    # --crop-predicted-margin were all deleted 2026-11-13; --crop-pool is applied by the TRAINER
    # and never by this file. The self-check is therefore always enforced rather than downgraded
    # to informational, which is what it was written for.
    _crop_mode = ""
    _crop_selfcheck_informational = bool(_crop_mode)
    if not a.ckpt and not a.no_self_check:
        if not selfchk:
            raise SystemExit("STOP: no recomputed ipTM could be compared against results.csv, so "
                             "the three joins are unverified and no metric below means anything.")
        mx = max(d for d, *_ in selfchk)
        bad = [t for t in selfchk if t[0] > a.iptm_tol]
        over = len(bad)
        print(f"\nSELF-CHECK vs results.csv: {len(selfchk)} row(s), max |diff| {mx:.6f}, "
              f"{over} over --iptm-tol {a.iptm_tol}")
        if over:
            # WHICH ROWS, GROUPED THE WAY THE THREE JOINS WOULD FAIL. An interface wrong in join 1
            # is wrong at EVERY seed, so it shows as a near-complete count; a join-2 problem hits
            # particular seeds of particular structures; a join-3 problem is scattered.
            per_if = collections.Counter(t[1] for t in bad)
            tot_if = collections.Counter(t[1] for t in selfchk)
            print(f"  offending interfaces: {len(per_if)} of {len(tot_if)} compared")
            for ifc, c in per_if.most_common(15):
                print(f"    {ifc}: {c} of {tot_if[ifc]} row(s), "
                      f"worst {max(t[0] for t in bad if t[1] == ifc):.6f}")
            worst = sorted(bad, key=lambda t: -t[0])[:5]
            print("  worst rows (iface, seed, sample: recomputed vs stored):")
            for d, ifc, sd2, k2, v2, sv in worst:
                print(f"    {ifc} {sd2} s{k2}: {v2:.4f} vs {sv:.4f}  (|diff| {d:.4f})")
            if a.selfcheck_out:
                with open(a.selfcheck_out, "w") as fh:
                    fh.write("iface,seed,sample,recomputed_iptm,stored_iptm,abs_diff\n")
                    for d, ifc, sd2, k2, v2, sv in sorted(bad, key=lambda t: -t[0]):
                        fh.write(f"{ifc},{sd2},{k2},{v2:.6f},{sv:.6f},{d:.6f}\n")
                print(f"  every offending row written to {a.selfcheck_out}")
            if _crop_selfcheck_informational:
                print(f"  NOT A FAILURE UNDER {_crop_mode}: results.csv holds "
                      f"the UNCROPPED shipped ipTM, and this pass scores the shipped head on "
                      f"CROPPED inputs on purpose, so these rows are EXPECTED to differ. The "
                      f"difference is the measurement. Join 1 therefore has no independent "
                      f"confirmation in this pass; coverage below still enforces every intended "
                      f"key, and the pose->sample join runs on uncropped coordinates.")
            else:
                raise SystemExit(
                    f"STOP: the shipped head does not reproduce {over} stored ipTM value(s). "
                    f"One of the three joins -- interface->chain pair, pose->sample, or "
                    f"(iface,seed,sample)->DockQ -- is wrong.")
        elif _crop_selfcheck_informational:
            # EVERY ROW WITHIN TOLERANCE UNDER A CROP IS ITS OWN ALARM: it means the crop changed
            # nothing the head reads, so the "cropped" number is the uncropped one.
            raise SystemExit(
                f"STOP: {_crop_mode} was applied and yet every recomputed "
                f"shipped ipTM still matches results.csv within {a.iptm_tol}. The crop cannot "
                f"have reached the head, so this metric is an uncropped number mislabelled.")
        else:
            print("  the joins reproduce the fold's own ipTM, so the metric below is measuring "
                  "the head")

    # ---- COVERAGE, ENFORCED KEY BY KEY BEFORE ANY NUMBER IS COMPUTED
    # `if dq is not None` silently dropped every pose without a DockQ, and the only coverage check
    # was that an interface had at least ONE scored pose -- so one of 500 passed. All four of the
    # reviewer's conditions are now required: every intended key, unique rows, finite scores.
    seen = collections.Counter((iface, sd_, k) for iface, sd_, k, _v, _dq, _vm in rows)
    dup = [kk for kk, c in seen.items() if c > 1]
    if dup:
        raise SystemExit(f"STOP: {len(dup)} duplicate (iface, seed, sample) row(s), e.g. {dup[:4]}. "
                         f"A duplicated pose would be weighted twice in the selection.")
    nonfinite = [(iface, sd_, k) for iface, sd_, k, v, dq, _vm in rows
                 if not np.isfinite(v) or (dq is not None and not np.isfinite(dq))]
    if nonfinite:
        raise SystemExit(f"STOP: {len(nonfinite)} row(s) carry a non-finite ipTM or DockQ, e.g. "
                         f"{nonfinite[:4]}. argmax over a nan is undefined.")
    nodq = sorted((f"{i}/{s_}/{k}") for i, s_, k, _v, dq, _vm in rows if dq is None)
    if nodq and not a.allow_partial:
        raise SystemExit(f"STOP: {len(nodq)} scored pose(s) have no DockQ in {a.results}, e.g. "
                         f"{nodq[:5]}. Dropping them silently changes what the mean is over; "
                         f"recover them with phase_j_fill_scores.py or pass --allow-partial.")
    byif = collections.defaultdict(list)
    for iface, sd_, k, v, dq, _vm in rows:
        if dq is not None:
            byif[iface].append((v, dq, sd_, k))
    missing_if = sorted(want_ifaces - set(byif))
    if missing_if:
        raise SystemExit(f"STOP: {len(missing_if)} pool interface(s) have no scored pose: "
                         f"{missing_if[:6]}. The metric would cover a subset.")
    # AND EVERY INTERFACE MUST CARRY THE WHOLE BUDGET. An interface with one scored pose out of 500
    # satisfied the check above; depth is what the word "budget" in the output is asserting.
    # THE EXACT INTENDED KEY SET, NOT MERELY EQUAL DEPTH. Requiring every interface to carry the
    # same number of poses passes when a seed is missing from EVERY interface: the depths stay equal
    # and the label still claims the full sweep. The intended set is the cross product the budget
    # asserts -- every pool interface x every seed in the budget x every sample.
    have = {(iface, int(sd_), int(k)) for iface, sd_, k, _v, dq, _vm in rows if dq is not None}
    # THE INTENDED SET IS PER STRUCTURE, because a trimmed fold keeps a different 30 seeds for
    # each one. interface_chain_pairs builds the id as f"{pdb}_{c1}{c2}", so one rsplit recovers
    # the structure; a wrong derivation cannot pass silently, because intended_seeds hard-stops on
    # a pdb the manifest does not name.
    want = {(iface, sd_, k) for iface in want_ifaces
            for sd_ in intended_seeds(a, iface.rsplit("_", 1)[0])
            for k in range(a.nsample)}
    extra = sorted(have - want)
    absent = sorted(want - have)
    if extra:
        raise SystemExit(f"STOP: {len(extra)} scored row(s) lie OUTSIDE the intended set, e.g. "
                         f"{extra[:4]}. The metric would average over poses the budget does not "
                         f"claim.")
    if absent and not a.allow_partial:
        _per = collections.Counter(i for i, _s, _k in absent)
        raise SystemExit(
            f"STOP: {len(absent)} of {len(want)} intended (interface, seed, sample) key(s) have no "
            f"scored pose, across {len(_per)} interface(s), e.g. {absent[:5]}.\n  A metric labelled "
            f"'budget {a.budget}' must cover every one of them. Equal depth is not the same "
            f"assertion: a seed missing from every interface leaves the depths equal. Pass "
            f"--allow-partial to report a number that does not mean what its label says.")
    n_pose = collections.Counter()
    for iface, sd_, _k, _v, dq, _vm in rows:
        if dq is not None:
            n_pose[iface] += 1
    want_depth = max(n_pose.values()) if n_pose else 0
    print(f"coverage: {len(byif)} interface(s), {want_depth} scored pose(s) each, "
          f"{len(rows)} row(s) total, {len(have)} of {len(want)} intended key(s), 0 duplicates, "
          f"0 non-finite")
    # THE TIE-BREAK IS EXPLICIT, so the selection cannot depend on row order. `max` returns the
    # FIRST maximal element, and row order differs between a sharded run (shards concatenated,
    # seeds interleaved) and a single-process one (seeds ascending), so two poses with equal ipTM
    # selected different DockQ depending on how the work was split. Ties now go to the lowest seed,
    # then the lowest sample -- the same answer either way.
    def _pick(v):
        return max(v, key=lambda t: (t[0], -t[2], -t[3]))

    # THE SECOND SELECTION, on the SAME poses. Built from the mean-pooled column when the pass
    # wrote one, so the two readouts differ only in which score chose the pose.
    # THE MEAN COLUMN GETS THE SAME CONTRACT AS THE MAX ONE.
    #
    # The first version simply skipped poses whose iptm_mean was missing or non-finite, so a pass
    # that scored every pose on the max and only some on the mean would report a mean-selected
    # metric over a SMALLER candidate set -- with the coverage check passing, because that check
    # reads the max column. A selector handed fewer candidates is not comparable to one handed
    # all of them, and the label would still say "budget N". Raised in review.
    byif_m = collections.defaultdict(list)
    _m_missing, _m_nonfinite = [], []
    for iface, sd_, k, _v, dq, vm in rows:
        if dq is None:
            continue
        if vm is None:
            _m_missing.append((iface, sd_, k))
            continue
        if not math.isfinite(vm):
            _m_nonfinite.append((iface, sd_, k))
            continue
        byif_m[iface].append((vm, dq, sd_, k))
    if (_m_missing or _m_nonfinite) and byif_m:
        # Partial means the column exists but does not cover the pass. Absent entirely is the
        # legacy format and is handled by the --select-by refusal below.
        msg = (f"{len(_m_missing)} pose(s) have no iptm_mean and {len(_m_nonfinite)} have a "
               f"non-finite one, out of {sum(len(v) for v in byif.values())} scored pose(s). "
               f"e.g. missing {_m_missing[:3]}, non-finite {_m_nonfinite[:3]}")
        # UNCONDITIONAL WHEN THE MEAN IS THE SELECTOR, and deliberately NOT waivable by
        # --allow-partial. That flag exists for the intended-key contract on the MAX column; using
        # it here would let the one thing this guard exists to prevent through on the same switch.
        # A mean-selected metric over fewer candidates than the max selector saw is not a weaker
        # version of the number, it is a different number wearing the same label.
        if a.select_by == "mean":
            raise SystemExit(
                f"STOP: {msg}\n  --select-by mean would then choose from a SMALLER candidate set "
                f"than the max selector saw, while the coverage check passes because it reads the "
                f"max column. Fix the pass (every scored pose needs a finite iptm_mean) or select "
                f"by the max; --allow-partial does NOT waive this.")
        print(f"  WARNING: {msg}; any mean-selected number below is over that reduced set.")
    if a.select_by == "mean" and not byif_m:
        raise SystemExit(
            "STOP: --select-by mean needs an iptm_mean column, which only --pool-mean writes. "
            "Selecting by a column that is not there would silently fall back to the max and "
            "report it under the wrong label.")
    if a.select_by == "mean":
        byif, byif_m = byif_m, byif
    sel = [_pick(v)[1] for v in byif.values()]
    orc = [max(dq for _v, dq, _s, _k in v) for v in byif.values()]
    n = len(sel)
    out = {"n_interfaces": n, "budget": a.budget,
           **({"dropped_ifaces": a.dropped_ifaces,
               "n_dropped_ifaces": len(a.dropped_ifaces)}
              if getattr(a, "dropped_ifaces", None) else {}),
           "mean_dockq": round(statistics.mean(sel), 4),
           "frac_acc": round(sum(1 for d in sel if d >= ACCEPTABLE) / n, 4),
           "oracle_mean_dockq": round(statistics.mean(orc), 4),
           "oracle_frac_acc": round(sum(1 for d in orc if d >= ACCEPTABLE) / n, 4),
           "ckpt": a.ckpt or "shipped"}
    out["select_by"] = a.select_by
    out["headroom"] = round(out["oracle_mean_dockq"] - out["mean_dockq"], 4)
    # WHICH SEEDS THIS NUMBER COVERS, RECORDED WITH THE NUMBER. budget + seed_base no longer imply
    # the intended set: a trimmed fold's kept ids come from a manifest, and two different manifests
    # with the same per-structure count would yield two different metrics whose json is otherwise
    # byte-shape identical. Review found the only trace was a print the trainer discards on success.
    out["seed_base"] = a.seed_base
    _sm = getattr(a, "seed_manifest", "")
    out["seed_manifest"] = os.path.abspath(_sm) if _sm else ""
    out["seed_manifest_digest"] = _SEED_MANIFEST_META.get(_sm, {}).get("digest", "")
    # BOTH POOLINGS REPORTED, so a mean-trained arm and the shipped max baseline can be read off
    # one pass and nobody has to trust that the two runs were otherwise identical.
    if byif_m:
        _alt = [_pick(v)[1] for v in byif_m.values()]
        _an = "max" if a.select_by == "mean" else "mean"
        out[f"mean_dockq_select_{_an}"] = round(statistics.mean(_alt), 4)
        out[f"frac_acc_select_{_an}"] = round(
            sum(1 for d in _alt if d >= ACCEPTABLE) / len(_alt), 4)
    print(f"\nMETRIC ({out['ckpt']}, {n} interfaces, budget {a.budget} seeds, "
          f"selected by {a.select_by.upper()}-pooled ipTM)")
    print(f"  mean_dockq {out['mean_dockq']:.4f}   frac_acc {out['frac_acc']:.4f}")
    for _k in ("mean_dockq_select_mean", "mean_dockq_select_max"):
        if _k in out:
            _f = _k.replace("mean_dockq", "frac_acc")
            print(f"  the OTHER pooling ({_k.rsplit('_', 1)[-1]}): mean_dockq "
                  f"{out[_k]:.4f}   frac_acc {out[_f]:.4f}")
    print(f"  oracle     {out['oracle_mean_dockq']:.4f}   {out['oracle_frac_acc']:.4f}   "
          f"headroom {out['headroom']:.4f}")
    # THE BASELINE IS THE SHIPPED HEAD OVER *THIS* POOL, DERIVED, NEVER HARDCODED.
    #
    # It used to be a fixed string -- 0.5775 / 0.7283, the FULL validation pool's numbers --
    # printed whatever --pool was actually scored. Every val10 log therefore invited the reader to
    # compare an 18-interface result against a 1,011-interface baseline: epoch 133 of the lr 1e-2
    # anchor read 0.2251, which looks like a collapse against 0.5775 and is in fact flat against
    # val10's own 0.2626. A baseline that does not move with --pool is worse than none.
    #
    # results.csv's `iptm` IS the shipped head's selection statistic, so the baseline is this same
    # selection run over the stored column instead of the recomputed one -- same _pick, same
    # tie-break, same interfaces, same budget, no second forward pass. It is the stored value at
    # 4dp rather than a fresh computation, which moved the full pool's mean by 0.0004 (0.5779
    # stored against 0.5775 recomputed), so it is labelled as the stored one.
    base_iptm = {(ifc, sd_, k): sv for _d, ifc, sd_, k, _v, sv in selfchk}
    b_by = collections.defaultdict(list)
    n_nobase = 0
    for iface, sd_, k, _v, dq, _vm in rows:
        if dq is None:
            continue
        sv = base_iptm.get((iface, sd_, k))
        if sv is None:
            n_nobase += 1
        else:
            b_by[iface].append((sv, dq, sd_, k))
    if n_nobase or set(b_by) != set(byif):
        # NO NUMBER RATHER THAN A PARTIAL ONE. A baseline over fewer poses, or over fewer
        # interfaces, is not the thing this result is being compared against -- and a silently
        # narrower baseline is the same class of error as the hardcoded one it replaced.
        print(f"  baseline to beat: UNAVAILABLE -- {n_nobase} scored pose(s) carry no stored ipTM "
              f"in {a.results}, and {len(set(byif) - set(b_by))} interface(s) carry none at all, "
              f"so the shipped head's selection cannot be reproduced over this pool.")
    else:
        b_sel = [_pick(v)[1] for v in b_by.values()]
        out["baseline_mean_dockq"] = round(statistics.mean(b_sel), 4)
        out["baseline_frac_acc"] = round(
            sum(1 for d in b_sel if d >= ACCEPTABLE) / len(b_sel), 4)
        print(f"  baseline   {out['baseline_mean_dockq']:.4f}   "
              f"{out['baseline_frac_acc']:.4f}   (shipped head over THESE {n} interface(s) at "
              f"budget {a.budget}, from results.csv's recorded ipTM at 4dp)")
    if a.out:
        with open(a.out, "w") as fh:
            import json
            json.dump(out, fh, indent=1)
        print(f"  wrote {a.out}")
    return 0


def _ca_coords(feat, px, pose):
    """(T, 3) CA coordinates for one pose, and which tokens have one.

    CA, NOT THE DISTOGRAM REPRESENTATIVE. distogram_rep_atom_mask is CB for ordinary residues and
    CA only for glycine; ABAG-Rank measures CA-CA. On 10fj the two differ at 1,158 of 4,795 atoms,
    1.42 A apart on average, and at a 12 A cutoff the CB convention loses 198 real pairs and adds
    430 spurious ones.

    DERIVED FROM THE TRUNK, not read from a label npz: pose_direction.derive_frames returns the
    pae_rep_atom_mask, which IS the CA mask, and it was verified exact against the label npz on all
    500 training structures. validation/fold has 53 trunks and no labels, so a label-based mask
    could not be used on the held-out set at all.
    """
    import pose_direction
    _fai, ca_mask = pose_direction.derive_frames(feat)
    ca_mask = ca_mask.detach().cpu().numpy() if hasattr(ca_mask, "detach") else np.asarray(ca_mask)
    a2t = feat["atom_to_token_idx"]
    a2t = a2t.detach().cpu().numpy() if hasattr(a2t, "detach") else np.asarray(a2t)
    asym = feat["asym_id"]
    asym = asym.detach().cpu().numpy() if hasattr(asym, "detach") else np.asarray(asym)
    T = len(asym)
    xyz = px[pose] if px.ndim == 3 else px
    out = np.full((T, 3), np.nan, dtype=np.float32)
    seen = np.zeros(T, dtype=bool)
    for ai in np.where(ca_mask.astype(bool))[0]:
        t = int(a2t[ai])
        if not seen[t]:
            out[t] = xyz[ai]
            seen[t] = True
    return out, seen, asym


_VALID_CACHE = {}


def _valid_tokens(a, pdb, n_tokens):
    """[T] bool: tokens whose residue is known, from abag_rank/restypes.py.

    REQUIRED, NOT OPTIONAL. Without it the closest-pair fallback can select a pair that the
    feature builder later drops for containing an X, and the correct fallback -- the closest pair
    of KNOWN residues -- may lie beyond the dump cutoff and be gone for good.
    """
    if pdb not in _VALID_CACHE:
        f = os.path.join(a.dump_pae_restypes, f"{pdb}.npz")
        if not os.path.exists(f):
            raise SystemExit(
                f"STOP: {pdb}: no residue letters at {f}. --dump-pae needs them to apply "
                f"upstream's X filter before choosing the closest-pair fallback; run "
                f"abag_rank/restypes.py for this population first.")
        one = np.load(f)["residue_one_letter"]
        _VALID_CACHE[pdb] = np.array([str(c) != "X" for c in one], dtype=bool)
    v = _VALID_CACHE[pdb]
    if len(v) != n_tokens:
        raise SystemExit(f"STOP: {pdb}: {len(v)} residue letter(s) against {n_tokens} token(s).")
    return v


_PAE_ACC = {"pdb": None, "sum": None, "n": 0}


def _pae_accumulate(a, pdb, tpp, shard_tag):
    """Running per-complex sum of the DENSE PAE, so pae_col_mean can be exact.

    WHY THE SPARSE DUMP IS NOT ENOUGH. Upstream's stage 1 writes EVERY inter-chain pair with no
    distance cutoff (01_process_af3_predictions.py:1184-1190), and stage 3 means over those, so
    `pae_col_mean` for a pair includes the poses in which that pair is far apart. Our dump keeps
    only pairs within 16 A, so a pair at 8 A in one pose and 25 A in another contributes once
    instead of twice and the mean is wrong -- not by a rounding error but by the whole weight of
    the omitted poses. Centring is one of this model's central features, so that is a semantic
    difference, not a storage one. Caught in review.

    The fix does not need dense per-pose storage: the dense tensor is already in hand on the GPU,
    so a running sum and count per complex gives the exact mean for ~1.3 MB per complex (float32
    T x T), against ~60 GB to keep every pose. Partials are written per shard and merged, because
    a complex's poses are split across shards.
    """
    acc = _PAE_ACC
    if acc["pdb"] != pdb:
        _pae_flush(a, shard_tag)
        acc["pdb"], acc["sum"], acc["n"] = pdb, np.zeros_like(tpp, dtype=np.float64), 0
    if acc["sum"].shape != tpp.shape:
        raise SystemExit(f"STOP: {pdb}: PAE shape {tpp.shape} against accumulator "
                         f"{acc['sum'].shape}. Not the same molecule.")
    acc["sum"] += tpp
    acc["n"] += 1


def _pae_flush(a, shard_tag):
    """Write this complex's partial sum. Merged across shards by abag_rank/merge_pae_mean.py."""
    acc = _PAE_ACC
    if acc["pdb"] is None or acc["n"] == 0:
        return
    d = os.path.join(a.dump_pae, acc["pdb"])
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"_paesum_{shard_tag}.npz")
    # ATOMIC: a partial sum that looks complete would silently skew every centred feature of this
    # complex, so it is written aside and renamed.
    # np.savez_compressed APPENDS .npz to a name that lacks it, so what it actually writes is
    # <stem>.tmp.npz, not <stem>.tmp. Renaming the name we passed raises FileNotFoundError and
    # leaves a stray .tmp.npz that a directory scan could mistake for a result -- which is exactly
    # what the first run of this code did. Caught by the smoke test, not by review.
    tmp = f + ".tmp"
    np.savez_compressed(tmp, pae_sum=acc["sum"].astype(np.float64),
                        n_poses=np.int64(acc["n"]), pdb=acc["pdb"])
    os.replace(tmp + ".npz", f)
    acc["pdb"], acc["sum"], acc["n"] = None, None, 0


def _dump_pose(a, pdb, seed, sample, tpp, feat, px, pose):
    """Write ONLY what the reranker consumes, for one pose.

    WHAT IS SAVED AND WHY NOTHING MORE. A dense PAE matrix is 1.5 MB a pose, ~60 GB over our
    38,150; every inter-chain pair regardless of distance is ~2 MB a pose, ~76 GB. The reranker
    only ever sees pairs near the interface, so only pairs under --dump-pae-cutoff are written,
    plus -- unconditionally -- the single closest pair of every ordered chain pair, so the
    zero-contact fallback upstream applies can always be satisfied without going back to the GPU.
    That is ~3 GB in total.

    The cutoff here is deliberately WIDER than the 12 A the model uses, so the dataset can apply
    its own cutoff, sorting and truncation without a second pass.

    asym_i/asym_j travel with each pair because an interface is a CHAIN PAIR, and which asym pair
    an interface corresponds to is decided per pose by OST -- not by a static table. Keeping the
    chain of each pair lets the dataset select an interface's pairs by that per-pose answer.
    """
    rep, seen, asym = _ca_coords(feat, px, pose)
    # UPSTREAM DROPS X PAIRS BEFORE THE CUTOFF AND BEFORE THE FALLBACK
    # (preprocess_shards.process_single_sample), so a fallback chosen over all tokens can pick a
    # pair that features.py then discards, leaving the interface with no pair at all and no way to
    # recover one without another forward. restypes.py establishes that X residues are real here:
    # 103 of 5,549 sequences contain one. Caught in review.
    valid = _valid_tokens(a, pdb, len(asym))
    if tpp.shape[0] != len(asym):
        raise SystemExit(f"STOP: {pdb}/{seed}/{sample}: PAE is {tpp.shape[0]} tokens against "
                         f"{len(asym)} in the trunk. Not the same molecule.")
    d = np.linalg.norm(rep[:, None, :] - rep[None, :, :], axis=-1)
    ok = seen[:, None] & seen[None, :] & valid[:, None] & valid[None, :]
    inter = asym[:, None] != asym[None, :]
    keep = inter & ok & (d <= a.dump_pae_cutoff)
    # the closest pair of every ordered chain pair, so the fallback never needs the GPU again
    for ci in np.unique(asym):
        for cj in np.unique(asym):
            if ci == cj:
                continue
            m = (asym[:, None] == ci) & (asym[None, :] == cj) & ok
            if not m.any():
                continue
            dd = np.where(m, d, np.inf)
            i0, j0 = np.unravel_index(np.argmin(dd), dd.shape)
            if np.isfinite(dd[i0, j0]):
                keep[i0, j0] = True
    ii, jj = np.where(keep)
    out = os.path.join(a.dump_pae, pdb)
    os.makedirs(out, exist_ok=True)
    _f = os.path.join(out, f"{seed}_{sample}.npz")
    # ATOMIC. An interrupted export must not leave a file that looks complete to a dataset builder
    # enumerating the directory. Caught in review.
    np.savez_compressed(
        _f + ".tmp",
        inter_idx=ii.astype(np.int32), inter_jdx=jj.astype(np.int32),
        asym_i=asym[ii].astype(np.int16), asym_j=asym[jj].astype(np.int16),
        interchain_pae_vals=tpp[ii, jj].astype(np.float32),
        interchain_ca_distances=d[ii, jj].astype(np.float32),
        n_tokens=np.int32(len(asym)), cutoff=np.float32(a.dump_pae_cutoff),
        pdb=pdb, seed=np.int32(seed), sample=np.int32(sample))
    os.replace(_f + ".tmp.npz", _f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/validation/fold")
    ap.add_argument("--results", default=f"{ROOT}/validation/fold/results.csv")
    ap.add_argument("--pool", default=f"{ROOT}/validation/fold/pool.csv")
    ap.add_argument("--roles", default=f"{ROOT}/validation/fold/chain_roles.csv")
    ap.add_argument("--ckpt", default="", help="a head state dict from phase_k_train; empty = shipped")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--budget", type=int, default=100, help="seeds per interface, from --seed-base")
    ap.add_argument("--seed-base", type=int, default=400000)
    ap.add_argument("--seed-manifest", default="",
                    help="json recording WHICH seeds a trimmed fold kept per structure "
                         "({structures: {pdb: {seeds: [...]}}, seed_range: [lo, hi]}). Without it "
                         "the intended set is the contiguous window from --seed-base, which a "
                         "trimmed fold does not satisfy.")
    ap.add_argument("--nsample", type=int, default=5,
                    help="poses per unit, as the fold wrote them. Part of the coverage contract: "
                         "the intended key set is interfaces x seeds x THIS, so a budget label "
                         "asserts a specific number of rows rather than merely equal depth.")
    ap.add_argument("--iptm-tol", type=float, default=2e-3,
                    help="how closely the recomputed ipTM must reproduce results.csv with the "
                         "SHIPPED head. bf16 trunk tensors are stored and reloaded and the head runs "
                         "under autocast, so exact equality is not the bar.")
    ap.add_argument("--frame-direction", action="store_true",
                    help="the checkpoint was trained with phase_k_train --frame-direction, so the "
                         "head carries a frame_dir_proj the stock model does not. Installs that "
                         "pathway before loading, and reads frame_atom_index / pae_rep_atom_mask "
                         "from the label npz beside each trunk. phase_k_train passes this "
                         "automatically; without it the checkpoint load refuses the extra tensor.")
    # CROPPING, SO A HEAD CAN BE SCORED ON THE INPUTS IT WAS TRAINED ON.
    #
    # This evaluator had no crop path at all, so every number it has ever produced is on FULL
    # structures -- while phase_k_train crops EVERY pose it trains on (o2qczq: 199,800 of 199,800,
    # median 53% of tokens kept). "In-sample" therefore meant the structures were training
    # structures and NOT that the inputs were training inputs, and a head read off its training
    # distribution is not being given a fair reading.
    #
    # THE SAME TWO FUNCTIONS THE TRAINER CALLS, not a reimplementation of them:
    # BC.interface_crop_tokens then BC.slice_cached_unit. The crop has no RNG -- it is a function
    # of the trunk, the label and (contact, margin, all_poses) -- so passing the arm's own values
    # reproduces its crop exactly. Verified on 40 randomly sampled (pdb, seed) units against the
    # trainer's own crop_sizes_rank*.csv: 40 of 40 kept_tok reproduced exactly.
    #
    # UNCROPPED IS STILL THE DEFAULT, and should stay the default: inference runs on whole
    # structures, so the uncropped number is the one that says whether a head is USEFUL. The
    # cropped number says whether it LEARNED something, which is a different question.
    ap.add_argument("--crop-contact", type=float, default=5.0,
                    help="heavy-atom distance defining an interface token, matching "
                         "phase_k_train --crop-contact. Read by --crop-margin and "
                         "--crop-predicted-margin.")
    ap.add_argument("--crop-all-poses", action="store_true",
                    help="define the crop from the union over all poses rather than pose 0, "
                         "matching phase_k_train --crop-all-poses. MUST match the arm being "
                         "scored; the arms in this project do not pass it.")
    ap.add_argument("--pose-split-check", action="store_true",
                    help="score one pose per forward pass with NO crop. Scores must match the "
                         "batched path exactly; this exists because --crop-predicted-margin has "
                         "to score poses one at a time, and if splitting the pose dimension "
                         "changed a score on its own, every per-pose crop number would be "
                         "measuring the split rather than the crop.")
    ap.add_argument("--labels", default="",
                    help="label npz root for --crop-margin; defaults to <fold>/labels. The crop "
                         "needs ground-truth coordinates, which is why this evaluator could "
                         "previously avoid reading labels entirely.")
    ap.add_argument("--pose-perblock", action="store_true",
                    help="the checkpoint was trained with phase_k_train --pose-perblock, so the "
                         "head carries pose_perblock_proj tensors the stock model does not. "
                         "phase_k_train passes this automatically; without it the checkpoint load "
                         "refuses the extra tensors.")
    ap.add_argument("--pose-perblock-tied", action="store_true",
                    help="the checkpoint was trained with weight sharing across injected blocks. "
                         "The tensor COUNT differs between tied and untied, so this must match "
                         "how the head was trained or the load fails.")
    ap.add_argument("--pose-perblock-copy", action="store_true",
                    help="scoring counterpart of the trainer's --pose-perblock-copy: each injected\n"
                         "block gets its OWN copy of block 0's injection, so the "
                         "checkpoint carries pose_perblock_linear_no_bias_d* tensors "
                         "the stock model lacks. UNLIKE --pose-perblock-share this mode "
                         "DOES add tensors, so the state-dict guard below catches its "
                         "omission rather than silently scoring one-depth injection.")
    ap.add_argument("--pose-perblock-share", action="store_true",
                    help="the checkpoint was trained with --pose-perblock-share: block 0's OWN "
                         "pretrained injection (linear_no_bias_d AND linear_no_bias_d_wo_onehot) "
                         "re-applied at every block, with ZERO extra tensors. The tensor COUNT is "
                         "therefore IDENTICAL to a stock head, so unlike the other pathway flags "
                         "this one cannot be caught by the checkpoint-load guard -- omit it and "
                         "the checkpoint loads cleanly and is scored with the pose injected at "
                         "ONE depth instead of four, silently and wrongly. phase_k_train passes "
                         "it automatically. Not accepting it is what made arm wd656y exit 1 after "
                         "its first epoch: the flag was forwarded to a parser that did not know it.")
    # --------------------------------------------------------------------------------------
    # IN-BLOCK POSE INJECTION, the pathway that had no scoring counterpart at all.
    #
    # phase_k_train --pose-inblock registers pose_inblock_d / pose_inblock_d_wo_onehot (and two
    # more with --pose-inblock-pae) on the confidence head. _head_ckpt_tensors saves every head
    # tensor, so those keys are IN the checkpoint -- and this evaluator knew nothing about them,
    # so the exact key-set guard below refused every in-block checkpoint as carrying tensors the
    # model does not have. The arm could train to completion and have no held-out DockQ at all.
    # Found by review before any in-block arm was launched, which is the only reason it was not
    # discovered as a dead eval shard after an epoch of GPU time.
    ap.add_argument("--pose-inblock", action="store_true",
                    help="the checkpoint was trained with phase_k_train --pose-inblock, so the "
                         "head carries pose_inblock_d* tensors the stock model does not. Installs "
                         "the same injection at the same point (inside the block, after "
                         "tri_mul_in) before loading. phase_k_train passes this automatically; "
                         "without it the checkpoint load refuses the extra tensors.")
    ap.add_argument("--pose-inblock-block", type=int, default=-1,
                    help="which pairformer block carries the injection. -1 (the default) means the "
                         "LAST block, matching the trainer's default. THE DEPTH IS PART OF THE "
                         "OPERATOR: the same two tensors at a different depth is a different "
                         "function, and the key-set guard cannot see the difference because the "
                         "tensor names do not carry the block index.")
    ap.add_argument("--pose-inblock-pae", action="store_true",
                    help="the checkpoint also carries the SECOND injection, after pae_ln in the "
                         "PAE readout (pose_inblock_pae_d*). phase_k_train passes this "
                         "automatically whenever it trained with it.")
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--allow-roles-fallback", action="store_true",
                    help="use chain_roles.csv when OST's per-pose mapping is unusable. OFF by "
                         "default: that table assigns one model chain per reference chain for the "
                         "whole structure, which is measured WRONG for chemically identical chains "
                         "(9eg6 on every pose, 7zra on some), and --ckpt runs skip the self-check "
                         "that caught it. An unmapped chain that is FORCED by elimination needs no "
                         "fallback and no flag.")
    ap.add_argument("--selfcheck-out", default="",
                    help="write EVERY row that fails the self-check to this csv, so the failure can "
                         "be analysed without re-running the evaluator.")
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--crop-pool", default="",
                    help="DIRECTORY of per-structure crop pools, the SAME format the trainer's "
                         "--crop-pool takes (run_logging/select_crop_pool.py). Scores every unit "
                         "on one crop instead of the whole structure.\n"
                         "WHICH CROP: --crop-index, default 0, the SAME one for every unit and "
                         "every epoch. The trainer rotates `epoch %% n` because it wants a "
                         "different view each epoch; an evaluator must not, or two checkpoints "
                         "would be scored on different inputs and the comparison between them "
                         "would be of the crops as much as of the heads.\n"
                         "THIS IS A PROXY, NOT A DEPLOYMENT NUMBER. The pool is selected so each "
                         "crop's DockQ regret against the UNCROPPED selection is within a "
                         "threshold (0.05 on validation/crop_final), which is what makes it "
                         "usable to rank checkpoints cheaply -- roughly 3x faster, since the cost "
                         "is quadratic in tokens and these are 256 against a 482-token median. "
                         "Every number this project reports against shipped is uncropped, and a "
                         "cropped one is not comparable with them.")
    ap.add_argument("--crop-index", type=int, default=0,
                    help="which crop of each structure's pool to use. Fixed across units and "
                         "checkpoints on purpose; see --crop-pool.")
    ap.add_argument("--join-cache", default="",
                    help="cache the pose->sample join (unit_dir and sample indices) in this json "
                         "and reuse it. The join is derived from the trunk's coordinates and the "
                         "pose cifs, neither of which changes between epochs, so re-deriving it "
                         "every pass costs a measured 3.6 min per shard for an answer that cannot "
                         "differ. OFF by default: it changes nothing about the metric, but a stale "
                         "cache would, so it is opt-in and keyed on the trunk's mtime and size.")
    ap.add_argument("--dump-pae", default="",
                    help="write per-pose EXPECTED ALIGNED ERROR (Angstroms) for the ABAG-Rank "
                         "export, into <dir>/<pdb>/<seed>_<sample>.npz. Only the inter-chain pairs "
                         "in that complex's fixed pair set are written -- dense matrices would be "
                         "~60 GB over 38,150 poses against ~1.4 GB sparse. Requires "
                         "--pair-iptm-only (that path already computes the PAE probabilities) and "
                         "--dump-pae-pairsets. The sample index comes from the coordinate-identity "
                         "join, not from a (pdb, seed, sample) guess.")
    ap.add_argument("--dump-pae-restypes", default="",
                    help="directory of per-token residue letters from abag_rank/restypes.py. "
                         "Required with --dump-pae: upstream drops X pairs BEFORE the cutoff and "
                         "before the closest-pair fallback, so the fallback must know them.")
    ap.add_argument("--dump-pae-cutoff", type=float, default=16.0,
                    help="CA-CA cutoff for what the dump keeps. Wider than the 12 A the model "
                         "uses, so the dataset can apply its own cutoff and the closest-pair "
                         "fallback without a second GPU pass.")
    ap.add_argument("--select-by", choices=("max", "mean"), default="max",
                    help="WHICH POOLED ipTM SELECTS THE POSE. max is shipped ipTM and the default, "
                         "so every existing number stays comparable. mean requires --pool-mean and "
                         "is what an arm trained with --rank-pool mean must be scored by: training "
                         "one quantity and selecting by another measures neither. Both selections "
                         "are REPORTED whenever --pool-mean is on, so the choice only moves which "
                         "one is the headline. Added after review found the trainer optimising the "
                         "mean while this evaluator selected by the max.")
    ap.add_argument("--pool-mean", action="store_true",
                    help="ALSO compute chain_pair ipTM pooled as the MEAN over anchor rows and "
                         "write it as an extra iptm_mean column. ADDITIVE AND DEFAULT OFF: the "
                         "selection, the self-check and every existing column stay on the MAX, "
                         "which is what shipped ipTM is and what every baseline in this repo "
                         "used, so the two poolings can be compared offline from one pass "
                         "instead of being swapped and recomputed. Needs --pair-iptm-only, "
                         "whose path already has the PAE probabilities.")
    ap.add_argument("--pair-iptm-only", action="store_true",
                    help="compute chain_pair_iptm directly from PAE instead of the whole Protenix "
                         "summary, whose other fields this script computes and discards. Measured "
                         "bit-identical (largest difference 0.000e+00 over 25 units) and 2.7 min "
                         "faster per 26.1 min pass; see phase_k_eval_stage_timing.py. OFF by "
                         "default because a metric path is not changed on a 25-unit sample without "
                         "being asked.")
    ap.add_argument("--prefetch", action="store_true",
                    help="read the next unit's trunk on a background thread while the GPU scores "
                         "this one. Load is a measured 4.7 min per shard of a 26.1 min pass and is "
                         "strictly serial with the forward today, at 813 MB/s against a disk that "
                         "does 1782. OFF by default.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report a metric that does NOT cover its stated budget: missing trunks, "
                         "poses without DockQ, or interfaces of unequal depth. OFF by default, "
                         "because a number labelled 'budget 100' computed from one seed is worse "
                         "than no number.")
    ap.add_argument("--out", default="")
    # ---- SHARDING. The evaluator is embarrassingly parallel: every unit loads its own trunk and
    # scores independently, so nothing has to be communicated between shards and no process group
    # is needed -- which also keeps it clear of the NCCL fault on this box. A shard does the GPU
    # work for its slice and writes per-pose ROWS; the aggregate checks (self-check, coverage,
    # metric) can only be judged on the UNION, so they run once in --rows-in mode, on CPU.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--drop-unmapped-poses", action="store_true",
                    help="IN --rows-out MODE ONLY: write the rows that resolved, record the "
                         "unmapped poses in the meta, and EXIT 0 instead of refusing. The shard "
                         "previously hard-stopped, which killed every shard of the pass and left "
                         "the merge nothing to drop -- 4zto's 50-of-75 failure records took down "
                         "a 67,425-pose in-sample run three times that way. The interfaces whose "
                         "poses went missing are then short of their sub-pool's depth and are "
                         "dropped BY NAME by --drop-incomplete-ifaces at the merge, which is "
                         "where that decision belongs. It is deliberately inert outside "
                         "--rows-out: a single-process metric run still refuses, because there "
                         "no later step would catch the gap.")
    ap.add_argument("--drop-incomplete-ifaces", action="store_true",
                    help="MERGE ONLY. Report over the interfaces that are COMPLETE and drop the "
                         "ones that are not, naming them in the metric json and on stdout. "
                         "Without it a single unresolvable pose discards every finished pose in "
                         "the run: 4zto's OST chain mapping failed when its poses were generated "
                         "(\"Target chain M is not available in target processed\"), 50 of its 75 "
                         "pose jsons are failure records, and that refused a merge holding 67,325 "
                         "of 67,425 deep rows -- an hour of GPU for nothing, twice. An interface "
                         "is dropped WHOLE rather than counted at partial depth, because a "
                         "selection over 25 of 75 candidates is a different and easier problem "
                         "than the one the budget label claims. The baseline must then be computed "
                         "over the same reduced interface set or the comparison is invalid.")
    ap.add_argument("--rows-out", default="",
                    help="write this shard's per-pose rows here and exit before the aggregate "
                         "checks, which are meaningless on a slice.")
    ap.add_argument("--rows-in", default="",
                    help="comma-separated row csvs from --rows-out. Runs the self-check, the "
                         "coverage contract and the metric on their union, on CPU, with no model.")
    a = ap.parse_args()
    # CHECKED FROM argv, BEFORE ANYTHING LOADS. Two crop rules at once is knowable here, and a
    # refusal that waits for the model load arrives minutes later and on a worker.
    if a.dump_pae:
        if not a.pair_iptm_only:
            raise SystemExit("STOP: --dump-pae needs --pair-iptm-only; the PAE probabilities it "
                             "reuses are computed only on that path.")
        if not a.dump_pae_restypes or not os.path.isdir(a.dump_pae_restypes):
            raise SystemExit(f"STOP: --dump-pae needs --dump-pae-restypes; got "
                             f"{a.dump_pae_restypes!r}. Upstream's X filter runs before the "
                             f"cutoff and before the fallback, so the dump must apply it too.")
        os.makedirs(a.dump_pae, exist_ok=True)
        # RUN-LEVEL PROVENANCE. Every npz records its structure, seed and sample but nothing about
        # WHICH MODEL produced the PAE. Exporting half a dataset, then resuming into the same
        # directory with a different checkpoint, yields valid files carrying two models' PAE and
        # nothing to detect it. So the identity is written once and checked on reuse. Caught in
        # review.
        _prov = {"ckpt": os.path.abspath(a.ckpt) if a.ckpt else "",
                 "model_name": a.model_name, "fold": os.path.abspath(a.fold),
                 "pool": os.path.abspath(a.pool) if a.pool else "",
                 "seed_base": a.seed_base, "budget": a.budget,
                 "seed_manifest": os.path.abspath(a.seed_manifest) if a.seed_manifest else "",
                 "cutoff": a.dump_pae_cutoff, "nshards": a.nshards,
                 "restypes": os.path.abspath(a.dump_pae_restypes)}
        _pf = os.path.join(a.dump_pae, "_provenance.json")
        if os.path.exists(_pf):
            try:
                _old = json.load(open(_pf))
            except (ValueError, OSError) as _e:
                raise SystemExit(f"STOP: {_pf} is unreadable ({type(_e).__name__}: {_e}). With "
                                 f"the atomic write below this should be impossible; an "
                                 f"unparseable provenance means the directory is being written "
                                 f"by something else.")
            _diff = {k for k in set(_old) | set(_prov) if _old.get(k) != _prov.get(k)}
            _diff -= {"nshards"}          # sharding is a layout choice, not a model change
            if _diff:
                raise SystemExit(
                    f"STOP: {a.dump_pae} was exported under different settings and resuming into "
                    f"it would mix models in one dataset:\n    "
                    + "\n    ".join(f"{k}: existing {_old.get(k)!r} vs now {_prov.get(k)!r}"
                                     for k in sorted(_diff))
                    + "\n  Export to a new directory, or delete that one deliberately.")
        else:
            # ATOMIC, because all shards start within milliseconds and this path is racy three
            # ways: two shards can both find the file absent and both write it, and a third can
            # read it mid-write and fail to parse. A unique temp name plus os.replace makes every
            # read see either no file or a complete one, and makes concurrent writers idempotent
            # -- they are in the same run, so their content is identical. Caught in review.
            _tmp = f"{_pf}.{os.getpid()}.tmp"
            with open(_tmp, "w") as _fh:
                json.dump(_prov, _fh, indent=1)
            os.replace(_tmp, _pf)

    if a.rows_in:
        merge_rows(a)
        return

    import torch
    import phase_j_rescore_cache as RC
    import phase_j_build_cache as BC

    pairs_by_pdb, unresolved = interface_chain_pairs(a.pool, a.roles)
    refs_by_pdb = pool_ref_pairs(a.pool)
    n_ost, n_static, n_differ, differ_pdbs = 0, 0, 0, collections.Counter()
    n_chain_gone, chain_gone = 0, []
    fallback_poses = []          # poses scored WHOLE because their crop was undefinable
    unmapped = []
    if unresolved:
        raise SystemExit(f"STOP: {len(unresolved)} pool interface(s) cannot be mapped to model "
                         f"chains via {a.roles}: {unresolved[:6]}. The metric would silently cover "
                         f"a subset.")
    want_ifaces = {i for v in pairs_by_pdb.values() for i, _x, _y in v}
    print(f"pool: {len(want_ifaces)} interface(s) over {len(pairs_by_pdb)} structure(s)")

    # join 3, read once
    dockq, stored_iptm = {}, {}
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            try:
                k = (r["iface"], int(r["seed"]), int(r["sample"]))
                dockq[k] = float(r["dockq"]); stored_iptm[k] = float(r["iptm"])
            except (TypeError, ValueError):
                pass
    print(f"results.csv: {len(dockq)} scored (iface, seed, sample) row(s)")

    # EVERY INTENDED UNIT, AND A MISSING TRUNK IS FATAL. This appended only the trunks that
    # happened to exist, so a "100-seed" metric could be computed from one seed and still print
    # "budget 100" -- review round 9 reproduced exactly that. The budget is a CLAIM about coverage,
    # so it is enforced rather than described.
    units, missing_trunk, n_intended = [], [], 0
    for pdb in sorted(pairs_by_pdb):
        for sd_ in intended_seeds(a, pdb):
            n_intended += 1
            tp = os.path.join(a.fold, "trunks", pdb, f"free_{sd_}.npz")
            (units if os.path.exists(tp) else missing_trunk).append(
                (pdb, sd_, tp) if os.path.exists(tp) else f"{pdb}/free_{sd_}")
    if missing_trunk and not a.allow_partial:
        raise SystemExit(
            f"STOP: {len(missing_trunk)} of {n_intended} intended unit(s) have no "
            f"trunk under {a.fold}/trunks, e.g. {missing_trunk[:5]}.\n  A metric labelled "
            f"'budget {a.budget}' must cover {a.budget} seeds for every structure. Lower --budget, "
            f"or pass --allow-partial to report a number that does NOT mean what its label says.")
    if a.limit_units:
        if not a.allow_partial:
            raise SystemExit("STOP: --limit-units subsets the data, so the metric would not be the "
                             "metric. Pass --allow-partial to acknowledge that explicitly.")
        units = units[:a.limit_units]
    if a.nshards > 1:
        if not a.rows_out:
            raise SystemExit("STOP: --nshards without --rows-out would compute a metric from a "
                             "slice and label it with the full budget. A shard writes rows; the "
                             "metric is computed once, in --rows-in mode, over their union.")
        # STRIDED, not contiguous. units are ordered by structure, and structure sizes vary 2-3x, so
        # contiguous blocks would hand one shard the large complexes. A stride interleaves them.
        units = units[a.shard::a.nshards]
        print(f"shard {a.shard} of {a.nshards}: {len(units)} unit(s)")
    print(f"units: {len(units)} of {len(pairs_by_pdb)*a.budget} intended "
          f"({len(pairs_by_pdb)} structures x {a.budget} seeds)"
          + ("  [PARTIAL -- --allow-partial]" if a.allow_partial else ""))

    # THE MODEL IS LOADED LAST, after everything that can be checked from the filesystem has
    # been. Enumerating units after the build meant a coverage failure cost a 40 s model load and
    # needed a free GPU to even observe, which is also why it could not be tested without one.
    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold (without it coordinates are NaN)")
    dev = torch.device("cuda")
    model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    # INSTALLED BEFORE THE CHECKPOINT IS LOADED, because a head trained with --frame-direction
    # carries frame_dir_proj.weight and the guard below refuses any tensor the model does not
    # have. That refusal is the right behaviour -- scoring a trained head with the pathway absent
    # would measure a different function than the one trained -- so this flag is how the trainer
    # keeps the two in step, and phase_k_train passes it automatically.
    # ACTIVATION CHECKPOINTING OFF BEFORE ANY PATHWAY IS INSTALLED.
    #
    # Scoring runs entirely under torch.no_grad(), so checkpointing buys nothing here -- there is
    # no backward to recompute activations for. It was left at Protenix's config default of 1,
    # which was harmless until per-block orientation arrived: those copies read
    # pose_direction's direction cache, which the outer forward clears after the stack, so a
    # recomputed block would find it missing or holding a LATER sample's geometry.
    # pose_perblock.install refuses that combination, and the refusal fired here first -- run
    # 2g65a6 trained epoch 1 and then died because all three eval shards raised
    # "per-block orientation needs blocks_per_ckpt off, got 1". The trainer passes
    # --blocks-per-ckpt to itself but not to the evaluator, and the evaluator never set it at all.
    #
    # Set unconditionally rather than only for the orientation arms: under no_grad it is the
    # right value for every arm, and a conditional would leave the same trap for the next
    # pathway that caches per-sample state.
    model.confidence_head.pairformer_stack.blocks_per_ckpt = None

    # SAME REASON, SAME PLACE, and the refusals are the trainer's refusals verbatim so that a
    # combination which is a hard stop when training is a hard stop when scoring too. A pathway
    # whose evaluator is more permissive than its trainer is how a number gets reported for an
    # architecture that was never trained.
    #
    # AND BEFORE pose_direction, BECAUSE THAT IS THE TRAINER'S ORDER (phase_k_train.py:1820 then
    # :1826). This block was first written after pose_direction and pose_perblock, which inverted
    # the memory_efficient_forward nesting between the two processes whenever --frame-direction was
    # also set: both modules wrap that method, so whichever installs second is the outer wrapper,
    # and train and eval would have disagreed about which one that is. pose_direction does not
    # touch _prep_blocks, so the inversion is probably numerically harmless -- "probably" is the
    # reason to match the order rather than reason about it. NOTE --pose-inblock with
    # --frame-direction is still an UNEXERCISED combination in both processes.
    if a.pose_inblock:
        if a.pose_perblock:
            raise SystemExit(
                "STOP: --pose-inblock with --pose-perblock. Both replace "
                "pairformer_stack._prep_blocks, so one of the two injections is silently dropped "
                "depending on install order. The trainer refuses this combination; so does this.")
        import pose_inblock
        _pib_blk = a.pose_inblock_block
        if _pib_blk < 0:
            _pib_blk = len(model.confidence_head.pairformer_stack.blocks) - 1
        pose_inblock.install(model, block_index=_pib_blk, pae_inject=a.pose_inblock_pae)
    elif a.pose_inblock_pae:
        raise SystemExit("STOP: --pose-inblock-pae without --pose-inblock. The readout injection "
                         "reads the pose geometry the in-block wrapper stashes, so on its own it "
                         "would never fire and the flag would be a silent no-op.")
    if a.frame_direction:
        import pose_direction
        pose_direction.install(model)
    # SAME REASON, SAME PLACE: a head trained with --pose-perblock carries pose_perblock_proj
    # tensors the stock model does not, and the guard below refuses any tensor the model lacks.
    if a.pose_perblock:
        import pose_perblock
        pose_perblock.install(model, tied=a.pose_perblock_tied,
                              share_pretrained=a.pose_perblock_share,
                              copy_pretrained=a.pose_perblock_copy)
    elif a.pose_perblock_share or a.pose_perblock_copy:
        raise SystemExit("STOP: --pose-perblock-share/-copy without --pose-perblock changes "
                         "nothing, and scoring would silently use one-depth injection.")
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        have = dict(model.named_parameters())
        # BOTH DIRECTIONS, OR THE ARCHITECTURE CAN DIFFER SILENTLY.
        #
        # This used to check only that every checkpoint key exists in the model. The other
        # direction was unchecked, so a head whose confidence_head has MORE trainable tensors than
        # the checkpoint loaded cleanly and left the extras at their initial values -- and because
        # both pose pathways are zero-init, "initial" means zero, i.e. the pathway silently
        # absent. Review reproduced a TIED pathway-A checkpoint, which holds one projection,
        # loading into an UNTIED model, which wants three, leaving two at zero and the run
        # reporting a number for an architecture it was not given.
        #
        # The checkpoint is written as exactly the trainable head tensors, so set equality against
        # the model's own head parameters is the right test and needs no extra metadata.
        prefixes = [n for n, _m in model.named_modules() if n.endswith("confidence_head")]
        if not prefixes:
            raise SystemExit("STOP: the model has no module named *confidence_head, so the head "
                             "parameter set cannot be established to check the checkpoint "
                             "against.")
        head_names = {n for n in have
                      if any(n == pr or n.startswith(pr + ".") for pr in prefixes)}
        extra = sorted(set(sd) - set(have))
        absent = sorted(head_names - set(sd))
        if extra or absent:
            raise SystemExit(
                f"STOP: {a.ckpt} does not match this model's head.\n"
                + (f"  {len(extra)} checkpoint tensor(s) the model does not have, e.g. "
                   f"{extra[:4]}\n" if extra else "")
                + (f"  {len(absent)} head tensor(s) the checkpoint does not supply, e.g. "
                   f"{absent[:4]}\n" if absent else "")
                + f"  The head has {len(head_names)} trainable tensor(s); the checkpoint has "
                  f"{len(sd)}.\n"
                  f"  Pass the same pathway flags the run trained with -- --frame-direction, "
                  f"--pose-perblock, --pose-perblock-tied, --pose-perblock-share, "
                  f"--pose-inblock, --pose-inblock-pae -- or the "
                  f"pathway is scored as though it were absent. NOTE "
                  f"--pose-perblock-share adds NO tensors, so this guard cannot catch its "
                  f"absence: the load succeeds and the score is simply wrong.")
        with torch.no_grad():
            for k, v in sd.items():
                have[k].data.copy_(v.to(have[k].device))
        print(f"loaded {len(sd)} head tensor(s) from {a.ckpt}, exact match against "
              f"{len(head_names)} head parameter(s)")
    else:
        print("using the SHIPPED head (no --ckpt)")

    # THE POSE->SAMPLE JOIN IS CACHEABLE, and it is the only part of this loop that is. It is a
    # function of the trunk's own coordinates and the pose cifs on disk, so for a fixed validation
    # set it returns the same answer every epoch -- 3.6 min per shard, fifteen times over, for a
    # constant. The key carries the trunk's size and mtime so a refold or a rebuild invalidates it
    # rather than silently reusing a mapping to superseded poses, which is the exact failure
    # resolve_seed_dir was written to prevent.
    join_cache, join_hits, join_new = {}, 0, 0
    # ONE FILE PER SHARD. The sharded driver passes --extra to EVERY shard, so a single path would
    # have three processes writing one json and the last writer would win with a third of the
    # entries -- or catch a half-written file mid-read. Each shard owns its own, keyed by index.
    if a.join_cache and a.nshards > 1:
        _st, _ext = os.path.splitext(a.join_cache)
        a.join_cache = f"{_st}_shard{a.shard}{_ext or '.json'}"
    if a.join_cache and os.path.exists(a.join_cache):
        try:
            join_cache = json.load(open(a.join_cache))
            print(f"join cache: {len(join_cache)} entry(ies) from {a.join_cache}")
        except Exception as e:
            print(f"join cache: ignoring {a.join_cache} ({type(e).__name__}: {e})")
            join_cache = {}

    def _join_key(pdb, sd_, tp):
        st = os.stat(tp)
        return f"{pdb}/{sd_}|{st.st_size}|{int(st.st_mtime)}"

    # PREFETCH READS BYTES, NOT TENSORS. The thread only pulls the npz off disk into the page cache;
    # load_unit still does the numpy-to-device work on the main thread, so no CUDA call is made from
    # a second thread and the device tensors are built in exactly the order they were before.
    _pf_pool = None
    if a.prefetch:
        from concurrent.futures import ThreadPoolExecutor
        _pf_pool = ThreadPoolExecutor(max_workers=1)

        def _warm(path):
            try:
                with open(path, "rb", buffering=0) as fh:
                    while fh.read(8 << 20):
                        pass
            except OSError:
                pass
        print("prefetch: one background reader warming the next trunk")
    _pf_future = None

    rows, selfchk, n_nosample = [], [], 0
    _CROPPOOL = {}
    if a.crop_pool:
        if not os.path.isdir(a.crop_pool):
            raise SystemExit(f"STOP: --crop-pool {a.crop_pool} is not a directory.")
        # ONE FILE PER STRUCTURE, N masks of possibly DIFFERENT lengths -- a pool can mix rungs,
        # because a structure that found no passing crop at 256 escalates. So the masks are a
        # list of index arrays, not one rectangular block. Same reader as the trainer's.
        for _f in sorted(os.listdir(a.crop_pool)):
            if not _f.endswith(".npz"):
                continue
            _z = np.load(os.path.join(a.crop_pool, _f), allow_pickle=True)
            _ks = sorted((k for k in _z.files if k.startswith("c") and k[1:].isdigit()),
                         key=lambda k: int(k[1:]))
            _rows = [np.asarray(r, dtype=np.int64) for k in _ks for r in _z[k]]
            if _rows:
                _CROPPOOL[_f[:-4]] = _rows
        if not _CROPPOOL:
            raise SystemExit(f"STOP: --crop-pool {a.crop_pool} produced no masks.")
        _nn = [len(v) for v in _CROPPOOL.values()]
        print(f"CROP POOL: {len(_CROPPOOL)} structure(s) from {a.crop_pool}, "
              f"{min(_nn)}..{max(_nn)} crop(s) each, using index {a.crop_index} for all of them "
              f"-- a PROXY, not comparable with any uncropped number")

    # _CROPTOK AND --crop-tokens ARE GONE. The only crop rule left is --crop-pool's,
    # which the TRAINER applies; the evaluator is uncropped except for
    # --crop-predicted-margin, which derives its window from predicted coordinates
    # and so needs no mask file.
    _n_cropped = _n_nocrop = _tok_full = _tok_kept = 0
    _labels_root = a.labels or os.path.join(a.fold, "labels")
    if a.pose_split_check:
        print("POSE-SPLIT CHECK: one pose per forward pass, NO crop. These scores must reproduce "
              "the batched path exactly, or every per-pose crop number is measuring the split.")
    _crop_sizes = {}          # pdb -> [kept tokens per pose], the premise of the per-pose mode
    for n, (pdb, sd_, tp) in enumerate(units, 1):
        if _pf_pool is not None:
            if _pf_future is not None:
                _pf_future.result()          # this unit's bytes; started one iteration ago
            if n < len(units):
                _pf_future = _pf_pool.submit(_warm, units[n][2])
        _perpose = []
        _fallback_pi = set()     # cached pose indices; translated to sample ids after the join
        # THE JOIN NEEDS THE UNCROPPED COORDINATES. BC.resolve_seed_dir matches ci_coordinate
        # against the pose cifs by coordinate IDENTITY, so handing it a cropped array would break
        # the one join in this file that is not a (pdb, seed, sample) guess -- and silently, since
        # a failed join just drops the unit. The crop applies to what the HEAD consumes; px stays
        # whole. This also keeps every existing join cache valid, because the join is unchanged.
        _px_full = None
        # --crop-tokens AND --crop-margin WERE HERE and went with their flags: the first read a
        # mask whose builder no longer exists, the second derived its window from
        # label_coordinate so nothing it produced was a deployment number. --crop-pool is the one
        # rule left, and this is the evaluator's half of it.
        if a.crop_pool:
            # A MISSING POOL ENTRY IS FATAL, never a quiet fall-through to the whole structure:
            # a pass where some units were cropped and some were not would still write a complete
            # metric and name itself after the crop. Same refusal the trainer makes.
            _rows = _CROPPOOL.get(pdb)
            if _rows is None:
                raise SystemExit(f"STOP: --crop-pool has no entry for {pdb}. The pool holds "
                                 f"{len(_CROPPOOL)} structure(s); this pool needs {pdb} too.")
            if a.crop_index >= len(_rows) or a.crop_index < 0:
                raise SystemExit(f"STOP: --crop-index {a.crop_index} but {pdb} has "
                                 f"{len(_rows)} crop(s).")
            _zc = np.load(tp)
            _px_full = np.asarray(_zc["ci_coordinate"])
            _ntok = int(np.asarray(_zc["ci_feat_asym_id"]).reshape(-1).shape[0])
            _idx = _rows[a.crop_index]
            if _idx.size and int(_idx.max()) >= _ntok:
                raise SystemExit(f"STOP: {pdb} crop {a.crop_index} indexes token "
                                 f"{int(_idx.max())} but its cached trunk has {_ntok}. The pool "
                                 f"was built against a different cache and slicing with it would "
                                 f"scramble the structure.")
            _kt = np.zeros(_ntok, dtype=bool)
            _kt[_idx] = True
            # THE LABEL NPZ IS STILL NEEDED, and that is bookkeeping rather than supervision:
            # slice_cached_unit re-indexes the unit's cached label arrays alongside the features.
            # The CROP itself comes from the pool and reads no ground truth.
            _lp = os.path.join(_labels_root, pdb, f"free_{sd_}.npz")
            if not os.path.exists(_lp):
                raise SystemExit(f"STOP: --crop-pool needs the label npz to slice the unit's "
                                 f"arrays and {_lp} does not exist.")
            _lab = np.load(_lp, allow_pickle=True)
            _a2t = np.asarray(_zc["ci_feat_atom_to_token_idx"]).reshape(-1)
            _inr = (_a2t >= 0) & (_a2t < _ntok)
            _ka = np.zeros(_a2t.shape[0], dtype=bool)
            _ka[_inr] = _kt[_a2t[_inr]]
            feat, s_in, s_tr, z, x, nc = BC.slice_cached_unit(_zc, _lab, _kt, _ka, dev, torch)[:6]
            _n_cropped += 1
            _tok_full += int(_kt.shape[0]); _tok_kept += int(_kt.sum())
        elif a.pose_split_check:
            # ONE SLICE PER POSE, WITH NOTHING CROPPED. --crop-predicted-margin was deleted
            # 2026-11-13 along with --crop-tokens and --crop-margin, leaving --crop-pool (applied
            # by the TRAINER) as the only crop rule; the evaluator is uncropped. What survives
            # here is the pose SPLIT, which was the control that made the cropped numbers
            # readable and is still the way to check that splitting a unit per pose changes
            # nothing on its own.
            _zc = np.load(tp)
            _px_full = np.asarray(_zc["ci_coordinate"])
            _perpose = []
            # LOADED ONCE: all five slices are identical here, and re-reading the unit per pose
            # put five copies of a 333 MB npz on the device at once for no change in the result.
            _u6_shared = RC.load_unit(tp, dev)[:6]
            for _pi in range(_px_full.shape[0]):
                _perpose.append((_pi, _u6_shared))
            feat = _perpose[0][1][0]
        else:
            feat, s_in, s_tr, z, x, nc = RC.load_unit(tp, dev)
        if a.frame_direction and not _perpose:
            # DERIVED FROM THE TRUNK, not read from a label npz. An earlier version read
            # lf_frame_atom_index and lf_pae_rep_atom_mask from the labels beside the trunk and
            # hard-stopped when they were absent -- which made --frame-direction unscorable on
            # validation/fold, where there are 53 trunks and no labels, i.e. on the only data the
            # measurement means anything. pose_direction.derive_frames reconstructs both from
            # atom_to_token_idx and atom_to_tokatom_idx, which load_unit already returns, and that
            # reconstruction was verified EXACT against the label npz on all 500 training
            # structures with zero mismatches.
            _fai, _pae = pose_direction.derive_frames(feat)
            pose_direction.set_unit(_fai, feat["has_frame"], _pae, pdb=pdb)
        _CONF = ((lambda *_a: RC.pair_iptm_only(*_a, want_pae=a.dump_pae, want_mean=a.pool_mean))
                 if a.pair_iptm_only else RC.confidence)
        if _perpose:
            summ = []
            for _pi, (_f, _si, _st, _z, _x, _nc) in _perpose:
                if a.frame_direction:
                    # PER POSE, because each pose carries its own crop and therefore its own token
                    # numbering. Setting this once per unit would point pose 3's frames at pose
                    # 0's residues.
                    _fai, _pd = pose_direction.derive_frames(_f)
                    pose_direction.set_unit(_fai, _f["has_frame"], _pd, pdb=pdb)
                _s = _CONF(model, configs, _f, _si, _st, _z, _x[_pi:_pi + 1], _nc)
                _s = _s if isinstance(_s, (list, tuple)) else [_s]
                if len(_s) != 1:
                    raise SystemExit(f"STOP: scoring one pose returned {len(_s)} summaries. The "
                                     f"pose dimension is not what drives the output, so slicing "
                                     f"it would silently mis-pair scores with poses.")
                summ.append(_s[0])
        else:
            # THE UNIT IS NAMED WHEN THIS RAISES. A confidence-head failure 2,000 units into a
            # 2,185-unit shard printed a bare torch device error and no pdb, so the only way to
            # find the structure was to reconstruct the strided unit order by hand -- after an
            # hour of GPU had already been spent and discarded. The traceback is re-raised
            # unchanged; this only adds the one fact it was missing.
            try:
                summ = _CONF(model, configs, feat, s_in, s_tr, z, x, nc)
            except Exception as _e_conf:
                _nt = (int(feat["asym_id"].shape[-1])
                       if hasattr(feat.get("asym_id"), "shape") else "?")
                _dev = {k: str(getattr(feat.get(k), "device", "n/a"))
                        for k in ("asym_id", "has_frame", "atom_to_token_idx", "is_ligand")}
                print(f"  CONFIDENCE HEAD FAILED on {pdb}/free_{sd_}: {type(_e_conf).__name__}: "
                      f"{_e_conf}\n    tokens={_nt}  feat devices={_dev}  "
                      f"x.device={getattr(x, 'device', 'n/a')}", flush=True)
                raise
            summ = summ if isinstance(summ, (list, tuple)) else [summ]
        # join 2: pose -> sample, by coordinate identity. The pose DIRECTORY is itself resolved by
        # the trunk, because a refold leaves two finished directories for one seed and neither
        # filesystem order nor poses/index.tsv says which run wrote this trunk.
        px = (_px_full if _px_full is not None else
              (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)))
        _jk = _join_key(pdb, sd_, tp) if a.join_cache else None
        try:
            if _jk is not None and _jk in join_cache:
                unit_dir, ks = join_cache[_jk][0], list(join_cache[_jk][1])
                join_hits += 1
            else:
                unit_dir, ks = BC.resolve_seed_dir(a.fold, pdb, sd_, px, 0.01, 0.5)
                if _jk is not None:
                    join_cache[_jk] = [unit_dir, [int(v) for v in ks]]
                    join_new += 1
        except Exception as e:
            n_nosample += 1
            if n_nosample <= 5:
                print(f"  {pdb}/{sd_}: cannot join poses to samples ({type(e).__name__}: "
                      f"{str(e)[:90]})")
            continue
        # WHICH asym_id VECTOR produced each pose's matrix. Under per-pose cropping every pose
        # has its own, so one per-unit vector would remap the wrong chains.
        _asym_for = ((lambda _i: _perpose[_i][1][0]["asym_id"]) if _perpose
                     else (lambda _i: feat["asym_id"]))
        if _fallback_pi:
            # AFTER THE JOIN, NOT BEFORE. rows are keyed by the joined sample id ks[i]; the
            # fallback was detected at crop time, where only the cached pose index exists. Writing
            # the pose index meant the sidecar could not be matched to a row without redoing the
            # coordinate join -- which is the whole reason the join cache exists.
            for _pi in sorted(_fallback_pi):
                if _pi < len(ks):
                    fallback_poses.append((pdb, int(sd_), int(ks[_pi])))
        for i, s_i in enumerate(summ):
            # THE PAE DUMP RIDES THE CANONICAL JOIN. `ks` comes from BC.resolve_seed_dir, which
            # matches poses to samples BY COORDINATE IDENTITY -- the only join in this file that is
            # not a (pdb, seed, sample) guess. Writing the dump anywhere else would reintroduce
            # exactly the mismatch that made the ipTM join verification fail on 7zra_BC/7zra_AC.
            if a.dump_pae:
                # FATAL, NOT SKIPPED. want_pae=True always returns this key, which is exactly why
                # its absence means something changed underneath us -- and a pose silently missing
                # from the export while evaluation carries on is how a dataset ends up short
                # without anything saying so. Caught in review.
                if "token_pair_pae" not in s_i:
                    raise SystemExit(f"STOP: {pdb}/{sd_}: --dump-pae is on but the confidence "
                                     f"path returned no token_pair_pae.")
                _tp = s_i["token_pair_pae"]
                _tp = _tp.detach().cpu().numpy() if hasattr(_tp, "detach") else np.asarray(_tp)
                _tp = _tp[0] if _tp.ndim == 3 else _tp
                _pae_accumulate(a, pdb, _tp, f"s{a.shard}")
                _dump_pose(a, pdb, int(sd_), int(ks[i]), _tp, feat, px, i)
            cpi = s_i["chain_pair_iptm"]
            cpi = cpi.detach().cpu().numpy() if hasattr(cpi, "detach") else np.asarray(cpi)
            cpi = cpi[0] if cpi.ndim == 3 else cpi
            cim = None
            if a.pool_mean:
                if "chain_pair_itm_mean" not in s_i:
                    raise SystemExit("STOP: --pool-mean is on but the confidence path returned no "
                                     "chain_pair_itm_mean. Silently writing an empty column would "
                                     "look like 'mean pooling does nothing'.")
                cim = s_i["chain_pair_itm_mean"]
                cim = (cim.detach().cpu().numpy() if hasattr(cim, "detach")
                       else np.asarray(cim))
                cim = cim[0] if cim.ndim == 3 else cim
            k = int(ks[i])
            # JOIN 1, PER POSE. OST's own chain mapping for THIS pose, falling back to the static
            # chain_roles table only when it cannot be read -- and counting both, so the output says
            # how much of the metric rests on which.
            pose_pairs, why = ost_chain_pairs(unit_dir, pdb, k, refs_by_pdb[pdb])
            if pose_pairs is None:
                # NO SILENT FALLBACK. chain_roles has been measured wrong for chemically identical
                # chains, and with --ckpt the self-check that would have caught it does not run, so
                # an unresolvable mapping is recorded and the run stops rather than scoring.
                n_static += 1
                unmapped.append(f"{pdb}/{sd_} sample {k}: {why}")
                if a.allow_roles_fallback:
                    pose_pairs = pairs_by_pdb[pdb]
                else:
                    continue
            else:
                n_ost += 1
                if pose_pairs != pairs_by_pdb[pdb]:
                    n_differ += 1
                    differ_pdbs[pdb] += 1
            # PROTENIX COMPACTS THE ASYM IDS before it builds chain_pair_iptm
            # (sample_confidence.py:508-512: when len(unique) != max+1 it remaps them to
            # 0..N-1). A crop that removes a WHOLE chain therefore shifts every row of that
            # matrix, and indexing it with the original ids reads a different chain pair --
            # silently, because the old `max(ai,aj) >= cpi.shape[0]` guard just dropped the
            # overflowing pair and scored the rest against the wrong partner. Rebuilding the
            # same mapping from the asym_id actually handed to the head is the fix; it is the
            # identity whenever nothing was dropped, so uncropped runs are unaffected.
            _cmap = _cpi_index(_asym_for(i))
            for iface, ai, aj in pose_pairs:
                mi, mj = _cmap.get(int(ai)), _cmap.get(int(aj))
                if mi is None or mj is None:
                    # AN EVALUATED CHAIN DID NOT SURVIVE THE CROP. There is no score for a pair
                    # whose chain is absent, and inventing one from a neighbouring row is the
                    # bug above. Recorded per pose and surfaced at the end.
                    n_chain_gone += 1
                    if len(chain_gone) < 20:
                        chain_gone.append(f"{pdb}/{sd_} sample {k}: {iface} needs asym "
                                          f"{ai},{aj}; crop kept {sorted(_cmap)}")
                    continue
                if max(mi, mj) >= cpi.shape[0]:
                    raise SystemExit(f"STOP: {pdb}/{sd_} sample {k}: remapped chain index "
                                     f"{mi},{mj} is outside a {cpi.shape[0]}-chain matrix. The "
                                     f"asym_id handed to the head and the matrix it returned "
                                     f"disagree.")
                # THE TWO MATRICES DO NOT SHARE AN INDEXING CONVENTION, and this is the one
                # place that reads both.
                #   cpi  comes from Protenix, which COMPACTS asym ids to 0..N-1
                #        (sample_confidence.py:508-512)            -> index with mi, mj
                #   cim  comes from our _chain_pair_itm_mean, which allocates max(id)+1 rows and
                #        writes at the ORIGINAL ids
                #        (phase_j_rescore_cache.py:155)            -> index with ai, aj
                # Using the compacted index on BOTH was the bug: with chains 0,2,3 surviving, the
                # MEAN for pair (0,2) read row 1, an all-zero row, and (0,3) read (0,2)'s value.
                # They agree whenever no chain is missing, which is why this never showed up
                # uncropped.
                v = float(cpi[mi, mj])
                vm = None
                if cim is not None:
                    if max(int(ai), int(aj)) >= cim.shape[0]:
                        raise SystemExit(
                            f"STOP: {pdb}/{sd_} sample {k}: original chain index {ai},{aj} is "
                            f"outside a {cim.shape[0]}-row MEAN matrix. That matrix is allocated "
                            f"at max(asym_id)+1, so this means the pair's chains are not in the "
                            f"asym_id the head was given.")
                    vm = float(cim[int(ai), int(aj)])
                key = (iface, sd_, k)
                rows.append((iface, sd_, k, v, dockq.get(key), vm))
                if key in stored_iptm:
                    # THE KEY, NOT JUST THE MAGNITUDE. Reporting "1014 rows disagree, max 0.2378"
                    # says a join is wrong and nothing about WHICH, so the next step is a 73-minute
                    # re-run to find out. Carrying the key costs nothing and makes the failure
                    # actionable: whether it is a handful of interfaces (join 1), a set of seeds
                    # (join 2) or scattered (join 3) is visible in the output itself.
                    selfchk.append((abs(v - stored_iptm[key]), iface, sd_, k,
                                    v, stored_iptm[key]))
        if n % 200 == 0:
            print(f"  ...{n}/{len(units)} units", flush=True)

    if _pf_pool is not None:
        if _pf_future is not None:
            _pf_future.result()
        _pf_pool.shutdown(wait=True)

    # WRITTEN ONLY ON A CLEAN PASS. The cache is a shortcut past resolve_seed_dir, which exists to
    # refuse an ambiguous pose directory; persisting entries from a pass that then failed to join
    # some unit would hand the next pass a half-built shortcut. So it is written after the
    # no-join check below, not here. The counters print either way.
    # THE LAST COMPLEX'S PARTIAL SUM. Without this flush the final pdb of every shard would have
    # no _paesum file and its pae_col_mean would be computed from fewer poses than it has -- the
    # same semantic error the accumulator exists to prevent, applied to one complex per shard.
    if a.dump_pae:
        _pae_flush(a, f"s{a.shard}")
        print(f"  PAE means: partial sums written as _paesum_s{a.shard}.npz "
              f"({a.nshards} shard(s)); run abag_rank/merge_pae_mean.py before building features")
    if a.join_cache:
        print(f"join cache: {join_hits} hit(s), {join_new} newly derived "
              f"({join_hits / max(1, join_hits + join_new) * 100:.0f}% reused)")

    if n_nosample:
        raise SystemExit(f"STOP: {n_nosample} unit(s) could not be joined to their sample cifs. "
                         f"Selecting a pose and reading another sample's DockQ would score the "
                         f"wrong thing, so this is fatal rather than skipped.")

    if a.join_cache and join_new:
        tmp = a.join_cache + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(join_cache, fh)
        os.replace(tmp, a.join_cache)       # atomic: a half-written cache is the one a reader takes
        print(f"join cache -> {a.join_cache} ({len(join_cache)} entry(ies))")

    # THE CROP IS REPORTED, NEVER ASSUMED. A --crop-margin that silently cropped nothing would
    # produce an "uncropped" number under a cropped label, which is the whole failure this run is
    # about.
    if a.crop_pool:
        _kf = (_tok_kept / _tok_full) if _tok_full else float("nan")
        print(f"\ncrop: {_n_cropped} unit(s) on --crop-pool index {a.crop_index}; kept "
              f"{_tok_kept}/{_tok_full} token(s) ({100 * _kf:.1f}%). PROXY -- not comparable "
              f"with an uncropped number.")
    # THE PER-POSE CROP REPORT WAS HERE, with its kept-token fraction and the crop-size
    # spread diagnostic. It went with --crop-predicted-margin on 2026-11-13; the evaluator
    # no longer crops anything, so there is nothing to report.

    print(f"\njoin 1: OST's per-pose chain mapping used on {n_ost} pose(s), unresolvable on "
          f"{n_static}")
    if a.rows_out:
        # The shard's own work is done and verified per pose: every unit joined to its samples, and
        # every pose's chain pair resolved from OST or refused. What CANNOT be judged here is
        # coverage -- a shard holds a slice by construction -- so the rows go out and the aggregate
        # checks wait for the union.
        # WRITE FIRST, REFUSE SECOND. An unmapped pose is a bookkeeping failure at the join,
        # discovered after every unit in the shard has already been through the GPU. Raising
        # before the write threw away a measured ~50 min of finished forward passes on two of
        # three shards, so the rows and the meta go to disk unconditionally and the refusal
        # comes after. The meta records the unmapped poses and the aggregate refuses to merge
        # a shard that has any, so a partial shard can be kept without being silently counted.
        with open(a.rows_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iface", "seed", "sample", "iptm", "dockq", "stored_iptm"]
                       + (["iptm_mean"] if a.pool_mean else []))
            for iface, sd2, k2, v2, dq, vm in rows:
                key = (iface, sd2, k2)
                sv = stored_iptm.get(key)
                # FULL PRECISION, NOT 6 DECIMALS. repr() round-trips a float exactly.
                # Rounding here manufactured ties that do not exist in the computed
                # values, and the selection is an argmax -- so a tie was broken by
                # whichever row the merge read first, which depends on shard order.
                # Review reproduced a selected DockQ moving 0.9 -> 0.1 from the rounding
                # alone. dockq keeps 4 decimals: that is the precision results.csv has.
                w.writerow([iface, sd2, k2, repr(v2),
                            "" if dq is None else f"{dq:.4f}",
                            "" if sv is None else repr(sv)]
                           + ([("" if vm is None else repr(vm))] if a.pool_mean else []))
        with open(a.rows_out + ".meta.json", "w") as fh:
            json.dump({"shard": a.shard, "nshards": a.nshards, "units": len(units),
                       "rows": len(rows), "n_ost": n_ost, "n_unresolvable": n_static,
                       "n_differ": n_differ, "differ_pdbs": dict(differ_pdbs),
                       "n_unmapped": len(unmapped), "unmapped": unmapped[:50],
                       "budget": a.budget, "ckpt": a.ckpt}, fh, indent=1)
        print(f"shard {a.shard}: {len(rows)} row(s) -> {a.rows_out}")
        if unmapped and a.drop_unmapped_poses:
            for u in unmapped[:10]:
                print(f"    {u}")
            print(f"  --drop-unmapped-poses: {len(unmapped)} pose(s) in this shard have no usable "
                  f"OST chain mapping and were NOT written. They are named in the meta; their "
                  f"interfaces will be short of depth and must be dropped by name at the merge.",
                  flush=True)
            return
        if unmapped:
            for u in unmapped[:10]:
                print(f"    {u}")
            raise SystemExit(f"STOP: {len(unmapped)} pose(s) in this shard have no usable OST chain "
                             f"mapping and none forced by elimination. The rows that DID resolve "
                             f"are on disk and the meta names the ones that did not; the aggregate "
                             f"will refuse this shard until they are resolved or excluded.")
        return
    if unmapped and not a.allow_roles_fallback:
        for u in unmapped[:10]:
            print(f"    {u}")
        raise SystemExit(
            f"STOP: {len(unmapped)} pose(s) have no usable OST chain mapping and none that is forced "
            f"by elimination. chain_roles.csv is NOT an acceptable substitute -- it is measured wrong "
            f"for chemically identical chains, and with --ckpt the shipped-head self-check does not "
            f"run, so a wrong chain pair here would mis-score silently. Pass "
            f"--allow-roles-fallback only if you have established the static table is right for "
            f"these structures, and say why.")
    if unmapped:
        print(f"  FELL BACK to chain_roles on {len(unmapped)} pose(s) because "
              f"--allow-roles-fallback was given; the static table is wrong for chemically "
              f"identical chains, so this metric is only as good as that assumption")
    if n_differ:
        print(f"  the two DISAGREED on {n_differ} pose(s), across {len(differ_pdbs)} structure(s): "
              + ", ".join(f"{k}:{v}" for k, v in differ_pdbs.most_common(8)))
        print(f"  OST's is authoritative -- it is the mapping results.csv's own DockQ was computed "
              f"under, and for chemically identical chains a static table cannot be right for every "
              f"pose.")

    aggregate(a, rows, selfchk, want_ifaces)

if __name__ == "__main__":
    sys.exit(main())
