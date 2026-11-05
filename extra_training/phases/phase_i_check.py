#!/usr/bin/env python
"""Phase I's verification and score recovery. Called by phase_i_fold_validation.sh.

WHY THIS IS PYTHON AND NOT SHELL. The shell version of both routines was wrong in ways shell makes
easy: it read pool.csv with `IFS=, read`, which keeps the \\r of a CRLF file in the last field and
produced interfaces like "6mnq_AB\\r" that scorer.py rejects outright; and its verification let a
crashed subprocess fall back to zero failures. Both needed a real CSV reader and a real exit status.

    verify   exact coverage of expected (pdb, seed, interface, sample), finite scores, readable
             caches. Exit 1 on any shortfall.
    rescore  score only what is missing, with the dataset's own scorer, without duplicating rows.

COMPLETION MEANS SCORED AND CACHED, NOT FOLDED. folder.py writes run.json before scoring, so counting
run.json files announces success over units that have no interface scores. And a run.json is only
evidence for the unit it names: globbing poses/<pdb>/<seed>-*/run.json without reading it let an
unrelated run compensate for a missing expected one.
"""
import fcntl
import argparse
import collections
import csv
import glob
import json
import math
import os
import subprocess
import sys

# the repo root, from this file, so renaming the checkout needs no edit here
_REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_logging"))
import pose_sets                                                           # noqa: E402


def read_pool(path):
    """in: pool.csv.  out: {pdb: {iface}} -- via the csv module, so CRLF cannot leak into a name"""
    out = collections.defaultdict(set)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            pdb = r["pdb"].strip()
            out[pdb].add(f"{pdb}_{r['chain1'].strip()}{r['chain2'].strip()}")
    return out


def read_results(path):
    """in: results.csv.  out: {(iface, seed): {sample: (dockq, iptm)}}, and the header's field names"""
    have = collections.defaultdict(dict)
    if not os.path.exists(path):
        return have
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                have[(r["iface"].strip(), r["seed"].strip())][r["sample"].strip()] = (
                    r.get("dockq", ""), r.get("iptm", ""))
            except KeyError:
                continue
    return have


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def read_skips(path):
    """in: val_fold/skipped.csv.  out: {pdb: reason}

    A SKIP IS A RECORDED DECISION, NOT AN ABSENCE. Settled 2026-10-11: a structure that will not fold
    is dropped from the validation set rather than accommodated by reducing N_sample or N_cycle,
    because changing the fold settings part-way would make the cached tensors non-comparable across
    the set -- and comparability is the whole point of caching them.

    So the skip has to be written down with a reason, and verify() reports the skipped structures
    separately instead of folding them into "unfolded". Without this file a skip and a fault look
    identical, and "INCOMPLETE" in the morning would not say which it was.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            pdb = str(r.get("pdb", "")).strip()
            why = str(r.get("reason", "")).strip()
            if pdb:
                out[pdb] = why or "NO REASON RECORDED"
    return out


def expected_units(jobs, suffix, seeds, seed0):
    pdbs = sorted(os.path.basename(f)[:-len(suffix)] for f in glob.glob(os.path.join(jobs, "*" + suffix)))
    return pdbs, [(p, str(seed0 + i)) for p in pdbs for i in range(seeds)]


def run_dir_for(E, pdb, seed):
    """out: the run directory whose run.json NAMES this (pdb, seed), or None

    Reading it matters: a directory merely matching poses/<pdb>/<seed>-* could belong to another run,
    and globbing alone let an unrelated run.json stand in for a missing expected unit.
    """
    for d in sorted(glob.glob(os.path.join(E, "poses", pdb, f"{seed}-*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        try:
            with open(rj) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(m.get("pdb")) == pdb and str(m.get("seed")) == str(seed):
            return d
    return None


def check_npz(path, nsample):
    """out: "" if the cache is usable for an offline replay, else why not"""
    if not os.path.exists(path):
        return "absent"
    try:
        import numpy as np
        d = np.load(path, allow_pickle=True)
        # THE SAME LIST THE SMOKE TEST ASSERTS. If verify required less than smoke, a cache that
        # smoke would have rejected could pass the final check on 1,540 units.
        need = ["z_trunk", "s_trunk", "ci_s_inputs", "ci_s_trunk_f32", "ci_coordinate",
                "ci_contact_probs", "ci_N_cycle", "ci_z_dtype", "ci_s_dtype",
                "ci_s_inputs_dtype", "rng_py", "rng_torch"] + [f"ci_feat_{k}" for k in (
                    "distogram_rep_atom_mask", "atom_to_token_idx", "atom_to_tokatom_idx",
                    "asym_id", "has_frame", "is_ligand")] + ["dtype"]
        missing = [k for k in need if k not in d.files]
        if missing:
            return f"missing {missing[:4]}"
        sh = [x for x in d["ci_coordinate"].shape]
        while len(sh) > 1 and sh[0] == 1:
            sh.pop(0)
        if not sh or sh[0] != nsample:
            return f"ci_coordinate holds {sh[0] if sh else 0} poses, expected {nsample}"
        if str(d["dtype"]) != "bf16":
            return f"dtype {d['dtype']}, expected bf16"
        # THE TENSORS MUST BE USABLE, not merely present. A cache passing the key list can still hold
        # NaN coordinates or a z_trunk of the wrong rank, and both read as a clean verify while an
        # offline replay on them produces nonsense.
        z, st_, si = d["z_trunk"], d["s_trunk"], d["ci_s_inputs"]
        if z.ndim != 3 or z.shape[0] != z.shape[1]:
            return f"z_trunk shape {tuple(z.shape)}, expected [N, N, c]"
        n_tok = z.shape[0]
        if st_.ndim != 2 or st_.shape[0] != n_tok:
            return f"s_trunk shape {tuple(st_.shape)}, expected [{n_tok}, c]"
        if si.ndim != 2 or si.shape[0] != n_tok:
            return f"ci_s_inputs shape {tuple(si.shape)}, expected [{n_tok}, c]"
        xyz = d["ci_coordinate"]
        if xyz.ndim < 2 or xyz.shape[-1] != 3:
            return f"ci_coordinate shape {tuple(xyz.shape)}, expected [..., n_atom, 3]"
        n_atom = xyz.shape[-2]
        # ATOM AND TOKEN MAPPINGS MUST AGREE WITH THOSE SHAPES, or the confidence head would gather
        # from the wrong rows and the cache would silently describe a different molecule.
        for k, n in (("ci_feat_atom_to_token_idx", n_atom),
                     ("ci_feat_atom_to_tokatom_idx", n_atom),
                     ("ci_feat_distogram_rep_atom_mask", n_atom),
                     ("ci_feat_is_ligand", n_atom),
                     ("ci_feat_asym_id", n_tok),
                     ("ci_feat_has_frame", n_tok)):
            a = d[k]
            if a.shape[0] != n:
                return f"{k} has {a.shape[0]} entries, expected {n}"
        a2t = d["ci_feat_atom_to_token_idx"]
        if len(a2t) and (int(a2t.max()) >= n_tok or int(a2t.min()) < 0):
            return (f"atom_to_token_idx ranges {int(a2t.min())}..{int(a2t.max())}, outside "
                    f"[0, {n_tok - 1}]")
        # FINITENESS LAST, because it is the most expensive and the cheapest checks should fire first.
        for k in ("ci_coordinate", "ci_s_inputs", "ci_s_trunk_f32", "ci_contact_probs"):
            if not np.isfinite(d[k]).all():
                return f"{k} holds NaN or inf"
    except Exception as e:                                      # noqa: BLE001
        return f"unreadable: {type(e).__name__}"
    return ""


def _ost():
    """out: the ost_dockq module, or None

    OST'S OWN PARSER, not a hand-rolled one. The project rule is that chain mapping and DockQ come
    from OST; reimplementing "does this json contain this interface" here would be a second, divergent
    reader of a format we do not own.
    """
    import importlib.util
    p = f"{_REPO}/eval/ost_dockq.py"
    if not os.path.exists(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("ost_dockq_ro", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    except Exception:                                           # noqa: BLE001
        return None


def classify_missing(rundir, scorer, k, iface=None, invalidate=False):
    """in: a run directory, the scorer name, a sample index.  out: ("", why) or (bucket, why)

    A MISSING ROW IS NOT ALWAYS A MISSING SCORE. scorer.py:85 skips a pose in which the two chains do
    not contact at all -- "this pose has no such interface -- a real condition, not an error" -- so a
    legitimately non-contacting pose produces no row. Demanding nsample rows unconditionally would
    report INCOMPLETE forever and send rescore() into a loop that re-scores the same poses and
    appends nothing.

    The two cases are distinguishable on disk. scorer.py:81 writes <dir>/<scorer>_json/<k>.json
    through mod.score() BEFORE it checks the interface, so that file existing proves pose k was put
    through the scorer. And the pose itself must exist at all:

        no pose cif            the fold never produced that sample -- a fault, and NOT recoverable by
                               rescoring, because repeating a seed does not reproduce its poses
        pose cif, no scorer json   scoring never reached this pose -- a fault rescore() can fix
        pose cif and json      scored, and this pose has no such interface -- expected, reported
    """
    cif = glob.glob(os.path.join(rundir, "**", f"*_sample_{k}.cif"), recursive=True)
    if not cif:
        return "no_pose", f"sample {k}: no pose cif"
    sj = os.path.join(rundir, f"{scorer}_json", f"{k}.json")
    if not os.path.exists(sj):
        return "unscored", f"sample {k}: pose exists, never scored"

    # THE JSON IS PARSED, NOT MERELY COUNTED. Its existence was treated as proof the pose went
    # through the scorer, but ost_dockq.run():76 reuses ANY nonempty file as a result -- so a
    # truncated json from a killed OST run is cached forever, parse() returns {}, no row is written,
    # and "the json exists" turned a permanent scoring failure into "no contact". Five such files
    # made a fold with zero score rows report COMPLETE.
    mod = _ost() if scorer == "ost" else None
    if mod is None:
        return "unscored", f"sample {k}: cannot verify the {scorer} cache (parser unavailable)"
    # GUARDED. parse()'s docstring promises it returns {} rather than raising on a truncated file,
    # and for truncated JSON it does -- but a file that parses as JSON with a short
    # dockq_interfaces entry raises IndexError (ost_dockq.py:120). A bad cache must not take the
    # verifier down with it, so any exception here is treated as an unusable cache.
    try:
        res = mod.parse(sj)
    except Exception as e:                                      # noqa: BLE001
        res = {}
        why_parse = f"{type(e).__name__}"
    else:
        why_parse = "empty"

    # A SUCCESSFUL RUN THAT MAPPED NO INTERFACE IS NOT AN UNUSABLE CACHE. parse() returns {} both
    # for a truncated file and for a complete OST result whose dockq_interfaces is [] --
    # ost_dockq.usable() accepts the latter as a real outcome, and this did not. Raised in review,
    # and it had a concrete cost: with invalidate=True this moved a VALID json aside and sent the
    # unit back through OST, which on the same input, options and version returns exactly the same
    # empty result. 200 training units are in that state; re-running the same automatic mapping is
    # not a repair. usable() is not called directly here because it renames a file that does not
    # parse, and classify_missing must stay read-only unless invalidate says otherwise.
    try:
        raw = json.load(open(sj))
    except (OSError, ValueError):
        raw = None
    complete_run = isinstance(raw, dict) and isinstance(raw.get("dockq_interfaces"), list)
    if not res and complete_run and not raw["dockq_interfaces"]:
        return "unmapped", (f"sample {k}: {scorer} ran and mapped no interface at all "
                            f"(dockq_interfaces []), so re-running it changes nothing")
    if not res:
        # INVALIDATED, or the retry is futile: run() would reuse this file and never re-invoke OST.
        if invalidate:
            os.replace(sj, sj + ".invalid")
            return "unscored", (f"sample {k}: unusable {scorer} cache ({why_parse}), "
                                "moved aside for a retry")
        return "unscored", f"sample {k}: unusable {scorer} cache ({why_parse}): {sj}"

    # AN EXPLICIT PER-INTERFACE OUTCOME. A valid json that does report this chain pair, while
    # results.csv has no row for it, is not "no contact" -- it is a row that went missing.
    if iface:
        chains = iface.split("_")[-1]
        if len(chains) == 2:
            v = mod.interface(res, chains[0], chains[1])
            # PRESENT IN THE CACHE AT ALL MEANS IT IS NOT A NO-CONTACT. Testing
            # `v.get("dockq") is not None` sent the dockq=None case to the no-contact branch, and
            # that case is a scoring failure, not an absent interface: parse() builds an entry for
            # every dockq_interfaces row and fills dockq from a parallel list, so a cache with
            # dockq_interfaces=[["A","B","A","B"]] and dockq=[] yields {"A,B": {"dockq": None}} --
            # OST reported the interface and produced no number for it. Five such caches made a
            # fold with zero score rows report COMPLETE.
            if v is not None:
                if v.get("dockq") is None:
                    if invalidate:
                        os.replace(sj, sj + ".invalid")
                        return "unscored", (f"sample {k}: {scorer} cache reports this interface "
                                            "with no usable DockQ, moved aside for a retry")
                    return "unscored", (f"sample {k}: {scorer} cache reports this interface but its "
                                        f"DockQ is null -- a scoring failure, not a no-contact ({sj})")
                return "unscored", (f"sample {k}: the {scorer} cache reports this interface "
                                    f"(dockq {v['dockq']:.4f}) but results.csv has no row")
    # ABSENT IS TWO DIFFERENT OUTCOMES, AND ONLY ONE OF THEM IS A NO-CONTACT. OST's
    # dockq_interfaces holds the reference interfaces it COULD MAP, so a pair missing from it means
    # either that the reference's own chains do not touch -- nothing will ever produce a row -- or
    # that one of them was never mapped to a model chain, which is a limitation of the mapping and
    # not a statement about the pose. chain_mapping distinguishes them, and the earlier text called
    # both "none is this chain pair". Raised in review.
    cm = raw.get("chain_mapping") if isinstance(raw, dict) else None
    if iface and isinstance(cm, dict):
        chains = iface.split("_")[-1]
        if len(chains) == 2:
            unmapped = [c for c in chains if c not in cm]
            if unmapped:
                return "unmapped", (f"sample {k}: {scorer} left reference chain {unmapped} "
                                    f"unmapped, so this interface was never a candidate "
                                    f"({len(res)} other interface(s) scored)")
    return "", f"sample {k}: scored, {len(res)} interface(s) found, none is this chain pair"


def ref_touching(refs, iface):
    """out: True / False / None -- does this interface's chain pair contact in the REFERENCE?

    WHY VERIFY MEASURES THIS. classify_missing() can say that the scorer ran, mapped both of the
    interface's reference chains, and still reported no DockQ for the pair. It called that a
    non-contact, which is the right guess but was never more than a guess. The guess happened to
    hold for all 550 interface-samples that carried the label -- measured in 2e49f389, 0 contacts
    against 4-414 for their scored siblings -- and a guess that happens to hold is exactly the kind
    that stops holding without anyone noticing. So the claim is now checked per interface, and a
    pair that DOES touch but has no score is reported as a gap instead of a result.
    """
    sys.path.insert(0, os.path.join(_REPO, "extra_training", "run_logging"))
    import contacts as _contacts
    chains = iface.split("_")[-1]
    if len(chains) != 2:
        return None
    return _contacts.touching(os.path.join(refs, f"{iface.split('_')[0]}.cif"),
                              chains[0], chains[1])


def queue_units(a):
    """out: [(pdb, seed, arm)] -- the units this invocation is responsible for

    WHY BOTH MODES NEED THE SAME READER. rescore honoured --worklist and verify did not: verify
    enumerated every job json in --jobs instead, so on the training fold it expected 856 structures
    times 10 seeds -- 8,560 units -- and failed on the first job that is not part of the queued
    3,560 at all (8gpu, which has no interfaces in pool.csv because it was never queued). The scope
    of a check and the scope of a recovery have to be the same thing, or the check answers a
    question nobody asked.
    """
    if a.worklist:
        sys.path.insert(0, os.path.join(_REPO, "run_model"))
        _argv, sys.argv = sys.argv, ["folder.py"]
        import folder
        sys.argv = _argv
        units = [(u[1], str(u[4]), u[3]) for u in folder.read_queue(a.worklist)]
        seen, uniq = set(), []
        for u in units:
            if u not in seen:
                seen.add(u); uniq.append(u)
        units = uniq
        print(f"worklist: {len(units)} unit(s) from {a.worklist}")
        if a.expect_units and len(units) != a.expect_units:
            raise SystemExit(f"STOP: worklist holds {len(units)} unit(s), expected "
                             f"{a.expect_units}. Refusing to rescore a scope I was not asked for.")
    else:
        _pdbs, u2 = expected_units(a.jobs, a.suffix, a.seeds, a.seed0)
        units = [(p, sd, "free") for p, sd in u2]
        print(f"worklist: {len(units)} unit(s) enumerated from {a.jobs} "
              f"({a.seeds} seed(s) from {a.seed0}); pass --worklist to use the queue instead")
    return units


def verify(a):
    pool = read_pool(os.path.join(a.e, "pool.csv"))
    have = read_results(os.path.join(a.e, "results.csv"))
    scope = queue_units(a)
    units = [(p, sd) for p, sd, _arm in scope]
    pdbs = sorted({p for p, _sd, _arm in scope})
    if not pdbs:
        print(f"FAIL no unit(s) in scope: no job files matching *{a.suffix} in {a.jobs}"
              + (f" and nothing readable in {a.worklist}" if a.worklist else ""))
        return 1
    skips = read_skips(os.path.join(a.e, "skipped.csv"))
    if skips:
        print(f"skipped by decision: {len(skips)} structure(s) -- excluded from the expected set")
        for pdb, why in sorted(skips.items()):
            print(f"   {pdb}: {why}")
        # A SKIP WITHOUT A REASON IS NOT A DECISION. Left unchecked, skipped.csv becomes the place
        # anything inconvenient goes, and the validation set quietly shrinks to whatever folded.
        blank = [p for p, w in skips.items() if w == "NO REASON RECORDED"]
        if blank:
            print(f"FAIL {len(blank)} skip(s) carry no reason: {blank}. A skip is a recorded "
                  "decision; without a reason it is indistinguishable from a fault.")
            return 1
        unknown = [p for p in skips if p not in pdbs]
        if unknown:
            print(f"FAIL skipped.csv names {len(unknown)} structure(s) that are not in the job set: "
                  f"{unknown}. Either the skip is stale or the job is missing.")
            return 1
        pdbs = [p for p in pdbs if p not in skips]
        units = [(p, sd) for p, sd in units if p not in skips]

    # EVERY ACTIVE JOB MUST BE IN pool.csv. The per-unit loop below iterates pool.get(pdb, ()), so a
    # structure absent from pool.csv has an EMPTY interface list, every pose and score check is
    # skipped for it, and verify reports COMPLETE on a structure it never looked at. pool.csv is what
    # names the interfaces to score, so a job missing from it is a fold that scores nothing.
    unpooled = [p for p in pdbs if not pool.get(p)]
    if unpooled:
        print(f"FAIL {len(unpooled)} job(s) have no interfaces in pool.csv: {unpooled[:10]}. Their "
              "poses and scores would not be checked at all, and the fold scores nothing for them.")
        return 1

    unfolded, no_rows, bad_scores, bad_cache = [], [], [], []
    no_pose, no_contact, short_poses, unmapped = [], [], [], []
    touching_but_missing, uncheckable = [], []
    verified_absent = set()      # (iface) measured to have no contact in the reference
    # WHAT IS INCOMPLETE PER STRUCTURE, not just in total. 230 outstanding units spread over two
    # structures and 230 spread over ninety are different problems, and the total cannot tell them
    # apart. Raised in review together with keeping the buckets separate.
    by_pdb = {}
    for pdb, seed in units:
        d = run_dir_for(a.e, pdb, seed)
        if d is None:
            unfolded.append((pdb, seed))
            continue
        # POSE FILES ARE CHECKED INDEPENDENTLY OF pool.csv. The interface loop below can only see
        # what pool.csv names; a unit that produced too few poses would otherwise be invisible
        # whenever its interfaces happen to be scored from the poses that do exist.
        n_cif = len(glob.glob(os.path.join(d, "predictions", "*_sample_*.cif")))
        if n_cif != a.nsample:
            short_poses.append((pdb, seed, n_cif))
        # EXACT SAMPLE COVERAGE, not "a row exists". One of nsample samples scored is not a scored
        # unit, and a blank or NaN score is not a score.
        for ifc in sorted(pool.get(pdb, ())):
            got = have.get((ifc, seed), {})
            for k in range(a.nsample):
                if str(k) in got:
                    continue
                bucket, why = classify_missing(d, a.scorer, k, iface=ifc)
                if bucket == "no_pose":
                    no_pose.append((ifc, seed, why))
                elif bucket == "unscored":
                    no_rows.append((ifc, seed, why))
                elif bucket == "unmapped":
                    # THE THIRD CATEGORY, KEPT SEPARATE ON PURPOSE. The scorer ran and the pose is
                    # fine; OST mapped no model chain to one of this interface's reference chains,
                    # so the interface was never a candidate. Re-running the identical call cannot
                    # produce a row, which is why rescore() does not retry these -- but they are
                    # missing scores, not the "no contact in this pose" result below, and counting
                    # them as either one overstates how complete the dataset is.
                    unmapped.append((ifc, seed, why))
                else:
                    # MEASURED, NOT ASSUMED. The scorer's silence about this pair is consistent
                    # with the chains not touching, and also with a mapping that put the pose on
                    # the wrong chains -- those need different answers, so the reference decides.
                    t = ref_touching(a.refs, ifc)
                    if t is False:
                        verified_absent.add(ifc)
                        no_contact.append((ifc, seed, why))
                    elif t is True:
                        touching_but_missing.append((ifc, seed, why))
                    else:
                        uncheckable.append((ifc, seed, why))
                if bucket:
                    lbl = bucket
                    if bucket == "no_contact" or bucket == "":
                        lbl = "no_contact"
                    by_pdb.setdefault(ifc.split("_")[0], {}).setdefault(lbl, 0)
                    by_pdb[ifc.split("_")[0]][lbl] += 1
            for k, (dq, ip) in got.items():
                if not finite(dq) or not finite(ip):
                    bad_scores.append((ifc, seed, k, dq, ip))
        # ONLY WHEN THE FOLD WROTE A CACHE. The training fold deliberately writes none, and a
        # verifier that reported 9,660 missing npz would bury the failures that matter.
        if a.trunks:
            why = check_npz(os.path.join(a.e, "trunks", pdb, f"free_{seed}.npz"), a.nsample)
            if why:
                bad_cache.append((pdb, seed, why))

    print(f"expected units {len(units)} over {len(pdbs)} structures"
          + (f" ({len(skips)} skipped by decision)" if skips else "") + "; "
          f"interfaces per structure {sum(len(v) for v in pool.values())} total")
    print(f"  unfolded units                     : {len(unfolded)}")
    print(f"  units with != {a.nsample} pose files            : {len(short_poses)}")
    print(f"  poses absent (unrecoverable)       : {len(no_pose)}")
    print(f"  poses never scored (rescore fixes) : {len(no_rows)}")
    print(f"  rows with a non-finite score       : {len(bad_scores)}")
    print(f"  unusable or absent trunk caches    : {len(bad_cache)}")
    print(f"  scored but the interface was never mapped (rescore cannot fix): {len(unmapped)}")
    print(f"  NO SCORE THOUGH THE REFERENCE CHAINS TOUCH : {len(touching_but_missing)}")
    print(f"  no score and the contact could not be checked: {len(uncheckable)}")
    # REPORTED, NOT FAILED. A pose in which the two chains do not contact has no DockQ to report;
    # scorer.py treats that as a real condition and so does this. It is printed because a large
    # number of them is informative about the target, not about the pipeline.
    print(f"  (no contact at that interface IN THE REFERENCE, measured, so there is no DockQ to "
          f"report: {len(no_contact)} sample(s) over {len(verified_absent)} interface(s))")
    # THE DENOMINATOR CANNOT INCLUDE WHAT CANNOT EXIST. pool.csv pairs every antibody chain with
    # every antigen chain and never checks contact, so it names interfaces that are two chains
    # sitting apart in the structure. Counting those as expected scores made the dataset look
    # permanently incomplete at 95.7% when the reachable total was 97.3%, and no amount of
    # rescoring could ever close the difference.
    expected = sum(len(pool.get(p, ())) for p, _sd in units) * a.nsample
    print(f"  expected interface-samples: {expected} named by pool.csv, "
          f"{expected - len(no_contact)} of them scoreable "
          f"({len(no_contact)} have no contact in the reference)")
    for lbl, items in (("unfolded", unfolded), ("short poses", short_poses), ("no pose", no_pose),
                       ("unscored", no_rows), ("unmapped", unmapped),
                       ("TOUCHING BUT UNSCORED", touching_but_missing),
                       ("contact uncheckable", uncheckable),
                       ("non-finite", bad_scores), ("cache", bad_cache)):
        for x in items[:3]:
            print(f"     {lbl}: {x}")
    gaps = {p: v for p, v in by_pdb.items() if set(v) - {"no_contact"}}
    if gaps:
        print(f"  structures with a missing score: {len(gaps)} of {len(pdbs)}")
        for pdb in sorted(gaps, key=lambda x: -sum(v for k, v in gaps[x].items()
                                                   if k != "no_contact")):
            print(f"     {pdb}: " + ", ".join(f"{k} {v}" for k, v in sorted(gaps[pdb].items())))
    # short_poses COUNTS TOWARDS THE VERDICT. A unit that wrote fewer poses than asked for is an
    # incomplete fold whether or not pool.csv happens to name an interface that the surviving poses
    # can be scored at.
    # unmapped COUNTS TOWARDS THE VERDICT TOO, in its own words. Those interface-samples have no
    # DockQ, and a run that calls itself COMPLETE with eight hundred of them would be the overclaim
    # this verifier exists to prevent -- even though no rescore can recover them.
    rc = 1 if (unfolded or short_poses or no_pose or no_rows or bad_scores or bad_cache
               or unmapped or touching_but_missing or uncheckable) else 0
    if rc == 0:
        print("COMPLETE: every expected unit folded, every pose scored at every interface, "
              "every cache replayable")
    elif touching_but_missing:
        print(f"INCOMPLETE: {len(touching_but_missing)} interface-sample(s) have no score although "
              f"the two reference chains measurably touch. That is a scoring failure wearing a "
              f"non-contact's clothes -- the case this check exists to catch.")
    elif not (unfolded or short_poses or no_pose or no_rows or bad_scores or bad_cache):
        print(f"INCOMPLETE: nothing left to fold or rescore; {len(unmapped)} interface-sample(s) "
              f"have no score because the scorer mapped no model chain to one of their reference "
              f"chains. Recovering them needs a different mapping, not a re-run.")
    else:
        print("INCOMPLETE")
    return rc


_LOCKS = []   # open lock files, kept alive for the process lifetime


def rescore(a):
    """score only the (interface, sample) rows that are missing, with the dataset's own scorer"""
    pool = read_pool(os.path.join(a.e, "pool.csv"))
    res = os.path.join(a.e, "results.csv")

    # DO NOT RUN WHILE ANOTHER WRITER APPENDS TO THIS CSV. Rescoring re-reads immediately before
    # appending, which narrows the window but cannot close it: a fold worker can still append the
    # same sample between that read and this write, and the duplicate double-weights the pose in
    # every average over its seed. Raised in review, and the right answer is sequencing rather than
    # locking the live folding driver. Once the fold workers exit, this can run alongside a
    # VALIDATION fold safely, because that writes a different results.csv.
    live = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout
    writers = [ln.split(":")[0] for ln in live.splitlines()
               if ln.startswith(("w1fold", "w1val", "w2_"))]
    mine = "validation" if "validation" in os.path.abspath(a.e) else "training"
    clash = [w for w in writers if (w.startswith("w1val") == (mine == "validation"))]
    if clash and not a.allow_concurrent_writers:
        raise SystemExit(f"STOP: {clash} are folding into the same results.csv as --e {a.e}. "
                         f"Rescoring now can duplicate a sample that lands between this call's "
                         f"read and its append. Wait for them to exit, or pass "
                         f"--allow-concurrent-writers if you accept that risk.")

    # A SECOND RESCORE IS A WRITER TOO, AND THE SCAN ABOVE CANNOT SEE ONE. It looks for tmux fold
    # sessions; a rescore is a plain subprocess. Overnight TWO rescore watchers sat behind the same
    # "wait for the fold workers to exit" loop, so both would have started appending to this csv at
    # the same moment -- the exact duplicate-row hazard the sequencing above exists to prevent. An
    # exclusive, non-blocking flock held for the whole run makes the second one refuse instead.
    _lk = open(res + ".rescore.lock", "a+")
    try:
        fcntl.flock(_lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lk.seek(0)
        who = _lk.read().strip() or "unknown"
        raise SystemExit(f"STOP: another rescore already holds {res}.rescore.lock (pid {who}). "
                         f"Two rescores appending to the same csv duplicate rows. Let that one "
                         f"finish; this one has written nothing.")
    _lk.seek(0); _lk.truncate(); _lk.write(f"{os.getpid()}\n"); _lk.flush()
    _LOCKS.append(_lk)   # module-level, so the fd is not closed by gc and the lock outlives this frame

    # THE WORKLIST COMES FROM THE QUEUE WHEN ONE IS GIVEN. expected_units enumerates every job json
    # in --jobs and expands --seeds per structure, which for training/jobs is 856 structures and
    # 8,560 candidate units rather than the 3,560 actually queued -- so an intended recovery could
    # reach into older incomplete work. Raised in review.
    units = queue_units(a)

    have = read_results(res)
    added = skipped = ambiguous = 0
    timed_out, failed, no_contact, still_missing, unmapped = [], [], [], [], []
    for pdb, seed, arm in units:
        # THE RUN MUST HOLD A COMPLETE POSE SET, AND ONLY ONE RUN MAY. run_dir_for returns the first
        # directory whose run.json names this unit without resolving several matches, and a run
        # holding two of five poses cannot be recovered by scoring. Two complete runs are different
        # folds and nothing here may choose between them.
        st, d, detail = pose_sets.classify(os.path.join(a.e, "poses"), pdb, seed,
                                           a.nsample, arm=arm)
        if st == "ambiguous":
            print(f"  AMBIGUOUS {pdb} {seed}: {detail}; not rescored")
            ambiguous += 1
            continue
        if st != "complete":
            continue
        # NO SECOND OPINION FROM run_dir_for. It returns the FIRST directory whose run.json names
        # the unit, whether or not its pose set is whole, so a complete run was being rejected
        # merely because an older partial run sorted before it. pose_sets.classify already checked
        # provenance -- pdb, seed and arm -- on the run it returned, so that is the run to score.
        for ifc in sorted(pool.get(pdb, ())):
            got = have.get((ifc, seed), {})
            # ONLY THE RECOVERABLE ONES. A pose that was scored and simply has no contact at this
            # interface will never produce a row however many times it is re-scored, so including it
            # here would make rescore() re-run the scorer on every call and append nothing -- and
            # verify() would stay INCOMPLETE forever.
            # invalidate=True: a cached json that cannot be parsed is moved aside, because
            # ost_dockq.run() would otherwise reuse it and the retry would be a no-op.
            miss = {str(k) for k in range(a.nsample) if str(k) not in got
                    and classify_missing(d, a.scorer, str(k), iface=ifc,
                                         invalidate=True)[0] == "unscored"}
            if not miss:
                continue
            # THE DATASET'S OWN SCORER. Recovery must not change the method: sabdab_val scores with
            # 'ost' and an earlier version of this hardcoded 'pxm', which would have mixed two DockQ
            # implementations in one results.csv.
            cmd = [a.py, os.path.join(a.code, "scorer.py"), "--dir", d, "--iface", ifc,
                   "--seed", seed, "--arm", "free", "--donor", "-1", "--chains", "",
                   "--ref", os.path.join(a.refs, f"{pdb}.cif"), "--scorer", a.scorer]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT {ifc} {seed}")
                timed_out.append((ifc, seed))
                continue
            if out.returncode != 0:
                print(f"  FAILED {ifc} {seed}: {out.stderr.strip()[:160]}")
                failed.append((ifc, seed, out.stderr.strip()[:120]))
                continue
            # ONLY THE MISSING SAMPLES. scorer.py emits a row per pose, so appending all of them
            # would duplicate the ones already there and double-weight that seed in every average.
            keep = []
            for line in out.stdout.splitlines():
                f = line.split(",")
                if len(f) >= 8 and f[5].strip() in miss:
                    keep.append(line)
                elif len(f) >= 8:
                    skipped += 1
            if keep:
                # RE-READ IMMEDIATELY BEFORE APPENDING. `have` was read once, before the scorer
                # ran; a fold worker scoring the same unit in the meantime would have added rows
                # this call still believes are missing, and appending them again double-weights
                # that seed in every average -- the harm the comment above guards against, by a
                # different route. Rescoring therefore stays safe to run while a fold is live.
                fresh = set()
                try:
                    with open(res) as fh:
                        for r in csv.DictReader(fh):
                            if r.get("iface") == ifc and r.get("seed") == seed:
                                fresh.add(str(r.get("sample")))
                except OSError:
                    pass
                raced = [l for l in keep if l.split(",")[5].strip() in fresh]
                keep = [l for l in keep if l.split(",")[5].strip() not in fresh]
                if raced:
                    print(f"  {ifc} {seed}: {len(raced)} sample(s) landed from another writer "
                          f"while scoring; not appended")
                    skipped += len(raced)
            if keep:
                # one write() per line under O_APPEND, so a concurrent writer cannot split a line
                with open(res, "a") as fh:
                    for line in keep:
                        fh.write(line + "\n")
                        fh.flush()
                added += len(keep)
                print(f"  {ifc} {seed}: added samples {[l.split(',')[5] for l in keep]}")
    # A POSE THAT LEGITIMATELY HAS NO CONTACT AT THIS INTERFACE IS NOT A FAILURE, and the two must
    # not be conflated -- that distinction is what keeps "incomplete recovery" meaningful.
    # classify_missing returns the bucket "" with "none is this chain pair" for a pose whose two
    # chains do not touch: scorer.py skips it by design, so no row will ever exist and no number of
    # rescores will change that. "unscored" is the recoverable bucket, and anything still in it
    # after a successful scorer call did not recover.
    #
    # Only rows that are STILL MISSING are classified, so this does not re-walk the whole queue.
    after = read_results(res)
    for pdb, seed, arm in units:
        for ifc in sorted(pool.get(pdb, ())):
            got = after.get((ifc, seed), {})
            miss = [k for k in range(a.nsample) if str(k) not in got]
            if not miss:
                continue
            st, d, _dt = pose_sets.classify(os.path.join(a.e, "poses"), pdb, seed,
                                            a.nsample, arm=arm)
            if st != pose_sets.COMPLETE:
                continue
            for k in miss:
                bucket, why = classify_missing(d, a.scorer, str(k), iface=ifc)
                if bucket == "":
                    no_contact.append((ifc, seed, k))
                elif bucket == "unmapped":
                    # NOT A FAILED RECOVERY. OST ran, and this interface was never a candidate
                    # because one of its reference chains was not mapped to a model chain. Running
                    # the same scorer again on the same pose cannot change that, so counting it as
                    # incomplete recovery would make every future rescore report failure for work
                    # it was never able to do. It is reported on its own line instead.
                    unmapped.append((ifc, seed, k, why[:90]))
                else:
                    still_missing.append((ifc, seed, k, bucket, why[:80]))

    print(f"rescore added {added} row(s); left {skipped} existing row(s) untouched")
    print(f"  not rescored, ambiguous or disagreeing saved run: {ambiguous}")
    print(f"  scorer timed out:                                {len(timed_out)}")
    print(f"  scorer exited nonzero:                           {len(failed)}")
    print(f"  pose has no contact at that interface (expected, not a failure): {len(no_contact)}")
    print(f"  reference chain never mapped by {a.scorer} -- NOT recoverable by rescoring: "
          f"{len(unmapped)}")
    for ifc, seed, k, why in unmapped[:6]:
        print(f"     {ifc} {seed} sample {k}: {why}")
    if len(unmapped) > 6:
        print(f"     ... and {len(unmapped) - 6} more")
    print(f"  still missing and NOT explained by no-contact:    {len(still_missing)}")
    for ifc, seed, k, bucket, why in still_missing[:10]:
        print(f"     STILL MISSING {ifc} {seed} sample {k}: {bucket} -- {why}")
    for ifc, seed in timed_out[:10]:
        print(f"     TIMEOUT {ifc} {seed}")
    for ifc, seed, err in failed[:10]:
        print(f"     FAILED  {ifc} {seed}: {err}")

    # EXIT NONZERO WHEN RECOVERY IS INCOMPLETE. Raised in review: returning 0 regardless meant a
    # run with timeouts, scorer failures or skipped ambiguous units looked successful, and the
    # launcher printed DONE over it. Non-contacting poses are excluded from that judgement because
    # they are a property of the structure, not of this run.
    bad = len(timed_out) + len(failed) + ambiguous + len(still_missing)
    if bad:
        print(f"INCOMPLETE: {bad} unit(s)/interface(s) did not recover; exiting nonzero")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "rescore"])
    ap.add_argument("--worklist", default="",
                    help="a queue.tsv whose units define the rescore scope. Without it the scope "
                         "is every job json in --jobs times --seeds, which reaches past the "
                         "queue into older work.")
    ap.add_argument("--expect-units", type=int, default=0,
                    help="refuse unless the worklist expands to exactly this many units, so a "
                         "recovery cannot silently widen its own scope")
    ap.add_argument("--allow-concurrent-writers", action="store_true",
                    help="run even though a fold is writing the same results.csv. The re-read "
                         "before appending narrows the duplicate window but does not close it.")
    ap.add_argument("--e", required=True, help="the val_fold directory")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--suffix", default="-final-updated.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=400000)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--scorer", required=True, help="the dataset's scorer; recovery must not change it")
    ap.add_argument("--py", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--trunks", action=argparse.BooleanOptionalAction, default=True,
                    help="check the confidence-head caches. --no-trunks for a fold run with "
                         "SAVE_TRUNKS=0, which stores poses and scores only")
    a = ap.parse_args()
    sys.exit(verify(a) if a.mode == "verify" else rescore(a))


if __name__ == "__main__":
    main()
