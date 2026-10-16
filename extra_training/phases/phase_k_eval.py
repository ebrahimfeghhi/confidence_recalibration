"""PHASE K -- score a confidence head on THE validation metric, the one the project is judged on.

WHAT IS COMPUTED. For each validation interface, select the pose with the highest PER-INTERFACE
pair_iptm over a seed budget, then report

    mean_dockq  -- mean over interfaces of the SELECTED pose's DockQ
    frac_acc    -- fraction of interfaces whose selected pose clears DockQ 0.23 (CAPRI acceptable)

Baseline to beat, measured 2026-10-15 over 92 interfaces at 100 seeds with the SHIPPED head:
mean_dockq 0.5779, frac_acc 0.7283, against an oracle of 0.7171 / 0.8696. The 0.1392 gap is what a
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
import json
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
    ap.add_argument("--iptm-tol", type=float, default=2e-3,
                    help="how closely the recomputed ipTM must reproduce results.csv with the "
                         "SHIPPED head. bf16 trunk tensors are stored and reloaded and the head runs "
                         "under autocast, so exact equality is not the bar.")
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
    ap.add_argument("--allow-partial", action="store_true",
                    help="report a metric that does NOT cover its stated budget: missing trunks, "
                         "poses without DockQ, or interfaces of unequal depth. OFF by default, "
                         "because a number labelled 'budget 100' computed from one seed is worse "
                         "than no number.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import phase_j_rescore_cache as RC
    import phase_j_build_cache as BC

    pairs_by_pdb, unresolved = interface_chain_pairs(a.pool, a.roles)
    refs_by_pdb = pool_ref_pairs(a.pool)
    n_ost, n_static, n_differ, differ_pdbs = 0, 0, 0, collections.Counter()
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
    units, missing_trunk = [], []
    for pdb in sorted(pairs_by_pdb):
        for sd_ in range(a.seed_base, a.seed_base + a.budget):
            tp = os.path.join(a.fold, "trunks", pdb, f"free_{sd_}.npz")
            (units if os.path.exists(tp) else missing_trunk).append(
                (pdb, sd_, tp) if os.path.exists(tp) else f"{pdb}/free_{sd_}")
    if missing_trunk and not a.allow_partial:
        raise SystemExit(
            f"STOP: {len(missing_trunk)} of {len(pairs_by_pdb)*a.budget} intended unit(s) have no "
            f"trunk under {a.fold}/trunks, e.g. {missing_trunk[:5]}.\n  A metric labelled "
            f"'budget {a.budget}' must cover {a.budget} seeds for every structure. Lower --budget, "
            f"or pass --allow-partial to report a number that does NOT mean what its label says.")
    if a.limit_units:
        if not a.allow_partial:
            raise SystemExit("STOP: --limit-units subsets the data, so the metric would not be the "
                             "metric. Pass --allow-partial to acknowledge that explicitly.")
        units = units[:a.limit_units]
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
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        have = dict(model.named_parameters())
        missing = [k for k in sd if k not in have]
        if missing:
            raise SystemExit(f"STOP: {a.ckpt} has {len(missing)} tensor(s) absent from the model, "
                             f"e.g. {missing[:4]}")
        with torch.no_grad():
            for k, v in sd.items():
                have[k].data.copy_(v.to(have[k].device))
        print(f"loaded {len(sd)} head tensor(s) from {a.ckpt}")
    else:
        print("using the SHIPPED head (no --ckpt)")

    rows, selfchk, n_nosample = [], [], 0
    for n, (pdb, sd_, tp) in enumerate(units, 1):
        feat, s_in, s_tr, z, x, nc = RC.load_unit(tp, dev)
        summ = RC.confidence(model, configs, feat, s_in, s_tr, z, x, nc)
        summ = summ if isinstance(summ, (list, tuple)) else [summ]
        # join 2: pose -> sample, by coordinate identity. The pose DIRECTORY is itself resolved by
        # the trunk, because a refold leaves two finished directories for one seed and neither
        # filesystem order nor poses/index.tsv says which run wrote this trunk.
        px = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        try:
            unit_dir, ks = BC.resolve_seed_dir(a.fold, pdb, sd_, px, 0.01, 0.5)
        except Exception as e:
            n_nosample += 1
            if n_nosample <= 5:
                print(f"  {pdb}/{sd_}: cannot join poses to samples ({type(e).__name__}: "
                      f"{str(e)[:90]})")
            continue
        for i, s_i in enumerate(summ):
            cpi = s_i["chain_pair_iptm"]
            cpi = cpi.detach().cpu().numpy() if hasattr(cpi, "detach") else np.asarray(cpi)
            cpi = cpi[0] if cpi.ndim == 3 else cpi
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
            for iface, ai, aj in pose_pairs:
                if max(ai, aj) >= cpi.shape[0]:
                    continue
                v = float(cpi[ai, aj])
                key = (iface, sd_, k)
                rows.append((iface, sd_, k, v, dockq.get(key)))
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

    if n_nosample:
        raise SystemExit(f"STOP: {n_nosample} unit(s) could not be joined to their sample cifs. "
                         f"Selecting a pose and reading another sample's DockQ would score the "
                         f"wrong thing, so this is fatal rather than skipped.")

    print(f"\njoin 1: OST's per-pose chain mapping used on {n_ost} pose(s), unresolvable on "
          f"{n_static}")
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

    # ---- THE SELF-CHECK
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
            raise SystemExit(f"STOP: the shipped head does not reproduce {over} stored ipTM value(s). "
                             f"One of the three joins -- interface->chain pair, pose->sample, or "
                             f"(iface,seed,sample)->DockQ -- is wrong.")
        print("  the joins reproduce the fold's own ipTM, so the metric below is measuring the head")

    # ---- COVERAGE, ENFORCED KEY BY KEY BEFORE ANY NUMBER IS COMPUTED
    # `if dq is not None` silently dropped every pose without a DockQ, and the only coverage check
    # was that an interface had at least ONE scored pose -- so one of 500 passed. All four of the
    # reviewer's conditions are now required: every intended key, unique rows, finite scores.
    seen = collections.Counter((iface, sd_, k) for iface, sd_, k, _v, _dq in rows)
    dup = [kk for kk, c in seen.items() if c > 1]
    if dup:
        raise SystemExit(f"STOP: {len(dup)} duplicate (iface, seed, sample) row(s), e.g. {dup[:4]}. "
                         f"A duplicated pose would be weighted twice in the selection.")
    nonfinite = [(iface, sd_, k) for iface, sd_, k, v, dq in rows
                 if not np.isfinite(v) or (dq is not None and not np.isfinite(dq))]
    if nonfinite:
        raise SystemExit(f"STOP: {len(nonfinite)} row(s) carry a non-finite ipTM or DockQ, e.g. "
                         f"{nonfinite[:4]}. argmax over a nan is undefined.")
    nodq = sorted((f"{i}/{s_}/{k}") for i, s_, k, _v, dq in rows if dq is None)
    if nodq and not a.allow_partial:
        raise SystemExit(f"STOP: {len(nodq)} scored pose(s) have no DockQ in {a.results}, e.g. "
                         f"{nodq[:5]}. Dropping them silently changes what the mean is over; "
                         f"recover them with phase_j_fill_scores.py or pass --allow-partial.")
    byif = collections.defaultdict(list)
    for iface, sd_, k, v, dq in rows:
        if dq is not None:
            byif[iface].append((v, dq))
    missing_if = sorted(want_ifaces - set(byif))
    if missing_if:
        raise SystemExit(f"STOP: {len(missing_if)} pool interface(s) have no scored pose: "
                         f"{missing_if[:6]}. The metric would cover a subset.")
    # AND EVERY INTERFACE MUST CARRY THE WHOLE BUDGET. An interface with one scored pose out of 500
    # satisfied the check above; depth is what the word "budget" in the output is asserting.
    n_pose = collections.Counter()
    for iface, sd_, _k, _v, dq in rows:
        if dq is not None:
            n_pose[iface] += 1
    want_depth = max(n_pose.values()) if n_pose else 0
    shallow = sorted((i, n_pose[i]) for i in byif if n_pose[i] != want_depth)
    if shallow and not a.allow_partial:
        raise SystemExit(
            f"STOP: {len(shallow)} interface(s) do not carry the same number of scored poses as the "
            f"deepest ({want_depth}): e.g. {shallow[:5]}.\n  Averaging over interfaces of different "
            f"depth makes the budget label meaningless; pass --allow-partial to accept it.")
    print(f"coverage: {len(byif)} interface(s), {want_depth} scored pose(s) each, "
          f"{len(rows)} row(s) total, 0 duplicates, 0 non-finite")
    sel = [max(v, key=lambda t: t[0])[1] for v in byif.values()]
    orc = [max(dq for _v, dq in v) for v in byif.values()]
    n = len(sel)
    out = {"n_interfaces": n, "budget": a.budget,
           "mean_dockq": round(statistics.mean(sel), 4),
           "frac_acc": round(sum(1 for d in sel if d >= ACCEPTABLE) / n, 4),
           "oracle_mean_dockq": round(statistics.mean(orc), 4),
           "oracle_frac_acc": round(sum(1 for d in orc if d >= ACCEPTABLE) / n, 4),
           "ckpt": a.ckpt or "shipped"}
    out["headroom"] = round(out["oracle_mean_dockq"] - out["mean_dockq"], 4)
    print(f"\nMETRIC ({out['ckpt']}, {n} interfaces, budget {a.budget} seeds)")
    print(f"  mean_dockq {out['mean_dockq']:.4f}   frac_acc {out['frac_acc']:.4f}")
    print(f"  oracle     {out['oracle_mean_dockq']:.4f}   {out['oracle_frac_acc']:.4f}   "
          f"headroom {out['headroom']:.4f}")
    print(f"  baseline to beat: mean_dockq 0.5779  frac_acc 0.7283  (shipped head, 100 seeds)")
    if a.out:
        with open(a.out, "w") as fh:
            import json
            json.dump(out, fh, indent=1)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
