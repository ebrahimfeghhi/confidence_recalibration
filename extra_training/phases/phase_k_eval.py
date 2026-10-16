"""PHASE K -- score a confidence head on THE validation metric, the one the project is judged on.

WHAT IS COMPUTED. For each validation interface, select the pose with the highest PER-INTERFACE
pair_iptm over a seed budget, then report

    mean_dockq  -- mean over interfaces of the SELECTED pose's DockQ
    frac_acc    -- fraction of interfaces whose selected pose clears DockQ 0.23 (CAPRI acceptable)

Baseline to beat, measured 2026-10-15 over 92 interfaces at 100 seeds with the SHIPPED head:
mean_dockq 0.5776, frac_acc 0.7283, against an oracle of 0.7184 / 0.8696. The 0.1408 gap is what a
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
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold (without it coordinates are NaN)")

    import torch
    import phase_j_rescore_cache as RC
    import phase_j_build_cache as BC

    pairs_by_pdb, unresolved = interface_chain_pairs(a.pool, a.roles)
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

    units = []
    for pdb in sorted(pairs_by_pdb):
        for sd_ in range(a.seed_base, a.seed_base + a.budget):
            tp = os.path.join(a.fold, "trunks", pdb, f"free_{sd_}.npz")
            if os.path.exists(tp):
                units.append((pdb, sd_, tp))
    if a.limit_units:
        units = units[:a.limit_units]
    print(f"units: {len(units)} (budget {a.budget} seeds)")

    rows, selfchk, n_nosample = [], [], 0
    for n, (pdb, sd_, tp) in enumerate(units, 1):
        feat, s_in, s_tr, z, x, nc = RC.load_unit(tp, dev)
        summ = RC.confidence(model, configs, feat, s_in, s_tr, z, x, nc)
        summ = summ if isinstance(summ, (list, tuple)) else [summ]
        # join 2: pose -> sample, by coordinate identity
        unit_dir = next(iter(glob.glob(os.path.join(a.fold, "poses", pdb, f"{sd_}-*"))), None)
        cifs = BC.sample_cifs(unit_dir, pdb) if unit_dir else []
        px = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        try:
            ks = BC.sample_indices(px, cifs, 0.01, 0.5)
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
            for iface, ai, aj in pairs_by_pdb[pdb]:
                if max(ai, aj) >= cpi.shape[0]:
                    continue
                v = float(cpi[ai, aj])
                key = (iface, sd_, k)
                rows.append((iface, sd_, k, v, dockq.get(key)))
                if key in stored_iptm:
                    selfchk.append(abs(v - stored_iptm[key]))
        if n % 200 == 0:
            print(f"  ...{n}/{len(units)} units", flush=True)

    if n_nosample:
        raise SystemExit(f"STOP: {n_nosample} unit(s) could not be joined to their sample cifs. "
                         f"Selecting a pose and reading another sample's DockQ would score the "
                         f"wrong thing, so this is fatal rather than skipped.")

    # ---- THE SELF-CHECK
    if not a.ckpt and not a.no_self_check:
        if not selfchk:
            raise SystemExit("STOP: no recomputed ipTM could be compared against results.csv, so "
                             "the three joins are unverified and no metric below means anything.")
        mx = max(selfchk)
        over = sum(1 for d in selfchk if d > a.iptm_tol)
        print(f"\nSELF-CHECK vs results.csv: {len(selfchk)} row(s), max |diff| {mx:.6f}, "
              f"{over} over --iptm-tol {a.iptm_tol}")
        if over:
            raise SystemExit(f"STOP: the shipped head does not reproduce {over} stored ipTM value(s). "
                             f"One of the three joins -- interface->chain pair, pose->sample, or "
                             f"(iface,seed,sample)->DockQ -- is wrong.")
        print("  the joins reproduce the fold's own ipTM, so the metric below is measuring the head")

    # ---- THE METRIC
    byif = collections.defaultdict(list)
    for iface, sd_, k, v, dq in rows:
        if dq is not None:
            byif[iface].append((v, dq))
    missing_if = sorted(want_ifaces - set(byif))
    if missing_if:
        raise SystemExit(f"STOP: {len(missing_if)} pool interface(s) have no scored pose: "
                         f"{missing_if[:6]}. The metric would cover a subset.")
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
    print(f"  baseline to beat: mean_dockq 0.5776  frac_acc 0.7283  (shipped head, 100 seeds)")
    if a.out:
        with open(a.out, "w") as fh:
            import json
            json.dump(out, fh, indent=1)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
