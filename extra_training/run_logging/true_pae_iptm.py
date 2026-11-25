#!/usr/bin/env python
"""GROUND-TRUTH PAE -> ipTM. Does the metric rank poses when the PREDICTION is perfect?

THE QUESTION. The within-seed measurement (run_logging/within_seed_iptm_dockq.py) shows the
shipped ipTM orders a seed's poses correctly only 52.7% of the time on held-out structures against
50% chance, while the pair representation it is read from carries plenty of pose-dependent signal
(run_logging/pose_pair_signal_sites.py). Two readings: the PAE head predicts the alignment error
badly, or the ipTM aggregation throws away the ordering even when the error is known exactly.

HOW THIS SEPARATES THEM. PAELoss.calculate_label already computes the EXPERIMENTALLY DERIVED
aligned error for every frame-token pair -- it is the label the PAE term is trained against, from a
real frame alignment of the pose onto the deposited structure -- and returns it as a ONE-HOT
distribution over the same 64 bins the head predicts. Substituting that one-hot for the head's
softmax and running the IDENTICAL aggregation (sample_confidence.calculate_chain_based_ptm, the
same call rank_loss.iptm makes) gives the ipTM a perfect PAE predictor would produce. No model is
loaded and no network runs: the inputs are the cached poses and the cached labels.

  true-PAE ipTM ranks the poses  -> the metric is sound and the PAE PREDICTION is the problem
  true-PAE ipTM does not         -> the aggregation discards the ordering and no amount of better
                                    PAE prediction will fix selection

THE AGGREGATION IS NOT REIMPLEMENTED. _iptm_from_prob below is rank_loss.iptm with the softmax
removed, down to the asym-id compaction remap, because calculate_chain_based_ptm renumbers chains
whenever a chain is absent and indexing a compacted matrix with uncompacted ids silently reads a
different chain pair.

ONE PREMISE, NAMED. true_bins is [N_frame, N_token] while the aggregation wants
[N_token, N_token]; the two coincide only because every token in this pool has a frame (measured
856 of 856 structures, the same premise tm_centered_loss.pair_errors documents). Asserted per unit
rather than assumed.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")


def load_configs(model_name="protenix-v2"):
    """The model configs WITHOUT building the model. Same assembly as
    phase_j_rescore_cache.build_model, minus Protenix(configs) and the checkpoint -- the PAE bin
    parameters have to come from the config the head was trained with, not from retyped
    defaults."""
    from collections.abc import Mapping
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import model_configs
    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs[model_name])
    return parse_configs(configs=base, arg_str="", fill_required_with_null=True)


def _iptm_from_prob(pae_prob, feat, configs, pair):
    """chain_pair_iptm[pair] from an ALREADY-NORMALISED pae distribution.

    rank_loss.iptm with the softmax removed. Everything else is identical, including the remap:
    calculate_chain_based_ptm compacts asym ids to 0..N-1 whenever a chain is missing, so an
    uncompacted pair index is either out of range or, worse, in range and pointing at a different
    chain pair.
    """
    import torch
    from protenix.model import sample_confidence
    from protenix.utils.torch_utils import autocasting_disable_decorator
    token_is_ligand = torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0
    out = autocasting_disable_decorator(True)(sample_confidence.calculate_chain_based_ptm)(
        pae_prob, has_frame=feat["has_frame"].bool(), asym_id=feat["asym_id"],
        token_is_ligand=token_is_ligand,
        **sample_confidence.get_bin_params(configs.loss.pae))
    cpi = out["chain_pair_iptm"]
    if cpi.dim() == 3:
        cpi = cpi[0]
    i, j = pair
    _u = torch.unique(feat["asym_id"].reshape(-1).long())
    _remap = {int(o): n for n, o in enumerate(_u.tolist())}
    if i not in _remap or j not in _remap:
        return None
    i, j = _remap[i], _remap[j]
    if i >= cpi.shape[0] or j >= cpi.shape[1]:
        return None
    return float(cpi[i, j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--train-structures",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--units", type=int, default=0, help="0 = every unit with a trunk and a label")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="per-pose csv: iface,seed,sample,dockq,"
                                                "iptm_pred,iptm_true")
    a = ap.parse_args()

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    import numpy as np
    import torch
    import phase_j_rescore_cache as RC
    import rank_loss as RKL
    from protenix.model.loss import PAELoss

    roles_join = RKL.load_roles(a.roles)
    cfg = load_configs()
    pae_loss = PAELoss(min_bin=cfg.loss.pae.min_bin, max_bin=cfg.loss.pae.max_bin,
                       no_bins=cfg.loss.pae.no_bins)
    dev = torch.device(a.device)

    # (pdb, seed) -> {iface: {sample: (dockq, iptm_pred)}}
    want = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") != "free":
                continue
            p = RKL.parse_iface(r["iface"], roles_join)
            if p is None:
                continue
            pdb, _pair = p
            try:
                want[(pdb, int(r["seed"]))][r["iface"]][int(r["sample"])] = (
                    float(r["dockq"]), float(r["iptm"]))
            except (ValueError, KeyError, TypeError):
                continue

    units = sorted(want)
    if a.units:
        units = units[:int(a.units)]
    print(f"{len(units)} unit(s) to score", flush=True)

    rows, nskip, nfail = [], 0, 0
    for ui, (pdb, seed) in enumerate(units, 1):
        tp = os.path.join(a.fold, "trunks", pdb, f"free_{seed}.npz")
        lp = os.path.join(a.fold, "labels", pdb, f"free_{seed}.npz")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            nskip += 1
            continue
        try:
            feat, _si, _st, _zt, poses, _nc = RC.load_unit(tp, dev)
            lab = np.load(lp, allow_pickle=True)
            fai = torch.as_tensor(lab["lf_frame_atom_index"]).to(dev)
            ram = torch.as_tensor(lab["lf_pae_rep_atom_mask"]).to(dev)
            hfr = torch.as_tensor(lab["lf_has_frame"]).to(dev).bool()
            # THE PREMISE, ASSERTED. true_bins is [N_frame, N_token]; the aggregation wants
            # [N_token, N_token]. Equal only when every token has a frame.
            if int(hfr.sum()) != int(hfr.numel()):
                nskip += 1
                continue
            # THE POSE INDEX IS NOT THE SAMPLE INDEX. results_with_trunks.csv is keyed by the
            # fold's SAMPLE number while the trunk and the label npz are in POSE order, and the
            # label npz carries the permutation in `sample_index` -- non-identity in 3,971 of
            # 4,000 training label files (99.3%), e.g. 10fj/free_500000 -> [0, 4, 3, 2, 1].
            # Joining by position paired each pose's true-PAE ipTM with another pose's DockQ and
            # turned the association NEGATIVE (pooled Pearson -0.088, pairwise 0.481 on the first
            # 100 units) -- the exact trap phase_k_train.py:7086 documents for the rel target.
            si = np.asarray(lab["sample_index"]).reshape(-1)
            npose = min(int(poses.shape[0]), int(lab["label_coordinate"].shape[0]), int(a.poses))
            if si.shape[0] < npose:
                nskip += 1
                continue
            for i in range(npose):
                samp = int(si[i])
                tc = torch.as_tensor(lab["label_coordinate"][i], dtype=torch.float32, device=dev)
                cm = torch.as_tensor(lab["label_coordinate_mask"][i].astype(np.float32),
                                     device=dev)
                with torch.no_grad():
                    _sq, true_bins, _m = pae_loss.calculate_label(
                        pred_coordinate=poses[i].float(), true_coordinate=tc,
                        coordinate_mask=cm, frame_atom_index=fai,
                        rep_atom_mask=ram, has_frame=hfr)
                    # NO EXTRA SAMPLE DIM. calculate_label already returns
                    # [1, N_frame, N_token, no_bins]; a further [None] made it 5-dimensional and
                    # calculate_chain_based_ptm died with "index 1 is out of bounds for dimension 1
                    # with size 1" from inside its chain loop.
                    prob = true_bins.float()                 # the one-hot IS a distribution
                    for iface, bys in want[(pdb, seed)].items():
                        if samp not in bys:
                            continue
                        pr = RKL.parse_iface(iface, roles_join)
                        if pr is None:
                            continue
                        sc = _iptm_from_prob(prob, feat, cfg, tuple(int(v) for v in pr[1]))
                        if sc is None:
                            nfail += 1
                            continue
                        dq, ipp = bys[samp]
                        rows.append({"pdb": pdb, "iface": iface, "seed": seed, "sample": samp,
                                     "dockq": dq, "iptm_pred": ipp,
                                     "iptm_true": round(sc, 6)})
                del tc, cm, true_bins, prob
            del feat, poses
            torch.cuda.empty_cache()
        except Exception as e:                                # noqa: BLE001
            print(f"  [{ui}/{len(units)}] {pdb}/{seed}: {type(e).__name__}: {e}", flush=True)
            nfail += 1
            continue
        if ui % 100 == 0:
            print(f"  [{ui}/{len(units)}] {len(rows)} pose row(s)", flush=True)

    print(f"rows {len(rows)}  skipped units {nskip}  failures {nfail}")
    if not rows:
        raise SystemExit("STOP: nothing scored")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["pdb", "iface", "seed", "sample", "dockq",
                                            "iptm_pred", "iptm_true"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"per-pose rows -> {a.out}")


if __name__ == "__main__":
    main()
