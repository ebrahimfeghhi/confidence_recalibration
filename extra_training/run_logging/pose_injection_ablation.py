#!/usr/bin/env python
"""IS THE CONFIDENCE HEAD ACTUALLY READING THE 3D POSE? Zero the only two layers that carry it.

THE PATHWAY IS EXACTLY TWO LAYERS WIDE. The predicted coordinates enter ConfidenceHead at
confidence.py:279-305 and nowhere else: x_pred_rep_coords -> torch.cdist -> distance_pred, and
distance_pred is consumed only by

    linear_no_bias_d            one_hot(distance, 39 bins) -> c_z      (39 x 128 weights)
    linear_no_bias_d_wo_onehot  the raw distance in angstrom -> c_z    ( 1 x 128 weights)

whose outputs are ADDED to z_pair before the pairformer stack. Everything downstream -- pae, pde,
pLDDT, resolved -- is a function of (s_trunk, z_pair), and ipTM is a function of the PAE
probabilities plus static masks (asym_id, has_frame); sample_confidence.calculate_iptm takes no
coordinates. s_trunk and z_trunk are per-SEED, not per-pose.

SO THE ABLATION IS DEGENERATE WITHIN A SEED, BY CONSTRUCTION. With both weight matrices zeroed the
head's input is bit-identical for all 5 poses of a seed, so all 5 get the SAME ipTM and within-seed
pairwise ranking accuracy is exactly 0.5 (every pair an ipTM tie). That is not a finding, it is
arithmetic -- and this script asserts it, because an ablated spread that is NOT zero would mean
some other pose-dependent path exists that the reading above missed.

WHAT IS THEREFORE THE MEASUREMENT. The per-pose displacement

    delta_i = ipTM(pose i, full head) - ipTM(pose-free head)

which is the entire contribution of the 3D pose to this unit's ipTM, pose by pose. Three questions
it answers that the ablated ranking cannot:
  MAGNITUDE   how far does knowing the pose move ipTM at all? If max_i |delta_i| is at the level of
              bf16 noise the head is ignoring the geometry it was handed.
  DIRECTION   does delta_i rise with DockQ? A large but DockQ-uncorrelated delta means the pose is
              read and misread, which is a different failure from not reading it.
  SHARE       |delta| against the full head's own within-seed ipTM spread: how much of the little
              within-seed movement ipTM has is pose-driven rather than numerical.

UNITS ARE PICKED FOR GENUINE WITHIN-SEED DIVERSITY -- one seed whose 5 poses span near-0 to near-1
DockQ -- so a head that reads the pose at all has the largest possible signal to show. If delta is
flat HERE it is flat everywhere.

POSE INDEX IS NOT SAMPLE INDEX. The labels npz carries `sample_index`, a non-identity permutation
in 99.3% of training label files; DockQ and the stored ipTM are keyed by SAMPLE, so every join goes
through it. Same convention as run_logging/validate_iptm_path.py, whose recomputation path this
reuses verbatim -- the full-head pass doubles as that validation, since it must reproduce the
stored ipTM.
"""
import argparse
import collections
import csv
import functools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
print = functools.partial(print, flush=True)

# the two layers, and nothing else, that carry x_pred_coords into the head
POSE_LAYERS = ("linear_no_bias_d", "linear_no_bias_d_wo_onehot")


def pairwise_acc(score, truth):
    """Mean over pairs with distinct truth: 1 correct, 0.5 score tie, 0 wrong. nan if no pair."""
    n, hit, tot = len(score), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if truth[i] == truth[j]:
                continue
            tot += 1
            if score[i] == score[j]:
                hit += 0.5
            elif (score[i] > score[j]) == (truth[i] > truth[j]):
                hit += 1.0
    return (hit / tot if tot else float("nan")), tot


def spearman(a, b):
    import numpy as np

    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=f"{ROOT}/../scratchpad/posefree_units.json",
                    help="json list of {pdb, iface, seed}")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    import numpy as np
    import torch
    import phase_j_rescore_cache as RC
    import rank_loss as RKL

    units = json.load(open(a.units))
    roles = RKL.load_roles(a.roles)

    # stored DockQ and ipTM, keyed by SAMPLE index, for the units we asked for
    want_keys = {(u["pdb"], int(u["seed"]), u["iface"]) for u in units}
    stored = collections.defaultdict(dict)       # (pdb, seed, iface) -> sample -> (dockq, iptm)
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") not in (None, "", "free"):
                continue
            k = (r.get("pdb") or "", int(r["seed"]) if r["seed"].isdigit() else -1, r["iface"])
            p = RKL.parse_iface(r["iface"], roles)
            if p is None:
                continue
            k = (p[0], k[1], r["iface"])
            if k not in want_keys:
                continue
            try:
                stored[k][int(r["sample"])] = (float(r["dockq"]), float(r["iptm"]))
            except (ValueError, KeyError, TypeError):
                continue

    dev = torch.device("cuda")
    model, cfg = RC.build_model(a.ckpt_dir, a.model_name, dev)
    head = model.confidence_head
    keep = {}
    for nm in POSE_LAYERS:
        w = getattr(head, nm).weight
        keep[nm] = w.detach().clone()
        print(f"pose layer {nm:28s} shape {tuple(w.shape)}  "
              f"|w| mean {w.detach().abs().mean().item():.5f}")

    def set_pose_path(on):
        with torch.no_grad():
            for nm in POSE_LAYERS:
                w = getattr(head, nm).weight
                w.copy_(keep[nm] if on else torch.zeros_like(w))

    rows = []
    for ui, u in enumerate(units, 1):
        pdb, seed, iface = u["pdb"], int(u["seed"]), u["iface"]
        key = (pdb, seed, iface)
        if key not in stored:
            print(f"  [{ui}] {pdb}/{seed} {iface}: no stored row, skipped")
            continue
        tp = os.path.join(a.fold, "trunks", pdb, f"free_{seed}.npz")
        lp = os.path.join(a.fold, "labels", pdb, f"free_{seed}.npz")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            print(f"  [{ui}] {pdb}/{seed}: missing trunk or label npz, skipped")
            continue
        feat, s_inputs, s_trunk, zt, poses, n_cycle = RC.load_unit(tp, dev)
        si = np.asarray(np.load(lp, allow_pickle=True)["sample_index"]).reshape(-1)
        pair = tuple(int(v) for v in RKL.parse_iface(iface, roles)[1])
        n_pose = min(int(poses.shape[0]), si.shape[0])

        def score(i):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _pl, pae, _pd, _rs = model.run_confidence_head(
                    input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                    pair_mask=None, x_pred_coords=poses[i:i + 1],
                    triangle_multiplicative=cfg.triangle_multiplicative,
                    triangle_attention=cfg.triangle_attention,
                    inplace_safe=False, chunk_size=None)
                logits = pae.detach().float()
                while logits.dim() > 4:
                    logits = logits[0]
                sc = RKL.iptm(logits, feat, cfg, pair)
            del pae, logits
            torch.cuda.empty_cache()
            return None if sc is None else float(sc)

        set_pose_path(True)
        full = [score(i) for i in range(n_pose)]
        set_pose_path(False)
        zero = [score(i) for i in range(n_pose)]
        set_pose_path(True)

        for i in range(n_pose):
            samp = int(si[i])
            if samp not in stored[key] or full[i] is None or zero[i] is None:
                continue
            dq, ip = stored[key][samp]
            rows.append({"pdb": pdb, "iface": iface, "seed": seed, "pose": i, "sample": samp,
                         "dockq": dq, "iptm_stored": ip,
                         "iptm_full": full[i], "iptm_zero": zero[i],
                         "delta": full[i] - zero[i]})
        print(f"  [{ui}/{len(units)}] {pdb}/{seed} {iface}  {n_pose} pose(s)  "
              f"full {min(x for x in full if x is not None):.4f}-"
              f"{max(x for x in full if x is not None):.4f}  "
              f"pose-free spread {max(zero) - min(zero):.2e}")
        del feat, s_inputs, s_trunk, zt, poses
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("STOP: nothing scored")

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["pdb"], r["seed"], r["iface"])].append(r)

    print()
    print("REPRODUCTION CHECK -- the full-head pass must reproduce the stored ipTM")
    d = np.array([r["iptm_full"] - r["iptm_stored"] for r in rows])
    print(f"  rows {len(rows)}   max |full - stored| {np.abs(d).max():.6f}   "
          f"mean {np.abs(d).mean():.6f}")

    print()
    print("ASSERTION -- with the pose path zeroed every pose of a seed must get ONE ipTM")
    sp = np.array([max(x["iptm_zero"] for x in v) - min(x["iptm_zero"] for x in v)
                   for v in by.values()])
    print(f"  units {len(sp)}   max within-seed pose-free ipTM spread {sp.max():.3e}   "
          f"{'OK' if sp.max() < 1e-6 else 'VIOLATED -- another pose path exists'}")

    print()
    hdr = (f"{'pdb':6} {'interface':12} {'seed':>7} {'dockq range':>13} {'iptm full':>13} "
           f"{'spread':>8} {'posefree':>9} {'max|delta|':>11} {'acc full':>9} {'rho full':>9} "
           f"{'rho delta':>10}")
    print(hdr)
    accs, rhos, rhod, mdel, spreads = [], [], [], [], []
    for (pdb, seed, iface), v in sorted(by.items()):
        v.sort(key=lambda r: r["pose"])
        dq = [r["dockq"] for r in v]
        fu = [r["iptm_full"] for r in v]
        ze = [r["iptm_zero"] for r in v]
        de = [r["delta"] for r in v]
        acc, _ = pairwise_acc(fu, dq)
        rf, rd = spearman(fu, dq), spearman(de, dq)
        accs.append(acc); rhos.append(rf); rhod.append(rd)
        mdel.append(max(abs(x) for x in de)); spreads.append(max(fu) - min(fu))
        print(f"{pdb:6} {iface:12} {seed:7d} {min(dq):6.3f}-{max(dq):6.3f} "
              f"{min(fu):6.4f}-{max(fu):6.4f} {max(fu) - min(fu):8.4f} {ze[0]:9.4f} "
              f"{max(abs(x) for x in de):11.4f} {acc:9.3f} {rf:9.3f} {rd:10.3f}")

    print()
    print(f"over {len(by)} unit(s), {len(rows)} pose(s)")
    print(f"  within-seed pairwise accuracy, full head        {np.nanmean(accs):.4f}")
    print(f"  within-seed pairwise accuracy, pose path zeroed 0.5000  (all pairs tie, by construction)")
    print(f"  mean Spearman(ipTM_full, DockQ) within a seed   {np.nanmean(rhos):+.4f}")
    print(f"  mean Spearman(delta,     DockQ) within a seed   {np.nanmean(rhod):+.4f}")
    print(f"  mean max|delta| per unit                        {np.mean(mdel):.4f}")
    print(f"  mean full-head within-seed ipTM spread          {np.mean(spreads):.4f}")
    print(f"  pose path's share of that spread                "
          f"{np.mean(mdel) / np.mean(spreads) if np.mean(spreads) else float('nan'):.2f}x")
    al = np.array([r["delta"] for r in rows])
    dqa = np.array([r["dockq"] for r in rows])
    print(f"  pooled Spearman(delta, DockQ) over all poses    {spearman(al, dqa):+.4f}")
    print(f"  pooled |delta|: mean {np.abs(al).mean():.4f}  median {np.median(np.abs(al)):.4f}  "
          f"max {np.abs(al).max():.4f}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  rows -> {a.out}")


if __name__ == "__main__":
    main()
