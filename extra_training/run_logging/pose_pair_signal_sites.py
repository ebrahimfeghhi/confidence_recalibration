#!/usr/bin/env python
"""WHERE DOES THE POSE GO? Pose-driven variation in the confidence head's pair representation,
measured at the geometry injection and after each Pairformer block, on antibody x antigen pairs.

THE QUESTION. ipTM holds 2.6% of the within-interface variance among a seed's 5 samples while
DockQ holds 49% -- the head barely distinguishes poses it should rank. Two very different causes:
the Pairformer stack SUPPRESSES the injected geometry (an input/representation problem), or the
geometry survives the stack and is lost in the PAE readout or the ipTM aggregation (an output
problem). They need different fixes, so this measures which one it is.

WHAT IS MEASURED. For one (structure, seed) unit, its 5 diffusion samples differ only in
coordinates: s_inputs, s_trunk and z_trunk are the SAME tensors for all five, so every difference
downstream is pose-driven by construction. Each pose is run through the confidence head ON ITS OWN
(N_sample=1), and at each site the antibody x antigen sub-block of the pair representation is
captured. The statistic is the fraction of that sub-block's activation energy that is pose
dependent:

    r = sqrt( mean_e Var_p(z_e) / mean_e mean_p z_e^2 )

over elements e of the ab x ag sub-block (all c_z channels) and the P=5 poses. The denominator is
the overall activation magnitude at that site, which is the normalisation asked for: r is
dimensionless and comparable across layers whose scales differ by orders of magnitude.

THE SITES, in order:
    inject   z_pair immediately after the two distance terms are added (the input to the stack)
    block0..3  z_pair after each Pairformer block
    pae      the PAE logits on the same pairs (the readout)
    itm      the token-token iTM matrix t_ij read out of the PAE distribution (what ipTM pools)
    iptm     the pooled chain-pair ipTM scalar (the aggregation)

READING IT. r falling monotonically from inject through block3 is suppression in the stack. r
holding up to block3 and then collapsing at pae / itm / iptm is a readout or aggregation problem.
For iptm, a scalar, r is the across-pose std over the mean -- the same ratio, one element.

MEMORY. The per-element pose mean needs one tensor of the sub-block's shape per site; the second
moment is accumulated as a scalar, using
mean_e Var_p = (1/PE) sum_{p,e} z^2 - (1/E) sum_e ((1/P) sum_p z)^2.
So the cost is one sub-block per site, not two, and not P of them.
"""
import argparse
import collections
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")

AB_ROLES = {"H", "L"}


def is_ag(role):
    """chain_roles_all.csv names antigens ag0, ag1, ... -- not "A". A set of {"A"} matched
    nothing and the selection came back empty."""
    return bool(role) and role.startswith("ag")


def load_roles_full(path):
    """{pdb: {ref_chain: (model_chain, role)}} -- the role column as well as the join."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["pdb"], {})[r["ref_chain"]] = (r["model_chain"], r["role"].strip())
    return out


def select_units(a):
    """[(pdb, seed, (ax, ay), iface, spread, tokens)] -- the units to measure, most-varying first.

    ONE UNIT PER STRUCTURE. Sorting 74k rows by within-seed DockQ spread and taking the top 50
    returns a handful of structures several times over, and a per-site ratio averaged over those
    would describe those few systems rather than the set.
    """
    import rank_loss as RKL
    roles_join = RKL.load_roles(a.roles)
    roles_full = load_roles_full(a.roles)

    tok = {}
    with open(a.indices) as fh:
        for r in csv.DictReader(fh):
            try:
                tok[r["pdb_id"]] = int(r["num_tokens"])
            except (KeyError, ValueError):
                pass

    by = collections.defaultdict(dict)
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") != "free":
                continue
            try:
                by[(r["iface"], int(r["seed"]))][int(r["sample"])] = float(r["dockq"])
            except (ValueError, KeyError, TypeError):
                continue

    cand = []
    for (iface, seed), d in by.items():
        if len(d) != int(a.poses):
            continue
        p = RKL.parse_iface(iface, roles_join)
        if p is None:
            continue
        pdb, pair = p
        # AB x AG ONLY. The pool also carries heavy-light and antigen-antigen pairs, and the
        # question is about the epitope, so an H/L pair would answer a different one.
        rm = roles_full.get(pdb) or {}
        rr = {m: ro for m, ro in rm.values()}
        ra = rr.get(chr(ord("A") + pair[0]))
        rb = rr.get(chr(ord("A") + pair[1]))
        if ra is None or rb is None:
            continue
        if not ((ra in AB_ROLES and is_ag(rb)) or (is_ag(ra) and rb in AB_ROLES)):
            continue
        n = tok.get(pdb)
        if n is None or n > int(a.max_tokens):
            continue
        tp = os.path.join(a.fold, "trunks", pdb, f"free_{seed}.npz")
        if not os.path.exists(tp):
            continue
        spread = max(d.values()) - min(d.values())
        if spread < float(a.min_spread):
            continue
        cand.append((pdb, int(seed), pair, iface, spread, n, tp))

    cand.sort(key=lambda t: -t[4])
    out, seen = [], set()
    for c in cand:
        if c[0] in seen:
            continue
        seen.add(c[0])
        out.append(c)
        if len(out) >= int(a.units):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--units", type=int, default=50)
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--min-spread", type=float, default=0.2,
                    help="minimum within-seed max-min ground-truth DockQ for a unit to qualify")
    ap.add_argument("--max-tokens", type=int, default=1600,
                    help="the sub-block tensors are one per site; past this they stop fitting")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--out", default="", help="per-unit csv")
    ap.add_argument("--list-only", action="store_true", help="print the selection and stop")
    a = ap.parse_args()

    units = select_units(a)
    print(f"selected {len(units)} unit(s), DockQ spread "
          f"{min(u[4] for u in units):.3f}-{max(u[4] for u in units):.3f}, tokens "
          f"{min(u[5] for u in units)}-{max(u[5] for u in units)}", flush=True)
    for u in units[:5]:
        print(f"   {u[3]:14s} seed {u[1]}  pair {u[2]}  spread {u[4]:.3f}  tokens {u[5]}")
    if a.list_only:
        return

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    import numpy as np
    import torch
    import phase_j_rescore_cache as RC
    import rank_loss as RKL
    import tm_centered_loss as TMC

    dev = torch.device("cuda")
    model, cfg = RC.build_model(a.ckpt_dir, a.model_name, dev)
    ch = model.confidence_head
    nblk = len(ch.pairformer_stack.blocks)
    SITES = ["inject"] + [f"block{k}" for k in range(nblk)] + ["pae", "itm", "iptm"]
    print(f"confidence head: {nblk} pairformer block(s); sites {SITES}", flush=True)

    grab = {}

    def pre_hook(_m, args):
        # args = (s_trunk, z_pair, pair_mask); z_pair here is AFTER both distance terms.
        grab["inject"] = args[1]
        return None

    def blk_hook(k):
        def _h(_m, _args, out):
            grab[f"block{k}"] = out[1]
            return None
        return _h

    hs = [ch.pairformer_stack.register_forward_pre_hook(pre_hook)]
    for k in range(nblk):
        hs.append(ch.pairformer_stack.blocks[k].register_forward_hook(blk_hook(k)))

    bp = RKL.__dict__  # keep the module referenced; bins come from cfg below
    del bp
    pae_cfg = cfg.loss.pae
    rows = []
    try:
        for ui, (pdb, seed, pair, iface, spread, ntok, tp) in enumerate(units, 1):
            feat, s_inputs, s_trunk, zt, poses, n_cycle = RC.load_unit(tp, dev)
            asym = feat["asym_id"].reshape(-1).long()
            mx = (asym == int(pair[0]))
            my = (asym == int(pair[1]))
            if int(mx.sum()) == 0 or int(my.sum()) == 0:
                print(f"  [{ui}/{len(units)}] {iface} seed {seed}: SKIP, a chain of the pair has "
                      f"no token in the cached unit", flush=True)
                continue
            ix = torch.nonzero(mx, as_tuple=True)[0]
            iy = torch.nonzero(my, as_tuple=True)[0]
            P = int(poses.shape[0]) if hasattr(poses, "shape") else len(poses)
            P = min(P, int(a.poses))

            # d0 FOR THE CHAIN PAIR, the same normalisation ipTM uses, so t_ij here is the t_ij
            # the pooled score is built from.
            npt = int(mx.sum()) + int(my.sum())
            w = TMC.per_bin_weight(TMC.d0_for_pair(npt), pae_cfg.min_bin, pae_cfg.max_bin,
                                   pae_cfg.no_bins, dev)

            acc_sum = {}            # site -> per-element sum over poses (cpu float32)
            acc_sq = collections.defaultdict(float)   # site -> sum over poses+elements of z^2
            acc_n = {}              # site -> element count
            iptms = []

            for i in range(P):
                grab.clear()
                x = poses[i:i + 1] if torch.is_tensor(poses) else torch.as_tensor(
                    poses[i:i + 1], device=dev)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _plddt, pae, _pde, _res = model.run_confidence_head(
                        input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk,
                        z_trunk=zt, pair_mask=None, x_pred_coords=x,
                        triangle_multiplicative=cfg.triangle_multiplicative,
                        triangle_attention=cfg.triangle_attention,
                        inplace_safe=False, chunk_size=None)
                site_t = {}
                for s in SITES[:1 + nblk]:
                    z = grab.get(s)
                    if z is None:
                        raise SystemExit(f"STOP: site {s} was never captured; the hook did not "
                                         f"fire, so nothing can be said about that layer.")
                    site_t[s] = z.detach().float()[..., ix, :, :][..., iy, :]
                pl = pae.detach().float()
                while pl.dim() > 3:
                    pl = pl[0]
                site_t["pae"] = pl[ix][:, iy]
                prob = torch.softmax(pl.float(), dim=-1)
                t_ij = (prob * w).sum(-1)
                site_t["itm"] = t_ij[ix][:, iy]
                # THE SAMPLE DIM IS REQUIRED. sample_confidence.calculate_chain_based_ptm writes
                # chain_pair_iptm[:, aid_1, aid_2], so a [N_tok, N_tok, bins] tensor raises
                # "too many indices for tensor of dimension 2". Training passes pred["pae"],
                # which still carries it.
                sc = RKL.iptm(pl[None], feat, cfg, tuple(int(v) for v in pair))
                if sc is None:
                    raise SystemExit(f"STOP: no chain_pair_iptm for {iface} pair {pair}")
                iptms.append(float(sc))

                for s, t in site_t.items():
                    v = t.reshape(-1)
                    acc_sq[s] += float((v.double() ** 2).sum())
                    acc_n[s] = v.numel()
                    cpu = v.cpu()
                    acc_sum[s] = cpu if s not in acc_sum else acc_sum[s] + cpu
                del site_t, pae, pl, prob, t_ij
                torch.cuda.empty_cache()

            rec = {"iface": iface, "seed": seed, "pdb": pdb, "tokens": ntok,
                   "n_ab": int(mx.sum()), "n_ag": int(my.sum()), "poses": P,
                   "dockq_spread": round(spread, 4)}
            for s in SITES[:-1]:
                E, n = acc_n[s], P
                m2 = acc_sq[s] / (n * E)                      # mean_e mean_p z^2
                mean_e = (acc_sum[s].double() / n)
                v_mean = m2 - float((mean_e ** 2).sum()) / E   # mean_e Var_p(z)
                rec[s] = round(float((max(v_mean, 0.0) / m2) ** 0.5), 6) if m2 > 0 else float("nan")
            ia = np.asarray(iptms, dtype=float)
            rec["iptm"] = round(float(ia.std() / abs(ia.mean())), 6) if ia.mean() else float("nan")
            rec["iptm_mean"] = round(float(ia.mean()), 4)
            rec["iptm_spread"] = round(float(ia.max() - ia.min()), 4)
            rows.append(rec)
            print(f"  [{ui}/{len(units)}] {iface} seed {seed} spread {spread:.3f}  "
                  + "  ".join(f"{s} {rec[s]:.4f}" for s in SITES), flush=True)
            del feat, s_inputs, s_trunk, zt, poses, acc_sum
            torch.cuda.empty_cache()
    finally:
        for h in hs:
            h.remove()

    if not rows:
        raise SystemExit("STOP: no unit produced a measurement")
    print(f"\n=== mean over {len(rows)} unit(s): pose-dependent fraction of activation energy ===")
    print(f"{'site':8s} {'mean r':>9s} {'median':>9s} {'vs inject':>10s}")
    base = None
    for s in SITES:
        v = np.asarray([r[s] for r in rows if r[s] == r[s]], dtype=float)
        if base is None:
            base = float(np.mean(v))
        print(f"{s:8s} {np.mean(v):9.5f} {np.median(v):9.5f} {np.mean(v) / base:10.3f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cols = ["pdb", "iface", "seed", "tokens", "n_ab", "n_ag", "poses", "dockq_spread"] \
            + SITES + ["iptm_mean", "iptm_spread"]
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        print(f"per-unit rows -> {a.out}")


if __name__ == "__main__":
    main()
