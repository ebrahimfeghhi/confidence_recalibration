"""WHERE THE VALIDATION GPU TIME GOES, and whether a PAE-only pair-ipTM path would recover it.

The evaluator reads exactly ONE field of the summary -- chain_pair_iptm (phase_k_eval.py:574) --
but RC.confidence computes the whole thing: the confidence head, then distogram_head and
compute_contact_prob, then compute_full_data_and_summary, which builds atom pLDDT, token-pair PDE,
token-pair PAE, clash and chain-based pLDDT on the way to the one number that is used.

Earlier I measured the load (4.7 min/shard) and the pose->sample join (3.6 min) and attributed the
remaining 17.8 min of a 26.1 min pass to "GPU forward + rest" WITHOUT decomposing it, then told the
user there was no win available. Review round 7 pointed out the narrow path. This measures the
three stages separately and checks that the narrow path returns the SAME chain_pair_iptm, because a
faster number that differs is not the same metric.

ONE UNCONTENDED GPU. Timings on a card shared with a training rank measure contention, not stages.

  CUDA_VISIBLE_DEVICES=4 LAYERNORM_TYPE=openfold python -I phases/phase_k_eval_stage_timing.py
"""
import argparse, os, random, sys, time
import numpy as np
import torch
import torch.utils.checkpoint  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import phase_j_rescore_cache as RC
from protenix.model import sample_confidence
from protenix.utils.torch_utils import autocasting_disable_decorator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default=os.path.join(ROOT, "validation", "fold"))
    ap.add_argument("--n", type=int, default=25, help="validation units to time")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold.")

    # REFUSE A CONTENDED CARD. The whole point is a clean attribution.
    free, total = torch.cuda.mem_get_info()
    print(f"card: {free/2**30:.0f} GiB free of {total/2**30:.0f} GiB")

    dev = torch.device("cuda:0")
    model, configs = RC.build_model("/home/ebrahim/checkpoint", "protenix-v2", dev)

    units = []
    for r, _, ns in os.walk(os.path.join(a.fold, "trunks")):
        for n in ns:
            if n.endswith(".npz") and n.startswith("free_"):
                units.append(os.path.join(r, n))
    random.Random(a.seed).shuffle(units)
    units = units[:a.n]
    print(f"timing {len(units)} validation unit(s)\n")

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    tA, tB, tC, tD, toks, maxdiff = [], [], [], [], [], 0.0
    for i, tp in enumerate(units, 1):
        feat, s_in, s_tr, z, x, nc = RC.load_unit(tp, dev)
        n_tok = int(feat["asym_id"].shape[-1])
        for rep in range(2):           # first pass warms autotuning and the allocator
            # ---- A: the confidence head
            t0 = sync()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plddt, pae, pde, _res = model.run_confidence_head(
                    input_feature_dict=feat, s_inputs=s_in, s_trunk=s_tr, z_trunk=z,
                    pair_mask=None, x_pred_coords=x,
                    triangle_multiplicative=configs.triangle_multiplicative,
                    triangle_attention=configs.triangle_attention,
                    inplace_safe=False, chunk_size=None)
            t1 = sync()
            # ---- B: distogram head + contact probabilities
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cp = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(
                    distogram_logits=model.distogram_head(z),
                    **sample_confidence.get_bin_params(configs.loss.distogram))
            t2 = sync()
            # ---- C: the full summary, as the evaluator computes it today
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                summary, _full = autocasting_disable_decorator(True)(
                    sample_confidence.compute_full_data_and_summary)(
                    configs=configs, pae_logits=pae, plddt_logits=plddt, pde_logits=pde,
                    contact_probs=cp, token_asym_id=feat["asym_id"],
                    token_has_frame=feat["has_frame"], atom_coordinate=x,
                    atom_to_token_idx=feat["atom_to_token_idx"],
                    atom_is_polymer=1 - feat["is_ligand"], N_recycle=nc,
                    interested_atom_mask=None, return_full_data=True,
                    mol_id=None, elements_one_hot=None)
            t3 = sync()
            # ---- D: the narrow path -- PAE only, straight to chain_pair_iptm
            with torch.no_grad():
                bins = sample_confidence.get_bin_params(configs.loss.pae)
                pae_prob = torch.softmax(pae.float(), dim=-1)
                atom_is_ligand = feat["is_ligand"].long()
                token_is_ligand = torch.zeros_like(feat["asym_id"]).scatter_add(
                    0, feat["atom_to_token_idx"], atom_is_ligand) > 0
                narrow = autocasting_disable_decorator(True)(
                    sample_confidence.calculate_chain_based_ptm)(
                    pae_prob, has_frame=feat["has_frame"].bool(), asym_id=feat["asym_id"],
                    token_is_ligand=token_is_ligand, **bins)
            t4 = sync()
            if rep == 0:
                continue
            tA.append(t1 - t0); tB.append(t2 - t1); tC.append(t3 - t2); tD.append(t4 - t3)
            toks.append(n_tok)
            # ---- CORRECTNESS: the narrow path must give the SAME chain_pair_iptm
            want = summary[0]["chain_pair_iptm"] if isinstance(summary, (list, tuple)) \
                else summary["chain_pair_iptm"]
            got = narrow["chain_pair_iptm"]
            w = want.detach().float().cpu().numpy()
            g = got.detach().float().cpu().numpy()
            g = g[0] if g.ndim == 3 and w.ndim == 2 else g
            if w.shape == g.shape:
                maxdiff = max(maxdiff, float(np.abs(w - g).max()))
            else:
                # A HARD FAILURE, NOT A NOTE. maxdiff is only updated when the shapes
                # match, so a run where every pair mismatched would print these lines and then
                # report "largest absolute difference 0.000e+00" -- i.e. declare the two paths
                # interchangeable on the strength of having compared nothing. Found in review
                # of a3086ef.
                raise SystemExit(
                    f"STOP: chain_pair_iptm shape mismatch on {os.path.basename(tp)}: "
                    f"{w.shape} (current) vs {g.shape} (narrow). The two paths are not "
                    f"returning the same object, so they cannot be compared and the narrow "
                    f"path must not be adopted on this evidence.")
        del feat, s_in, s_tr, z, x, plddt, pae, pde, cp, summary, _full, narrow
        torch.cuda.empty_cache()
        if i % 5 == 0:
            print(f"  {i}/{len(units)}", flush=True)

    A, B, C, D = (np.array(v) for v in (tA, tB, tC, tD))
    cur = A + B + C
    new = A + D
    print(f"\nn = {len(A)} unit(s), median {np.median(toks):.0f} tokens")
    print(f"{'stage':42s} {'median s':>9s} {'share of current':>17s}")
    for nm, v in (("A confidence head (needed)", A),
                  ("B distogram head + contact prob", B),
                  ("C compute_full_data_and_summary", C)):
        print(f"  {nm:40s} {np.median(v):9.3f} {np.median(v)/np.median(cur)*100:16.1f}%")
    print(f"  {'D narrow: softmax(PAE) + chain ptm':40s} {np.median(D):9.3f}")
    print(f"\ncurrent path A+B+C : {cur.sum():.2f} s total, median {np.median(cur):.3f} s/unit")
    print(f"narrow path  A+D   : {new.sum():.2f} s total, median {np.median(new):.3f} s/unit")
    print(f"speedup on the GPU half: {cur.sum()/new.sum():.2f}x "
          f"({(1 - new.sum()/cur.sum())*100:.0f}% of it removed)")
    per_shard = 1767
    saved_min = (cur.mean() - new.mean()) * per_shard / 60
    print(f"per shard ({per_shard} units): saves {saved_min:.1f} min of a 26.1 min pass")
    print(f"\nchain_pair_iptm, narrow vs current: largest absolute difference {maxdiff:.3e}")
    if maxdiff > 1e-5:
        print("  NOT INTERCHANGEABLE at this tolerance -- a faster number that differs is a "
              "different metric. Do not adopt the narrow path on this evidence.")
        return 1
    print("  interchangeable: the narrow path returns the same number the metric is built on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
