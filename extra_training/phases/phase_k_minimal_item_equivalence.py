"""GATE: does the minimal loss input reproduce the full dataset item's loss, exactly?

The item cache is the training run's memory -- 82-95 GB per rank, measured -- and it exists only to
hand feat_ds and label_dict to ProtenixLoss. A key-access probe on four structures (128-1,581
tokens, 2-5 chains) found the loss reads nine feature keys and two label keys, all of which the
cached label npz already holds. This asserts the consequence: for every unit tested, the loss
computed from BC.minimal_loss_inputs must equal the loss computed the old way, term by term.

It is a gate, not a benchmark. Exit 1 on any term that differs by more than --tol, so a launch can
depend on it. Run it before trusting --minimal-items on a real run.

  CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold python -I phases/phase_k_minimal_item_equivalence.py
"""
import argparse, csv, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import phase_j_dataset as DS
import phase_j_build_cache as BC
import phase_j_rescore_cache as RC
from protenix.data.pipeline.dataset import get_datasets
from protenix.model.loss import ProtenixLoss

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structures", default="",
                    help="comma-separated pdb ids; default spans the token range")
    ap.add_argument("--poses", type=int, default=2, help="poses per unit to compare")
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed |old-new| per term. The two paths feed the SAME tensors to "
                         "the same loss, so they should agree to the bit; the tolerance exists only "
                         "so a formatting difference does not read as a numerical one.")
    a = ap.parse_args()

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold, or the trunk's coordinates are NaN.")

    rows = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "train_val_splits", "cache_train_structures.csv")))
        if r["tokens"].isdigit()]
    if a.structures:
        want = [x.strip() for x in a.structures.split(",") if x.strip()]
        picks = [r for r in rows if r["pdb"] in want]
    else:
        rows.sort(key=lambda r: int(r["tokens"]))
        skip = {"9dc3", "9bbn", "8vba", "9q7b", "9q5v", "9e7x", "9b9j", "9dzf", "9q5m", "9mx1"}
        ok = [r for r in rows if r["pdb"] not in skip]
        picks = [ok[0], ok[len(ok) // 4], ok[len(ok) // 2], ok[3 * len(ok) // 4], ok[-1]]
    print(f"comparing {len(picks)} structure(s): "
          + ", ".join(f"{r['pdb']}({r['tokens']}t)" for r in picks))

    dev = torch.device("cuda:0")
    cfg = DS.build_configs(3400,
                           os.path.join(ROOT, "training/prep/job_indices.csv"),
                           os.path.join(ROOT, "training/prep/job_bioassembly"),
                           os.path.join(ROOT, "training/prep/job_cifs"),
                           os.path.join(ROOT, "building_msa/work/mmcif_msa"),
                           os.path.join(ROOT, "building_msa/work/pdb_seqs/seq_to_pdb_index.json"),
                           os.path.join(ROOT, "building_msa/work/mmcif_msa"))
    cfg.train_confidence_only = True
    train_ds, _ = get_datasets(cfg, os.path.join(ROOT, "training/prep/dataset_errors"))
    byp = BC.merged_lookup(train_ds)
    model, configs = RC.build_model("/home/ebrahim/checkpoint", "protenix-v2", dev)
    configs.train_confidence_only = True
    loss_fn = ProtenixLoss(configs)

    worst, n_cmp, failures = 0.0, 0, []
    print(f"\n{'structure':10s} {'pose':>4s} | {'term':14s} {'full item':>13s} {'minimal':>13s} {'|diff|':>10s}")
    for r in picks:
        pdb = r["pdb"]
        trunk = os.path.join(ROOT, "training/fold/trunks", pdb, f"free_{a.seed}.npz")
        labp = os.path.join(ROOT, "training/fold/labels", pdb, f"free_{a.seed}.npz")
        if not (os.path.exists(trunk) and os.path.exists(labp)):
            print(f"  {pdb}: no trunk/label for seed {a.seed}; skipped"); continue
        lab = np.load(labp, allow_pickle=True)
        feat_head, s_inputs, s_trunk, zt, poses, _ = RC.load_unit(trunk, dev)

        # THE OLD PATH, exactly as phase_k_train assembled it before this change.
        it = train_ds[byp[pdb][0]]
        feat_ds, label_dict = it["input_feature_dict"], it["label_dict"]
        feat_live = {k: (v.to(dev) if torch.is_tensor(v)
                         else torch.as_tensor(np.asarray(v)).to(dev))
                     for k, v in feat_ds.items()
                     if torch.is_tensor(v) or isinstance(v, (np.ndarray, int, float))}
        for k, v in feat_ds.items():
            feat_live.setdefault(k, v)
        feat_old = dict(feat_live)
        for k in BC.LOSS_FEATS:
            v = np.asarray(lab[f"lf_{k}"])
            feat_old[k] = torch.as_tensor(v).to(dev) if v.dtype != object else lab[f"lf_{k}"]

        for i in range(min(a.poses, poses.shape[0])):
            x = poses[i:i + 1].to(dev).float()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plddt, pae, pde, resolved = model.run_confidence_head(
                    input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                    pair_mask=None, x_pred_coords=x,
                    triangle_multiplicative=configs.triangle_multiplicative,
                    triangle_attention=configs.triangle_attention, inplace_safe=False,
                    chunk_size=None)
            # ONE prediction, scored twice. Running the head twice would fold cuDNN's
            # non-determinism into the comparison and make a real difference unreadable.
            pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                    "resolved": resolved.float(), "coordinate_mini": x}

            ld_old = {}
            for k, v in label_dict.items():
                if torch.is_tensor(v):
                    ld_old[k] = v.to(dev)
                elif isinstance(v, np.ndarray) and v.dtype != object:
                    ld_old[k] = torch.as_tensor(v).to(dev)
                else:
                    ld_old[k] = v
            ld_old["coordinate"] = torch.as_tensor(lab["label_coordinate"][i],
                                                   dtype=torch.float32, device=dev)
            ld_old["coordinate_mask"] = torch.as_tensor(
                lab["label_coordinate_mask"][i].astype(np.float32), device=dev)

            feat_new, ld_new = BC.minimal_loss_inputs(lab, i, dev, torch)

            with torch.no_grad():
                _, met_old = loss_fn.calculate_losses(feat_old, pred, ld_old, mode="train")
                _, met_new = loss_fn.calculate_losses(feat_new, pred, ld_new, mode="train")

            for t in TERMS + ("loss",):
                o, nw = float(met_old[t]), float(met_new[t])
                d = abs(o - nw)
                worst = max(worst, d); n_cmp += 1
                flag = "" if d <= a.tol else "  MISMATCH"
                if d > a.tol:
                    failures.append((pdb, i, t, o, nw, d))
                print(f"{pdb:10s} {i:4d} | {t:14s} {o:13.8f} {nw:13.8f} {d:10.2e}{flag}")
        del feat_old, feat_live, label_dict, it
        torch.cuda.empty_cache()

    print(f"\n{n_cmp} comparison(s); largest absolute difference {worst:.3e} (tolerance {a.tol:g})")
    if failures:
        print(f"GATE FAILED: {len(failures)} term(s) differ.")
        for pdb, i, t, o, nw, d in failures[:10]:
            print(f"  {pdb} pose {i} {t}: {o:.8f} vs {nw:.8f}  (|diff| {d:.3e})")
        return 1
    print("GATE PASSED: the minimal input reproduces the full item's loss on every term.")
    print("  -> --minimal-items can be trusted; the item cache is not needed to compute the loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
