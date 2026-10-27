#!/usr/bin/env python
"""Per-token residue identity for the ABAG-Rank export. CPU only, no GPU.

WHY THIS EXISTS AS A SEPARATE PASS. ABAG-Rank needs `residue_one_letter` for its one-hot feature
AND for the filter upstream applies before the distance cutoff and before choosing the zero-contact
fallback: preprocess_shards drops every pair whose residue is 'X'. That filter is not optional for
us -- 103 of the 5,549 sequences in seq_to_pdb_index.json contain X.

AND IT IS NOT IN THE CACHE. The trunk npz carries six feature arrays (asym_id, has_frame,
atom_to_token_idx, atom_to_tokatom_idx, distogram_rep_atom_mask, is_ligand) and no restype, because
the confidence head does not need one; ci_s_inputs carries no restype one-hot either, 400 of its
449 columns being identically zero. abag_rank_plan.md claimed residue letters came free from fold
metadata. They do not.

SO IT COMES FROM THE SAME FEATURISER THAT BUILT THE CACHE, not from re-parsing a cif. phase_j's
training dataset returns `input_feature_dict["restype"]`, and phase_j_build_cache's own
check_alignment() asserts that dataset's asym_id / atom_to_token_idx / rep-atom masks agree
ELEMENTWISE with the cached npz. Running that same check here turns "the token orders probably
match" into a proof, which is the only basis on which a residue may be attached to a token index.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

# Protenix's restype index -> one letter. Index 20 is the unknown residue, which is what upstream
# filters on; anything past the 20 standard types is reported as X for the same reason.
AA = "ARNDCQEGHILKMFPSTWYV"


def letters(restype_onehot):
    """[T] of one-letter codes from the [T, 32] restype one-hot."""
    idx = np.asarray(restype_onehot).argmax(-1)
    return np.array([AA[i] if i < len(AA) else "X" for i in idx], dtype="U1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="training/full_train/structures_no9dzj.csv")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--out", default="training/abag/restypes")
    ap.add_argument("--limit", type=int, default=0)
    # phase_j_build_cache's own defaults, repeated so this pass cannot be pointed somewhere else.
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index",
                    default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    a = ap.parse_args()

    import phase_j_build_cache as BC
    import phase_j_dataset as DS
    import torch

    pdbs = [r["pdb"] for r in csv.DictReader(open(os.path.join(ROOT, a.selection)))]
    if a.limit:
        pdbs = pdbs[:a.limit]
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    # EXACTLY phase_j_build_cache's construction (its lines 859-866), same arguments and same
    # defaults, so this is the dataset that produced the cached arrays rather than a lookalike.
    from protenix.data.pipeline.dataset import get_datasets
    print("  building the phase_j dataset (same featuriser that built the cache)...", flush=True)
    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    os.makedirs(err, exist_ok=True)
    train, _ = get_datasets(cfg, err)
    byp = BC.merged_lookup(train)

    n_x = done = 0
    for pdb in pdbs:
        dst = os.path.join(out, f"{pdb}.npz")
        if os.path.exists(dst):
            done += 1
            continue
        if pdb not in byp:
            raise SystemExit(f"STOP: {pdb} is not in the phase_j dataset; its residues cannot be "
                             f"attached to token indices.")
        item = train[byp[pdb][0]]
        feat_ds = item["input_feature_dict"]
        trunks = sorted(glob.glob(os.path.join(ROOT, a.fold, "trunks", pdb, "free_*.npz")))
        if not trunks:
            raise SystemExit(f"STOP: {pdb} has no trunk npz to align against.")
        _z, npz_feat, _x = BC.load_trunk(trunks[0])
        # THE PROOF, not an assumption: six arrays must agree elementwise or the residues would be
        # attached to the wrong tokens, which is invisible downstream.
        BC.check_alignment(npz_feat, feat_ds, pdb)
        rt = feat_ds["restype"]
        rt = rt.detach().cpu().numpy() if torch.is_tensor(rt) else np.asarray(rt)
        one = letters(rt)
        if len(one) != len(npz_feat["asym_id"]):
            raise SystemExit(f"STOP: {pdb}: {len(one)} restype(s) against "
                             f"{len(npz_feat['asym_id'])} token(s).")
        n_x += int((one == "X").sum())
        np.savez_compressed(dst, residue_one_letter=one, asym_id=npz_feat["asym_id"])
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(pdbs)}  last {pdb}: T={len(one)}, "
                  f"X={(one == 'X').sum()}", flush=True)
    print(f"\n  {done} structure(s); {n_x} X token(s) in total")
    print("  (a non-zero X count is why the exporter must filter before the cutoff and before the "
          "closest-pair fallback, as upstream does)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
