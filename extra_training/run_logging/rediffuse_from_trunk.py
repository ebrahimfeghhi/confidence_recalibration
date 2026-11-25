#!/usr/bin/env python
"""MORE DIFFUSION SAMPLES FROM A CACHED TRUNK. The trunk is NOT rerun.

WHY THIS IS POSSIBLE. The expensive half of a fold is the trunk -- InputFeatureEmbedder, the MSA
module and N_cycle passes of the pairformer -- and its whole output for a unit is already on disk:
`s_inputs`, `s_trunk` and `z_trunk` in training/fold/trunks/<pdb>/<unit>.npz, 7,309 units over 856
structures. `sample_diffusion` consumes exactly those three plus an input_feature_dict, and takes
`pair_z=None, p_lm=None, c_l=None` -- protenix.py:737 already does that and the comment there says
why: passing None makes the sampler recompute the diffusion-conditioning cache from z_trunk, which
is what the model does whenever enable_diffusion_shared_vars_cache is off.

WHAT IS NOT CACHED, AND WHY RE-DERIVING IT IS NOT "RERUNNING THE NETWORK". The npz keeps only the
confidence head's slice of the feature dict (asym_id, atom_to_token_idx, has_frame, is_ligand,
distogram_rep_atom_mask, atom_to_tokatom_idx). The diffusion module additionally reads ref_pos,
ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid and the local-atom geometry
d_lm / v_lm / pad_info. Every one of those is FEATURISATION, not inference: the ref_* come off the
bioassembly through protenix.data.core.featurizer, and d_lm / v_lm / pad_info are three lines of
arithmetic on ref_pos in protenix.model.protenix.update_input_feature_dict. No pairformer, no MSA
module, no recycling. The same dataset object phase_j_build_cache used to write the labels
produces them, so the atom order is the order the cached trunk was computed in -- asserted, not
assumed, by BC.check_alignment.

WHAT THE SAMPLES ARE. Fresh diffusion noise against a FIXED trunk, i.e. more samples of the SAME
seed -- the within-seed axis. The cached unit's own 5 poses stay on disk untouched; this writes a
new npz beside nothing and deletes nothing.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--unit", default="", help="trunk unit stem, e.g. free_500000. "
                                              "Empty = the structure's only unit, or fail.")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--seed", type=int, default=777000,
                    help="the DIFFUSION noise seed. Not the fold seed: the trunk is fixed.")
    ap.add_argument("--out", default="", help="npz to write; empty = report only, write nothing")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index",
                    default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    a = ap.parse_args()

    # WITHOUT IT THE COORDINATES ARE SILENTLY NaN -- build_model refuses rather than let that
    # through, and this is the same contract every other entry point in the repo signs.
    os.environ.setdefault("LAYERNORM_TYPE", "openfold")

    import numpy as np
    import torch
    import phase_j_build_cache as BC
    import phase_j_dataset as DS
    import phase_j_rescore_cache as RC

    unit = a.unit
    if not unit:
        import glob
        got = sorted(glob.glob(f"{a.fold}/trunks/{a.pdb}/*.npz"))
        if len(got) != 1:
            raise SystemExit(f"STOP: {a.pdb} has {len(got)} trunk unit(s); name one with --unit")
        unit = os.path.basename(got[0])[:-4]
    trunk_p = f"{a.fold}/trunks/{a.pdb}/{unit}.npz"
    if not os.path.exists(trunk_p):
        raise SystemExit(f"STOP: no cached trunk at {trunk_p}")

    dev = torch.device("cuda")
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.model.protenix import update_input_feature_dict

    import time
    _t = {}
    _t0 = time.perf_counter()
    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    os.makedirs(err, exist_ok=True)
    train, _ = get_datasets(cfg, err)
    _t["dataset_build"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    byp = BC.merged_lookup(train)
    _t["lookup"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    if a.pdb not in byp:
        raise SystemExit(f"STOP: {a.pdb} is not in the dataset built from {a.indices}")
    item = train[byp[a.pdb][0]]
    feat = item["input_feature_dict"]
    _t["item_featurise"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()

    feat_head, s_inputs, s_trunk, zt, poses, n_cycle = RC.load_unit(trunk_p, dev)
    # THE ATOM ORDER IS ASSERTED, NOT ASSUMED. The cached trunk and this freshly featurised dict
    # come from the same dataset code, but a drift in the stores would otherwise show up as poses
    # that look plausible and are wrong.
    # ON THE HOST. check_alignment calls np.asarray on each tensor, which refuses a CUDA one
    # ("can't convert cuda:0 device type tensor to numpy"); load_unit returns device tensors.
    BC.check_alignment({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in feat_head.items()},
                       {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in feat.items()}, a.pdb)

    _t["load_unit"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    model, mcfg = RC.build_model(a.ckpt_dir, a.model_name, dev)
    _t["build_model"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    feat = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feat.items()}
    # `relp` AND the local-atom geometry, the two things the npz does not carry and the diffusion
    # module reads. Both are FEATURISATION, not inference: generate_relp is a no-weight method on
    # RelativePositionEncoding (one-hots of residue/token/chain index differences, protenix.py:1836
    # calls it on exactly this dict) and update_input_feature_dict is three lines of arithmetic on
    # ref_pos. Without relp the diffusion module raises KeyError: 'relp' at diffusion.py:397.
    feat = model.relative_position_encoding.generate_relp(feat)
    feat = update_input_feature_dict(feat)

    _t["relp_plus_dlm"] = time.perf_counter() - _t0
    n_step = int(mcfg.sample_diffusion["N_step"])
    _t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        xyz = model.sample_diffusion(
            denoise_net=model.diffusion_module,
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=zt,
            pair_z=None,
            p_lm=None,
            c_l=None,
            N_sample=int(a.nsample),
            noise_schedule=model.inference_noise_scheduler(
                N_step=n_step, device=dev, dtype=s_inputs.dtype),
            enable_efficient_fusion=model.enable_efficient_fusion,
        )
    torch.cuda.synchronize()
    _t["sample_diffusion"] = time.perf_counter() - _t0
    print("  timing (s): " + "  ".join(f"{k} {v:.1f}" for k, v in _t.items()))
    new = xyz.detach().float().cpu().numpy().astype(np.float32)
    old = poses.detach().float().cpu().numpy().astype(np.float32) if torch.is_tensor(poses) \
        else np.asarray(poses, dtype=np.float32)
    print(f"{a.pdb}/{unit}: cached poses {old.shape}  new poses {new.shape}  "
          f"N_step {n_step}  diffusion seed {a.seed}")

    # A POSE MUST NOT BE A COPY OF A CACHED ONE, and must not be a copy of its own siblings.
    # Reported as plain atom-wise RMSD after no superposition at all: the sampler emits all
    # samples in one frame, so a zero here means an identical draw, which is the failure this
    # guards against.
    def rmsd(p, q):
        return float(np.sqrt(((p - q) ** 2).sum(-1).mean()))
    print("  new-vs-cached RMSD (no superposition), min over cached per new pose:")
    for i in range(new.shape[0]):
        d = [rmsd(new[i], old[j]) for j in range(old.shape[0])]
        print(f"    new {i}: min {min(d):8.3f} A   (per cached: "
              + " ".join(f"{x:.2f}" for x in d) + ")")
    print("  new-vs-new RMSD:")
    for i in range(new.shape[0]):
        print("    " + " ".join(f"{rmsd(new[i], new[j]):8.3f}" for j in range(new.shape[0])))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "wb") as fh:
            np.savez_compressed(
                fh, coords=new,
                atom_to_token_idx=feat["atom_to_token_idx"].cpu().numpy().astype(np.int32),
                asym_id=feat["asym_id"].cpu().numpy().astype(np.int32),
                pdb=np.str_(a.pdb), unit=np.str_(unit),
                diffusion_seed=np.int32(a.seed), n_step=np.int32(n_step))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
