#!/usr/bin/env python
"""+5 MORE POSES PER CACHED TRUNK, over the whole store. The trunk is never rerun.

See run_logging/rediffuse_from_trunk.py for why this is possible and what is and is not cached.
This is that script as a resumable batch: the model and the dataset are built once, each
structure is featurised once and reused across all of its seeds, and every unit writes its own
npz so a kill costs at most one unit.

THE NEW POSES ARE AN INDEPENDENT DRAW FROM THE SAME TRUNK, not a continuation of the original
RNG stream. torch is seeded per unit with --seed-base + the unit's fold seed, so the run is
reproducible and no two units share noise, but the 5 new poses are not "samples 5-9" of the
original sampling -- they are 5 more samples of the same conditioning. For the within-seed
question that is exactly the quantity wanted; for anything that needs the original stream
continued, it is not.

WHAT THIS DOES NOT PRODUCE. Coordinates only. A new pose has no DockQ (OST against the
reference), no ipTM (a confidence-head pass) and no label_coordinate (symmetric permutation
against the deposited structure). Those are separate jobs and the expensive one is the first.
"""
import argparse
import collections
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/fold/poses_extra")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--only", default="", help="comma-separated pdb ids")
    ap.add_argument("--limit-structures", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = no cap")
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

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    import numpy as np
    import torch
    import phase_j_build_cache as BC
    import phase_j_dataset as DS
    import phase_j_rescore_cache as RC
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.model.protenix import update_input_feature_dict

    units = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{a.fold}/trunks/*/*.npz")):
        units[os.path.basename(os.path.dirname(p))].append(p)
    if a.only:
        keep = {x.strip() for x in a.only.split(",") if x.strip()}
        units = {k: v for k, v in units.items() if k in keep}

    tok = {}
    import csv as _csv
    with open(a.indices) as fh:
        for r in _csv.DictReader(fh):
            try:
                tok[r["pdb_id"]] = int(r["num_tokens"])
            except (KeyError, ValueError):
                pass
    if a.max_tokens:
        units = {k: v for k, v in units.items()
                 if tok.get(k, 10 ** 9) <= int(a.max_tokens)}
    pdbs = sorted(units)
    if a.limit_structures:
        pdbs = pdbs[:int(a.limit_structures)]
    todo = sum(1 for p in pdbs for u in units[p]
               if not os.path.exists(os.path.join(a.out_dir, p,
                                                  os.path.basename(u))))
    print(f"{len(pdbs)} structure(s), {sum(len(units[p]) for p in pdbs)} unit(s), "
          f"{todo} still to do", flush=True)
    if not todo:
        print("nothing to do")
        return

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    os.makedirs(err, exist_ok=True)
    train, _ = get_datasets(cfg, err)
    byp = BC.merged_lookup(train)
    model, mcfg = RC.build_model(a.ckpt_dir, a.model_name, dev)
    n_step = int(mcfg.sample_diffusion["N_step"])
    print(f"setup {time.perf_counter() - t0:.0f}s   N_step {n_step}   "
          f"seed-base {a.seed_base}   out {os.path.relpath(a.out_dir, ROOT)}", flush=True)

    done = fail = 0
    t_start = time.perf_counter()
    for pi, pdb in enumerate(pdbs, 1):
        mine = [u for u in units[pdb]
                if not os.path.exists(os.path.join(a.out_dir, pdb, os.path.basename(u)))]
        if not mine:
            continue
        if pdb not in byp:
            print(f"  {pdb}: not in the dataset, {len(mine)} unit(s) skipped", flush=True)
            fail += len(mine)
            continue
        try:
            # ONCE PER STRUCTURE. The feature dict does not depend on the seed, and building it
            # is ~5-10 s of CPU that would otherwise be paid 15 times over.
            feat0 = train[byp[pdb][0]]["input_feature_dict"]
            feat = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feat0.items()}
            feat = model.relative_position_encoding.generate_relp(feat)
            feat = update_input_feature_dict(feat)
        except Exception as e:                                # noqa: BLE001
            print(f"  {pdb}: featurise failed, {type(e).__name__}: {e}", flush=True)
            fail += len(mine)
            continue
        for up in mine:
            unit = os.path.basename(up)[:-4]
            try:
                fh_, s_inputs, s_trunk, zt, poses, _nc = RC.load_unit(up, dev)
                BC.check_alignment({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                    for k, v in fh_.items()},
                                   {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                    for k, v in feat.items()}, pdb)
                sd = int(unit.split("_")[-1]) if unit.split("_")[-1].isdigit() else 0
                torch.manual_seed(int(a.seed_base) + sd)
                torch.cuda.manual_seed_all(int(a.seed_base) + sd)
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                    xyz = model.sample_diffusion(
                        denoise_net=model.diffusion_module, input_feature_dict=feat,
                        s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                        pair_z=None, p_lm=None, c_l=None, N_sample=int(a.nsample),
                        noise_schedule=model.inference_noise_scheduler(
                            N_step=n_step, device=dev, dtype=s_inputs.dtype),
                        enable_efficient_fusion=model.enable_efficient_fusion)
                new = xyz.detach().float().cpu().numpy().astype(np.float32)
                os.makedirs(os.path.join(a.out_dir, pdb), exist_ok=True)
                tmp = os.path.join(a.out_dir, pdb, os.path.basename(up) + ".part")
                with open(tmp, "wb") as f:
                    np.savez_compressed(
                        f, coords=new,
                        atom_to_token_idx=feat["atom_to_token_idx"].cpu().numpy().astype(np.int32),
                        asym_id=feat["asym_id"].cpu().numpy().astype(np.int32),
                        pdb=np.str_(pdb), unit=np.str_(unit),
                        diffusion_seed=np.int32(int(a.seed_base) + sd),
                        n_step=np.int32(n_step))
                # ATOMIC. A kill mid-write would otherwise leave a short npz that the
                # skip-existing check would accept as finished.
                os.replace(tmp, os.path.join(a.out_dir, pdb, os.path.basename(up)))
                done += 1
                del fh_, s_inputs, s_trunk, zt, poses, xyz, new
            except Exception as e:                            # noqa: BLE001
                print(f"  {pdb}/{unit}: {type(e).__name__}: {e}", flush=True)
                fail += 1
            torch.cuda.empty_cache()
        del feat, feat0
        torch.cuda.empty_cache()
        if pi % 10 == 0 or pi == len(pdbs):
            el = time.perf_counter() - t_start
            rate = done / el if el else 0
            left = (todo - done) / rate / 3600 if rate else float("nan")
            print(f"  [{pi}/{len(pdbs)}] {done} unit(s) in {el / 60:.0f} min "
                  f"({rate * 3600:.0f}/h), {fail} failed, ~{left:.1f} h left", flush=True)
    print(f"DONE  {done} unit(s) written, {fail} failed")


if __name__ == "__main__":
    main()
