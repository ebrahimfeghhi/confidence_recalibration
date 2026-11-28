#!/usr/bin/env python
"""+5 MORE POSES PER CACHED TRUNK, over the whole store. The trunk is never rerun.

See run_logging/rediffuse_from_trunk.py for why this is possible and what is and is not cached.
This is that script as a resumable batch: the model and the dataset are built once, each
structure is featurised once and reused across all of its seeds, and every unit writes its own
npz so a kill costs at most one unit.

THE NEW POSES ARE AN INDEPENDENT DRAW FROM THE SAME TRUNK, not a continuation of the original
RNG stream. torch is seeded per unit with --seed-base + the unit's fold seed, so no two units
share noise, but the 5 new poses are not "samples 5-9" of the original sampling -- they are 5
more samples of the same conditioning. For the within-seed question that is exactly the quantity
wanted; for anything that needs the original stream continued, it is not.

THE SEED DOES NOT MAKE THIS BIT-REPRODUCIBLE, measured rather than assumed. Two processes ran the
same unit with the same seed during an accidental overlap and their ipTM disagreed by up to
0.0615 -- the sampler's kernels are not deterministic, so a rerun gives a valid draw from the
same conditioning and NOT the same coordinates. Anything that needs the exact poses must read the
npz; regenerating is not a substitute for it.

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
    ap.add_argument("--sample-base", type=int, default=100,
                    help="sample id of the first new pose. 100 was the 5->10 batch; the 10->15 "
                         "batch passes 105 so the two never collide in the merged tables.")
    ap.add_argument("--shard", default="",
                    help="I/N -- take every Nth structure starting at I (0-based) off the "
                         "size-sorted list, so N instances on N cards split the work with no "
                         "overlap. Each shard locks and writes its own scores csv; the pose "
                         "npz paths are disjoint because the structures are.")
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
    ap.add_argument("--score", action="store_true", default=True,
                    help="score ipTM for every new pose in the same loop (default). The model and"
                         " the trunk are already resident, so this costs one confidence-head "
                         "forward per unit against a 200-step diffusion -- a few percent -- "
                         "whereas a second pass would reload and re-read everything.")
    ap.add_argument("--no-score", dest="score", action="store_false")
    ap.add_argument("--pool-csv", default=f"{ROOT}/training/combined/pool.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--scores-out", default=f"{ROOT}/training/fold/poses_extra/results_extra.csv",
                    help="SEPARATE FROM results_with_trunks.csv on purpose: the 5-pose and "
                         "10-pose worlds stay separable until we deliberately join them.")
    a = ap.parse_args()

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")

    # ONE INSTANCE AT A TIME, ENFORCED BY A LOCK RATHER THAN BY CARE.
    #
    # Two instances pointed at the same --out-dir write the same npz paths and append to the
    # same csv. That happened: a chained shell outlived the kill of its parent, fired when the
    # first pass ended, and ran beside a manual relaunch. The damage was not loud -- 15
    # duplicate rows over 3 units, which looked like a cosmetic dedupe until the two copies
    # turned out to DISAGREE on ipTM by up to 0.0615, meaning one process's coordinates were on
    # disk under the other's scores. An exclusive flock makes the second instance refuse at
    # startup instead.
    import fcntl
    _shard_tag = ("_shard" + a.shard.replace("/", "of")) if a.shard else ""
    _lockp = os.path.join(os.path.dirname(os.path.abspath(a.out_dir)) or ".",
                          f".{os.path.basename(a.out_dir)}{_shard_tag}.lock")
    if a.shard:
        _si, _sn = (int(x) for x in a.shard.split("/"))
        if not 0 <= _si < _sn:
            raise SystemExit(f"STOP: --shard {a.shard} is out of range")
        _root, _ext = os.path.splitext(a.scores_out)
        a.scores_out = f"{_root}{_shard_tag}{_ext}"
    os.makedirs(os.path.dirname(_lockp) or ".", exist_ok=True)
    _lock = open(_lockp, "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"STOP: another rediffuse_batch already holds {_lockp}. Two instances on one "
            f"--out-dir corrupt it silently; stop the other one or pass a different --out-dir.")
    _lock.write(f"{os.getpid()}\n")
    _lock.flush()

    import numpy as np
    import torch
    import phase_j_build_cache as BC
    import phase_j_dataset as DS
    import phase_j_rescore_cache as RC
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.model.protenix import update_input_feature_dict
    import rank_loss as RKL

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
    # CHEAPEST FIRST. Cost goes as tokens^1.71, so alphabetical order mixes 300-token and
    # 2,300-token structures at random and the usable set grows at the mean rate. Ascending size
    # finishes far more structures per hour early and leaves the expensive tail at the end,
    # where it can be cut without losing the rest. A structure with no token count sorts last.
    pdbs = sorted(units, key=lambda p: (tok.get(p, 10 ** 9), p))
    if a.shard:
        # INTERLEAVED, NOT CONTIGUOUS. The list is sorted cheapest-first, so three contiguous
        # blocks would hand one card every expensive structure and it would still be running
        # long after the other two finished.
        pdbs = pdbs[_si::_sn]
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

    # THE SCORED CHAIN PAIRS, per structure, through the SAME join the trainer and phase_k_eval
    # use -- model asym indices, never the iface letters. run_logging/validate_iptm_path.py shows
    # this path reproduces Protenix's own stored ipTM on 99.34% of rows to 2e-3.
    pairs = collections.defaultdict(list)
    if a.score:
        roles_join = RKL.load_roles(a.roles)
        with open(a.pool_csv) as fh:
            for r in _csv.DictReader(fh):
                ifc = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
                pr = RKL.parse_iface(ifc, roles_join)
                if pr is not None:
                    pairs[pr[0]].append((ifc, (int(pr[1][0]), int(pr[1][1]))))
        print(f"scoring ON: {sum(len(v) for v in pairs.values())} interface(s) over "
              f"{len(pairs)} structure(s) -> {os.path.relpath(a.scores_out, ROOT)}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)), exist_ok=True)
        if not os.path.exists(a.scores_out):
            with open(a.scores_out, "w", newline="") as fh:
                _csv.writer(fh).writerow(["pdb", "iface", "seed", "sample", "iptm",
                                          "iptm_mean", "diffusion_seed", "source"])
        # THE HEADER MUST MATCH WHAT THE ROWS CARRY. A stale header is not a cosmetic
        # problem: csv.DictReader keys every row by it, so an eighth column read back under a
        # seven-name header is silently dropped or shifted. Checked here rather than trusted,
        # because the first version of this file shipped exactly that mismatch.
        with open(a.scores_out) as fh:
            _hdr = next(_csv.reader(fh), [])
        _want_hdr = ["pdb", "iface", "seed", "sample", "iptm", "iptm_mean",
                     "diffusion_seed", "source"]
        if _hdr != _want_hdr:
            raise SystemExit(
                f"STOP: {a.scores_out} has header {_hdr}, expected {_want_hdr}. An existing "
                f"file from an older column set must be moved aside rather than appended to.")

        # WHAT IS ALREADY SCORED, so a resume does not duplicate rows.
        scored = set()
        with open(a.scores_out) as fh:
            for r in _csv.DictReader(fh):
                scored.add((r["pdb"], int(r["seed"])))
        print(f"  {len(scored)} unit(s) already scored", flush=True)
    else:
        scored = set()

    done = fail = nscore = 0
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
                # ipTM FOR THE NEW POSES, in this loop, while the trunk and the model are
                # resident. Sample ids are --sample-base + i so a new pose can never be confused with one
                # of the original five, in any table that later joins the two.
                if a.score and (pdb, sd) not in scored and pairs.get(pdb):
                    with torch.no_grad(), torch.autocast(device_type="cuda",
                                                         dtype=torch.bfloat16):
                        _pl, pae_n, _pd, _rs = model.run_confidence_head(
                            input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk,
                            z_trunk=zt, pair_mask=None, x_pred_coords=xyz,
                            triangle_multiplicative=mcfg.triangle_multiplicative,
                            triangle_attention=mcfg.triangle_attention,
                            inplace_safe=False, chunk_size=None)
                        lg = pae_n.detach().float()
                        rowsout = []
                        for i_s in range(int(a.nsample)):
                            one = lg[i_s:i_s + 1] if lg.dim() == 4 else lg[0][i_s:i_s + 1]
                            for ifc, pr_ in pairs[pdb]:
                                # BOTH POOLINGS. `iptm` is the shipped MAX over anchor rows, which
                                # rests a pose's whole score on one row; `iptm_mean` is the same
                                # quantity pooled by the mean, which the ceiling work found
                                # recovers more of the oracle headroom (67.7% vs 58.6%). Computed
                                # here because the logits are already in hand.
                                sc_ = RKL.iptm(one, feat, mcfg, pr_)
                                if sc_ is None:
                                    continue
                                sm_ = RKL.iptm_pooled(one, feat, mcfg, pr_, pool="mean")
                                rowsout.append([pdb, ifc, sd, a.sample_base + i_s,
                                                round(float(sc_), 6),
                                                (round(float(sm_), 6) if sm_ is not None
                                                 else ""),
                                                int(a.seed_base) + sd, "rediffuse"])
                    if rowsout:
                        with open(a.scores_out, "a", newline="") as fh:
                            _csv.writer(fh).writerows(rowsout)
                        nscore += len(rowsout)
                    del pae_n, lg
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
                  f"({rate * 3600:.0f}/h), {fail} failed, {nscore} ipTM row(s), "
                  f"~{left:.1f} h left", flush=True)
    print(f"DONE  {done} unit(s) written, {fail} failed, {nscore} ipTM row(s)")


if __name__ == "__main__":
    main()
