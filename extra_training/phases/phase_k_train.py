"""PHASE K -- the FT3 finetune: train the confidence head on our own pose distribution.

WHAT THIS OPTIMISES, AND WHAT IT IS JUDGED ON. The head predicts pLDDT, PAE, PDE and resolved. Only
PAE matters for the thing we are measured on: sample_confidence.py builds EVERY ipTM number out of
the PAE logits (`pae_prob = logits_to_score(pae_logits, ...)`, then `calculate_iptm(pae_prob, ...)`
and `calculate_chain_based_ptm(pae_prob, ...)`), and `chain_pair_iptm` is what selects a pose. So
`alpha_pae` is not a tuning knob here -- at Protenix's shipped default of 0.0 the PAE term is
multiplied by zero, ipTM cannot move, and the whole experiment is a no-op whose total loss still
falls because pLDDT, PDE and resolved keep training. Their own comment says `# or 1 in finetuning
stage 3`. This script REFUSES to start if any of the four weights is zero.

The other three terms stay on as regularisers: they keep the head honest about per-atom and
distance-error confidence instead of letting it distort everything to fit PAE.

THE TARGET, measured 2026-10-15 over 92 validation interfaces at 100 seeds
(analysis/val_selection_metric.py): selecting by max pair_iptm gives mean DockQ 0.5776 and
frac-acceptable 0.7283, against an oracle of 0.7184 / 0.8696. So there is 0.1408 of mean DockQ
available to a better ranker AT ZERO EXTRA COMPUTE, and that gap is the deliverable. Note ipTM
selection PEAKS near 25 seeds and DECAYS to 100 while the oracle keeps climbing -- more sampling
produces better poses and a greater chance confidence prefers a confidently wrong one.

WHY THE FORWARD PASS IS IMPORTED, NOT WRITTEN. phase_j_replay.py already assembles the model, sets
train_confidence_only, builds the loss, runs the head and backpropagates, and it is the GATE that
proved a cached unit reproduces live FT3 exactly -- 0.00e+00 on the label and on every loss term over
5 units and 25 poses. A trainer that rebuilt that path could drift from the thing that was verified,
so this imports the same helpers the replay and the rescorer use.

WHAT IS FROZEN, ASSERTED RATHER THAN ASSUMED. 19.8M of 464.4M parameters (4.3%) are the confidence
head. protenix.py's train_confidence_only puts the trunk in eval mode, but it does NOT tell the
optimiser what it may touch, so the optimiser is given the head's parameters by module-tree name and
--grad-check-every steps re-assert that no gradient has appeared outside it. The replay measured the
expected answer: 165 head tensors with a gradient, 0 outside.

ONE UNIT IS ONE (structure, seed) WITH ITS 5 POSES. The dataset item is per STRUCTURE and is
expensive to build, so units are grouped by structure and the item is built once per structure --
which also means the unit order is not shuffled freely; see --shuffle.
"""
import argparse
import copy
import csv
import functools
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
print = functools.partial(print, flush=True)

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss")


def head_param_names(model):
    """The parameters FT3 may move, taken from the model's own module tree.

    Not a hand-written list: the confidence head is its own module, so its name prefix is the honest
    definition and a rename upstream surfaces as "no module named *confidence_head" rather than as a
    silently empty optimiser.
    """
    prefixes = [n for n, _m in model.named_modules() if n.endswith("confidence_head")]
    if not prefixes:
        raise SystemExit("STOP: the model has no module named *confidence_head, so the set of "
                         "trainable parameters cannot be established from the module tree.")
    names = [n for n, _p in model.named_parameters()
             if any(n == p or n.startswith(p + ".") for p in prefixes)]
    if not names:
        raise SystemExit(f"STOP: {prefixes} matched no parameters; the optimiser would be empty.")
    return prefixes, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--labels", default="", help="default <fold>/labels")
    ap.add_argument("--out", default=f"{ROOT}/training/ft3", help="checkpoints and trajectory")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--alpha-pae", type=float, default=1.0,
                    help="FT3 wants 1. At Protenix's shipped 0.0 the PAE term is multiplied by zero "
                         "and ipTM -- the only number this project is judged on -- cannot move.")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Protenix's global default is 0.0018, tuned for finetuning the whole model. "
                         "This trains 19.8M parameters (4.3%%), so the default is not transferable "
                         "and --lr-probe exists to measure it rather than inherit it.")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--accum", type=int, default=8,
                    help="poses per optimiser step. Token counts vary per structure so units cannot "
                         "be batched into one tensor; accumulation is how the effective batch is set.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit-units", type=int, default=0, help="debug: cap the number of units")
    ap.add_argument("--shuffle", default="structure",
                    choices=["structure", "none"],
                    help="`structure` shuffles the ORDER OF STRUCTURES and the units within each, "
                         "which is as much as can be shuffled while building each structure's "
                         "dataset item once. `none` is deterministic for debugging.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preload-items", type=int, default=0,
                    help="how many STRUCTURE dataset items to hold in memory at once. 0 keeps one, "
                         "which rebuilds every structure every epoch. MEASURED 2026-10-15: an item "
                         "takes 3.6-10.4 s to build (avg 6.8) and occupies 331-886 MB (avg 518), so "
                         "500 structures cost ~57 min per epoch to rebuild and ~253 GB to hold. With "
                         "837 GB free that fits, and every epoch after the first saves the 57 min. "
                         "Use 0 for a single-epoch run, where there is nothing to amortise.")
    ap.add_argument("--max-nonfinite", type=int, default=10,
                    help="how many non-finite-gradient steps to tolerate before stopping. Each one "
                         "drops its gradients and takes no step; past this the data or the learning "
                         "rate is wrong and continuing trains on noise.")
    ap.add_argument("--grad-check-every", type=int, default=25,
                    help="re-assert that no gradient has appeared outside the confidence head")
    ap.add_argument("--save-every", type=int, default=250, help="optimiser steps per checkpoint")
    ap.add_argument("--lr-probe", default="",
                    help="comma-separated learning rates. Runs --probe-steps steps at each from the "
                         "SAME initial weights and reports the loss trend, then exits without "
                         "training. This is how --lr is chosen instead of inherited.")
    ap.add_argument("--probe-steps", type=int, default=30)
    ap.add_argument("--require-replay", default="",
                    help="path to a replay result that must say PASS before training starts. "
                         "Cached training is only FT3 if the cache reproduces it; see "
                         "phase_j_replay.py. Pass 'skip' to train without the gate and say why.")
    a = ap.parse_args()
    a.labels = a.labels or os.path.join(a.fold, "labels")

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold. Without it the trunk's coordinates are "
                         "silently NaN -- phase_j_replay.py's guard fired on this first.")

    # THE GATE. A cached-training run is only an FT3 run if the cached halves reproduce what the live
    # code computes. That is phase_j_replay.py's job and it PASSES; requiring its result here is what
    # stops a future run from quietly skipping it.
    if a.require_replay != "skip":
        if not a.require_replay:
            raise SystemExit(
                "STOP: --require-replay not given. Run\n"
                "  CUDA_VISIBLE_DEVICES=0 LAYERNORM_TYPE=openfold python phases/phase_j_replay.py "
                "--unit <pdb>/free_<seed> > replay.log\n"
                "and pass that log, or pass --require-replay skip with a reason recorded.")
        if not os.path.exists(a.require_replay):
            raise SystemExit(f"STOP: no replay result at {a.require_replay}")
        txt = open(a.require_replay).read()
        if "PASS:" not in txt:
            raise SystemExit(f"STOP: {a.require_replay} does not say PASS. Cached training would be "
                             f"optimising something other than FT3.")
        print(f"replay gate: PASS ({a.require_replay})")

    import torch
    import phase_j_dataset as DS
    import phase_j_build_cache as BC
    import phase_j_rescore_cache as RC
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.model.loss import ProtenixLoss
    from protenix.utils.permutation.permutation import SymmetricPermutation

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    configs.train_confidence_only = True
    configs.loss.weight.alpha_pae = float(a.alpha_pae)
    loss_fn = ProtenixLoss(configs)
    w = {k: loss_fn.loss_weight[k] for k in ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")}
    print("loss weights: " + "  ".join(f"{k.replace('_loss','')} {v:g}" for k, v in w.items()))
    zero = [k for k, v in w.items() if not v]
    if zero:
        raise SystemExit(f"STOP: {', '.join(zero)} carries weight 0, so that term cannot train. "
                         f"FT3 wants alpha_pae = 1 (Protenix's own comment, configs_base.py:402); "
                         f"at 0 the ipTM this project is judged on cannot move.")

    prefixes, trainable = head_param_names(model)
    head_ids = {id(p) for n, p in model.named_parameters() if n in set(trainable)}
    n_head = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
    n_all = sum(p.numel() for _n, p in model.named_parameters())
    print(f"trainable: {len(trainable)} tensor(s), {n_head/1e6:.1f}M of {n_all/1e6:.1f}M parameters "
          f"({n_head/n_all:.1%}), under {prefixes}")
    for n, p in model.named_parameters():
        p.requires_grad_(id(p) in head_ids)

    # ---- THE MANIFEST: every unit's label AND trunk resolved up front, and nothing skipped
    # SILENTLY SKIPPING A UNIT IS TRAINING ON A SUBSET CHOSEN BY WHICH FILES HAPPENED TO EXIST.
    # run_unit used to `return` when a trunk or a dataset entry was absent, so a missing input
    # reduced the training set with no record -- the same class of silent selection this project
    # keeps finding. Every unit is now resolved here, against every root in --fold, and a unit whose
    # trunk or dataset entry cannot be found STOPS the run with the list.
    roots = [r.strip() for r in a.fold.split(",") if r.strip()]
    for r in roots:
        if not os.path.isdir(r):
            raise SystemExit(f"STOP: --fold root {r} does not exist")
    label_dirs = [x.strip() for x in a.labels.split(",") if x.strip()]
    labels_found = []
    for d in label_dirs:
        labels_found += sorted(glob.glob(os.path.join(d, "*", "*.npz")))
    if not labels_found:
        raise SystemExit(f"STOP: no label npz under {label_dirs}; run phase_j_build_cache.py")

    manifest, no_trunk = [], []
    for lp in labels_found:
        pdb = os.path.basename(os.path.dirname(lp))
        arm_seed = os.path.basename(lp)[:-4]
        tp = next((os.path.join(r, "trunks", pdb, f"{arm_seed}.npz") for r in roots
                   if os.path.exists(os.path.join(r, "trunks", pdb, f"{arm_seed}.npz"))), None)
        if tp is None:
            no_trunk.append(f"{pdb}/{arm_seed}")
            continue
        manifest.append((pdb, lp, tp))
    if no_trunk:
        raise SystemExit(f"STOP: {len(no_trunk)} labelled unit(s) have no trunk under any of "
                         f"{roots}, e.g. {no_trunk[:5]}. Training would silently drop them.")

    bypdb = {}
    for pdb, lp, tp in manifest:
        bypdb.setdefault(pdb, []).append((lp, tp))
    order = sorted(bypdb)
    rng = random.Random(a.seed)
    if a.shuffle == "structure":
        rng.shuffle(order)
        for pp in order:
            rng.shuffle(bypdb[pp])
    flat = [(pp, lp, tp) for pp in order for lp, tp in bypdb[pp]]
    if a.limit_units:
        flat = flat[:a.limit_units]
    print(f"manifest: {len(flat)} unit(s) over {len(set(x[0] for x in flat))} structure(s) from "
          f"{len(roots)} fold root(s), shuffle={a.shuffle}, seed={a.seed}")

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    train_ds, _ = get_datasets(cfg, err)
    byp = BC.merged_lookup(train_ds)
    perm = SymmetricPermutation(cfg, error_dir=err)
    # EVERY STRUCTURE IN THE MANIFEST MUST HAVE A DATASET ENTRY, asserted here rather than
    # discovered per unit and skipped.
    nodata = sorted({pdb for pdb, _lp, _tp in flat if pdb not in byp})
    if nodata:
        raise SystemExit(f"STOP: {len(nodata)} structure(s) in the manifest have no dataset "
                         f"datapoint: {nodata[:8]}. Their units would be skipped silently.")

    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if n in set(trainable)],
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    init_state = copy.deepcopy({n: p.detach().clone() for n, p in model.named_parameters()
                                if n in set(trainable)})
    init_opt = copy.deepcopy(opt.state_dict())

    # PERSIST THE TRAJECTORY PER STEP, not at the end. A run that dies at step 900 of 1000 must still
    # be analysable, and a loss curve is the only way to see a learning rate that is wrong.
    traj_p = os.path.join(a.out, "trajectory.csv")
    traj_cols = ["wall_s", "step", "unit_i", "pdb", "unit", "pose", "lr"] + list(TERMS) + \
                ["grad_norm", "outside_head", "epoch"]
    # the per-structure item build is the cost --preload-items exists to amortise, so it is measured
    # rather than assumed; reported at the end against the number of structures actually visited.
    item_build_s = []
    nonfinite = {"n": 0}

    def run_unit(pdb, lab_p, trunk_p, item_cache):
        """Forward + loss for every pose of one unit. Yields (pose_index, loss_tensor, metrics).

        Both paths are resolved by the manifest and asserted to exist before the run starts, so
        there is nothing here to skip.
        """
        arm_seed = os.path.basename(lab_p)[:-4]
        if pdb not in item_cache:
            # EVICT ONLY WHEN THE BUDGET IS REACHED. Holding one item at a time means every structure
            # is rebuilt every epoch, which the numbers above make the dominant cost of a multi-epoch
            # run. FIFO rather than LRU because --shuffle visits each structure's units consecutively,
            # so recency carries no information worth tracking.
            cap = max(1, a.preload_items)
            while len(item_cache) >= cap:
                item_cache.pop(next(iter(item_cache)))
            _t = time.time()
            it = train_ds[byp[pdb][0]]
            item_cache[pdb] = (it["input_feature_dict"], it["label_dict"], it["label_full_dict"])
            item_build_s.append(time.time() - _t)
        feat_ds, label_dict, label_full = item_cache[pdb]
        lab = np.load(lab_p, allow_pickle=True)
        # THE CACHE'S PROVENANCE IS REQUIRED, NOT OPTIONAL. build_cache stores a `fingerprint` of
        # the job cif, bioassembly, index rows, trunk and code the label was built from, plus a
        # `schema`. Training read neither, so a label written by older code or from inputs that have
        # since changed would be trained on with nothing noticing. The lf_* loss features are the
        # same story: `if kk in lab` silently fell back to the LIVE features, which are not the ones
        # the label was built against -- exactly the substitution the replay gate exists to detect.
        for req in ("fingerprint", "schema", "label_coordinate", "label_coordinate_mask",
                    "sample_index"):
            if req not in lab:
                raise SystemExit(f"STOP: {lab_p} has no `{req}`. It was written by an older "
                                 f"build_cache; rebuild it before training.")
        if int(np.asarray(lab["schema"])) != int(BC.SCHEMA):
            raise SystemExit(f"STOP: {lab_p} is schema {int(np.asarray(lab['schema']))}, this code "
                             f"expects {int(BC.SCHEMA)}. Rebuild the cache.")
        _missing_lf = [k for k in BC.LOSS_FEATS if f"lf_{k}" not in lab]
        if _missing_lf:
            raise SystemExit(f"STOP: {lab_p} stores no lf_ array for {_missing_lf}. Falling back to "
                             f"the live features would score the label against features it was not "
                             f"built with. Rebuild the cache.")
        # BOTH LOADERS, as the replay does. load_trunk gives the NUMPY features that
        # check_alignment compares against the dataset's -- it calls np.asarray, so handing it the
        # device tensors raises "can't convert cuda:0 device type tensor to numpy". load_unit gives
        # the device tensors the head consumes, and reverses the bf16-as-uint16 storage.
        _z, feat_npz, _poses_np = BC.load_trunk(trunk_p)
        BC.check_alignment(feat_npz, feat_ds, pdb)
        feat_head, s_inputs, s_trunk, zt, poses, _nc = RC.load_unit(trunk_p, dev)

        def to_dev(d):
            return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
        feat_live = to_dev({k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v)))
                            for k, v in feat_ds.items()
                            if torch.is_tensor(v) or isinstance(v, (np.ndarray, int, float))})
        for k, v in feat_ds.items():
            if k not in feat_live:
                feat_live[k] = v
        # the stored loss features are the ones the label was built with
        feat_cache = dict(feat_live)
        for k in BC.LOSS_FEATS:
            v = np.asarray(lab[f"lf_{k}"])          # presence asserted above
            feat_cache[k] = torch.as_tensor(v).to(dev) if v.dtype != object else lab[f"lf_{k}"]

        n_pose = poses.shape[0] if hasattr(poses, "shape") else len(poses)
        for i in range(n_pose):
            x = poses[i:i + 1].to(dev).float() if torch.is_tensor(poses) \
                else torch.as_tensor(poses[i:i + 1], dtype=torch.float32, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plddt, pae, pde, resolved = model.run_confidence_head(
                    input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                    pair_mask=None, x_pred_coords=x,
                    triangle_multiplicative=configs.triangle_multiplicative,
                    triangle_attention=configs.triangle_attention, inplace_safe=False,
                    chunk_size=None)
            pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                    "resolved": resolved.float(), "coordinate_mini": x}
            ld = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in label_dict.items()}
            ld["coordinate"] = torch.as_tensor(lab["label_coordinate"][i], dtype=torch.float32,
                                               device=dev)
            ld["coordinate_mask"] = torch.as_tensor(
                lab["label_coordinate_mask"][i].astype(np.float32), device=dev)
            cum, met = loss_fn.calculate_losses(feat_cache, pred, ld, mode="train")
            absent = [k for k in TERMS if k not in met]
            if absent:
                raise SystemExit(f"STOP: the loss produced no {absent} for {pdb}/{arm_seed} pose {i}; "
                                 f"a confidence term that stopped being produced reads as a lower "
                                 f"loss, which is the failure this check exists for.")
            if not math.isfinite(float(met["loss"])):
                raise SystemExit(f"STOP: {pdb}/{arm_seed} pose {i} loss is {float(met['loss'])}, "
                                 f"not finite. Training on it would destroy the head.")
            yield i, cum, met

    def outside_head_grads():
        return sum(1 for n, p in model.named_parameters()
                   if n not in set(trainable) and p.grad is not None
                   and bool(p.grad.abs().sum() > 0))

    def _apply_step(step):
        """Clip, assert, step. Returns (grad_norm, n_outside_head, ok).

        A FINITE LOSS DOES NOT MEAN FINITE GRADIENTS -- the loss is checked per pose already, and
        that is not the same assertion. error_if_nonfinite makes clip_grad_norm_ raise rather than
        hand inf or nan to AdamW, which would silently poison every head parameter through the
        exponential averages and never recover. The gradients are dropped and the run continues,
        counted, rather than the step being taken.
        """
        params = [pp for n, pp in model.named_parameters() if n in set(trainable)]
        try:
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        except RuntimeError as e:
            nonfinite["n"] += 1
            print(f"  NON-FINITE GRADIENT at step {step} ({e}); gradients dropped, no step taken "
                  f"({nonfinite['n']} so far)")
            opt.zero_grad(set_to_none=True)
            if nonfinite["n"] > a.max_nonfinite:
                raise SystemExit(f"STOP: {nonfinite['n']} non-finite gradient(s), over "
                                 f"--max-nonfinite {a.max_nonfinite}. The learning rate or the data "
                                 f"is wrong; continuing would train on noise.")
            return float("nan"), -1, False
        outside = -1
        if step % a.grad_check_every == 0:
            outside = outside_head_grads()
            if outside:
                raise SystemExit(
                    f"STOP: {outside} parameter(s) OUTSIDE the confidence head carry a non-zero "
                    f"gradient at step {step}. FT3 freezes the trunk; this is not FT3 any more.")
        opt.step()
        opt.zero_grad(set_to_none=True)
        return gn, outside, True

    def train_loop(steps_cap, lr, tag, write_traj):
        for g in opt.param_groups:
            g["lr"] = lr
        rows, t0, step, n_in_accum, item_cache = [], time.time(), 0, 0, {}
        opt.zero_grad(set_to_none=True)
        # AN EPOCH LOOP, BECAUSE --epochs USED TO DO NOTHING. It scaled steps_cap while the loop
        # traversed `flat` exactly once, so a request for more steps than the data provides simply
        # stopped early -- review round 7 asked for 5 optimiser steps and got 2. The data is now
        # re-traversed, reshuffled each epoch so the order is not identical, until the step cap or
        # the epoch count is reached.
        n_epochs = max(1, math.ceil(a.epochs))
        done = False
        for ep in range(n_epochs):
            if done:
                break
            ep_units = list(flat)
            if a.shuffle == "structure" and ep:
                # reshuffle BY STRUCTURE, keeping each structure's units together, so a preloaded
                # item is still used for all of them
                _by = {}
                for _t in ep_units:
                    _by.setdefault(_t[0], []).append(_t)
                _o = list(_by)
                _r = random.Random(a.seed + ep)
                _r.shuffle(_o)
                for _k in _o:
                    _r.shuffle(_by[_k])
                ep_units = [t for _k in _o for t in _by[_k]]
            # a fractional last epoch takes that fraction of the units
            if ep == n_epochs - 1 and a.epochs < n_epochs:
                frac = a.epochs - (n_epochs - 1)
                ep_units = ep_units[:max(1, int(len(ep_units) * frac))]
            print(f"  epoch {ep + 1}/{n_epochs}: {len(ep_units)} unit(s)")
            for ui, (pdb, lab_p, trunk_p) in enumerate(ep_units):
                for i, cum, met in run_unit(pdb, lab_p, trunk_p, item_cache):
                    cur_lr = lr * min(1.0, (step + 1) / max(1, a.warmup_steps))
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    (cum / a.accum).backward()
                    n_in_accum += 1
                    gn, outside = float("nan"), -1
                    if n_in_accum >= a.accum:
                        gn, outside, ok = _apply_step(step)
                        if not ok:
                            n_in_accum = 0
                            continue
                        step += 1
                        n_in_accum = 0
                        if a.save_every and step % a.save_every == 0 and write_traj:
                            cp = os.path.join(a.out, f"head_step{step}.pt")
                            torch.save({n: pp.detach().cpu() for n, pp in model.named_parameters()
                                        if n in set(trainable)}, cp)
                            print(f"  saved {cp}")
                    rows.append({"wall_s": round(time.time() - t0, 1), "step": step, "unit_i": ui,
                                 "pdb": pdb, "unit": os.path.basename(lab_p)[:-4], "pose": i,
                                 "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                                 "epoch": ep + 1,
                                 **{k: float(met[k]) for k in TERMS}})
                    if write_traj and len(rows) % 20 == 0:
                        _flush(traj_p, traj_cols, rows); rows = []
                    if steps_cap and step >= steps_cap:
                        done = True
                        break
                if done:
                    break
        # THE FINAL PARTIAL ACCUMULATION. Poses left in the accumulator when the data runs out used
        # to be dropped, so the last fraction of an epoch contributed gradients that were computed
        # and then thrown away.
        if n_in_accum:
            print(f"  flushing a partial accumulation of {n_in_accum}/{a.accum} pose(s)")
            _apply_step(step)
            step += 1
        if write_traj:
            _flush(traj_p, traj_cols, rows)
        return rows

    def _flush(path, cols, rows):
        if not rows:
            return
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            if new:
                wr.writeheader()
            wr.writerows(rows)

    # ---- LR PROBE: measure the learning rate instead of inheriting one tuned for a different job
    if a.lr_probe:
        lrs = [float(x) for x in a.lr_probe.split(",")]
        print(f"\nLR PROBE: {a.probe_steps} optimiser step(s) at each of {lrs}, all from the SAME "
              f"initial weights. Protenix's 0.0018 default is for finetuning the whole model; this "
              f"moves 4.3% of it, so the default is not transferable.")
        out = []
        for lr in lrs:
            for n, p in model.named_parameters():
                if n in set(trainable):
                    p.data.copy_(init_state[n])
            opt.load_state_dict(copy.deepcopy(init_opt))
            rows = train_loop(a.probe_steps, lr, f"probe{lr}", write_traj=False)
            if not rows:
                print(f"  lr {lr:g}: no rows"); continue
            k = max(1, len(rows) // 5)
            first = sum(r["loss"] for r in rows[:k]) / k
            last = sum(r["loss"] for r in rows[-k:]) / k
            fin = all(math.isfinite(r["loss"]) for r in rows)
            out.append((lr, first, last, last - first, fin))
            print(f"  lr {lr:<9g} loss first{k} {first:.6f} -> last{k} {last:.6f} "
                  f"(delta {last-first:+.6f}) finite={fin}")
        with open(os.path.join(a.out, "lr_probe.csv"), "w", newline="") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["lr", "loss_first", "loss_last", "delta", "all_finite"])
            wr.writerows(out)
        print(f"\nwrote {os.path.join(a.out, 'lr_probe.csv')}")
        print("A PROBE IS NOT A RESULT. It shows which rates are stable and which diverge over a few "
              "dozen steps; it cannot show which generalises. Pick from it, then judge the run on "
              "analysis/val_selection_metric.py, never on this loss.")
        return 0

    total_poses = int(len(flat) * 5 * a.epochs)
    steps_cap = max(1, total_poses // a.accum)
    print(f"\ntraining: lr {a.lr:g}, warmup {a.warmup_steps}, accum {a.accum}, "
          f"~{steps_cap} optimiser step(s) for {a.epochs} epoch(s)")
    print(f"trajectory -> {traj_p}")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    train_loop(steps_cap, a.lr, "train", write_traj=True)
    if item_build_s:
        import statistics as _st
        print(f"\ndataset items built: {len(item_build_s)}, median {_st.median(item_build_s):.2f}s, "
              f"total {sum(item_build_s)/60:.1f} min"
              + (f" -- --preload-items {a.preload_items} held them, so later epochs pay none of this"
                 if a.preload_items > 1 else
                 f" -- --preload-items is {a.preload_items}, so every epoch pays this again"))
    cp = os.path.join(a.out, "head_final.pt")
    import torch as _t
    _t.save({n: p.detach().cpu() for n, p in model.named_parameters() if n in set(trainable)}, cp)
    print(f"\nsaved {cp}")
    print("THE LOSS IS NOT THE RESULT. Score this checkpoint with "
          "analysis/val_selection_metric.py after rescoring the validation poses with it; the target "
          "is mean DockQ 0.5776 and frac-acceptable 0.7283 at 100 seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
