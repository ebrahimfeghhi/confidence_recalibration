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
(analysis/val_selection_metric.py): selecting by max pair_iptm gives mean DockQ 0.5779 and
frac-acceptable 0.7283, against an oracle of 0.7171 / 0.8696. So there is 0.1392 of mean DockQ
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
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Protenix ships TWO: configs_base.py has 0.0018 and their own "
                         "finetune_demo.sh passes 0.001. FT3 is a finetune, so 0.001 is the default "
                         "here. Neither was tuned for training 4.3%% of the model at batch 256, so "
                         "--lr-probe exists to measure it rather than inherit it.")
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="SHORT BECAUSE THE RUN IS SHORT. Protenix's finetune_demo.sh warms up over "
                         "2,000 steps of a 100,000-step run (2%%). At batch 256 three epochs of our "
                         "cache is 292 optimiser steps, so 2,000 would be 685%% of the run and the "
                         "learning rate would never reach its set value. 20 is 6.8%%.")
    ap.add_argument("--accum", type=int, default=256,
                    help="POSES PER OPTIMISER STEP -- the effective batch. The micro-batch is forced "
                         "to 1: token counts run 263-3,400 across our structures, so poses cannot be "
                         "stacked into one tensor without padding the head's pair representations. "
                         "256 at the user's direction. For reference, Protenix's own finetune runs "
                         "one structure per step with iters_to_accumulate=1 and scales by world_size; "
                         "256 is AF3's paper figure, for full-model training across many devices. At "
                         "256 an epoch of our 25,000 poses is 97 steps.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-every-epoch", action="store_true",
                    help="score the validation metric at the end of every epoch. THE METRIC IS THE "
                         "RESULT, not the loss: mean DockQ and frac-acceptable of the pose selected "
                         "by max pair_iptm, against the 0.5779 / 0.7283 baseline at 100 seeds.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also score every N optimiser steps. 0 disables. A full pass is 5,300 "
                         "validation units at 0.82 s/unit = ~73 min, MEASURED on GATE 1 2026-10-16 (73 units/min over 5,300). An earlier 0.187 s/unit here came from a small probe of small structures with the model already warm and was 4.4x optimistic, so at 97 steps/epoch this "
                         "is normally left off in favour of --eval-every-epoch.")
    ap.add_argument("--limit-units", type=int, default=0, help="debug: cap the number of units")
    ap.add_argument("--shuffle", default="structure",
                    choices=["structure", "none"],
                    help="`structure` shuffles the ORDER OF STRUCTURES and the units within each, "
                         "which is as much as can be shuffled while building each structure's "
                         "dataset item once. `none` is deterministic for debugging.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="500000-500009",
                    help="the seed range the manifest expects per structure, inclusive.")
    ap.add_argument("--manifest-selection",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv",
                    help="THE FROZEN POPULATION. The manifest is this times --seeds, not whatever "
                         "label files happen to exist; a label that was never written must be an "
                         "error, not a silently smaller training set.")
    ap.add_argument("--manifest-skip", default="9mx1",
                    help="comma-separated pdb ids the manifest must NOT expect, each a deliberate "
                         "documented exclusion. 9mx1 is 2,941 residues against Protenix's "
                         "2,560-token limit, so it cannot be folded at all.")
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
    ap.add_argument("--max-rss-gb", type=float, default=300.0,
                    help="HARD CAP ON THIS PROCESS'S OWN MEMORY, checked before every item is added "
                         "to the cache. The user's constraint is that our processes do not exceed "
                         "300 GB, and --preload-min-free-gb answers a different question: it looks "
                         "at MemAvailable, so on an idle box it would let one run take 700 GB. Past "
                         "this the cache stops growing and evicts; the run never fails for it.")
    ap.add_argument("--preload-min-free-gb", type=float, default=200.0,
                    help="FLOOR ON FREE RAM, checked before every item is added to the cache -- not "
                         "once at startup. The cache GROWS to ~253 GB over the first epoch (500 "
                         "items at a measured 331-886 MB, avg 518), so a startup check would pass "
                         "and the box could still be driven out of memory later, taking the fold "
                         "workers with it. When MemAvailable falls below this the cache stops "
                         "growing and evicts instead; it never fails the run.")
    ap.add_argument("--grad-check-every", type=int, default=25,
                    help="re-assert that no gradient has appeared outside the confidence head")
    ap.add_argument("--save-every", type=int, default=250, help="optimiser steps per checkpoint")
    ap.add_argument("--dist-timeout-min", type=float, default=240.0,
                    help="COLLECTIVE TIMEOUT IN MINUTES, and it must exceed the per-epoch "
                         "evaluation. torch's default is 30 min; the evaluation at --eval-budget "
                         "100 covers 5,300 validation units and takes ~73 min, measured. Rank 0 "
                         "evaluates while the other ranks wait at a barrier, so on the default the "
                         "barrier would time out and kill the run at the FIRST epoch boundary -- "
                         "after hours of correct training, with the failure looking like a network "
                         "fault rather than a timeout someone chose. 240 min leaves room for an "
                         "evaluation that runs long under contention.")
    ap.add_argument("--debug-save-rank-heads", action="store_true",
                    help="EVERY rank also writes head_final_rank<r>.pt. Off by default and not for "
                         "training runs: the replicas are supposed to be identical, so these files "
                         "are redundant in normal use. phase_k_train_ddp_selftest.py turns it on to "
                         "CHECK that they are identical after several steps, which is the only way "
                         "to catch replicas drifting apart -- a single-rank checkpoint cannot show "
                         "it, and drift is exactly what a missed all-reduce would cause.")
    ap.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"),
                    help="the collective backend for multi-GPU. DEFAULT gloo, NOT nccl, because "
                         "NCCL is broken on this box: a 20-line two-rank all_reduce on cards 1 and "
                         "3 fails with `CUDA error: an illegal memory access was encountered`, with "
                         "P2P enabled and disabled alike, on NCCL 2.26.2+cuda12.2 under a torch "
                         "built for 12.8 and driver 595.99.02. It is not this script -- the same "
                         "failure reproduces with no model loaded. gloo is a CPU backend, so the "
                         "gradients are staged through host memory: measured 60 ms per 79 MB "
                         "all-reduce, about 6 s per epoch over ~98 steps, against an epoch of "
                         "hours. Switch to nccl if the driver/NCCL pairing is ever fixed.")
    ap.add_argument("--nsample", type=int, default=5,
                    help="poses per unit, as the fold wrote them. Only asserted under multi-GPU, "
                         "where a unit with a different count would shift one rank's optimiser step "
                         "onto a different pose and deadlock the next all-reduce.")
    ap.add_argument("--lr-probe", default="",
                    help="comma-separated learning rates. Runs --probe-steps steps at each from the "
                         "SAME initial weights and reports the loss trend, then exits without "
                         "training. This is how --lr is chosen instead of inherited.")
    ap.add_argument("--probe-steps", type=int, default=30)
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="accept a label whose fingerprint differs ONLY in the `code` field, with "
                         "cif, bio, idx, trunk, loss and schema all matching. The data provenance "
                         "is what determines whether a label is CORRECT; the code hash additionally "
                         "says it is REPRODUCIBLE, and it covers whole files, so editing a docstring "
                         "in phase_j_build_cache.py invalidates all 5,000 labels. OFF by default: a "
                         "drift in any other field, or several fields, is still fatal, and every "
                         "accepted drift is printed with both hashes.")
    ap.add_argument("--wandb-project", default="",
                    help="wandb project. Empty disables wandb entirely. Every argparse value is "
                         "logged as config, the four loss terms + lr + grad_norm per step, and the "
                         "two validation metrics at each epoch boundary.")
    ap.add_argument("--wandb-run", default="")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--eval-script", default=f"{ROOT}/phases/phase_k_eval.py")
    ap.add_argument("--eval-device", default="",
                    help="CUDA index for the epoch-end evaluation. The subprocess would otherwise "
                         "INHERIT CUDA_VISIBLE_DEVICES from the trainer and land on the same card "
                         "while the trainer is still resident and holding its memory -- the "
                         "evaluation does not get the trainer's allocator back. Give it an idle "
                         "card. Empty means share, and then the memory is checked first and the "
                         "evaluation is refused rather than risking an OOM mid-run.")
    ap.add_argument("--eval-min-free-mib", type=int, default=20000,
                    help="when the evaluation must share the training card, how much VRAM has to be "
                         "free for it to run. The eval loads a second 464M-parameter model; "
                         "measured at ~16 GB, so 20 GB is the floor.")
    ap.add_argument("--eval-budget", type=int, default=100,
                    help="seeds per interface for the epoch-end metric. 100 is the full budget and "
                         "the one the 0.5779 / 0.7283 baseline is quoted at; a pass is ~73 min, measured.")
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

    # ---- MULTI-GPU: SYNCHRONOUS REPLICAS, NOT DistributedDataParallel ----
    #
    # DDP wraps a module's forward, and ours is RC.confidence(model, configs, ...) over a FROZEN
    # trunk. DDP's hooks would both fight that custom call and expect gradients for the 444M
    # parameters that deliberately have none. So every rank holds an identical replica -- same
    # checkpoint, hence identical initial parameters -- and only the HEAD gradients are summed
    # across ranks before each step.
    #
    # WHY A SUM AND NOT A MEAN. Each pose is already scaled by the global accumulation at
    # backward(), so summing the ranks' accumulated gradients reconstructs exactly the mean over all
    # --accum poses that one GPU would have computed. After the all-reduce every rank holds
    # bit-identical gradients, so the clip, the non-finite decision, the AdamW state and therefore
    # the parameters stay in lockstep with no further communication and no parameter broadcast.
    #
    # RANKS MUST TAKE THE SAME NUMBER OF STEPS or they deadlock on the all-reduce. That is enforced
    # two ways: shards are whole structures truncated to an equal count (shard_by_structure), and
    # each unit is asserted to hold the same number of poses as every other.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    dist = None
    if world > 1:
        import torch.distributed as dist
        import datetime
        dist.init_process_group(backend=a.dist_backend,
                                timeout=datetime.timedelta(minutes=a.dist_timeout_min))
        torch.cuda.set_device(local_rank)
        print(f"[rank {rank}/{world}] cuda:{local_rank} backend={a.dist_backend} "
              f"timeout={a.dist_timeout_min:.0f} min", flush=True)
    dev = torch.device(f"cuda:{local_rank}") if world > 1 else torch.device("cuda")

    # The effective batch is what the ranks actually sum to, which is not --accum when the world
    # size does not divide it. It is reported rather than silently assumed, and it is what the loss
    # is scaled by, so the gradient is a true mean over the poses that contributed.
    accum_local = max(1, a.accum // world)
    accum_eff = accum_local * world
    if world > 1:
        print(f"  accum: {a.accum} requested -> {accum_local}/rank x {world} ranks = {accum_eff} "
              f"poses per optimiser step")
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

    wb = None
    # RANK 0 ONLY. Three ranks initialising the same run name would either collide or create three
    # runs logging a third of the steps each; the gradients are identical after the all-reduce, so
    # rank 0's view is the run's view.
    if a.wandb_project and is_main:
        import wandb
        wb = wandb.init(project=a.wandb_project, name=a.wandb_run or None, mode=a.wandb_mode,
                        config={**vars(a), "n_head_params": n_head, "n_all_params": n_all,
                                "head_frac": n_head / n_all, "n_head_tensors": len(trainable),
                                "loss_weights": {k.replace("_loss", ""): float(v)
                                                 for k, v in w.items()},
                                "baseline_mean_dockq": 0.5779, "baseline_frac_acc": 0.7283})
        print(f"wandb: {wb.url if hasattr(wb, 'url') else a.wandb_project} (mode={a.wandb_mode})")

    # ---- THE MANIFEST COMES FROM THE FROZEN SELECTION, NOT FROM WHATEVER FILES EXIST ----
    # GLOBBING THE LABEL DIRECTORY CANNOT ESTABLISH COMPLETENESS. It did: the manifest was built by
    # globbing *.npz and then checking that each DISCOVERED label had a trunk, so a label that was
    # never written simply did not appear and the training set shrank with no error. Review round 8.
    # The population is now the selection csv times the seed range, minus named exclusions, and a
    # missing label is fatal.
    roots = [r.strip() for r in a.fold.split(",") if r.strip()]
    for r in roots:
        if not os.path.isdir(r):
            raise SystemExit(f"STOP: --fold root {r} does not exist")
    label_dirs = [x.strip() for x in a.labels.split(",") if x.strip()]
    lo_s, hi_s = (int(x) for x in a.seeds.split("-"))
    skip = {x.strip().lower() for x in a.manifest_skip.split(",") if x.strip()}
    with open(a.manifest_selection) as fh:
        sel_pdbs = [r["pdb"].strip().lower() for r in csv.DictReader(fh)]
    want = [(p, sd) for p in sel_pdbs if p not in skip for sd in range(lo_s, hi_s + 1)]
    print(f"manifest: {len(sel_pdbs)} selected structure(s) minus {sorted(skip) or 'none'} "
          f"x seeds {lo_s}-{hi_s} = {len(want)} intended unit(s), from {a.manifest_selection}")

    manifest, no_label, no_trunk = [], [], []
    for pdb, sd in want:
        arm_seed = f"free_{sd}"
        lp = next((os.path.join(d, pdb, f"{arm_seed}.npz") for d in label_dirs
                   if os.path.exists(os.path.join(d, pdb, f"{arm_seed}.npz"))), None)
        if lp is None:
            no_label.append(f"{pdb}/{arm_seed}"); continue
        cands = [os.path.join(r, "trunks", pdb, f"{arm_seed}.npz") for r in roots
                 if os.path.exists(os.path.join(r, "trunks", pdb, f"{arm_seed}.npz"))]
        if not cands:
            no_trunk.append(f"{pdb}/{arm_seed}"); continue
        manifest.append((pdb, lp, cands))        # ALL candidates; provenance picks below
    if no_label or no_trunk:
        raise SystemExit(
            f"STOP: the training set is INCOMPLETE.\n"
            f"  {len(no_label)} intended unit(s) have no label, e.g. {no_label[:5]}\n"
            f"  {len(no_trunk)} have no trunk under {roots}, e.g. {no_trunk[:5]}\n"
            f"  Build them, or name a deliberate exclusion in --manifest-skip and record it in "
            f"docs/build_cache_exclusions.md.")

    bypdb = {}
    for pdb, lp, cands in manifest:
        bypdb.setdefault(pdb, []).append((lp, cands))
    order = sorted(bypdb)
    rng = random.Random(a.seed)
    if a.shuffle == "structure":
        rng.shuffle(order)
        for pp in order:
            rng.shuffle(bypdb[pp])
    flat = [(pp, lp, cands) for pp in order for lp, cands in bypdb[pp]]
    if a.limit_units:
        flat = flat[:a.limit_units]
    print(f"  resolved {len(flat)} unit(s) over {len(set(x[0] for x in flat))} structure(s) "
          f"from {len(roots)} fold root(s), shuffle={a.shuffle}, seed={a.seed}")

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    train_ds, _ = get_datasets(cfg, err)
    byp = BC.merged_lookup(train_ds)
    perm = SymmetricPermutation(cfg, error_dir=err)
    # THE SAME FINGERPRINT CONTEXT build_cache USES, imported rather than reimplemented. A
    # reimplementation that drifted would reject every label or accept every label; round 8 found
    # the latter, because `fingerprint` was required to be PRESENT and never compared.
    code_fp, loss_fp = BC.fingerprint_context(cfg)
    print(f"fingerprint context: code={code_fp} loss={loss_fp}")
    # EVERY STRUCTURE IN THE MANIFEST MUST HAVE A DATASET ENTRY, asserted here rather than
    # discovered per unit and skipped.
    nodata = sorted({pdb for pdb, _lp, _cands in flat if pdb not in byp})
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
                ["grad_norm", "outside_head", "epoch", "step_taken", "nonfinite_grad"]
    # the per-structure item build is the cost --preload-items exists to amortise, so it is measured
    # rather than assumed; reported at the end against the number of structures actually visited.
    item_build_s = []
    nonfinite = {"n": 0}
    ram_floor = {"hit": False, "n": 0}
    code_drift = {"n": 0, "warned": False}

    def resolve_trunk(pdb, lab_p, cands, want_fp_cache={}):
        """Pick the trunk whose PROVENANCE matches this label, not the first filename that exists.

        THE FIRST MATCHING FILENAME IS NOT THE RIGHT TRUNK. The multi-root lookup took it, so a
        label built from one root's trunk could be paired with another root's trunk of the same name
        -- and since the label is a PER-POSE permutation of its own trunk's poses, that pairs a pose
        with another run's permuted label and trains on it. Review round 8 reproduced exactly that
        with a fold_regen label and a fold trunk.

        The stored fingerprint names its trunk by content hash, so the trunk is chosen BY it. Two
        candidates matching is ambiguous and refused rather than resolved arbitrarily.
        """
        stored = str(np.asarray(np.load(lab_p, allow_pickle=True)["fingerprint"]))
        ok = []
        for tp in cands:
            key = (pdb, tp)
            if key not in want_fp_cache:
                want_fp_cache[key] = BC.unit_fingerprint(a, pdb, tp, code_fp, loss_fp)
            if want_fp_cache[key] == stored:
                ok.append(tp)
        if len(ok) == 1:
            return ok[0]
        if not ok:
            # WHICH FIELD DIFFERS IS THE ACTIONABLE PART. "stale" with no detail cannot be acted on,
            # and the fields are not equally serious: cif, bio, idx and trunk say the label was
            # built from DIFFERENT DATA, while code says only that the builder's source has changed.
            def fields(fp):
                out = {}
                for part in fp.split("|"):
                    if "=" in part:
                        k, v = part.split("=", 1); out[k] = v
                return out
            want_f = fields(want_fp_cache[(pdb, cands[0])])
            got_f = fields(stored)
            diff = sorted(k for k in set(want_f) | set(got_f) if want_f.get(k) != got_f.get(k))
            if diff == ["code"] and a.allow_code_drift:
                if not code_drift["warned"]:
                    print(f"  ACCEPTING CODE DRIFT (--allow-code-drift): labels were built by "
                          f"code={got_f.get('code')}, this run is code={want_f.get('code')}. Every "
                          f"data field matches -- cif, bio, idx, trunk, loss, schema -- so the "
                          f"labels are correct but not reproducible from this checkout.")
                    code_drift["warned"] = True
                code_drift["n"] += 1
                return cands[0]
            raise SystemExit(
                f"STOP: {lab_p} does not match its inputs. No trunk among {cands} reproduces its "
                f"fingerprint.\n  field(s) differing: {diff}\n"
                + "".join(f"    {k}: label {got_f.get(k)} vs now {want_f.get(k)}\n" for k in diff)
                + (f"  Only `code` differs, so the DATA provenance is intact; pass "
                   f"--allow-code-drift to accept that explicitly.\n" if diff == ["code"] else "")
                + f"  Otherwise rebuild with phase_j_build_cache.py --only {pdb}.")
        raise SystemExit(f"STOP: {len(ok)} trunks match {lab_p}'s fingerprint: {ok}. Ambiguous "
                         f"provenance is refused rather than resolved arbitrarily.")

    def run_unit(pdb, lab_p, trunk_p, item_cache):
        """Forward + loss for every pose of one unit. Yields (pose_index, loss_tensor, metrics).

        Both paths are resolved by the manifest and asserted to exist before the run starts, so
        there is nothing here to skip.
        """
        arm_seed = os.path.basename(lab_p)[:-4]
        if pdb not in item_cache:
            # RE-CHECKED EVERY TIME, from MemAvailable rather than a remembered number
            try:
                _free_gb = next(int(l.split()[1]) for l in open("/proc/meminfo")
                                if l.startswith("MemAvailable:")) / 1048576.0
            except Exception:
                _free_gb = float("inf")
            try:
                _rss_gb = int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * \
                          os.sysconf("SC_PAGE_SIZE") / 2**30
            except Exception:
                _rss_gb = 0.0
            if (_rss_gb > a.max_rss_gb or _free_gb < a.preload_min_free_gb) and len(item_cache) >= 1:
                if not ram_floor["hit"]:
                    print(f"  RAM CAP: this process holds {_rss_gb:.0f} GB (cap "
                          f"{a.max_rss_gb:.0f}), {_free_gb:.0f} GB free (floor "
                          f"{a.preload_min_free_gb:.0f}). The item cache stops "
                          f"growing at {len(item_cache)} structure(s) and evicts from here on; "
                          f"structures will be rebuilt rather than held. Nothing fails.")
                    ram_floor["hit"] = True
                ram_floor["n"] += 1
                while len(item_cache) >= max(1, len(item_cache)):
                    item_cache.pop(next(iter(item_cache)))
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
        # THE FINGERPRINT IS COMPARED, NOT MERELY PRESENT. resolve_trunk already proved one trunk
        # reproduces it; this re-asserts against the trunk actually being used, so the two cannot
        # come apart if resolve_trunk is ever bypassed.
        _want = BC.unit_fingerprint(a, pdb, trunk_p, code_fp, loss_fp)
        _got = str(np.asarray(lab["fingerprint"]))
        if _got != _want:
            _dw = dict(x.split("=", 1) for x in _want.split("|") if "=" in x)
            _dg = dict(x.split("=", 1) for x in _got.split("|") if "=" in x)
            _df = sorted(k for k in set(_dw) | set(_dg) if _dw.get(k) != _dg.get(k))
            if not (_df == ["code"] and a.allow_code_drift):
                raise SystemExit(f"STOP: {lab_p} fingerprint differs in {_df}. Rebuild it.")
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
        if world > 1 and n_pose != a.nsample:
            raise SystemExit(f"STOP: {pdb}/{arm_seed} holds {n_pose} pose(s), not --nsample "
                             f"{a.nsample}. Across ranks that shifts the optimiser step onto a "
                             f"different pose and the next all-reduce deadlocks.")
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
        if world > 1:
            # BEFORE THE CLIP, NOT AFTER. Clipping each rank's partial gradient to norm 1 and then
            # averaging is not the same function as clipping the summed gradient, so the multi-GPU
            # run would not reproduce the single-GPU one. A rank with no gradient for a parameter
            # still has to contribute a tensor, or the collective's shapes do not line up.
            for pp in params:
                if pp.grad is None:
                    pp.grad = torch.zeros_like(pp)
            # ONE COLLECTIVE, NOT 248. The head's gradients are flattened into a single buffer, so
            # the per-call latency is paid once per step rather than once per tensor. gloo is a CPU
            # backend, so that buffer is staged through host memory and copied back.
            flat = torch.cat([pp.grad.reshape(-1) for pp in params])
            if a.dist_backend == "gloo":
                host = flat.cpu()
                dist.all_reduce(host, op=dist.ReduceOp.SUM)
                flat = host.to(flat.device)
            else:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off = 0
            for pp in params:
                k = pp.grad.numel()
                pp.grad.copy_(flat[off:off + k].view_as(pp.grad))
                off += k
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

    def eval_checkpoint(step, epoch):
        """Save the head, score it on the validation metric, log it. Returns the metric dict.

        RUN AS A SUBPROCESS, not imported. phase_k_eval carries its own guards -- the pool-coverage
        check, the pose-to-sample join by coordinate identity, and the self-check that the SHIPPED
        head reproduces results.csv's own ipTM -- and running it as the script keeps all of them,
        including its refusal to report a number when a join is unverified. It also keeps a second
        model off this process's GPU memory.
        """
        import json
        import subprocess
        cp = os.path.join(a.out, f"head_ep{epoch}_step{step}.pt")
        if not is_main:
            return          # the caller already holds the other ranks at a barrier
        torch.save({n: pp.detach().cpu() for n, pp in model.named_parameters()
                    if n in set(trainable)}, cp)
        mj = os.path.join(a.out, f"metric_ep{epoch}.json")
        cmd = [sys.executable, "-I", a.eval_script, "--ckpt", cp, "--budget", str(a.eval_budget),
               "--out", mj]
        env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
        if a.eval_device:
            env["CUDA_VISIBLE_DEVICES"] = str(a.eval_device)
            print(f"  evaluating epoch {epoch} on GPU {a.eval_device} (the trainer keeps its card)")
        else:
            # SHARING MEANS CHECKING FIRST. The trainer is still resident and does not hand its
            # allocator back, so a second 464M-parameter model has to fit alongside it.
            free_mib = None
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
                idx = int(str(os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0])
                free_mib = int(q.stdout.strip().splitlines()[idx])
            except Exception:
                pass
            if free_mib is not None and free_mib < a.eval_min_free_mib:
                raise SystemExit(
                    f"STOP: the epoch-{epoch} evaluation would share the training card with only "
                    f"{free_mib} MiB free, under --eval-min-free-mib {a.eval_min_free_mib}. It "
                    f"loads a second 464M-parameter model (~16 GB) while the trainer stays "
                    f"resident. Pass --eval-device <idle gpu>, or raise the floor deliberately.")
            print(f"  evaluating epoch {epoch} on the TRAINING card, {free_mib} MiB free")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            raise SystemExit("STOP: the validation evaluation failed, so this checkpoint cannot be "
                             "scored and a training curve without it is not interpretable:\n  "
                             + "\n  ".join(tail))
        m = json.load(open(mj))
        print(f"  epoch {epoch}: mean_dockq {m['mean_dockq']:.4f} (baseline 0.5779)  "
              f"frac_acc {m['frac_acc']:.4f} (baseline 0.7283)  headroom {m['headroom']:.4f}")
        if wb is not None:
            wb.log({"val/mean_dockq": m["mean_dockq"], "val/frac_acc": m["frac_acc"],
                    "val/oracle_mean_dockq": m["oracle_mean_dockq"],
                    "val/headroom": m["headroom"], "epoch": epoch, "step": step})
        return m

    def shard_by_structure(units):
        """Split `units` across ranks by WHOLE STRUCTURE, equal in count on every rank.

        By structure, not by unit, because --shuffle=structure keeps a structure's units
        consecutive so one built dataset item serves all of them; sharding by unit would make every
        rank build every structure and multiply the item cache by the world size for nothing.

        Truncated to the smallest rank's share so all ranks run the same number of steps. 500
        structures over 3 ranks is 167/167/166, so one structure is dropped from an epoch; the
        per-epoch reshuffle means it is a different one each time rather than the same data being
        excluded from training.
        """
        if world == 1:
            return units, []
        by, seen = {}, []
        for t in units:
            if t[0] not in by:
                by[t[0]] = []
                seen.append(t[0])
            by[t[0]].append(t)
        keep = min(len(seen[r::world]) for r in range(world))
        mine = seen[rank::world]
        dropped = mine[keep:]
        return [t for k in mine[:keep] for t in by[k]], dropped

    def train_loop(steps_cap, lr, tag, write_traj):
        for g in opt.param_groups:
            g["lr"] = lr
        rows, t0, step, n_in_accum, item_cache = [], time.time(), 0, 0, {}
        opt.zero_grad(set_to_none=True)

        def _apply_partial():
            """Step on FEWER than accum_local poses, scaled by the poses that actually contributed.

            Every pose divided its loss by accum_eff, so a partial accumulation holds sum/accum_eff:
            a gradient scaled DOWN by n_contributing/accum_eff, not a mean over the poses present.
            The old code stepped on that directly, making the last step of an epoch a fraction of
            the size of every other step -- 1/256th of one, if a single pose was left. Multiplying by
            accum_eff/n_contributing restores the mean.

            The count is reduced across ranks rather than assumed to be n_local * world: the shards
            are equal length so it should be, but a step whose size depends on that assumption
            should not take it on trust.

            It also respects _apply_step's success flag. The old code incremented `step` whether or
            not a step was taken, so a non-finite gradient here advanced the step counter anyway.
            """
            nonlocal step, n_in_accum
            n_tot = n_in_accum * world
            if world > 1:
                t = torch.tensor([float(n_in_accum)], dtype=torch.float32)
                if a.dist_backend == "gloo":
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                else:
                    t = t.to(dev)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                n_tot = int(round(float(t.item())))
            n_in_accum = 0
            if n_tot <= 0:
                return
            scale = accum_eff / n_tot
            tset = set(trainable)
            for _n, pp in model.named_parameters():
                if _n in tset and pp.grad is not None:
                    pp.grad.mul_(scale)
            print(f"  flushing a partial accumulation of {n_tot}/{accum_eff} pose(s), gradient "
                  f"scaled by {scale:.4f} so the step is the same size as every other")
            _gn, _outside, ok = _apply_step(step)
            if ok:
                step += 1
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
            ep_units, _dropped = shard_by_structure(ep_units)
            if world > 1:
                if _dropped:
                    print(f"  [rank {rank}] epoch {ep + 1}: dropped {_dropped} to keep the ranks "
                          f"equal; the reshuffle means a different structure each epoch")
                # THE RANKS MUST AGREE ON THE UNIT COUNT. If they do not, they take a different
                # number of optimiser steps and the first mismatched all-reduce hangs forever
                # rather than failing, so it is checked here where it can still be reported.
                _t = torch.tensor([len(ep_units)], device=dev)
                _mx, _mn = _t.clone(), _t.clone()
                dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
                dist.all_reduce(_mn, op=dist.ReduceOp.MIN)
                if int(_mx) != int(_mn):
                    raise SystemExit(f"STOP: rank unit counts disagree ({int(_mn)}..{int(_mx)}). "
                                     f"The ranks would take different numbers of steps and deadlock "
                                     f"on the first all-reduce.")
            # a fractional last epoch takes that fraction of the units
            if ep == n_epochs - 1 and a.epochs < n_epochs:
                frac = a.epochs - (n_epochs - 1)
                ep_units = ep_units[:max(1, int(len(ep_units) * frac))]
            print(f"  epoch {ep + 1}/{n_epochs}: {len(ep_units)} unit(s)")
            ep_start_step = step
            for ui, (pdb, lab_p, cands) in enumerate(ep_units):
                trunk_p = resolve_trunk(pdb, lab_p, cands)
                for i, cum, met in run_unit(pdb, lab_p, trunk_p, item_cache):
                    cur_lr = lr * min(1.0, (step + 1) / max(1, a.warmup_steps))
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    (cum / accum_eff).backward()
                    n_in_accum += 1
                    gn, outside, took, bad_grad = float("nan"), -1, 0, 0
                    if n_in_accum >= accum_local:
                        gn, outside, ok = _apply_step(step)
                        took = 1
                        if not ok:
                            bad_grad = 1
                            n_in_accum = 0
                            rows.append({"wall_s": round(time.time() - t0, 1), "step": step,
                                         "unit_i": ui, "pdb": pdb,
                                         "unit": os.path.basename(lab_p)[:-4], "pose": i,
                                         "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                                         "epoch": ep + 1, "step_taken": 0, "nonfinite_grad": 1,
                                         **{k: float(met[k]) for k in TERMS}})
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
                                 "epoch": ep + 1, "step_taken": took,
                                 "nonfinite_grad": bad_grad,
                                 **{k: float(met[k]) for k in TERMS}})
                    if wb is not None:
                        wb.log({f"train/{k.replace('_loss','')}": float(met[k]) for k in TERMS}
                               | {"train/lr": cur_lr, "train/grad_norm": gn, "step": step,
                                  "epoch": ep + 1})
                    if write_traj and len(rows) % 20 == 0:
                        _flush(traj_p, traj_cols, rows); rows = []
                    if steps_cap and step >= steps_cap:
                        done = True
                        break
                if done:
                    break
            # THE PARTIAL ACCUMULATION IS FLUSHED HERE, BEFORE THE EVALUATION, SO THE HEAD THAT
            # IS SCORED IS THE HEAD THAT IS SAVED. It used to be flushed after the epoch loop, which
            # is after the LAST evaluation: at production settings the run evaluated step 292, then
            # took step 293, then wrote head_final.pt -- so the shipped head was one optimiser step
            # away from the only head whose validation metric we had, and the number reported for
            # the run did not belong to the checkpoint the run produced.
            if n_in_accum:
                _apply_partial()
            if a.eval_every_epoch and step > ep_start_step:
                # RANK 0 EVALUATES, THE OTHERS WAIT. The evaluator is a separate process holding
                # its own model, so letting every rank launch one would put three on the cards at
                # once. The barriers keep the ranks together across the ~16 min it takes; without
                # the second one, rank 0 would re-enter training a whole evaluation behind.
                if world > 1:
                    dist.barrier()
                if is_main:
                    _flush(traj_p, traj_cols, rows); rows = []
                    eval_checkpoint(step, ep + 1)
                if world > 1:
                    dist.barrier()
        # NOTHING MAY BE LEFT HERE. Every epoch flushes its own remainder above, before evaluating,
        # so a non-zero accumulator at this point would mean a path out of the loop that skipped the
        # flush -- and the saved head would again differ from the evaluated one. Flushed and reported
        # rather than dropped, because dropping poses silently is what the flush was added to stop.
        if n_in_accum:
            print(f"  WARNING: {n_in_accum} pose(s) still accumulated after the epoch loop; the "
                  f"evaluated head is NOT the saved head for this run")
            _apply_partial()
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
    # PER RANK. Each rank sees total_poses/world poses and steps every accum_local of them, so the
    # step count is total_poses/(world*accum_local) = total_poses/accum_eff either way, and a
    # multi-GPU run does the same number of optimiser steps as a single-GPU one -- in a third of the
    # wall clock, rather than a third of the training.
    steps_cap = max(1, total_poses // accum_eff)
    print(f"\ntraining: lr {a.lr:g}, warmup {a.warmup_steps}, accum {a.accum}, "
          f"~{steps_cap} optimiser step(s) for {a.epochs} epoch(s)")
    print(f"trajectory -> {traj_p}")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    train_loop(steps_cap, a.lr, "train", write_traj=is_main)
    if ram_floor["n"]:
        print(f"\nRAM FLOOR was hit on {ram_floor['n']} structure(s): the item cache was capped "
              f"below --preload-items to stay above {a.preload_min_free_gb:.0f} GB free, so those "
              f"structures were rebuilt rather than held.")
    if code_drift["n"]:
        print(f"\nCODE DRIFT ACCEPTED on {code_drift['n']} unit(s) via --allow-code-drift. The "
              f"labels' data provenance matched in every field; only the builder's source hash "
              f"differed. This run is NOT reproducible from a clean checkout of that code.")
    if item_build_s:
        import statistics as _st
        print(f"\ndataset items built: {len(item_build_s)}, median {_st.median(item_build_s):.2f}s, "
              f"total {sum(item_build_s)/60:.1f} min"
              + (f" -- --preload-items {a.preload_items} held them, so later epochs pay none of this"
                 if a.preload_items > 1 else
                 f" -- --preload-items is {a.preload_items}, so every epoch pays this again"))
    # RANK 0 WRITES THE CHECKPOINT. Every rank holds the same parameters after the all-reduce, so
    # the content would be identical, but two processes writing one path concurrently can interleave
    # and leave a truncated file -- the smoke test showed both ranks reporting the same save.
    cp = os.path.join(a.out, "head_final.pt")
    if is_main or a.debug_save_rank_heads:
        import torch as _t
        _sd = {n: p.detach().cpu() for n, p in model.named_parameters() if n in set(trainable)}
        if is_main:
            _t.save(_sd, cp)
            print(f"\nsaved {cp}")
        if a.debug_save_rank_heads:
            _rp = os.path.join(a.out, f"head_final_rank{rank}.pt")
            _t.save(_sd, _rp)
            print(f"saved {_rp}")
    print("THE LOSS IS NOT THE RESULT. Score this checkpoint with "
          "analysis/val_selection_metric.py after rescoring the validation poses with it; the target "
          "is mean DockQ 0.5779 and frac-acceptable 0.7283 at 100 seeds.")
    # Tear the group down so the ranks exit together rather than on NCCL's timeout.
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
