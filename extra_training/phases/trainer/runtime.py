"""`trainer/runtime.py` -- the glue between the extracted regions.

EXTRACTED from `phases/phase_k_train.py`: 15 gaps, 988
lines (source `8534cd30`).

WHAT THIS IS. The named regions -- bootstrap, dataset, unit, loop -- are the parts of
main() that LOOK like functions. Between them sits the state they all read: TERMS, the
seed columns, the prefetch and trunk caches, the fingerprint cache, the timers, the
trajectory paths. 99 of 115 context reads had no writer because this was never extracted.

EVERY GAP IS KEPT and the lean pass removes culled blocks inside them. The first version
dropped a whole gap when its FIRST statement guarded a culled flag, which threw away the
trajectory and selection setup sitting after `if a.l2_sp:` and `if a.save_init:`.
"""
import collections
import csv
import glob
import hashlib
import json
import math
import os
import queue
import random
import re
import threading
import time

import numpy as np
import torch


def build_runtime(rt, a):
    """Set everything the extracted stages read off the context."""
    if a.skip_dead_ops:
        import dead_ops as DOPS
        # THE KERNEL NAMES COME FROM configs, THE SAME PLACE run_confidence_head GETS THEM.
        # Both default to "cuequivariance", and with hidden_scale_up the tri-mul cueq gate is
        # open -- so probing the "torch" reference implementation would certify a code path this
        # run never executes. Found in review after the first version hardcoded "torch".
        _dops = DOPS.install(rt.model, autocast_dtype=torch.bfloat16,
                             triangle_multiplicative=rt.configs.triangle_multiplicative,
                             triangle_attention=rt.configs.triangle_attention)
    else:
        DOPS = None

    # ---- THE FROZEN-PREFIX CACHE ---------------------------------------------------------------
    # INSTALLED HERE AND NOT AT THE pose_inblock SITE, which is ~1,100 lines above. Its central
    # precondition is that every prefix tensor reads requires_grad=False, and requires_grad is not
    # set until the line above -- `trainable` is not final until --train-only, LoRA and
    # --lean-single have all had their say. Installing it earlier would check the flag's intent
    # instead of the model's state, which is the distinction the whole guard exists to make.
    _pfc = None
    if a.prefix_cache_gb > 0:
        import prefix_cache as PFC
        _pfc = PFC.install(rt.model, a.prefix_cache_gb, lean_single=a.lean_single, device=rt.dev,
                           crop_margin=0.0, crop_tokens='',
                           crop_all_poses=False)
    else:
        PFC = None

    _head_names = {n for n, _p in rt.model.named_parameters()
                   if any(n == pr or n.startswith(pr + ".") for pr in rt.prefixes)}



    # ---- --save-init: THE HEAD BEFORE ANY STEP IS TAKEN ----------------------------------------
    # PLACED HERE, which is after every pathway has been installed (lean_readouts, --lean-single,
    # pose_inblock, pose_direction, pose_perblock all happen around :1776-1835) and after
    # _head_names is fixed, but far above the training loop. So the file holds exactly the
    # architecture the run was configured for, with the injections still at their zero init.
    if a.save_init:
        if rt.world > 1:
            raise SystemExit(
                f"STOP: --save-init with WORLD_SIZE={rt.world}. It writes the head and exits, and "
                f"exiting one rank of {rt.world} leaves the others parked on the next collective "
                f"until the elastic agent times out. Run it on one card.")
        _si_d = os.path.dirname(os.path.abspath(a.save_init))
        if _si_d:
            os.makedirs(_si_d, exist_ok=True)
        _si_t = _head_ckpt_tensors()
        torch.save(_si_t, a.save_init)
        # THE ZERO-INIT CLAIM, ASSERTED AT THE MOMENT IT IS SAVED rather than trusted. If a future
        # change initialises either injection non-zero, step-zero parity against the shipped head
        # stops being the right test and this says so instead of the parity run failing later with
        # a number nobody can attribute.
        _si_inj = {k: v for k, v in _si_t.items() if "pose_inblock" in k or "pose_perblock" in k
                   or "frame_dir" in k}
        _si_nz = sorted(k for k, v in _si_inj.items() if float(v.abs().max()) != 0.0)
        print(f"--save-init: wrote {len(_si_t)} head tensor(s) to {a.save_init} before any step. "
              f"{len(_si_inj)} injected tensor(s) present; "
              + ("ALL EXACTLY ZERO, so this head is bit-identical to the un-injected one and its "
                 "PAE/ipTM must reproduce the shipped head."
                 if not _si_nz else
                 f"NOT ZERO: {_si_nz}. Step-zero parity against the shipped head is NOT the right "
                 f"check for this configuration."))
        sys.stdout.flush()
        raise SystemExit(0)
    # -------------------------------------------------------------------------------------------

    # NO EXPERIMENT TRACKER. Runs are recorded by the files this script already writes --
    # trajectory.csv, metric_ep<N>.json, args.json -- and rendered by run_logging/render_report.py.
    # wandb was removed on 2026-10-16 at the user's direction: it needed an account, it put the
    # record somewhere the repo cannot read, and instrumenting the trainer for a viewer means a
    # restart whenever the viewer changes. The records outlive whatever reads them.

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
        _sel_rows = list(csv.DictReader(fh))
    sel_pdbs = [r["pdb"].strip().lower() for r in _sel_rows]
    # THE TOKEN COUNT, KEPT THIS TIME. It was parsed and discarded, which left --epoch-order no
    # way to sort by size -- so "manifest" order silently meant whatever order `flat` happened to
    # be in, which is sorted by pdb. Absent or non-numeric `tokens` is simply missing here and
    # sorts first, named by the builder rather than guessed at.
    _sel_tokens = {r["pdb"].strip().lower(): int(r["tokens"])
                   for r in _sel_rows if str(r.get("tokens", "")).strip().isdigit()}
    if a.manifest_nshards and a.manifest_nshards > 1:
        if not (0 <= a.manifest_shard < a.manifest_nshards):
            raise SystemExit(f"STOP: --manifest-shard {a.manifest_shard} is not in "
                             f"[0, {a.manifest_nshards}).")
    # A PER-STRUCTURE SEED SET, because the population is no longer uniform in depth. 437
    # structures carry seeds 500000-500014 and 416 carry 500000 alone -- their other trunks were
    # reclaimed on 2026-11-04 -- so a cross product with ONE range is wrong for whichever group it
    # is not describing, and the incompleteness guard below then refuses the whole run for being
    # the shape it is meant to be. An optional `seeds` column in --manifest-selection declares each
    # structure's own depth; a row without one falls back to --seeds, so every existing arm's
    # selection csv keeps its exact present meaning.
    #
    # DECLARED, NOT DISCOVERED, and that is the whole point of this file. Globbing the trunks would
    # make a lost unit look like a smaller training set instead of an error, which is the defect
    # review round 8 removed. The seeds column is a contract: if it names a seed with no label or
    # no trunk, the run stops.


    _seed_col = {}
    for _r in _sel_rows:
        _p = _r["pdb"].strip().lower()
        _txt = str(_r.get("seeds", "") or "").strip()
        if _txt:
            _seed_col[_p] = _parse_seed_field(_txt, f"{a.manifest_selection}:{_p}")
    want = [(p, sd) for p in sel_pdbs if p not in skip
            for sd in sorted(_seed_col.get(p) or range(lo_s, hi_s + 1))]
    if _seed_col:
        _depths = collections.Counter(len(_seed_col.get(p) or range(lo_s, hi_s + 1))
                                      for p in sel_pdbs if p not in skip)
        print(f"manifest: per-structure seed depths from {a.manifest_selection}: "
              f"{dict(sorted(_depths.items()))} "
              f"({len(_seed_col)} of {len(sel_pdbs)} row(s) declare `seeds`)")
    _rank_check_coverage(want)
    print(f"manifest: {len(sel_pdbs)} selected structure(s) minus {sorted(skip) or 'none'} "
          f"x seeds {lo_s}-{hi_s} (fallback for rows with no `seeds` column) = {len(want)} "
          f"intended unit(s), from {a.manifest_selection}")

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
    # THE POPULATION, AS ONE HASH. The resume guard compares this rather than the arguments that
    # produced it: --manifest-selection, --manifest-skip, --seeds and --limit-units can all change
    # which units exist, and a guard that checked the flags would still miss a rewritten selection
    # csv. Review 2026-10-16 reproduced a resume accepted after the exclusions changed.
    # --eval-every IS PARSED AND NEVER READ. Found in review: the flag has no effect anywhere in
    # this file, so launch_lr_probe_full.sh asked for validation every 1,452 steps and would never
    # have produced a single pass. Refused rather than implemented: a second eval trigger has to
    # reproduce the barrier discipline of the epoch path exactly -- rank 0 evaluates while the
    # others wait -- and a mistake there deadlocks rather than fails. The two working triggers do
    # cover the need.
    if a.eval_every:
        raise SystemExit(
            f"STOP: --eval-every {a.eval_every} has no effect -- it is parsed and never read, so "
            f"it would promise a validation pass that never happens. Use --eval-every-epoch for "
            f"the validation set every epoch, or --train-eval-pool with --train-eval-every N "
            f"(optionally --train-eval-fold / --train-eval-seed-base / --train-eval-budget) to "
            f"score any pool every N epochs. Both are exercised by the current arms.")
    _manifest_digest = hashlib.sha256(
        "\n".join(f"{p}/{os.path.basename(lp)}" for p, lp, _c in sorted(flat)).encode()
    ).hexdigest()[:16]
    print(f"  manifest digest {_manifest_digest} over {len(flat)} unit(s)")
    _pae_mask_coverage(flat)
    # THE PREFLIGHT IS REQUIRED, AND IT MUST BE THIS POPULATION'S. A marker from a different
    # manifest proves nothing about the structures this run will train on.
    # UNCONDITIONAL NOW, NOT GATED ON --minimal-items. The dataset-item path is gone, so the
    # minimal path is the ONLY path and check_alignment can no longer run per structure. Leaving
    # this behind the flag meant omitting --minimal-items silently dropped the alignment check
    # altogether instead of falling back to the per-structure one that used to exist.
    if True:
        if not a.alignment_preflight:
            raise SystemExit(
                "STOP: --alignment-preflight is REQUIRED. It is not conditional on "
                "--minimal-items: the full dataset-item path is gone, so the minimal path is the "
                "only path and check_alignment can no longer run per structure. Omitting "
                "--minimal-items does NOT avoid this -- the message used to say it did, and that "
                "cost a launch attempt. check_alignment is what proves the trunk's features agree "
                "atom for atom with the dataset's, the one failure that would pair a pose with "
                "another molecule's label. Run:\n"
                "  phases/phase_k_alignment_preflight.py --out <marker>\n"
                "and pass that marker.")
        if not os.path.exists(a.alignment_preflight):
            raise SystemExit(f"STOP: no alignment preflight marker at {a.alignment_preflight}")
        _pf = json.load(open(a.alignment_preflight))
        # A SUPERSET MARKER AUTHORISES A SUBSET RUN, which exact digest equality cannot express.
        # Raised by the owner: a memory probe or a --limit-units benchmark builds a SHORT manifest,
        # so its digest differs from the full marker's and the run was refused -- even though a
        # marker proving alignment for 851 structures plainly vouches for the 2 a probe touches.
        # That made short runs impossible and blocked re-measuring --ckpt-token-threshold for a new
        # population, which is what let an OOM reach 2.6 h into an epoch.
        #
        # THE SUBSET TEST IS BY NAME AND STRICTER THAN THE DIGEST, not looser: every structure this
        # run will train must appear in the marker's `checked` list, and the marker must still
        # record ok and complete. A marker with no `checked` list cannot be used this way and falls
        # back to requiring digest equality, so an old marker never silently authorises more than
        # it did before.
        # THE MARKER ALREADY PROVES THIS WITHOUT A NAME LIST, which is the cheap version I missed
        # first. `complete: true` means the preflight checked EVERY structure in the selection it
        # names, after that selection's skips. So a run using the SAME --manifest-selection and the
        # SAME --manifest-skip is necessarily a subset of what was checked, whatever subset of UNITS
        # it walks. No per-structure list is needed and no rebuild of the marker is needed; I
        # regenerated one for 20 minutes of CPU, competing with a live trainer, before noticing.
        #
        # The `checked` list the preflight now also writes is a strictly tighter check and is used
        # when present -- it covers the case of a run whose selection differs but whose structures
        # happen to be covered.
        _pf_checked = set(_pf.get("checked") or [])
        _run_pdbs = {p for p, _lp, _c in flat}

        def _skipset(txt):
            return {x.strip().lower() for x in (rt.txt or "").split(",") if x.strip()}

        # A WIDER SKIP LIST OVER THE SAME SELECTION IS STILL A SUBSET, which equality cannot see.
        # The marker is `complete` over (selection - its own skip). This run trains
        # (selection - this run's skip). So whenever the run's skip set CONTAINS the marker's,
        # the run's structures are a subset of what was checked -- set algebra, not a heuristic:
        #     run_skip >= pf_skip  =>  (sel - run_skip) <= (sel - pf_skip) = checked
        #
        # THE CASE THAT FORCED IT: ten structures were excluded for not fitting uncropped, so the
        # skip went from '9mx1,4zto,9q5m' to that plus ten names. Same selection, 841 of the 851
        # already-checked structures, and the digest changed -- so this refused a marker that
        # plainly vouched for every structure in the run. Equality of the skip STRING is also
        # order- and whitespace-sensitive, which would refuse a reordered list of identical names.
        #
        # STRICTLY NARROWER THAN THE DIGEST, never wider: a skip list that REMOVES a name the
        # marker skipped means the run trains a structure the preflight never saw, and that fails
        # this test and is refused as before.
        _pf_skip, _run_skip = _skipset(_pf.get("manifest_skip")), _skipset(a.manifest_skip)
        _same_population = (
            bool(_pf.get("complete"))
            and (_pf.get("manifest_selection") or "") == (a.manifest_selection or "")
            and _run_skip >= _pf_skip)
        _subset_ok = _same_population or (bool(_pf_checked) and _run_pdbs <= _pf_checked)
        if _pf.get("manifest_digest") != _manifest_digest and _subset_ok:
            _why = ((f"it is COMPLETE over the same selection and this run skips "
                     f"{len(_run_skip)} structure(s) against the marker's {len(_pf_skip)} "
                     f"(a superset: {sorted(_run_skip - _pf_skip) or 'identical'}), so every "
                     f"structure this run touches was checked") if _same_population else
                    f"this run's {len(_run_pdbs)} structure(s) are a SUBSET of the "
                    f"{len(_pf_checked)} named in its `checked` list")
            print(f"  alignment preflight: digest differs ({_pf.get('manifest_digest')} vs "
                  f"{_manifest_digest}) but {_why}, so it authorises this run.")
        elif _pf.get("manifest_digest") != _manifest_digest:
            raise SystemExit(
                f"STOP: the alignment preflight was run over manifest "
                f"{_pf.get('manifest_digest')}, this run is {_manifest_digest}. It says nothing "
                f"about the structures this run will train on; re-run it.\n"
                + (f"  Its selection/skip are {_pf.get('manifest_selection')!r} / "
                   f"{_pf.get('manifest_skip')!r} against this run's "
                   f"{a.manifest_selection!r} / {a.manifest_skip!r}, and it records no `checked` "
                   f"list, so neither subset path applies."
                   if not _pf_checked else
                   f"  {len(_run_pdbs - _pf_checked)} of this run's structure(s) are NOT in the "
                   f"marker's checked list, e.g. {sorted(_run_pdbs - _pf_checked)[:5]}."))
        if not _pf.get("ok"):
            raise SystemExit(f"STOP: {a.alignment_preflight} does not record a pass.")
        # THE MARKER MUST COVER THE WHOLE MANIFEST, not just carry its digest. The digest is
        # computed over every unit even when the preflight ran with --limit, so a one-structure
        # debug marker used to pass both checks above and authorise all 491. Asserted on BOTH
        # sides: the preflight now refuses to write ok=true when incomplete, and this refuses a
        # marker that does not say so -- including one written before that change existed.
        if not _pf.get("complete"):
            _lim = _pf.get("limit")
            raise SystemExit(
                f"STOP: {a.alignment_preflight} does not record complete=true. It checked "
                f"{_pf.get('n_structures')} of {_pf.get('n_manifest_structures', '?')} structure(s)"
                + (f" (--limit {_lim})" if _lim else "")
                + ", so it says nothing about the rest. Re-run the preflight without --limit.")
        print(f"  alignment preflight: PASS over {_pf.get('n_structures')} structure(s), "
              f"digest {_pf.get('manifest_digest')}")
        print(f"  MINIMAL ITEMS: the dataset item is not built; the loss reads the cached label "
              f"only ({len(rt.BC.LOSS_FEATS)} features + coordinate/coordinate_mask)")
    # THE SHARD IS APPLIED HERE, AFTER the manifest digest, the PAE mask coverage and the
    # alignment preflight have all run over the WHOLE population. Slicing sel_pdbs earlier -- the
    # first version of this -- made each child compute a digest over its own 17-18 structures and
    # compare it to a marker written for all 53, so every child exited with "the alignment
    # preflight was run over manifest X, this run is Y". All three shards would have failed, and
    # the any-shard-fails policy would then have dropped the held-out point EVERY epoch: a
    # permanently dead monitor that looks like a configuration problem. Caught in review.
    #
    # Filtering by STRUCTURE keeps every seed of a structure in one child, which is convenient but
    # is NOT what makes the pooled mean correct -- disjoint, complete pose coverage is, and that
    # is asserted by the parent against the expected population before it publishes a number.
    if a.manifest_nshards and a.manifest_nshards > 1:
        _all_pdbs = sorted({x[0] for x in flat})
        _mine = set(_all_pdbs[a.manifest_shard::a.manifest_nshards])
        _n_before = len(flat)
        flat = [x for x in flat if x[0] in _mine]
        print(f"  manifest SHARD {a.manifest_shard} of {a.manifest_nshards}: "
              f"{len(_mine)} of {len(_all_pdbs)} structure(s), {len(flat)} of {_n_before} unit(s), "
              f"interleaved -- digest and preflight above cover the FULL population")
        # MACHINE-READABLE, because the parent cannot derive the FULL population itself: units are
        # dropped for missing labels or trunks after the selection is read, so
        # structures x seeds overstates it. Only the child knows _n_before. The parent asserts the
        # shards agree on it and that they sum to it, which is what makes the pooled mean a mean
        # over the same population every epoch.
        print(f"SHARD COVERAGE shard={a.manifest_shard} nshards={a.manifest_nshards} "
              f"units_mine={len(flat)} units_all={_n_before} nsample={a.nsample}", flush=True)
    print(f"  resolved {len(flat)} unit(s) over {len(set(x[0] for x in flat))} structure(s) "
          f"from {len(roots)} fold root(s), shuffle={a.shuffle}, seed={a.seed}")

    cfg = rt.DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    err = os.path.join(rt.ROOT, "training", "prep", "dataset_errors")
    perm = rt.SymmetricPermutation(cfg, error_dir=err)
    # THE SAME FINGERPRINT CONTEXT build_cache USES, imported rather than reimplemented. A
    # reimplementation that drifted would reject every label or accept every label; round 8 found
    # the latter, because `fingerprint` was required to be PRESENT and never compared.
    code_fp, loss_fp = rt.BC.fingerprint_context(cfg)
    print(f"fingerprint context: code={code_fp} loss={loss_fp}")



    # PERSIST THE TRAJECTORY PER STEP, not at the end. A run that dies at step 900 of 1000 must still
    # be analysable, and a loss curve is the only way to see a learning rate that is wrong.
    # CROP BOOKKEEPING, counted rather than assumed. An arm whose crop silently failed to apply on
    # many structures is not the arm it claims to be, and the retained fraction is the only record
    # that says which happened.
    crop_stats = {"n": 0, "tok": 0, "kept": 0, "frames": 0, "undefined": 0}
    crop_rows = []
    crop_cols = ["pdb", "unit", "n_tok", "kept_tok", "frames_cleared", "status"]
    # PER RANK, because each rank crops its own third of the units and only rank 0 was flushing:
    # the other two accumulated rows forever and wrote none, so the csv described a third of the
    # run while reading like the whole of it. Review round 6 caught that.
    crop_p = ""

    traj_p = os.path.join(a.out, "trajectory.csv")
    # AND A PER-RANK COPY, for the same reason crop_sizes.csv got one above. trajectory.csv is
    # written by rank 0 alone (write_traj=is_main), so its per-pose columns describe the third of
    # the units rank 0 walked while reading like the whole run -- which is how five plotters came
    # to report a rank-local mean. The per-EPOCH curve is safe either way, because batch_<term> is
    # already reduced across ranks; what is NOT recoverable from rank 0 alone is any per-pose
    # question that filters by pdb (an unwinnable-structure filter, a per-structure histogram),
    # because the other ranks' poses were never written down anywhere.
    #
    # trajectory.csv is left exactly as it is so nothing that reads it today changes. This is an
    # ADDITIONAL file per rank, rank 0 included, so the full set is self-describing rather than
    # "the main one plus the others".
    traj_rank_p = os.path.join(a.out, f"trajectory_rank{rt.rank}.csv")
    # THE BATCH COLUMNS ARE THE CURVE; the bare TERMS are the single pose that happened to be last.
    # batch_<term> is the mean over ALL accum_eff poses in the step, reduced across every rank, and
    # batch_n is how many actually contributed -- accum_eff on a normal step, fewer on the partial
    # step an epoch ends with. Added after review 2026-10-16: reporting read the stepping pose's own
    # loss as the batch loss, which is 1 pose of 255, and no amount of reader-side work can
    # reconstruct the other ranks' poses because they were never written.
    traj_cols = ["wall_s", "step", "unit_i", "pdb", "unit", "pose", "lr"] + list(rt.TERMS) + \
                ["batch_" + k for k in rt.TERMS] + ["batch_n", "partial_step", "step_ok"] + \
                ["grad_norm", "outside_head", "epoch", "step_taken", "nonfinite_grad"] + \
                (["pae_loss_unweighted"] if float(0.0) > 0 else [])
    # pae_loss IS THE WEIGHTED TERM once --pae-row-frac is on, because it is what is trained, and
    # a mean-1 row weight does NOT preserve the value: the weighted mean equals the unweighted one
    # only if the per-row loss is uncorrelated with the weight, and the weight is chosen BY t_i.
    # So the comparable curve lives in pae_loss_unweighted, read back every pose from the replayed
    # label, and plot_train_pae.py --unweighted reads that column instead.
    # the per-structure item build is the cost --preload-items exists to amortise, so it is measured
    # rather than assumed; reported at the end against the number of structures actually visited.
    # WHERE AN EPOCH'S WALL CLOCK ACTUALLY GOES, accumulated per rank and printed at every epoch
    # boundary. Always on: a perf_counter pair is ~100 ns against a 0.58 s pose, and the last time
    # this was guessed at rather than measured, 0.67 s of a 3.10 s step attributed to the collective
    # turned out to be straggler wait (b829879d). The point is to order the optimisations by
    # measurement instead of by which is easiest to write.
    #
    # READ EPOCH 2, NOT EPOCH 1. Two costs are first-touch only: resolve_trunk memoises
    # unit_fingerprint in a mutable default argument, so the full-trunk hash runs once per unit per
    # PROCESS, and the dataset item cache fills on first use. Epoch 1 pays both for all 4880 units
    # and is not representative of the 499 that follow.
    _T = collections.defaultdict(float)
    _TN = collections.defaultdict(int)
    # [units run UNCHECKPOINTED, units run CHECKPOINTED] under --ckpt-token-threshold. Reported
    # per epoch because a threshold that silently matches nothing -- or everything -- looks
    # exactly like a threshold that works until the timings fail to move.
    # THE PRECOMPUTED CROP MASKS. __n and __meta are BOOKKEEPING, not structures: the evaluator's
    # loader filters only __n and so counts __meta as a structure, which is why it prints one more
    # than it has. Filter both.
    _CROPTOK = {}



    # THE EPOCH THE TOKEN DRAW IS SEEDED FROM. Set once at each epoch boundary BEFORE the
    # prefetch workers start, then read-only for the epoch -- which is what makes it safe for
    # those threads to read without a lock. It has to be a closure cell rather than a parameter
    # because _cpu_crop runs in the worker, which crops units before the step is known.
    _crop_epoch = [0]
    _tok_drop_stat = {"kept": 0.0, "tot": 0.0, "n": 0, "topped": 0, "topped_tok": 0}
    _ckpt_hist = [0, 0]
    # THE EPOCH'S PEAK, KEPT BY HAND, because --mem-log resets the CUDA high-water mark before
    # every pose to make the per-pose number mean the pose. Without this the epoch line would
    # report the LAST pose's peak and call it the epoch's. Zeroed at each epoch boundary beside
    # reset_peak_memory_stats. Unused when --mem-log is off: the CUDA counter is then already the
    # epoch's, because nothing reset it mid-epoch.
    _mem_ep = [0]



    # GPU TIME NEEDS CUDA EVENTS, NOT WALL CLOCK. A kernel launch returns before the kernel runs,
    # so perf_counter around .backward() times the launch and reports a few milliseconds for work
    # that takes hundreds. `pose_total` happens to be honest because float(met["loss"]) inside it
    # forces a device sync, but the forward/backward split is not, and that split is what decides
    # whether disabling activation checkpointing could pay.
    #
    # DRAINED ONCE PER STEP, not per pose. elapsed_time() requires the events to have completed, so
    # querying per pose would impose ~84 syncs per step and distort exactly what it measures. The
    # pairs are held and read at the step boundary, where the gradient reduction synchronises
    # anyway.
    _ev_pending = []


    nonfinite = {"n": 0}
    code_drift = {"n": 0, "warned": False}

    # ONE FINGERPRINT CACHE, SHARED BY BOTH CALLERS. unit_fingerprint SHAs the trunk (~205 MB),
    # the job cif and the bioassembly pickle. resolve_trunk memoised it privately in a default
    # argument, but run_unit re-asserts it per unit and did NOT, so a full SHA of all three ran on
    # every unit of every epoch: at 1620 units per rank that is roughly half of this rank's trunk
    # I/O spent re-proving something proved in epoch 1.
    #
    # THE SAFETY PROPERTY IS UNCHANGED. The re-assertion exists so the check cannot be skipped if
    # resolve_trunk is ever bypassed -- and a bypass leaves this cache without the key, so it still
    # computes and still asserts. What is skipped is only the SECOND and later computations of a
    # value that is a pure function of files this process has already read.
    #
    # RAM: one short string per (pdb, trunk path), 4880 keys on the full set -- about 1 MB.
    _fp_cache = {}
    # AND THE CONTENT HALF ALONE, which is the half that costs anything and the only half that is
    # persisted. _fp_cache holds the COMPOSED string this process compares against labels;
    # _fp_content holds `cif|bio|idx|trunk` for the same key, so a new process (or a sibling rank)
    # can skip the four SHAs and recompose. Both are filled by _unit_fp and nothing else.
    _fp_content = {}
    # AND THE STAT SIGNATURE AS IT WAS WHEN THE CONTENT WAS HASHED, so what is persisted describes
    # the files that were actually read. See _unit_fp.
    _fp_sig_at_hash = {}
    # THE RESOLVED TRUNK PATH, per (pdb, label). resolve_trunk is a pure function of the label's
    # stored fingerprint and the candidate trunks' own, and the per-unit re-assertion in run_unit
    # still checks the chosen trunk every epoch -- so caching the CHOICE skips a label npz open
    # and a dict comparison, not a safety check.
    #
    # MEASURED: the one-unit-ahead prefetch calls resolve_trunk on the next unit, and epoch 1 of
    # x3k2m4 spent 8.1 min of 77.9 inside prefetch_issue doing exactly that -- my own overhead,
    # larger than the trunk read it was meant to hide. With the choice cached it is a dict lookup
    # plus one fadvise syscall.
    _trunk_choice = {}

    # ---- THE FINGERPRINT CACHE, PERSISTED BESIDE THE FOLD -------------------------------------
    #
    # WHY. The in-memory cache above saves ~8 min per epoch, but only after the FIRST epoch of each
    # PROCESS: it is not in the resume bundle, so every restart re-pays the hashing of 4,880
    # trunks, cifs and bioassembly pickles. Review caught this, and it matters more than it looks
    # -- this project restarted four times in one day, and a 500-epoch run will restart again.
    #
    # WHAT IT TRUSTS, AND WHAT THAT GIVES UP. unit_fingerprint is a pure function of SCHEMA,
    # code_fp, loss_fp, the job cif, the bioassembly pickle, the indices row and the trunk. A fresh
    # process currently re-reads all of those, which re-verifies them. Persisting replaces that
    # with a stat signature -- size and mtime_ns of every file the fingerprint covers. That is the
    # same bargain phase_k_eval's --join-cache already makes ("the cache key carries each trunk's
    # size and mtime, so a refold or rebuild invalidates it"), and it is a real reduction in
    # checking: a file rewritten with its size and mtime preserved would go unnoticed. Everything
    # that rewrites these files -- a refold, a rebuild, the 2026-10-14 geometry fix -- changes mtime.
    #
    # ONLY THE CONTENT HALF IS PERSISTED, AND THAT IS THE ~43 MIN FIX. Version 1 stored the WHOLE
    # fingerprint and refused a file whose stored code_fp or loss_fp differed from this process's,
    # dropping all ~11,040 entries at once -- so every edit to phase_j_build_cache or to the loss
    # config threw away the entire cache and the run re-SHA'd ~243 MB per unit from scratch, ~43 min
    # off the front of the launch. But `code` and `loss` are three short strings this process
    # already holds and they cannot change what sha256(trunk) is. So the file now stores only
    # BC.unit_content_fingerprint's output -- cif|bio|idx|trunk -- and the schema/code/loss prefix
    # is recomposed at every use by BC.unit_fingerprint(..., content=...). The composed string is
    # byte-identical to version 1's, gated by phases/test_unit_fingerprint_split.py against a
    # literal of the pre-split format, so the provenance comparison is unchanged: a stale code_fp
    # still mismatches the label's stored fingerprint and still raises, exactly as before. code_fp
    # and loss_fp are still written into the file, as a record of what wrote it, and are no longer
    # read for invalidation.
    #
    # THE STAT SIGNATURE IS STILL THE WHOLE INVALIDATION, and it covers every file the content half
    # hashes -- trunk, cif, bioassembly, indices. Nothing is adopted from disk without it matching.
    #
    # PER RANK, AND ALL RANKS' FILES ARE READ. Each rank sees a third of the units and the shuffle
    # moves that third every epoch, so one rank's file covers only part of the manifest. Writing
    # one file per rank and loading every one of them accumulates coverage across epochs and across
    # runs without any rank having to write what another measured.
    _FP_CACHE_VER = 2
    _fp_disk = os.path.join(roots[0], f"unit_fingerprints.r{rt.rank}.json")



    # ---- UNIT PREFETCH: the CPU half of the crop, run ahead on a worker thread ---------------
    #
    # WHY. Measured on epoch 5: crop_slice is 23.0 min of a 73.1 min epoch and the GPU is idle for
    # all of it -- each rank prepares a unit on ONE core (90% of one core on a 288-core box),
    # then runs that unit's 5 poses on the GPU, then prepares the next. GPU busy time is ~20 min
    # of 73. The two phases take turns when they could overlap.
    #
    # NOT PARALLELISM -- OVERLAP. Threading the slice itself does not help: measured 0.088 s at
    # OMP=1, 0.084 at 4, 0.144 at 16. It is memory-bound numpy indexing, not BLAS. What helps is
    # doing it while the GPU is busy with the previous unit.
    #
    # THE WORKER NEVER TOUCHES CUDA. It calls slice_cached_unit with dev=cpu and hands back host
    # tensors; the main thread moves them. Only one thread ever issues CUDA work, which is the
    # whole reason this is safe to do with a thread rather than a process.
    #
    # BOUNDED AT --prefetch-units, default 1. Each cached slice is ~30 MB at our crop sizes, so a
    # depth of 1 costs ~30 MB per rank and a depth of 4 ~120 MB -- nothing against 250 GB, but the
    # queue is bounded anyway so a fast worker cannot run away with memory while the GPU lags.
    _pf_q = None
    _pf_thread = None
    _pf_stop = threading.Event()



    _pf_pending = {}
    # UNITS THAT WERE DELIVERED AND THEN EVICTED. A worker emits each unit exactly ONCE, so an
    # evicted result can never return to the queue -- and _pf_take's loop only exits on a match or
    # on the budget expiring, so without this it waits the FULL 120 s for something that no longer
    # exists, then recrops inline. The eviction pops in insertion order, i.e. the lowest-numbered
    # stashed unit, which is precisely the one the main loop asks for next: so one eviction becomes
    # a cascade of 120 s waits. Reproduced on CPU in run_logging/prefetch_deadlock_repro.py --
    # W=1 gives 400 hits and 0 stalls, W=4 gives 40 consecutive full-budget stalls and does not
    # recover. That is ggfhj5's epoch 12: every rank at 0% GPU in futex_wait, nine minutes with no
    # trajectory write, and --dist-timeout-min 240 keeping anything from erroring out. Recording
    # the eviction turns each of those waits into an immediate inline recrop.
    _pf_dropped = set()
    # HIT/MISS IS REPORTED, because "crop_slice did not move" is how the first version's total
    # failure went unnoticed -- the inline fallback ticks the same counter as the fast path.
    _pf_stat = {"hit": 0, "miss": 0, "wait_s": 0.0, "dropped": 0, "exhausted": 0}



    # DID THE CROP PREFETCH ACTUALLY START THIS EPOCH? Read by the epoch report, which cannot use
    # _pf_thread (it is cleared by _pf_end, which runs first) and must not use a.prefetch_units
    # (set on uncropped arms, where _pf_start now declines).
    _pf_started = [False]



    # ---- THE STEP-SCOPED TRUNK CACHE, for the UNCROPPED arm -------------------------------------
    #
    # WHY. On an uncropped arm the only per-unit I/O is RC.load_unit, and the cross-seed objective
    # calls run_unit TWICE on every unit of a step -- a no_grad scoring pass A, then the backward
    # pass B -- so each unit's ~243 MB was materialised and copied to the device twice. Rank 0,
    # epoch 3 of mqird6: 4,600 load_unit calls for 2,300 units, 37.0 min of `unit_setup` with
    # 0.2 min in named leaves and 31.0 of the epoch's 35.9 min residual INSIDE units. Holding the
    # HOST arrays for the duration of ONE step halves the reads outright, and warming them on a
    # worker thread puts the surviving half behind the GPU.
    #
    # KEYED BY TRUNK PATH AND NEVER POPPED, which is the whole design and is not negotiable. The
    # cropped prefetch (_pf_take) is a POP QUEUE keyed by label with a 120 s blocking wait, and
    # pass A walking the same units as pass B consumed each unit's result and left pass B waiting
    # out the full budget per unit -- ~3 hours an epoch, observed on probe dsswrj, looking like a
    # hang with flat CPU and 0% GPU. A read-only dict cannot do that: both passes LOOK UP the same
    # key, a lookup never blocks, and a miss just reads inline.
    #
    # PURELY ADVISORY, SO NOTHING CAN DEADLOCK ON IT. Every path that consults it falls back to the
    # ordinary read, so a worker that is slow, failed, capped out or never started costs time and
    # nothing else. There is no wait, no budget and no eviction cascade to get wrong.
    #
    # ITS OWN CAP, IN BYTES, SEPARATE FROM THE CROP PREFETCH'S. _pf_pending's max(16, 8*workers)
    # would be 32 entries, and at 243 MB per uncropped trunk that is ~7.8 GB of RSS per rank
    # against --max-rss-gb. This caps on BYTES, defaults off, and REFUSES rather than evicts when
    # full: every entry in a step is wanted twice, so dropping the oldest would drop one pass B is
    # about to ask for, while refusing the newest only forgoes a saving. A deep step of the largest
    # remaining structures is 10 units x 1,650 tokens ~= 8.5 GB, so a 4 GB cap covers a typical
    # step and partially covers the worst, which is the intended behaviour.
    #
    # CLEARED AT EVERY STEP BOUNDARY. Scope is one step because that is the span over which a unit
    # is read twice; holding longer would grow without bound, since the epoch's units are distinct.
    _tr_cap = int(float(a.trunk_cache_gb) * (1 << 30))
    # CUDA IS PART OF THE PRECONDITION, not an accident of this box. unit_from_host aliases the
    # host arrays when the device is CPU, so two passes sharing one dict would share any in-place
    # write; `.to(cuda)` always copies. dev is cuda at line 1528, and this says so out loud.
    #
    # AND ONLY WHERE THERE IS A PASS A. The cache is scoped to a step and warmed at a step
    # boundary, both of which come from the cross-seed grouping; without it `_xs_at` is empty,
    # _tr_step_start is never called, every unit is read exactly once and there is nothing to hit.
    # The cache would then fill to its cap, refuse everything after that and hold the RSS for no
    # benefit -- so the same condition that gates _xs_prepare gates this.
    _tr_on = (_tr_cap > 0 and not (0.0 > 0 or '')
              and getattr(rt.dev, "type", "") == "cuda"
              and a.rank_cross_seed and a.rank_loss and not a.loss_only)
    _tr_cache = {}
    # PATH -> RESERVED BYTES, not a bare set of paths. A set records THAT a read is in flight; the
    # cap needs to know HOW MUCH. Review found that _tr_bytes counted only ADMITTED entries, so the
    # pre-read check `_tr_bytes[0] + _sz > _tr_cap` was evaluated against a number that four
    # concurrent warmers all saw as the same low value: each passed, each allocated its whole
    # ~243 MB, and TRUNK_CACHE_GB bounded the cache while the PROCESS went ~4x over it. The
    # reservation is taken under the lock at the same instant the check passes and released in the
    # `finally`, so the cap now bounds admitted + in-flight, which is what host RSS actually is.
    _tr_inflight = {}
    _tr_lock = threading.Lock()
    _tr_bytes = [0]
    _tr_resv = [0]
    # SHARED WITH THE WARMERS, AND `+=` ON A DICT ENTRY IS NOT ATOMIC, so a simultaneous put can
    # lose one sample out of thousands. Acceptable for a diagnostic and NOT acceptable for anything
    # the run branches on; nothing branches on these. Same bargain as _cpu_crop's counters.
    _tr_stat = {"hit": 0, "miss": 0, "put": 0, "full": 0, "err": 0, "dup": 0}
    _tr_threads = [None]
    # EVERY warmer ever started and not yet reaped. _tr_threads[0] holds only the CURRENT step's,
    # which is what made an older generation's thread unjoinable.
    _tr_live = []
    # A GENERATION COUNTER, NOT AN EVENT, AND THAT IS A BUG FIX. The first version used one shared
    # threading.Event: _tr_step_start did set() -> join(timeout=30) -> clear(). A warmer still
    # inside a 243 MB read when the join expired then resumed its loop, found the flag CLEARED, and
    # went on warming the PREVIOUS step's remaining units -- spending the new step's cap on trunks
    # nobody would ask for, so the step's real units were refused and read twice anyway. It was
    # never joined again either, because _tr_threads[0] had been replaced, so up to
    # --prefetch-workers orphans could leak per step boundary and live for the rest of the run, all
    # of them writing _fp_cache / _fp_content / _T without a lock while the main thread iterated
    # them at the epoch boundary. A generation number cannot be "un-cancelled": a worker compares
    # its own `gen` against this cell and a stale worker can only exit.
    #
    # AND IT LETS THE JOIN GO, which is the other half. Joining W warmers on the MAIN thread at
    # every one of 230 step boundaries put the main thread behind whatever read was in flight --
    # inside the very critical path this exists to shorten. Nothing is joined per step now; the
    # epoch boundary still joins, once, so no warmer races _fp_cache_save.
    _tr_gen = [0]


        # A WARMER THAT OUTLIVES THE JOIN CAN STILL BE HOLDING ONE READ, and that is all it can do:
        # its generation is retired, so _tr_put drops the result and the loop exits at its next
        # top. Nothing it does after this line reaches the cache or the byte count.

    # LOADED HERE, after the helpers above are bound and before the first resolve_trunk call.
    # Calling it beside DS.build_configs raised UnboundLocalError: the name is local to main() and
    # that line runs ~280 lines before the def.
    _fp_cache_load()



    # ONE LIST, NAMES AND TENSORS TOGETHER, at main level so both _apply_step and the gradient
    # probe read the same objects in the same order. They were built independently before, and
    # `params` lived inside _apply_step where _one_pose could not see it -- the probe would have
    # raised NameError the first time it ran. Zipping two separately-built lists also risks
    # pairing a name with the wrong tensor. Found in review of a3086ef.
    trainable_params = [(n, pp) for n, pp in rt.model.named_parameters() if n in set(rt.trainable)]
    if a.rank_loss and not trainable_params:
        raise SystemExit("STOP: no trainable tensors found for the ranking term.")



    # DERIVED FROM --seeds, NOT FROM --eval-budget. The validation fold holds 100 seeds per
    # interface; the training fold holds exactly the ones this run trained on, and asking the
    # evaluator for more than exist makes it refuse rather than quietly report a thinner number.


    # ---- LR PROBE: measure the learning rate instead of inheriting one tuned for a different job
    if a.lr_probe:
        lrs = [float(x) for x in a.lr_probe.split(",")]
        print(f"\nLR PROBE: {a.probe_steps} optimiser step(s) at each of {lrs}, all from the SAME "
              f"initial weights. Protenix's 0.0018 default is for finetuning the whole model; this "
              f"moves 4.3% of it, so the default is not transferable.")
        out = []
        for lr in lrs:
            for n, p in rt.model.named_parameters():
                if n in set(rt.trainable):
                    p.data.copy_(rt.init_state[n])
            rt.opt.load_state_dict(rt.copy.deepcopy(rt.init_opt))
    # The call to train_loop that stood here is main()'s epilogue, not glue. Keeping it
    # would have this stage run the training loop as a side effect of being built.
            if not rows:
                print(f"  lr {lr:g}: no rows"); continue
            # THE BATCH LOSS ON STEPS ACTUALLY TAKEN, not every per-pose row. `rows` is one entry
            # per POSE on rank 0; averaging those compares learning rates on rank 0's slice of a
            # handful of structures, and review 2026-10-16 built a case where the global batch loss
            # fell while rank 0's pose losses rose, so this recommended LOWERING the lr. The
            # cross-rank batch mean is in the row the optimiser stepped on.
            _st = [r for r in rows if r.get("step_taken") and r.get("step_ok")
                   and r.get("batch_loss") is not None]
            if not _st:
                print(f"  lr {lr:g}: no completed optimiser step to judge; skipped"); continue
            k = max(1, len(_st) // 5)
            first = sum(r["batch_loss"] for r in _st[:k]) / k
            last = sum(r["batch_loss"] for r in _st[-k:]) / k
            fin = all(math.isfinite(r["batch_loss"]) for r in _st)
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

    # THE ESTIMATE USES --nsample, not a hardcoded 5. It was 5 because every unit has 5 poses in
    # this pilot, which is a fact about the data and not about the code.
    total_poses = int(len(flat) * a.nsample * a.epochs)
    # PER RANK. Each rank sees total_poses/world poses and steps every accum_local of them, so the
    # step count is total_poses/(world*accum_local) = total_poses/accum_eff either way, and a
    # multi-GPU run does the same number of optimiser steps as a single-GPU one -- in a third of the
    # wall clock, rather than a third of the training.
    # THE CAP IS FOR DEBUGGING ONLY, AND IS OFF FOR REAL TRAINING. total_poses // accum_eff counts
    # full accumulations, but the loop ALSO flushes a partial one at every epoch boundary, so the
    # step counter advances faster than the cap anticipates and the last epoch could be cut short.
    # Review 2026-10-16 reproduced three epochs of ten poses running only 24 of 30. The epoch loop
    # already knows when it is finished; `--limit-units` and the lr probe remain for capped runs.
    # 9mfb4y escaped by one step (cap 481, 480 taken) only because total_poses counts all 491
    # structures while the loop runs the 489 that survive shard truncation -- two errors cancelling.
    # THE HORIZON MUST COME FROM THE EPOCH BUILDER, NOT FROM len(flat) // world. build_epoch
    # takes floor(n_seeds / world) WHOLE seed-groups of each structure, so at ten seeds and world 3
    # one seed per structure sits out and the rank walks 3 groups, not 10/3. len(flat) // world
    # counts that sitting-out seed, and the error lands on the LEARNING RATE: at 368 structures it
    # claimed 205 steps per epoch against an actual 184, which stretches the decay window by 11%.
    # The configured "1e-2 flat to epoch 20, 1e-4 by epoch 50" came out as flat to 22.3 and 1e-4 at
    # 55.7, leaving lr ~7.95e-4 at epoch 50 -- eight times the intended value -- so the schedule
    # the arm is testing was not the schedule it ran. Caught in review of 38a83283.
    _units_by_pdb = {}
    for _p, _lp, _c in flat:
        _units_by_pdb[_p] = _units_by_pdb.get(_p, 0) + 1
    _units_per_rank = rt._LRS.units_per_rank(_units_by_pdb, rt.world,
                                          bool(a.rank_cross_seed), len(flat))
    _spe_ep = rt._LRS.steps_per_epoch(_units_by_pdb, rt.world, a.nsample, rt.accum_local,
                                   cross_seed=bool(a.rank_cross_seed), n_units=len(flat),
                                   structs_per_step=(a.rank_structs_per_step
                                                     if a.rank_cross_seed else 0),
                                   preserve_order=('shuffle' in ("manifest", "tokens")))
    # THE SPLIT, REPORTED HONESTLY. _full is whole accumulations and the partial exists only when
    # the poses do not divide accum_local -- at 368 structures they do (1,104 x 5 / 30 = 184), so
    # there is no partial step and the old unconditional "+ 1 partial" overstated the horizon.
    _poses_pr = _units_per_rank * a.nsample
    _full = _poses_pr // max(1, rt.accum_local)
    _part = 1 if _poses_pr % max(1, rt.accum_local) else 0
    _expect = _spe_ep * max(1, math.ceil(a.epochs))
    _naive = ((len(flat) // max(1, rt.world)) * a.nsample // rt.accum_local + 1)
    if _naive != _spe_ep:
        print(f"  steps/epoch {_spe_ep} from the epoch builder, NOT {_naive} from "
              f"len(flat)//world: {len(flat)} unit(s) over {len(_units_by_pdb)} structure(s), of "
              f"which this rank walks {_units_per_rank} "
              f"(floor(seeds/{rt.world}) whole group(s) per structure; the remainder sits out and is "
              f"reseeded each epoch). The lr schedule uses the former.")
    steps_cap = 0
    rt._lr_horizon[0] = _expect
    # WHAT THE HORIZON WAS BUILT FROM, so build_epoch_cross_seed can refuse if it packs a
    # different number of steps than the schedule was calibrated for.
    rt._xs_spe[0] = _spe_ep if a.rank_cross_seed and a.rank_structs_per_step else 0
    # --lr-steps IS CHECKED FIRST, because it takes precedence over --lr-final and the decay
    # window (see its help). This line derived the description from lr_final alone, so a run with
    # a staircase printed "flat after warmup (no decay)" while the staircase was in fact applied
    # -- parse_lr_steps' own docstring warns that "the run trains at the wrong lr for twenty
    # epochs and nothing in the log says so", and the startup line was saying exactly nothing.
    if a.lr_steps:
        _sched = "STAIRCASE: " + f"{a.lr:g}" + "".join(
            f" -> {_v:g} from epoch {_e:g}" for _e, _v in rt._LRS0.parse_lr_steps(a.lr_steps))
    elif 0.0 <= 0.0:
        _sched = "flat after warmup (no decay)"
    else:
        _sched = f"COSINE {a.lr:g} -> {0.0:g} over steps {a.warmup_steps}..{_expect}"
    print(f"\ntraining: lr {a.lr:g}, warmup {a.warmup_steps}, accum {a.accum}, "
          f"~{_expect} optimiser step(s) for {a.epochs} epoch(s) "
          f"({_full} full + {_part} partial per epoch); no step cap, the epoch loop decides")
    print(f"  lr schedule: {_sched}")
    print(f"trajectory -> {traj_p}")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    # --grad-probe STOPS HERE. The return inside train_loop only exits train_loop; main() then
    # ran on through the summaries to the unconditional head_final.pt save, so a probe wrote a
    # checkpoint indistinguishable from a trained one WHILE PRINTING that it had not. Found in
    # review of 523013a. It also corrects the claim: the probe DOES take one real optimiser step,
    # by design -- what it must not do is leave a head behind that looks like an arm's.
    if a.grad_probe:
        if rt.is_main:
            print(f"  --grad-probe complete. ONE optimiser step was taken (deliberately) and NO "
                  f"checkpoint is written: {a.out} holds no head_*.pt from this invocation, so it "
                  f"cannot be mistaken for a trained arm.", flush=True)
        if rt.world > 1:
            rt.dist.barrier()
            rt.dist.destroy_process_group()
        return

    if a.pae_scored_pairs:
        # ACROSS EVERY RANK, same reason as the crop summary: each rank masks its own third of the
        # units, so one rank's counters describe a third of the run. These were being counted and
        # printed NOWHERE, which made a mask that covered nothing indistinguishable in the log from
        # one that covered everything.
        _mk = ("masked", "unmasked")
        _ms = dict(rt._pae_mask_stats)
        if rt.world > 1:
            _t = torch.tensor([float(rt._pae_mask_stats[k]) for k in _mk], dtype=torch.float64)
            rt.dist.all_reduce(_t, op=rt.dist.ReduceOp.SUM)
            _ms = {k: int(v) for k, v in zip(_mk, _t.tolist())}
        if rt.is_main:
            _tot = _ms["masked"] + _ms["unmasked"]
            print(f"\nPAE mask: {_ms['masked']}/{_tot} pose-unit(s) masked to scored chain pairs "
                  f"({_ms['masked'] / _tot * 100:.1f}%) across {rt.world} rank(s), from "
                  f"{a.pae_scored_pairs}")
            if _ms["unmasked"]:
                print(f"  {_ms['unmasked']} fell back to the UNMASKED PAE term -- the structure is "
                      f"absent from the pool, or no scored pair survived the crop. Those poses "
                      f"trained the ungraded blocks too.")

    if code_drift["n"]:
        print(f"\nCODE DRIFT ACCEPTED on {code_drift['n']} unit(s) via --allow-code-drift. The "
              f"labels' data provenance matched in every field; only the builder's source hash "
              f"differed. This run is NOT reproducible from a clean checkout of that code.")
    # RANK 0 WRITES THE CHECKPOINT. Every rank holds the same parameters after the all-reduce, so
    # the content would be identical, but two processes writing one path concurrently can interleave
    # and leave a truncated file -- the smoke test showed both ranks reporting the same save.
    cp = os.path.join(a.out, "head_final.pt")
    if a.loss_only:
        # NOTHING TRAINED, SO NOTHING IS SAVED. --save-every is already refused for this mode, but
        # this save is unconditional and wrote head_final.pt on the first --loss-only pass: a file
        # indistinguishable from a trained head, holding the resumed bundle's weights under a name
        # that says otherwise. The measurement's output is the trajectory.
        _src = a.resume or "the shipped head"
        print(f"\n--loss-only: no checkpoint written -- {cp} would duplicate {_src} under a name "
              f"implying it was trained")
    elif rt.is_main or a.debug_save_rank_heads:
        import torch as _t
        _sd = {n: p.detach().cpu() for n, p in rt.model.named_parameters() if n in set(rt.trainable)}
        if rt.is_main:
            _t.save(_sd, cp)
            print(f"\nsaved {cp}")
        if a.debug_save_rank_heads:
            _rp = os.path.join(a.out, f"head_final_rank{rt.rank}.pt")
            _t.save(_sd, _rp)
            print(f"saved {_rp}")
    if a.loss_only:
        print("THE LOSS IS THE RESULT HERE, and only that: this mode measures a frozen head's loss "
              "on the fold it was pointed at. Aggregate the per-pose pae_loss column of "
              f"{os.path.join(a.out, 'trajectory.csv')} -- every pose is measured exactly once, so "
              "the plain mean over all rows is the pose-weighted figure. Do NOT use the "
              "step_taken==1 windows: --accum leaves a tail of poses outside the last complete "
              "window, which a training run re-traverses next epoch and this single pass does not.")
    else:
        # NO HARDCODED TARGET. This printed "the target is mean DockQ 0.5775 and frac-acceptable
        # 0.7283 at 100 seeds" after every run, which are the FULL validation pool's numbers --
        # wrong for any run scored on a subset, and val10's are 0.2626 / 0.5000. The same string
        # was hardcoded in phase_k_eval and is now derived there from the pool actually scored, so
        # the baseline belongs in that output, not in a constant here.
        print("THE LOSS IS NOT THE RESULT. Score this checkpoint with phases/phase_k_eval.py, which "
              "reports the shipped head's baseline over the pool it scored alongside the metric.")
    # Tear the group down so the ranks exit together rather than on NCCL's timeout.
    if rt.world > 1:
        rt.dist.barrier()
        rt.dist.destroy_process_group()
    return 0

    # EVERY LOCAL THIS STAGE BOUND GOES ONTO THE CONTEXT, by name.
    for _n in ('DOPS', 'PFC', '_CROPTOK', '_FP_CACHE_VER', '_T', '_TN', '_all_pdbs', '_c', '_ckpt_hist', '_crop_epoch', '_depths', '_dops', '_e', '_ev_pending', '_expect', '_fp_cache', '_fp_content', '_fp_disk', '_fp_sig_at_hash', '_full', '_head_names', '_lim', '_lp', '_manifest_digest', '_mem_ep', '_mine', '_mk', '_ms', '_n', '_n_before', '_naive', '_p', '_part', '_pf', '_pf_checked', '_pf_dropped', '_pf_pending', '_pf_q', '_pf_skip', '_pf_started', '_pf_stat', '_pf_stop', '_pf_thread', '_pfc', '_poses_pr', '_r', '_rp', '_run_pdbs', '_run_skip', '_same_population', '_sched', '_sd', '_seed_col', '_sel_rows', '_sel_tokens', '_si_d', '_si_inj', '_si_nz', '_si_t', '_skipset', '_spe_ep', '_src', '_st', '_subset_ok', '_t', '_tok_drop_stat', '_tot', '_tr_bytes', '_tr_cache', '_tr_cap', '_tr_gen', '_tr_inflight', '_tr_live', '_tr_lock', '_tr_on', '_tr_resv', '_tr_stat', '_tr_threads', '_trunk_choice', '_txt', '_units_by_pdb', '_units_per_rank', '_v', '_why', 'arm_seed', 'build_runtime', 'bypdb', 'cands', 'cfg', 'code_drift', 'code_fp', 'cp', 'crop_cols', 'crop_p', 'crop_rows', 'crop_stats', 'd', 'err', 'fh', 'fin', 'first', 'flat', 'hi_s', 'k', 'label_dirs', 'last', 'lo_s', 'loss_fp', 'lp', 'lr', 'lrs', 'manifest', 'n', 'no_label', 'no_trunk', 'nonfinite', 'order', 'out', 'p', 'pdb', 'perm', 'pp', 'pr', 'r', 'rng', 'roots', 'rows', 'sd', 'sel_pdbs', 'skip', 'steps_cap', 'total_poses', 'trainable_params', 'traj_cols', 'traj_p', 'traj_rank_p', 'v', 'want', 'wr', 'x'):
        if _n in locals():
            setattr(rt, _n, locals()[_n])
    return rt
