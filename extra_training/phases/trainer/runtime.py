"""`trainer/runtime.py` -- the glue between the extracted regions.

EXTRACTED from `phases/phase_k_train.py`: 12 gaps, 507 lines (source `7fc6870c`).

WHAT THIS IS, AND WHY IT WAS MISSED. The named regions -- bootstrap, dataset, unit, loop
-- are the parts of main() that look like functions. Between them sits the state they
all read: TERMS, the module handles, the seed columns, the prefetch and trunk caches,
the fingerprint cache, the timers. 99 of 115 `rt.<attr>` reads had no writer because
this was never extracted; I took what looked like code and left what looked like setup.

Gaps whose first statement guards a culled flag are NOT ported: if a.save_init:, if a.l2_sp:, if a.lr_probe:.
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



    _seed_col = {}
    for _r in rt._sel_rows:
        _p = _r["pdb"].strip().lower()
        _txt = str(_r.get("seeds", "") or "").strip()
        if _txt:
            _seed_col[_p] = _parse_seed_field(_txt, f"{a.manifest_selection}:{_p}")
    want = [(p, sd) for p in rt.sel_pdbs if p not in rt.skip
            for sd in sorted(_seed_col.get(p) or range(rt.lo_s, rt.hi_s + 1))]
    if _seed_col:
        _depths = collections.Counter(len(_seed_col.get(p) or range(rt.lo_s, rt.hi_s + 1))
                                      for p in rt.sel_pdbs if p not in rt.skip)
        print(f"manifest: per-structure seed depths from {a.manifest_selection}: "
              f"{dict(sorted(_depths.items()))} "
              f"({len(_seed_col)} of {len(rt.sel_pdbs)} row(s) declare `seeds`)")
    _rank_check_coverage(want)
    print(f"manifest: {len(rt.sel_pdbs)} selected structure(s) minus {sorted(rt.skip) or 'none'} "
          f"x seeds {rt.lo_s}-{rt.hi_s} (fallback for rows with no `seeds` column) = {len(want)} "
          f"intended unit(s), from {a.manifest_selection}")

    manifest, no_label, no_trunk = [], [], []
    for pdb, sd in want:
        arm_seed = f"free_{sd}"
        lp = next((os.path.join(d, pdb, f"{arm_seed}.npz") for d in rt.label_dirs
                   if os.path.exists(os.path.join(d, pdb, f"{arm_seed}.npz"))), None)
        if lp is None:
            no_label.append(f"{pdb}/{arm_seed}"); continue
        cands = [os.path.join(r, "trunks", pdb, f"{arm_seed}.npz") for r in rt.roots
                 if os.path.exists(os.path.join(r, "trunks", pdb, f"{arm_seed}.npz"))]
        if not cands:
            no_trunk.append(f"{pdb}/{arm_seed}"); continue
        manifest.append((pdb, lp, cands))        # ALL candidates; provenance picks below
    if no_label or no_trunk:
        raise SystemExit(
            f"STOP: the training set is INCOMPLETE.\n"
            f"  {len(no_label)} intended unit(s) have no label, e.g. {no_label[:5]}\n"
            f"  {len(no_trunk)} have no trunk under {rt.roots}, e.g. {no_trunk[:5]}\n"
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
          f"from {len(rt.roots)} fold root(s), shuffle={a.shuffle}, seed={a.seed}")

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
    _fp_disk = os.path.join(rt.roots[0], f"unit_fingerprints.r{rt.rank}.json")



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

    # EVERYTHING THIS STAGE BOUND GOES ONTO THE CONTEXT. The extracted stages read these
    # as rt.<name>; inside main() they were simply locals still in scope. Exported by name
    # rather than by vars() so the set is visible and a rename is a failure, not a silent
    # miss.
    for _n in ('DOPS', 'PFC', '_FP_CACHE_VER', '_all_pdbs', '_c', '_ckpt_hist', '_crop_epoch', '_depths', '_dops', '_ev_pending', '_fp_cache', '_fp_content', '_fp_disk', '_fp_sig_at_hash', '_head_names', '_lim', '_lp', '_manifest_digest', '_mem_ep', '_mine', '_n_before', '_p', '_pf', '_pf_checked', '_pf_dropped', '_pf_pending', '_pf_q', '_pf_skip', '_pf_started', '_pf_stat', '_pf_stop', '_pf_thread', '_pfc', '_r', '_run_pdbs', '_run_skip', '_same_population', '_seed_col', '_subset_ok', '_tok_drop_stat', '_tr_bytes', '_tr_cache', '_tr_cap', '_tr_gen', '_tr_inflight', '_tr_live', '_tr_lock', '_tr_on', '_tr_resv', '_tr_stat', '_tr_threads', '_trunk_choice', '_txt', '_why', 'arm_seed', 'bypdb', 'cands', 'cfg', 'code_drift', 'code_fp', 'd', 'err', 'flat', 'loss_fp', 'lp', 'manifest', 'n', 'no_label', 'no_trunk', 'nonfinite', 'order', 'p', 'pdb', 'perm', 'pp', 'pr', 'r', 'rng', 'sd', 'trainable_params', 'want', 'x'):
        if _n in locals():
            setattr(rt, _n, locals()[_n])
    return rt
