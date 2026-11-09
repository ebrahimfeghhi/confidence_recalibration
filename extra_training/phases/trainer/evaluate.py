"""`trainer/evaluate.py` -- the held-out and val-loss passes.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 6356-6831 (`val_loss_checkpoint`) and 6833-7042 (`eval_checkpoint`) (source `d34baebc`), dedented one
level from inside `main()`. Statements and comments are the original.

THESE WERE CLOSURES. Inside `main()` they read dozens of enclosing locals directly; here the
names they close over arrive as `ctx`, a plain namespace built by the caller. Nothing else about
them changed, and test_region_equivalence.py compares every line against the live source so a
change there fails the test instead of drifting.
"""
import collections
import csv
import json
import math
import os
import time

import torch

def val_loss_checkpoint(ctx, epoch, bundle, step):
    """Held-out PAE for the bundle this epoch just wrote. Returns the pose mean, or None.

    A SUBPROCESS, for the same reason eval_checkpoint is one: --loss-only carries its own
    guards -- the alignment preflight, the mask-coverage check, the manifest completeness
    check, the refusal to write a checkpoint -- and running it as the script keeps all of
    them. It also means the held-out fold's dataset config, labels and crop live entirely in
    another process, so there is no path by which measuring can perturb the weights being
    measured. That property is asserted by the caller rather than assumed.

    IT READS THE BUNDLE FROM DISK, not the live parameters. The epoch boundary has already
    written resume_ep<N>.pt, which carries all 248 head tensors, so the thing measured is
    exactly the thing a later scorer would load -- not a copy made by a different code path.
    """
    import json
    import subprocess
    out = os.path.join(ctx.a.out, f"val_loss_ep{epoch}")
    labels = ctx.a.val_loss_labels or os.path.join(ctx.a.val_loss_fold, "labels")
    cmd = [sys.executable, "-I", os.path.abspath(ctx.__file__),
           # A MEASUREMENT IS NOT A FORK, but it resumes the bundle with a different fold,
           # selection, seed range and epoch count, so every one of those reads as drift.
           # --resume-note is mandatory whenever drift is accepted, and the subprocess cannot
           # be the one exception -- a silent exemption is how the val monitor would die on
           # the next restart with a message about a note nobody could have passed.
           "--loss-only", "--resume", bundle, "--resume-allow-drift",
           "--resume-note", (f"held-out PAE measurement of {os.path.basename(bundle)} at "
                             f"epoch {epoch}; --loss-only reads the weights and writes no "
                             f"checkpoint, so the bundle's schedule is not continued"),
           "--fold", ctx.a.val_loss_fold, "--labels", labels,
           "--manifest-selection", ctx.a.val_loss_selection,
           "--alignment-preflight", ctx.a.val_loss_preflight, "--minimal-items",
           "--pae-scored-pairs", ctx.a.val_loss_pool,
           "--seeds", ctx.a.val_loss_seeds,
           # THE RUN'S OWN GEOMETRY AND WEIGHTS. A different crop or a different alpha would
           # make the held-out number incomparable to the training curve it is read against,
           # which is the only thing it is for.
           # --crop-margin 0 IS WHAT "UNCROPPED" MEANS here; the argument's own help says 0
           # is the whole structure. crop_contact and crop_size are still passed because they
           # are inert at margin 0 and passing them keeps the recorded config complete.
           "--crop-margin", ("0" if ctx.a.val_loss_uncropped else str(ctx.a.crop_margin)),
           "--crop-contact", str(ctx.a.crop_contact),
           "--crop-size", str(ctx.a.crop_size), "--accum", str(ctx.a.accum),
           # THE PATH IS PER SHARD, and the sharded launcher below rewrites it the same way
           # it rewrites --out. Handing every child this one path meant three processes
           # opening it with "w" and truncating each other: rows overwritten, interleaved or
           # lost, and the existing coverage checks look at trajectory.csv, not at this file,
           # so the corruption would not have been detected. Found in review of 511519e7.
           *(["--emit-iptm-rows", os.path.join(out, "iptm_rows.csv")]
             if ctx.a.val_loss_emit_iptm else []),
           # EIGHTH INSTANCE OF THE --loss-only INHERITANCE CLASS. The three crop args above
           # are passed because "a different crop would make the held-out number incomparable
           # to the training curve it is read against"; --crop-all-poses changes the crop by
           # the same argument and was not passed, so a parent using the union would have been
           # monitored on a pose-0 crop. Caught in review.
           *(["--crop-all-poses"]
             if (ctx.a.crop_all_poses and not ctx.a.val_loss_uncropped) else []),
           # ALPHA_PAE 0 IS A TRAINING CHOICE, NOT A MEASUREMENT ONE. A ranking-only arm sets
           # --alpha-pae 0 so no PAE gradient is taken, but the held-out number this
           # subprocess exists to produce IS the PAE loss, and passing 0 through would report
           # val10 PAE as exactly 0.0 every epoch -- silently destroying the one diagnostic
           # that is comparable across every arm (shipped 2.9756). So the monitor always
           # measures at alpha_pae 1. SIXTH instance of the --loss-only inheritance class:
           # --train-only, LoRA, --pose-perblock-share, the ranking guard fields, my own
           # coverage guard, now the loss weight itself.
           "--nsample", str(ctx.a.nsample),
           "--alpha-pae", str(ctx.a.alpha_pae if float(ctx.a.alpha_pae) > 0 else 1.0),
           "--alpha-confidence", str(ctx.a.alpha_confidence), "--clip-norm", str(ctx.a.clip_norm),
           "--blocks-per-ckpt", str(ctx.a.blocks_per_ckpt),
           "--epochs", "1", "--save-every", "0", "--warmup-steps", "0",
           "--shuffle", "structure", "--out", out]
    if ctx.a.pae_only:
        cmd.append("--pae-only")
    if ctx.a.drop_pde:
        cmd.append("--drop-pde")
    if ctx.a.frame_direction:
        cmd.append("--frame-direction")
    if ctx.a.pose_perblock:
        cmd.append("--pose-perblock")
        if ctx.a.pose_perblock_tied:
            cmd.append("--pose-perblock-tied")
        if ctx.a.pose_perblock_share:
            cmd.append("--pose-perblock-share")
        # PASSED TO THE CHILD, like the other two. A monitor that built a DIFFERENT
        # architecture than the parent would report a held-out number for a model that does
        # not exist -- the --loss-only inheritance class, now seven instances.
        if ctx.a.pose_perblock_copy:
            cmd.append("--pose-perblock-copy")
    # THE IN-BLOCK PATHWAY AND THE LEAN TRIO -- the NINTH AND TENTH instances of the
    # --loss-only inheritance class, both found by review before an arm was launched.
    # (Ninth and tenth, not eighth and ninth: :6165 already claims the eighth for
    # --crop-all-poses. The running count in these comments is off by one further up --
    # :6177 says "sixth" and :6200 "now seven instances" for what are the seventh and
    # eighth -- so the ordinals are a narrative, not an index. Worth one more than nothing
    # because the POINT of the running count is that this keeps happening.)
    #
    # --pose-inblock ADDS tensors, so a child without it fails the resume guard's parameter-set
    # comparison (phase_k_train.py:3373-3381) and the held-out point is simply absent.
    #
    # --lean-single is worse, because it would NOT fail loudly here. It changes no mathematics
    # on the PAE path, so the child would compute the right number -- but the guard compares
    # it, so the child refuses the bundle and the monitor reports nothing. Forwarded with the
    # two flags it hard-requires (phase_k_train.py:1788), since the child parses them itself.
    if ctx.a.pose_inblock:
        cmd += ["--pose-inblock", "--pose-inblock-block", str(ctx.a.pose_inblock_block)]
        if ctx.a.pose_inblock_pae:
            cmd.append("--pose-inblock-pae")
    # FORWARDED WHENEVER THE PARENT HAS IT, not only alongside --lean-single. lean_readouts
    # replaces memory_efficient_forward rather than adding or removing tensors, so its absence
    # in the child was invisible to both the key-set guard and the resume guard -- the child
    # was simply computing readouts the parent does not, slower and on a different code path.
    if ctx.a.lean_readouts:
        cmd.append("--lean-readouts")
    if ctx.a.lean_single:
        if not ctx.a.lean_readouts or not ctx.a.pae_only:
            raise SystemExit(
                "STOP: --lean-single without --lean-readouts and --pae-only reached "
                "val_loss_checkpoint. The parse-time check should have refused this run.")
        cmd.append("--lean-single")
    if ctx.a.val_loss_prep:
        cmd += ["--indices", os.path.join(ctx.a.val_loss_prep, "job_indices.csv"),
                "--bio-dir", os.path.join(ctx.a.val_loss_prep, "job_bioassembly"),
                "--mmcif-dir", os.path.join(ctx.a.val_loss_prep, "job_cifs")]
    # A SINGLE-PROCESS CHILD, SO THE PARENT'S DISTRIBUTED ENVIRONMENT MUST GO. This script
    # reads WORLD_SIZE/RANK/LOCAL_RANK at startup, so an inherited WORLD_SIZE=3 makes the
    # monitor try to join a process group that already has all three of its members -- it
    # hangs on init_process_group holding a card, and the parent is parked at a barrier
    # waiting for it. eval_checkpoint never hit this because phase_k_eval.py reads none of
    # these; this child is phase_k_train.py itself.
    env = {k: v for k, v in os.environ.items()
           if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK",
                        "ROLE_NAME", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE", "MASTER_ADDR",
                        "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT",
                        "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
                        "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE",
                        "TORCH_NCCL_ASYNC_ERROR_HANDLING")
           and not k.startswith("TORCHELASTIC_")}
    env["LAYERNORM_TYPE"] = "openfold"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    if ctx.a.val_loss_card:
        env["CUDA_VISIBLE_DEVICES"] = str(ctx.a.val_loss_card)
    # SHARDED ACROSS THE EVAL CARDS when there is more than one. This pass was a SINGLE
    # single-card child while the other two cards sat at the epoch-boundary barrier: measured
    # on j583i8, 12.8 min of a 46.0 min cycle, against a 24.1 min epoch. The children are
    # independent processes with no collective between them -- deliberately NOT a distributed
    # child, because --loss-only refuses WORLD_SIZE>1 and the comment there records the hang
    # that refusal prevents.
    _cards = [c.strip() for c in (ctx.a.eval_cards or "").split(",") if c.strip()]
    if ctx.a.val_loss_card:
        _cards = [str(ctx.a.val_loss_card)]
    _nsh = len(_cards) if len(_cards) > 1 else 1
    _where = (f"{_nsh} shard(s) over card(s) {','.join(_cards)}" if _nsh > 1
              else ('card ' + str(ctx.a.val_loss_card) if ctx.a.val_loss_card else 'the training card'))
    print(f"  held-out PAE for epoch {epoch} ({_where})", flush=True)
    if _nsh > 1:
        # EACH CHILD WRITES ITS OWN LOG FILE rather than a pipe the parent drains one at a
        # time. With stdout=PIPE and sequential communicate(), a later child that fills its
        # 64 KiB pipe buffer BLOCKS until the parent gets round to reading it -- serialising
        # the very passes this change exists to run in parallel. Files also keep the whole
        # diagnostic instead of the last four lines of a failure. Caught in review.
        os.makedirs(os.path.join(ctx.a.out, f"val_loss_ep{epoch}"), exist_ok=True)

        def _spawn(_i):
            _env = dict(env)
            _env["CUDA_VISIBLE_DEVICES"] = _cards[_i]
            _o = os.path.join(ctx.a.out, f"val_loss_ep{epoch}", f"shard{_i}")
            os.makedirs(_o, exist_ok=True)
            _lg = os.path.join(_o, "shard.log")
            # ROTATE EVERY OUTPUT OF A PREVIOUS ATTEMPT, NOT JUST THE LOG.
            # _flush() opens trajectory.csv with "a" and writes the header only when the file
            # is absent, so a shard that crashed AFTER emitting rows would have the retry
            # append a second copy of the poses it already wrote. The pooling then sees a
            # duplicate pose key and voids the epoch anyway -- meaning the retry would have
            # fixed CUDA-init failures (which happen before any row exists) and silently
            # failed to fix every later one. Caught in review; the five-case test does not
            # touch a real filesystem and could not have seen it.
            # They are rotated, not deleted: a partial trajectory is the evidence of where
            # the attempt died, and "never delete" applies to measurements too.
            _n = 0
            while os.path.exists(os.path.join(_o, f"shard.log.failed{_n or ''}")):
                _n += 1
            _sfx = f".failed{_n or ''}"
            for _old in ([os.path.join(_o, _f) for _f in
                          ("shard.log", "trajectory.csv", "iptm_rows.csv")]
                         + sorted(glob.glob(os.path.join(_o, "crop_sizes_rank*.csv")))):
                if os.path.exists(_old):
                    os.rename(_old, _old + _sfx)
            _cmd = [x for x in cmd]
            _cmd[_cmd.index("--out") + 1] = _o
            # AND THE EMIT PATH, for the same reason --out is rewritten: the emitter opens it
            # with "w", so a shared path means three children truncating one file.
            if "--emit-iptm-rows" in _cmd:
                _cmd[_cmd.index("--emit-iptm-rows") + 1] = \
                    os.path.join(_o, "iptm_rows.csv")
            _cmd += ["--manifest-shard", str(_i), "--manifest-nshards", str(_nsh)]
            _fh = open(_lg, "w")
            # IF Popen RAISES, CLOSE THE HANDLE. Otherwise the fd leaks and, worse, the
            # exception escapes heldout_pae() into the training loop -- a measurement is not
            # allowed to end a run. Caught in review.
            try:
                _p = subprocess.Popen(_cmd, stdout=_fh, stderr=subprocess.STDOUT,
                                      text=True, env=_env)
            except Exception:                 # noqa: BLE001
                try:
                    _fh.close()
                except Exception:             # noqa: BLE001
                    pass
                raise
            return (_i, _o, _lg, _fh, _p)

        def _await(_ps):
            _failed = []
            for _i, _o, _lg, _fh, _p in _ps:
                _p.wait()
                try:
                    _fh.close()
                except Exception:             # noqa: BLE001
                    pass
                if _p.returncode != 0:
                    try:
                        _tail = open(_lg).read().strip().splitlines()[-6:]
                    except OSError:
                        _tail = [f"(no readable log at {_lg})"]
                    _failed.append((_i, _tail))
            return _failed

        # A 2 s STAGGER, AND IT IS MEASURED TO BE NEARLY POINTLESS -- kept only because it
        # costs 4 seconds and cannot hurt. Say so plainly rather than let it read as a fix.
        # On tv6hue epoch 4 two of three children died inside torch._C._cuda_init() -- "CUDA
        # driver initialization failed" and "No CUDA GPUs are available" -- while the third
        # completed normally. Not OOM (that says "CUDA out of memory") and not compute-mode
        # exclusivity (all three cards report Default).
        # WHY THE STAGGER DOES ALMOST NOTHING, measured on a real child: the model build,
        # which is where CUDA is first touched, takes 67.6 s. A child reaches context
        # creation about a minute after it is launched, so a 2 s offset at launch is well
        # inside the jitter and all three can still initialise at the same instant. Raised in
        # review and confirmed by instrumenting the child.
        # The RETRY is what the guarantee rests on, and on the retry path the children are
        # run strictly ONE AT A TIME, which is the only thing that actually removes the
        # concurrency. The first attempt stays parallel because that is the 30 min/epoch path.
        # THE ROOT CAUSE IS STILL NOT ESTABLISHED. A probe reproducing the suspected
        # conditions -- inherited --membind=0 against node-1 GPUs, three contexts created at
        # once -- initialised 36 times out of 36 on this box, so neither NUMA inheritance nor
        # simple concurrency is sufficient on its own. The child now records its affinity,
        # memory policy, RSS, torch version and every card's free memory, so the next
        # occurrence produces evidence instead of another guess.
        _handles = {}
        for _i in range(_nsh):
            if _i:
                time.sleep(2.0)
            _handles[_i] = _spawn(_i)
        _bad = _await(list(_handles.values()))

        # RETRY THE FAILED SHARDS BEFORE VOIDING THE EPOCH. Losing the whole epoch's held-out
        # point to a transient that cost one child 40 seconds is a bad trade: the measurement
        # is 2.1 min of a 30 min cycle, it reads a bundle already written to disk, and the
        # weights cannot move while it runs, so re-running a failed shard measures exactly the
        # same model as its siblings. Before this, ANY shard failure voided the number, and
        # tv6hue epoch 4 is the first time in the project that actually happened.
        # ONLY THE FAILED INDICES ARE RE-RUN. Re-running a shard that already succeeded would
        # throw away good work and give the pooling a second chance to see a duplicate.
        for _try in range(1, int(getattr(ctx.a, "val_loss_shard_retries", 2) or 0) + 1):
            if not _bad:
                break
            _idx = [_i for _i, _ in _bad]
            print(f"  held-out shard(s) {_idx} of epoch {epoch} failed; retry {_try} of "
                  f"{int(getattr(ctx.a, 'val_loss_shard_retries', 2) or 0)}", flush=True)
            for _i, _tail in _bad:
                print(f"    shard {_i} said:\n      " + "\n      ".join(_tail), flush=True)
            time.sleep(15.0 * _try)
            # ONE AT A TIME ON A RETRY, NOT TWO SECONDS APART. A 2 s launch gap does not put
            # 2 s between two CUDA context creations: each child imports torch and builds the
            # model first, and those take long enough to swallow the offset entirely -- which
            # is exactly how three children launched in a loop ended up initialising at the
            # same instant in the first place. Waiting for each retry to EXIT before starting
            # the next is the only way to guarantee no concurrency, and on the retry path the
            # cost is real but small: failures are rare, a shard is ~2.1 min, and 4 extra
            # minutes beats losing the epoch's held-out point. The FIRST attempt stays
            # parallel, because that is the 30 min/epoch path and it almost always works.
            # Raised in review.
            _still = []
            for _i in _idx:
                _handles[_i] = _spawn(_i)
                _still += _await([_handles[_i]])
            _bad = _still
        procs = [_handles[_i] for _i in range(_nsh)]
        if _bad:
            # ANY shard failing voids the epoch's number. Pooling the survivors would report a
            # mean over a BIASED subset of the held-out set -- a different population than
            # every other epoch -- which is worse than no point at all, because it is silently
            # comparable-looking. The next epoch tries again and the gap shows in val_loss.csv.
            for _i, _tail in _bad:
                print(f"  WARNING: held-out shard {_i} of epoch {epoch} failed:\n    "
                      + "\n    ".join(_tail), flush=True)
            print(f"  WARNING: epoch {epoch} has no held-out point; {len(_bad)} of {_nsh} "
                  f"shard(s) failed after "
                  f"{int(getattr(ctx.a, 'val_loss_shard_retries', 2) or 0)} retry/retries and a "
                  f"partial pool would be a different population.", flush=True)
            return None
        # POOLED AS A PLAIN MEAN OVER POSES. Each shard owns whole structures, every pose
        # appears in exactly one shard exactly once, so concatenating the rows and taking one
        # mean IS the pose-weighted figure -- the same identity the unsharded path relies on.
        # EXACT POSE POPULATION, ASSERTED. Arithmetic alone does not make the pooled mean
        # right: a child can exit 0 having measured fewer poses than it owned, and the mean
        # over a short population would look entirely normal while being a different number
        # from every other epoch. So the identities are checked, not just the count.
        _vals, _ids, _cov = [], set(), {}
        for _i, _o, _lg, _fh, _p in procs:
            try:
                with open(_lg) as fh:
                    for ln in fh:
                        if ln.startswith("SHARD COVERAGE "):
                            _cov[_i] = dict(kv.split("=", 1)
                                            for kv in ln.split()[2:] if "=" in kv)
            except OSError:
                pass
            try:
                with open(os.path.join(_o, "trajectory.csv")) as fh:
                    for row in csv.DictReader(fh):
                        if not row.get("pae_loss"):
                            continue
                        _k = (row.get("pdb"), row.get("unit"), row.get("pose"))
                        if _k in _ids:
                            print(f"  WARNING: epoch {epoch} held-out pose {_k} appears in "
                                  f"more than one shard; the shards are not disjoint and the "
                                  f"epoch has no held-out point", flush=True)
                            return None
                        _ids.add(_k)
                        _vals.append(float(row["pae_loss"]))
            except OSError as e:
                print(f"  WARNING: held-out shard {_i} of epoch {epoch} wrote no readable "
                      f"trajectory ({e}); epoch {epoch} has no held-out point", flush=True)
                return None
        if not _vals:
            print(f"  WARNING: the epoch-{epoch} held-out shards have no pae_loss rows",
                  flush=True)
            return None
        if len(_cov) != _nsh:
            print(f"  WARNING: epoch {epoch} got coverage lines from {len(_cov)} of {_nsh} "
                  f"shard(s); cannot prove the population is complete, so no held-out point",
                  flush=True)
            return None
        _alls = {int(c["units_all"]) for c in _cov.values()}
        _nsamp = {int(c["nsample"]) for c in _cov.values()}
        _sum_mine = sum(int(c["units_mine"]) for c in _cov.values())
        if len(_alls) != 1 or len(_nsamp) != 1:
            print(f"  WARNING: epoch {epoch} shards disagree on the full population "
                  f"(units_all={_alls}, nsample={_nsamp}); no held-out point", flush=True)
            return None
        _exp_units, _ns = _alls.pop(), _nsamp.pop()
        if _sum_mine != _exp_units:
            print(f"  WARNING: epoch {epoch} shards cover {_sum_mine} of {_exp_units} unit(s); "
                  f"the population is incomplete, so no held-out point", flush=True)
            return None
        _exp_poses = _exp_units * _ns
        if len(_vals) != _exp_poses or len(_ids) != _exp_poses:
            print(f"  WARNING: epoch {epoch} pooled {len(_vals)} pose(s) ({len(_ids)} unique) "
                  f"against {_exp_poses} expected ({_exp_units} unit(s) x {_ns}); no held-out "
                  f"point", flush=True)
            return None
        # MERGE THE PER-SHARD ipTM FILES, with the same disjointness standard the pose
        # population above is held to. Each shard now writes its own iptm_rows.csv (they used
        # to share one path and truncate each other), so the merged file is the only thing
        # dockq_from_iptm.py can read. A duplicate or an unreadable shard file leaves NO
        # merged file rather than a partial one: dockq_from_iptm refuses duplicates anyway,
        # and a silently short file is exactly the "metric over whatever showed up" it warns
        # against.
        if ctx.a.val_loss_emit_iptm:
            _merged = os.path.join(ctx.a.out, f"val_loss_ep{epoch}", "iptm_rows.csv")
            _rows_i, _keys_i, _bad_i = [], set(), []
            for _i, _o, _lg, _fh, _p in procs:
                _f = os.path.join(_o, "iptm_rows.csv")
                try:
                    with open(_f) as fh:
                        _rd = csv.DictReader(fh)
                        for _r in _rd:
                            _k = (_r["iface"], _r["seed"], _r["sample"])
                            if _k in _keys_i:
                                _bad_i.append(f"duplicate {_k} in shard {_i}")
                                break
                            _keys_i.add(_k)
                            _rows_i.append(_r)
                except OSError as _e:
                    _bad_i.append(f"shard {_i} unreadable ({_e})")
            # COMPLETENESS, NOT JUST UNIQUENESS. Rejecting duplicates says no row appears
            # twice; it says nothing about a row that never appeared. RKL.iptm can return
            # None for a scored interface and the emitter skips that row silently, the child
            # still exits 0, and the merged file comes out SHORT -- which the trajectory
            # coverage check above cannot see, because it counts poses and these are
            # per-interface rows. So the merged key set is checked against the population the
            # pool and the seed range define: every interface in --val-loss-pool must carry
            # exactly (seeds x nsample) rows. Caught in review of 8bcf8202; it is not a
            # blocker while the separate eval still runs, and it is one the moment it does
            # not.
            if not _bad_i:
                try:
                    _pool_if = set()
                    with open(ctx.a.val_loss_pool) as fh:
                        for _r in csv.DictReader(fh):
                            _pool_if.add(f"{_r['pdb']}_{_r['chain1']}{_r['chain2']}")
                    _lo_s, _, _hi_s = ctx.a.val_loss_seeds.partition("-")
                    _nseed = (int(_hi_s) - int(_lo_s) + 1) if _hi_s else 1
                    _exp_depth = _nseed * int(ctx.a.nsample)
                    _depth = {}
                    for _r in _rows_i:
                        _depth[_r["iface"]] = _depth.get(_r["iface"], 0) + 1
                    _miss_if = sorted(_pool_if - set(_depth))
                    _extra_if = sorted(set(_depth) - _pool_if)
                    _short = sorted(k for k, v in _depth.items() if v != _exp_depth)
                    if _miss_if:
                        _bad_i.append(f"{len(_miss_if)} pool interface(s) emitted no rows, "
                                      f"e.g. {_miss_if[:3]}")
                    if _extra_if:
                        _bad_i.append(f"{len(_extra_if)} emitted interface(s) are not in the "
                                      f"pool, e.g. {_extra_if[:3]}")
                    if _short:
                        _bad_i.append(f"{len(_short)} interface(s) do not have the expected "
                                      f"{_exp_depth} row(s) ({_nseed} seed(s) x "
                                      f"{ctx.a.nsample} sample(s)), e.g. "
                                      + ", ".join(f"{k}={_depth[k]}" for k in _short[:3]))
                except (OSError, KeyError, ValueError) as _e:
                    _bad_i.append(f"could not establish the expected ipTM population ({_e})")
            if _bad_i:
                print(f"  WARNING: epoch {epoch} ipTM rows NOT merged: {_bad_i[0]}"
                      + (f" (+{len(_bad_i)-1} more)" if len(_bad_i) > 1 else "")
                      + f"; {_merged} not written", flush=True)
            else:
                with open(_merged, "w", newline="") as fh:
                    _w = csv.writer(fh)
                    _w.writerow(["iface", "seed", "sample", "iptm"])
                    for _r in _rows_i:
                        _w.writerow([_r["iface"], _r["seed"], _r["sample"], _r["iptm"]])
                print(f"  merged {len(_rows_i)} ipTM row(s) from {_nsh} shard(s) "
                      f"-> {_merged}", flush=True)
        _mean = sum(_vals) / len(_vals)
        print(f"epoch {epoch} VALLOSS: pae {_mean:.4f}  poses {len(_vals)}", flush=True)
        _record_val_loss(epoch, step, bundle, _mean, len(_vals))
        return _mean
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        # LOUD, BUT NOT FATAL. Training is the run; a monitor that cannot compute its number
        # must not destroy 37 minutes of epoch because a card was momentarily full. The next
        # epoch tries again and the gap is visible in val_loss.csv.
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        print(f"  WARNING: the epoch-{epoch} held-out PAE pass failed; training continues and "
              f"this epoch has no held-out point:\n    " + "\n    ".join(tail), flush=True)
        return None
    tp = os.path.join(out, "trajectory.csv")
    vals = []
    try:
        with open(tp) as fh:
            for row in csv.DictReader(fh):
                if row.get("pae_loss"):
                    vals.append(float(row["pae_loss"]))
    except OSError as e:
        print(f"  WARNING: the epoch-{epoch} held-out pass wrote no readable trajectory ({e})",
              flush=True)
        return None
    if not vals:
        print(f"  WARNING: the epoch-{epoch} held-out trajectory has no pae_loss rows", flush=True)
        return None
    # EVERY POSE EXACTLY ONCE, so the plain mean IS the pose-weighted figure. Deliberately not
    # the step_taken==1 windows: --accum leaves a tail outside the last complete window, which
    # a training epoch re-traverses next epoch and this single pass never does.
    mean = sum(vals) / len(vals)
    # PARSEABLE, in the same shape as the TRAIN: line the plot already reads. PRINTED BEFORE
    # the csv append, so the number survives any failure in the bookkeeping that follows it.
    print(f"epoch {epoch} VALLOSS: pae {mean:.4f}  poses {len(vals)}", flush=True)
    # RECORDING THE MEASUREMENT MUST NOT BE ABLE TO END THE RUN. A NameError on this exact
    # line killed epoch 1 of x3k2m4 at 09:34 after the pass had already succeeded: `step` was
    # not in scope, the mean was computed, and 78 minutes of training died on the bookkeeping.
    # The subprocess's own trajectory.csv is the durable record either way -- that is where the
    # lost number was recovered from -- so a failure here is reported and dropped.
    _record_val_loss(epoch, step, bundle, mean, len(vals))
    return mean

def eval_checkpoint(ctx, step, epoch, train_pool="", crop_tokens="", tag="", fold="",
                    seed_base=-1, budget=0):
    """One sharded evaluation. tag names its outputs; crop_tokens crops its INPUTS.

    tag EXISTS BECAUSE TWO PASSES CAN SHARE A POOL. A cropped and an uncropped reading of
    valfull are different numbers, and writing both under metric_valfull_ep<N>.json would
    silently overwrite one with the other -- the same class of mistake as the five
    metric_train files that held the only held-out numbers a run produced. The caller names
    each pass and the name reaches the metric file and the shard directory.
    """
    """Save the head, score it, log it. Returns the metric dict.

    With `train_pool`, scores that pool of TRAINING interfaces using the run's own --fold and
    --seeds instead of the validation set, and writes metric_<pool>_ep<N>.json. Everything else
    -- the subprocess, the sharding, the guards -- is the same path, because the two metrics
    have to be comparable and a second implementation would not stay that way.

    RUN AS A SUBPROCESS, not imported. phase_k_eval carries its own guards -- the pool-coverage
    check, the pose-to-sample join by coordinate identity, and the self-check that the SHIPPED
    head reproduces results.csv's own ipTM -- and running it as the script keeps all of them,
    including its refusal to report a number when a join is unverified. It also keeps a second
    model off this process's GPU memory.
    """
    import json
    import subprocess
    cp = os.path.join(ctx.a.out, f"head_ep{epoch}_step{step}.pt")
    if not ctx.is_main:
        return          # the caller already holds the other ranks at a barrier
    torch.save(_head_ckpt_tensors(), cp)
    # NAMED AFTER THE POOL, like the files it writes. "pool@fold" named the FOLD, which is
    # "fold" for every pool in this project and so said nothing; and the flags that drive this
    # are spelled --eval-pool-* precisely so no line anywhere calls an evaluation "train".
    kind = (os.path.basename(os.path.dirname(os.path.abspath(train_pool)).rstrip("/")) or "pool"
            if train_pool else "validation")
    # NAMED AFTER THE POOL, NOT THE MECHANISM. These used to be metric_train_ep<N>.json
    # whatever --train-eval-pool pointed at, so on this arm -- which points it at
    # training/valfull/pool.csv -- five files called "metric_train" held the only HELD-OUT
    # numbers the run produced. A pool is in-sample or it is not and only the pool knows, so
    # the pool's directory names the file: metric_valfull_ep25.json. "heldout" would be the
    # wrong word to standardise on, since --train-eval-pool aimed at the training pool really
    # is in-sample. run_logging/eval_dirs.pool_tag is the reader-side counterpart of this one
    # line and test_eval_dir_names.py asserts the two cannot drift apart.
    _ptag = (os.path.basename(os.path.dirname(os.path.abspath(train_pool)).rstrip("/"))
             or "pool") if train_pool else ""
    # AN EXPLICIT TAG WINS over the pool's directory name, and a cropped primary pass is
    # renamed so it cannot be read as the uncropped one. plot_val_2x3.py reads <pool>, and it
    # must keep getting the deployment-shaped number.
    if tag:
        _ptag = tag
    elif _ptag and crop_tokens:
        _ptag = f"{_ptag}_cropped"
    mj = os.path.join(ctx.a.out, (f"metric_{_ptag}_ep%d.json" if train_pool else "metric_ep%d.json")
                      % epoch)
    extra = []
    if train_pool:
        # THE POOL'S OWN FOLD, which is not always this run's. Scoring a validation subset
        # here means --fold, --results, --roles and --seed-base all have to come from that
        # fold, not from the one being trained on: a results.csv from the wrong fold would
        # look up DockQ for interfaces it does not contain and the coverage check would fail,
        # and a wrong --seed-base would ask for seeds the fold never wrote.
        _tef = fold or ctx.a.train_eval_fold or ctx.a.fold
        _seed_base = (seed_base if seed_base >= 0
                      else (ctx.a.train_eval_seed_base if ctx.a.train_eval_seed_base >= 0
                            else ctx._train_eval_seed_base))
        # THE ROLES TABLE CAN ALSO LIVE BESIDE THE POOL, same rule as the seed manifest
        # below and for the same reason. The fold's own chain_roles.csv names only the 53
        # structures of the original valfull wave, so an 83-structure pool spanning BOTH
        # waves leaves 49 of its 141 interfaces unmappable and phase_k_eval stops on them.
        # A pool directory describes its own pool: pool.csv, chain_roles.csv, seed_manifest
        # .json. Falling back to the fold keeps every existing pool working untouched.
        _prd = os.path.dirname(os.path.abspath(train_pool))
        _srol = os.path.join(_prd, "chain_roles.csv")
        _roles = _srol if os.path.exists(_srol) else os.path.join(_tef, "chain_roles.csv")
        extra = ["--fold", _tef,
                 "--results", os.path.join(_tef, "results.csv"),
                 "--pool", train_pool,
                 "--roles", _roles,
                 "--seed-base", str(_seed_base)]
        # AND WHICH SEEDS THAT POOL KEPT. A trimmed fold keeps 30 of a 100-wide sweep, chosen
        # per structure, so the evaluator's contiguous window asks for seeds the fold never
        # wrote and its coverage guard stops on them -- which is exactly what killed the
        # epoch-5 valfull eval. The manifest sits beside the POOL, not under the fold, because
        # it describes what this pool's metric claims. Forwarded only when it exists, so an
        # untrimmed pool keeps the contiguous behaviour.
        _sman = os.path.join(_prd, "seed_manifest.json")
        if os.path.exists(_sman):
            extra += ["--seed-manifest", _sman]
    # THE EVALUATOR MUST RUN THE SAME FUNCTION WE TRAINED. Without this it builds a head with
    # no frame_dir_proj, and its own guard refuses the checkpoint for holding a tensor the
    # model does not have -- a loud failure rather than a silently different metric, but a
    # failure either way. Appended outside the train_pool branch so it reaches the validation
    # path too; both the sharded driver (--extra) and the single-card path forward `extra`.
    if ctx.a.frame_direction:
        extra = extra + ["--frame-direction"]
    if ctx.a.pose_perblock:
        extra = extra + ["--pose-perblock"]
        if ctx.a.pose_perblock_tied:
            extra = extra + ["--pose-perblock-tied"]
        if ctx.a.pose_perblock_share:
            extra = extra + ["--pose-perblock-share"]
        # AND THE COPY MODE. This one ADDS tensors, so omitting it makes the evaluator's
        # state-dict guard refuse outright rather than silently score a different
        # architecture -- but passing it is still the fix, not relying on the refusal.
        if ctx.a.pose_perblock_copy:
            extra = extra + ["--pose-perblock-copy"]
    # AND THE IN-BLOCK PATHWAY, which had no evaluator counterpart at all until review found
    # it. The block index goes too: pose_inblock_d* carry no depth in their names, so an
    # evaluator that installed at a DIFFERENT block would pass the key-set guard and score a
    # different operator. --lean-single is deliberately NOT forwarded: it sets c_s = 0 on
    # built blocks, leaves every module in the tree, and so changes neither the checkpoint's
    # key set nor the PAE path's mathematics -- the evaluator has no such flag and needs none.
    if ctx.a.pose_inblock:
        extra = extra + ["--pose-inblock",
                         "--pose-inblock-block", str(ctx.a.pose_inblock_block)]
        if ctx.a.pose_inblock_pae:
            extra = extra + ["--pose-inblock-pae"]
    # THE THREE OPTIMISATIONS THE EVALUATOR ALREADY HAS, which this path never passed.
    #
    # --join-cache is the big one. The pose->sample join is a function of the trunk's own
    # coordinates and the pose cifs on disk, so for a fixed pool it returns the same answer
    # every time -- and phase_k_eval_sharded's docstring measured it at 3.6 min per shard.
    # Recomputing a constant on every eval is most of why a 10-structure pool took 8.9 min
    # when its 156 GB of trunks is only ~1.7 min of I/O. The cache key carries each trunk's
    # size and mtime, so a refold or rebuild invalidates it rather than silently reusing a
    # mapping to superseded poses -- which is the exact failure resolve_seed_dir exists to
    # prevent. It lives beside the POOL, not in the run directory, so the first eval of the
    # first arm pays for it and every later eval and every later arm is free.
    #
    # --pair-iptm-only computes chain_pair_iptm straight from PAE instead of building the
    # whole Protenix summary and discarding every other field. Measured bit-identical.
    # --prefetch overlaps the next unit's trunk read with the current unit's compute.
    _jc = os.path.join(os.path.dirname(os.path.abspath(train_pool or ctx.a.train_eval_pool or ".")),
                       "join_cache.json") if train_pool else os.path.join(ctx.a.out, "join_cache.json")
    extra = extra + ["--join-cache", _jc, "--pair-iptm-only", "--prefetch"]
    # THE EVALUATOR MUST SELECT BY WHAT THE RUN IS TRAINING. --rank-pool mean optimises the
    # mean-pooled ipTM, and this pass selected by the MAX -- so the automatic validation curve
    # measured a different rule from the objective and neither number meant what it said.
    # Caught in review. --pool-mean also makes the evaluator report the shipped MAX selection
    # from the same pass, so the baseline is preserved rather than traded away: both appear as
    # mean_dockq / mean_dockq_select_max in its json.
    if ctx.a.rank_pool == "mean" and ctx.a.rank_loss:
        extra = extra + ["--pool-mean", "--select-by", "mean"]
    # THE CROP, FORWARDED. phase_k_eval's --crop-tokens takes the same npz --crop-tokens does
    # here, so a cropped reading of a pool is the evaluator's own code path and not a
    # reimplementation. Absent, the pass is uncropped, which is what every pass has been.
    if crop_tokens:
        extra = extra + ["--crop-tokens", os.path.abspath(crop_tokens)]
    # SHARDED OVER EVERY CARD THE RUN WAS GIVEN. The other ranks are parked at a barrier for
    # the whole evaluation, so their cards are idle: at 73 min a pass, evaluating on one card
    # wasted ~47% of the run's wall clock with two cards doing nothing. The driver puts one
    # shard on each card -- ~6 GB beside a parked rank's ~6 GB on a 98 GB card -- and merges.
    cards = ctx.a.eval_cards or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    shard_work = os.path.join(ctx.a.out, (f"{_ptag}_eval_shards_ep%d" if train_pool
                                       else "eval_shards_ep%d") % epoch)
    budget = (budget if budget else
              (ctx._train_eval_budget if train_pool else ctx.a.eval_budget))
    if len([c for c in cards.split(",") if c.strip()]) > 1:
        cmd = [sys.executable, "-I",
               os.path.join(os.path.dirname(os.path.abspath(ctx.__file__)),
                            "phase_k_eval_sharded.py"),
               "--cards", cards, "--ckpt", cp, "--budget", str(budget),
               "--work", shard_work, "--out", mj]
        if extra:
            cmd += ["--extra", " ".join(extra)]
    else:
        cmd = [sys.executable, "-I", ctx.a.eval_script, "--ckpt", cp,
               "--budget", str(budget), "--out", mj, *extra]
    env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
    sharded = "phase_k_eval_sharded.py" in cmd[2]
    if sharded:
        print(f"  evaluating epoch {epoch} ({kind}) sharded over GPU(s) {cards}")
    elif ctx.a.eval_device:
        # ONLY ON THE SINGLE-CARD PATH. Pinning CUDA_VISIBLE_DEVICES here would hand the sharding
        # driver one card and silently undo the split.
        env["CUDA_VISIBLE_DEVICES"] = str(ctx.a.eval_device)
        print(f"  evaluating epoch {epoch} on GPU {ctx.a.eval_device} (the trainer keeps its card)")
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
        if free_mib is not None and free_mib < ctx.a.eval_min_free_mib:
            raise SystemExit(
                f"STOP: the epoch-{epoch} evaluation would share the training card with only "
                f"{free_mib} MiB free, under --eval-min-free-mib {ctx.a.eval_min_free_mib}. It "
                f"loads a second 464M-parameter model (~16 GB) while the trainer stays "
                f"resident. Pass --eval-device <idle gpu>, or raise the floor deliberately.")
        print(f"  evaluating epoch {epoch} on the TRAINING card, {free_mib} MiB free")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        raise SystemExit(f"STOP: the epoch-{epoch} {kind} evaluation failed, so this "
                         f"checkpoint cannot be scored and a training curve without it is not "
                         f"interpretable:\n  " + "\n  ".join(tail))
    m = json.load(open(mj))
    if train_pool:
        print(f"  epoch {epoch} TRAIN: mean_dockq {m['mean_dockq']:.4f}  "
              f"frac_acc {m['frac_acc']:.4f}  oracle {m['oracle_mean_dockq']:.4f} / "
              f"{m['oracle_frac_acc']:.4f}  headroom {m['headroom']:.4f}  "
              f"({m['n_interfaces']} interfaces)")
    else:
        print(f"  epoch {epoch}: mean_dockq {m['mean_dockq']:.4f} (baseline 0.5775)  "
              f"frac_acc {m['frac_acc']:.4f} (baseline 0.7283)  headroom {m['headroom']:.4f}")
    return m
