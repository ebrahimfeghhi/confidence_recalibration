"""`trainer/loop.py` -- the epoch and optimiser-step loop.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 7300-8276 (`train_loop`) (source `d34baebc`), dedented one
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

def train_loop(ctx, steps_cap, lr, tag, write_traj, from_epoch=0, from_step=0):
    for g in ctx.opt.param_groups:
        g["lr"] = lr
    rows, t0, step, n_in_accum = [], time.time(), from_step, 0
    ep_loss = {"sum": 0.0, "n": 0}
    stop_state = {"best": float("inf"), "bad": 0}
    ctx.opt.zero_grad(set_to_none=True)
    # LOCAL SUMS OVER THE WINDOW, reduced once per step. Summed rather than averaged so the
    # reduction is exact regardless of how unevenly the ranks are filled.
    batch_sum = {k: 0.0 for k in ctx.TERMS}

    def _batch_means():
        """Mean of each loss term over EVERY pose in this step, across all ranks, and the count.

        ONE COLLECTIVE for the terms and the count together: len(TERMS)+1 floats, against the
        79 MB gradient buffer already reduced every step, so the cost is not measurable. The
        count is reduced rather than assumed to be n_local * world -- the shards are equal
        length so it should be, but a number the curve depends on should not be taken on trust.

        Resets the accumulator, so it must be called exactly once per step.
        """
        nonlocal n_in_accum
        vals = [batch_sum[k] for k in ctx.TERMS] + [float(n_in_accum)]
        if ctx.world > 1:
            t = torch.tensor(vals, dtype=torch.float64)
            if ctx.a.dist_backend == "gloo":
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            else:
                t = t.to(ctx.dev)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t = t.cpu()
            vals = [float(x) for x in t]
        n_tot = int(round(vals[-1]))
        # THE EPOCH'S LOSS, ACCUMULATED HERE because `vals` is already reduced over every rank
        # and is therefore bit-identical on all of them. Summing it anywhere rank-local would
        # give each rank its own epoch mean, and an early-stop decision that differs between
        # ranks leaves some in the next epoch and the rest at a barrier.
        ep_loss["sum"] += vals[ctx.TERMS.index("loss")]
        ep_loss["n"] += n_tot
        for k in batch_sum:
            batch_sum[k] = 0.0
        n_in_accum = 0
        means = {"batch_" + k: (vals[i] / n_tot if n_tot else None)
                 for i, k in enumerate(ctx.TERMS)}
        means["batch_n"] = n_tot
        return means, n_tot

    def _apply_partial(ctx=None):
        ctx = ctx or {}
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
        nonlocal step
        # The count comes back from the same collective that reduces the loss sums, so the
        # partial step is logged with the same numbers it was taken on.
        means, n_tot = _batch_means()
        if n_tot <= 0:
            return
        scale = ctx._step_poses_g[0] / n_tot
        tset = set(ctx.trainable)
        for _n, pp in ctx.model.named_parameters():
            if _n in tset and pp.grad is not None:
                pp.grad.mul_(scale)
        print(f"  flushing a partial accumulation of {n_tot}/{ctx._step_poses_g[0]} pose(s), gradient "
              f"scaled by {scale:.4f} so the step is the same size as every other")
        _gn, _outside, ok = _apply_step(step)
        # LOGGED, because this step used to leave no row at all: the reader then spliced the
        # epoch's trailing poses into the next epoch's first window and dropped the run's final
        # remainder. partial_step marks it; batch_n says how many poses it covered.
        rows.append({"wall_s": round(time.time() - t0, 1), "step": step,
                     "unit_i": ctx.get("unit_i", -1), "pdb": ctx.get("pdb", ""),
                     "unit": ctx.get("unit", ""), "pose": ctx.get("pose", -1),
                     "lr": ctx.get("lr", float("nan")), "grad_norm": _gn,
                     "outside_head": _outside, "epoch": ctx.get("epoch", 0),
                     "step_taken": 1 if ok else 0, "nonfinite_grad": 0 if ok else 1,
                     "partial_step": 1, "step_ok": 1 if ok else 0, **means})
        if ok:
            step += 1
    def _save_resume(epochs_done, step_now):
        """Everything needed to continue: the head, AdamW's moments, the step, and a guard.

        UNDER --loss-only THIS WRITES NOTHING. AdamW never stepped, so the moments are the
        resumed bundle's or empty, and a bundle carrying them under a new epoch number would
        be a forgery of training that did not happen.

        THE OPTIMISER STATE IS THE POINT. AdamW carries a per-parameter exp_avg and exp_avg_sq,
        which is what makes its steps adaptive; restoring weights alone restarts those from zero
        and the first steps after a resume are effectively un-adapted. It roughly triples the
        file -- 76 MiB of weights, ~158 MiB of moments -- which is nothing against the 1.4 TB of
        trunks this run reads.

        The guard records what determines batch composition, so a resume into different settings
        is refused rather than silently training with moments that belong elsewhere.
        """
        if ctx.a.loss_only:
            return
        p = os.path.join(ctx.a.out, f"resume_ep{epochs_done}.pt")
        tmp = p + ".tmp"
        torch.save({"head": {n: pp.detach().cpu() for n, pp in ctx.model.named_parameters()
                             if n in set(ctx.trainable)},
                    "opt": ctx.opt.state_dict(),
                    "epoch": epochs_done, "step": step_now,
                    "guard": {"drift_history": ctx._drift_history,
                              "lora_rank": ctx.a.lora_rank, "lora_alpha": ctx.a.lora_alpha,
                              "lora_targets": ctx.a.lora_targets,
                              "lora_dropout": ctx.a.lora_dropout,
                              "train_only": ctx.a.train_only,
                              "world": ctx.world, "seed": ctx.a.seed, "accum": ctx.a.accum,
                              "nsample": ctx.a.nsample, "lr": ctx.a.lr,
                              "lr_final": ctx.a.lr_final,
                              # ALSO HALF-WIRED, and this one decides the lr at every step of
                              # the cosine arm: the decay window and shape were compared but
                              # never written, so a bundle moved without its args.json fell
                              # through to the pre-crop defaults (0, 0, "cosine") and a resume
                              # would silently run a WHOLE-RUN cosine instead of the 20-50
                              # window it was configured for. Found by test_guard_fields.py
                              # the first time it ran, not by reading the diff.
                              "lr_decay_start_epoch": ctx.a.lr_decay_start_epoch,
                              "lr_decay_end_epoch": ctx.a.lr_decay_end_epoch,
                              "lr_decay_shape": ctx.a.lr_decay_shape,
                              "lr_steps": ctx.a.lr_steps,
                              "pair_dropout": ctx.a.pair_dropout,
                              "token_keep_union": ctx.a.token_keep_union,
                              "token_keep_other": ctx.a.token_keep_other,
                              "warmup_steps": ctx.a.warmup_steps, "epochs": ctx.a.epochs,
                              "manifest_skip": ctx.a.manifest_skip,
                              "manifest_digest": ctx._manifest_digest,
                              "crop_size": ctx.a.crop_size,
                              "alpha_confidence": ctx.a.alpha_confidence,
                              "pae_scored_pairs": ctx.a.pae_scored_pairs,
                              "pae_only": ctx.a.pae_only,
                              "lean_single": ctx.a.lean_single,
                              "pose_inblock": ctx.a.pose_inblock,
                              "pose_inblock_pae": ctx.a.pose_inblock_pae,
                              "pose_inblock_block": ctx.a.pose_inblock_block,
                              # SAVED, NOT ONLY COMPARED. frame_direction went into the
                              # comparison side but not here, so a bundle carried no record of
                              # it: a resume beside its args.json still matched, because the
                              # guard falls back to that file, but a bundle moved on its own
                              # fell through to the pre-crop default and read as C DISABLED.
                              # Third time a guard field has been half-wired. Found in review.
                              "frame_direction": ctx.a.frame_direction,
                              "pose_perblock": ctx.a.pose_perblock,
                              "pose_perblock_tied": ctx.a.pose_perblock_tied,
                              # FOURTH TIME, five fields at once. pose_perblock_share, l2_sp
                              # and the three ranking fields were all in the comparison side
                              # and none of them here, so a bundle moved without its args.json
                              # carried no record of the architecture OR the objective. Found
                              # in review of 04503a9.
                              "pose_perblock_share": ctx.a.pose_perblock_share,
                              "pose_perblock_copy": ctx.a.pose_perblock_copy,
                              "l2_sp": ctx.a.l2_sp,
                              "l2_sp_exclude": ctx.a.l2_sp_exclude,
                              # FIFTH TIME. The three row-weight fields went into the
                              # comparison side and not here, so a bundle moved without its
                              # args.json would carry no record of the PAE objective and read
                              # as UNWEIGHTED. Found in review of c431fb93. The pattern is now
                              # tested rather than re-noticed: test_guard_fields.py asserts
                              # every compared field is also saved, so a sixth is a failing
                              # gate instead of a comment.
                              "pae_row_frac": ctx.a.pae_row_frac,
                              "pae_row_mass": ctx.a.pae_row_mass,
                              "pae_row_floor": ctx.a.pae_row_floor,
                              "rank_loss": ctx.a.rank_loss, "alpha_rank": ctx.a.alpha_rank,
             "rank_pool": ctx.a.rank_pool, "rel_labels": ctx.a.rel_labels,
             "rel_standardize": ctx.a.rel_standardize,
                              "rank_weight_by_gap": ctx.a.rank_weight_by_gap,
                              "rank_form": ctx.a.rank_form,
                              "rank_tau": (float(ctx.a.rank_tau)
                                           if ctx.a.rank_tau is not None else -1.0),
                              "rank_cross_seed": ctx.a.rank_cross_seed,
                              "rank_structs_per_step": ctx.a.rank_structs_per_step,
                              # FIFTH TIME. rank_pairs, rank_within_weight and rank_topk_decay
                              # went into the comparison side with a comment claiming both
                              # sides were edited together; they were not, and
                              # test_guard_fields.py said so. Found in review.
                              "rank_pairs": ctx.a.rank_pairs,
                              "rank_within_weight": ctx.a.rank_within_weight,
                              "rank_topk_decay": ctx.a.rank_topk_decay,
                              "rank_straddle": ctx.a.rank_straddle,
                              "alpha_pae": ctx.a.alpha_pae, "drop_pde": ctx.a.drop_pde,
                              "clip_norm": ctx.a.clip_norm,
                              "crop_margin": ctx.a.crop_margin,
                              "crop_tokens": ctx._CROP_TOK_DIGEST,
                              "crop_contact": ctx.a.crop_contact,
             # THE CROP DEFINITION, for the same reason as the two above: a16911de changed
             # the DEFAULT, so resuming a pre-change bundle under post-change code blends two
             # crops into one curve unless the guard compares it. Benign only while pose 0
             # equals the union; the guard's standard is that the setting is compared, not
             # that it currently happens to be a no-op.
             "crop_all_poses": ctx.a.crop_all_poses,
                              "commit": ctx._git_commit(), "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}},
                   tmp)
        # ATOMIC: a bundle half-written by a crash at the epoch boundary would be the one thing
        # a reader reaches for, so it is renamed into place only once it is complete.
        os.replace(tmp, p)
        print(f"  saved {p} ({os.path.getsize(p)/2**20:.0f} MiB: head + AdamW moments + step)",
              flush=True)

    # AN EPOCH LOOP, BECAUSE --epochs USED TO DO NOTHING. It scaled steps_cap while the loop
    # traversed `flat` exactly once, so a request for more steps than the data provides simply
    # stopped early -- review round 7 asked for 5 optimiser steps and got 2. The data is now
    # re-traversed, reshuffled each epoch so the order is not identical, until the step cap or
    # the epoch count is reached.
    n_epochs = max(1, math.ceil(ctx.a.epochs))
    done = False
    for ep in range(from_epoch, n_epochs):
        if done:
            break
        # THE EPOCH ORDER, AND WITH --rank-cross-seed ITS STEP SHAPE TOO. The cross-seed
        # builder returns a per-step unit quota because S structures x their seeds need not
        # divide `world`, so the step boundary cannot be the constant accum_local.
        ep_quotas = None
        if ctx.a.rank_cross_seed:
            ep_units, ep_quotas, _xs_nunits = ctx.build_epoch_cross_seed(ep)
            # EVERY STEP HOLDS THE SAME TOTAL unit count across ranks -- only the per-rank
            # split rotates -- so the global pose count per step is a constant.
            ctx._step_poses_g[0] = _xs_nunits * ctx.a.nsample
        else:
            ep_units = build_epoch(ep)
        if ctx.world > 1:
            # THE RANKS MUST AGREE ON THE UNIT COUNT. If they do not, they take a different
            # number of optimiser steps and the first mismatched all-reduce hangs forever
            # rather than failing, so it is checked here where it can still be reported.
            _t = torch.tensor([len(ep_units)], device=ctx.dev)
            _mx, _mn = _t.clone(), _t.clone()
            dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
            dist.all_reduce(_mn, op=dist.ReduceOp.MIN)
            if int(_mx) != int(_mn):
                raise SystemExit(f"STOP: rank unit counts disagree ({int(_mn)}..{int(_mx)}). "
                                 f"The ranks would take different numbers of steps and deadlock "
                                 f"on the first all-reduce.")
        # a fractional last epoch takes that fraction of the units
        if ep == n_epochs - 1 and ctx.a.epochs < n_epochs:
            frac = ctx.a.epochs - (n_epochs - 1)
            ep_units = ep_units[:max(1, int(len(ep_units) * frac))]
        print(f"  epoch {ep + 1}/{n_epochs}: {len(ep_units)} unit(s)")
        # BEFORE _pf_start, NOT AFTER. The token draw is seeded from this cell and the
        # prefetch workers crop units in their own threads from the moment they start, so
        # setting it afterwards left those units drawn against the PREVIOUS epoch's value
        # while the inline path used the current one -- different token sets for poses of the
        # same structure, which is exactly the invariant the draw exists to hold. Caught by
        # test_token_dropout.py, which pins this ordering.
        ctx._crop_epoch[0] = ep
        ctx._tok_drop_stat["kept"] = ctx._tok_drop_stat["tot"] = 0.0
        ctx._tok_drop_stat["n"] = ctx._tok_drop_stat["topped"] = 0
        ctx._tok_drop_stat["topped_tok"] = 0
        # STARTED PER EPOCH, over THIS epoch's shuffled order, so the worker walks the units
        # in the order the main loop will consume them.
        _pf_start(ep_units)
        # PER EPOCH, like _pf_stat and for the same reason: the printed line reports an epoch,
        # and a run-cumulative counter printed as one would be a different number every time.
        for _k in ctx._tr_stat:
            ctx._tr_stat[_k] = 0
        _T0, _TN0 = dict(ctx._T), dict(ctx._TN)
        # PEAK MEMORY PER EPOCH, reset here so each epoch reports its own rather than the
        # high-water mark of the whole run. This is the number that decides whether activation
        # checkpointing can stay off: --mem-log answers it too but emits one line per pose,
        # 8,100 per rank per epoch.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ctx._mem_ep[0] = 0
        ep_start_step = step
        ep_t0 = time.time()
        ep_loss["sum"], ep_loss["n"] = 0.0, 0
        # UNIT INDEX -> (step index, the step's whole unit list), so pass A can score a
        # step's poses before the first backward of that step. Built once per epoch.
        _xs_steps, _xs_at = [], {}
        if ep_quotas:
            _o = 0
            for _k, _q in enumerate(ep_quotas):
                if _o >= len(ep_units):
                    break
                _xs_steps.append(ep_units[_o:_o + _q])
                _xs_at[_o] = _k
                _o += _q
            if _o != len(ep_units):
                raise SystemExit(
                    f"STOP: the cross-seed quotas sum to {_o} unit(s) but the epoch holds "
                    f"{len(ep_units)}. The step boundary would drift from the quota and the "
                    f"ranks would step at different times.")
        for ui, (pdb, lab_p, cands) in enumerate(ep_units):
            if ui in _xs_at:
                ctx._step_poses[0] = len(_xs_steps[_xs_at[ui]]) * ctx.a.nsample
                # BEFORE PASS A, so the warmer is already reading unit 1 while pass A is on
                # the GPU with unit 0. It also drops the PREVIOUS step's arrays, which is what
                # bounds the cache to one step.
                _tr_step_start(_xs_steps[_xs_at[ui]])
                # BEFORE PASS A AND AFTER THE OPTIMISER STEP THAT ENDED THE LAST ONE. Both
                # halves matter: the entries of the step that just finished are dead (its poses
                # never recur), and this is also where the cache re-verifies that nothing moved
                # the frozen prefix -- an optimiser step is the only thing that could have.
                if ctx._pfc is not None:
                    ctx.PFC.step_reset()
                # THE DEAD-OP RE-CHECK IS DELIBERATELY *NOT* HERE. This site is inside
                # `if ui in _xs_at:`, which is only reached under --rank-cross-seed, so a
                # check placed here runs zero times on every other arm -- silently. It lives
                # immediately after opt.step() in _apply_step instead, which is unconditional
                # and is the actual instant a stubbed tensor could have been written.
                _xs_prepare(_xs_steps[_xs_at[ui]], ep)
            trunk_p = resolve_trunk(pdb, lab_p, cands)
            # THE NEXT UNIT'S TRUNK, WARMED WHILE THIS ONE COMPUTES. Resolving it is cheap
            # after epoch 1 because the fingerprint is cached; on the first epoch it pays the
            # hash it would have paid a moment later anyway.
            if ui + 1 < len(ep_units):
                _n_pdb, _n_lab, _n_cands = ep_units[ui + 1]
                _t_pf = time.perf_counter()
                # EXCLUSIVE OF THE HASH NESTED INSIDE IT. resolve_trunk computes
                # unit_fingerprint, which accumulates into fingerprint_hash -- and because the
                # prefetch reaches unit i+1 before the main loop does, it pays EVERY first-touch
                # hash. Epoch 1 of x3k2m4 therefore reported fingerprint_hash 8.1m and
                # prefetch_issue 8.1m as if they were separate costs when they are the same
                # 8.1 minutes; across ranks the pairs were 8.1/8.1, 7.6/7.6, 8.5/8.6. The
                # budget subtracted it twice and understated the residual by ~8 min.
                _h0 = ctx._T["fingerprint_hash"]
                try:
                    _prefetch(resolve_trunk(_n_pdb, _n_lab, _n_cands))
                except SystemExit:
                    # A BAD NEXT UNIT MUST NOT KILL THE CURRENT ONE HERE. resolve_trunk raises
                    # SystemExit on an unresolvable fingerprint, which is the right behaviour
                    # when that unit is reached for real -- one iteration later, with its own
                    # diagnostics. Swallowing it here only declines to prefetch.
                    pass
                _tick("prefetch_issue", _t_pf)
                ctx._T["prefetch_issue"] -= ctx._T["fingerprint_hash"] - _h0
            for _m in ctx.run_unit(pdb, lab_p, trunk_p):
                i, met = _m["pose"], _m      # floats only; the tensors are already freed
                # THE SCHEDULE LIVES IN phases/lr_schedule.py so it can be asserted before
                # a sixty-epoch run rather than inferred after one. Driven by `step`, which
                # resume restores, so a resumed run picks the schedule up where it left off.
                # _lr_horizon[0] is steps-per-epoch x epochs, derived from the EPOCH BUILDER's
                # grouping -- see the comment at its assignment for the 205-vs-184 bug.
                cur_lr = ctx._LRS.lr_at_step(
                    step, lr, ctx.a.lr_final, ctx.a.warmup_steps,
                    ctx._lr_horizon[0] / max(1.0, float(ctx.a.epochs)), ctx.a.epochs,
                    ctx.a.lr_decay_start_epoch, ctx.a.lr_decay_end_epoch, ctx.a.lr_decay_shape,
                    ctx._LR_STEPS)
                for g in ctx.opt.param_groups:
                    g["lr"] = cur_lr
                n_in_accum += 1
                # EVERY POSE CONTRIBUTES TO THE WINDOW SUM, which is what the batch columns
                # average. The per-pose TERMS stay in the row too, for provenance.
                for _k in ctx.TERMS:
                    batch_sum[_k] += float(met[_k])
                gn, outside, took, bad_grad = float("nan"), -1, 0, 0
                means = {"batch_" + _k: None for _k in ctx.TERMS}
                means["batch_n"] = 0
                if n_in_accum >= ctx._step_poses[0]:
                    # REDUCED BEFORE THE STEP IS TAKEN, so the numbers logged are the numbers
                    # the step was taken on even if the step is then refused. _batch_means
                    # resets the accumulator and n_in_accum, so neither is reset again below.
                    # TIMED SEPARATELY, because it is the step's FIRST collective and it was
                    # invisible. _batch_means all_reduces the loss statistics before
                    # _apply_step touches gradients, so an early-arriving rank blocks HERE, not
                    # at the gradient reduction -- and none of that wait appeared in
                    # grad_collective_plus_wait. That is why 0.6 min of 77.9 could not be read
                    # as "synchronisation is cheap": the expensive-looking collective was timed
                    # and the one that absorbs the straggler was not.
                    _t_bm = time.perf_counter()
                    means, _n_tot = _batch_means()
                    _tick("loss_stats_collective_plus_wait", _t_bm)
                    _t_as = time.perf_counter()
                    gn, outside, ok = _apply_step(step)
                    _tick("apply_step_total", _t_as)
                    took = 1
                    if not ok:
                        bad_grad = 1
                        rows.append({"wall_s": round(time.time() - t0, 1), "step": step,
                                     "unit_i": ui, "pdb": pdb,
                                     "unit": os.path.basename(lab_p)[:-4], "pose": i,
                                     "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                                     "epoch": ep + 1, "step_taken": 0, "nonfinite_grad": 1,
                                     "partial_step": 0, "step_ok": 0,
                                     **{k: met[k] for k in ctx.TERMS},
                                     **({"pae_loss_unweighted":
                                         met.get("pae_loss_unweighted", float("nan"))}
                                        if float(ctx.a.pae_row_frac) > 0 else {}),
                                     **means})
                        continue
                    step += 1
                    if ctx.a.save_every and step % ctx.a.save_every == 0 and write_traj:
                        cp = os.path.join(ctx.a.out, f"head_step{step}.pt")
                        torch.save(_head_ckpt_tensors(), cp)
                        print(f"  saved {cp}")
                rows.append({"wall_s": round(time.time() - t0, 1), "step": step, "unit_i": ui,
                             "pdb": pdb, "unit": os.path.basename(lab_p)[:-4], "pose": i,
                             "lr": cur_lr, "grad_norm": gn, "outside_head": outside,
                             "epoch": ep + 1, "step_taken": took,
                             "nonfinite_grad": bad_grad,
                             "partial_step": 0, "step_ok": 1 if took else 0,
                             **{k: met[k] for k in ctx.TERMS},
                             **({"pae_loss_unweighted":
                                 met.get("pae_loss_unweighted", float("nan"))}
                                if float(ctx.a.pae_row_frac) > 0 else {}),
                             **means})
                # EVERY RANK FLUSHES ITS OWN FILE; only rank 0 writes trajectory.csv.
                # The condition used to be `write_traj and ...`, which is is_main, so ranks 1+
                # never flushed AND never cleared `rows`: they accumulated the whole epoch in
                # memory and wrote nothing. That is the same defect the crop log was fixed for
                # in review round 6, and it is why every per-pose figure described a third of
                # the run.
                if len(rows) % 20 == 0:
                    _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
                    if write_traj:
                        _flush(ctx.traj_p, ctx.traj_cols, rows)
                    rows = []
                # THE CROP LOG FLUSHES ON EVERY RANK, and with the trajectory rather than only
                # at the epoch boundary: a run that dies mid-epoch -- as this one did, on a
                # hangup -- would otherwise leave no record of what it had cropped. Outside the
                # write_traj branch because crop_p is per-rank and every rank owns its file.
                if ctx.crop_p and len(ctx.crop_rows) >= 20:
                    _flush(ctx.crop_p, ctx.crop_cols, ctx.crop_rows); ctx.crop_rows.clear()
                if steps_cap and step >= steps_cap:
                    done = True
                    break
            # --grad-probe: one unit done. Report after N of them and stop; this mode exists to
            # measure the gradient scale, not to train.
            if ctx.a.grad_probe:
                ctx._probe["units"] += 1
                # EXACTLY ONE WINDOW. accum_local poses close the window and the optimiser
                # steps; one unit further and the probe is summing g(theta_0) with
                # g(theta_1), which is not the gradient of any single step. Found in review
                # of 523013a, where the launcher asked for 8 units against a 6-unit window.
                # THE WINDOW IS THE STEP, AND THE STEP IS NO LONGER A CONSTANT. accum_local
                # // nsample is the unit count per step only when every step holds the same
                # number; --rank-cross-seed rotates a 7/7/6 split, so the window has to come
                # from _step_poses[0], which the epoch loop sets per step. Measuring one unit
                # past the boundary sums g(theta_0) with g(theta_1), which is the gradient of
                # no single step -- the defect review found in 523013a.
                if ctx._probe["units"] >= max(1, ctx._step_poses[0] // max(1, ctx.a.nsample)):
                    # RAISES on failure, so the launcher's exit status carries the verdict.
                    _grad_probe_report(ctx._probe, ctx.trainable, ctx.a, step > 0, ctx.accum_local, ctx.world)
                    ctx._probe["done"] = True
                    return
            if done:
                break
        # HOW LONG THE EPOCH TOOK, logged rather than left to be derived. Timing has been
        # estimated wrong four times on this pilot, every time from a median or a short probe
        # treated as a mean; an epoch that reports its own duration cannot be misquoted.
        _ep_min = (time.time() - ep_t0) / 60.0
        _ep_poses = step - ep_start_step
        print(f"  epoch {ep + 1} took {_ep_min:.1f} min for {len(ep_units)} unit(s), "
              f"{_ep_poses} optimiser step(s)"
              + (f", {_ep_min * 60 / max(1, len(ep_units) * ctx.a.nsample):.2f} s/pose"
                 if ep_units else ""))

        # THE PARTIAL ACCUMULATION IS FLUSHED HERE, BEFORE THE EARLY-STOP CHECK AND BEFORE
        # THE EVALUATION, SO THE HEAD THAT IS SCORED IS THE HEAD THAT IS SAVED AND THE EPOCH
        # LOSS COVERS EVERY POSE. The flush used to sit BELOW the stop check, so `ep_loss`
        # was read with the epoch's last partial step still unaccumulated: the rule averaged
        # 1,428 of 1,500 poses and never saw the remaining 72. That step is a handful of
        # structures and its loss ranged 1.21-2.78 across krah8z's 29 epochs, against ~1.99
        # for the epoch, so the two differed by up to 0.04 and disagreed in SIGN: epoch 29
        # read 1.9935 (up, and the eighth strike that ended the run) where the full epoch was
        # 1.9558, its lowest. The stop rule and the plotted curve were not the same number.
        #
        # Before that it sat after the EPOCH LOOP, which is after the last evaluation: at
        # production settings the run evaluated step 292, then took step 293, then wrote
        # head_final.pt -- so the shipped head was one optimiser step away from the only head
        # whose validation metric we had, and the number reported for the run did not belong to
        # the checkpoint the run produced.
        if n_in_accum:
            _apply_partial({"epoch": ep + 1, "lr": cur_lr, "pdb": pdb, "unit_i": ui,
                            "unit": os.path.basename(lab_p)[:-4], "pose": i})

        if ctx.a.early_stop_patience and ep_loss["n"]:
            _ep_loss = ep_loss["sum"] / ep_loss["n"]
            if _ep_loss < stop_state["best"] - ctx.a.early_stop_min_delta:
                stop_state["best"], stop_state["bad"] = _ep_loss, 0
            else:
                stop_state["bad"] += 1
                # AGAINST THE BEST SO FAR, not the previous epoch: this loss rises on
                # individual epochs while still trending down, so a previous-epoch comparison
                # would stop a run that is still learning.
                print(f"  early stop: epoch {ep + 1} loss {_ep_loss:.4f} did not beat "
                      f"{stop_state['best']:.4f} by {ctx.a.early_stop_min_delta:g} "
                      f"({stop_state['bad']}/{ctx.a.early_stop_patience} without progress)")
            if stop_state["bad"] >= ctx.a.early_stop_patience:
                print(f"\nEARLY STOP after epoch {ep + 1}: {stop_state['bad']} consecutive "
                      f"epoch(s) failed to beat the best training loss "
                      f"({stop_state['best']:.4f}) by {ctx.a.early_stop_min_delta:g}. The epoch "
                      f"below still evaluates and still writes its resume bundle, so this "
                      f"stops the run without discarding the epoch that triggered it.")
                done = True

        # THE RESUME BUNDLE, WRITTEN BEFORE THE EVALUATION. The evaluator is a subprocess that
        # does not touch these weights, so either side would be correct -- but writing first
        # means a crash IN the evaluator still leaves this epoch resumable, and the evaluator
        # has crashed before (OST chain mapping, a card with no free memory).
        # WHERE THIS EPOCH'S WALL CLOCK WENT, per rank, as deltas over the epoch rather than
        # run totals -- the first-touch costs (the per-unit fingerprint hash, the item cache)
        # land entirely in epoch 1 and would otherwise contaminate every later epoch's figure.
        _pf_end()
        # AND THE TRUNK WARMER, so its arrays are not still held while the evaluator
        # subprocesses have to fit beside this rank -- the same reason empty_cache() runs
        # below for the device side.
        _tr_end()
        # SO THE EVALUATOR SUBPROCESSES DO NOT INHERIT UP TO --prefix-cache-gb OF HELD HOST
        # MEMORY. Same reason _tr_end is unconditional.
        if ctx._pfc is not None:
            ctx.PFC.step_reset()
        # ANY EVENTS LEFT FROM THE EPOCH'S TAIL, so the last partial accumulation window's
        # GPU time is not carried silently into the next epoch's figures.
        _drain_gpu_spans()
        _ep_s = time.time() - ep_t0
        # THE PREFETCH WAIT IS A REAL BUCKET AND WAS IN NO BUCKET AT ALL. _pf_take is called
        # between the label_open tick and the start of crop_slice, so no timer was open across
        # it and its minutes landed only in the unexplained residual -- which is why, on this
        # arm, "residual" and "prefetch wait" happened to be the same 6.7 min and could be
        # mistaken for two separate costs. Publishing it means the residual means the same
        # thing in every arm instead of only in the ones without a pass A.
        # ACCUMULATED, NOT ASSIGNED. _parts reports _T[k] - _T0[k], which is right only for
        # the run-cumulative counters _tick produces. _pf_stat is RESET per epoch by
        # _pf_start, so assigning it here printed w_N - w_{N-1} -- negative exactly when the
        # wait shrank, which is the phenomenon this bucket was added to track -- and a count
        # delta of 0 whenever two epochs had the same unit count, which filtered the bucket
        # out of both the printed line and _acc from epoch 2 onward. Only epoch 1 was right.
        ctx._T["prefetch_wait"] += ctx._pf_stat["wait_s"]
        ctx._TN["prefetch_wait"] += ctx._pf_stat["hit"] + ctx._pf_stat["miss"]
        # sorted(list(_T)), BECAUSE A WARMER CAN ADD A KEY MID-ITERATION. _tr_worker's
        # resolve_trunk ticks fingerprint_hash_warm, and a defaultdict grows on the FIRST tick
        # of a key -- which is enough for "dictionary changed size during iteration" at an
        # epoch boundary. The snapshot is 20-odd strings.
        _parts = [(k, ctx._T[k] - _T0.get(k, 0.0), ctx._TN[k] - _TN0.get(k, 0))
                  for k in sorted(list(ctx._T))]
        # cropped_tokens is a WORKLOAD counter, not a duration: reported as a mean so the
        # per-rank imbalance is readable directly. The pair track is quadratic in tokens, so
        # equal unit counts across ranks still means unequal compute.
        _shown = "  ".join(
            (f"{k} mean {v/c:.0f}" if k == "cropped_tokens" else f"{k} {v/60:.1f}m/{c}")
            for k, v, c in _parts if c)
        # THE RESIDUAL, COMPUTED HERE RATHER THAN BY HAND. The buckets below are mutually
        # exclusive and sum to accounted time; gpu_forward/gpu_backward are NOT included
        # because they are CUDA-event spans that overlap pose_total rather than adding to it.
        # Printing the residual stops a reader summing the wrong subset -- which is how an
        # ~8 min double-count of the fingerprint hash survived into a committed document.
        # EVERY EXCLUSIVE HOST BUCKET, so the printed residual is actually unexplained time.
        # It omitted prefetch_wait and grad_collective_plus_wait, and the three pass-A buckets
        # that only ranking arms have -- so the same residual meant 6.7 min of prefetch wait
        # on j583i8 and 10-12 min of pass A on zg5fnw and 8fkrsu, with nothing to say which.
        # cpucrop_* are deliberately NOT here: they are a breakdown of work already counted by
        # crop_slice and by the worker, so adding them would double count.
        # THE THREE trunk_* BUCKETS ARE EXCLUSIVE LEAVES TOO. They are the UNCROPPED path's
        # load_unit, so they never co-occur with trunk_read_plus_crop_tokens or crop_slice
        # (the crop branch returns its own tensors and load_unit is not called), and they sit
        # in unit_setup before any pose, so they do not overlap pose_total either. Under
        # --rank-cross-seed pass A ticks them into the same _T, which is already why
        # rank_pass_a is kept out of this list.
        _EXCL = ("pose_total", "crop_slice", "fingerprint_hash", "prefetch_issue",
                 "trunk_npz_open", "trunk_npz_read", "trunk_h2d",
                 "trunk_read_plus_crop_tokens", "label_open", "loss_feature_prep",
                 "loss_stats_collective_plus_wait", "apply_step_total",
                 "prefetch_wait", "grad_collective_plus_wait",
                 # rank_gather and pose_a_total are leaves. rank_pass_a is NOT: it wraps the
                 # score_only run_unit calls, which tick pose_a_total, label_open,
                 # fingerprint_hash, trunk_read_plus_crop_tokens and loss_feature_prep into
                 # the same _T. Including it counted pass A two or three times and could push
                 # the residual negative on a --rank-cross-seed arm -- the same double-count
                 # reasoning that correctly kept cpucrop_* out.
                 "rank_gather", "pose_a_total")
        _acc = sum(v for k, v, c in _parts if k in _EXCL and c)
        print(f"  [rank {ctx.rank}] epoch {ep + 1} took {_ep_s/60:.1f} min -- {_shown}", flush=True)
        # DID THE PREFETCH ACTUALLY FIRE? crop_slice alone cannot answer it: a miss falls back
        # to the inline crop and ticks the same counter, which is exactly how the first
        # version's 100% miss rate looked identical to no prefetch at all.
        # ON _pf_thread, NOT ON a.prefetch_units. _pf_start now declines to start on an
        # uncropped arm, where nothing ever consumed its results -- but the default arm still
        # passes --prefetch-units 2, so keying the print on the flag made every epoch of every
        # uncropped run report "prefetch (4 worker(s)): 0/0 hit (0.0%)", which reads as a
        # regression to 0% rather than as "there is no prefetch on this path".
        if ctx._pf_started[0]:
            _pf_n = ctx._pf_stat["hit"] + ctx._pf_stat["miss"]
            print(f"  [rank {ctx.rank}] epoch {ep + 1} prefetch "
                  f"({max(1, ctx.a.prefetch_workers)} worker(s)): {ctx._pf_stat['hit']}/{_pf_n} hit "
                  f"({100.0 * ctx._pf_stat['hit'] / max(1, _pf_n):.1f}%), "
                  f"{ctx._pf_stat['wait_s']/60:.1f} min waiting on the worker"
                  # REPORTED, because these two are the multi-worker failure made visible. A
                  # non-zero `dropped` means results were delivered and evicted -- harmless
                  # now (an immediate inline recrop) but it is the condition that used to
                  # cost 120 s each and cascade. `exhausted` means the workers finished while
                  # the main loop was still asking. Both were invisible before and both are
                  # the thing to watch when --prefetch-workers goes above 1.
                  + (f", {ctx._pf_stat['dropped']} evicted-then-wanted, "
                     f"{ctx._pf_stat['exhausted']} asked after the workers finished"
                     if (ctx._pf_stat['dropped'] or ctx._pf_stat['exhausted']) else ""), flush=True)
        # DID THE TRUNK CACHE ACTUALLY FIRE? The same question and the same reason: a miss
        # reads inline and ticks the same trunk_* buckets, so without a hit rate a cache that
        # never fires is indistinguishable from one that always does.
        #
        # HOW TO READ THE RATE, stated because the first version printed "ceiling 50%" and that
        # was wrong in the direction that stops you tuning. 50% is what the cache alone gives:
        # pass A misses, fills, and pass B hits. A unit the WARMER got to before pass A is a
        # hit in BOTH passes, so the range is 50% (the warmers contributed nothing) to 100%
        # (every unit was warmed in time). Printing 50% as the ceiling would read a healthy 80%
        # as impossible and a useless 50% as perfect. Note the cap also bounds it: at 4 GiB a
        # 30-unit step of ~7.3 GB fits about half its units, so ~75% is the practical target
        # there and `full` says how many were turned away.
        if ctx._tr_on:
            _tr_n = ctx._tr_stat["hit"] + ctx._tr_stat["miss"]
            print(f"  [rank {ctx.rank}] epoch {ep + 1} trunk cache "
                  f"({ctx.a.trunk_cache_gb:g} GiB, {max(1, ctx.a.prefetch_workers)} warmer(s)): "
                  f"{ctx._tr_stat['hit']}/{_tr_n} hit "
                  f"({100.0 * ctx._tr_stat['hit'] / max(1, _tr_n):.1f}%; 50% = the warmers "
                  f"contributed nothing, 100% = every unit was warmed before pass A), "
                  f"{ctx._tr_stat['put']} cached, {ctx._tr_stat['full']} refused for cap, "
                  f"{ctx._tr_stat['dup']} skipped as present/full, {ctx._tr_stat['err']} "
                  f"warmer error(s)", flush=True)
        # DID THE PREFIX CACHE ACTUALLY FIRE? Same question, same reason: a miss recomputes
        # the prefix and ticks the same gpu_forward bucket, so without a hit rate a cache that
        # never fires looks exactly like one that always does.
        #
        # HOW TO READ THE RATE. The ceiling here really IS 50%: every pose is forwarded twice
        # per step, pass A always misses (it is the fill) and pass B hits, so 50% means the
        # cache worked on every pose and anything below it is poses that were refused for cap.
        # That is the opposite of the trunk cache's range and the reason both lines say so.
        if ctx._pfc is not None:
            _ps = ctx.PFC.stats()
            _pn = _ps["hit"] + _ps["miss"]
            print(f"  [rank {ctx.rank}] epoch {ep + 1} prefix cache "
                  f"({ctx.a.prefix_cache_gb:g} GiB, cut in block {ctx._pfc['block_index']} after "
                  f"tri_mul_in, sig {ctx._pfc['sig']}): "
                  f"{_ps['hit']}/{_pn} hit ({100.0 * _ps['hit'] / max(1, _pn):.1f}%; 50% is "
                  f"the ceiling -- pass A fills and pass B hits), {_ps['put']} cached, "
                  f"{_ps['full']} refused for cap, {_ps['shape']} dropped on a shape "
                  f"mismatch, {_ps['nokey']} forward(s) with no pose key, {_ps['err']} "
                  f"copy error(s); {_ps['d2h_bytes'] / (1 << 30):.1f} GiB staged to host, "
                  f"{_ps['h2d_bytes'] / (1 << 30):.1f} GiB read back", flush=True)
            ctx.PFC.reset_stats()
        # DID THE DEAD-OP SKIP ACTUALLY FIRE? Same question as the two caches above, and the
        # same reason: a stub that is never reached costs nothing and saves nothing, and the
        # epoch time is the only other signal. The count is per stub invocation, so under
        # --prefix-cache-gb expect FEWER calls than stubs x poses x passes -- on a cache hit
        # the blocks before the cut are identities and their stubs are not reached at all.
        if ctx._dops is not None:
            _ds = ctx.DOPS.stats()
            print(f"  [rank {ctx.rank}] epoch {ep + 1} dead ops: {_ds['calls']:,} stub call(s) "
                  f"across {_ds['stubbed']} skipped module(s) "
                  f"({_ds['tensors']} tensor(s) held at the floor); each one is an O(N^3) or "
                  f"O(N^2) pair operation NOT computed, and each returned a zeros tensor the "
                  f"block added to z -- bit-identical to running it", flush=True)
            ctx.DOPS.reset_stats()
        # THE DRIFT IS THE EXPERIMENT under --l2-sp, so it is reported, not inferred. The
        # full head reached ||dW||^2 = 67,052 unconstrained over 30 epochs; this says where
        # the penalised run actually sits.
        # THE REALISED DRAW, reported rather than assumed. The configured probabilities say
        # what was asked for; this says what the epoch actually trained on, which is the
        # number to compare against the fixed crop's 0.598.
        if ctx._TOK_DROP and ctx._tok_drop_stat["n"]:
            _tk = ctx._tok_drop_stat["kept"] / max(ctx._tok_drop_stat["tot"], 1.0)
            print(f"  [rank {ctx.rank}] epoch {ep + 1} token draw: kept {_tk:.4f} of tokens over "
                  f"{ctx._tok_drop_stat['n']} unit(s) "
                  f"(union p={ctx.a.token_keep_union:g}, other p={ctx.a.token_keep_other:g}; the "
                  f"fixed crop was 0.598), so ~{_tk * _tk / 0.598 ** 2:.2f}x the fixed "
                  f"crop's pair work; {ctx._tok_drop_stat['topped']} chain(s) topped up to the "
                  f"4-token floor, {ctx._tok_drop_stat['topped_tok']} token(s) added", flush=True)
        if ctx.a.l2_sp and ctx._l2sp_drift["n"]:
            # AS A SHARE OF THE OBJECTIVE, not just in absolute terms. A coefficient that
            # LOOKS weak is not weak if the squared displacement is large: at the 67,052 the
            # unconstrained head reached, lambda 1e-4 is a penalty of 6.7 against a loss near
            # 2.9 -- more than twice the objective. So "is 1e-4 weak here" is only answerable
            # against the epoch's own loss, and this prints the ratio rather than leaving a
            # reader to do it. Raised in review.
            _l2p = ctx.a.l2_sp * ctx._l2sp_drift["sq"]
            _l2_ref = (ep_loss["sum"] / ep_loss["n"]) if ep_loss["n"] else float("nan")
            _l2_share = (_l2p / _l2_ref) if (_l2_ref == _l2_ref and _l2_ref > 0) else \
                float("nan")
            print(f"  [rank {ctx.rank}] epoch {ep + 1} l2-sp: ||W-W0||^2 = "
                  f"{ctx._l2sp_drift['sq']:.1f}, penalty = {_l2p:.4f} "
                  f"= {100.0 * _l2_share:.1f}% of the epoch's mean loss {_l2_ref:.4f} "
                  f"(unconstrained full head reached 67,052 by epoch 30, which at "
                  f"lambda {ctx.a.l2_sp:g} would be {ctx.a.l2_sp * 67052.0:.2f})", flush=True)
        # THE RANKING LOSS IS THE EXPERIMENT under --rank-loss. ZERO is the reference, not
        # ln2: this loss is a normalised covariance between score gaps and DockQ gaps, so
        # equal scores give exactly 0 and so does an ordering uncorrelated with DockQ.
        # NEGATIVE is correctly aligned, positive is inverted. ln2 = 0.6931 was the reference
        # for the softplus version and was left here when the loss went linear -- caught in
        # review of cd8e6ab7, before it misled a single training run.
        #
        # AND IT DOES NOT MEASURE INVERSIONS. c_i depends only on DockQ, so a pose already
        # ranked above everything still gets pushed up: the loss can fall by widening gaps
        # that were already correct, with the inverted fraction unchanged. The within-seed
        # inversion rate has to be read separately -- see run_logging/rank_rho.py.
        if ctx.a.rank_loss and ctx.a.rank_cross_seed and ctx._xs_stat["groups"]:
            # THE SOFTPLUS LOSS IS NOT sum_i c_i s_i, so it cannot be reconstructed from the
            # coefficients the way the linear one is. group_coeffs returns the value in pass A
            # and it is accumulated there; 0 is the floor (every pair ordered), not "no
            # alignment".
            # THE PAIR FAMILY IS a.rank_pairs, NOT the string "cross-seed". That was
            # hardcoded here from a time when every arm formed cross-seed pairs, so x5frz6 --
            # which forms WITHIN-SEED pairs only -- printed "cross-seed" at every epoch while
            # the startup banner said the opposite. The pair count was the only thing that
            # gave it away: 6,656 against the ~86,600 a cross-seed objective would form.
            if ctx.a.rank_form == "rel" and ctx._rel_stat.get("rho"):
                def _med(v):
                    w = sorted(v)
                    return w[len(w) // 2] if w else float("nan")
                _rh = ctx._rel_stat["rho"]
                print(f"  [rank {ctx.rank}] epoch {ep + 1} WITHIN-INTERFACE SPEARMAN "
                      f"(predicted mean ipTM vs DockQ): median {_med(_rh):+.4f} over "
                      f"{len(_rh)} interface-group(s)   [the head was +0.3170 at d32kqp ep66; "
                      f"the TRUE iTM reaches +0.6900 -- this is the arm's prediction]")
                if ctx._rel_stat.get("sd_ratio"):
                    print(f"  [rank {ctx.rank}] epoch {ep + 1} SCORE SPREAD: median std(predicted) "
                          f"{_med(ctx._rel_stat['sd_s']):.4f} against std(true iTM) "
                          f"{_med(ctx._rel_stat['sd_g']):.4f}, median ratio "
                          f"{_med(ctx._rel_stat['sd_ratio']):.3f}   "
                          f"[<1 collapsing, >1 exaggerating, ~1 right magnitude]")
                for _k3 in ("rho", "sd_s", "sd_g", "sd_ratio"):
                    ctx._rel_stat[_k3] = []
            # THE LINE MUST NAME THE OBJECTIVE THAT RAN. Raised in review: this printed
            # "({rank_form}, {rank_pairs}-seed pairs)" for EVERY form, so a rel run -- which
            # has no pair family at all and ignores --rank-pairs entirely -- was logged as
            # "(rel, cross-seed pairs)". This is the one line most likely to be read back
            # months later to establish what an arm trained on, and it was asserting a pair
            # family for an objective that forms no pairs.
            _obj = ("relative regression on true iTM, mean-centred per interface"
                    if ctx.a.rank_form == "rel"
                    else f"{ctx.a.rank_form}, {ctx.a.rank_pairs}-seed pairs")
            print(f"  [rank {ctx.rank}] epoch {ep + 1} rank-loss ({_obj}): "
                  f"{ctx._xs_stat['loss'] / ctx._xs_stat['groups']:+.5f} over "
                  f"{ctx._xs_stat['groups']} interface-group(s) in {ctx._xs_stat['steps']} step(s)"
                  # THE REASON MUST MATCH THE FORM. Raised in review: this said "no
                  # straddling pair" whatever the objective, which is false under rel -- rel
                  # forms no pairs, and a skipped rel step means the step held no interface
                  # with at least two labelled poses to centre over.
                  + ((f" [{ctx._xs_stat['skipped']} step(s) SKIPPED: "
                      + ("no interface with two or more labelled poses to centre over"
                         if ctx.a.rank_form == "rel" else
                         "no straddling pair in either structure")
                      + ", so no task gradient -- not handed to AdamW]")
                     if ctx._xs_stat.get("skipped") else "") + ", "
                  # THE COUNT IS POSES FOR THE EXPECTED FORM, NOT PAIRS, and its floor is
                  # -max(DockQ) rather than 0. Calling 150 poses "150 pair(s)" and claiming a
                  # zero floor would misreport the objective in the one line most likely to
                  # be read back later.
                  f"{ctx._xs_stat['pairs']} "
                  # POSES FOR rel TOO, not just for expected. rank_loss.rel_coeffs returns
                  # (coeffs, n_poses, loss), so this counter holds a POSE count under rel and
                  # calling it "pair(s)" overstated the group size by the pair-count factor.
                  + ("pose(s)" if ctx.a.rank_form in ("expected", "rel") else "pair(s)")
                  + ("  (0 is the FLOOR -- every pair ordered)"
                     if ctx.a.rank_form == "softplus" else "")
                  + ("  (0 is the FLOOR -- centred prediction equals centred true iTM)"
                     if ctx.a.rank_form == "rel" else "")
                  + ("  (NEGATIVE: this is -expected DockQ, floor -max(DockQ) per group)"
                     if ctx.a.rank_form == "expected" else ""), flush=True)
            for _k in ("groups", "pairs", "steps", "skipped"):
                ctx._xs_stat[_k] = 0
            ctx._xs_stat["loss"] = 0.0
        elif ctx.a.rank_loss and ctx._rank_seen["units"]:
            _rk_mean = ctx._rank_seen["loss"] / ctx._rank_seen["units"]
            print(f"  [rank {ctx.rank}] epoch {ep + 1} rank-loss: {_rk_mean:+.5f} over "
                  f"{ctx._rank_seen['units']} unit(s) (0 = no alignment, negative is better; "
                  f"NOT an inversion rate)", flush=True)
            ctx._rank_seen["loss"] = 0.0
            ctx._rank_seen["units"] = 0
        print(f"  [rank {ctx.rank}] epoch {ep + 1} accounted {_acc/60:.1f}m of {_ep_s/60:.1f}m, "
              f"residual {(_ep_s - _acc)/60:.1f}m "
              f"(exclusive buckets only; gpu_* overlap pose_total, "
              f"unit_setup overlaps the per-unit leaves)", flush=True)
        # THE RESIDUAL, SPLIT. unit_setup is a SUPERSET of label_open, fingerprint_hash,
        # trunk_read_plus_crop_tokens, crop_slice and loss_feature_prep, so it is deliberately
        # NOT in _EXCL -- including it would double-count the same minutes, which is the trap
        # the comment beside rank_pass_a describes. What it buys is the part of the residual
        # it EXPLAINS: setup time not already claimed by those leaves is inside-unit work, and
        # whatever is left after that is between-unit work, which is a different fix.
        _us = dict((k, v) for k, v, c in _parts if c)
        if _us.get("unit_setup"):
            # THE THREE trunk_* BUCKETS ARE LEAVES TOO, and leaving them out of this sum after
            # instrumenting load_unit would report the very time they now name as still
            # unaccounted -- the measurement would look like it had changed nothing.
            _leaves = sum(_us.get(k, 0.0) for k in
                          ("label_open", "fingerprint_hash", "trunk_read_plus_crop_tokens",
                           "crop_slice", "cpucrop_label", "cpucrop_slice", "cpucrop_tokens",
                           "cpucrop_trunk", "loss_feature_prep", "prefetch_issue",
                           "prefetch_wait", "trunk_npz_open", "trunk_npz_read", "trunk_h2d"))
            print(f"  [rank {ctx.rank}] epoch {ep + 1} unit_setup {_us['unit_setup']/60:.1f}m, "
                  f"of which {_leaves/60:.1f}m is already in named leaves -> "
                  f"{(_us['unit_setup'] - _leaves)/60:.1f}m of the residual is INSIDE units, "
                  f"the rest is between them", flush=True)
        if torch.cuda.is_available():
            _gb = 1 << 30
            # UNDER --mem-log THE CUDA COUNTER IS NOT THE EPOCH'S, see _mem_ep.
            _ep_peak = max(ctx._mem_ep[0], torch.cuda.max_memory_allocated()) if ctx.a.mem_log \
                else torch.cuda.max_memory_allocated()
            print(f"  [rank {ctx.rank}] epoch {ep + 1} memory: peak allocated "
                  f"{_ep_peak/_gb:.2f} GB, peak reserved "
                  f"{torch.cuda.max_memory_reserved()/_gb:.2f} GB, "
                  f"blocks_per_ckpt={ctx.model.confidence_head.pairformer_stack.blocks_per_ckpt}"
                  + (f", ckpt_threshold={ctx.a.ckpt_token_threshold} tok -> "
                     f"{ctx._ckpt_hist[0]} unit(s) UNCHECKPOINTED, {ctx._ckpt_hist[1]} checkpointed"
                     if ctx.a.ckpt_token_threshold > 0 else ""),
                  flush=True)
        ctx._ckpt_hist[0] = ctx._ckpt_hist[1] = 0
        # EVERY RANK SAVES ITS OWN, at the boundary rather than at exit: x3k2m4 died after
        # epoch 1 and would have discarded all 4,880 hashes it had just computed.
        _fp_cache_save()
        # AND READ BACK WHAT THE OTHER RANKS JUST WROTE. The three ranks jointly cover all
        # 6,903 units in epoch 1, but the shuffle moves a third of them onto a different rank
        # every epoch -- so without this a rank re-hashes, from epoch 2 onward, every unit it
        # has not personally seen, while the hash it needs is already on disk beside its own.
        # Entries this process computed are never replaced; see _fp_cache_load.
        #
        # NOT SYNCHRONISED, AND IT DOES NOT NEED TO BE. Every rank writes its own file
        # atomically via os.replace, so a sibling either sees the previous version or the new
        # one and never a torn one; a miss here costs one re-hash and the next boundary picks
        # it up. os.replace is also why this is safe to do while the siblings are writing.
        _fp_cache_load(label=f"epoch {ep + 1} boundary")
        # HAND THE ALLOCATOR'S CACHE BACK BEFORE THE VAL HOOKS RUN. The ranks are parked at a
        # barrier for the held-out pass and the DockQ eval, not computing -- but PyTorch holds
        # freed activation blocks as *reserved*, so a parked rank still looks ~20-30 GB wide to
        # the subprocesses that have to fit beside it. Measured at the p100 crop: allocated
        # 2.28 GB against reserved 30.66, so ~28 GB per card is cached-but-free.
        #
        # THIS MATTERS MORE WITHOUT ACTIVATION CHECKPOINTING, where the stored activations push
        # the peak up and therefore the reserved pool with it. Costs a few hundred ms of
        # re-allocation on the next epoch's first poses, at an epoch boundary, once per epoch.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ctx.is_main and write_traj:
            _save_resume(ep + 1, step)
            # THE EPOCH'S HEAD, WRITTEN HERE AND NOT BY THE EVALUATOR.
            #
            # IT USED TO BE WRITTEN ONLY WHEN AN EVAL RAN. The one `torch.save` of
            # head_ep<N>_step<S>.pt sits INSIDE the evaluation function, before it spawns
            # phase_k_eval -- so under NOEVAL=1 the function is never called and NO
            # epoch-boundary head was ever written. The live arm mqird6 has only
            # head_step250/500/750/1000.pt from the periodic --save-every path, and at 230
            # steps per epoch those land at arbitrary points MID-epoch: there is no head
            # checkpoint corresponding to any completed epoch of that run.
            #
            # THE HEAD IS THE ARTIFACT AND THE EVAL IS ONE CONSUMER OF IT. Coupling them meant
            # turning off the consumer silently cost the artifact. Nothing was lost on the
            # current run -- resume_ep<N>.pt is written unconditionally just above and contains
            # the head -- but it has to be extracted, which is exactly the friction this
            # removes.
            #
            # SAME NAME, so every downstream scorer, plot and eval_dirs reader keeps working,
            # and the evaluator's own save becomes a no-op rewrite of an identical file rather
            # than the only copy. ~79 MB per epoch against a 3,072 GiB quota.
            _ep_cp = os.path.join(ctx.a.out, f"head_ep{ep + 1}_step{step}.pt")
            try:
                torch.save(_head_ckpt_tensors(), _ep_cp)
                print(f"  saved {_ep_cp} "
                      f"({os.path.getsize(_ep_cp)/2**20:.0f} MiB: the epoch's head)",
                      flush=True)
            except OSError as _e:
                # A CHECKPOINT IS WORTH REPORTING AND NOT WORTH THE RUN, the same bargain
                # _fp_cache_save makes. The resume bundle above already carries the head.
                print(f"  WARNING: could not write {_ep_cp} ({_e}); the epoch's head is still "
                      f"inside resume_ep{ep + 1}.pt", flush=True)
        # NO write_traj IN THIS CONDITION. It is `is_main`, so including it meant ranks 1
        # and 2 skipped the whole block INCLUDING both barriers, while rank 0 entered them.
        # dist.barrier() is untagged, so rank 0's two calls paired with the other ranks' two
        # train-eval calls and rank 0 then blocked forever on its third -- a deadlock at the
        # first epoch boundary, not a dropped measurement. Every rank enters the barriers;
        # only rank 0 does the work inside, exactly as the two evaluation paths below.
        if (ctx.a.val_loss_every and (ep + 1) % ctx.a.val_loss_every == 0
                and step > ep_start_step):
            # SAME BARRIER DISCIPLINE as the two evaluation paths: one measuring process, the
            # other ranks parked for its duration. Without the second barrier rank 0 re-enters
            # training a whole pass behind the others.
            if ctx.world > 1:
                dist.barrier()
            _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
            if ctx.is_main:
                _flush(ctx.traj_p, ctx.traj_cols, rows)
            rows.clear()
            if ctx.is_main:
                _bundle = os.path.join(ctx.a.out, f"resume_ep{ep + 1}.pt")
                if not os.path.exists(_bundle):
                    print(f"  WARNING: no {_bundle}; skipping the held-out PAE for this epoch")
                else:
                    # THE WEIGHTS MUST BE UNTOUCHED, ASSERTED NOT ASSUMED. The measuring
                    # process reads a bundle off disk and shares nothing with this one, so it
                    # cannot reach these tensors -- but "cannot" is the kind of claim that
                    # stops being true after an edit, and a monitor that silently perturbed
                    # the run it monitors would invalidate every number after it.
                    _before = hashlib.sha256()
                    for _n2 in ctx.trainable:
                        _before.update(dict(ctx.model.named_parameters())[_n2]
                                       .detach().cpu().numpy().tobytes())
                    ctx.val_loss_checkpoint(ep + 1, _bundle, step)
                    _after = hashlib.sha256()
                    for _n2 in ctx.trainable:
                        _after.update(dict(ctx.model.named_parameters())[_n2]
                                      .detach().cpu().numpy().tobytes())
                    if _before.digest() != _after.digest():
                        raise SystemExit(
                            f"STOP: the head changed across the epoch-{ep + 1} held-out PAE "
                            f"pass. It runs as a subprocess precisely so it cannot do that, so "
                            f"something now shares state with it and every number after this "
                            f"point is suspect.")
            if ctx.world > 1:
                dist.barrier()
        if ctx.a.eval_every_epoch and step > ep_start_step:
            # RANK 0 EVALUATES, THE OTHERS WAIT. The evaluator is a separate process holding
            # its own model, so letting every rank launch one would put three on the cards at
            # once. The barriers keep the ranks together across the ~16 min it takes; without
            # the second one, rank 0 would re-enter training a whole evaluation behind.
            if ctx.world > 1:
                dist.barrier()
            _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
            if ctx.is_main:
                _flush(ctx.traj_p, ctx.traj_cols, rows)
            rows.clear()
            if ctx.is_main:
                ctx.eval_checkpoint(step, ep + 1)
            if ctx.world > 1:
                dist.barrier()
        if (ctx.a.train_eval_pool and ctx.a.train_eval_every
                and (ep + 1) % ctx.a.train_eval_every == 0 and step > ep_start_step):
            # SAME BARRIER DISCIPLINE as the validation path above: one evaluator, sharded
            # over every card, the other ranks parked for its duration.
            if ctx.world > 1:
                dist.barrier()
            _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
            if ctx.is_main:
                _flush(ctx.traj_p, ctx.traj_cols, rows)
            rows.clear()
            if ctx.is_main:
                ctx.eval_checkpoint(step, ep + 1, train_pool=ctx.a.train_eval_pool,
                                crop_tokens=ctx.a.eval_pool_crop_tokens)
            if ctx.world > 1:
                dist.barrier()
        # EVERY EXTRA PASS, each on its own cadence. SAME BARRIER DISCIPLINE: one evaluator
        # sharded over every card while the other ranks park, so these must not interleave.
        for _ee in ctx._EXTRA_EVAL:
            if not ((ep + 1) % _ee["every"] == 0 and step > ep_start_step):
                continue
            if ctx.world > 1:
                dist.barrier()
            _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
            if ctx.is_main:
                _flush(ctx.traj_p, ctx.traj_cols, rows)
            rows.clear()
            if ctx.is_main:
                print(f"  extra eval '{_ee['name']}': pool {_ee['pool']}, budget "
                      f"{_ee['budget'] or 'default'}, every {_ee['every']} epoch(s), "
                      f"{'CROPPED' if _ee['crop_tokens'] else 'uncropped'}", flush=True)
                ctx.eval_checkpoint(step, ep + 1, train_pool=_ee["pool"],
                                crop_tokens=_ee["crop_tokens"], tag=_ee["name"],
                                fold=_ee["fold"], seed_base=_ee["seed_base"],
                                budget=_ee["budget"])
            if ctx.world > 1:
                dist.barrier()
    # NOTHING MAY BE LEFT HERE. Every epoch flushes its own remainder above, before evaluating,
    # so a non-zero accumulator at this point would mean a path out of the loop that skipped the
    # flush -- and the saved head would again differ from the evaluated one. Flushed and reported
    # rather than dropped, because dropping poses silently is what the flush was added to stop.
    if n_in_accum:
        print(f"  WARNING: {n_in_accum} pose(s) still accumulated after the epoch loop; the "
              f"evaluated head is NOT the saved head for this run")
        _apply_partial({"epoch": ep + 1, "lr": cur_lr, "pdb": pdb, "unit_i": ui,
                        "unit": os.path.basename(lab_p)[:-4], "pose": i})
    if write_traj:
        _flush(ctx.traj_p, ctx.traj_cols, rows)
    # EVERY RANK, including rank 0. _flush clears nothing, so the same rows go to both files;
    # the final clear below is what retires them.
    if rows:
        _flush(ctx.traj_rank_p, ctx.traj_cols, list(rows))
    if ctx.crop_p and ctx.crop_rows:        # every rank, its own file
        _flush(ctx.crop_p, ctx.crop_cols, ctx.crop_rows)
        ctx.crop_rows.clear()
    return rows
