"""`trainer/resume.py` -- Restore weights, optimiser moments and the epoch/step counter from a bundle.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 3451-3858 (source `d34baebc`).
Statements and comments are the original; only the names this closed over inside
main() became parameters. test_resume_equivalence.py compares it line for line,
so a change to the live trainer fails the test instead of drifting silently.
"""
import json
import os
import math

import torch
import torch.distributed as dist


def load_resume(rt, a, model, opt):
    # INITIALISED BEFORE THE BRANCH, because `if a.resume:` is the only thing that sets them and
    # the live arm runs with `resume: ""`. phase_k_train.py:3424 does this one line above the
    # extracted span; cutting the region at the `if` left the function raising UnboundLocalError
    # on exactly the configuration that is running. Found by the bloat audit, not by a test --
    # the equivalence test compares the region, and this line is outside it.
    start_epoch, start_step = 0, 0
    if a.resume:
        if not os.path.exists(a.resume):
            raise SystemExit(f"STOP: no resume bundle at {a.resume}")
        _b = torch.load(a.resume, map_location="cpu", weights_only=False)
        _g = _b.get("guard", {})
        # INHERITED ON EVERY RESUME, NOT ONLY A DRIFTING ONE. A clean continuation of a run that
        # was itself a fork is still a fork's descendant, and a chain that dropped its history at
        # the first non-drifting link would read as pristine two bundles later.
        _drift_history = list(_g.get("drift_history") or [])
        # WHAT MUST MATCH, in two groups. The first changes which poses land in which batch, so the
        # saved moments describe a different problem. The second changes the objective or the
        # schedule, so continuing would silently blend two experiments. Checking only the first four
        # let a resume through after the exclusions, the loss weights and the lr had all changed
        # (review 2026-10-16).
        _mine = {"world": rt.world, "seed": a.seed, "accum": a.accum, "nsample": a.nsample,
                 "manifest_digest": rt._manifest_digest, "crop_size": a.crop_size,
                 "alpha_confidence": a.alpha_confidence, "alpha_pae": a.alpha_pae,
                 "drop_pde": False, "clip_norm": a.clip_norm,
                 "rank_loss": a.rank_loss, "alpha_rank": a.alpha_rank,
                 "rank_pool": a.rank_pool, "rel_labels": a.rel_labels,
                 "rel_standardize": a.rel_standardize,
                 # THE FOUR CROSS-SEED FIELDS. Every one of them changes the objective or the step
                 # shape, so a resume that moved any of them silently would be a different
                 # experiment wearing the same run id. FOURTH time a field has been added to this
                 # comparison without being added to the saved bundle below -- so both sides are
                 # edited in the same patch, and test_guard_fields.py asserts they agree.
                 "rank_form": a.rank_form,
                 "rank_cross_seed": a.rank_cross_seed,
                 "rank_structs_per_step": a.rank_structs_per_step,
                 # L2-SP CHANGES THE OBJECTIVE and was missing here: a resume could silently move
                 # lambda and the drift guard would not notice. Caught in review of cd8e6ab7.
                 # AND SO DOES THE SCHEDULE. lr_final changes the lr at every step after warmup,
                 # so a resume that added or dropped the cosine would be a different experiment.
                 "lr": a.lr, "warmup_steps": a.warmup_steps,
                 # ROW WEIGHTING CHANGES THE OBJECTIVE. The PAE term becomes a weighted mean
                 # over anchor rows, so moving frac, mass or the floor on resume blends two
                 # objectives into one curve -- the same failure the l2_sp and lr_final entries
                 # above exist to stop. Caught in review of 38a83283.
                 # The decay WINDOW and SHAPE change the lr at every step just as lr_final does,
                 # so they belong beside it rather than being silently re-derived on resume.
                 # THE STAIRCASE IS A SCHEDULE CHANGE like lr_final and the decay window, so a
                 # resume that silently changed it would splice two schedules into one curve.
                 "lr_steps": a.lr_steps,
                 # DROPOUT IS PART OF THE OBJECTIVE, not of the optimiser: flipping it on a
                 # resume changes the gradient's distribution at every step.
                 # THE AUGMENTATION DISTRIBUTION. Changing either probability changes what the
                 # head sees at every step, so a resume that silently changed them would splice
                 # two augmentations into one curve. Raised in review.
                 # CROPPING CHANGES THE OBJECTIVE -- the loss averages over a different set of
                 # tokens and pairs -- so resuming across a crop change would blend two experiments
                 # into one curve, which is exactly what this guard exists to stop.
                 "crop_contact": 5.0,
                 # A CROP RULE CHANGE IS AN OBJECTIVE CHANGE: the loss averages over a different
                 # set of tokens and pairs, so resuming across it would blend two objectives.
                 # THE CROP DEFINITION ITSELF, not just its radii. a16911de changed the DEFAULT
                 # from the union over all nsample poses to pose 0 alone, so a resume across that
                 # change blends two crop definitions into one curve. f1268fe5 added the field to
                 # the saved bundle and stopped there -- which is the inert guard described two
                 # entries down, repeated one commit later. It is compared HERE and defaulted in
                 # _PRE_CROP, which are the two halves that make a field actually checked.
                 # SAME REASON AS CROPPING. --pae-scored-pairs changes which pairs the PAE term
                 # averages over and --pae-only changes which terms exist at all, so resuming
                 # across either blends two objectives into one curve, carrying the first one's
                 # AdamW moments. These were added to the saved bundle and to _PRE_CROP but NOT
                 # here, which left the guard inert: it compared neither, and the _PRE_CROP
                 # defaults for them were never consulted.
                 "pae_scored_pairs": a.pae_scored_pairs, "pae_only": a.pae_only,
                 # THE PARAMETER SET AGAIN, AND THIS ONE CORRUPTS SILENTLY IF IT IS NOT GUARDED.
                 # --lean-single drops 158 tensors from the optimiser's param group. AdamW's
                 # state_dict is keyed by the parameter's INDEX in that group, so resuming a
                 # --lean-single bundle without the flag (or the reverse) would load exp_avg and
                 # exp_avg_sq onto DIFFERENT tensors -- no shape error, because many of them are
                 # the same shape, and no way to notice afterwards. It changes no mathematics in
                 # the forward, which is exactly why nothing else would catch it.
                 "lean_single": a.lean_single,
                 # THE PARAMETER SET AGAIN: --pose-inblock ADDS trainable tensors (two, or four
                 # with --pose-inblock-pae), so a bundle written without it is missing them and
                 # the _missing check below refuses it anyway. Compared here so the refusal names
                 # the setting instead of reporting an absent tensor -- the same reason
                 # frame_direction is compared. The BLOCK matters too: the same two tensors at a
                 # different depth is a different operator.
                 "pose_inblock": a.pose_inblock, "pose_inblock_pae": a.pose_inblock_pae,
                 "pose_inblock_block": a.pose_inblock_block,
                 # NOT JUST THE OBJECTIVE -- THE PARAMETER SET. --frame-direction adds a trainable
                 # tensor, so a bundle written without it is missing one and the _missing check
                 # below refuses it anyway. It is compared here so the refusal names the setting
                 # that caused it instead of reporting an absent tensor.
                 "frame_direction": a.frame_direction,
                 # SAME REASON AS frame_direction: A adds trainable tensors, so a bundle written
                 # without it is missing some and the _missing check below refuses it anyway.
                 # Compared here so the refusal names the setting rather than an absent tensor.
                 "pose_perblock": a.pose_perblock,
                 "pose_perblock_tied": a.pose_perblock_tied,
                 "pose_perblock_share": a.pose_perblock_share,
                 "pose_perblock_copy": a.pose_perblock_copy,
                 # THE PARAMETER SET ITSELF. A bundle written at rank 8 holds 282 adapter
                 # tensors and one written without LoRA holds 248 head tensors; resuming one as
                 # the other fails on the missing-tensor check with a message about tensors
                 # rather than about the setting that changed.
                 "lora_rank": 0, "lora_alpha": 0.0,
                 "lora_targets": '', "lora_dropout": 0.0,
                 "train_only": a.train_only}

        # A BUNDLE WRITTEN BY OLDER CODE IS NOT DRIFT. The guard grew from four fields to twelve
        # while run 9mfb4y was in flight, so its bundles carry only the original eight: the running
        # process had the old module loaded and kept writing the old shape while HEAD moved under
        # it. Reading the six newer fields as "<absent>" made every one of them a mismatch, which
        # would have reported six settings as changed when none had, and a resume as a FORK when it
        # is a continuation.
        #
        # So an absent field falls back to the run's own args.json, which sits beside the bundle and
        # does record them. That is the same evidence, written by the same process at launch.
        _side = os.path.join(os.path.dirname(os.path.abspath(a.resume)), "args.json")
        _fallback = {}
        if os.path.exists(_side):
            try:
                _fallback = json.load(open(_side))
            except Exception as e:
                print(f"  (could not read {_side}: {type(e).__name__}: {e})")

        # A FIELD THAT POSTDATES THE BUNDLE IS NOT AN UNKNOWN FIELD. crop_margin and crop_contact
        # joined the guard after run 9mfb4y was already in flight, so none of its bundles carry
        # them -- not resume_ep1/2/3, and not the ep4/ep5 ones being written tonight either, since
        # that process loaded its code before the fields existed. Reading them as "recorded
        # nowhere" made the _unknown hard-stop fire on every pre-crop bundle and refused the
        # resume outright, which --resume-allow-drift could not even override because the stop
        # runs first. I turned a safety check into a wall by adding two fields every existing
        # bundle lacks; review round 6 reproduced it.
        #
        # Absence is EVIDENCE here, not ignorance: a bundle written by code with no cropping was
        # written with cropping off. So these migrate to the pre-crop defaults and are compared
        # like any other field -- which means a resume that ENABLES cropping shows up as real
        # drift and needs --resume-allow-drift, because it genuinely changes the objective.
        _PRE_CROP = {"lora_rank": 0, "lora_alpha": 0.0, "lora_targets": "", "lora_dropout": 0.0,
                     # A bundle written before these flags existed ran the full-run schedule, which
                     # is exactly what the 0 defaults mean, and "cosine" is what it computed.
                     "lr_decay_start_epoch": 0.0, "lr_decay_end_epoch": 0.0,
                     "lr_decay_shape": "cosine", "lr_steps": "",
                     "pair_dropout": -1.0,
                     "token_keep_union": -1.0, "token_keep_other": -1.0,
                     # ABSENCE IS EVIDENCE: a bundle written before --pae-row-frac existed ran the
                     # unweighted term, which is exactly what frac 0 means. The mass and floor
                     # defaults are inert at frac 0 but are migrated so a resume that only turns
                     # weighting ON still shows up as drift on the frac field.
                     "pae_row_frac": 0.0, "pae_row_mass": 0.8, "pae_row_floor": 10,
                 "train_only": "",
                     "crop_contact": 5.0, "pae_scored_pairs": "",
                     # ABSENCE IS EVIDENCE, as for the two above: a bundle written before
                     # a16911de was written by code that had no such flag and always took the
                     # union, so it migrates to True. A resume that switches to pose 0 therefore
                     # reads as real drift and must be named with --resume-override crop_all_poses.
                     "crop_all_poses": True,
                     "pae_only": False,
                     # A BUNDLE PREDATING THE FLAG TRAINED THE FULL PARAMETER GROUP.
                     "lean_single": False,
                     # A BUNDLE PREDATING THE FLAG HAD NO IN-BLOCK INJECTION AT ALL.
                     "pose_inblock": False, "pose_inblock_pae": False,
                     "pose_inblock_block": -1,
                     "frame_direction": False,
                     "pose_perblock": False, "pose_perblock_tied": False,
                     "pose_perblock_share": False, "pose_perblock_copy": False,
                     "rank_loss": False, "alpha_rank": 1.0,
                     "rank_pool": "max", "rel_labels": "", "rel_standardize": False,
                     "rank_weight_by_gap": True, "l2_sp": 0.0, "l2_sp_exclude": "",
                     # A BUNDLE PREDATING THESE RAN THE LINEAR WITHIN-SEED LOSS, which is what
                     # these defaults say. rank_tau -1.0 means "unset", matching how the
                     # comparison side encodes None.
                     "rank_form": "linear", "rank_tau": -1.0,
                     "rank_cross_seed": False, "rank_structs_per_step": 2,
                     # A BUNDLE PREDATING THESE THREE RAN CROSS-SEED PAIRS ONLY, AT NATURAL
                     # WEIGHT, WITH NO RANK DECAY -- which is what these defaults say. Their
                     # absence is evidence of the inert values, not of an unknown setting, so
                     # they belong here rather than in _unknown: the STOP at "recorded nowhere
                     # for this bundle" fires BEFORE --resume-allow-drift is consulted and so
                     # cannot be overridden, which would have made every pre-patch bundle --
                     # including the live zg5fnw's per-epoch held-out subprocess, which resumes
                     # its own bundle through this same file -- unresumable.
                     "rank_pairs": "cross", "rank_within_weight": 1.0, "rank_topk_decay": 0.0,
                     "rank_straddle": 0.0,
                     # A BUNDLE PREDATING --lr-final HELD lr FLAT AFTER WARMUP, which is exactly
                     # what lr_final 0.0 means -- the flag's own help says 0 "keeps the historical
                     # schedule ... hold --lr flat for the whole run, which is what every arm
                     # before this one did". So its absence is evidence of 0.0, not of an unknown
                     # setting, and it belongs here rather than in _unknown: that STOP fires
                     # BEFORE --resume-allow-drift is consulted and so cannot be overridden,
                     # which made ocqtyt's epoch-26 bundle unresumable (run 4av7s8, exit 1).
                     "lr_final": 0.0}
        _migrated = []

        _bad, _from_args, _unknown = {}, [], []
        for k, v in _mine.items():
            if k in _g:
                if _g[k] != v:
                    _bad[k] = (_g[k], v)
            elif k in _fallback:
                if _fallback[k] != v:
                    _bad[k] = (_fallback[k], v)
                else:
                    _from_args.append(k)
            elif k in _PRE_CROP:
                _migrated.append(k)
                if _PRE_CROP[k] != v:
                    _bad[k] = (f"{_PRE_CROP[k]} (pre-crop default)", v)
            else:
                _unknown.append(k)
        if _migrated:
            print(f"  resume guard: {len(_migrated)} field(s) postdate this bundle and are read as "
                  f"the pre-crop default: "
                  f"{', '.join(f'{k}={_PRE_CROP[k]}' for k in sorted(_migrated))}")
        if _from_args:
            print(f"  resume guard: {len(_from_args)} field(s) absent from the bundle and matched "
                  f"against {os.path.basename(_side)} instead: {', '.join(sorted(_from_args))}")
        if _unknown:
            # manifest_digest is the only field in neither place. It is a hash of the unit list,
            # which is determined by the selection file and the skip list -- and manifest_skip IS
            # in the bundle and is compared above. So the population is still checked, by its
            # inputs rather than by its hash. Said out loud rather than passed over, because an
            # unverified field is not a verified one.
            print(f"  resume guard: {len(_unknown)} field(s) in neither the bundle nor "
                  f"{os.path.basename(_side)}, so NOT compared: {', '.join(sorted(_unknown))}")
            for k in _unknown:
                if k != "manifest_digest":
                    raise SystemExit(
                        f"STOP: {k} is recorded nowhere for this bundle, so a resume cannot show "
                        f"the setting is unchanged. Relaunch from scratch instead.")
            print(f"    manifest_digest is a hash of the unit list; manifest_skip "
                  f"({_g.get('manifest_skip')!r}) and the selection file determine that list and "
                  f"manifest_skip is compared, so the population is checked by its inputs.")
        # NAMED OVERRIDES, NOT A BLANKET ONE. --resume-allow-drift accepts every difference,
        # including the one you did not know you had made -- which is how a resume that was meant
        # to change only the lr can also silently change the crop. --resume-override names the
        # fields, so an unnamed difference is still refused.
        _ovr = {f.strip() for f in a.resume_override.split(",") if f.strip()}
        _unknown_ovr = sorted(_ovr - set(_mine))
        if _unknown_ovr:
            raise SystemExit(
                f"STOP: --resume-override names {len(_unknown_ovr)} field(s) the guard does not "
                f"have: {_unknown_ovr}.\n"
                f"  A typo here protects nothing -- the field it was meant to allow stays refused "
                f"and the one it mistyped is not a field at all.\n"
                f"  The guard's fields are: {', '.join(sorted(_mine))}")
        # A STALE OVERRIDE IS ALSO AN ERROR. A name left on the command line after the setting it
        # covered went back to matching still disarms the guard for that field on the NEXT resume
        # that reuses the line, and nothing would say so.
        _stale = sorted(_ovr - set(_bad))
        if _stale and not a.resume_allow_drift:
            raise SystemExit(
                f"STOP: --resume-override names {len(_stale)} field(s) that do not differ from the "
                f"bundle: {_stale}.\n"
                f"  Drop them. Left in place they disarm the guard for those fields on the next "
                f"resume that reuses this command line.")
        _unnamed = {k: v for k, v in _bad.items() if k not in _ovr}
        if _unnamed and not a.resume_allow_drift:
            raise SystemExit(
                "STOP: this resume bundle was written under different settings:\n"
                + "".join(f"  {k}: bundle {b!r} vs this run {m!r}\n"
                          for k, (b, m) in _unnamed.items())
                + "  Changes to world/seed/accum/nsample/manifest_digest/crop_size move which poses\n"
                  "  land in which batch, so the saved AdamW moments belong to a different problem.\n"
                  "  Changes to the alphas, drop_pde, clip_norm, lr or warmup change the objective or\n"
                  "  the schedule, so continuing would blend two experiments into one curve.\n"
                  "  Match the settings, relaunch from scratch, or accept these deliberately with\n"
                  f"    --resume-override {','.join(sorted(_unnamed))} --resume-note '<why>'\n"
                  "  (--resume-allow-drift accepts everything at once, including what you did not\n"
                  "  intend). Either way it is recorded in resume_drift.json and carried forward in\n"
                  "  every bundle this run writes.")
        # AN UNEXPLAINED DEVIATION IS THE ONE A LATER READER CANNOT TELL FROM A MISTAKE --
        # EXCEPT UNDER --loss-only, WHICH IS NOT A DEVIATION AT ALL. A measurement does not
        # continue the bundle's schedule; it reads the bundle's weights, writes no checkpoint and
        # takes no step, so there is nothing for a later reader to mistake. The same reasoning
        # already exempts it from the epoch-count refusal below.
        #
        # THIS WAS LEARNED THE EXPENSIVE WAY, on the live run. The epoch-7 held-out pass died
        # with "accepts 5 changed setting(s) (lr, manifest_digest, pae_scored_pairs,
        # warmup_steps, world) and --resume-note is empty" -- a held-out fold legitimately
        # differs from the training bundle in all five. Passing the note from the subprocess
        # command does NOT fix it: the trainer that spawns that subprocess has its own code
        # loaded in memory and keeps sending the old argument list, while the subprocess reads
        # this file fresh from disk. Any mandatory flag added here breaks every RUNNING trainer's
        # monitor the moment the file is saved. A measurement must never be able to do that.
        if _bad and not a.resume_note.strip() and not a.loss_only:
            raise SystemExit(
                f"STOP: this resume accepts {len(_bad)} changed setting(s) "
                f"({', '.join(sorted(_bad))}) and --resume-note is empty.\n"
                f"  Say why in one line. It is written to resume_drift.json and travels in every "
                f"bundle this run writes, which is what lets the next reader tell a deliberate "
                f"fork from a mismatch nobody noticed.")
        if _bad:
            _via = "--resume-allow-drift" if (a.resume_allow_drift and _unnamed) \
                else "--resume-override"
            print(f"ACCEPTING RESUME DRIFT ({_via}); this run is a FORK of the bundle, "
                  f"not a continuation of it:")
            for k, (b, m) in _bad.items():
                print(f"  {k}: bundle {b!r} -> this run {m!r}"
                      + ("" if k in _ovr else "   <- NOT NAMED, accepted by the blanket flag"))
            print(f"  why: {a.resume_note.strip()}")
            # CARRIED FORWARD, NOT JUST PRINTED. A fork of a fork must still be readable as one,
            # so this run's deviation is appended to whatever history the bundle already carried
            # and the whole chain goes into every bundle this run writes.
            _drift_history.append({
                "from_bundle": os.path.abspath(a.resume),
                "bundle_commit": _g.get("commit", "?"),
                "bundle_written": _g.get("written", "?"),
                "bundle_epoch": int(_b["epoch"]), "bundle_step": int(_b["step"]),
                "accepted_at": rt._dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "via": _via,
                "note": a.resume_note.strip(),
                "fields": {k: {"bundle": b, "this_run": m} for k, (b, m) in _bad.items()},
                "named": sorted(_ovr & set(_bad)),
                "unnamed": sorted(_unnamed),
            })
            if rt.is_main:
                _dp = os.path.join(a.out, "resume_drift.json")
                try:
                    with open(_dp, "w") as _fh:
                        json.dump(_drift_history, _fh, indent=1, default=str)
                    print(f"  recorded -> {_dp}")
                except Exception as _e:
                    print(f"  (could not write {_dp}: {type(_e).__name__}: {_e})")
        # NOT UNDER --loss-only. A measurement does not continue the bundle's schedule, it reads
        # the bundle's weights: "the bundle has 141 epochs done and --epochs is 1" is a refusal
        # about training that is not happening, and the epoch counter below is reset to 0 so the
        # single pass actually runs rather than falling into range(141, 1) and measuring nothing.
        if not a.loss_only and _b["epoch"] >= max(1, math.ceil(a.epochs)):
            raise SystemExit(f"STOP: the bundle has {_b['epoch']} epoch(s) done and --epochs is "
                             f"{a.epochs}; there is nothing left to run.")
        # A --train-only BUNDLE CARRIES ONLY THE TENSORS IT TRAINED, which is correct: the ones it
        # omits were never updated, so they are still the shipped head's. But the held-out monitor
        # spawns itself WITHOUT --train-only -- the running trainer builds that command from its
        # own in-memory arguments, so adding the flag there cannot reach a trainer already running
        # -- and its trainable set is therefore all 248. This guard then refused every measurement
        # of a --train-only run: "the bundle is missing 245 trainable tensor(s)". It killed panel
        # 2, the only panel that matters, on arm ggfhj5 at epoch 1.
        #
        # SO INHERIT THE BUNDLE'S OWN SET, and only under --loss-only, where there is no optimiser
        # to keep consistent and no checkpoint to write. THIS IS NOT A BLANKET WEAKENING: the
        # bundle must carry EXACTLY what its own recorded train_only selects, so a TRUNCATED
        # bundle -- the case this check exists for -- is still refused.
        _b_only = (_b.get("guard") or {}).get("train_only") or ""
        if a.loss_only and _b_only and not a.train_only:
            import re as _re_i
            _pfx_i = rt.prefixes[0]
            _pats_i = [t.strip() for t in _b_only.split(",") if t.strip()]
            _want_i = {_n for _n in trainable
                       if any(_re_i.search(_p, _n[len(_pfx_i) + 1:]) for _p in _pats_i)}
            if _want_i and _want_i == set(_b["head"]):
                trainable = sorted(_want_i)
                if rt.is_main:
                    print(f"  --loss-only: inheriting the bundle's --train-only '{_b_only}'. It "
                          f"carries exactly the {len(_want_i)} tensor(s) that selects; the rest of "
                          f"the head was never trained and stays at the shipped weights, which is "
                          f"the head this measures.")
        _tset = set(trainable)
        _missing = _tset - set(_b["head"])
        if _missing:
            raise SystemExit(f"STOP: the bundle is missing {len(_missing)} trainable tensor(s), "
                             f"e.g. {sorted(_missing)[:3]}")
        with torch.no_grad():
            for _n, _p in model.named_parameters():
                if _n in _tset:
                    _p.data.copy_(_b["head"][_n].to(_p.device, _p.dtype))
        # THE OPTIMISER IS NOT RESTORED FOR A MEASUREMENT. AdamW never steps under --loss-only, so
        # its moments are dead weight -- and after inheriting a 3-tensor set above, the live param
        # group still holds 248, so load_state_dict would raise on the size mismatch.
        if not a.loss_only:
            opt.load_state_dict(_b["opt"])
        start_epoch, start_step = int(_b["epoch"]), int(_b["step"])
        if a.loss_only:
            _measured_epoch, _measured_step = start_epoch, start_step
            start_epoch, start_step = 0, 0
        if rt.is_main:
            # SAY WHAT ACTUALLY HAPPENED TO THE OPTIMISER. This line claimed "AdamW state for 298
            # tensor(s) restored" on a --loss-only pass that deliberately does NOT restore it --
            # a log that misreports its own run is worse than a quiet one.
            _opt_note = (f"AdamW state for {len(_b['opt'].get('state', {}))} tensor(s) restored"
                         if not a.loss_only else
                         "AdamW state NOT restored (measurement takes no step)")
            print(f"RESUMED from {a.resume}: {start_epoch} epoch(s) done, step {start_step}, "
                  f"{_opt_note}; "
                  f"continuing at epoch {start_epoch + 1}/{max(1, math.ceil(a.epochs))}")
            print(f"  bundle commit {_g.get('commit', '?')[:8]}, written {_g.get('written', '?')}")

    # THE L2-SP ANCHOR IS THE SHIPPED HEAD, AND ONLY A FRESH RUN STILL HOLDS IT. Set up AFTER the
    # resume block on purpose: at this point a fresh run's weights are untouched (init == shipped)
    # and can be snapshotted, while a resumed run's weights are 60 epochs of training away from
    # shipped and must read the anchor back from disk instead.
    _l2sp_anchor, _l2sp_params = None, []
    _l2sp_drift = {"sq": 0.0, "n": 0}

    # EVERY LOCAL THIS STAGE BOUND GOES ONTO THE CONTEXT. Inside main() these were simply
    # still in scope for everything below; here the hand-off has to be explicit. Written
    # out by name rather than with vars() so the set is visible and a rename fails loudly.
    for _n in ('_PRE_CROP', '_b', '_b_only', '_bad', '_dp', '_drift_history', '_fallback', '_fh', '_from_args', '_g', '_l2sp_anchor', '_l2sp_drift', '_l2sp_params', '_measured_epoch', '_measured_step', '_migrated', '_mine', '_missing', '_n', '_opt_note', '_ovr', '_p', '_pats_i', '_pfx_i', '_re_i', '_side', '_stale', '_tset', '_unknown', '_unknown_ovr', '_unnamed', '_via', '_want_i', 'b', 'f', 'k', 'load_resume', 'm', 'start_epoch', 'start_step', 't', 'trainable', 'v'):
        if _n in locals():
            setattr(rt, _n, locals()[_n])
    return start_epoch, start_step
