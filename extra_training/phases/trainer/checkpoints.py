"""Resume-guard comparison: may this bundle continue this run, or is it a fork?

WHY THIS IS THE PIECE WORTH EXTRACTING. The resume block is 401 lines, but almost all of it is
loading tensors and printing. The DECISION -- which recorded settings disagree with the ones about
to run -- is the 15-line loop below, and it is pure dict logic over three inputs. It could not be
exercised without starting a training run; now it can be checked against every bundle on disk.

THREE PLACES A FIELD CAN COME FROM, in order:
  the bundle's own guard        the normal case
  the run's args.json beside it a bundle written by older code lacks fields the guard later grew;
                                absence is not drift, and the sidecar records the same evidence
  the pre-crop defaults         a field that POSTDATES the bundle. Absence is evidence, not
                                ignorance: a bundle written by code with no cropping was written
                                with cropping off, so it migrates to that default and is compared
                                like any other field. A resume that ENABLES cropping then shows up
                                as real drift, which is correct.
Anything in none of the three is reported as not compared, rather than silently passing.

Both tables are moved verbatim from phase_k_train.py; the comments record which run exposed each
case and are the reason the defaults are what they are.
"""


_PRE_CROP = {"lora_rank": 0, "lora_alpha": 0.0, "lora_targets": "", "lora_dropout": 0.0,
             # A bundle written before these flags existed ran the full-run schedule, which
             # is exactly what the 0 defaults mean, and "cosine" is what it computed.
             "lr_decay_start_epoch": 0.0, "lr_decay_end_epoch": 0.0,
             "lr_decay_shape": "cosine", "lr_steps": "",
             "pair_dropout": -1.0, "crop_tokens": "",
             "token_keep_union": -1.0, "token_keep_other": -1.0,
             # ABSENCE IS EVIDENCE: a bundle written before --pae-row-frac existed ran the
             # unweighted term, which is exactly what frac 0 means. The mass and floor
             # defaults are inert at frac 0 but are migrated so a resume that only turns
             # weighting ON still shows up as drift on the frac field.
             "pae_row_frac": 0.0, "pae_row_mass": 0.8, "pae_row_floor": 10,
         "train_only": "",
             "crop_margin": 0.0, "crop_contact": 5.0, "pae_scored_pairs": "",
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


def compare_guard(mine, bundle_guard, fallback, migrations=None):
    """Which settings disagree. Pure.

    mine          what this process is about to run with
    bundle_guard  the `guard` dict inside resume_ep<N>.pt
    fallback      the run's args.json sitting beside the bundle, for fields the bundle predates
    migrations    defaults for fields that postdate the bundle (default: _PRE_CROP)

    Returns (bad, from_args, unknown, migrated):
      bad        {field: (recorded, mine)} -- real disagreements; non-empty means FORK
      from_args  fields matched against args.json because the bundle lacked them
      unknown    fields recorded in none of the three, so NOT compared
      migrated   fields read as a pre-crop default
    """
    # The extracted loop refers to _mine / _g / _fallback / _PRE_CROP by their original names, so
    # they are bound here rather than renamed in the loop -- the loop is verbatim.
    _mine = mine
    _g, _fallback = bundle_guard or {}, fallback or {}
    _PRE_CROP_LOCAL = _PRE_CROP if migrations is None else migrations
    _bad, _from_args, _unknown, _migrated = {}, [], [], []
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
    return _bad, _from_args, _unknown, _migrated
