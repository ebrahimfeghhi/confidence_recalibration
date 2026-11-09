"""`trainer/bootstrap.py` -- build the confidence head and install everything that mutates it.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 1774-1935 (source `d34baebc`). The body
below is the original statements, comments included, with only the names they close over turned
into parameters. Nothing is reordered, because in this region ORDER IS THE CONTRACT:

  model.confidence_head.train()   before anything wraps it -- build_model() comes from the
                                  RESCORING helper and ends in .eval(); Protenix's head has an
                                  inference-only offload keyed on `not self.training` that moved
                                  pae/pde to the CPU above 2000 tokens and killed two runs 20-24
                                  minutes in, pointing at the loss rather than the mode flag.
  blocks_per_ckpt                 assigned explicitly, including 0 -> None. The first version
                                  skipped the assignment on 0, so Protenix's own config value of 1
                                  stood and checkpointing was ALWAYS on: every conclusion about
                                  what it cost was drawn from two identical configurations.
  lean_readouts.install()         FIRST of the forward-patch chain. It REPLACES
                                  memory_efficient_forward; pose_direction and pose_perblock each
                                  CAPTURE and wrap it. Installing the mirror later overwrites the
                                  outermost wrapper and drops the injections.
  --lean-single                   after install(), before the pose sites.
  pose_inblock / pose_perblock    mutually exclusive; both replace _prep_blocks.
  pose_direction                  before pose_perblock, whose orientation copies read its cache.
  pose_readout                    last; it refuses unless a stack injection already published the
                                  per-forward geometry.

THE DEPENDENCY LEDGER for this region, which is what the review asked for before any extraction:

  reads    a (parsed args), dev, RC.build_model, ProtenixLoss is NOT touched here
  returns  model, configs, dead_head
  mutates  model.confidence_head: .train(), blocks_per_ckpt, p_drop on every block,
           memory_efficient_forward and pairformer_stack._prep_blocks (patch chain),
           new modules added by the pose installers
  writes   stdout only
  refuses  6 SystemExit paths, preserved verbatim -- a refusal here is not advice
"""
import os
import time

import torch

import phase_j_rescore_cache as RC


def build_head(a, dev, t_cuda=None):
    """Build the model and install every head mutation this run asks for.

    Returns (model, configs, dead_head). `dead_head` is the tensor list --lean-single proved
    gradient-free, which the optimiser build downstream drops.
    """
    _t_cuda = time.time() if t_cuda is None else t_cuda
    try:
        model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    except Exception as _e:                       # noqa: BLE001
        # The traceback alone cost us the diagnosis last time. Say what the device state was at
        # the moment it failed, then re-raise unchanged so the caller's handling is untouched.
        print(f"  CHILD BUILD FAILED after {time.time() - _t_cuda:.2f}s: "
              f"{type(_e).__name__}: {_e}", flush=True)
        try:
            import torch as _t
            print(f"  at failure: device_count={_t.cuda.device_count()} "
                  f"is_available={_t.cuda.is_available()} "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
        except Exception:                         # noqa: BLE001
            pass
        raise
    if a.loss_only:
        print(f"  child built model in {time.time() - _t_cuda:.2f}s", flush=True)
    configs.train_confidence_only = True
    configs.loss.weight.alpha_pae = float(a.alpha_pae)
    configs.loss.weight.alpha_confidence = float(a.alpha_confidence)

    # THE HEAD WE ARE TRAINING MUST BE IN TRAINING MODE.
    #
    # build_model() comes from phase_j_rescore_cache, which exists to RESCORE, so it ends with
    # model.eval(). Training through it left confidence_head.training == False, and Protenix's
    # confidence head has an inference-only memory optimisation keyed on exactly that flag
    # (modules/confidence.py:233): above PROTENIX_CONF_OFFLOAD_TOKENS tokens (default 2000) and when
    # `not self.training`, it moves pae and pde to the CPU and empties the cache. The loss then gets
    # two of its four logit tensors on the CPU while the labels are on the GPU and dies inside
    # Protenix's PDE loss as "cuda:2 and cpu" -- pointing at the loss rather than at the mode flag.
    # Data-dependent, so it killed two runs 20-24 min in, once the shuffle reached a structure over
    # 2000 tokens (9dzf).
    #
    # Switching the mode changes NO ARITHMETIC, which is why the steps already taken were valid:
    # the head's four Dropout modules all have p=0 and it holds no BatchNorm, so train() and eval()
    # compute the same function. The trunk stays in eval, which is what FT3 requires.
    model.confidence_head.train()
    # ACTIVATION CHECKPOINTING ON THE HEAD'S PAIR STACK. Set here rather than at construction,
    # because build_model comes from the rescoring helper. PairformerStack reads this attribute at
    # forward time and ignores it when grad is off, so the evaluator is unaffected.
    # 0 DISABLES IT, rather than silently leaving whatever the config set. The first version was
    # `if a.blocks_per_ckpt:`, which skipped the assignment on 0 -- and Protenix's config already
    # sets blocks_per_ckpt=1, so checkpointing was ALWAYS on, --blocks-per-ckpt 1 was a no-op, and
    # --blocks-per-ckpt 0 did not turn it off. Every conclusion drawn about what checkpointing cost
    # or saved was drawn from two identical configurations.
    model.confidence_head.pairformer_stack.blocks_per_ckpt = (a.blocks_per_ckpt
                                                              if a.blocks_per_ckpt else None)
    _nblk = len(getattr(model.confidence_head.pairformer_stack, "blocks", []) or [])
    print(f"pair stack: {_nblk} block(s), blocks_per_ckpt="
          f"{model.confidence_head.pairformer_stack.blocks_per_ckpt}, chunking off")
    print(f"confidence_head.training = {model.confidence_head.training} "
          f"(trunk stays eval; train_confidence_only={configs.train_confidence_only})")
    # THE THIRD POSE PATHWAY, INSTALLED BEFORE head_param_names() READS THE MODULE TREE. That
    # function derives the trainable set from the tree rather than from a hand-written list, so the
    # new projection is picked up automatically -- but only if it exists by then. Installed after
    # .train() and the blocks_per_ckpt assignment so neither is applied to a module that is about
    # to be wrapped.
    # INSTALLED BEFORE BOTH POSE PATHWAYS, AND THAT ORDER IS LOAD-BEARING.
    #
    # pose_direction.py:234 and pose_perblock.py:232 each capture
    # `orig_mef = head.memory_efficient_forward` and REPLACE the attribute with a wrapper that
    # calls what it captured. They are a chain, not independent patches. Installing the lean
    # mirror afterwards would overwrite the outermost wrapper and silently drop the orientation
    # and per-block injections -- except it would not even be silent: the mirror still calls
    # self.pairformer_stack, whose _prep_blocks is also wrapped and raises _NoPose when the
    # one-hot cache is unset, so the run would die at pose 0 instead. Installing FIRST makes the
    # mirror the base of the chain, which is what it is a mirror of. Found in review.
    if a.pair_dropout >= 0.0:
        _blocks = list(getattr(model.confidence_head.pairformer_stack, "blocks", []))
        if not _blocks:
            raise SystemExit("STOP: --pair-dropout: the confidence head's pairformer_stack has "
                             "no .blocks, so there is nothing to set the rate on.")
        _was = sorted({float(getattr(b, "p_drop", -1)) for b in _blocks})
        for _b in _blocks:
            # p_drop IS THE ONE THAT MATTERS: all four pair-rep sites in PairformerBlock pass
            # self.p_drop to dropout_add_rowwise. dropout_row is the unused non-fused path; it is
            # set too so the two cannot disagree if a refactor ever switches paths.
            _b.p_drop = float(a.pair_dropout)
            _dr = getattr(_b, "dropout_row", None)
            if _dr is not None:
                for _obj in (_dr, getattr(_dr, "dropout", None)):
                    for _attr in ("p", "r"):
                        if _obj is not None and hasattr(_obj, _attr):
                            setattr(_obj, _attr, float(a.pair_dropout))
        _now = sorted({float(b.p_drop) for b in _blocks})
        if _now != [float(a.pair_dropout)]:
            raise SystemExit(f"STOP: --pair-dropout did not take: p_drop is {_now} after "
                             f"assignment, not [{a.pair_dropout}].")
        print(f"PAIR DROPOUT: row-wise rate on z set to {a.pair_dropout:g} on {len(_blocks)} "
              f"pairformer block(s), was {_was}. "
              f"confidence_head.training={model.confidence_head.training} "
              f"-- dropout is a NO-OP unless that is True")
    if a.lean_readouts:
        import lean_readouts
        _lr_sha = lean_readouts.install(model, pae_only=a.pae_only)
        print(f"LEAN READOUTS: pde, pLDDT and resolved are NOT COMPUTED. "
              f"memory_efficient_forward pinned at {_lr_sha[:16]}")
    # THE COMPUTE THAT FED THE DEAD READOUTS, which --lean-readouts did not touch. See
    # lean_readouts.drop_single_pathway for why z never depends on s and why this is an attribute
    # flip rather than a rebuild. MUST run after install(), because install() refuses to be the top
    # of the patch chain and this reads the same module tree.
    _dead_head = []
    if a.lean_single:
        import lean_readouts as _LS
        if not (a.lean_readouts and a.pae_only):
            raise SystemExit(
                "STOP: --lean-single requires --lean-readouts AND --pae-only. It turns off the "
                "pairformer's single track, which is what feeds pLDDT and resolved; without both "
                "of those the two heads ARE trained and would be trained against a representation "
                "that is no longer updated -- silently, since the shapes still line up.")
        _n_blk, _n_dead = _LS.drop_single_pathway(model)
        _dead_head = _LS.dead_head_params(model.confidence_head, single_dropped=True)
        print(f"LEAN SINGLE: attention_pair_bias and single_transition are NOT CALLED in "
              f"{_n_blk} pairformer block(s) (c_s set to 0; the modules stay in the tree, so the "
              f"checkpoint's key set is unchanged). {_n_dead/1e6:.3f}M head parameter(s) across "
              f"{len(_dead_head)} tensor(s) now provably receive no gradient; they are dropped "
              f"from the optimiser below. PairformerBlock.forward pinned at "
              f"{_LS.PINNED_BLOCK_SHA[:16]}, PairformerStack.forward at "
              f"{_LS.PINNED_STACK_SHA[:16]}")
    # AFTER lean_readouts AND --lean-single, BEFORE pose_direction/pose_perblock. It wraps
    # memory_efficient_forward and replaces the stack's _prep_blocks, so lean_readouts -- which
    # REPLACES that attribute rather than wrapping it -- must already be in place; its refusal list
    # carries this module's marker so the wrong order is a hard stop.
    if a.pose_inblock:
        # AND NOT ALONGSIDE --pose-perblock. Both replace the stack's _prep_blocks, so whichever
        # installs second wins and the other's injection silently disappears -- pose_perblock also
        # raises _NoPose from its own _prep_blocks if it is the loser, which is a symptom and not a
        # check. This combination has not been reasoned through and is refused rather than ordered.
        if a.pose_perblock:
            raise SystemExit(
                "STOP: --pose-inblock with --pose-perblock. Both replace "
                "pairformer_stack._prep_blocks, so one of the two injections is silently dropped "
                "depending on install order. If you want the pose at every depth AND inside the "
                "last block, that is a third design and needs its own module.")
        import pose_inblock
        _pib_blk = a.pose_inblock_block
        if _pib_blk < 0:
            _pib_blk = len(model.confidence_head.pairformer_stack.blocks) - 1
        _pib = pose_inblock.install(model, block_index=_pib_blk, pae_inject=a.pose_inblock_pae)
    elif a.pose_inblock_pae:
        raise SystemExit("STOP: --pose-inblock-pae without --pose-inblock. The readout injection "
                         "shares the pose geometry the in-block wrapper computes and stashes, so "
                         "on its own it would never fire and the flag would be a silent no-op.")
    if a.frame_direction:
        import pose_direction
        pose_direction.install(model)
    if a.pose_perblock:
        import pose_perblock
        pose_perblock.install(model, tied=a.pose_perblock_tied,
                              share_pretrained=a.pose_perblock_share,
                              copy_pretrained=a.pose_perblock_copy)
    elif a.pose_perblock_tied or a.pose_perblock_share or a.pose_perblock_copy:
        raise SystemExit("STOP: --pose-perblock-tied/-share/-copy without --pose-perblock does "
                         "nothing. They select how A shares its weights, not whether A is "
                         "installed.")
    if a.pose_readout:
        # AFTER both stack injections, because it refuses unless one of them is installed: it
        # reads their per-forward geometry rather than recomputing cdist and the one-hot.
        import pose_readout
        _prd = pose_readout.install(model, copy_distance=a.pose_readout_copy_distance)
    elif a.pose_readout_copy_distance:
        raise SystemExit("STOP: --pose-readout-copy-distance without --pose-readout. It selects "
                         "how the fifth site is initialised, not whether it exists.")
    return model, configs, _dead_head
