"""`trainer/trainable.py` -- Derive the trainable tensor set from the module tree, then apply --train-only.

EXTRACTED VERBATIM from `phases/phase_k_train.py` lines 2906-3059 (source `d34baebc`).
Statements and comments are the original; only the names this closed over inside
main() became parameters. test_trainable_equivalence.py compares it line for line,
so a change to the live trainer fails the test instead of drifting silently.
"""
import os
import math
import time

import torch
import torch.distributed as dist


def select_trainable(rt, a, model):
    prefixes, trainable = rt.head_param_names(model)
    # LoRA REPLACES THE TRAINABLE SET, it does not add to it. Applied to the confidence head only
    # -- head_param_names already established that module is the honest boundary -- and after the
    # head is built so the adapters inherit its device and dtype.
    if a.train_only:
        import re as _re
        _pats = [t.strip() for t in a.train_only.split(",") if t.strip()]
        _pfx = prefixes[0]
        _rel = {n: n[len(_pfx) + 1:] for n in trainable}
        _kept = [n for n in trainable if any(_re.search(_p, _rel[n]) for _p in _pats)]
        if not _kept:
            raise SystemExit(
                f"STOP: --train-only matched no parameter, so the optimiser would be empty and "
                f"the run would report success while training nothing.\n"
                f"  patterns: {_pats}\n"
                f"  head has {len(trainable)} tensor(s), e.g. "
                f"{[_rel[n] for n in trainable[:6]]}")
        _before = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        trainable = _kept
        _after = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        print(f"--train-only {_pats}: {len(_kept)} of {len(_rel)} head tensor(s), "
              f"{_after:,} of {_before:,} parameters ({100.0*_after/max(1,_before):.3f}% of the "
              f"head). Everything else is FROZEN.")
        for n in _kept:
            print(f"    {_rel[n]:46} {tuple(dict(model.named_parameters())[n].shape)}")

    # A LoRA BUNDLE HOLDS ADAPTERS, NOT HEAD TENSORS, and the held-out monitor spawns itself
    # without --lora-rank -- the running trainer builds that command from its in-memory arguments,
    # so adding the flag there cannot reach a trainer already running. Its trainable set was
    # therefore the 248 plain head tensors, none of which a rank-8 bundle contains, and every
    # measurement died on
    #     STOP: the bundle is missing 248 trainable tensor(s)
    # Arm m8rote trained three epochs and produced ZERO held-out points before this was caught --
    # the same failure as ggfhj5 epoch 1, which was --train-only instead of LoRA.
    #
    # So under --loss-only, inherit the bundle's own LoRA configuration. The guard already records
    # all four fields for exactly this reason ("resuming one as the other fails on the
    # missing-tensor check"). Only when this run named no rank of its own, so an explicit
    # --lora-rank still wins and a non-LoRA bundle is untouched.
    # THE SAME CLASS OF BUG, FOURTH INSTANCE. --train-only, then LoRA, then the evaluator's
    # --pose-perblock-share, each broke because a setting the trainer passes was not also inherited
    # or forwarded. The ranking fields are in the guard because they change the objective, so a
    # --loss-only monitor that does not name them is refused on a rank-loss bundle -- and the
    # monitor CANNOT name them, because a running trainer keeps sending the argument list its own
    # loaded code built. So inherit them here instead. The ranking term is skipped under
    # --loss-only anyway (there is no backward to add it to), so this only quiets the guard.
    if a.loss_only and a.resume and not a.rank_loss:
        try:
            _gr = (torch.load(a.resume, map_location="cpu", weights_only=False).get("guard")
                   or {})
        except Exception as _e:                              # noqa: BLE001
            _gr = {}
            print(f"  (could not read {a.resume} for its ranking config: {type(_e).__name__})")
        if _gr.get("rank_loss"):
            a.rank_loss = True
            a.alpha_rank = float(_gr.get("alpha_rank") or 0.0)
            a.rank_weight_by_gap = bool(_gr.get("rank_weight_by_gap", True))
            if rt.is_main:
                print(f"  --loss-only: inheriting the bundle's ranking configuration "
                      f"(alpha_rank {a.alpha_rank:g}, "
                      f"weight_by_gap={a.rank_weight_by_gap}) so the guard matches; the ranking "
                      f"term itself does not run -- a measurement takes no step.")

    if a.loss_only and a.resume and not a.lora_rank:
        try:
            _g = (torch.load(a.resume, map_location="cpu", weights_only=False).get("guard")
                  or {})
        except Exception as _e:                              # noqa: BLE001
            _g = {}
            print(f"  (could not read {a.resume} for its LoRA config: {type(_e).__name__})")
        if int(_g.get("lora_rank") or 0):
            a.lora_rank = int(_g["lora_rank"])
            a.lora_alpha = float(_g.get("lora_alpha") or 0.0)
            a.lora_targets = _g.get("lora_targets") or ""
            a.lora_dropout = float(_g.get("lora_dropout") or 0.0)
            if rt.is_main:
                print(f"  --loss-only: inheriting the bundle's LoRA configuration "
                      f"(rank {a.lora_rank}, alpha {a.lora_alpha:g}, dropout {a.lora_dropout:g}) "
                      f"-- it holds adapter tensors, so the model must carry the same adapters "
                      f"for its weights to load at all.")

    lora_report = None
    if a.lora_rank:
        import lora as LORA
        _hm = dict(model.named_modules())[prefixes[0]]
        _lnames, lora_report = LORA.apply_lora(
            _hm, a.lora_rank, a.lora_alpha or 2.0 * a.lora_rank,
            [t.strip() for t in a.lora_targets.split(",") if t.strip()], a.lora_dropout)
        trainable = [f"{prefixes[0]}.{n}" for n in _lnames]

        print(f"LoRA rank {a.lora_rank} alpha {a.lora_alpha or 2.0*a.lora_rank:g} "
              f"dropout {a.lora_dropout:g}: {lora_report['modules_adapted']} module(s) adapted, "
              f"{lora_report['lora_params']/1e6:.3f}M adapter parameters against "
              f"{lora_report['frozen_params']/1e6:.2f}M frozen "
              f"({100*lora_report['fraction_of_head']:.2f}% of the head)")
        if lora_report["skipped_rank_too_high"]:
            print(f"  {len(lora_report['skipped_rank_too_high'])} module(s) skipped: rank "
                  f"{a.lora_rank} exceeds a dimension, e.g. "
                  f"{lora_report['skipped_rank_too_high'][:3]}")
        if lora_report["adapted_with_bias"]:
            print(f"  NOTE {len(lora_report['adapted_with_bias'])} adapted module(s) have a bias; "
                  f"it stays FROZEN -- the adapter is the only thing that moves.")
    # THE DEAD HEAD PARAMETERS COME OUT LAST, so --train-only and LoRA cannot put them back.
    #
    # WHY IT IS SAFE RATHER THAN MERELY CHEAP. Nothing reads these tensors' outputs, so autograd
    # produces no gradient for them -- which _apply_step then turns into an explicit zeros tensor
    # ("a rank with no gradient for a parameter still has to contribute a tensor, or the
    # collective's shapes do not line up"). So today they are all-reduced as zeros, AdamW allocates
    # exp_avg and exp_avg_sq for them, and with weight_decay=0.0 their update is exactly zero. They
    # are therefore bit-identical whether they are in the optimiser or not; removing them removes
    # ~84 MB of moment state and ~42 MB of the 79 MB flat collective buffer per step and NOTHING
    # else. Verified on CPU by run_logging/test_lean_single.py, which checks the single pathway's
    # parameters get .grad is None with the track ON as well as off.
    #
    # AND THE REPORTED NUMBER BECOMES HONEST. Every --pae-only arm in this project has printed
    # "258 trainable tensor(s), 19.8M of 464.5M parameters (4.3%)" while 158 of those tensors
    # received no gradient. The --grad-probe path already said so out loud and nothing acted on it.
    if rt._dead_head:
        _dead_abs = {f"{prefixes[0]}.{n}" for n in rt._dead_head}
        _before_n, _before_p = len(trainable), sum(
            p.numel() for n, p in model.named_parameters() if n in set(trainable))
        trainable = [n for n in trainable if n not in _dead_abs]
        if not trainable:
            raise SystemExit(
                "STOP: --lean-single left the trainable set EMPTY. Every tensor --train-only or "
                "LoRA selected is one nothing reads under --pae-only, so the run would report "
                "success while training nothing. Widen --train-only or drop --lean-single.")
        _after_p = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
        print(f"--lean-single: {_before_n - len(trainable)} of {_before_n} trainable tensor(s) "
              f"dropped from the optimiser as provably gradient-free "
              f"({(_before_p - _after_p)/1e6:.3f}M of {_before_p/1e6:.3f}M parameters, "
              f"{100.0 * (_before_p - _after_p) / max(1, _before_p):.2f}%). The remaining "
              f"{len(trainable)} tensor(s) / {_after_p/1e6:.3f}M parameters are the EFFECTIVE "
              f"capacity -- which is what the line below now prints.")
    head_ids = {id(p) for n, p in model.named_parameters() if n in set(trainable)}
    n_head = sum(p.numel() for n, p in model.named_parameters() if n in set(trainable))
    n_all = sum(p.numel() for _n, p in model.named_parameters())
    print(f"trainable: {len(trainable)} tensor(s), {n_head/1e6:.1f}M of {n_all/1e6:.1f}M parameters "
          f"({n_head/n_all:.1%}), under {prefixes}")
    for n, p in model.named_parameters():
        p.requires_grad_(id(p) in head_ids)

    # ---- DEAD OPERATION ELIMINATION ------------------------------------------------------------
    # INSTALLED HERE, AND HERE FOR THE SAME REASON THE PREFIX CACHE IS: its central refusal is "a
    # module I would stub is TRAINABLE", read from requires_grad on the tensors, and requires_grad
    # is not final until the line above has applied --train-only, LoRA and --lean-single. Any
    # earlier and it would be checking a flag's intent instead of the model's state.
    #
    # BEFORE THE PREFIX CACHE, though either order is correct -- the husks leave named_parameters()
    # and named_modules() unchanged and move no value, so the cache's prefix set, its dropout scan
    # and its digest are identical either way. This order only means a dead_ops refusal happens
    # before the cache turns on, and that the cache's log line describes the tree that will run.
    # phases/dead_ops.py's docstring states the full order.
    _dops = None
    return prefixes, trainable, _dops
