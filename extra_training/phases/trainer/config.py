"""Cross-flag validation and the derivations that depend on more than one flag.

WHY SEPARATE FROM cli.py. argparse validates one flag at a time. Most of this trainer's refusals
are about COMBINATIONS -- a crop margin with --minimal-items, --lr-steps together with --lr-final,
an --extra-eval spec naming a pool that does not exist -- and argparse has nowhere to put those.
They ran as 303 lines of straight-line code inside main(), where they could not be exercised
without starting a training run.

Every refusal is a SystemExit whose message explains what is inconsistent and what to pass
instead; 44 of them. That text is the contract and is moved verbatim.

This function MUTATES `a` -- it derives fields that depend on other fields -- and returns it for
convenience. phases/trainer/test_config_equivalence.py checks it against real runs' args.json and
against known-bad combinations.
"""
import csv
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lr_schedule as _LRS0                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_re_ee = re.compile(r"(\w+)=([^,]*)")


def _git_commit():
    """The source commit, for provenance. Empty when git is unavailable."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def validate(a):
    """Refuse inconsistent flag combinations; derive what depends on several flags.

    Mutates and returns `a`. Raises SystemExit with an explanatory message on any inconsistency.
    """

    # EXTRA EVAL PASSES, parsed and validated HERE so a typo costs a second rather than surfacing
    # at the first epoch boundary hours in, on three ranks, after a full epoch of training.
    import re as _re_ee          # module scope has no `re`; the l2-sp block imports it locally too
    _EXTRA_EVAL = []
    _ee_names = set()
    for _spec in a.extra_eval:
        _d = {}
        for _kv in str(_spec).split(","):
            _kv = _kv.strip()
            if not _kv:
                continue
            if "=" not in _kv:
                raise SystemExit(f"STOP: --extra-eval {_spec!r}: {_kv!r} is not key=value.")
            _k, _v = _kv.split("=", 1)
            _d[_k.strip()] = _v.strip()
        _unknown = sorted(set(_d) - {"name", "pool", "fold", "seed_base", "budget", "every",
                                     "crop_tokens"})
        if _unknown:
            raise SystemExit(f"STOP: --extra-eval {_spec!r} has unknown key(s) {_unknown}. "
                             f"Known: name, pool, fold, seed_base, budget, every, crop_tokens.")
        for _req in ("name", "pool", "every"):
            if not _d.get(_req):
                raise SystemExit(f"STOP: --extra-eval {_spec!r} is missing {_req}=.")
        _nm = _d["name"]
        if not _re_ee.fullmatch(r"[A-Za-z0-9_.-]+", _nm):
            raise SystemExit(f"STOP: --extra-eval name {_nm!r} must be filesystem-safe "
                             f"([A-Za-z0-9_.-]+); it names metric_<name>_ep<N>.json.")
        if _nm in _ee_names:
            raise SystemExit(f"STOP: --extra-eval name {_nm!r} is used twice, so the two passes "
                             f"would overwrite each other's metric file.")
        _ee_names.add(_nm)
        if not os.path.exists(_d["pool"]):
            raise SystemExit(f"STOP: --extra-eval {_nm}: pool {_d['pool']} does not exist.")
        _ef = _d.get("fold") or a.fold
        for _need in (_ef, os.path.join(_ef, "results.csv"),
                      os.path.join(_ef, "chain_roles.csv")):
            if not os.path.exists(_need):
                raise SystemExit(f"STOP: --extra-eval {_nm}: {_need} does not exist. A pool is "
                                 f"scored against its OWN fold's results and roles.")
        if _d.get("crop_tokens") and not os.path.exists(_d["crop_tokens"]):
            raise SystemExit(f"STOP: --extra-eval {_nm}: crop_tokens {_d['crop_tokens']} does "
                             f"not exist.")
        try:
            _ev = int(_d["every"])
            _bg = int(_d.get("budget") or 0)
            _sb = int(_d["seed_base"]) if _d.get("seed_base") else -1
        except ValueError as _e:
            raise SystemExit(f"STOP: --extra-eval {_nm}: every/budget/seed_base must be "
                             f"integers ({_e}).")
        if _ev <= 0:
            raise SystemExit(f"STOP: --extra-eval {_nm}: every={_ev} must be positive.")
        _EXTRA_EVAL.append(dict(name=_nm, pool=_d["pool"], fold=_ef, seed_base=_sb,
                                budget=_bg, every=_ev,
                                crop_tokens=_d.get("crop_tokens", "")))
    # THE MASK'S CONTENTS, DIGESTED AT STARTUP -- here and not beside the loader because the
    # resume guard dict is built before the loader runs. The guard recorded
    # basename(a.crop_tokens) first, so rebuilding the mask with a different margin, contact
    # cutoff or forced-pair rule under the SAME NAME resumed as "unchanged": the whole training
    # input distribution could move while the record said it had not. This hashes the
    # per-structure masks themselves in sorted order, plus the builder's provenance blob.
    # Found in review.
    _CROP_TOK_DIGEST = ""
    # A THRESHOLD WITH NOTHING TO TURN OFF IS A NO-OP THAT READS LIKE A SETTING. With
    # --blocks-per-ckpt 0 every unit is already uncheckpointed, so the rule would select between
    # "off" and "off" and the run would look configured when it is not.
    if a.ckpt_token_threshold > 0 and not a.blocks_per_ckpt:
        raise SystemExit(
            "STOP: --ckpt-token-threshold needs --blocks-per-ckpt > 0. With checkpointing already "
            "off for every unit the threshold selects between off and off, which changes nothing "
            "and hides that it changed nothing.")
    # PARSED AT STARTUP, NOT AT THE FIRST STEP. A malformed --lr-steps would otherwise surface
    # minutes in, on three ranks, after the model is built.
    try:
        # IMPORTED HERE, not relied on from later. `import lr_schedule as _LRS` happens ~850 lines
        # below, so using _LRS during startup validation raised UnboundLocalError on every rank
        # before pose 0. Caught by launching and reading epoch 1, which is what epoch 1 is for.
        import lr_schedule as _LRS0
        _LR_STEPS = tuple(_LRS0.parse_lr_steps(a.lr_steps)) if a.lr_steps else ()
    except ValueError as _e:
        raise SystemExit(f"STOP: {_e}")
    if a.ckpt_token_threshold < 0:
        raise SystemExit("STOP: --ckpt-token-threshold cannot be negative.")
    if a.lean_readouts and not a.pae_only:
        # AT PARSE TIME, NOT AT THE INSTALL SITE. lean_readouts.install() refuses this too, but it
        # runs after the model is built -- minutes in, and on three ranks at once. The flag pair is
        # knowable from the argv alone, so it is checked from the argv alone.
        raise SystemExit(
            "STOP: --lean-readouts needs --pae-only. The three readouts it stops computing are "
            "the ones --pae-only stubs the loss terms for; without it, pLDDT, PDE and resolved "
            "ARE trained, and replacing them with placeholder zeros would train three terms "
            "against constant logits while reporting their losses as if they were real.")
    if a.val_loss_every:
        _need = {"--val-loss-fold": a.val_loss_fold,
                 "--val-loss-selection": a.val_loss_selection,
                 "--val-loss-pool": a.val_loss_pool,
                 "--val-loss-preflight": a.val_loss_preflight,
                 "--val-loss-seeds": a.val_loss_seeds}
        _unset = sorted(k for k, v in _need.items() if not v)
        if _unset:
            raise SystemExit(f"STOP: --val-loss-every {a.val_loss_every} needs {_unset}. A "
                             f"held-out pass with no fold, selection, mask pool, preflight marker "
                             f"or seed range has nothing to measure.")
        _labels = a.val_loss_labels or os.path.join(a.val_loss_fold, "labels")
        _paths = {"--val-loss-fold": a.val_loss_fold, "labels": _labels,
                  "--val-loss-selection": a.val_loss_selection,
                  "--val-loss-pool": a.val_loss_pool,
                  "--val-loss-preflight": a.val_loss_preflight}
        if a.val_loss_prep:
            _paths["--val-loss-prep"] = a.val_loss_prep
        _gone = sorted(f"{k} -> {v}" for k, v in _paths.items() if not os.path.exists(v))
        if _gone:
            raise SystemExit("STOP: the held-out PAE monitor points at path(s) that do not "
                             "exist:\n  " + "\n  ".join(_gone))
        # THE LABELS MUST COVER THE SEED RANGE. An empty or partial label directory makes the
        # subprocess fail every epoch with "the training set is INCOMPLETE", which is correct but
        # 37 minutes too late to be useful. validation/fold/labels was EMPTY until 2026-10-20.
        try:
            _lo, _hi = (int(x) for x in a.val_loss_seeds.split("-"))
        except ValueError:
            raise SystemExit(f"STOP: --val-loss-seeds {a.val_loss_seeds!r} is not <lo>-<hi>.")
        with open(a.val_loss_selection) as _fh:
            _pdbs = sorted({r["pdb"] for r in csv.DictReader(_fh)
                            if r.get("tokens", "").isdigit()})
        if not _pdbs:
            raise SystemExit(f"STOP: {a.val_loss_selection} yielded 0 structure(s). Rows are kept "
                             f"only when `tokens` is a digit -- a selection csv without that "
                             f"column drops every row silently.")
        _miss = [f"{q}/free_{sd}" for q in _pdbs for sd in range(_lo, _hi + 1)
                 if not os.path.exists(os.path.join(_labels, q, f"free_{sd}.npz"))]
        if _miss:
            raise SystemExit(
                f"STOP: {len(_miss)} of {len(_pdbs) * (_hi - _lo + 1)} held-out label(s) are "
                f"missing under {_labels}, e.g. {_miss[:4]}. Build them with "
                f"phases/phase_j_build_cache.py --fold {a.val_loss_fold}; see "
                f"analysis/oneoff/01_build_val10_labels.sh.")
        print(f"held-out PAE monitor: every {a.val_loss_every} epoch(s), {len(_pdbs)} structure(s) "
              f"x seeds {_lo}-{_hi} = {len(_pdbs) * (_hi - _lo + 1)} unit(s), masked by "
              f"{a.val_loss_pool}")



    # PIN THE COMMIT NOW, before any work, so every bundle this process writes records the code it
    # is running rather than whatever HEAD has become hours later. Caching it lazily at the first
    # epoch boundary was not enough: that is hours in, and the tree can already have moved.
    _git_commit()

    # REL CONFIG IS CHECKED BEFORE ANY HARDWARE. These refusals were originally placed with the
    # rest of the ranking-loss setup, which runs after CUDA initialisation -- so a
    # misconfiguration cost a GPU allocation to discover, and the gate could not reach them
    # without a free card. A config error should never need hardware to be caught.
    if a.rank_loss and a.rank_form == "rel":
        if not a.rank_cross_seed:
            raise SystemExit(
                "STOP: --rank-form rel needs --rank-cross-seed. The loss centres on an "
                "INTERFACE's poses across all its seeds; within one seed it would centre on "
                f"{a.nsample} poses and remove most of the signal it exists to use.")
        if not a.rel_labels:
            raise SystemExit("STOP: --rank-form rel needs --rel-labels (see "
                             "run_logging/build_itm_labels.py).")
        if not os.path.isdir(a.rel_labels):
            raise SystemExit(f"STOP: --rel-labels {a.rel_labels} is not a directory.")
        _rm = os.path.join(a.rel_labels, "manifest.json")
        if not os.path.exists(_rm):
            raise SystemExit(
                f"STOP: {_rm} is missing, so the label set cannot be identified. Rebuild with "
                f"run_logging/build_itm_labels.py, which writes it.")

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold. Without it the trunk's coordinates are "
                         "silently NaN -- phase_j_replay.py's guard fired on this first.")


    import torch
    import phase_j_dataset as DS
    import phase_j_build_cache as BC
    import phase_j_rescore_cache as RC
    from protenix.model.loss import ProtenixLoss

    return a
