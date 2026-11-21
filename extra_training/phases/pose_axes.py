"""THE THREE AXES OF POSE INJECTION, and the ONE place flags become axes.

WHAT WAS WRONG. Pose injection has three independent axes:

  SITES  which places inject -- pairformer blocks 1..n-1 (block 0 is never wrapped: its
         injection already exists as the pretrained pre-stack linear_no_bias_d), the readout
         site BEFORE pae_ln, the readout site AFTER pae_ln.
  TERMS  which projections each site carries -- the binned one-hot distance (num_bins -> c_z),
         the raw distance (1 -> c_z), the frame-relative unit direction (3 -> c_z).
  INIT   zero, or a copy of block 0's pretrained injection.

The flags fused them into "modes", so legitimate points in that space were unexpressible. Plain
--pose-perblock meant "one-hot only, at zero"; --pose-perblock-copy meant "one-hot AND raw
distance, but ONLY at pretrained init". "Blocks carry one-hot AND raw distance, initialised to
zero" -- a one-variable comparison against the copy arm -- could not be asked for at all, and
pose_perblock's forward branched on head._pose_perblock_copy, an INIT-named flag deciding the
STRUCTURAL question of which terms get added. That is the defect this module removes: terms
decide the forward, init decides only the starting value.

A FOURTH AXIS, PARAMETER SHARING, is kept as it was: untied (one matrix per depth), tied (one
matrix for the whole stack) or shared (block 0's OWN tensors re-applied, zero new parameters).
It governs the DISTANCE terms only -- the orientation copies are one per depth in every mode,
exactly as before.

THE LEGACY FLAGS STILL WORK, as aliases desugared HERE and nowhere else. Reproducing an earlier
arm depends on that, so `normalise()` is the single normalisation point for both the trainer and
the evaluator, and it writes the LEGACY keys back onto the namespace as well as the axis keys so
that args.json keeps carrying both: the resume guard, run_index.py and the half-dozen
run_logging helpers that rebuild an eval argv from args.json all read the legacy keys, and a
silently different architecture in a child eval is the failure this project has paid for most
(commit 7d9d4d10).

WHY THE AXIS KEYS GO INTO args.json TOO, rather than the legacy keys alone: the legacy keys
CANNOT express the new points, so two different architectures would record identical flags. The
axis keys are the complete record, `argv()` is the one builder every forwarding path uses, and
`backfill()` reads the axes back out of an OLD args.json that has only the legacy keys.
"""

#: Canonical order, which is also the order every spelling is normalised to.
TERMS = ("onehot", "dist", "orient")
INITS = ("zero", "pretrained")
WEIGHTS = ("untied", "tied", "shared")
READOUT_SITES = ("pre", "post")

#: The distance terms. `orient` is not one of them: it has no pretrained source to copy (it
#: copies frame_dir_proj, which pose_direction zero-inits and verifies) and it is never tied.
DIST_TERMS = ("onehot", "dist")

#: The argparse dests this module owns, on both the trainer and the evaluator side.
AXIS_KEYS = ("pose_block_terms", "pose_block_init", "pose_block_weights",
             "pose_readout_sites", "pose_readout_terms", "pose_readout_init")

#: The legacy dests it desugars from and writes back.
LEGACY_KEYS = ("pose_perblock", "pose_perblock_tied", "pose_perblock_share",
               "pose_perblock_copy", "pose_readout", "pose_readout_prenorm",
               "pose_readout_post_zero", "pose_readout_copy_distance")

#: What a run with no pose injection at all records. Also the migration default for a resume
#: bundle written before these keys existed AND carrying no legacy pose flag either.
EMPTY = {"pose_block_terms": "", "pose_block_init": "zero", "pose_block_weights": "untied",
         "pose_readout_sites": "", "pose_readout_terms": "", "pose_readout_init": "zero"}


class PoseConfigError(Exception):
    """An illegal combination. The callers turn it into SystemExit; tests read the message."""


def _refuse(msg):
    raise PoseConfigError("STOP: " + msg)


def parse_terms(spec, flag):
    """A comma list of terms -> a canonical tuple, or a refusal naming the legal set."""
    items = [t.strip().lower() for t in str(spec or "").split(",") if t.strip()]
    if not items:
        _refuse(f"{flag} is empty. Pass the terms this site carries, e.g. "
                f"{flag} onehot,dist -- the legal terms are {', '.join(TERMS)} "
                f"(binned one-hot distance, raw distance, frame-relative unit direction).")
    bad = [t for t in items if t not in TERMS]
    if bad:
        _refuse(f"{flag}: unknown term(s) {bad}. The legal terms are {', '.join(TERMS)}.")
    dupes = sorted({t for t in items if items.count(t) > 1})
    if dupes:
        _refuse(f"{flag}: term(s) {dupes} given more than once. Each term is added ONCE per "
                f"site; repeating it in the list does not double it and is more likely a typo.")
    return tuple(t for t in TERMS if t in items)


def parse_readout_sites(spec):
    """A comma list of readout sites -> a canonical tuple, or a refusal."""
    items = [t.strip().lower() for t in str(spec or "").split(",") if t.strip()]
    if not items:
        _refuse("--pose-readout-sites is empty. Pass pre, post, or pre,post -- `pre` injects "
                "into pae_ln's INPUT (the layernorm then renormalises the sum, so the term can "
                "turn the pair feature but not scale it) and `post` into its OUTPUT.")
    bad = [t for t in items if t not in READOUT_SITES]
    if bad:
        _refuse(f"--pose-readout-sites: unknown site(s) {bad}. The legal sites are "
                f"{', '.join(READOUT_SITES)}.")
    dupes = sorted({t for t in items if items.count(t) > 1})
    if dupes:
        _refuse(f"--pose-readout-sites: site(s) {dupes} given more than once.")
    return tuple(s for s in READOUT_SITES if s in items)


def check_block_axes(terms, init, weights, have_direction):
    """Refuse an illegal point in the block space, naming what to pass instead."""
    terms = tuple(t for t in TERMS if t in set(terms))
    if not terms:
        _refuse("the per-block sites were asked for with no terms. Pass "
                "--pose-block-terms onehot[,dist][,orient].")
    if init not in INITS:
        _refuse(f"--pose-block-init {init!r} is not one of {', '.join(INITS)}.")
    if weights not in WEIGHTS:
        _refuse(f"--pose-block-weights {weights!r} is not one of {', '.join(WEIGHTS)}.")
    if "orient" in terms and not have_direction:
        _refuse("--pose-block-terms includes `orient` but --frame-direction is not on. The "
                "orientation geometry IS --frame-direction's: it computes the frame-relative "
                "unit direction and publishes it per forward, and these copies project it. "
                "Pass --frame-direction, or drop `orient` from the term list.")
    if weights == "shared":
        if tuple(t for t in DIST_TERMS if t in terms) != DIST_TERMS:
            _refuse("--pose-block-weights shared re-applies block 0's OWN injection, and block "
                    "0 adds BOTH terms -- linear_no_bias_d(one_hot) and "
                    "linear_no_bias_d_wo_onehot(distance). Re-adding only one of them would be "
                    "a different operator wearing the same name, and there are no new tensors "
                    "to leave out the other with. Pass --pose-block-terms onehot,dist (plus "
                    "orient if you want it), or use --pose-block-weights untied/tied.")
        if init != "zero":
            _refuse("--pose-block-weights shared with --pose-block-init pretrained is a "
                    "contradiction: shared adds NO new tensors, it re-applies block 0's own "
                    "pretrained ones, so there is nothing to initialise. Drop the init flag, or "
                    "pass --pose-block-weights untied --pose-block-init pretrained for one "
                    "pretrained COPY per depth (the old --pose-perblock-copy).")
    return terms


def check_readout_axes(sites, terms, init, have_direction):
    """Refuse an illegal point in the readout space, naming what to pass instead."""
    sites = tuple(s for s in READOUT_SITES if s in set(sites))
    terms = tuple(t for t in TERMS if t in set(terms))
    if not sites:
        _refuse("the readout was asked for with no site. Pass --pose-readout-sites pre, post, "
                "or pre,post.")
    if not terms:
        _refuse("the readout was asked for with no terms. Pass "
                "--pose-readout-terms onehot[,dist][,orient].")
    if init not in INITS:
        _refuse(f"--pose-readout-init {init!r} is not one of {', '.join(INITS)}.")
    if "orient" in terms and not have_direction:
        _refuse("--pose-readout-terms includes `orient` but --frame-direction is not on, so "
                "there is no orientation geometry to project. Pass --frame-direction, or drop "
                "`orient` from the term list.")
    if init == "pretrained" and not any(t in terms for t in DIST_TERMS):
        _refuse("--pose-readout-init pretrained copies block 0's DISTANCE injection, and the "
                "term list has neither onehot nor dist. There is no pretrained orientation "
                "matrix anywhere in this project to copy. Add a distance term or pass "
                "--pose-readout-init zero.")
    return sites, terms


# ----------------------------------------------------------------------------------------------
# THE DESUGARING. One table, one direction, one place.
# ----------------------------------------------------------------------------------------------
def legacy_block_axes(d):
    """The axes the legacy --pose-perblock* flags mean, from a dict-like of the legacy keys.

        --pose-perblock                      terms onehot            init zero        untied
        --pose-perblock --pose-perblock-tied terms onehot            init zero        tied
        --pose-perblock --pose-perblock-copy terms onehot,dist       init pretrained  untied
        --pose-perblock --pose-perblock-share terms onehot,dist      (no new tensors) shared

    `orient` joins the terms of EVERY one of those four whenever --frame-direction is on, which
    is exactly what pose_perblock.install did by inspecting whether pose_direction had installed.
    """
    if not d.get("pose_perblock"):
        return None
    share = bool(d.get("pose_perblock_share"))
    copy = bool(d.get("pose_perblock_copy"))
    tied = bool(d.get("pose_perblock_tied"))
    modes = [k for k, v in (("--pose-perblock-tied", tied), ("--pose-perblock-share", share),
                            ("--pose-perblock-copy", copy)) if v]
    if len(modes) > 1:
        _refuse(f"{' and '.join(modes)} are different experiments and at most one may be set. "
                f"In the new surface they are different points on three separate axes: "
                f"--pose-block-weights tied / shared and --pose-block-init pretrained.")
    terms = ["onehot"] + (["dist"] if (share or copy) else [])
    if d.get("frame_direction"):
        terms.append("orient")
    return {"terms": tuple(t for t in TERMS if t in terms),
            "init": "pretrained" if copy else "zero",
            "weights": "shared" if share else ("tied" if tied else "untied")}


def legacy_readout_axes(d):
    """The axes the legacy --pose-readout* flags mean.

        --pose-readout                                 sites post      terms onehot,dist
        --pose-readout --pose-readout-prenorm          sites pre       terms onehot,dist
        ... --pose-readout-post-zero                   sites pre,post  terms onehot,dist
        ... --pose-readout-copy-distance               init pretrained

    `orient` joins the terms whenever --frame-direction is on, as install() decided by
    inspection before. The init applies to the PRIMARY site only -- `pre` when it exists, else
    `post` -- and the second site is always zero, which is what --pose-readout-post-zero's name
    says and what pose_readout.install did.
    """
    if not d.get("pose_readout"):
        return None
    pre = bool(d.get("pose_readout_prenorm"))
    post_zero = bool(d.get("pose_readout_post_zero"))
    if post_zero and not pre:
        _refuse("--pose-readout-post-zero adds a SECOND site after pae_ln on top of the "
                "pre-norm one; without --pose-readout-prenorm there is no second site -- that "
                "is just the default post-LN arm. In the new surface the two sites are named "
                "directly: --pose-readout-sites pre,post.")
    sites = ("pre", "post") if (pre and post_zero) else (("pre",) if pre else ("post",))
    terms = ["onehot", "dist"] + (["orient"] if d.get("frame_direction") else [])
    return {"sites": sites, "terms": tuple(t for t in TERMS if t in terms),
            "init": "pretrained" if d.get("pose_readout_copy_distance") else "zero"}


def _new_block_given(d):
    # A NON-DEFAULT INIT OR WEIGHTS COUNTS AS ASKING, even with no terms: otherwise
    # `--pose-block-init pretrained` alone would fall through to the legacy branch and be a
    # silent no-op, which is the class of defect this file exists to remove.
    return (bool(d.get("pose_block_terms"))
            or str(d.get("pose_block_init") or "zero") != EMPTY["pose_block_init"]
            or str(d.get("pose_block_weights") or "untied") != EMPTY["pose_block_weights"])


def _legacy_block_given(d):
    return any(d.get(k) for k in ("pose_perblock", "pose_perblock_tied", "pose_perblock_share",
                                  "pose_perblock_copy"))


def _legacy_readout_given(d):
    return any(d.get(k) for k in ("pose_readout", "pose_readout_prenorm",
                                  "pose_readout_post_zero", "pose_readout_copy_distance"))


def _new_readout_given(d):
    # SAME REASON AS _new_block_given: a non-default init alone must not be silently inert.
    return (bool(d.get("pose_readout_sites")) or bool(d.get("pose_readout_terms"))
            or str(d.get("pose_readout_init") or "zero") != EMPTY["pose_readout_init"])


def resolve(d):
    """(block_axes | None, readout_axes | None) from a dict-like of BOTH flag surfaces.

    This is the whole decision procedure, and it is pure: `normalise()` is the thin wrapper that
    writes the answer back onto an argparse namespace.
    """
    have_dir = bool(d.get("frame_direction"))
    new_blk, old_blk = _new_block_given(d), _legacy_block_given(d)
    if new_blk and old_blk:
        _refuse("the per-block sites were asked for twice, through both surfaces. Pass either "
                "the legacy --pose-perblock[-tied|-copy|-share] OR the axes "
                "--pose-block-terms/--pose-block-init/--pose-block-weights, not both. "
                "--pose-perblock-copy is --pose-block-terms onehot,dist "
                "--pose-block-init pretrained; plain --pose-perblock is "
                "--pose-block-terms onehot.")
    if new_blk:
        if not d.get("pose_block_terms"):
            _refuse("--pose-block-init/--pose-block-weights select how the per-block sites are "
                    "built, not WHETHER they are. Pass --pose-block-terms too, or drop them.")
        blk = {"terms": parse_terms(d["pose_block_terms"], "--pose-block-terms"),
               "init": str(d.get("pose_block_init") or "zero"),
               "weights": str(d.get("pose_block_weights") or "untied")}
        blk["terms"] = check_block_axes(blk["terms"], blk["init"], blk["weights"], have_dir)
    else:
        if old_blk and not d.get("pose_perblock"):
            _refuse("--pose-perblock-tied/-share/-copy without --pose-perblock does nothing. "
                    "They select how the per-block sites share and initialise their weights, "
                    "not whether those sites exist.")
        blk = legacy_block_axes(d)
        if blk is not None:
            blk["terms"] = check_block_axes(blk["terms"], blk["init"], blk["weights"], have_dir)

    new_rd, old_rd = _new_readout_given(d), _legacy_readout_given(d)
    if new_rd and old_rd:
        _refuse("the readout site was asked for twice, through both surfaces. Pass either the "
                "legacy --pose-readout[-prenorm|-post-zero|-copy-distance] OR the axes "
                "--pose-readout-sites/--pose-readout-terms/--pose-readout-init, not both. "
                "--pose-readout --pose-readout-prenorm --pose-readout-post-zero is "
                "--pose-readout-sites pre,post --pose-readout-terms onehot,dist.")
    if new_rd:
        if not (d.get("pose_readout_sites") or d.get("pose_readout_terms")):
            _refuse("--pose-readout-init selects how the readout site is initialised, not "
                    "whether it exists. Pass --pose-readout-sites and --pose-readout-terms too, "
                    "or drop it.")
        if not d.get("pose_readout_sites"):
            _refuse("--pose-readout-terms names what the readout sites carry but not WHERE they "
                    "are. Pass --pose-readout-sites pre, post or pre,post too.")
        if not d.get("pose_readout_terms"):
            _refuse("--pose-readout-sites names where the readout injects but not WHAT. Pass "
                    "--pose-readout-terms onehot[,dist][,orient] too -- the legacy "
                    "--pose-readout meant onehot,dist, plus orient with --frame-direction.")
        rd = {"sites": parse_readout_sites(d["pose_readout_sites"]),
              "terms": parse_terms(d["pose_readout_terms"], "--pose-readout-terms"),
              "init": str(d.get("pose_readout_init") or "zero")}
        rd["sites"], rd["terms"] = check_readout_axes(rd["sites"], rd["terms"], rd["init"],
                                                      have_dir)
    else:
        if old_rd and not d.get("pose_readout"):
            if d.get("pose_readout_copy_distance"):
                _refuse("--pose-readout-copy-distance without --pose-readout. It selects how the "
                        "readout site is initialised, not whether it exists.")
            _refuse("--pose-readout-prenorm/--pose-readout-post-zero move or add the readout "
                    "site; they do not install it. Pass --pose-readout too, or drop them.")
        rd = legacy_readout_axes(d)
        if rd is not None:
            rd["sites"], rd["terms"] = check_readout_axes(rd["sites"], rd["terms"], rd["init"],
                                                          have_dir)
    if rd is not None and blk is None and not d.get("pose_inblock"):
        _refuse("the readout site needs a stack injection to be installed: it READS the "
                "per-forward pose geometry that wrapper publishes rather than recomputing cdist "
                "and the one-hot, so on its own the hook would never fire and the flag would be "
                "a silent no-op. Pass --pose-block-terms (or the legacy --pose-perblock), or "
                "--pose-inblock.")
    return blk, rd


def record(blk, rd):
    """The canonical args.json fields for a resolved pair -- axis keys AND legacy keys.

    BOTH, DELIBERATELY. The axis keys are the complete record: the legacy keys cannot express
    "onehot+dist at zero init", so on their own two different architectures would be recorded
    identically. The legacy keys are kept because the resume guard, run_index.py and six
    run_logging helpers read them, and because every arm already on disk is described by them --
    dropping them would make this run's record incomparable with its own history. Where legacy
    cannot express the config its keys are the WEAKER true statement (pose_perblock is really
    true; pose_perblock_copy is really false), never a guess, and the argv builders below read
    the axis keys, so nothing downstream reconstructs an architecture from the weaker half.
    """
    out = dict(EMPTY)
    if blk is not None:
        out["pose_block_terms"] = ",".join(blk["terms"])
        out["pose_block_init"] = blk["init"]
        out["pose_block_weights"] = blk["weights"]
    if rd is not None:
        out["pose_readout_sites"] = ",".join(rd["sites"])
        out["pose_readout_terms"] = ",".join(rd["terms"])
        out["pose_readout_init"] = rd["init"]
    out["pose_perblock"] = blk is not None
    out["pose_perblock_tied"] = bool(blk and blk["weights"] == "tied")
    out["pose_perblock_share"] = bool(blk and blk["weights"] == "shared")
    # THE LEGACY FLAG MEANT ONE EXACT POINT, so it is recorded true only at that point: untied
    # copies of BOTH distance terms. A pretrained one-hot-only arm, which legacy could not ask
    # for, records false -- weaker, and true.
    out["pose_perblock_copy"] = bool(
        blk and blk["weights"] == "untied" and blk["init"] == "pretrained"
        and all(t in blk["terms"] for t in DIST_TERMS))
    out["pose_readout"] = rd is not None
    out["pose_readout_prenorm"] = bool(rd and "pre" in rd["sites"])
    out["pose_readout_post_zero"] = bool(rd and set(rd["sites"]) == set(READOUT_SITES))
    out["pose_readout_copy_distance"] = bool(rd and rd["init"] == "pretrained")
    return out


def normalise(a):
    """Resolve the axes ONCE and write the canonical record back onto the namespace.

    Called immediately after parse_args in BOTH the trainer and the evaluator, before anything
    reads a pose flag. Everything downstream -- the installs, args.json, the resume guard, every
    forwarded child argv -- then sees one consistent picture, and the legacy flags are a
    spelling rather than a code path.
    """
    blk, rd = resolve(vars(a))
    for k, v in record(blk, rd).items():
        if hasattr(a, k):
            setattr(a, k, v)
    return blk, rd


def backfill(d):
    """The axis keys an OLD record implies, for a dict that has only the legacy ones.

    A resume bundle or args.json written before the axes existed records pose_perblock_copy and
    friends. Absence of the axis keys there is evidence of the legacy point in the space, not of
    an unknown setting, so the resume guard reads them through this rather than against a
    constant -- a constant would report drift on every legacy resume. Returns {} when the record
    already has the axis keys.
    """
    if any(k in d for k in AXIS_KEYS):
        return {}
    if not (_legacy_block_given(d) or _legacy_readout_given(d)):
        return {}
    try:
        blk, rd = resolve({k: d.get(k) for k in LEGACY_KEYS + ("frame_direction", "pose_inblock")})
    except PoseConfigError:
        return {}
    return {k: v for k, v in record(blk, rd).items() if k in AXIS_KEYS}


#: The axis flags each process understands. The evaluator has no init flags, for the same reason
#: it never had --pose-readout-copy-distance: every head tensor comes from the checkpoint, so an
#: initialisation is overwritten before anything is scored, and the tensor NAMES are a function
#: of the terms alone. The trainer's own --loss-only child DOES need them, because a resume
#: bundle carries only the TRAINABLE tensors and a tensor left out of --train-only keeps its init.
_ARGV_KEYS = {
    "train": ("pose_block_terms", "pose_block_init", "pose_block_weights",
              "pose_readout_sites", "pose_readout_terms", "pose_readout_init"),
    "eval": ("pose_block_terms", "pose_block_weights", "pose_readout_sites",
             "pose_readout_terms"),
}


def eval_long_opts(src=None):
    """The long options phase_k_eval.py's own argparse declares, read out of the source.

    The flags are spelled literally in that file (`ap.add_argument("--pose-block-terms", ...)`)
    rather than registered by a shared helper, which is deliberate: run_logging/pathway_flags.py
    finds the evaluator's vocabulary with this same regex, so a flag added through a helper would
    be invisible to it and silently dropped from a derived eval argv. Returns None when the source
    cannot be read, which means "cannot check" rather than "has no flags".
    """
    import os
    import re
    src = src or os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_k_eval.py")
    try:
        with open(src) as fh:
            return set(re.findall(r'ap\.add_argument\("(--[a-z0-9-]+)"', fh.read()))
    except OSError:
        return None


def argv(src, target="eval", defined=None):
    """(flags, notes): the pose argv a child process needs to build THIS architecture.

    `src` is an argparse namespace this process normalised, or an args.json dict. ONE builder for
    every forwarding path, because the duplicated per-flag `if` blocks are what let a child eval
    build a different architecture than its parent (commit 7d9d4d10) -- four copies of the same
    ladder, each of which had to be edited whenever a flag was added, and one of which was not.

    `defined` is the set of long options the target's parser really declares, when the caller
    knows it; a flag this builder wants and the target does not have is a WIRING BUG and is
    reported in `notes` rather than dropped in silence.
    """
    d = dict(src) if isinstance(src, dict) else dict(vars(src))
    d = dict(d, **backfill(d))
    out, notes = [], []
    if target not in _ARGV_KEYS:
        raise ValueError(f"argv(target={target!r}): expected one of {sorted(_ARGV_KEYS)}")
    for k in _ARGV_KEYS[target]:
        v = d.get(k, EMPTY[k])
        if not v or v == EMPTY[k]:
            # EMPTY IS THE DEFAULT, not a value to pass. "" means the sites are off; "zero" and
            # "untied" are the parser's own defaults, so passing them adds noise to every argv.
            continue
        flag = "--" + k.replace("_", "-")
        if defined is not None and flag not in defined:
            notes.append(f"{k}={v!r} but the target parser has no {flag} -- a wiring bug: the "
                         f"child would build a DIFFERENT architecture than this process did")
            continue
        out += [flag, str(v)]
    return out, notes
