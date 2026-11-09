"""The single definition of "code this copy does not carry", used by the cleaner AND the test.

WHY ONE DEFINITION. Deleting dead branches from the copy would make a verbatim comparison
against `phase_k_train.py` fail for a good reason, and a test that fails for good reasons gets
switched off. So the rule lives here: `strip(source)` removes the branches that depend on flags
`cli.py` cannot express, and `test_region_equivalence.py` applies it to the LIVE source before
comparing. Deliberate leanness then matches; real drift still fails.

A BRANCH IS REMOVED ONLY IF ITS CONDITION CANNOT BE TRUE. The test must reference a culled flag
and nothing else; `if a.l2_sp and a.resume:` is left alone, because reading it requires knowing
which operand decides. When such an `if` has an `else`, the else body is kept and promoted --
it is the path this copy always takes.
"""
import ast
import re

#: flags dropped from trainer/cli.py. Attribute reads of these raise AttributeError, so the
#: branches guarding them are not merely unreachable, they are broken.
CULLED = frozenset({
    "lora_rank", "lora_alpha", "lora_dropout", "lora_targets",
    "l2_sp", "l2_sp_exclude",
    "crop_margin", "crop_tokens", "crop_contact", "crop_all_poses",
    "token_keep_union", "token_keep_other",
    "pae_row_frac", "pae_row_mass", "pae_row_floor", "pae_row_log_every",
    "rank_tau", "rank_pairs", "rank_within_weight", "rank_topk_decay", "rank_straddle",
    "rank_weight_by_gap", "rank_labels", "rank_spearman_tau", "rank_dist_weight",
    "lr_final", "lr_decay_start_epoch", "lr_decay_end_epoch", "lr_decay_shape",
    "early_stop_patience", "early_stop_min_delta",
    "epoch_order", "structures_per_step", "drop_pde", "pair_dropout",
    "eval_device", "eval_min_free_mib", "eval_pool_crop_tokens",
    "allow_unmasked_fallback",
})

#: NOT culled, pending an owner decision. --grad-probe is the only way to re-measure alpha-rank
#: on this frozen architecture, and --loss-only IS the per-epoch held-out monitor.
OWNER_DECISION = frozenset({"grad_probe", "loss_only"})


def _attrs(node):
    """Every `a.<name>` read under `node`, including `rt.a.<name>`.

    BOTH SPELLINGS MATTER. The extraction rewrote names the closures read from main()'s scope to
    `rt.<name>`, so the args object is `a` in the modules that take it as a parameter and `rt.a`
    in the ones that reach it through the context. Matching only the first found 53 removals and
    silently missed every one in loop.py, unit.py, evaluate.py and helpers.py.
    """
    out = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Attribute):
            continue
        base = n.value
        if isinstance(base, ast.Name) and base.id == "a":
            out.add(n.attr)
        elif (isinstance(base, ast.Attribute) and base.attr == "a"
              and isinstance(base.value, ast.Name) and base.value.id == "rt"):
            out.add(n.attr)
    return out


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def dead_test(test):
    """True when this condition reads culled flags and nothing that could decide it otherwise."""
    at = _attrs(test)
    if not at or not (at <= CULLED):
        return False
    # any other free name could carry the decision; refuse to guess
    other = _names(test) - {"a", "rt"}
    return not other


def _only_culled(node):
    """The expression reads culled flags and nothing else that could carry meaning."""
    at = _attrs(node)
    return bool(at) and at <= CULLED


def _span(node):
    return node.lineno, node.end_lineno


def dead_spans(tree, lines):
    """[(first, last)] 1-based inclusive line ranges to delete, plus a list of unhandled sites.

    LINE RANGES, NOT A TRANSFORMED TREE. `ast.unparse` would round-trip the code and drop every
    comment -- and the comments here are the record of what each guard cost to learn. So the AST
    is used only to LOCATE, and the deletion happens on the source text.

    A leading run of comment lines immediately above a deleted block goes with it: a comment
    explaining a branch that no longer exists is worse than no comment.
    """
    spans, unhandled = [], []

    def take(node, why):
        lo, hi = _span(node)
        while lo > 1:
            prev = lines[lo - 2].strip()
            if prev.startswith("#"):
                lo -= 1
            else:
                break
        spans.append((lo, hi, why))

    for n in ast.walk(tree):
        if isinstance(n, ast.If) and (dead_test(n.test) or dead_const(n.test)):
            if n.orelse:
                unhandled.append((n.lineno, "if/else -- else body would need promoting"))
            else:
                take(n, "dead branch")
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and _only_culled(n.value):
            take(n, "report of a culled setting")
        elif isinstance(n, ast.Dict):
            # ONLY WHEN THE ENTRY OWNS ITS LINES. These guard dicts pack several keys per line:
            # `"drop_pde": a.drop_pde, "clip_norm": a.clip_norm,`. Deleting by the value's line
            # span there takes a live key with it, which is how resume.py lost "clip_norm".
            on_line = {}
            for k in n.keys:
                if k is not None:
                    on_line[k.lineno] = on_line.get(k.lineno, 0) + 1
            for k, v in zip(n.keys, n.values):
                if v is None or not _only_culled(v):
                    continue
                lo = min(k.lineno if k is not None else v.lineno, v.lineno)
                if on_line.get(lo, 0) != 1 or v.end_lineno != lo:
                    unhandled.append((lo, "dict entry shares its line"))
                    continue
                rest = lines[lo - 1].rstrip()
                if not rest.endswith((",", "{", "}")):
                    unhandled.append((lo, "dict entry line has a tail"))
                    continue
                spans.append((lo, v.end_lineno, "guard field"))

    # merge overlaps, outermost wins
    spans.sort()
    merged = []
    for lo, hi, why in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi), merged[-1][2])
        else:
            merged.append((lo, hi, why))
    return merged, unhandled


def strip_source(text):
    """(new_text, n_lines_removed, unhandled)."""
    lines = text.splitlines(keepends=True)
    spans, unhandled = dead_spans(ast.parse(text), lines)
    drop = set()
    for lo, hi, _ in spans:
        drop.update(range(lo, hi + 1))
    out = [l for i, l in enumerate(lines, 1) if i not in drop]
    text = "".join(out)

    # AN EMPTIED BLOCK NEEDS A BODY. Removing the only statement under a live `if` leaves
    # "expected an indented block", which is how dataset.py and trainable.py were skipped
    # entirely rather than leaned. Put a `pass` back, with a comment saying why it is there.
    for _ in range(40):
        try:
            ast.parse(text)
            break
        except SyntaxError as e:
            if not e.msg.startswith("expected an indented block"):
                break
            # THE LINE TO INDENT UNDER IS IN THE MESSAGE, not in e.lineno. e.lineno points at
            # the first line that is wrongly dedented; the header is "... after 'if' statement
            # on line N". Using e.lineno put the `pass` below the next statement instead of
            # under the empty block, and the loop never converged.
            m = re.search(r"on line (\d+)", e.msg)
            if not m:
                break
            at = int(m.group(1))
            ls = text.splitlines(keepends=True)
            head = ls[at - 1] if at <= len(ls) else ""
            indent = len(head) - len(head.lstrip())
            ls.insert(at, " " * (indent + 4)
                      + "pass  # body removed with a flag this copy does not carry\n")
            text = "".join(ls)

    return text, len(drop), unhandled


def culled_in(tree):
    """{flag: n} for culled flags still read -- these need a human."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in CULLED:
            base = n.value
            ok = ((isinstance(base, ast.Name) and base.id == "a")
                  or (isinstance(base, ast.Attribute) and base.attr == "a"
                      and isinstance(base.value, ast.Name) and base.value.id == "rt"))
            if ok:
                out[n.attr] = out.get(n.attr, 0) + 1
    return out


#: what each culled flag held on the live arm 87ja6a. Substituting these is what lets the
#: remaining references go: `weight_by_gap=rt.a.rank_weight_by_gap` becomes `weight_by_gap=True`,
#: exactly what the live run passes, and `if rt.a.crop_tokens` becomes `if ""` -- now provably
#: dead, so a second strip pass removes the branch the first could not.
LIVE_VALUES = {
    "allow_unmasked_fallback": False, "crop_all_poses": False, "crop_contact": 5.0,
    "crop_margin": 0.0, "crop_tokens": "", "drop_pde": False,
    "early_stop_min_delta": 0.002, "early_stop_patience": 0, "epoch_order": "shuffle",
    "eval_device": "", "eval_min_free_mib": 20000, "eval_pool_crop_tokens": "",
    "l2_sp": 0.0, "l2_sp_exclude": "", "lora_alpha": 0.0, "lora_dropout": 0.0,
    "lora_rank": 0, "lora_targets": "", "lr_decay_end_epoch": 0.0,
    "lr_decay_shape": "cosine", "lr_decay_start_epoch": 0.0, "lr_final": 0.0,
    "pae_row_floor": 10, "pae_row_frac": 0.0, "pae_row_log_every": 50,
    "pae_row_mass": 0.8, "pair_dropout": -1.0, "rank_dist_weight": 0.1,
    "rank_labels": None, "rank_pairs": "cross", "rank_spearman_tau": 0.0,
    "rank_straddle": 0.0, "rank_tau": None, "rank_topk_decay": 0.0,
    "rank_weight_by_gap": True, "rank_within_weight": 1.0, "structures_per_step": 0,
    "token_keep_other": -1.0, "token_keep_union": -1.0,
}

_NO = object()


def substitute(text):
    """Replace every READ of a culled flag with the constant the live arm passed.

    Reads only: `a.rank_weight_by_gap = ...` is a Store, and substituting there produces
    `True = ...`, which is not valid Python. Resume inherits several of these from a bundle's
    guard, so the Store form is common rather than a corner case.

    Rewritten on the source text by AST position, right to left, so comments survive and earlier
    offsets stay valid.
    """
    tree = ast.parse(text)
    hits = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and n.attr in LIVE_VALUES):
            continue
        if not isinstance(n.ctx, ast.Load):
            continue
        base = n.value
        ok = ((isinstance(base, ast.Name) and base.id == "a")
              or (isinstance(base, ast.Attribute) and base.attr == "a"
                  and isinstance(base.value, ast.Name) and base.value.id == "rt"))
        if ok:
            hits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, n.attr))
    lines = text.splitlines(keepends=True)
    for ln, c0, _ln2, c1, name in sorted(set(hits), reverse=True):
        lines[ln - 1] = lines[ln - 1][:c0] + repr(LIVE_VALUES[name]) + lines[ln - 1][c1:]
    return "".join(lines), len(hits)


def _const(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return _NO


def dead_const(test):
    """True when the condition has folded to something always falsy."""
    v = _const(test)
    if v is not _NO:
        return not v
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        return all(dead_const(x) for x in test.values)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        l, r = _const(test.left), _const(test.comparators[0])
        if l is _NO or r is _NO:
            return False
        op = test.ops[0]
        try:
            for cls, fn in ((ast.Gt, lambda: l > r), (ast.GtE, lambda: l >= r),
                            (ast.Lt, lambda: l < r), (ast.LtE, lambda: l <= r),
                            (ast.Eq, lambda: l == r), (ast.NotEq, lambda: l != r)):
                if isinstance(op, cls):
                    return not fn()
        except TypeError:
            return False
    return False
