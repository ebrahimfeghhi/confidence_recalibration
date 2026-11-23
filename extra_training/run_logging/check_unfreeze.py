#!/usr/bin/env python
"""Did exactly the intended tensors receive a gradient? Answered by MOVEMENT, not by inspection.

--grad-check-every already asserts that nothing OUTSIDE the confidence head carries a gradient.
It does not assert that everything inside --train-only does, so an unfreeze that selected a
tensor nothing reads looks identical to one that worked: both print the same trainable count.

A tensor that moved between --save-init and a later checkpoint received a gradient. One that did
not move did not. That is the empirical version of the check and it needs no instrumentation.

TWO FAILURES ARE REPORTED SEPARATELY because they mean different things: a SELECTED tensor that
did not move is an unfreeze that did nothing (the pattern missed, or nothing reads it), while an
UNSELECTED tensor that moved is a leak -- the frozen set is not frozen.
"""
import argparse, re, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--after", required=True)
ap.add_argument("--train-only", required=True,
                help="the comma-separated --train-only patterns the arm ran with")
a = ap.parse_args()

i = torch.load(a.init, map_location="cpu", weights_only=False)
f = torch.load(a.after, map_location="cpu", weights_only=False)
i = i.get("head", i) if isinstance(i, dict) and "head" in i else i
pats = [p.strip() for p in a.train_only.split(",") if p.strip()]

# MATCHED THE WAY THE TRAINER MATCHES, which is re.search against the name with the head's own
# prefix STRIPPED (phase_k_train: `_rel = {n: n[len(_pfx) + 1:]}` then
# `any(re.search(p, _rel[n]) for p in _pats)`).
#
# IT USED TO BE `any(p in k for p in pats)` -- a plain substring test against the FULL name. That
# silently broke for every arm whose pattern is a regex rather than a list of substrings. Arm I's
# first term is the negative lookahead `^(?!pairformer_stack\.)`, which occurs in no tensor name,
# so NOTHING was selected, all 106 moved tensors were reported as leaks, and the verdict was
# MISMATCH on a correctly-behaving arm. The failure is worse than cosmetic: with an empty selected
# set the check cannot distinguish a real leak from a working arm either, so the one thing it
# exists to catch was unreachable.
#
# THE PREFIX IS STRIPPED PER KEY, not assumed: a bundle's keys carry `confidence_head.` while the
# trainer's patterns are written relative to the head, so `^` anchors would never match otherwise.
_PFX = "confidence_head."


def _rel(k):
    return k[len(_PFX):] if k.startswith(_PFX) else k


def _hit(k):
    r = _rel(k)
    for p in pats:
        try:
            if re.search(p, r):
                return True
        except re.error:
            # A PATTERN THAT IS NOT VALID REGEX falls back to the substring test rather than
            # killing the check, because some arms really do pass bare names.
            if p in r:
                return True
    return False


sel = {k for k in f if _hit(k)}
if not sel:
    print(f"  WARNING: --train-only matched NO tensor in the bundle. Patterns: {pats[:3]}"
          f"{' ...' if len(pats) > 3 else ''}\n"
          f"  Every moved tensor below will read as a leak, which is a fault in this check and not "
          f"in the arm. Example bundle key: {next(iter(f), '(empty)')}")

moved, still, leaked = [], [], []
for k in f:
    if k not in i:
        continue
    d = float((f[k].float() - i[k].float()).abs().max())
    if k in sel:
        (moved if d > 0 else still).append((k, d))
    elif d > 0:
        leaked.append((k, d))

print(f"  selected by --train-only : {len(sel)}")
print(f"    moved (got a gradient) : {len(moved)}")
print(f"    did NOT move           : {len(still)}")
for k, _ in still[:12]:
    print(f"      {k}")
print(f"  NOT selected but moved   : {len(leaked)}")
for k, d in leaked[:12]:
    print(f"      {k}  max|delta| {d:.3e}")
ok = not still and not leaked
print("  VERDICT:", "exactly the selected set trained" if ok else "MISMATCH, see above")
sys.exit(0 if ok else 1)
