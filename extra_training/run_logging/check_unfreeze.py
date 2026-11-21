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
import argparse, sys, torch

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
sel = {k for k in f if any(p in k for p in pats)}

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
