#!/usr/bin/env python
"""resume_ep<N>.pt -> head_ep<N>.pt: the flat head state dict the evaluators want.

WHY THIS IS NEEDED. phase_k_eval and train_eval_metrics take --ckpt and immediately torch.load it
into a dict they compare key-for-key against the model's confidence_head parameters. A RESUME
BUNDLE is not that shape: it is {head, opt, epoch, step, guard}, so passing one directly fails the
set-equality guard rather than loading a model. fawgtu wrote only bundles -- it ran with no eval
pass, so nothing ever asked it for a head-only file -- and this converts them.

NOTHING IS RECOMPUTED. The bundle's "head" IS the trained state dict; this only unwraps it. The
output is written next to the bundle so the two stay together, and an existing file is left alone
unless --force, because the point of these is to be the fixed object a later evaluation refers to.

CPU ONLY, NO GPU. Safe to run while a training job holds every card.

Usage:  python -I run_logging/extract_heads.py <run_id> [--epochs 5,10,15] [--force]
"""
import argparse
import glob
import os
import re

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--epochs", default="", help="comma list; empty = every bundle present")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

d = os.path.join(ROOT, "training", "runs", a.run)
want = {int(x) for x in a.epochs.split(",") if x.strip()} if a.epochs.strip() else None
found = []
for f in sorted(glob.glob(os.path.join(d, "resume_ep*.pt"))):
    m = re.search(r"resume_ep(\d+)\.pt$", f)
    if not m:
        continue
    ep = int(m.group(1))
    if want is not None and ep not in want:
        continue
    found.append((ep, f))
if not found:
    raise SystemExit(f"STOP: no matching resume bundle in {d}")

for ep, f in found:
    b = torch.load(f, map_location="cpu")
    # NAMED head_ep<N>_step<S>.pt, THE PROJECT'S CONVENTION, because train_eval_metrics.py
    # auto-discovers a run's latest checkpoint by globbing exactly that pattern and picking the
    # highest (epoch, step). A bare head_ep<N>.pt is invisible to it, which would silently fall
    # back to a different checkpoint than the one asked for.
    step = b.get("step")
    out = os.path.join(d, f"head_ep{ep}_step{step}.pt" if step is not None
                       else f"head_ep{ep}.pt")
    if os.path.exists(out) and not a.force:
        print(f"  epoch {ep:>3}: {os.path.basename(out)} exists, skipping")
        continue
    if "head" not in b:
        raise SystemExit(f"STOP: {os.path.basename(f)} has no 'head' key; keys are {list(b)}. "
                         f"That is not a resume bundle this script understands, and guessing "
                         f"which entry holds the weights is how the wrong tensors get evaluated.")
    head = b["head"]
    # ATOMIC, so an interrupted write cannot leave a half-file that later looks like a checkpoint.
    tmp = out + ".tmp"
    torch.save(head, tmp)
    os.replace(tmp, out)
    print(f"  epoch {ep:>3}: {len(head)} tensor(s) -> {os.path.basename(out)}"
          + (f"  (bundle says epoch {b.get('epoch')}, step {b.get('step')})"
             if b.get("epoch") is not None else ""))
print(f"\n  {len(found)} bundle(s) processed in {os.path.join('training', 'runs', a.run)}")
