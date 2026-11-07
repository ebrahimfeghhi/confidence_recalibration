#!/usr/bin/env python
"""Patch a resume bundle so a run can CONTINUE with --frame-direction added.

WHY A PATCH AND NOT A FLAG. phase_k_train's resume refuses a bundle that lacks any trainable
tensor -- "STOP: the bundle is missing N trainable tensor(s)" -- and --resume-allow-drift does not
cover that, because it governs ARGUMENT drift, not the parameter set. Adding --frame-direction
creates four new tensors (one 3->c_z projection before the pairformer stack, three more inside it
when --pose-perblock is on), so a plain resume stops. This adds them to the bundle instead.

ZEROS ARE THE CORRECT VALUE, not a convenience. pose_direction zero-initialises its projection and
install() verifies the output is exactly zero, so a head with these tensors at zero computes
bit-identically to the head without them. The loss curve therefore continues with no step, and any
later improvement is attributable to the new path rather than to a restart.

THE ADAMW STATE IS RESET, DELIBERATELY. Its state_dict is keyed by PARAMETER INDEX within the
group, and pose_direction installs BEFORE pose_perblock, so the new tensors land in the middle of
named_parameters() and shift every index after them. Restoring the old moments by index would
silently attach them to the wrong tensors -- a corruption with no error message. Resetting costs
~20 steps of rebuild at betas=(0.9, 0.95), against 271 steps per epoch.

Usage:
  python -I run_logging/add_orientation_to_bundle.py <in.pt> <out.pt> [--c-z 256] [--blocks 3]
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--c-z", type=int, default=256)
ap.add_argument("--blocks", type=int, default=3, help="per-block orientation copies")
ap.add_argument("--prefix", default="confidence_head")
a = ap.parse_args()

b = torch.load(a.src, map_location="cpu")
for k in ("head", "opt", "epoch", "step"):
    if k not in b:
        raise SystemExit(f"STOP: {a.src} has no '{k}'; keys are {list(b)}")
head = dict(b["head"])
n0 = len(head)

# THE NAMES MUST MATCH WHAT THE MODULES REGISTER, or the resume's set check fails again with the
# same error for a different reason. pose_direction registers frame_dir_proj on the head;
# pose_perblock registers a ModuleList named pose_perblock_frame_dir, indexed 0..blocks-1 -- the
# names seen in this project's own gradient logs (confidence_head.pose_perblock_frame_dir.0.weight).
want = [f"{a.prefix}.frame_dir_proj.weight"]
want += [f"{a.prefix}.pose_perblock_frame_dir.{i}.weight" for i in range(a.blocks)]

ref = head.get(f"{a.prefix}.linear_no_bias_d_wo_onehot.weight")
if ref is None:
    raise SystemExit(f"STOP: no {a.prefix}.linear_no_bias_d_wo_onehot.weight in the bundle to "
                     f"copy dtype/device from; is this a confidence-head bundle?")
added = []
for n in want:
    if n in head:
        print(f"  {n} already present, left alone")
        continue
    head[n] = torch.zeros((a.c_z, 3), dtype=ref.dtype)
    added.append(n)

if not added:
    raise SystemExit("STOP: nothing to add -- the bundle already carries every orientation tensor.")

b["head"] = head
# EMPTY STATE, MATCHING GROUPS. AdamW accepts a state_dict whose 'state' is empty and initialises
# moments lazily on the first step; what it cannot accept is a param_groups length that disagrees
# with the live group. So the groups are rebuilt to the new count with the old hyperparameters.
old = b["opt"]
groups = []
for g in old.get("param_groups", []):
    g2 = {k: v for k, v in g.items() if k != "params"}
    groups.append(g2)
if len(groups) != 1:
    raise SystemExit(f"STOP: expected one param group, found {len(groups)}; the index remap this "
                     f"avoids would not be safe to guess at.")
groups[0]["params"] = list(range(len(head)))
b["opt"] = {"state": {}, "param_groups": groups}

g = b.get("guard") or {}
g["orientation_patched_from"] = os.path.basename(a.src)
g["orientation_added"] = added
b["guard"] = g

tmp = a.dst + ".tmp"
torch.save(b, tmp)
os.replace(tmp, a.dst)
print(f"\n  {a.src}")
print(f"    head tensors {n0} -> {len(head)}  (+{len(added)})")
for n in added:
    print(f"      + {n}  {tuple(head[n].shape)}  all-zero: {bool((head[n] == 0).all())}")
print(f"    AdamW state reset; param_groups now list {len(head)} param(s)")
print(f"    epoch {b['epoch']}, step {b['step']} preserved")
print(f"  -> {a.dst}")
