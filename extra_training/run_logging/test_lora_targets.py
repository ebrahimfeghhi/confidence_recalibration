#!/usr/bin/env python
"""GATE: where the LoRA adapters go, and what stays dense, on the REAL head tensor names.

WHAT THE FIRST VERSION GOT WRONG, recorded because the failure was invisible. It asserted over
module names it had invented -- `tri_att_start.linear_no_bias_q` and friends -- when the real
TriangleAttention leaves are `<op>.linear` and `<op>.mha.linear_{q,k,v,g,o}`. 35 of its 50 "dead"
names could never be adapted because they do not exist, so "PASS none of the dead modules match"
was 70% assertions about strings. Its control re-tested a name already in its own DEAD list. And
it never ran: it was not wired into the launcher's gate sequence. This version reads every name
from a real checkpoint.

WHAT IT CHECKS, all against the patterns read from the launcher itself:
  1. no adapter on a module inside a measured-DEAD pairformer op;
  2. no adapter on a readout that is dead under --pae-only (lean_readouts.DEAD_UNDER_PAE_ONLY);
  3. no adapter on ANY pose tensor -- they are meant to be dense, not dense+adapter;
  4. every one of the 12 pose tensors matches --lora-keep-trainable, INCLUDING both depth-0 terms;
  5. the live pairformer ops ARE adapted, so the pattern has not been truncated to nothing;
  6. rank 8 fits every module that is adapted, so nothing is silently skipped for rank-too-high.
"""
import os
import re
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
from lora import _targets_ok                                        # noqa: E402

REF = os.path.join(ROOT, "training/runs/q6mgyc/head_ep50_step7250.pt")
RANK = int(os.environ.get("LORA_RANK", "8"))
LAUNCHER = os.path.join(ROOT, "shell_scripts", "launch_pae_crop_arm.sh")


def _from_launcher(var):
    """Read a variable by EXECUTING the launcher's own assignments, never a retyped copy."""
    out = subprocess.run(
        ["bash", "-c",
         f'eval "$(grep -m1 \'^TRAIN_ONLY=\' {LAUNCHER})"; '
         f'eval "$(grep -m1 \'^LORA_EXCLUDE=\' {LAUNCHER})"; '
         f'eval "$(grep -m1 \'^LORA_TARGETS=\' {LAUNCHER})"; '
         f'eval "$(grep -m1 \'^LORA_KEEP=\' {LAUNCHER})"; printf %s "${{{var}}}"'],
        capture_output=True, text=True).stdout
    return [p.strip() for p in out.split(",") if p.strip()]


TARGETS, KEEP = _from_launcher("LORA_TARGETS"), _from_launcher("LORA_KEEP")
if not TARGETS or not KEEP:
    sys.exit(f"STOP: could not read patterns from {os.path.basename(LAUNCHER)}")

sd = torch.load(REF, map_location="cpu", weights_only=False)
sd = sd if not isinstance(sd.get("head"), dict) else sd["head"]
REL = {n.split("confidence_head.", 1)[-1]: t for n, t in sd.items()}
# ADAPTABLE = a module carrying a 2-D `weight`, which is what iter_adaptable yields.
ADAPTABLE = {n[:-len(".weight")]: tuple(t.shape)
             for n, t in REL.items() if n.endswith(".weight") and t.dim() == 2}

DEAD_OPS = ["blocks.0.tri_mul_out", "blocks.0.tri_mul_in", "blocks.0.tri_att_start",
            "blocks.0.tri_att_end", "blocks.1.tri_att_start", "blocks.1.tri_att_end",
            "blocks.1.pair_transition", "blocks.2.tri_att_start", "blocks.2.tri_att_end",
            "blocks.3.tri_att_end"]
DEAD_READOUT = ("plddt_weight", "resolved_weight", "plddt_ln.", "resolved_ln.",
                "linear_no_bias_pde.", "pde_ln.", "input_strunk_ln.")
POSE = sorted(n for n in REL
              if "frame_dir" in n or "pose_perblock" in n or n.startswith("linear_no_bias_d"))

adapted = sorted(n for n, shp in ADAPTABLE.items() if _targets_ok(n, TARGETS))
print(f"patterns read from {os.path.basename(LAUNCHER)}: "
      f"{len(TARGETS)} target(s), {len(KEEP)} keep")
print(f"{len(ADAPTABLE)} adaptable module(s) in the reference head; {len(adapted)} would be adapted")

fail = []


def check(name, bad, note=""):
    if bad:
        print(f"FAIL  {name}: {len(bad)} -> {sorted(bad)[:4]}")
        fail.append(name)
    else:
        print(f"PASS  {name}{note}")


check("no adapter inside a measured-dead pairformer op",
      [n for n in adapted if any(f"pairformer_stack.{op}." in n + "." for op in DEAD_OPS)],
      f"  ({sum(1 for n in ADAPTABLE if any(f'pairformer_stack.{op}.' in n + '.' for op in DEAD_OPS))} such modules exist)")
check("no adapter on a readout dead under --pae-only",
      [n for n in adapted if any(d.rstrip('.') in n for d in DEAD_READOUT)])
check("no adapter on any pose tensor",
      [n for n in adapted if n + ".weight" in set(POSE)],
      f"  (pose is dense, not dense+adapter)")
check("every pose tensor matches --lora-keep-trainable",
      [n for n in POSE if not any(re.search(k, n) for k in KEEP)],
      f"  (all {len(POSE)})")
check("the live pairformer ops ARE adapted",
      [] if any("blocks.3.tri_mul_out." in n for n in adapted)
         and any("blocks.1.tri_mul_in." in n for n in adapted) else ["live ops not adapted"])
check(f"rank {RANK} fits every adapted module",
      [f"{n}{ADAPTABLE[n]}" for n in adapted if RANK > min(ADAPTABLE[n])])

print(f"\n  adapted: {len(adapted)} module(s); pose kept dense: {len(POSE)} tensor(s)")
sys.exit(1 if fail else 0)
