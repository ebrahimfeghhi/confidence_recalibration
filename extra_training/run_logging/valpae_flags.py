#!/usr/bin/env python
"""Derive the val-PAE measurement flags for one run, from its own args.json.

WHY THIS EXISTS AT ALL: hand-building the argv dropped the pose flags once and measured a head
without the pathway it was trained with. The architecture must come from the run, never from
memory.

BALANCE AND ROW WEIGHTING ARE DELIBERATELY NOT COPIED. They are TRAINING choices -- they reweight
which poses the gradient listens to. The measurement has to be the same quantity for every arm or
the arms cannot be compared, and that quantity is the PLAIN per-pose mean. Copying
--pae-balance-accept from a balanced arm would return 0.5/n_good and 0.5/n_bad weighted loss and
put it in a table next to shipped's unweighted 3.0991, which is two different numbers under one
heading. Arms B, E and G all had balance 0.0, so every measurement so far is already unweighted
and stays comparable.

--pae-split-accept IS copied: it only decides which poses are called good or bad in the REPORT,
and it must match the training split or the groups mean different things.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import pose_axes as _PAX   # noqa: E402  the pose axes, and the ONE argv builder

# THE POSE INJECTION FLAGS ARE NO LONGER LISTED HERE. They came off this table one key at a time
# -- pose_perblock_share is missing from it, which is the silent-wrong-architecture case the
# module docstring warns about -- so they now come from pose_axes.argv, the single builder every
# forwarding path uses. What stays is the rest: the loss shape and the lean trio.
ARCH_FLAGS = [
    ("pae_only", "--pae-only"), ("drop_pde", "--drop-pde"),
    ("frame_direction", "--frame-direction"),
    ("lean_readouts", "--lean-readouts"), ("lean_single", "--lean-single"),
    ("pose_mlp", "--pose-mlp"),
]
VALUE_FLAGS = [
    ("train_only", "--train-only"), ("alpha_pae", "--alpha-pae"), ("nsample", "--nsample"),
    ("blocks_per_ckpt", "--blocks-per-ckpt"), ("pae_split_accept", "--pae-split-accept"),
]
# NEVER COPIED: pae_balance_accept, pae_row_frac -- see the module docstring.
TRAINING_ONLY = ("pae_balance_accept", "pae_row_frac")


def main():
    rid = sys.argv[1]
    a = json.load(open(f"training/runs/{rid}/args.json"))
    a = a.get("args", a)
    flags = [f for k, f in ARCH_FLAGS if a.get(k)]
    # AND THE POSE INJECTION AXES, from the one builder. The `train` target because this file's
    # flags go to phase_k_train --loss-only: that process builds the head from scratch and loads
    # only the TRAINABLE tensors out of the bundle, so a pose tensor outside --train-only keeps
    # its initialisation and the init flags are part of the architecture here.
    flags += _PAX.argv(a, target="train")[0]
    for k, f in VALUE_FLAGS:
        if a.get(k) not in (None, "", 0):
            flags += [f, str(a[k])]
    out = f"training/runs/{rid}/valpae_flags.txt"
    with open(out, "w") as fh:
        fh.write(" ".join(flags))
    skipped = {k: a.get(k) for k in TRAINING_ONLY if a.get(k) not in (None, "", 0)}
    print(f"  {rid}: {len(flags)} flag token(s) -> {out}")
    print(f"     lora_rank={a.get('lora_rank')}  expect_trainable={a.get('expect_trainable')}  "
          f"split={a.get('pae_split_accept')}")
    print("     NOT copied (training-only): "
          + (", ".join(f"{k}={v}" for k, v in skipped.items()) if skipped else "none set"))


if __name__ == "__main__":
    main()
