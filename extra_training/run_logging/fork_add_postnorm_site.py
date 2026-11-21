#!/usr/bin/env python
"""Fork a 5-site pose-injection bundle into a 6-site one by adding the post-LayerNorm site.

WHY A FORK AND NOT A NEW ARM. Training a sixth-site arm from scratch costs 60 epochs and compares
against a different random trajectory. Forking at epoch 40 and replaying 41-50 with the same data
order compares the extra geometry input against the five-site run ON THE SAME held-out interfaces
at the same epochs, which the five-site CSV already records -- a matched comparison for a sixth of
the compute.

WHY --resume ALONE CANNOT DO THIS. phase_k_train.py:4593 hard-stops when a trainable tensor is
absent from the bundle, and the five-site bundle has no pose_readout_post_* tensors; line 4604
then restores AdamW param groups directly, which would mismatch 15 live params against 18. So the
bundle itself has to be extended before the trainer sees it.

WHAT THIS PRESERVES AND WHAT IT CREATES FRESH. Every existing weight, every existing AdamW moment,
and the step counter are carried over untouched. The three new tensors are zero and have NO
optimiser state, so AdamW initialises them on their first step -- fresh moments, which is correct
for a parameter that has never moved.

THE INDEX ASSUMPTION IS ASSERTED, NOT ASSUMED. AdamW state is keyed by integer position in
model.named_parameters() order (phase_k_train.py:4151), so appending tensors is only safe if the
new ones land AFTER all fifteen existing ones. pose_readout.py registers the post_* modules
immediately after pose_readout_frame_dir, so they should be 15/16/17 -- but if the live order ever
differs, every restored moment would attach to the wrong weight silently. This refuses rather than
guess: run with --verify-order and a built 6-site model to check before forking.

ZERO-INIT MEANS STEP 0 IS BIT-IDENTICAL. The post-norm site adds its term through zero matrices,
so the forked head must reproduce the parent's PAE logits exactly on the same poses before the
first update. Verify that (--loss-only on both, same pool, compare logits) BEFORE reading anything
into what the sixth site learns.
"""
import argparse
import os
import sys

import torch

NEW = ["confidence_head.pose_readout_post_d.weight",
       "confidence_head.pose_readout_post_d_wo_onehot.weight",
       "confidence_head.pose_readout_post_frame_dir.weight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the five-site resume_ep<N>.pt")
    ap.add_argument("--out", required=True, help="where to write the six-site bundle")
    ap.add_argument("--order", default="",
                    help="a file with the SIX-SITE model's trainable named_parameters(), one per "
                         "line, in order. Produced by the trainer at startup. Without it this "
                         "refuses to write, because the AdamW index mapping cannot be checked.")
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--num-bins", type=int, default=39)
    ap.add_argument("--force-append", action="store_true",
                    help="write without an --order file, trusting that the three new tensors "
                         "append at the end. Only for a dry inspection.")
    a = ap.parse_args()

    b = torch.load(a.src, map_location="cpu", weights_only=False)
    head, opt = b["head"], b["opt"]
    names = list(head)
    n_old = len(names)
    print(f"source: {a.src}  epoch {b['epoch']}  step {b['step']}  {n_old} trainable tensor(s)")

    # THE BUNDLE'S OWN CONSISTENCY FIRST. Index i of the optimiser state must describe tensor i of
    # the head dict; if that is not true, everything below is attaching moments to wrong weights.
    for i, n in enumerate(names):
        st = opt["state"].get(i)
        if st is None:
            sys.exit(f"STOP: optimiser state has no entry for index {i} ({n})")
        if tuple(st["exp_avg"].shape) != tuple(head[n].shape):
            sys.exit(f"STOP: index {i} ({n}) head {tuple(head[n].shape)} against moment "
                     f"{tuple(st['exp_avg'].shape)} -- the bundle's order is not what it seems")
    print(f"  bundle self-check: all {n_old} moments match their tensor by shape")

    for n in NEW:
        if n in head:
            sys.exit(f"STOP: {n} is already in the bundle; this is not a five-site checkpoint")

    if a.order:
        want = [x.strip() for x in open(a.order) if x.strip()]
        if want[:n_old] != names:
            sys.exit("STOP: the six-site model's first "
                     f"{n_old} trainable tensors are not the five-site order.\n"
                     f"  bundle: {names[:3]} ...\n  model : {want[:3]} ...\n"
                     "Appending would attach every restored moment to the wrong weight.")
        if sorted(want[n_old:]) != sorted(NEW):
            sys.exit(f"STOP: expected exactly {NEW} after the first {n_old}, got {want[n_old:]}")
        new_order = want[n_old:]
        print(f"  order file: the three new tensors are at indices "
              f"{list(range(n_old, n_old + 3))}, existing indices unchanged")
    elif a.force_append:
        new_order = NEW
        print("  WARNING: no --order file; assuming the three new tensors append at the end. "
              "Do not train from a bundle written this way.")
    else:
        sys.exit("STOP: pass --order with the six-site model's trainable parameter order, or "
                 "--force-append for a dry inspection. The AdamW index mapping cannot be "
                 "checked without it.")

    shapes = {NEW[0]: (a.c_z, a.num_bins), NEW[1]: (a.c_z, 1), NEW[2]: (a.c_z, 3)}
    for n in new_order:
        head[n] = torch.zeros(shapes[n], dtype=next(iter(head.values())).dtype)
    # NO OPTIMISER STATE FOR THE NEW ONES. param_groups grows to cover them; opt["state"] does
    # not, and AdamW creates exp_avg/exp_avg_sq at their first step. Writing zero moments by hand
    # would ALSO set step=0 for them, which is the same thing, but lazily is what the optimiser
    # does for a freshly added parameter and is one less thing to get wrong.
    opt["param_groups"][0]["params"] = list(range(n_old + len(new_order)))
    b["guard"] = b.get("guard", {})
    b["guard"]["forked_from"] = os.path.abspath(a.src)
    b["guard"]["forked_added"] = list(new_order)

    torch.save(b, a.out)
    print(f"wrote {a.out}: {len(head)} tensor(s), {len(opt['state'])} restored moment(s), "
          f"param group of {len(opt['param_groups'][0]['params'])}, epoch {b['epoch']} "
          f"step {b['step']}")
    print("\nBEFORE TRAINING FROM THIS, verify the fork reproduces the parent's PAE logits on the "
          "same poses:\n  the three new matrices are zero, so step 0 must be bit-identical. If it "
          "is not,\n  the post-norm site is not actually zero-init in this build and nothing "
          "downstream is\n  interpretable.")


if __name__ == "__main__":
    main()
