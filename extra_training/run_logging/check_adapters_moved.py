#!/usr/bin/env python
"""Did the LoRA adapters actually train, or did the arm quietly re-run its base?

WHY THIS EXISTS, and why a gradient check is not enough. Under cuequivariance the triangle
multiplication never calls linear_a_p.forward: the kernel reads `.weight` off the module and does
one fused matmul. LoRALinear serves that through a differentiable `weight` property, so the
adapter rides a path this project had never put a grad-requiring tensor through before. If that
kernel's backward dropped the weight gradient, lora_A.grad and lora_B.grad would come back None,
_apply_step would fill them with zeros so the all-reduce shapes still line up, --grad-check-every
only asserts nothing OUTSIDE the head has a gradient, and nothing anywhere asserts that a
TRAINABLE tensor received one. The arm would spend its whole budget producing a bit-exact replay
of its base under a new name, with an identical-looking grad_norm trace.

THE CHECK IS A WEIGHT DIFF, NOT A GRADIENT PROBE. The adapted weights are FROZEN in the base arm,
so the base arm's own checkpoint is a valid "before" for them: any movement at all is movement the
adapters produced. Checkpoints are written merged (adapter folded into the base weight), so the
names match and no LoRA-aware loading is needed.

EXACTLY ZERO MOVEMENT IS THE FAILURE. Not "small" -- zero. bf16 training moves a live weight by
something.
"""
import argparse
import re
import sys

import torch


def load(p):
    s = torch.load(p, map_location="cpu", weights_only=False)
    return s.get("head") if isinstance(s.get("head"), dict) else s.get("model", s.get("state_dict", s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="a checkpoint where the adapted weights are FROZEN")
    ap.add_argument("--after", required=True, help="the LoRA arm's checkpoint")
    ap.add_argument("--pattern", default=r"blocks\.[123]\.tri_mul_(in|out)\.linear_(a_p|b_p|z)\.weight$")
    ap.add_argument("--expect", type=int, default=18)
    a = ap.parse_args()

    b, c = load(a.base), load(a.after)
    rx = re.compile(a.pattern)
    keys = sorted(k for k in b if rx.search(k) and k in c)
    if len(keys) != a.expect:
        sys.exit(f"STOP: pattern matched {len(keys)} tensor(s) in both checkpoints, expected "
                 f"{a.expect}. The names or the pattern are wrong, so this check proves nothing.")
    dead, moved = [], []
    for k in keys:
        d = (c[k].float() - b[k].float()).abs().max().item()
        (moved if d > 0 else dead).append((k.split("confidence_head.")[-1], d))
    for n, d in moved:
        print(f"  moved   {n:62} max|d| {d:.3e}")
    for n, d in dead:
        print(f"  DEAD    {n:62} max|d| {d:.3e}")
    print(f"\n  {len(moved)}/{len(keys)} adapted weight(s) moved")
    if dead:
        sys.exit(f"FAIL: {len(dead)} adapted weight(s) did NOT move at all. The adapters are not "
                 f"training -- this arm is re-running its base. STOP IT.")
    print("  VERDICT: the adapters are training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
