#!/usr/bin/env python
"""The eval flags a checkpoint's own run trained with, READ OFF ITS args.json. Never retyped.

WHAT THIS PREVENTS, MEASURED TODAY. Arm B's head (training/runs/w57hjo) trained with
--pose-readout --pose-readout-prenorm on top of the three pathway flags score_checkpoint.sh
hardcodes. The scoring pass was launched without them, so all three shards loaded the model
WITHOUT the readout pathway and refused the checkpoint 90 seconds in:

    6 checkpoint tensor(s) the model does not have, e.g.
    ['confidence_head.pose_readout_d.weight', ...]

That refusal is the lucky case. phase_k_eval says so itself: --pose-perblock-share and
--pose-readout-prenorm add NO tensors, so when one of THOSE is the flag left off, the load
succeeds and the number is simply wrong -- an in-sample pass over 7,279 units reporting a head
scored through a pathway it was never trained through.

SO THE ARGV IS COPIED, NOT COMPOSED. The run directory beside the checkpoint holds args.json, the
record of what the run actually trained with; this turns it into the eval's flag string.

AND ONLY FLAGS phase_k_eval REALLY DEFINES. The list is intersected with the --long-options
phase_k_eval.py's own argparse declares, read out of the source. A train-only option (pose_batch,
pose_readout_copy_distance -- an init-time weight copy the checkpoint already carries) therefore
cannot be emitted as a flag the evaluator would reject, and a pathway key that is TRUE but has no
evaluator flag is printed on stderr rather than dropped in silence.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "phases", "phase_k_eval.py")
# PATHWAY KEYS ONLY. args.json holds the whole training argv; the crop, the schedule and the loss
# weights are not things the evaluator installs, and forwarding them would be inventing an argv
# rather than copying one.
PREFIX = ("pose_", "frame_")


def eval_flags(src=EVAL):
    return set(re.findall(r'ap\.add_argument\("(--[a-z0-9-]+)"', open(src).read()))


def flags_for(ckpt, defined=None):
    """(flag list, notes) for a checkpoint path, or ([], notes) when there is nothing to read."""
    defined = eval_flags() if defined is None else defined
    notes, run = [], os.path.dirname(os.path.abspath(ckpt))
    args = os.path.join(run, "args.json")
    if not os.path.exists(args):
        return [], [f"no args.json beside {ckpt}; nothing to derive"]
    try:
        with open(args) as fh:
            d = json.load(fh)
    except (ValueError, OSError) as e:
        return [], [f"STOP-WORTHY: {args} is unreadable ({type(e).__name__}: {e})"]
    out = []
    for k in sorted(d):
        if not k.startswith(PREFIX):
            continue
        v, flag = d[k], "--" + k.replace("_", "-")
        if v is True:
            (out if flag in defined else notes).append(
                flag if flag in defined else
                f"{k} is true in args.json and phase_k_eval has no {flag} -- train-only, or a "
                f"pathway the evaluator cannot install")
        elif isinstance(v, int) and not isinstance(v, bool) and flag in defined and v != -1:
            out += [flag, str(v)]
    return out, notes


def main():
    """Exit 0 with the flag string, or 2 when the record could not be READ at all.

    EVERY FAILURE USED TO EXIT 0, which made the caller's `|| exit 4` unreachable: an unreadable
    args.json printed a note to stderr and handed back an empty POSE_FLAGS, and because the
    derivation was empty the caller's `say` line was skipped too, so the pass log carried no
    trace of it. For an arm whose only extra flags are the two that add NO tensors
    (--pose-perblock-share, --pose-readout-prenorm) that is exactly the silently-wrong score this
    file exists to prevent. Found in review.

    ABSENT IS NOT THE SAME AS UNREADABLE, and only the second is fatal here. A checkpoint with no
    run directory beside it (the shipped head, a copied-out file) has nothing to derive and must
    keep working as it did; the caller says out loud that it derived nothing.
    """
    if len(sys.argv) != 2:
        sys.exit("usage: pathway_flags.py <path/to/head_epN_stepM.pt>")
    # AN EMPTY PATH IS THE SHIPPED HEAD, AND THE SHIPPED HEAD HAS NO FLAGS TO DERIVE. Refused
    # rather than answered, because every part of answering it is wrong:
    # os.path.dirname(os.path.abspath("")) is the CURRENT DIRECTORY, so this would look for
    # extra_training/args.json -- some other run's record if one is ever left there -- and
    # otherwise hand back an empty string that the caller cannot distinguish from "this
    # checkpoint needs no extra flag". That is exactly what happened on 2026-11-18: a shipped
    # pass was launched with CKPT="" through a copy of score_checkpoint.sh, the derivation said
    # nothing, and COMMON's hardcoded --pose-perblock --pose-perblock-copy --frame-direction were
    # left as the only flags in play. Twelve minutes of three cards scored
    # shipped-plus-injections under the name "shipped".
    if not sys.argv[1].strip():
        print("  pathway_flags: EMPTY checkpoint path. The shipped head has no run directory and "
              "no args.json, so there is nothing to derive from -- and an empty answer here is "
              "indistinguishable from 'this head needs no flag'. Score the shipped head with "
              "shell_scripts/score_shipped_head.sh, which installs NO pathway at all; this "
              "script is for a trained head's own checkpoint.", file=sys.stderr)
        return 2
    flags, notes = flags_for(sys.argv[1])
    for n in notes:
        print(f"  pathway_flags: {n}", file=sys.stderr)
    if any(n.startswith("STOP-WORTHY") for n in notes):
        return 2
    print(" ".join(flags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
