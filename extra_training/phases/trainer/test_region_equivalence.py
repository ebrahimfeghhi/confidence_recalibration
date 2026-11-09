"""Every extracted region must still be the live source, line for line.

This is the test that makes the copy safe to work on. The danger is not that an extraction fails
to run -- it is that someone tidies a comment or reorders two installs while moving them, and the
copy then trains a DIFFERENT model while everything still passes. So each region is compared
against `phase_k_train.py` after undoing the two mechanical transforms the extraction applied:
dedent by one level, and rewrite free names to `rt.<name>`.

A deliberate change to the live trainer FAILS this test. That is the point -- `trainer/cli.py`
drifted silently once already, and the arm's command line was refused as a result.
"""
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(os.path.dirname(HERE), "phase_k_train.py")

#: module -> (first-body marker, live span). Spans are the contract.
REGIONS = {
    "ddp.py":        ("world = int(os.environ", 1715, 1772, False),
    "bootstrap.py":  ("model, configs = RC.build_model", 1774, 1935, False),
    "trainable.py":  ("prefixes, trainable = head_param_names", 2906, 3059, False),
    "optimiser.py":  ("opt = torch.optim.AdamW", 3439, 3450, False),
    "resume.py":     ("if a.resume:", 3451, 3858, False),
    "unit.py":       ("def run_unit", 4946, 5666, True),
    "evaluate.py":   ("def val_loss_checkpoint", 6356, 6831, True),
    "loop.py":       ("def train_loop", 7300, 8276, True),
}
CTX = re.compile(r"\brt\.")


def norm(lines, dedent):
    t = "".join(l + "\n" for l in lines)
    if dedent:
        t = textwrap.dedent(t)
    return [CTX.sub("", l).rstrip() for l in t.splitlines()]


def main():
    fails = []
    live_all = open(LIVE).read().splitlines()
    for mod, (marker, lo, hi, dedent) in sorted(REGIONS.items()):
        want = norm(live_all[lo - 1:hi], dedent)
        got_all = open(os.path.join(HERE, mod)).read().splitlines()
        # STRIP ctx. BEFORE MATCHING. The extraction rewrote free names to rt.<name>, so the
        # marker as it appears in the live source does not appear verbatim in the copy.
        def _bare(l):
            return CTX.sub("", l).strip().replace("(rt, ", "(", 1)
        start = next((i for i, l in enumerate(got_all) if _bare(l).startswith(marker)), None)
        if start is None:
            print(f"  FAIL  {mod:<14} marker {marker!r} not found"); fails.append(mod); continue
        got = [CTX.sub("", l).rstrip() for l in got_all[start:start + len(want)]]
        # the def line gained a ctx parameter; compare it with that undone
        if dedent and got:
            got[0] = got[0].replace("(rt, ", "(", 1)
        diffs = [(lo + i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
        ok = not diffs and len(got) == len(want)
        print(("  PASS  " if ok else "  FAIL  ") +
              f"{mod:<14} vs phase_k_train.py:{lo}-{hi}   {len(want)} lines" +
              ("" if ok else f"   {len(diffs)} differ"))
        if not ok:
            fails.append(mod)
            for ln, w, g in diffs[:3]:
                print(f"        line {ln}\n          live: {w.strip()[:88]}\n          copy: {g.strip()[:88]}")
    print()
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print(f"all {len(REGIONS)} regions match the live source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
