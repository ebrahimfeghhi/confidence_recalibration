"""Every line of an extracted region must still exist in the live source, in order.

This is the test that makes the copy safe to work on. The danger is not that an extraction fails
to run -- it is that someone tidies a comment or reorders two installs while moving them, and the
copy then trains a DIFFERENT model while everything still passes. So each region is compared
against `phase_k_train.py` after undoing the two mechanical transforms the extraction applied:
dedent by one level, and rewrite free names to `rt.<name>`.

SUBSEQUENCE, NOT EQUALITY, and that is deliberate. The copy drops branches guarded by flags
cli.py cannot express, so it is strictly shorter. Requiring equality would fail for a good
reason -- and a test that fails for good reasons gets switched off. Requiring that every line
the copy KEPT still appears in the live region, in order, catches the thing that matters: a line
that was edited rather than deleted. Deleting is allowed; rewriting is not.

(The first attempt leaned the live side too, by running the same strip on it. That failed on the
regions that do not parse standalone -- bootstrap starts inside a `try:` -- so the live side
stayed fat and the comparison was meaningless for exactly the two largest regions.)

A deliberate change to the live trainer FAILS this test. That is the point -- `trainer/cli.py`
drifted silently once already, and the arm's command line was refused as a result.
"""
import os
import re
import sys
import textwrap

import lean

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

#: Lines the extraction ADDED, which therefore cannot appear in the live region. Kept as an
#: explicit allowlist rather than a loose rule, so anything else that appears in the copy and not
#: in the source is still a failure. These are the function scaffolding: `main()` had no returns,
#: because its locals simply stayed in scope.
ADDED = re.compile(r"^(return |def |pass  # body removed|import |from |seed_base=-1, budget=0\):)")


def norm(lines, dedent):
    t = textwrap.dedent("".join(l + "\n" for l in lines))
    return [CTX.sub("", l).rstrip() for l in t.splitlines()]


def subsequence(copy, live):
    """(ok, first_bad_copy_line, text) -- is `copy` an ordered subsequence of `live`?"""
    i = 0
    for n, cl in enumerate(copy):
        if not cl.strip():
            continue
        j = i
        while j < len(live) and live[j] != cl:
            j += 1
        if j == len(live):
            return False, n, cl
        i = j + 1
    return True, -1, ""


def main():
    fails = []
    live_all = open(LIVE).read().splitlines()
    for mod, (marker, lo, hi, dedent) in sorted(REGIONS.items()):
        want = norm(live_all[lo - 1:hi], dedent)
        got_all = open(os.path.join(HERE, mod)).read().splitlines()

        def _bare(l):
            return CTX.sub("", l).strip().replace("(rt, ", "(", 1)
        start = next((i for i, l in enumerate(got_all) if _bare(l).startswith(marker)), None)
        if start is None:
            print(f"  FAIL  {mod:<14} marker {marker!r} not found"); fails.append(mod); continue

        # the copy's region runs to the end of the file or to its own return
        raw = textwrap.dedent("\n".join(got_all[start:]) + "\n")
        got = [CTX.sub("", l).rstrip() for l in raw.splitlines()]
        got = [g for g in got if "pass  # body removed" not in g]
        if dedent and got:
            got[0] = got[0].replace("(rt, ", "(", 1)
        # only compare as far as the live region goes
        # STOP AT THE NEXT TOP-LEVEL def. evaluate.py holds two extracted functions and this
        # entry covers only the first; without this the second function's body is compared
        # against a region that does not contain it.
        keep, seen = [], 0
        for n_i, g in enumerate(got):
            if seen >= len(want):
                break
            if n_i and g.startswith("def "):
                break
            keep.append(g)
            if g.strip():
                seen += 1
        keep = [k for k in keep if not ADDED.match(k.strip())]
        ok, bad, text = subsequence(keep, want)
        kept = sum(1 for g in keep if g.strip())
        print(("  PASS  " if ok else "  FAIL  ") +
              f"{mod:<14} {kept:>4} kept of {sum(1 for w in want if w.strip()):>4} live lines" +
              ("" if ok else f"   first rewritten line: {text.strip()[:70]!r}"))
        if not ok:
            fails.append(mod)
    print()
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print(f"all {len(REGIONS)} regions are faithful subsequences of the live source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
