#!/usr/bin/env python
"""THE mixed= KEY ON --extra-eval ACTUALLY REACHES THE EVALUATOR'S ARGV.

Two halves, because a flag that parses but is never forwarded is the failure mode that matters:

  1. THE PARSER. The --extra-eval spec block is lifted out of phase_k_train.main() by its own
     source lines and exec'd against a fake namespace, so this tests the SHIPPING code rather
     than a copy that can drift. mixed=1 must set the flag, mixed=1 without a seed_manifest.json
     beside the pool must hard-stop, and a non-boolean must hard-stop.

  2. THE FORWARDING. eval_checkpoint builds `extra` and only adds --mixed-budget inside the
     `if os.path.exists(_sman)` branch, so the flag cannot arrive without the manifest it is
     double-locked against. Asserted on the source, because the real call needs three GPUs.

WHY A TEST AT ALL. The last two launcher defects in this project were a duplicate SKIP whose
empty value won and a `#` inside a backslash-continued env block -- both silently ran a DIFFERENT
experiment and both passed `bash -n`. An in-run pass that quietly scored 947 of 1,688 interfaces
while writing "the training set" on the label is the same class of mistake.
"""
import os, re, sys, ast

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases", "phase_k_train.py")
TEXT = open(SRC).read()
LINES = TEXT.splitlines()


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _block():
    """The spec-parsing block, located by sentinel rather than by line number."""
    start = next(i for i, l in enumerate(LINES) if l.strip() == "for _spec in a.extra_eval:")
    end = next(i for i, l in enumerate(LINES[start:], start)
               if "_EXTRA_EVAL.append(dict(name=_nm" in l)
    # the append is a continued call; take lines until the parens balance
    j, depth = end, 0
    while True:
        depth += LINES[j].count("(") - LINES[j].count(")")
        j += 1
        if depth <= 0:
            break
    src = "\n".join(l[4:] if l.startswith("    ") else l for l in LINES[start:j])
    ast.parse(src)
    return src


def run(specs, pool_exists=True):
    g = {"os": os, "re": re, "SystemExit": SystemExit,
         "_re_ee": re, "_EXTRA_EVAL": [], "_ee_names": set(),
         "a": _NS(extra_eval=specs, fold="training/fold")}
    exec(compile(_block(), "<extra_eval>", "exec"), g)
    return g["_EXTRA_EVAL"]


def expect_stop(specs, needle):
    try:
        run(specs)
    except SystemExit as e:
        assert needle in str(e), f"wrong stop: {e}"
        return str(e)
    raise AssertionError(f"expected a STOP containing {needle!r}, got none")


def main():
    MIXED = "name=trainfull,pool=training/fulltrain_mixed/pool.csv,fold=training/fold," \
            "seed_base=500000,budget=15,every=5,mixed=1"
    UNIFORM = "name=slice280,pool=training/slice280/pool.csv,fold=training/fold," \
              "seed_base=500000,budget=15,every=5"

    # 1. THE FLAG IS SET, AND THE DEFAULT IS FALSE -- the feature changes something.
    got = run([MIXED])
    assert len(got) == 1 and got[0]["mixed"] is True, got
    assert got[0]["budget"] == 15 and got[0]["pool"].endswith("fulltrain_mixed/pool.csv"), got
    print("ok   mixed=1 sets mixed True")

    got = run([UNIFORM])
    assert got[0]["mixed"] is False, got
    print("ok   a spec without mixed= stays False (every existing arm unchanged)")

    got = run([MIXED, UNIFORM])
    assert [g["mixed"] for g in got] == [True, False], got
    print("ok   the two pools keep their own flag in one run")

    # 2. mixed=1 WITHOUT A MANIFEST IS A HARD STOP. slice280 has no seed_manifest.json, so this
    #    is a real pool standing in for the mistake, not a fabricated path.
    assert not os.path.exists("training/slice280/seed_manifest.json"), \
        "fixture broke: slice280 now HAS a manifest, pick another uniform pool"
    expect_stop([UNIFORM + ",mixed=1"], "no seed_manifest.json beside the pool")
    print("ok   mixed=1 without a manifest hard-stops")

    # 3. A NON-BOOLEAN IS A HARD STOP, not silently false.
    expect_stop([MIXED.replace("mixed=1", "mixed=yes")], "must be 0 or 1")
    print("ok   mixed=yes hard-stops instead of reading as false")

    # 4. UNKNOWN KEYS STILL STOP, and the message now names mixed.
    msg = expect_stop([MIXED + ",crop_tokens=256"], "unknown key(s)")
    assert "mixed" in msg, msg
    print("ok   unknown keys still stop, and 'mixed' is listed as known")

    # 5. THE FORWARDING IS INSIDE THE MANIFEST BRANCH. Checked on the source: the only
    #    --mixed-budget in the trainer must sit under `if os.path.exists(_sman)`.
    idx = [i for i, l in enumerate(LINES) if '"--mixed-budget"' in l]
    assert len(idx) == 1, f"expected exactly one --mixed-budget site, found {len(idx)}"
    window = "\n".join(LINES[max(0, idx[0] - 14):idx[0] + 1])
    assert "if os.path.exists(_sman):" in window, window
    assert 'extra += ["--seed-manifest", _sman]' in window, window
    assert re.search(r"if mixed:\s*$", LINES[idx[0] - 1]), LINES[idx[0] - 3:idx[0] + 1]
    print("ok   --mixed-budget is forwarded only alongside the manifest, and only when asked")

    # 6. eval_checkpoint TAKES AND THE CALL SITE PASSES IT.
    assert re.search(r"def eval_checkpoint\([^)]*mixed=False\)", TEXT, re.S), "signature"
    assert 'mixed=_ee.get("mixed", False)' in TEXT, "extra_eval call site does not forward mixed"
    print("ok   eval_checkpoint accepts mixed and the extra-eval call site forwards it")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.exit(main())
