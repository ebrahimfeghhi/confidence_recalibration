#!/usr/bin/env python
"""Turn launch_default_arm.sh's DRY=1 line into the newline-per-arg file bench_ckpt_by_size wants.

WHY NOT WORD-SPLIT IT IN THE SHELL. The arm's argv contains --train-only with a quoted regex that
holds commas and backslashes, and --note with a paragraph of prose. `for a in $line` would shred
both. shlex splits it the way the shell that produced it would have.

WHY NOT HAND-WRITE THE FILE. bench_ckpt_by_size.py exists to measure the arm it is given, and its
--arm-argv default is captured from a LIVE run's /proc cmdline precisely so the benchmark cannot
drift from the run. For an arm that is not running yet there is no /proc to read, so the launcher's
own dry run is the only source that is still the launcher's output rather than someone's
transcription of it.
"""
import argparse
import shlex
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="a DRY=1 launch_default_arm.sh log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--script", default="phase_k_train.py")
    a = ap.parse_args()

    cand = [ln for ln in open(a.log, errors="replace").read().splitlines()
            if a.script in ln and "torchrun" in ln]
    if not cand:
        print(f"STOP: no line in {a.log} holds both 'torchrun' and {a.script!r}. "
              f"Either the dry run failed or the launcher's final line changed shape.")
        return 1
    if len(cand) > 1:
        print(f"STOP: {len(cand)} candidate lines in {a.log}; refusing to guess which run to "
              f"benchmark. Point --log at a log holding ONE dry run.")
        return 1

    parts = shlex.split(cand[0].strip())
    # FROM THE INTERPRETER ONWARD, dropping torchrun and its wrapper. bench_ckpt_by_size builds its
    # own single-rank invocation, so the elastic launcher's own flags must not be carried over --
    # they are what would make it join the live arm's process group.
    try:
        i = next(i for i, p in enumerate(parts) if p.endswith(a.script))
    except StopIteration:
        print(f"STOP: {a.script} is in the line but not as a whole argument.")
        return 1
    # NOT endswith("python"): torchrun's own --no-python ends with "python" and was picked up as
    # the interpreter, which put a flag in argv[0]. A path ending in /python, or the bare word.
    py = next((p for p in parts[:i]
               if not p.startswith("-") and (p.endswith("/python") or p == "python")), None)
    if py is None:
        print("STOP: no python interpreter found before the script in that line.")
        return 1
    argv = [py, "-u"] + parts[i:]

    # THE ELASTIC ENVIRONMENT MUST NOT SURVIVE, same reason the --loss-only child strips it.
    bad = [p for p in argv if p.startswith("--nproc") or p.startswith("--master_port")]
    if bad:
        print(f"STOP: torchrun flags survived the split: {bad}")
        return 1

    with open(a.out, "w") as fh:
        fh.write("\n".join(argv) + "\n")
    n_out = sum(1 for p in argv if p == "--out")
    print(f"wrote {len(argv)} argument(s) to {a.out} "
          f"({n_out} --out; bench_ckpt_by_size overrides it)")
    print(f"  interpreter {argv[0]}")
    print(f"  script      {argv[2]}")
    for flag in ("--pose-inblock", "--pose-inblock-pae", "--lean-single", "--blocks-per-ckpt",
                 "--train-only", "--manifest-skip"):
        if flag in argv:
            j = argv.index(flag)
            val = argv[j + 1] if j + 1 < len(argv) and not argv[j + 1].startswith("--") else ""
            print(f"  {flag:<20} {val[:90]}")
        else:
            print(f"  {flag:<20} ABSENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
