"""The extracted parser must produce the SAME config the ORIGINAL one did.

THE GROUND TRUTH IS NOT THE ORIGINAL PARSER. Comparing the extracted block against a copy of
itself proves nothing. Every run writes args.json from the values the ORIGINAL parser produced, so
parsing that run's real argv with the new parser and comparing field by field is a check against a
file this code did not write.

Two sources, and the fixture is the one that matters: fixtures/cli_argv_*.json holds a real run's
argv with machine paths templated, plus the args.json that run wrote, so the test keeps working
after the process exits and on another machine. The live /proc check runs too when a trainer
happens to be up, which catches a fixture that has gone stale.
"""
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.cli import build_parser, parse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # extra_training/
HOME = os.path.expanduser("~")
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def untemplate(v):
    return v.replace("{ROOT}", ROOT).replace("{HOME}", HOME) if isinstance(v, str) else v


def compare(tag, argv, expected):
    """Compare the fields THIS parser still has.

    The minimal CLI deliberately drops 55 flags the original carried (LoRA, cropping, the
    tau-based ranking forms, the probes). Those are absent by design, so the comparison is over
    the intersection -- and `shared` is asserted to cover essentially all of the MINIMAL parser's
    fields, which is the direction that matters: nothing this parser claims may disagree.
    """
    parsed = vars(parse(argv))
    shared = sorted(set(parsed) & set(expected))
    ok(len(shared) >= 0.9 * len(parsed), f"{tag}: args.json covers this parser's fields",
       f"{len(shared)} of {len(parsed)} ({len(set(expected) - set(parsed))} dropped by design)")
    diff = [(k, parsed[k], expected[k]) for k in shared if parsed[k] != expected[k]]
    ok(not diff, f"{tag}: every field matches what the ORIGINAL parser recorded",
       f"{len(shared)} identical" if not diff else f"{len(diff)} differ: {diff[:3]}")
    return parsed


def live_runs():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode() for a in raw if a]
        if not any(a.endswith("phase_k_train.py") for a in argv):
            continue
        rest = argv[next(k for k, a in enumerate(argv) if a.endswith("phase_k_train.py")) + 1:]
        if "--out" in rest:
            out.append((os.path.basename(rest[rest.index("--out") + 1]), rest))
    seen, uniq = set(), []        # three ranks share one argv; check it once
    for rid, argv in out:
        if rid not in seen:
            seen.add(rid); uniq.append((rid, argv))
    return uniq


def main():
    print("\n1  COMMITTED FIXTURES (portable, survive the run exiting)")
    fx = sorted(glob.glob(os.path.join(HERE, "fixtures", "cli_argv_*.json")))
    ok(bool(fx), "at least one fixture exists", f"{len(fx)} found")
    for f in fx:
        d = json.load(open(f))
        argv = [untemplate(t) for t in d["argv"]]
        exp = {k: untemplate(v) for k, v in d["expected"].items()}
        compare(os.path.basename(f), argv, exp)

    print("\n2  THE DEPENDENT DEFAULT IS APPLIED (argparse cannot express it)")
    a = parse(["--fold", "/tmp/somefold"])
    ok(a.labels == "/tmp/somefold/labels",
       "--labels defaults to <fold>/labels", a.labels)
    a2 = parse(["--fold", "/tmp/somefold", "--labels", "/tmp/explicit"])
    ok(a2.labels == "/tmp/explicit", "and an explicit --labels still wins", a2.labels)

    print("\n3  THE PARSER REFUSES WHAT THE ORIGINAL REFUSED")
    for bad, why in ((["--epochs", "notanumber"], "non-numeric --epochs"),
                     (["--no-such-flag"], "an unknown flag")):
        # argparse prints its whole usage block to stderr on refusal; swallow it so a passing
        # run is readable.
        _err = sys.stderr
        try:
            sys.stderr = io.StringIO()
            build_parser().parse_args(bad)
            sys.stderr = _err
            ok(False, f"rejects {why}", "accepted it")
        except SystemExit as e:
            sys.stderr = _err
            ok(e.code != 0, f"rejects {why}", f"exit {e.code}")
        finally:
            sys.stderr = _err

    print("\n4  LIVE PROCESS (runs only when a trainer is up; catches a stale fixture)")
    runs = live_runs()
    if not runs:
        print("        (no trainer running -- skipped, the fixtures above are the real check)")
    for rid, argv in runs:
        aj = os.path.join(ROOT, "training", "runs", rid, "args.json")
        if os.path.exists(aj):
            compare(f"live {rid}", argv, json.load(open(aj)))

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
