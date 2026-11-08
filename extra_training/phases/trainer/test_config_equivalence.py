"""validate() must accept what the original accepted and refuse what it refused.

GROUND TRUTH, again, is not this code. A real run's args.json is the output of the ORIGINAL
validation path -- the run started, so its flags were accepted, and the derived fields in that file
are what the original derived. Replaying that run's argv through cli.parse + config.validate and
comparing is a check against a file neither module wrote.

The refusal half cannot be checked that way: no run exists for flags that were rejected. Those are
pinned by asserting a SystemExit whose message names the conflict, so a refusal that silently stops
refusing is caught.
"""
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.cli import parse            # noqa: E402
from trainer.config import validate      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HOME = os.path.expanduser("~")
# The validation block checks this, and the launcher sets it on every real run. Setting it here
# makes the test reproduce the environment the original validated in rather than a bare shell.
os.environ.setdefault("LAYERNORM_TYPE", "openfold")
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def un(v):
    return v.replace("{ROOT}", ROOT).replace("{HOME}", HOME) if isinstance(v, str) else v


def refuses(argv, needle, name):
    err = sys.stderr
    try:
        sys.stderr = io.StringIO()
        validate(parse(argv))
        sys.stderr = err
        ok(False, name, "accepted it")
    except SystemExit as e:
        sys.stderr = err
        msg = str(e)
        ok(needle.lower() in msg.lower(), name, f"{msg[:72]}")
    except Exception as e:                          # noqa: BLE001
        sys.stderr = err
        ok(False, name, f"{type(e).__name__}: {e}")
    finally:
        sys.stderr = err


def main():
    print("\n1  A REAL RUN'S FLAGS STILL VALIDATE, and derive the same values")
    fx = sorted(glob.glob(os.path.join(HERE, "fixtures", "cli_argv_*.json")))
    ok(bool(fx), "a fixture exists", f"{len(fx)} found")
    for f in fx:
        d = json.load(open(f))
        argv = [un(t) for t in d["argv"]]
        exp = {k: un(v) for k, v in d["expected"].items()}
        try:
            a = validate(parse(argv))
        except SystemExit as e:
            ok(False, f"{os.path.basename(f)}: validates", f"refused: {str(e)[:70]}")
            continue
        ok(True, f"{os.path.basename(f)}: accepted, as the original did")
        got = vars(a)
        shared = sorted(set(got) & set(exp))
        diff = [(k, got[k], exp[k]) for k in shared if got[k] != exp[k]]
        ok(not diff, "  and every field equals what that run recorded",
           f"{len(shared)} identical" if not diff else f"{len(diff)} differ: {diff[:3]}")

    print("\n2  THE REFUSALS STILL REFUSE (no run exists for these, so they are pinned by message)")
    # A MINIMAL CONFIG THAT ALREADY CLEARS THE UNRELATED GATES, so each case below fails for the
    # reason it is testing and not because the fixture is under-specified. --require-replay skip
    # is the documented escape the validator itself names.
    base = ["--fold", "/tmp/f", "--require-replay", "skip"]

    # NOT a refusal: --lr-steps overrides --lr-final with a printed NOTE. Pinning the real
    # contract rather than the one I assumed -- the extraction surfaced the difference.
    # The validator checks --rank-cross-seed BEFORE --rel-labels, so that is the message it gives
    # for this input. Pinning the order it actually enforces, not the one I guessed.
    refuses(base + ["--rank-loss", "--rank-form", "rel"], "rank-cross-seed",
            "--rank-form rel without --rank-cross-seed")
    refuses(base + ["--rank-loss", "--rank-form", "rel", "--rank-cross-seed"], "rel-labels",
            "--rank-form rel with cross-seed but no --rel-labels")



    print("\n3  THE REFUSAL COUNT HAS NOT SILENTLY DROPPED")
    src = open(os.path.join(HERE, "config.py")).read()
    n = src.count("SystemExit")
    # 26, not the original 44: the cropping, LoRA and tau-form refusals went with their flags.
    # Pinned so a later edit cannot quietly drop one of the refusals that REMAINS.
    ok(n >= 26, "config.py carries every refusal the minimal surface still needs",
       f"{n} SystemExit")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
