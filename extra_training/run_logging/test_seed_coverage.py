#!/usr/bin/env python
"""seed_coverage's contract, on SYNTHETIC rows.

DELIBERATELY NOT THE PRODUCTION CSV. Raised in review and right: "437 of 437 at seed 500000" proves
the tool agrees with one dataset, not that it refuses what it promises to refuse. Every case here is
six hand-written rows where the answer is known by construction, including the two that the first
version got wrong -- an unexpected interface and a stray arm both returned 0 while the docstring
said they were refused.
"""
import csv, os, subprocess, sys, tempfile

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "seed_coverage.py")
SEED = "500000"
HDR = ["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"]
FAIL = []


def row(iface, sample, arm="free", dockq="0.5", iptm="0.5", seed=SEED):
    return [iface, seed, arm, "-1", "", str(sample), dockq, iptm]


def run(rows, pool, structures=("1abc",), nsample=5, plan=False):
    """out: (exit status, stdout, plan rows)"""
    d = tempfile.mkdtemp()
    rp = os.path.join(d, "results.csv")
    with open(rp, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(HDR); w.writerows(rows)
    pp = os.path.join(d, "pool.csv")
    with open(pp, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["pdb", "chain1", "chain2"])
        w.writerows(pool)
    sp = os.path.join(d, "structs.txt")
    open(sp, "w").write("\n".join(structures) + "\n")
    planp = os.path.join(d, "plan.csv")
    cmd = [PY, "-I", TOOL, "--results", rp, "--pool", pp, "--structures", sp,
           "--seed", SEED, "--nsample", str(nsample)]
    if plan:
        cmd += ["--plan-out", planp]
    p = subprocess.run(cmd, capture_output=True, text=True)
    got = []
    if plan and os.path.exists(planp):
        got = [tuple(r) for r in list(csv.reader(open(planp)))[1:]]
    return p.returncode, p.stdout + p.stderr, got


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    POOL1 = [("1abc", "A", "B")]                      # one interface, 1abc_AB
    POOL2 = [("1abc", "A", "B"), ("1abc", "A", "C")]  # two: 1abc_AB and 1abc_AC
    full1 = [row("1abc_AB", k) for k in range(5)]
    full2 = full1 + [row("1abc_AC", k) for k in range(5)]

    rc, out, _ = run(full1, POOL1)
    check("exact coverage of one interface passes", rc == 0, f"exit {rc}")

    # (i) EXACT COVERAGE PLUS AN EXTRA ARM. The first version put this in a list the exit status
    # never read, so it printed the problem and returned success.
    rc, out, _ = run(full1 + [row("1abc_AB", 0, arm="tmpl")], POOL1)
    check("exact coverage + a wrong-arm row FAILS", rc != 0, f"exit {rc}")
    check("  and says the arm is forbidden", "arm=" in out and "FORBIDDEN" in out)

    # (ii) EXACT COVERAGE PLUS AN EXTRA INTERFACE, and plus an extra sample index.
    rc, out, _ = run(full1 + [row("1abc_AZ", 0)], POOL1)
    check("exact coverage + an interface the pool does not name FAILS", rc != 0, f"exit {rc}")
    check("  and names it", "not named by the pool" in out)
    rc, out, _ = run(full1 + [row("1abc_AB", 7)], POOL1)
    check("exact coverage + a sample index outside 0..4 FAILS", rc != 0, f"exit {rc}")

    # (iii) MISSING ONE SAMPLE.
    rc, out, plan = run(full1[:-1], POOL1, plan=True)
    check("missing one sample FAILS", rc != 0, f"exit {rc}")
    check("  and the plan names exactly that one key", plan == [("1abc_AB", SEED, "4")], f"{plan}")

    # (iv) A DUPLICATE KEY.
    rc, out, _ = run(full1 + [row("1abc_AB", 2)], POOL1)
    check("a duplicated key FAILS", rc != 0, f"exit {rc}")
    check("  and says it appears twice", "appears 2 times" in out)

    # (v) ONE INTERFACE AT 1/5 AND ANOTHER AT 5/5 -- the 9vaw_AB shape, and the case the merge
    # contract could not express. The plan must name 4 keys, not 5 and not 10.
    rows = full1 + [row("1abc_AC", 0)]
    rc, out, plan = run(rows, POOL2, plan=True)
    check("5/5 beside 1/5 FAILS", rc != 0, f"exit {rc}")
    check("  and the plan names the 4 MISSING keys only",
          plan == [("1abc_AC", SEED, str(k)) for k in (1, 2, 3, 4)], f"{plan}")
    check("  and does NOT name the complete interface",
          not any(i == "1abc_AB" for i, _s, _k in plan))

    # (vi) NaN, AND OUT OF RANGE. Not a score, so the interface is short -- same outcome as absent.
    rc, out, plan = run(full1[:-1] + [row("1abc_AB", 4, dockq="nan")], POOL1, plan=True)
    check("a NaN row counts as missing, not as a score", rc != 0, f"exit {rc}")
    check("  and is reported as not a score", "NOT A SCORE" in out)
    check("  and the plan still asks for that key", plan == [("1abc_AB", SEED, "4")], f"{plan}")
    rc, out, _ = run(full1[:-1] + [row("1abc_AB", 4, iptm="7.5")], POOL1)
    check("an ipTM outside [0, 1] counts as missing", rc != 0, f"exit {rc}")

    # A ROW FOR A DIFFERENT SEED MUST BE IGNORED, not counted toward this one.
    rc, out, _ = run(full1[:-1] + [row("1abc_AB", 4, seed="500099")], POOL1)
    check("a row at another seed does not complete this seed", rc != 0, f"exit {rc}")

    # TWO COMPLETE INTERFACES PASS, so the gate is not simply always failing.
    rc, out, _ = run(full2, POOL2)
    check("both interfaces complete passes", rc == 0, f"exit {rc}")

    # A STRUCTURE WITH NO ROWS AT ALL, and its plan is the whole key set.
    rc, out, plan = run([], POOL1, plan=True)
    check("no rows at all FAILS", rc != 0, f"exit {rc}")
    check("  and the plan asks for all 5 keys", len(plan) == 5, f"{len(plan)}")

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
