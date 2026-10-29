#!/usr/bin/env python
"""Assert on a smoke run's artifacts: did the row weighting actually DO anything on real poses?

Reads only files the trainer wrote, so it checks the shipped glue rather than a reimplementation
of it. Usage:  check_smoke_rowweight.py <run_dir> <trainer_exit_code>
"""
import csv
import math
import os
import sys

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    out, rc = sys.argv[1], int(sys.argv[2])
    log = os.path.join(out, "run.log")
    txt = open(log).read() if os.path.exists(log) else ""
    if not txt:
        for alt in ("smoke.log",):
            p = os.path.join(out, alt)
            if os.path.exists(p):
                txt = open(p).read()
    check("the trainer exited 0", rc == 0, f"exit {rc}")
    check("the row-weight banner fired (the flag reached the hook)",
          "PAE term ROW-WEIGHTED" in txt,
          next((l.strip()[:120] for l in txt.splitlines() if "ROW-WEIGHTED" in l), "absent"))
    check("the steps/epoch correction printed",
          "from the epoch builder" in txt or "COSINE" in txt,
          next((l.strip()[:120] for l in txt.splitlines()
                if "from the epoch builder" in l), ""))
    check("the diagnostic did not report a failure",
          "row diagnostic failed" not in txt and "unweighted PAE readback failed" not in txt,
          next((l.strip()[:120] for l in txt.splitlines() if "failed" in l), "clean"))

    # the row diagnostic
    rows = []
    for f in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        if f.startswith("pae_rows_rank") and f.endswith(".csv"):
            rows += list(csv.DictReader(open(os.path.join(out, f))))
    check("pae_rows_rank*.csv has rows", len(rows) > 0, f"{len(rows)} row(s)")
    if rows:
        ess = [float(r["ess"]) for r in rows if r.get("ess")]
        nr = [float(r["n_scored_rows"]) for r in rows if r.get("n_scored_rows")]
        ok_ess = all(math.isfinite(e) and e > 0 for e in ess)
        check("every ESS is finite and positive", ok_ess,
              f"min {min(ess):.1f} max {max(ess):.1f}" if ess else "none")
        below = sum(1 for e, n in zip(ess, nr) if e < n)
        check("ESS is BELOW the scored row count (the weighting concentrated)",
              below == len(ess), f"{below} of {len(ess)}")
        w = [float(r["pae_weighted"]) for r in rows if r.get("pae_weighted")]
        u = [float(r["pae_unweighted"]) for r in rows if r.get("pae_unweighted")]
        diff = [abs(x - y) for x, y in zip(w, u)]
        check("weighted and unweighted PAE DIFFER on real poses (the no-op check)",
              bool(diff) and max(diff) > 1e-6,
              f"max |d| {max(diff):.6f} over {len(diff)} pose(s)" if diff else "no pairs")
        check("the unweighted value is finite everywhere",
              all(math.isfinite(x) for x in u), f"{sum(1 for x in u if not math.isfinite(x))} bad")
        rho = [float(r["rho_rowweight_vs_rowloss"]) for r in rows
               if r.get("rho_rowweight_vs_rowloss")
               and math.isfinite(float(r["rho_rowweight_vs_rowloss"]))]
        if rho:
            print(f"\n  FIRST LOOK AT THE PREMISE: rho(row weight, that row's PAE loss) over "
                  f"{len(rho)} pose(s), mean {sum(rho)/len(rho):+.4f}, "
                  f"range {min(rho):+.4f}..{max(rho):+.4f}")
            print("  Negative means the focused rows are already the BETTER-fit ones, which is the "
                  "reading under which this weighting has little headroom.")

    # the trajectory column
    traj = os.path.join(out, "trajectory.csv")
    if os.path.exists(traj):
        tr = list(csv.DictReader(open(traj)))
        check("trajectory.csv carries pae_loss_unweighted",
              bool(tr) and "pae_loss_unweighted" in tr[0], f"{len(tr)} row(s)")
        pairs = [(float(r["pae_loss"]), float(r["pae_loss_unweighted"])) for r in tr
                 if r.get("pae_loss") and r.get("pae_loss_unweighted")
                 and math.isfinite(float(r["pae_loss_unweighted"]))]
        check("the two trajectory columns differ", bool(pairs)
              and max(abs(a - b) for a, b in pairs) > 1e-6,
              f"max |d| {max(abs(a-b) for a, b in pairs):.6f}" if pairs else "no pairs")
    else:
        check("trajectory.csv exists", False)

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
