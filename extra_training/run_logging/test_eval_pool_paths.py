#!/usr/bin/env python
"""Gate for phase_k_eval's two execution paths under --pool-mean.

WHY IT EXISTS. The --pool-mean change broke BOTH paths and shipped:
  - mean_v was created in merge_rows() and read in aggregate(), so every sharded merge raised
    NameError -- including --select-by max, which does not use the mean at all.
  - the single-process path appended SIX fields per row while aggregate unpacked FIVE, so it
    raised ValueError.
Neither was caught, because no test exercised aggregate() at all. This does, on both row formats
and both selection rules, with synthetic rows rather than a model so it runs on CPU in a second.

THE PROPERTY THAT MATTERS is that the two paths agree. They cannot be compared end to end without
a GPU, so what is asserted instead is that aggregate() is INDIFFERENT to which path produced its
rows: the same poses in either representation must give the same metric.

Usage:  python -I run_logging/test_eval_pool_paths.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# TWO INTERFACES, THREE POSES EACH, with the max and mean columns deliberately disagreeing on
# which pose is best -- so a path that silently falls back to the max cannot pass by accident.
#
# THE MEANS MUST ALSO DIFFER, not just the picks. The first fixture had the two selections pick
# different poses whose DockQ happened to average to the same 0.5000, so the inequality assertion
# failed while the code was correct. Worked out explicitly below rather than tuned until green.
#
#            iface     seed  samp  iptm_max  dockq  iptm_mean
POSES = [
    ("aaaa_AB", 500000, 0, 0.90, 0.10, 0.20),   # max picks this  -> DockQ 0.10
    ("aaaa_AB", 500000, 1, 0.50, 0.80, 0.70),   # mean picks this -> DockQ 0.80
    ("aaaa_AB", 500000, 2, 0.40, 0.30, 0.10),
    ("bbbb_AB", 500000, 0, 0.30, 0.70, 0.60),   # mean picks this -> DockQ 0.70
    ("bbbb_AB", 500000, 1, 0.70, 0.30, 0.40),   # max picks this  -> DockQ 0.30
    ("bbbb_AB", 500000, 2, 0.20, 0.40, 0.10),
]
# by MAX : 0.10, 0.30 -> mean_dockq 0.2000, frac_acc 0.5000  (only 0.30 clears 0.23)
# by MEAN: 0.80, 0.70 -> mean_dockq 0.7500, frac_acc 1.0000
EXPECT = {"max": (0.2000, 0.5000), "mean": (0.7500, 1.0000)}


def shard_csvs(d, with_mean=True, n_shards=2):
    """Synthetic shard files in the exact format phase_k_eval writes."""
    paths = []
    per = len(POSES) // n_shards
    for sh in range(n_shards):
        p = os.path.join(d, f"rows_b1_s{sh}.csv")
        cols = ["iface", "seed", "sample", "iptm", "dockq", "stored_iptm"]
        if with_mean:
            cols.append("iptm_mean")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i, s_, k, mx, dq, mn in POSES[sh * per:(sh + 1) * per]:
                row = [i, s_, k, repr(mx), f"{dq:.4f}", ""]
                if with_mean:
                    row.append(repr(mn))
                w.writerow(row)
        json.dump({"shard": sh, "nshards": n_shards, "units": per, "rows": per,
                   "n_ost": per, "n_unresolvable": 0, "n_differ": 0, "differ_pdbs": {},
                   "n_unmapped": 0, "unmapped": [], "budget": 1, "ckpt": "synthetic"},
                  open(p + ".meta.json", "w"))
        paths.append(p)
    return paths


def synth_pool(d):
    """A pool and a chain_roles table covering ONLY the synthetic interfaces.

    Without this the evaluator checks the rows against the default valfull pool and refuses with
    "92 pool interface(s) have no scored pose" -- correctly, since the coverage contract is the
    point of that guard. The test has to supply its own universe, not disable the check.
    """
    pool = os.path.join(d, "pool.csv")
    with open(pool, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2"])
        for pdb in ("aaaa", "bbbb"):
            w.writerow([pdb, "A", "B"])
    roles = os.path.join(d, "chain_roles.csv")
    with open(roles, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "model_chain", "ref_chain", "auth_chain", "role"])
        for pdb in ("aaaa", "bbbb"):
            w.writerow([pdb, "A", "A", "A", "H"])
            w.writerow([pdb, "B", "B", "B", "L"])
    return pool, roles


def run_cli(d, with_mean, select_by, extra=()):
    """THE REAL CLI over the merge path, which is what exercises aggregate(). No namespace is
    faked: the first version of this test built one by hand and failed on seed_base and pool,
    which proves only that the fake was incomplete."""
    paths = shard_csvs(d, with_mean=with_mean)
    pool, roles = synth_pool(d)
    out = os.path.join(d, f"m_{with_mean}_{select_by}.json")
    cmd = [PY_BIN, "-I", os.path.join(R, "phases", "phase_k_eval.py"),
           "--rows-in", ",".join(paths), "--budget", "1",
           "--ckpt", "synthetic", "--no-self-check",
           "--pool", pool, "--roles", roles,
           # SEED BASE AND --allow-partial. The intended-key set is pool x seeds x samples, and a
           # six-pose fixture cannot satisfy a contract written for 10 seeds x 5 samples. The
           # coverage contract is NOT what this test covers -- it has its own guard and every real
           # run exercises it -- so it is stood down explicitly here rather than worked around by
           # reshaping the fixture until it happens to pass.
           "--seed-base", "500000", "--allow-partial",
           "--select-by", select_by, "--out", out, *extra]
    if with_mean:
        cmd.append("--pool-mean")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, LAYERNORM_TYPE="openfold"))
    j = json.load(open(out)) if os.path.exists(out) else None
    return r, j


def main():
    with tempfile.TemporaryDirectory() as d:
        # 1. the merge path must RUN -- it raised NameError on mean_v for every selection rule
        r, j = run_cli(d, True, "max")
        check("merge path runs with --pool-mean and --select-by max",
              r.returncode == 0 and j is not None,
              (r.stderr.strip().splitlines() or ["ok"])[-1][:90])
        if j:
            check("  and reports the max selection", "mean_dockq" in j,
                  f"mean_dockq {j['mean_dockq']:.4f} frac {j['frac_acc']:.4f}")
            check("  and reports the OTHER pooling from the same pass",
                  "mean_dockq_select_mean" in j,
                  f"{[k for k in j if 'select_' in k]}")

        # 2. selecting by the mean must pick different poses. Built into the table: by MAX
        #    aaaa picks sample 0 (DockQ 0.10) and bbbb picks 1 (0.90); by MEAN aaaa picks 1
        #    (0.80) and bbbb picks 0 (0.20).
        r2, j2 = run_cli(d, True, "mean")
        check("merge path runs with --select-by mean", r2.returncode == 0 and j2 is not None,
              (r2.stderr.strip().splitlines() or ["ok"])[-1][:90])
        if j and j2:
            check("--select-by mean selects DIFFERENT poses than max, with the EXPECTED values",
                  (j["mean_dockq"], j["frac_acc"]) == EXPECT["max"]
                  and (j2["mean_dockq"], j2["frac_acc"]) == EXPECT["mean"],
                  f"max {j['mean_dockq']:.4f}/{j['frac_acc']:.4f} (want "
                  f"{EXPECT['max'][0]:.4f}/{EXPECT['max'][1]:.4f})  mean "
                  f"{j2['mean_dockq']:.4f}/{j2['frac_acc']:.4f} (want "
                  f"{EXPECT['mean'][0]:.4f}/{EXPECT['mean'][1]:.4f})")
            check("the two runs agree on the OTHER pooling's number (same poses, both ways)",
                  abs(j2.get("mean_dockq_select_max", -1) - j["mean_dockq"]) < 1e-9
                  and abs(j.get("mean_dockq_select_mean", -1) - j2["mean_dockq"]) < 1e-9,
                  f"cross-check {j2.get('mean_dockq_select_max')} vs {j['mean_dockq']}")
            check("the chosen rule is recorded", j2.get("select_by") == "mean"
                  and j.get("select_by") == "max", f"{j.get('select_by')} / {j2.get('select_by')}")

        # 3. no mean column at all: max must still work, mean must REFUSE
        r3, j3 = run_cli(d, False, "max")
        check("merge path runs with NO iptm_mean column (the legacy format)",
              r3.returncode == 0 and j3 is not None,
              (r3.stderr.strip().splitlines() or ["ok"])[-1][:90])
        if j and j3:
            check("  and gives the same max metric as the --pool-mean pass",
                  j3["mean_dockq"] == j["mean_dockq"] and j3["frac_acc"] == j["frac_acc"],
                  f"{j3['mean_dockq']:.4f} vs {j['mean_dockq']:.4f}")
        r4, _ = run_cli(d, False, "mean")
        check("--select-by mean REFUSES without a mean column rather than falling back",
              r4.returncode != 0 and "needs an iptm_mean column" in (r4.stdout + r4.stderr),
              (( r4.stdout + r4.stderr).strip().splitlines() or ["(none)"])[-1][:90])


if __name__ == "__main__":
    main()
    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    sys.exit(1 if FAIL else 0)
