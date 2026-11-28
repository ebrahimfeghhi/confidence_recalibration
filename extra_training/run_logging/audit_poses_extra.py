#!/usr/bin/env python
"""IS THE REGENERATION PRODUCING WHAT WE ASKED FOR? A progress report that is also a check.

Counting files says a run is moving, not that it is useful. This asks, every time it runs:

  PROGRESS   units written, rate over the last window, ETA from the remaining units' own token
             counts rather than a flat average (cost goes as tokens^1.71, so the tail of large
             structures is slower than the mean suggests)
  SCORES     ipTM rows per unit, both poolings present, and units written but NOT scored
  CONTENT    on a random sample of finished units: the array is [nsample, n_atom, 3], every
             coordinate is finite, no two new poses are the same draw, the atom count matches
             the cached trunk's, and the new poses differ from the cached five
  COVERAGE   every scored unit has a row for every interface the pool lists for its structure
  FAILURES   what the run's own log reported

Exit code 1 if any check fails, so a watcher can say "stop and look" rather than printing a wall
of numbers nobody reads.
"""
import argparse
import collections
import csv
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/fold/poses_extra")
    ap.add_argument("--scores", default=f"{ROOT}/training/fold/poses_extra/results_extra.csv")
    ap.add_argument("--log", default="")
    ap.add_argument("--pool-csv", default=f"{ROOT}/training/combined/pool.csv")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--sample-units", type=int, default=12)
    ap.add_argument("--window-min", type=float, default=30.0)
    ap.add_argument("--rmsd-floor", type=float, default=0.5,
                    help="two poses closer than this (unsuperposed) are the same draw")
    a = ap.parse_args()

    import numpy as np

    bad = []
    expected = {}
    for p in sorted(glob.glob(f"{a.fold}/trunks/*/*.npz")):
        expected[(os.path.basename(os.path.dirname(p)), os.path.basename(p))] = p
    written = {}
    for p in sorted(glob.glob(f"{a.out_dir}/*/*.npz")):
        written[(os.path.basename(os.path.dirname(p)), os.path.basename(p))] = p
    tok = {}
    with open(a.indices) as fh:
        for r in csv.DictReader(fh):
            try:
                tok[r["pdb_id"]] = int(r["num_tokens"])
            except (KeyError, ValueError):
                pass
    # UNITS THE RUN CAN ACTUALLY DO: a structure absent from job_indices has no dataset item, so
    # it is skipped by design. Counting it as outstanding would make the ETA never converge.
    doable = {k: v for k, v in expected.items() if k[0] in tok}
    print(f"PROGRESS  {len(written)} written / {len(doable)} doable "
          f"({len(expected)} units in the trunk store, {len(expected) - len(doable)} from "
          f"structures with no dataset item)")

    now = time.time()
    recent = [p for p in written.values() if now - os.path.getmtime(p) < a.window_min * 60]
    rate = len(recent) / a.window_min * 60 if a.window_min else 0
    left = len(doable) - len(written)
    if rate > 0:
        # ETA BY THE REMAINING UNITS' OWN SIZE, not a flat rate: seconds ~ tokens^1.71.
        done_t = np.array([tok[k[0]] ** 1.71 for k in written if k[0] in tok]) if written else None
        left_t = np.array([tok[k[0]] ** 1.71 for k in doable if k not in written])
        if done_t is not None and len(done_t) and len(left_t):
            scale = left_t.mean() / done_t.mean()
        else:
            scale = 1.0
        print(f"          {rate:.0f} unit/h over the last {a.window_min:.0f} min, "
              f"{left} left, ETA {left / rate * scale:.1f} h "
              f"(size-weighted, remaining units are {scale:.2f}x the mean done)")
    else:
        print(f"          no unit written in the last {a.window_min:.0f} min, {left} left")
        if left:
            bad.append("no progress in the window")

    nrow = 0
    by_unit = collections.defaultdict(set)
    nomean = 0
    if os.path.exists(a.scores):
        with open(a.scores) as fh:
            rdr = csv.DictReader(fh)
            if rdr.fieldnames != ["pdb", "iface", "seed", "sample", "iptm", "iptm_mean",
                                  "diffusion_seed", "source"]:
                bad.append(f"scores header is {rdr.fieldnames}")
            for r in rdr:
                nrow += 1
                by_unit[(r["pdb"], f"free_{r['seed']}.npz")].add(r["iface"])
                if not (r.get("iptm_mean") or "").strip():
                    nomean += 1
    unscored = [k for k in written if k not in by_unit]
    print(f"SCORES    {nrow} row(s) over {len(by_unit)} unit(s); "
          f"{len(unscored)} written unit(s) unscored; {nomean} row(s) missing iptm_mean")
    if nomean:
        bad.append(f"{nomean} rows have no mean pooling")
    if len(unscored) > max(2, 0.02 * max(len(written), 1)):
        bad.append(f"{len(unscored)} written units are unscored")

    pool = collections.defaultdict(set)
    with open(a.pool_csv) as fh:
        for r in csv.DictReader(fh):
            pool[r["pdb"]].add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")
    short = [(k, sorted(pool[k[0]] - v)) for k, v in by_unit.items()
             if pool.get(k[0]) and (pool[k[0]] - v)]
    print(f"COVERAGE  {len(by_unit) - len(short)}/{len(by_unit)} scored unit(s) carry every "
          f"pool interface of their structure")
    if short:
        print(f"          e.g. {short[0][0]} missing {short[0][1][:3]}")
        if len(short) > 0.02 * max(len(by_unit), 1):
            bad.append(f"{len(short)} units are missing interfaces")

    rng = np.random.default_rng(int(now) // 1800)
    keys = list(written)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(a.sample_units, len(keys)),
                                        replace=False)] if keys else []
    nchk = problems = 0
    for k in pick:
        try:
            z = np.load(written[k], allow_pickle=True)
            c = z["coords"]
            t = np.load(expected[k], allow_pickle=True)["ci_coordinate"]
            nchk += 1
            if c.ndim != 3 or c.shape[0] != a.nsample or c.shape[2] != 3:
                problems += 1
                bad.append(f"{k}: coords shape {c.shape}")
                continue
            if not np.isfinite(c).all():
                problems += 1
                bad.append(f"{k}: non-finite coordinates")
            if c.shape[1] != t.shape[1]:
                problems += 1
                bad.append(f"{k}: {c.shape[1]} atoms vs the trunk's {t.shape[1]}")
                continue
            d = [float(np.sqrt(((c[i] - c[j]) ** 2).sum(-1).mean()))
                 for i in range(c.shape[0]) for j in range(i + 1, c.shape[0])]
            if min(d) < a.rmsd_floor:
                problems += 1
                bad.append(f"{k}: two new poses identical (min RMSD {min(d):.3f})")
            dn = [float(np.sqrt(((c[i] - t[j]) ** 2).sum(-1).mean()))
                  for i in range(c.shape[0]) for j in range(t.shape[0])]
            if min(dn) < a.rmsd_floor:
                problems += 1
                bad.append(f"{k}: a new pose reproduces a cached one (min RMSD {min(dn):.3f})")
        except Exception as e:                                # noqa: BLE001
            problems += 1
            bad.append(f"{k}: unreadable, {type(e).__name__}")
    print(f"CONTENT   {nchk} unit(s) checked, {problems} problem(s)")

    if a.log and os.path.exists(a.log):
        with open(a.log, errors="replace") as fh:
            lines = [ln.strip() for ln in fh
                     if ("Error" in ln or "error" in ln or "failed" in ln
                         or "not in the dataset" in ln) and "INFO" not in ln]
        print(f"FAILURES  {len(lines)} line(s) in the log" + (f"; last: {lines[-1][:110]}"
                                                              if lines else ""))

    if bad:
        print("\nCHECK FAILED:")
        for b in bad[:10]:
            print(f"  - {b}")
        sys.exit(1)
    print("\nall checks pass")


if __name__ == "__main__":
    main()
