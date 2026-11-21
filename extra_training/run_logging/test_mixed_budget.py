"""--mixed-budget must let the WHOLE training pool be scored without weakening any uniform pass.

WHAT THIS IS GUARDING. The training fold's depth is not uniform -- 459 of the 853 structures in
training/combined/pool.csv have 15 trunks and 394 have exactly one -- so a uniform `--budget 15`
over it stopped on 5,516 of 12,795 intended units ("A metric labelled 'budget 15' must cover 15
seeds for every structure"). That guard is right, and the fix must not be to remove it. So there
are two failures to rule out, and this file reproduces both:

  1. THE MIXED PASS NOT HAPPENING, or happening over a silently different candidate set.
  2. THE UNIFORM PASS LOSING ITS DEPTH ASSERTION -- the thing that would make every slice280 and
     val30 number in this repo unverifiable. Section 3 re-runs the real val manifest at budget 30
     and the known-bad mutations of it, so the old guard is proved still live, not assumed.

NO GPU ANYWHERE. Section 5 runs the aggregate half of the evaluator (--rows-in) over the REAL rows
of a completed budget-15 slice pass on disk, with half of its structures truncated to one seed. The
rows are real scores, the manifest is derived from them, and the metric is cross-checked against an
independent reimplementation of the selection in this file -- so the strata are verified against
something other than the code that produced them.

THE VACUITY TRAP, stated up front: a mixed pass whose pooled number equals the uniform one would
make every assertion here pass while proving nothing is at stake. Section 5 asserts the pooled
mixed number DIFFERS from the uniform number over the same interfaces, which is exactly why the
pooled number must not be compared against it.
"""
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

PY = sys.executable
EVAL = os.path.join(ROOT, "phases", "phase_k_eval.py")
MIX = os.path.join(ROOT, "training/fulltrain_mixed/seed_manifest.json")
MIX_POOL = os.path.join(ROOT, "training/fulltrain_mixed/pool.csv")
COMBINED = os.path.join(ROOT, "training/combined/pool.csv")
TRUNKS = os.path.join(ROOT, "training/fold/trunks")
RESULTS = os.path.join(ROOT, "training/fold/results.csv")
VAL_MAN = os.path.join(ROOT, "training/valfull83/seed_manifest.json")
SLICE_POOL = os.path.join(ROOT, "training/slice280/pool.csv")
# A COMPLETED REAL PASS, by name. Its rows are 280 interfaces x 15 seeds x 5 samples of genuine
# head scores, shipped ipTM and DockQ -- the only input to the aggregate half of the evaluator.
SLICE_ROWS = ("/tmp/claude-1004/-home-ebrahim-confidence-recalibration/"
              "d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/slice280_fs_ep20")
ACC = 0.23

FAIL = []
TMP = tempfile.mkdtemp(prefix="test_mixed_budget_")


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def raises(fn, needle, name):
    try:
        fn()
    except SystemExit as e:
        ok(needle in str(e), name, f"got: {str(e)[:90]}")
        return
    except Exception as e:                                  # noqa: BLE001
        ok(False, name, f"raised {type(e).__name__}, not SystemExit")
        return
    ok(False, name, "did not refuse at all")


class A:
    """phase_k_eval passes its argparse Namespace around as a config bag; this is that shape."""
    def __init__(self, **kw):
        self.seed_manifest, self.seed_base, self.budget = "", 500000, 15
        self.mixed_budget = False
        self.__dict__.update(kw)


def pool_ifaces(path):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        p = r["pdb"].strip()
        out[p].append(f"{p}_{r['chain1'].strip()}{r['chain2'].strip()}")
    return out


def write_manifest(tag, man):
    p = os.path.join(TMP, f"man_{tag}.json")
    with open(p, "w") as fh:
        json.dump(man, fh)
    return p


def pick(cand):
    """The evaluator's selection, reimplemented: max ipTM, ties to the lowest seed then sample."""
    return max(cand, key=lambda t: (t[0], -t[1], -t[2]))[3]


def main():
    import phase_k_eval as EV

    for p in (MIX, MIX_POOL, VAL_MAN, SLICE_POOL):
        if not os.path.exists(p):
            print(f"STOP: {p} is absent, so nothing here would be measured.")
            return 1

    def fresh(**kw):
        EV._SEED_MANIFEST.clear()
        EV._SEED_MANIFEST_META.clear()
        return A(**kw)

    man = json.load(open(MIX))
    st = man["structures"]
    depths = {p: len(v["seeds"]) for p, v in st.items()}

    print("\n1  THE BUILT MANIFEST DESCRIBES THE FOLD THAT IS ON DISK")
    ok(man.get("mixed") is True and man.get("k") is None,
       "it declares mixed:true and no k", f"algo {man.get('algo')}")
    pool = pool_ifaces(COMBINED)
    ok(set(st) == set(pool),
       "it names every structure of training/combined/pool.csv and no others",
       f"{len(st)} structure(s)")
    ok(len(set(depths.values())) > 1,
       "THE DEPTHS GENUINELY DIFFER (otherwise nothing here is at stake)",
       ", ".join(f"{c} at {d}" for d, c in
                 sorted(collections.Counter(depths.values()).items(), reverse=True)))
    miss = [f"{p}/free_{s}" for p, v in st.items() for s in v["seeds"]
            if not os.path.exists(os.path.join(TRUNKS, p, f"free_{s}.npz"))]
    ok(miss == [], "every seed it keeps has its trunk on disk",
       f"0 of {sum(depths.values())} missing" if not miss else str(miss[:3]))
    # AND EVERY KEPT SEED IS SCORED, which is the half a trunk check cannot see: 741 interfaces
    # carry DockQ for 10 seeds while only 1 trunk exists, so the two sources disagree in BOTH
    # directions and only their intersection is scoreable.
    scored = collections.defaultdict(set)
    for r in csv.DictReader(open(RESULTS)):
        try:
            scored[r["iface"]].add((int(r["seed"]), int(r["sample"])))
            float(r["dockq"])
        except (TypeError, ValueError):
            pass
    bad = [(i, s) for p, v in st.items() for i in pool[p] for s in v["seeds"]
           for k in range(5) if (s, k) not in scored.get(i, ())]
    ok(bad == [], "and every (interface, kept seed, sample) has a DockQ row",
       f"0 missing of {sum(len(pool[p]) * d for p, d in depths.items()) * 5}"
       if not bad else str(bad[:3]))
    deeper = [p for p, v in st.items() if len(scored[pool[p][0]]) // 5 > len(v["seeds"])]
    ok(len(deeper) > 0,
       "results.csv really is DEEPER than the trunks for some structures (so the intersection "
       "matters, and 'count the seeds in results.csv' would overclaim)",
       f"{len(deeper)} structure(s), e.g. {deeper[0]}: "
       f"{len(scored[pool[deeper[0]][0]]) // 5} scored seed(s) vs {depths[deeper[0]]} trunk(s)")

    print("\n2  THE FLAG AND THE MANIFEST ARE LOCKED TO EACH OTHER")
    maxd = max(depths.values())
    p15 = next(p for p, d in depths.items() if d == maxd)
    p1 = next(p for p, d in depths.items() if d < maxd)
    raises(lambda: EV.intended_seeds(fresh(seed_manifest=MIX, budget=maxd), p15),
           "--mixed-budget was not passed",
           "a mixed manifest WITHOUT --mixed-budget is fatal (the in-training eval auto-forwards "
           "--seed-manifest, so this is the path that must not open by itself)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=VAL_MAN, budget=30, seed_base=400000, mixed_budget=True), "22gi"),
        "does not declare mixed:true",
        "--mixed-budget on a UNIFORM manifest is fatal (the flag must never disable an assertion "
        "the pass satisfies)")
    a = fresh(seed_manifest=MIX, budget=maxd, mixed_budget=True)
    got = EV.intended_seeds(a, p1)
    ok(got == tuple(st[p1]["seeds"]) and len(got) < maxd,
       "with both, a shallow structure returns ITS OWN depth, not the budget",
       f"{p1}: {len(got)} seed(s) against --budget {maxd}")
    ok(EV.intended_seeds(a, p15) == tuple(st[p15]["seeds"]),
       "and a deep structure returns all of its seeds", f"{p15}: {maxd} seed(s)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=MIX, budget=maxd + 1, mixed_budget=True), p15),
        "deepest structure keeps",
        "--budget above the deepest structure is fatal (the label would overclaim the candidate "
        "set it is printed beside)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=MIX, budget=maxd - 1, mixed_budget=True), p15),
        "deepest structure keeps",
        "--budget below the deepest structure is fatal too")
    raises(lambda: EV.intended_seeds(fresh(seed_manifest=write_manifest(
        "withk", {**man, "k": maxd}), budget=maxd, mixed_budget=True), p15),
        "declares both mixed:true and k",
        "mixed:true AND a k is fatal (k is the claim mixed withdraws)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=MIX, budget=maxd, mixed_budget=True), "zzzz"),
        "does not name structure",
        "a structure the mixed manifest omits is fatal (it cannot be scored at an inferred depth)")
    _ragged = {k: v for k, v in man.items() if k != "mixed"}
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=write_manifest("ragged", _ragged), budget=maxd), p1),
        "but --budget is",
        "a ragged manifest that forgot to declare mixed:true still hits the ORIGINAL depth guard")

    print("\n3  THE UNIFORM PATH IS UNTOUCHED -- the real val manifest, at its real budget")
    av = fresh(seed_manifest=VAL_MAN, budget=30, seed_base=400000)
    ok(len(EV.intended_seeds(av, "22gi")) == 30, "valfull83 still resolves at budget 30")
    raises(lambda: EV.intended_seeds(fresh(seed_manifest=VAL_MAN, budget=29, seed_base=400000),
                                     "22gi"),
           "declares k=30", "and still refuses a budget its own k contradicts")
    _short = json.load(open(VAL_MAN))
    _short["k"] = None
    _short["structures"]["22gi"]["seeds"] = _short["structures"]["22gi"]["seeds"][:-1]
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=write_manifest("valshort", _short), budget=30, seed_base=400000),
        "22gi"),
        "but --budget is 30",
        "a uniform manifest keeping 29 of 30 is STILL fatal without the flag")
    ok(EV.intended_seeds(fresh(budget=15), "anything") == tuple(range(500000, 500015)),
       "and no manifest at all is still the contiguous window")

    print("\n4  MIXED DEPTH IS NOT INFERRED FROM THE ROWS, so short data cannot pass as shallow")
    # The same per-structure key set the uniform path asserts: this is the claim that makes the
    # mixed number trustworthy, so it is checked explicitly rather than left to section 5.
    a = fresh(seed_manifest=MIX, budget=maxd, mixed_budget=True)
    want = {(i, s, k) for p in (p15, p1) for i in pool[p]
            for s in EV.intended_seeds(a, p) for k in range(5)}
    ok(len(want) == (len(pool[p15]) * maxd + len(pool[p1]) * depths[p1]) * 5,
       "the intended key set is interfaces x THEIR OWN seeds x samples",
       f"{len(want)} key(s) over 2 structures")

    print("\n5  END TO END ON REAL ROWS, NO GPU: a budget-15 pass read back as a mixed one")
    if not os.path.isdir(SLICE_ROWS):
        ok(False, "a completed slice280 pass is on disk to read back", SLICE_ROWS)
        return 1 if FAIL else 0
    rows = []
    for f in sorted(os.listdir(SLICE_ROWS)):
        if f.startswith("rows_") and f.endswith(".csv"):
            rows += list(csv.DictReader(open(os.path.join(SLICE_ROWS, f))))
    sp = pool_ifaces(SLICE_POOL)
    ok(len(rows) == sum(len(v) for v in sp.values()) * 15 * 5,
       "the pass is complete: 280 interfaces x 15 seeds x 5 samples", f"{len(rows)} row(s)")
    # HALF THE STRUCTURES ARE TRUNCATED TO ONE SEED, which is the shape of the real training pool
    # (459 structures at 15, 394 at 1) with the proportions it really has.
    pdbs = sorted(sp)
    shallow = set(pdbs[1::2])
    mixman = {"mixed": True, "k": None, "algo": "test-truncation", "nsample": 5,
              "seed_range": [500000, 500014],
              "structures": {p: {"seeds": [500000] if p in shallow
                                 else list(range(500000, 500015))} for p in pdbs}}
    mixp = write_manifest("slice_mixed", mixman)
    keep = [r for r in rows
            if not (r["iface"].rsplit("_", 1)[0] in shallow and int(r["seed"]) != 500000)]
    work = os.path.join(TMP, "mixedwork")
    os.makedirs(work, exist_ok=True)
    paths = []
    for i in range(3):
        p = os.path.join(work, f"rows_mixed_s{i}.csv")
        with open(p, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in keep[i::3]:
                w.writerow(r)
        json.dump({"shard": i, "nshards": 3, "units": len(keep[i::3]), "rows": len(keep[i::3]),
                   "n_ost": 0, "n_unresolvable": 0, "n_differ": 0, "differ_pdbs": {},
                   "n_unmapped": 0, "unmapped": [], "budget": 15, "ckpt": "test-head"},
                  open(p + ".meta.json", "w"))
        paths.append(p)
    base = [PY, "-I", EVAL, "--rows-in", ",".join(paths), "--budget", "15",
            "--seed-base", "500000", "--pool", SLICE_POOL, "--results", RESULTS,
            "--nsample", "5", "--seed-manifest", mixp]
    outj = os.path.join(TMP, "mixed_metric.json")
    r = subprocess.run(base + ["--mixed-budget", "--out", outj],
                       capture_output=True, text=True, env={**os.environ,
                                                            "LAYERNORM_TYPE": "openfold"})
    ok(r.returncode == 0, "the mixed merge runs to a metric",
       (r.stdout + r.stderr).strip().splitlines()[-1][:110] if r.returncode else "exit 0")
    if r.returncode:
        print((r.stdout + r.stderr)[-2000:])
        return 1
    got = json.load(open(outj))
    ok(got.get("budget_mode") == "mixed-per-structure" and got.get("budget_max") == 15,
       "the json says it is a mixed-budget number and what the deepest structure was",
       f"depth_hist {got.get('depth_hist')}")
    ok(sorted(got.get("strata", {})) == ["depth_1", "depth_15"],
       "and carries one stratum per depth", str(sorted(got.get("strata", {}))))

    # INDEPENDENT RECOMPUTATION, from the same rows, in this file.
    by = collections.defaultdict(list)
    bs = collections.defaultdict(list)
    for rr in keep:
        by[rr["iface"]].append((float(rr["iptm"]), int(rr["seed"]), int(rr["sample"]),
                                float(rr["dockq"])))
        bs[rr["iface"]].append((float(rr["stored_iptm"]), int(rr["seed"]), int(rr["sample"]),
                                float(rr["dockq"])))
    def stat(ifs, src):
        sel = [pick(src[i]) for i in ifs]
        return (round(sum(sel) / len(sel), 4),
                round(sum(1 for d in sel if d >= ACC) / len(sel), 4), len(sel))
    deep_ifs = [i for i in by if i.rsplit("_", 1)[0] not in shallow]
    shal_ifs = [i for i in by if i.rsplit("_", 1)[0] in shallow]
    for ifs, key in ((deep_ifs, "depth_15"), (shal_ifs, "depth_1")):
        q, t1, n = stat(ifs, by)
        bq, bt1, _ = stat(ifs, bs)
        s = got["strata"][key]
        ok((s["n_interfaces"], s["mean_dockq"], s["frac_acc"]) == (n, q, t1),
           f"{key}: the stratum reproduces an independent selection over those rows",
           f"n={n} q={q} top1={t1}")
        ok((s.get("baseline_mean_dockq"), s.get("baseline_frac_acc")) == (bq, bt1),
           f"{key}: its baseline is the SHIPPED column of the SAME rows",
           f"q={bq} top1={bt1}")
        ok(s["candidates_per_interface"] == (75 if key == "depth_15" else 5),
           f"{key}: the candidate count is stated, not implied",
           str(s["candidates_per_interface"]))
    pq, pt1, pn = stat(list(by), by)
    ok((got["n_interfaces"], got["mean_dockq"], got["frac_acc"]) == (pn, pq, pt1),
       "the pooled number is the pooled selection over all of them", f"n={pn} q={pq} top1={pt1}")
    # THE POOLED NUMBER IS A DIFFERENT MEASUREMENT, demonstrated rather than asserted in prose.
    full_q, full_t1, _ = stat(list(by), {i: [t for t in by[i]] for i in by})
    deep_q, deep_t1, _ = stat(deep_ifs, by)
    ok(abs(pt1 - deep_t1) > 1e-9 or abs(pq - deep_q) > 1e-9,
       "and it DIFFERS from the deep stratum over the same pool -- which is why it must not be "
       "compared against a uniform-budget number",
       f"pooled {pq}/{pt1} vs depth-15 {deep_q}/{deep_t1}")
    _uni = os.path.join(os.path.dirname(SLICE_ROWS), "slice280_fs_ep20.json")
    if os.path.exists(_uni):
        _u = json.load(open(_uni))
        ok(_u["mean_dockq"] != pq,
           "and from the uniform budget-15 metric over the very same 280 interfaces",
           f"uniform {_u['mean_dockq']}/{_u['frac_acc']} vs mixed-pooled {pq}/{pt1}")

    # ---- AND UNDER --select-by mean, which SWAPS byif and byif_m before the strata are built.
    # THE RECURRING BUG CLASS in this file's history: the mean column has twice been handled on one
    # of the two paths only. A stratum built after the swap must report the MEAN selection as
    # "ours" while its oracle (same candidates) and its baseline (results.csv's stored ipTM) stay
    # exactly where they were.
    meanwork = os.path.join(TMP, "meanwork")
    os.makedirs(meanwork, exist_ok=True)
    mpaths = []
    for i, src in enumerate(paths):
        rs = list(csv.DictReader(open(src)))
        dst = os.path.join(meanwork, os.path.basename(src))
        with open(dst, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()) + ["iptm_mean"])
            w.writeheader()
            for rr in rs:
                # A DIFFERENT RANKING, not a rescaling of the max: a selector reading the wrong
                # column cannot then produce the same number by accident.
                h = int(hashlib.sha1(f"{rr['iface']}{rr['seed']}{rr['sample']}".encode())
                        .hexdigest()[:6], 16)
                rr["iptm_mean"] = f"{float(rr['iptm']) * 0.7 + (h % 1000) / 3000.0:.6f}"
                w.writerow(rr)
        json.dump(json.load(open(src + ".meta.json")), open(dst + ".meta.json", "w"))
        mpaths.append(dst)
    outm = os.path.join(TMP, "mixed_mean_metric.json")
    r4 = subprocess.run([PY, "-I", EVAL, "--rows-in", ",".join(mpaths), "--budget", "15",
                         "--seed-base", "500000", "--pool", SLICE_POOL, "--results", RESULTS,
                         "--nsample", "5", "--seed-manifest", mixp, "--mixed-budget",
                         "--select-by", "mean", "--out", outm],
                        capture_output=True, text=True,
                        env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    ok(r4.returncode == 0, "--select-by mean runs on the same mixed rows",
       (r4.stdout + r4.stderr).strip().splitlines()[-1][:100] if r4.returncode else "exit 0")
    if r4.returncode == 0:
        gm = json.load(open(outm))
        bym = collections.defaultdict(list)
        for rr in keep:
            h = int(hashlib.sha1(f"{rr['iface']}{rr['seed']}{rr['sample']}".encode())
                    .hexdigest()[:6], 16)
            bym[rr["iface"]].append((float(rr["iptm"]) * 0.7 + (h % 1000) / 3000.0,
                                     int(rr["seed"]), int(rr["sample"]), float(rr["dockq"])))
        for ifs, key in ((deep_ifs, "depth_15"), (shal_ifs, "depth_1")):
            q, t1, n = stat(ifs, bym)
            bq, bt1, _ = stat(ifs, bs)
            sm = gm["strata"][key]
            ok((sm["mean_dockq"], sm["frac_acc"]) == (q, t1),
               f"{key}: under --select-by mean the stratum selects by the MEAN column",
               f"q={q} top1={t1}")
            ok((sm.get("baseline_mean_dockq"), sm.get("baseline_frac_acc")) == (bq, bt1),
               f"{key}: and its baseline is STILL the shipped stored ipTM, unswapped",
               f"q={bq} top1={bt1}")
            ok(sm["oracle_mean_dockq"] == got["strata"][key]["oracle_mean_dockq"],
               f"{key}: and the oracle is unchanged -- the candidates did not move")

    print("\n6  THE GUARDS STILL FIRE ON THE SAME ROWS")
    r2 = subprocess.run(base, capture_output=True, text=True,
                        env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    ok(r2.returncode != 0 and "--mixed-budget was not passed" in (r2.stdout + r2.stderr),
       "the identical rows WITHOUT --mixed-budget are refused, not quietly averaged",
       (r2.stdout + r2.stderr).strip().splitlines()[-1][:90])
    # A ROW OUTSIDE THE INTENDED SET IS STILL FATAL. This is the check that makes the mixed
    # number mean something: the shallow structures' seeds 500001-500014 exist in the rows file
    # of the original pass, and a mixed pass must refuse them rather than quietly score 75
    # candidates for an interface its manifest says has 5.
    work2 = os.path.join(TMP, "mixedwork_extra")
    os.makedirs(work2, exist_ok=True)
    paths2 = []
    for i in range(3):
        p = os.path.join(work2, f"rows_extra_s{i}.csv")
        with open(p, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for rr in rows[i::3]:                      # the UNTRUNCATED rows
                w.writerow(rr)
        json.dump({"shard": i, "nshards": 3, "units": 0, "rows": 0, "n_ost": 0,
                   "n_unresolvable": 0, "n_differ": 0, "differ_pdbs": {}, "n_unmapped": 0,
                   "unmapped": [], "budget": 15, "ckpt": "test-head"},
                  open(p + ".meta.json", "w"))
        paths2.append(p)
    r3 = subprocess.run([PY, "-I", EVAL, "--rows-in", ",".join(paths2), "--budget", "15",
                         "--seed-base", "500000", "--pool", SLICE_POOL, "--results", RESULTS,
                         "--nsample", "5", "--seed-manifest", mixp, "--mixed-budget"],
                        capture_output=True, text=True,
                        env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    ok(r3.returncode != 0 and "lie OUTSIDE the intended set" in (r3.stdout + r3.stderr),
       "rows deeper than the manifest's depth are refused under --mixed-budget",
       (r3.stdout + r3.stderr).strip().splitlines()[-1][:90])

    print("\n7  WIRING: the shell path and the report can both run a whole-pool pass")
    sh = open(os.path.join(ROOT, "shell_scripts/score_checkpoint.sh")).read()
    ok("SLICE_MANIFEST" in sh and "--mixed-budget" in sh,
       "score_checkpoint.sh takes SLICE_MANIFEST and derives --mixed-budget from the file")
    ok('tail -3 "$SD/${SLICE_NAME}_$TAG.log"' in sh,
       "its ABORT handler tails the log the pass actually wrote (not slice280's)")
    ok('--slice-work "$SD/${SLICE_NAME}_$TAG"' in sh,
       "and the report reads the work dir the pass actually wrote")
    ok("POSE_FLAGS=$($PY -I run_logging/pathway_flags.py" in sh,
       "and the pathway flags are read off the run's args.json rather than retyped")
    sr = open(os.path.join(ROOT, "run_logging/score_report.py")).read()
    ok("--slice-depths" in sr and "d{d} ({len(_ifs)}, in-sample)" in sr,
       "score_report.py splits the in-sample table by depth instead of pooling it")
    ok('slice_name = ("TRAIN SLICE (280, in-sample)"' in sr,
       "and only the real 280-item slice is still labelled TRAIN SLICE, which is what "
       "plot_fullset.report_points greps for")
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "run_logging"))
    pf = importlib.import_module("pathway_flags")
    _fl, _notes = pf.flags_for(os.path.join(ROOT, "training/runs/w57hjo/head_ep20_step2900.pt"))
    ok("--pose-readout" in _fl and "--pose-readout-prenorm" in _fl,
       "and that derivation recovers the flags Arm B's failed pass was missing", " ".join(_fl))
    ok(all(f in pf.eval_flags() for f in _fl if f.startswith("--")),
       "every flag it emits is one phase_k_eval really defines")

    print("\n8  THE REVIEW'S FINDINGS, EACH REPRODUCED AS A CHECK")
    # Every one of these was a real defect found by review of this diff, so each gets a test that
    # fails if it comes back -- not a comment saying it was fixed.

    # 8.1 --mixed-budget with NO manifest returned the contiguous window and still wrote
    # budget_mode/strata, because intended_seeds returns before it reads any manifest.
    r5 = subprocess.run([PY, "-I", EVAL, "--rows-in", ",".join(paths), "--budget", "15",
                         "--seed-base", "500000", "--pool", SLICE_POOL, "--results", RESULTS,
                         "--nsample", "5", "--mixed-budget"],
                        capture_output=True, text=True,
                        env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    ok(r5.returncode != 0 and "--mixed-budget needs --seed-manifest" in (r5.stdout + r5.stderr),
       "8.1 --mixed-budget without a manifest refuses (it used to label a UNIFORM pass mixed)",
       (r5.stdout + r5.stderr).strip().splitlines()[-1][:80])

    # 8.2 the per-stratum baseline was gated on interface membership only, so it could be
    # computed over fewer POSES than its own 'ours' number while the pooled baseline refused.
    holework = os.path.join(TMP, "holework")
    os.makedirs(holework, exist_ok=True)
    hpaths, _blanked = [], set()
    for i, src in enumerate(paths):
        rs = list(csv.DictReader(open(src)))
        dst = os.path.join(holework, os.path.basename(src))
        with open(dst, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            for rr in rs:
                # ONE POSE PER INTERFACE, so every interface still HAS a baseline and only the
                # candidate set is narrowed -- the exact shape that slipped through.
                if rr["iface"] not in _blanked:
                    _blanked.add(rr["iface"])
                    rr = {**rr, "stored_iptm": ""}
                w.writerow(rr)
        json.dump(json.load(open(src + ".meta.json")), open(dst + ".meta.json", "w"))
        hpaths.append(dst)
    outh = os.path.join(TMP, "holed_metric.json")
    r6 = subprocess.run([PY, "-I", EVAL, "--rows-in", ",".join(hpaths), "--budget", "15",
                         "--seed-base", "500000", "--pool", SLICE_POOL, "--results", RESULTS,
                         "--nsample", "5", "--seed-manifest", mixp, "--mixed-budget",
                         "--out", outh],
                        capture_output=True, text=True,
                        env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    ok(r6.returncode == 0 and "baseline to beat: UNAVAILABLE" in r6.stdout,
       "8.2 a pass missing some stored ipTM still reports, with the pooled baseline UNAVAILABLE",
       f"exit {r6.returncode}")
    if r6.returncode == 0:
        gh = json.load(open(outh))
        ok("baseline_mean_dockq" not in gh,
           "8.2 and the pooled json carries no baseline")
        ok(all("baseline_mean_dockq" not in v for v in gh["strata"].values()),
           "8.2 AND NO STRATUM CARRIES ONE EITHER -- a per-depth ship column over a narrower "
           "candidate set than its own ours column was printed as the shipped reference",
           str(sorted(gh["strata"])))
        ok("AND THERE IS NO BASELINE COLUMN" in r6.stdout,
           "8.2 and the trailer stops telling the reader to compare against a column that is "
           "not there")

    # 8.3 score_report treated an OMITTED --slice-pool as "this is slice280", so a mixed or
    # bigger pool printed TRAIN SLICE and was read into the slice panel by plot_fullset.
    r7 = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging/score_report.py"),
                         "--slice-work", work, "--expect-slice", "0",
                         "--expect-cands", "val=150"],
                        capture_output=True, text=True, cwd=ROOT)
    ok(r7.returncode != 0 and "--slice-pool is required here" in (r7.stdout + r7.stderr),
       "8.3 score_report refuses to label a non-default pool TRAIN SLICE",
       (r7.stdout + r7.stderr).strip().splitlines()[-1][:80])

    # 8.4 pathway_flags returned 0 on every failure, so the caller's `|| exit 4` was unreachable
    # and a corrupt args.json silently yielded no flags.
    _br = os.path.join(TMP, "badrun")
    os.makedirs(_br, exist_ok=True)
    open(os.path.join(_br, "args.json"), "w").write("{not json")
    r8 = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging/pathway_flags.py"),
                         os.path.join(_br, "head_ep1_step1.pt")], capture_output=True, text=True)
    ok(r8.returncode == 2, "8.4 an unreadable args.json exits non-zero", f"exit {r8.returncode}")
    r9 = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging/pathway_flags.py"),
                         os.path.join(TMP, "no_run_dir", "head_ep1_step1.pt")],
                        capture_output=True, text=True)
    ok(r9.returncode == 0 and r9.stdout.strip() == "",
       "8.4 while an ABSENT args.json is not fatal -- a checkpoint with no run dir still scores",
       f"exit {r9.returncode}")

    # 8.5 the mixed probe lived inside the SIDES branch while the report always runs, so
    # SIDES=val as a separate invocation reached score_report with slice=75 expectations.
    _i_man = sh.index('if [ -n "$SLICE_MANIFEST" ]')
    _i_sid = sh.index('if [ "$SIDES" = both ] || [ "$SIDES" = slice ]')
    ok(_i_man < _i_sid,
       "8.5 the manifest probe is resolved BEFORE the SIDES branch, so a val-only invocation "
       "still reports the mixed pass correctly")
    ok('[ -f "$SLICE_POOL" ] ||' in sh,
       "8.5 and a missing SLICE_POOL is refused before three cards are spent")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
