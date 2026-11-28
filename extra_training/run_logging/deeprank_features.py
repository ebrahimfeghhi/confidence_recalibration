#!/usr/bin/env python
"""DEEPRANK-AB FEATURES FOR EVERY POSE WE HAVE, saved instead of thrown away.

WHY THIS EXISTS. The scoring run kept only `predicted_dockq` per pose: score_training.invoke
builds the graphs inside a tempfile.TemporaryDirectory and the features die with it. The features
are the interesting part -- bsa, voro_area, vdw, orientation are the only four that move between
two poses of one seed -- so this runs the SAME pipeline and keeps `*_graph.hdf5` (plus the
predictions csv, which comes along for free).

NOTHING ABOUT DEEPRANK IS REIMPLEMENTED. The ensemble writer, the case enumeration and the chain
roles are imported from score_training.py; the graph build is driver.py exactly as the scoring run
invoked it. The only thing added here is pose plumbing: samples 0-4 are cifs on disk, while the
regenerated samples are coordinates in an npz and have to be substituted into a template cif
first, which is run_logging/dockq_poses_extra.substitute -- also imported, not retyped.

THE WORKSPACE IS DELETED, THE FEATURES ARE NOT. One 75-pose workspace measured 118 MB, almost all
of it the ensemble PDB and the split per-model copies; 1,600 of those would be 190 GB against a
disk quota with no grace. The graph hdf5 is ~67 kB per pose, so the kept output is ~8 GB for every
pose we own. Only directories this script created under --work-dir are removed.

POSE BATCHES ARE SEPARATE FILES ON PURPOSE. --samples picks which samples go into one ensemble, and
the output is named after the batch tag. Samples 105-109 are still being generated at the time of
writing, so doing 0-104 tonight and 105-109 later costs nothing extra -- the graph build is per
model, so splitting the ensemble duplicates no work. One graph.hdf5 per (pdb, antigen, batch).
"""
import argparse
import collections
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))

# THE CEILING IS ENFORCED, NOT DOCUMENTED, the same way dockq_poses_extra.py does it: jobs *
# graph_cores is what actually lands on the box, and a later run with a bigger flag gets clamped
# rather than honoured, because the box is shared.
MAX_CORES = 32


def parse_samples(spec):
    """'0-4,100-104' -> [0,1,2,3,4,100,101,102,103,104]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def orig_cifs(fold, pdb, seeds):
    """{(seed, sample): path} for the fold's own sample cifs, by the scoring run's glob."""
    found = collections.defaultdict(list)
    for p in glob.glob(str(Path(fold) / "poses" / pdb / "*" / "predictions" / f"{pdb}_sample_*.cif")):
        p = Path(p)
        try:
            seed = int(p.parents[1].name.split("-", 1)[0])
            sample = int(p.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if seed in seeds:
            found[(seed, sample)].append(p)
    return {k: sorted(v)[0] for k, v in found.items()}


def extra_sources(extra_dirs, pdb):
    """{(seed, sample_base + i): (npz path, i)} for every regenerated pose of this structure."""
    import numpy as np
    out = {}
    for d, base in extra_dirs:
        for p in sorted(glob.glob(os.path.join(d, pdb, "*.npz"))):
            unit = os.path.basename(p)[:-4]
            tail = unit.split("_")[-1]
            if not tail.isdigit():
                continue
            seed = int(tail)
            try:
                n = int(np.load(p, allow_pickle=True)["coords"].shape[0])
            except Exception:                                      # noqa: BLE001
                continue
            for i in range(n):
                out[(seed, base + i)] = (p, i)
    return out


def build_case(job):
    """One (pdb, antigen) ensemble: materialise poses, run driver.py, keep the graph."""
    (case, fold, extra_dirs, samples, base, out_dir, work_root, graph_cores, tag) = job
    import numpy as np
    sys.path.insert(0, str(base))
    import score_training as ST
    import dockq_poses_extra as DQE

    pdb, ag, h, l, nag, rows = case
    prefix = f"{pdb}_{ag}__{tag}"
    graph_out = Path(out_dir) / f"{prefix}_graph.hdf5"
    if graph_out.is_file():
        return prefix, 0, 0, 0.0, "cached"
    started = time.monotonic()
    seeds = {int(r["seed"]) for r in rows}
    want = set(samples)
    os.makedirs(work_root, exist_ok=True)
    log = Path(out_dir) / "logs" / f"{prefix}.log"
    os.makedirs(log.parent, exist_ok=True)
    try:
        have_orig = orig_cifs(fold, pdb, seeds)
        have_extra = extra_sources(extra_dirs, pdb) if any(s >= 100 for s in want) else {}
        paths, tmpdir = {}, tempfile.mkdtemp(prefix=f"{prefix}-", dir=work_root)
        try:
            # A template per seed, used only for topology: every coordinate is overwritten.
            tmpl = {}
            for (seed, sample), p in sorted(have_orig.items()):
                if sample == 0:
                    tmpl[seed] = p
            for seed in sorted(seeds):
                for s in samples:
                    if s < 100:
                        if (seed, s) in have_orig:
                            paths[(seed, s)] = have_orig[(seed, s)]
                        continue
                    src = have_extra.get((seed, s))
                    if src is None or seed not in tmpl:
                        continue
                    npz, idx = src
                    coords = np.load(npz, allow_pickle=True)["coords"][idx]
                    mcif = os.path.join(tmpdir, f"{pdb}_{seed}_{s}.cif")
                    DQE.substitute(tmpl[seed], coords, mcif)
                    paths[(seed, s)] = Path(mcif)
            if not paths:
                return prefix, 0, 0, time.monotonic() - started, "no-poses"
            work = Path(tmpdir) / "run"
            work.mkdir()
            pdb_input = Path(tmpdir) / f"{prefix}.pdb"
            ST.write_ensemble(pdb_input, paths, [x for x in (h, l, ag) if x])
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                   "WEIGHT_PATH": str(base / "weights/esm2_t33_650M_UR50D.pt"),
                   "REG_WEIGHT_PATH":
                       str(base / "weights/esm2_t33_650M_UR50D-contact-regression.pt"),
                   "DEEPRANK_GRAPH_CORES": str(graph_cores)}
            cmd = [str(base / ".venv/bin/python"), str(base / "driver.py"), str(pdb_input),
                   "--heavy_chain_id", h, "--light_chain_id", l or "-", "--antigen_chain_id", ag]
            with log.open("w") as fh:
                subprocess.run(cmd, cwd=work, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               check=True)
            got = list(work.glob("*/*_graph.hdf5"))
            if len(got) != 1:
                raise ValueError(f"expected one graph hdf5, found {len(got)}")
            os.makedirs(out_dir, exist_ok=True)
            part = graph_out.with_suffix(".hdf5.partial")
            shutil.copyfile(got[0], part)
            os.replace(part, graph_out)
            pred = list(work.glob("*/*_predictions.csv"))
            if pred:
                shutil.copyfile(pred[0], Path(out_dir) / f"{prefix}_predictions.csv")
            return (prefix, len(paths), len(seeds) * len(samples),
                    time.monotonic() - started, "ok")
        finally:
            # ONLY the directory this call created. mkdtemp means the name is ours.
            shutil.rmtree(tmpdir, ignore_errors=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        with log.open("a") as fh:
            fh.write(f"\n{type(e).__name__}: {e}\n")
        return prefix, 0, 0, time.monotonic() - started, "failed"


def enumerate_cases(roles_csv, results_csv):
    """(pdb, antigen, H, L, n_antigen, rows) -- score_training.training_cases, parameterised.

    The train and val folds differ only in two filenames, and the val variant in
    score_validation.py additionally asserts exactly 500 poses per pair, which is false for us:
    only 30 of the 100 val seeds were kept. So this takes the paths as arguments and asserts the
    ROLES rather than the pose count. Keyed on ref_chain and emitting model_chain, the way
    score_training.py:26 does -- the iface letters are REF names and the pose cifs use MODEL
    names, and they are permuted.
    """
    roles = collections.defaultdict(dict)
    with open(roles_csv) as fh:
        for r in csv.DictReader(fh):
            roles[r["pdb"]][r["ref_chain"]] = (r["role"].strip(), r["model_chain"])
    cases = collections.defaultdict(list)
    with open(results_csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") not in (None, "", "free"):
                continue
            if "has_trunk" in r and r.get("has_trunk") not in (None, "", "1"):
                continue
            pdb, names = r["iface"].split("_", 1)
            if any(c not in roles.get(pdb, {}) for c in names):
                continue
            tags = [roles[pdb][c] for c in names]
            ags = [m for role, m in tags if role.startswith("ag")]
            abs_ = [m for role, m in tags if role in ("H", "L")]
            if len(ags) != 1 or len(abs_) != 1:
                continue
            cases[(pdb, ags[0])].append(r)
    for (pdb, ag), rows in sorted(cases.items()):
        ab = {role: chain for role, chain in roles[pdb].values() if role in ("H", "L")}
        nag = sum(role.startswith("ag") for role, _ in roles[pdb].values())
        if "H" not in ab:
            continue
        yield (pdb, ag, ab["H"], ab.get("L"), nag, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--base", default=f"{ROOT}/training/runs/deeprank_ab_cpu")
    ap.add_argument("--roles", default="",
                    help="chain roles csv; defaults to <fold>/chain_roles_all.csv. The val fold "
                         "names it chain_roles_all83.csv, which is the only difference.")
    ap.add_argument("--results", default="",
                    help="results csv naming the (iface, seed) units; defaults to "
                         "<fold>/results_with_trunks.csv, the val fold uses results.csv.")
    ap.add_argument("--extra", action="append", default=[],
                    help="DIR:SAMPLE_BASE of a regenerated pose set, repeatable. The base is the "
                         "sample id of pose 0 in each npz, matching rediffuse_batch --sample-base.")
    ap.add_argument("--samples", default="0-4",
                    help="which samples go into this batch, e.g. '0-4,100-104'")
    ap.add_argument("--tag", default="", help="output name tag; defaults to the --samples spec")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/fold/deeprank_features")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--graph-cores", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    base = Path(a.base)
    sys.path.insert(0, str(base))
    import score_training as ST

    if a.jobs * a.graph_cores > MAX_CORES:
        jobs = max(1, MAX_CORES // max(1, a.graph_cores))
        print(f"  --jobs {a.jobs} x --graph-cores {a.graph_cores} = "
              f"{a.jobs * a.graph_cores} cores, over the {MAX_CORES} ceiling; jobs -> {jobs}")
        a.jobs = jobs

    samples = parse_samples(a.samples)
    tag = a.tag or a.samples.replace(",", "_")
    extra_dirs = []
    for spec in a.extra:
        d, _, b = spec.rpartition(":")
        extra_dirs.append((d if d else spec, int(b) if b.isdigit() else 100))
    work_root = a.work_dir or os.path.join(a.out_dir, "work")

    cases = list(enumerate_cases(a.roles or f"{a.fold}/chain_roles_all.csv",
                                 a.results or f"{a.fold}/results_with_trunks.csv"))
    if a.limit:
        cases = cases[:int(a.limit)]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"{len(cases)} antibody-antigen pair(s); samples {samples}; tag {tag}; "
          f"{a.jobs} job(s) x {a.graph_cores} graph core(s) = {a.jobs * a.graph_cores} cores",
          flush=True)
    print(f"extra pose sets: {extra_dirs or 'none'}", flush=True)
    print(f"out -> {a.out_dir}", flush=True)

    jobs = [(c, a.fold, extra_dirs, samples, base, a.out_dir, work_root, a.graph_cores, tag)
            for c in cases]
    t0 = time.time()
    done = collections.Counter()
    poses = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        fut = [pool.submit(build_case, j) for j in jobs]
        for i, f in enumerate(as_completed(fut), 1):
            name, n, want, elapsed, status = f.result()
            done[status] += 1
            poses += n
            if status != "cached" or i % 100 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] {name} {status}: {n}/{want} poses in {elapsed:.0f}s   "
                      f"{poses} pose(s) kept, ~{(len(jobs) - i) * el / i / 3600:.1f} h left",
                      flush=True)
    print(f"\n{dict(done)}   {poses} pose(s) featurised in {(time.time() - t0) / 3600:.2f} h")
    print(f"graphs -> {a.out_dir}")


if __name__ == "__main__":
    main()
