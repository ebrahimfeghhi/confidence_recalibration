#!/usr/bin/env python
"""Build the reference-free 50-seed experiment end to end: pool -> templates, ready for the GPU.

SUMMARY
    One command for steps 1-7. It stops before folding, so the only thing it spends is CPU.

    WHY THIS EXISTS. The pool on disk is not the pool the experiment is supposed to use. It holds 95
    seed ids (300000-300094) because individual complexes were sampled deeper in earlier work, and
    some seeds have several fold directories from a precision side-experiment. Nothing in the seven
    steps ever capped either one, so the last run drew 302 of its 3,132 donors from seeds past 300049
    and the name "reffree_50" described an assumption no code checked.

    THE RULE, hardcoded here and nowhere else:

      seeds 300000-300049, and the legacy-no-provenance fold directory only.

    Verified against disk when this was written: all 123 complexes have all 50 seeds with 5 poses
    each under that run id, so the rule drops nothing it should keep -- 6,150 folds, 30,750 poses.

    WHAT IT RUNS, in order. Each step reads the previous step's output from <out>:

      1 anchor_relabel      name every pose's chains after one chosen pose
      2 pose_interfaces     which chain pairs are interfaces, by contact in the poses
      3 fetch_spurious      pair_iptm for interfaces the predictions declare and the reference lacks
      4 pose_to_pose_dockq  N x N symmetric pose-to-pose DockQ per interface
      5 diversity_filter    sort by ipTM, keep a pose only if unlike everything kept, cap 50
      6 build_templates     each donor pose -> a template Protenix can load
      7 preflight           ids, chains and a3m hits, before any GPU time

    RESUME. A step is skipped when its output is already present, so a re-run picks up where it
    stopped; --force <step> recomputes from that step onward. Step 4 is the expensive one.

    THE DIFF. Because donors are now drawn from a smaller pool, the selection no longer matches the
    previous run. --diff-against reports, by fold identity (donor pose, seed, pinned chains):

      carry over    folds that exist and are still selected -- reuse, do not refold
      omit          folds that exist but are no longer selected, i.e. the out-of-range donors
      to fold       folds newly selected that were never run

INPUTS
    --out:           experiment directory to build. Required.
    --diff-against:  an earlier experiment to compare fold-for-fold. Default experimental_data/reffree_50.
    --cap:           donors per complex. Default 50.
    --seed:          the seed every templated fold uses. Default 430000.
    --jobs:          worker processes for the parallel steps. Default 16.
    --force:         recompute from this step onward: pool, anchor, interfaces, spurious, dockq,
                     filter, templates, preflight. Default off, i.e. resume.
    --stop-after:    run no further than this step. Optional, for inspecting an intermediate.

OUTPUTS
    <out>/pool_50.csv          the filtered pool -- the rule, made explicit as a file
    <out>/anchor/              chain maps and roles in the anchor frame
    <out>/pose_interfaces.csv  the interfaces the poses declare
    <out>/spurious/            pair_iptm for the predicted-only interfaces
    <out>/matrices/            per-interface npz, built from the filtered pool only
    <out>/queue.tsv            the folds to run
    <out>/ns/                  the template namespace
    <out>/diff_vs_<name>.md    carry over / omit / to fold, as a table
    <out>/build.log            every step's stdout

FUNCTIONS
    filtered_pool():  write pool_50.csv, keeping only the hardcoded seeds and run directory
    run():            one step as a subprocess, logged, skipped when its output exists
    fold_id():        (donor pose, seed, pinned chains) -- what decides whether a fold can be reused
    diff():           new selection vs an earlier experiment, by fold identity
    main():           the seven steps in order, then the diff
"""
import argparse
import collections
import itertools
import csv
import os
import re
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from fmt import table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Two interpreters, because no single environment holds every dependency: anchor_relabel.py needs
# pxmeter's chain permutation, which only the pxmeter venv has, and everything else needs gemmi and
# torch, which only the Protenix venv has. Launching every step with sys.executable is what made
# step 2 die with ModuleNotFoundError. Steps name their interpreter rather than inheriting one.
PXM_PY = str(paths.PXMETER_PYTHON)
PY = str(paths.PROTENIX_PYTHON)
# Step 5's DockQ is exact all-pairs DockQ on GPU, not a CPU fallback, so it needs a device. Kept off
# the GPUs the folding baseline is using; override when those change.
DOCKQ_GPUS = [g for g in os.environ.get("DOCKQ_GPUS", "0,2,3").split(",") if g]
# The extra GPUs are borrowed above the standing 4-card budget, so they are time-limited.
DOCKQ_DEADLINE_S = int(os.environ.get("DOCKQ_DEADLINE_S", "1800"))

# the rule. seeds 300000-300049 and the legacy fold directory, nothing else.
POOL_SEEDS = {str(300000 + i) for i in range(50)}
RUN_DIR = paths.LEGACY_RUN

STEPS = ["pool", "anchor", "interfaces", "spurious", "dockq", "filter", "templates", "preflight"]


def filtered_pool(out, log):
    """in: the experiment directory.  out: the path to the filtered pool csv

    Keeps a row only if its seed is in range AND its pose resolves under the legacy run directory.
    The second test is what makes "legacy provenance" real rather than assumed: a row whose pose only
    exists in some other fold directory is dropped, so scores and coordinates cannot disagree.
    """
    dst = f"{out}/pool_50.csv"
    kept = dropped_seed = dropped_run = 0
    seeds = collections.defaultdict(set)
    poses = collections.defaultdict(set)
    log(f"  [pool] filtering {paths.PXM2025_POOL_CSV}")
    log(f"         rule: seeds {min(POOL_SEEDS)}-{max(POOL_SEEDS)} ({len(POOL_SEEDS)} ids), "
        f"run dir {RUN_DIR}")
    with open(str(paths.PXM2025_POOL_CSV)) as fh, open(dst, "w", newline="") as o:
        r = csv.DictReader(fh)
        w = csv.DictWriter(o, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if row["seed"] not in POOL_SEEDS:
                dropped_seed += 1
                continue
            pose = (f"{paths.FREE_DIR}/{row['pdb']}/{row['seed']}-{RUN_DIR}"
                    f"/predictions/{row['pdb']}_sample_{row['sample']}.cif")
            if not os.path.exists(pose):
                dropped_run += 1
                continue
            w.writerow(row)
            kept += 1
            seeds[row["pdb"]].add(row["seed"])
            poses[row["pdb"]].add((row["seed"], row["sample"]))
    log(f"         wrote {dst}")
    log(f"         kept {kept} rows; dropped {dropped_seed} out-of-range seeds, "
        f"{dropped_run} with no {RUN_DIR} pose")

    # the verification. every complex must carry all 50 seeds, or the rule is silently sampling less
    # than it claims and every downstream count inherits that
    bad = {p: sorted(v, key=int) for p, v in seeds.items() if len(v) != len(POOL_SEEDS)}
    nsamp = collections.Counter(len(v) for v in poses.values())
    log(f"  [pool] verification")
    log(f"         complexes: {len(seeds)}")
    log(f"         seeds per complex: all exactly {len(POOL_SEEDS)}" if not bad
        else f"         seeds per complex: {len(bad)} COMPLEXES OFF")
    log(f"         poses per complex: " + ", ".join(f"{k} poses x{v}" for k, v in
                                                    sorted(nsamp.items(), reverse=True)))
    log(f"         total poses: {sum(len(v) for v in poses.values())}")
    if bad:
        for p, v in sorted(bad.items())[:10]:
            miss = sorted(POOL_SEEDS - set(v), key=int)
            log(f"         ! {p}: {len(v)} seeds, missing {miss[:6]}")
        raise SystemExit(f"[pool] {len(bad)} complexes do not carry all {len(POOL_SEEDS)} seeds")
    return dst


def run(cmd, outputs, force, log, name, env=None):
    """in: the command, the paths it produces, whether to force, a logger, the step name, env extras.
    out: whether it ran
    """
    # A step that created its output and then died must not count as done. Two ways that happens: an
    # empty output DIRECTORY, and -- the one that actually bit -- a csv holding only its header row,
    # which anchor_relabel.py writes before the work starts. Treating that as present skipped step 2
    # and fed an empty chain map to every step after it, with no error anywhere.
    def there(p):
        if os.path.isdir(p):
            return bool(os.listdir(p))
        if not os.path.exists(p):
            return False
        if str(p).endswith(".csv"):
            with open(p) as fh:
                return sum(1 for _ in itertools.islice(fh, 2)) > 1
        return True

    if not force and outputs and all(there(p) for p in outputs):
        log(f"  [{name}] present, skipping")
        return False
    log(f"  [{name}] running {os.path.basename(str(cmd[1]))}")
    log(f"         python {cmd[0]}")
    if env:
        log(f"         env {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}")
    for p in outputs:
        log(f"         -> {p}")
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    for line in (p.stdout or "").splitlines():
        log(f"    {line}")
    if p.returncode != 0:
        for line in (p.stderr or "").splitlines()[-25:]:
            log(f"    ! {line}")
        raise SystemExit(f"[{name}] failed with exit {p.returncode} -- see the log")
    return True


def norm_chains(chains):
    """in: chains as a list, a plus-joined string, or None.  out: a canonical sorted string

    The two records of the same thing are spelled differently -- chosen.json writes ["A","B","D"] and
    results.csv writes "A+B+C" -- so both are reduced to sorted ids before they are compared. Without
    this every fold looks unlike every other and the diff reports nothing can be reused.
    """
    if not chains:
        return ""
    if isinstance(chains, str):
        chains = [c for c in chains.replace("+", " ").split() if c]
    return "+".join(sorted(str(c) for c in chains))


def merge_roles(anchor, spur, out, log, dst_name="roles_merged.csv"):
    """in: the anchor and spurious role csvs and the experiment dir.  out: the merged csv path

    WHY THIS EXISTS. pose_to_pose_dockq.py builds a matrix only for an interface it can find roles for
    (`k in roles`), and roles are written by two different steps: anchor_relabel.py covers the 292
    interfaces the reference contains, fetch_spurious_interface_iptm.py covers the 53 the predictions
    invent. Passing only the first silently dropped every predicted-only interface -- 53 of 345, and
    the ones the confident-decoy question is actually about. The step reported success because nothing
    compared its 292 matrices against the 345 interfaces upstream had declared.
    """
    dst = f"{out}/{dst_name}"
    seen, rows, hdr = set(), [], None
    for src in (anchor, spur):
        if not os.path.exists(src):
            log(f"  [roles] MISSING {src}")
            continue
        with open(src) as fh:
            r = csv.DictReader(fh)
            hdr = hdr or r.fieldnames
            n = 0
            for row in r:
                k = (row["pdb"], row["chain_1"], row["chain_2"])
                if k in seen:          # an interface named by both files keeps the anchor's roles
                    continue
                seen.add(k)
                rows.append(row)
                n += 1
        log(f"  [roles] {n} rows from {os.path.basename(src)}")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    log(f"         wrote {dst}  ({len(rows)} interfaces with roles)")
    return dst


def run_sharded(cmd, gpus, outputs, force, log, name, deadline_s):
    """in: the base command, GPUs to shard over, outputs, force, logger, step name, seconds allowed.
    out: whether it ran

    One process per GPU with --shard i/n. Safe to split because each shard writes its own npz files
    and its own build log, and every shard skips an interface whose npz already exists -- so a shard
    that is killed loses only the interface it was mid-way through.

    THE DEADLINE is a promise about borrowed hardware, not an optimisation: the extra GPUs are over
    the standing 4-card budget, so after deadline_s the extra shards are stopped and the remainder is
    finished on the FIRST gpu alone, which is back inside budget. Resuming costs nothing for the same
    reason a killed shard is cheap.
    """
    def there(p):
        return os.path.isdir(p) and bool(os.listdir(p))

    if not force and outputs and all(there(p) for p in outputs):
        log(f"  [{name}] present, skipping")
        return False
    n = len(gpus)
    log(f"  [{name}] running {os.path.basename(str(cmd[1]))} as {n} shards on GPUs {','.join(gpus)}")
    log(f"         deadline {deadline_s}s on the extra GPUs, then GPU {gpus[0]} finishes alone")
    for p in outputs:
        log(f"         -> {p}")
    procs = []
    for i, g in enumerate(gpus):
        c = [str(x) for x in cmd] + ["--shard", f"{i}/{n}"]
        procs.append((g, subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, env={**os.environ,
                                                          "CUDA_VISIBLE_DEVICES": g})))
    t0 = time.time()
    late = False
    while any(p.poll() is None for _, p in procs):
        if time.time() - t0 > deadline_s:
            late = True
            for g, p in procs:
                if p.poll() is None:
                    log(f"         ! deadline reached, stopping shard on GPU {g}")
                    p.terminate()
            break
        time.sleep(5)
    for g, p in procs:
        txt = (p.communicate()[0] or "") if p.stdout else ""
        tail = [l for l in txt.splitlines() if l.strip()][-3:]
        log(f"         GPU {g} exit {p.returncode}" + (f"   {tail[-1].strip()}" if tail else ""))
        if p.returncode not in (0, None) and not late:
            for l in txt.splitlines()[-15:]:
                log(f"    ! {l}")
            raise SystemExit(f"[{name}] shard on GPU {g} failed with exit {p.returncode}")
    if late:
        log(f"         finishing the remainder on GPU {gpus[0]} alone (within the 4-GPU budget)")
        run([str(x) for x in cmd], outputs, True, log, f"{name}-finish",
            env={"CUDA_VISIBLE_DEVICES": gpus[0]})
    return True


def validate(name, out, anchor, spur, mats, ns, log):
    """in: the step just finished and the experiment's directories.  out: nothing, raises on a problem

    The presence check in run() only asks whether an output is non-trivial. This asks whether it is
    RIGHT, which is a different question and the one that was missing: the previous run produced a
    selection drawing 302 donors from seeds outside the rule, and every step reported success. Each
    check below is a property the step's output must have if the 50-seed rule actually held. A failure
    stops the pipeline here rather than letting the next step inherit bad input.
    """
    def fail(msg):
        log(f"         ! FAIL {msg}")
        raise SystemExit(f"[{name}] validation failed: {msg}")

    def rows(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))

    log(f"  [{name}] validating")
    if name == "anchor":
        m, r = rows(f"{anchor}/chain_maps_anchor.csv"), rows(f"{anchor}/chain_roles_anchor.csv")
        pool = rows(f"{out}/pool_50.csv")
        want = {(x["pdb"], x["seed"], x["sample"]) for x in pool}
        got = {(x["pdb"], x["seed"], x["sample"]) for x in m}
        bad = {x["seed"] for x in m} - POOL_SEEDS
        if bad:
            fail(f"chain maps carry {len(bad)} seeds outside the rule: {sorted(bad)[:5]}")
        if not got >= want:
            fail(f"chain maps cover {len(got)} of {len(want)} pool poses "
                 f"({len(want - got)} missing, e.g. {sorted(want - got)[:3]})")
        log(f"         {len(m)} pose maps covering all {len(want)} pool poses, "
            f"{len(r)} role rows, 0 seeds outside the rule")

    elif name == "interfaces":
        r = rows(f"{out}/pose_interfaces.csv")
        pool = {x["pdb"] for x in rows(f"{out}/pool_50.csv")}
        miss = pool - {x["pdb"] for x in r}
        if miss:
            fail(f"{len(miss)} complexes have no interface declared: {sorted(miss)[:5]}")
        gt = sum(1 for x in r if x["have_gt"] == "1")
        log(f"         {len(r)} interfaces over {len(pool)} complexes: "
            f"{gt} with ground truth, {len(r) - gt} predicted-only")

    elif name == "spurious":
        r = rows(f"{spur}/spurious_pool.csv")
        bad = {x["seed"] for x in r} - POOL_SEEDS
        if bad:
            fail(f"spurious pool carries seeds outside the rule: {sorted(bad)[:5]}")
        if any(x["dockq"] for x in r):
            fail("spurious rows carry a dockq value; they have no reference and must be empty")
        log(f"         {len(r)} pose rows, all seeds in range, dockq empty as required")

    elif name == "dockq":
        import numpy as np
        files = sorted(f for f in os.listdir(mats) if f.endswith(".npz"))
        if not files:
            fail(f"no npz written to {mats}")
        over, seedn = [], []
        for f in files:
            d = np.load(f"{mats}/{f}", allow_pickle=True)
            s = {str(x) for x in d["seed"]}
            seedn.append(len(s))
            if s - POOL_SEEDS:
                over.append(f)
            if d["dockq"].shape[0] != d["seed"].shape[0]:
                fail(f"{f}: matrix is {d['dockq'].shape} but carries {d['seed'].shape[0]} poses")
        if over:
            fail(f"{len(over)} matrices hold seeds outside the rule: {over[:5]}")
        # The check that was missing: 292 matrices looked fine on their own, but 345 interfaces had
        # been declared upstream. A matrix count is only meaningful against what the previous steps
        # produced, never against itself.
        have = {(r["pdb"], r["chain_1"], r["chain_2"])
                for r in rows(f"{out}/roles_merged.csv")}
        built = {f[:-4] for f in files}
        want = {f'{p}_{a}{b}' for p, a, b in have}
        miss = want - built
        log(f"         {len(files)} interface matrices, seeds per matrix "
            f"{min(seedn)}-{max(seedn)} (cap {len(POOL_SEEDS)}), 0 outside the rule")
        log(f"         roles cover {len(want)} interfaces; {len(miss)} have no matrix")
        if len(miss) > len(want) * 0.15:
            fail(f"{len(miss)} of {len(want)} interfaces have no matrix, e.g. {sorted(miss)[:5]} -- "
                 f"a whole class of interface is probably being dropped")

    elif name == "filter":
        q = [l.rstrip("\n").split("\t") for l in open(f"{out}/queue.tsv") if l.strip()]
        if not q:
            fail("queue.tsv is empty; nothing was selected")
        seeds = {m.group(1) for m in (re.search(r"/free/[^/]+/(\d+)-", f[5]) for f in q if len(f) > 5)
                 if m}
        bad = seeds - POOL_SEEDS
        if bad:
            fail(f"queue draws donors from {len(bad)} seeds outside the rule: {sorted(bad)[:5]}")
        missing = [f[5] for f in q if len(f) > 5 and not os.path.exists(f[5])]
        if missing:
            fail(f"{len(missing)} donor poses do not exist on disk, e.g. {missing[:2]}")
        runs = {m.group(1) for m in (re.search(r"/free/[^/]+/\d+-([^/]+)/", f[5]) for f in q
                                    if len(f) > 5) if m}
        if runs != {RUN_DIR}:
            fail(f"donors come from fold directories other than {RUN_DIR}: {sorted(runs)}")
        per = collections.Counter(f[1] for f in q)
        log(f"         {len(q)} folds over {len(per)} complexes, {max(per.values())} max per complex "
            f"(cap {len(POOL_SEEDS)})")
        log(f"         all donors: {len(seeds)} seeds in range, run dir {RUN_DIR} only, all on disk")

    elif name == "templates":
        q = [l for l in open(f"{out}/queue.tsv") if l.strip()]
        # listdir(ns) counts jobs/ mmcif/ release_dates.json -- always 3, so the old check passed
        # no matter what. The namespaces are one level down, one per queued fold.
        made = set(os.listdir(f"{ns}/jobs")) if os.path.isdir(f"{ns}/jobs") else set()
        if not made:
            fail(f"no template namespaces written to {ns}/jobs")
        # COVERAGE, not equality. A re-selection leaves the previous run's namespaces behind, so the
        # directory legitimately holds more than the queue needs -- the union of two selections. What
        # would actually break folding is a QUEUED fold with no namespace, so that is what is checked;
        # leftovers are reported because they are clutter worth seeing, not a failure.
        want = {f"{f[2]}_{f[3]}" for f in (l.rstrip("\n").split("\t") for l in q) if len(f) > 3}
        missing = want - made
        if missing:
            fail(f"{len(missing)} queued folds have no template namespace, "
                 f"e.g. {sorted(missing)[:5]}")
        log(f"         all {len(want)} queued folds have a namespace in {ns}/jobs")
        if made - want:
            log(f"         note: {len(made - want)} stale namespaces from an earlier selection")

    else:
        log(f"         (no check defined)")


def fold_id(donor, seed, chains):
    """in: donor pose path, seed, pinned chains.  out: the reuse key

    A fold's output is fixed by these three. The tag is only a label, and the target interface is not
    part of the identity -- one fold scores every interface of its complex.
    """
    return (donor, str(seed), norm_chains(chains))


def chains_for(ns, iface, tag):
    """in: the namespace, an interface and tag.  out: the chains the builder pinned, or '' """
    import json
    p = f"{ns}/jobs/{iface}_{tag}/chosen.json"
    if not os.path.exists(p):
        return ""
    try:
        d = json.load(open(p))
        # chosen.json is a LIST of chosen templates, one per donor, not a dict
        return norm_chains(d[0].get("chains") if isinstance(d, list) and d else d.get("chains"))
    except (ValueError, OSError, KeyError, AttributeError, IndexError):
        return ""


def diff(out, old, ns):
    """in: this experiment, the earlier one, this namespace.  out: (rows, the three id sets)"""
    new = {}
    for line in open(f"{out}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            new[fold_id(f[5], f[4], chains_for(ns, f[2], f[3]))] = f
    have = {}
    src = f"{old}/results.csv"
    if os.path.exists(src):
        for r in csv.DictReader(open(src)):
            d = r.get("donor_pose", "")
            if d and d != "-1":
                have[fold_id(d, r["seed"], r.get("tmpl_chains", ""))] = r
    carry = set(new) & set(have)
    tofold = set(new) - set(have)
    omit = set(have) - set(new)
    outrange = sum(1 for k in omit if not any(f"/{s}-" in k[0] for s in POOL_SEEDS))
    rows = [["carry over (exists, still selected)", len(carry)],
            ["to fold (newly selected, never run)", len(tofold)],
            ["omit (exists, no longer selected)", len(omit)],
            ["   of those, donor seed out of range", outrange]]

    # The queue to actually SPEND GPU on. Claim names embed the tag, and the two selections assign
    # different tags to the same fold, so a carried fold cannot be pre-claimed to make workers skip it
    # -- it would need a claim under a name that never existed. Folding a reduced queue instead keeps
    # the two sources of score separate and honest: 757 folded here, 2,585 read from the earlier run.
    with open(f"{out}/queue_tofold.tsv", "w") as fh:
        for k in sorted(tofold):
            fh.write("\t".join(new[k]) + "\n")
    with open(f"{out}/carry_over_folds.tsv", "w") as fh:
        fh.write("donor_pose\tseed\ttmpl_chains\n")
        for d, sd, ch in sorted(carry):
            fh.write(f"{d}\t{sd}\t{ch}\n")
    return rows, (carry, tofold, omit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--diff-against", default=f"{paths.DATA_DIR}/reffree_50")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--force", choices=STEPS, default=None)
    ap.add_argument("--stop-after", choices=STEPS, default=None)
    A = ap.parse_args()

    out = A.out
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/build.log", "a")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    start = STEPS.index(A.force) if A.force else len(STEPS)
    stop = STEPS.index(A.stop_after) if A.stop_after else len(STEPS) - 1
    forced = {s: STEPS.index(s) >= start for s in STEPS}
    log(f"\n=== self_template_50seeds_pxm -> {out}   seeds 300000-300049, {RUN_DIR} only ===")

    anchor, spur = f"{out}/anchor", f"{out}/spurious"
    mats, ns = f"{out}/matrices", f"{out}/ns"
    pool = f"{out}/pool_50.csv"
    if forced["pool"] or not os.path.exists(pool):
        filtered_pool(out, log)
    else:
        log("  [pool] present, skipping")
    if stop == 0:
        return 0

    run([PXM_PY, f"{HERE}/anchor_relabel.py", "--pool", pool, "--out-dir", anchor, "--jobs", A.jobs],
        [f"{anchor}/chain_maps_anchor.csv", f"{anchor}/chain_roles_anchor.csv"],
        forced["anchor"], log, "anchor")
    validate("anchor", out, anchor, spur, mats, ns, log)
    if stop == 1:
        return 0

    # pose_interfaces.py's --out is a DIRECTORY it writes pose_interfaces.csv into, not a file path.
    # Passing the file path made it create a directory of that name with the csv nested inside.
    run([PY, f"{HERE}/pose_interfaces.py", "--pool", pool, "--anchor-dir", anchor,
         "--out", out, "--jobs", A.jobs],
        [f"{out}/pose_interfaces.csv"], forced["interfaces"], log, "interfaces")
    validate("interfaces", out, anchor, spur, mats, ns, log)
    if stop == 2:
        return 0

    run([PY, f"{HERE}/fetch_spurious_interface_iptm.py", "--ifaces", f"{out}/pose_interfaces.csv",
         "--pool", pool, "--out", spur, "--jobs", A.jobs],
        [f"{spur}/spurious_pool.csv", f"{spur}/spurious_roles.csv"],
        forced["spurious"], log, "spurious")
    validate("spurious", out, anchor, spur, mats, ns, log)
    if stop == 3:
        return 0

    roles = merge_roles(f"{anchor}/chain_roles_anchor.csv", f"{spur}/spurious_roles.csv", out, log)
    run_sharded([PY, f"{HERE}/pose_to_pose_dockq.py", "--pool", pool, "--extra-pool",
         f"{spur}/spurious_pool.csv", "--maps", f"{anchor}/chain_maps_anchor.csv",
         "--roles", roles, "--out", mats,
         "--device", "cuda"],
        DOCKQ_GPUS, [mats], forced["dockq"], log, "dockq", DOCKQ_DEADLINE_S)
    validate("dockq", out, anchor, spur, mats, ns, log)
    if stop == 4:
        return 0

    run([PY, f"{HERE}/structural_diversity_filter.py", "--out", out, "--cap", A.cap,
         "--seed", A.seed, "--no-reuse", "--pose-interfaces", "--anchor-dir", anchor,
         "--anchor-matrices", mats, "--xtal-matrices", mats, "--spur-matrices", mats],
        [f"{out}/queue.tsv"], forced["filter"], log, "filter")
    validate("filter", out, anchor, spur, mats, ns, log)
    # The frame roles the filter writes cover only the reference interfaces, exactly like the anchor
    # roles did -- so steps 6 and 7 need the same merge, or build_templates.py dies on the first
    # predicted-only interface it is asked to build (KeyError: '8w5d_DC').
    frame_roles = merge_roles(f"{out}/frame_roles.csv", f"{spur}/spurious_roles.csv", out, log,
                              dst_name="frame_roles_merged.csv")
    # the guard that makes this class of bug impossible to ship: every interface in the queue must
    # have roles, checked before any GPU time is spent rather than discovered by a crash
    have = {f'{r["pdb"]}_{r["chain_1"]}{r["chain_2"]}'
            for r in csv.DictReader(open(frame_roles))}
    want = {l.split("\t")[2] for l in open(f"{out}/queue.tsv") if l.strip()}
    if want - have:
        log(f"         ! FAIL {len(want - have)} queued interfaces have no roles: "
            f"{sorted(want - have)[:5]}")
        raise SystemExit("[roles] queued interfaces without roles")
    log(f"         all {len(want)} queued interfaces have roles")
    if stop == 5:
        return 0

    run([PY, f"{HERE}/build_templates.py", "--queue", f"{out}/queue.tsv", "--ns", ns,
         "--maps", f"{out}/frame_maps.csv", "--roles", frame_roles, "--jobs", A.jobs],
        [f"{ns}/release_dates.json"], forced["templates"], log, "templates")
    validate("templates", out, anchor, spur, mats, ns, log)
    if stop == 6:
        return 0

    run([PY, f"{HERE}/preflight.py", "--ns", ns, "--queue", f"{out}/queue.tsv",
         "--maps", f"{out}/frame_maps.csv", "--roles", frame_roles],
        [], True, log, "preflight")

    if A.diff_against and os.path.isdir(A.diff_against):
        rows, _ = diff(out, A.diff_against, ns)
        name = os.path.basename(A.diff_against.rstrip("/"))
        t = table(["fold", "count"], rows, align="l r")
        log(f"\n  fold-for-fold against {name}:\n{t}")
        open(f"{out}/diff_vs_{name}.md", "w").write(
            f"# {os.path.basename(out)} vs {name}\n\n"
            f"Fold identity is (donor pose, seed, pinned chains).\n\n{t}\n")
        log(f"  wrote {out}/diff_vs_{name}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
