#!/usr/bin/env python
"""Build the 50-seed self-templating experiment end to end: pool -> templates, ready for the GPU.

SUMMARY
    One command for every step before folding, so the only thing it spends is CPU and, for the DockQ
    matrices, a GPU for under an hour.

    NOTHING ON THE SELECTION PATH READS THE CRYSTAL. Antibody and antigen come from the input
    sequences, interfaces from contacts in the poses, the ranking from each pose's own confidence json,
    the similarity from pose-to-pose DockQ. The crystal is read only to score folds (scorer.py) and to
    report arms (score_experiment.py, compare_arms.py). The one exception is WHICH complexes form the
    benchmark -- the ids of the crystal-scored pool, the same 123 the free arms use -- and nothing per
    pose or per chain is read from it.

    THE POOL RULE, hardcoded here and nowhere else:

      seeds 300000-300049, and the legacy-no-provenance fold directory only.

    The pool on disk holds up to 95 seeds on some complexes and several fold directories per seed on
    others; an earlier build drew 302 of 3,132 donors from seeds past 300049 because nothing capped it.

    WHAT IT RUNS, in order. Each step reads the previous step's output from <out>:

      pool        poses.csv: every pose the rule admits, found on disk
      roles       chain_roles.py         antibody or antigen per input entity (ANARCI)
      anchor      anchor_relabel.py      each pose's contacting antibody-antigen pairs and their ipTM;
                                         the anchor per complex; every pose's chains named after it
      interfaces  pose_interfaces.py     antibody-antigen anchor pairs in contact in >= 1 pose
      dockq       pose_to_pose_dockq.py  N x N pose-to-pose DockQ and ipTM per interface (GPU)
      filter      structural_diversity_filter.py  ipTM order, diversity filter, round-robin, cap 50
      templates   build_templates.py     each donor pose -> a template Protenix can load
      preflight   preflight.py           donors, ids, chains, paths and hits, before any GPU time

    Each step is VALIDATED against the previous step's output, not only its own: a check that looks
    only at a step's own output cannot catch a step that lost data.

    RESUME. A step is skipped when its output is present; --force <step> recomputes from that step on.
    The DockQ step always runs and skips each interface already built.

    THE DIFF. --diff-against an earlier experiment reuses its scored folds by fold identity (donor pose,
    seed, pinned chains): carried folds' rows are copied into results.csv under this selection's tags,
    and only the rest go into queue_tofold.tsv. Folds the new selection omits are never copied.

INPUTS
    --out:           experiment directory to build. Required.
    --diff-against:  an earlier experiment whose scored folds may be reused. Optional.
    --varseed-out:   also write the variable-seed arm's queue into this directory. Optional.
    --cap:           donors per complex. Default 50.
    --seed:          the seed every templated fold uses. Default 430000.
    --jobs:          worker processes for the CPU steps. Default 16.
    --force:         recompute from this step onward. Default off, i.e. resume.
    --stop-after:    run no further than this step.

    DOCKQ_GPUS (env): GPUs for the DockQ step, comma-separated. Required when that step runs; never
    guessed on a shared machine. DOCKQ_DEADLINE_S (env): seconds before extra shards stop. Default 1800.

OUTPUTS
    <out>/poses.csv                  the pool: pdb, seed, sample
    <out>/chain_roles.csv            antibody or antigen per entity
    <out>/chain_frame/               pose_contacts.csv, chain_maps_anchor.csv, anchor_chains.csv,
                                     anchor_manifest.json
    <out>/interfaces.csv             the interfaces, in anchor letters
    <out>/pose_dockq_matrices/       one npz per interface
    <out>/queue.tsv                  every selected fold
    <out>/templates/                 the template namespace
    <out>/queue_tofold.tsv, carry_over_folds.tsv, results.csv, diff_vs_<name>.md   with --diff-against
    <varseed-out>/queue.tsv          with --varseed-out
    <out>/build.log                  every step's stdout

FUNCTIONS
    pool_poses():     write poses.csv from disk under the pool rule, and verify it is complete
    run():            one step as a subprocess, logged, skipped when its output exists
    run_sharded():    the DockQ step, one process per GPU
    validate():       check a finished step against the ones before it
    fold_id():        (donor pose, seed, pinned chains) -- what decides whether a fold can be reused
    diff():           new selection vs an earlier experiment; writes the carried rows and the to-fold queue
    varseed_queue():  this queue with one seed per donor, for the variable-seed arm
    main():           the steps in order, then the diff
"""
import argparse
import collections
import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fmt import table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Three interpreters, because no single environment holds every dependency: anchor_relabel.py needs
# pxmeter, chain_roles.py a working ANARCI, everything else gemmi and torch from the Protenix venv.
# Steps name their interpreter rather than inheriting one.
PXM_PY = str(paths.PXMETER_PYTHON)
ANARCI_PY = str(paths.ANARCI_PYTHON)
PY = str(paths.PROTENIX_PYTHON)
DOCKQ_DEADLINE_S = int(os.environ.get("DOCKQ_DEADLINE_S", "1800"))

# the rule. seeds 300000-300049 and the legacy fold directory, nothing else.
POOL_SEEDS = [str(300000 + i) for i in range(50)]
SAMPLES = range(5)
RUN_DIR = paths.LEGACY_RUN

STEPS = ["pool", "roles", "anchor", "interfaces", "dockq", "filter", "templates", "preflight"]


def pool_poses(dst, log):
    """in: the poses.csv path, a logger.  out: nothing; raises unless every pose the rule names exists

    Benchmark complexes are the ids in the crystal-scored pool csv (the evaluated set, the same 123
    the free arms use); only its pdb column is read.
    """
    pdbs = sorted({r["pdb"] for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV)))})
    rows, missing = [], []
    for pdb in pdbs:
        for seed in POOL_SEEDS:
            for x in SAMPLES:
                p = f"{paths.FREE_DIR}/{pdb}/{seed}-{RUN_DIR}/predictions/{pdb}_sample_{x}.cif"
                (rows if os.path.exists(p) else missing).append((pdb, seed, x))
    log(f"  [pool] {len(pdbs)} complexes x {len(POOL_SEEDS)} seeds x {len(SAMPLES)} samples, "
        f"{RUN_DIR} only: {len(rows)} poses on disk, {len(missing)} missing")
    if missing:
        raise SystemExit(f"[pool] {len(missing)} poses missing, e.g. {missing[:3]}")
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "seed", "sample"])
        w.writerows(rows)
    log(f"         wrote {dst}")


def run(cmd, outputs, force, log, name, env=None):
    """in: the command, the paths it produces, whether to force, a logger, the step name, env extras.
    out: whether it ran
    """
    # A step that created its output and then died must not count as done: an empty output directory,
    # or a csv holding only its header row, is what an interrupted step leaves.
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


def run_sharded(cmd, gpus, outputs, log, name, deadline_s):
    """in: the base command, GPUs to shard over, outputs, a logger, the step name, seconds allowed.
    out: nothing

    One process per GPU with --shard i/n. Safe to split because each shard writes its own npz files
    and its own build log, and every shard skips an interface whose npz already exists -- so a shard
    that is killed loses only the interface it was mid-way through.

    THE DEADLINE: after deadline_s the extra shards are stopped and the remainder is finished on the
    FIRST gpu alone. Resuming costs nothing for the same reason a killed shard is cheap.

    ALWAYS RUNS. A non-empty matrix directory is exactly what an interrupted run leaves; treating it as
    done once skipped 315 of 339 interfaces. The script itself skips every interface already built.
    """
    n = len(gpus)
    log(f"  [{name}] running {os.path.basename(str(cmd[1]))} as {n} shards on GPUs {','.join(gpus)}")
    for p in outputs:
        log(f"         -> {p}")
    procs = []
    for i, g in enumerate(gpus):
        c = [str(x) for x in cmd] + ["--shard", f"{i}/{n}"]
        procs.append((g, subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": g})))
    t0, late = time.time(), False
    while any(p.poll() is None for _, p in procs):
        if time.time() - t0 > deadline_s and n > 1:
            late = True
            for g, p in procs:
                if p.poll() is None:
                    log(f"         ! deadline reached, stopping shard on GPU {g}")
                    p.terminate()
            break
        time.sleep(5)
    for g, p in procs:
        txt = (p.communicate()[0] or "") if p.stdout else ""
        tail = [l for l in txt.splitlines() if l.strip()][-1:]
        log(f"         GPU {g} exit {p.returncode}" + (f"   {tail[0].strip()}" if tail else ""))
        if p.returncode not in (0, None) and not late:
            for l in txt.splitlines()[-15:]:
                log(f"    ! {l}")
            raise SystemExit(f"[{name}] shard on GPU {g} failed with exit {p.returncode}")
    if late:
        log(f"         finishing the remainder on GPU {gpus[0]} alone")
        run([str(x) for x in cmd], outputs, True, log, f"{name}-finish",
            env={"CUDA_VISIBLE_DEVICES": gpus[0]})


def validate(name, out, log):
    """in: the step just finished and the experiment directory.  out: nothing, raises on a problem

    Each check compares a step's output with what the steps before it produced.
    """
    def fail(msg):
        log(f"         ! FAIL {msg}")
        raise SystemExit(f"[{name}] validation failed: {msg}")

    def rows(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))

    poses = rows(f"{out}/poses.csv")
    pdbs = {r["pdb"] for r in poses}
    if name == "roles":
        r = rows(f"{out}/chain_roles.csv")
        got = {x["pdb"] for x in r}
        if got != pdbs:
            fail(f"roles cover {len(got)} of {len(pdbs)} complexes")
        n_ab = sum(1 for x in r if x["role"] == "antibody")
        log(f"         {len(r)} entities over {len(got)} complexes: {n_ab} antibody, "
            f"{len(r) - n_ab} antigen")

    elif name == "anchor":
        m = rows(f"{out}/chain_frame/chain_maps_anchor.csv")
        want = {(x["pdb"], x["seed"], x["sample"]) for x in poses}
        got = {(x["pdb"], x["seed"], x["sample"]) for x in m if x["chain_mapping"]}
        if got != want:
            fail(f"chain maps cover {len(got & want)} of {len(want)} pool poses")
        man = json.load(open(f"{out}/chain_frame/anchor_manifest.json"))
        if set(man["anchors"]) != pdbs:
            fail(f"anchors for {len(man['anchors'])} of {len(pdbs)} complexes")
        log(f"         {len(got)} pose maps covering every pool pose, {len(pdbs)} anchors")

    elif name == "interfaces":
        r = rows(f"{out}/interfaces.csv")
        miss = pdbs - {x["pdb"] for x in r}
        if miss:
            fail(f"{len(miss)} complexes have no interface: {sorted(miss)[:5]}")
        log(f"         {len(r)} interfaces over {len(pdbs)} complexes")

    elif name == "dockq":
        import numpy as np
        mats = f"{out}/pose_dockq_matrices"
        ifs = set(pool_index.load_roles(f"{out}/interfaces.csv"))
        files = {f[:-4] for f in os.listdir(mats) if f.endswith(".npz")}
        stray = files - ifs
        if stray:
            fail(f"{len(stray)} matrices name no interface of this build, e.g. {sorted(stray)[:3]}")
        # every interface without a matrix must be accounted for by the build log (too few contacts)
        logged = {}
        for f in os.listdir(mats):
            if f.startswith("build_log") and f.endswith(".csv"):
                for x in rows(f"{mats}/{f}"):
                    logged[x["interface"]] = x
        unexplained = [i for i in ifs - files if i not in logged or logged[i]["rows"] not in ("0",)]
        if unexplained:
            fail(f"{len(unexplained)} interfaces have no matrix and no logged reason: "
                 f"{sorted(unexplained)[:5]}")
        rule = set(POOL_SEEDS)
        for f in sorted(files):
            z = np.load(f"{mats}/{f}.npz")
            if {str(s) for s in z["seed"]} - rule:
                fail(f"{f}: seeds outside the rule")
            if not (z["dockq"].shape[0] == len(z["seed"]) == len(z["pair_iptm"])):
                fail(f"{f}: matrix and labels disagree in length")
        log(f"         {len(files)} matrices for {len(ifs)} interfaces; "
            f"{len(ifs - files)} below the contact minimum, per the build log")

    elif name == "filter":
        q = [l.rstrip("\n").split("\t") for l in open(f"{out}/queue.tsv") if l.strip()]
        if not q:
            fail("queue.tsv is empty")
        ifs = set(pool_index.load_roles(f"{out}/interfaces.csv"))
        if {f[2] for f in q} - ifs:
            fail("the queue names interfaces this build did not declare")
        seeds = {re.search(r"/free/[^/]+/(\d+)-", f[5]).group(1) for f in q}
        runs = {re.search(r"/free/[^/]+/\d+-([^/]+)/", f[5]).group(1) for f in q}
        if seeds - set(POOL_SEEDS) or runs != {RUN_DIR}:
            fail(f"donors outside the rule: seeds {sorted(seeds - set(POOL_SEEDS))[:5]}, runs {runs}")
        if len({(f[1], f[5]) for f in q}) != len(q) or len({(f[1], f[3]) for f in q}) != len(q):
            fail("a donor or a tag repeats within a complex")
        per = collections.Counter(f[1] for f in q)
        log(f"         {len(q)} folds over {len(per)} complexes, at most {max(per.values())} per "
            f"complex; all donors in the rule, none repeated")

    elif name == "templates":
        want = {f"{f[2]}_{f[3]}" for f in (l.rstrip("\n").split("\t")
                                           for l in open(f"{out}/queue.tsv") if l.strip())}
        made = set(os.listdir(f"{out}/templates/jobs"))
        if want - made:
            fail(f"{len(want - made)} queued folds have no template namespace")
        log(f"         all {len(want)} queued folds have a namespace")


def norm_chains(chains):
    """in: chains as a list or a plus-joined string.  out: a canonical sorted plus-joined string"""
    if isinstance(chains, str):
        chains = [c for c in chains.replace("+", " ").split() if c]
    return "+".join(sorted(str(c) for c in chains or []))


def fold_id(donor, seed, chains):
    """in: donor pose path, seed, pinned chains.  out: the reuse key

    A fold's output is fixed by these three. The tag is only a label, and the interface is not part of
    the identity -- one fold scores every interface of its complex.
    """
    return (donor, str(seed), norm_chains(chains))


def diff(out, old):
    """in: this experiment and an earlier one.  out: table rows

    Writes queue_tofold.tsv (the folds to spend GPU on), carry_over_folds.tsv, and results.csv: the
    carried folds' rows under THIS selection's tags. A fold the new selection omits is never copied.
    """
    new = {}
    for line in open(f"{out}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            ch = json.load(open(f"{out}/templates/jobs/{f[2]}_{f[3]}/chosen.json"))[0]["chains"]
            new[fold_id(f[5], f[4], ch)] = f
    src = f"{old}/results.csv"
    have = {fold_id(r["donor_pose"], r["seed"], r["tmpl_chains"]) for r in csv.DictReader(open(src))
            if r.get("donor_pose") not in ("", "-1", None)}
    carry, tofold, omit = set(new) & have, set(new) - have, have - set(new)

    with open(f"{out}/queue_tofold.tsv", "w") as fh:
        fh.writelines("\t".join(new[k]) + "\n" for k in sorted(tofold))
    with open(f"{out}/carry_over_folds.tsv", "w") as fh:
        fh.write("donor_pose\tseed\ttmpl_chains\n")
        fh.writelines(f"{d}\t{s}\t{c}\n" for d, s, c in sorted(carry))
    dst = f"{out}/results.csv"
    if os.path.exists(dst) and sum(1 for _ in itertools.islice(open(dst), 2)) > 1:
        raise SystemExit(f"{dst} already holds scored rows; refusing to overwrite")
    n = 0
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"])
        for r in csv.DictReader(open(src)):
            k = fold_id(r.get("donor_pose", ""), r["seed"], r.get("tmpl_chains", ""))
            if k in carry:
                w.writerow([r["iface"], r["seed"], new[k][3], r["donor_pose"], k[2],
                            r["sample"], r["dockq"], r["iptm"]])
                n += 1
    return [["carry over (scored, still selected)", len(carry)],
            ["to fold (selected, never scored)", len(tofold)],
            ["omit (scored, no longer selected)", len(omit)],
            ["rows carried into results.csv", n]]


def varseed_queue(queue, dst):
    """in: this experiment's queue.tsv, the destination queue path.  out: lines written

    The variable-seed arm folds the SAME donors with the SAME templates, but each donor on its own seed
    instead of all on one. Seeds are 300100-300149 -- the 50 the compute-matched free arm used, so the
    extra folds are the same seeds spent differently. Per complex: sort by (pdb, interface, tag),
    shuffle the 50 seeds with random.Random(f"varseed-{pdb}"), assign in order. Reproduces the
    2026-09-24 hand-built queue byte for byte from that build's fixed queue.
    """
    lines = [l.rstrip("\n").split("\t") for l in open(queue) if l.strip()]
    per = collections.Counter(f[1] for f in lines)
    if max(per.values()) > 50:
        raise SystemExit(f"a complex has {max(per.values())} donors; only 50 seeds to assign")
    out, seeds, n = [], {}, collections.Counter()
    for f in sorted(lines, key=lambda f: (f[1], f[2], f[3])):
        if f[1] not in seeds:
            s = list(range(300100, 300150))
            random.Random(f"varseed-{f[1]}").shuffle(s)
            seeds[f[1]] = s
        out.append(f[:4] + [str(seeds[f[1]][n[f[1]]])] + f[5:])
        n[f[1]] += 1
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as fh:
        fh.writelines("\t".join(f) + "\n" for f in out)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--diff-against", default=None)
    ap.add_argument("--varseed-out", default=None)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--force", choices=STEPS, default=None)
    ap.add_argument("--stop-after", choices=STEPS, default=None)
    A = ap.parse_args()

    out = os.path.abspath(A.out)
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/build.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    forced = {s: A.force is not None and STEPS.index(s) >= STEPS.index(A.force) for s in STEPS}
    last = STEPS.index(A.stop_after) if A.stop_after else len(STEPS) - 1
    # the template namespace's path is baked into every job json, so it is named here, once
    frame, mats, ns = f"{out}/chain_frame", f"{out}/pose_dockq_matrices", f"{out}/templates"
    poses, roles, ifaces = f"{out}/poses.csv", f"{out}/chain_roles.csv", f"{out}/interfaces.csv"
    maps = f"{frame}/chain_maps_anchor.csv"
    log(f"\n=== self_template_50seeds_pxm -> {out}   seeds 300000-300049, {RUN_DIR} only ===")

    for i, step in enumerate(STEPS[:last + 1]):
        if step == "pool":
            if forced["pool"] or not os.path.exists(poses):
                pool_poses(poses, log)
            else:
                log("  [pool] present, skipping")
            continue
        if step == "roles":
            run([ANARCI_PY, f"{HERE}/chain_roles.py", "--pdbs", poses, "--out", roles], [roles],
                forced[step], log, step)
        elif step == "anchor":
            run([PXM_PY, f"{HERE}/anchor_relabel.py", "--poses", poses, "--roles", roles,
                 "--out-dir", frame, "--jobs", A.jobs],
                [maps, f"{frame}/anchor_chains.csv", f"{frame}/pose_contacts.csv"], forced[step], log,
                step)
        elif step == "interfaces":
            run([PY, f"{HERE}/pose_interfaces.py", "--anchor-dir", frame, "--out", ifaces], [ifaces],
                forced[step], log, step)
        elif step == "dockq":
            gpus = [g for g in os.environ.get("DOCKQ_GPUS", "").split(",") if g]
            if not gpus:
                raise SystemExit("[dockq] set DOCKQ_GPUS: this machine is shared, GPUs are never guessed")
            run_sharded([PY, f"{HERE}/pose_to_pose_dockq.py", "--poses", poses, "--maps", maps,
                         "--roles", ifaces, "--out", mats, "--device", "cuda"],
                        gpus, [mats], log, step, DOCKQ_DEADLINE_S)
        elif step == "filter":
            run([PY, f"{HERE}/structural_diversity_filter.py", "--matrices", mats,
                 "--out", f"{out}/queue.tsv", "--cap", A.cap, "--seed", A.seed],
                [f"{out}/queue.tsv"], forced[step], log, step)
        elif step == "templates":
            run([PY, f"{HERE}/build_templates.py", "--queue", f"{out}/queue.tsv", "--ns", ns,
                 "--maps", maps, "--roles", ifaces, "--jobs", A.jobs],
                [f"{ns}/release_dates.json"], forced[step], log, step)
        elif step == "preflight":
            run([PY, f"{HERE}/preflight.py", "--ns", ns, "--queue", f"{out}/queue.tsv",
                 "--maps", maps, "--roles", ifaces], [], True, log, step)
            continue
        log(f"  [{step}] validating")
        validate(step, out, log)

    if last < len(STEPS) - 1:
        return 0
    if A.diff_against:
        name = os.path.basename(A.diff_against.rstrip("/"))
        t = table(["fold", "count"], diff(out, A.diff_against), align="l r")
        log(f"\n  fold-for-fold against {name}:\n{t}")
        open(f"{out}/diff_vs_{name}.md", "w").write(
            f"# {os.path.basename(out)} vs {name}\n\nFold identity is (donor pose, seed, pinned "
            f"chains).\n\n{t}\n")
    if A.varseed_out:
        n = varseed_queue(f"{out}/queue.tsv", f"{A.varseed_out}/queue.tsv")
        log(f"  wrote {A.varseed_out}/queue.tsv ({n} folds, one seed per donor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
