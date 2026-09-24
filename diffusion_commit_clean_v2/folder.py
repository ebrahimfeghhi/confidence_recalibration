#!/usr/bin/env python
"""Fold a queue of units with the checkpoint loaded once, score each, append rows to a csv.

    folder.py --queue q.tsv --out <dir> --csv <dir>/results.csv [--templates <ns>]
    folder.py --report --csv <results.csv>      summarise what has been scored
    folder.py --verify --csv <csv> --out <dir>  cross-check the csv against files on disk

SUMMARY
    The one driver for every kind of fold. The checkpoint is 464M parameters and takes 50-80s to load,
    so one long-lived process works through many units rather than one process per fold. Several
    processes can share a queue: each claims a unit by creating a directory before folding it.

    THREE THINGS THAT WILL BITE YOU
      LAYERNORM_TYPE=openfold is mandatory. Without it the model emits NaN coordinates silently -- no
      error, five plausible-looking cifs, and the failure only surfaces when the scorer dies.

      Templates must be fully built BEFORE the first fold. Protenix caches release dates once per
      process, so an id registered mid-run is rejected and every template is dropped while the fold
      still completes and scores normally.

INPUTS
    --queue:   one fold per line, tab separated. Required.

        kind   pdb    target     tag    seeds            poses
        free   8vqf   *          free   300050-300099
        tmpl   9vca   9vca_AC    A06    300020           /abs/pose.cif

      kind     `free` = no self-templated poses. `tmpl` = template on the listed poses. free does NOT
               mean template-free: Protenix still finds ~4 real PDB templates per chain, so the
               contrast is "PDB templates" vs "PDB templates replaced by our own poses".
      target   the interface this unit was chosen FOR, or `*` for none in particular. It does not
               restrict scoring: every fold is scored against every interface of its complex.
      tag      names the arm. A unit is identified by (seed, tag) and NOT by target, so two lines
               sharing a tag and seed are the same unit even if their targets differ.
      poses    absolute paths, tmpl only.

    --templates:  namespace built by the template builder. Required for tmpl units.
    --out:        where folds land. Default paths.FREE_DIR.
    --csv:        results csv, and the to-do list: an (iface, seed, arm) already present is skipped.
                  Default <FREE_DIR>/results.csv.
    --claim-dir:  atomic mkdir claims so several workers share one queue. Default <FREE_DIR>/claims.
    --pool:       csv whose interfaces define what `target=*` scores. Default paths.PXM2025_POOL_CSV.
    --dataset:    pxm2025 or foldbench. Default pxm2025.
    --ncycle:     recycles per fold. Default 10.
    --nsample:    poses per fold. Default 5.
    --dtype:      bf16, fp32 or fp16. Default bf16.
    --keep-cifs:  keep the poses. On by default; --drop-cifs deletes them after scoring.
    --skip-by-template: skip a unit already folded with the same (complex, seed, template). Off by
                  default, because a csv that records the required pair rather than the full chain set
                  will not match and would refold everything.
    --force:      refold even if that unit was already folded. The new fold gets its own directory.
    --report:     print what has been scored and exit.
    --verify:     cross-check the csv against the poses on disk and exit.

OUTPUTS
    <out>/<pdb>/<seed>-<run_id>/predictions/   the poses. run_id is random because folds do not
                                               reproduce -- two folds of one seed are different data.
    <out>/<pdb>/<seed>-<run_id>/run.json       per-fold provenance
    <out>/index.tsv                            one line per fold, greppable, with the template md5
    <csv>                                      one row per scored pose:

        iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm

    `arm` is the tag: a label. `donor_pose` and `tmpl_chains` are the provenance -- a templated fold is
    identified by (pose, seed, pinned chains). The tag alone used to be the only link back to the
    template, and when two interfaces of one complex both used tag K00 they silently swapped donors.

EXAMPLE
    folder.py --queue q.tsv --out experimental_data/free
    folder.py --queue exp/queue.tsv --out exp/out --csv exp/results.csv \
              --templates exp/ns --claim-dir exp/claims --drop-cifs
    folder.py --verify --csv exp/results.csv --out exp/out

FUNCTIONS
    parse_args():       the command line; defaults point at the free pool
    read_queue():       the queue file -> one (kind, pdb, target, tag, seed, poses) unit per fold
    parse_seeds():      "300050-300099" -> a list of seeds
    check_queue():      refuse a bad queue up front: no namespace, incomplete namespace, or a
                        templated queue aimed at the shared free pool
    interfaces_by_pdb(): the pool csv -> what a unit with target=* gets scored on
    open_csv():         create the results csv if absent; return what is already scored
    unit_already_folded(): has this (complex, seed, template) been folded before
    build_runner():     load the checkpoint once; every unit reuses the returned runner
    Ctx                 the handles every unit shares: args, model, interfaces, done-sets, namespace
    run_unit():         claim one unit, fold it, score it, release the claim if anything fails
    named_job():        copy a unit's job json under a unique name, which is what fixes its output dir
    fold():             run one fold; assert it produced poses and the expected template count
    provenance():       the unit -> (donor_pose, tmpl_chains), the two fields that identify a fold
    score_fold():       run scorer.py per interface and append its rows to the csv
    reconcile_claims(): at startup, free claims whose unit was never scored (a killed worker's work)
    append_index():     one line per fold in index.tsv
    runmeta_md5()       checksum a pose, so run.json pins the exact file templated on
    report():           print what has been scored, per (interface, seed, arm)
    verify():           cross-check the csv against the poses on disk
    main():             load the model once, then run every unit
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths     # noqa: E402  every location comes from here, never a literal

sys.path.insert(0, str(paths.PROTENIX_SRC))
sys.path.insert(0, str(paths.CODE_DIR))
import runmeta   # noqa: E402

DC = os.path.dirname(os.path.realpath(__file__))   # scorer.py must be the copy beside this file
CLEAN_CODE = str(paths.CLEAN_CODE_DIR)   # the template builder still lives there
PY = str(paths.PROTENIX_PYTHON)

# mandatory: without it the model emits NaN coordinates silently
os.environ.setdefault("LAYERNORM_TYPE", "openfold")

# template flags that carry cross-chain geometry; shell-overridable for bisecting
TMPL_ENV = {k: os.environ.get(k, v) for k, v in {
    "PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
    "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATE_CANDIDATES": "20"}.items()}

KINDS = ("free", "tmpl")
ALIASES = {"all5": "tmpl"}          # old queue files spell tmpl as all5
_TEMPLATE_COUNTS = []


class ArmError(RuntimeError):
    """A unit failed in a way that must not be silently skipped."""


class _CountHandler(logging.Handler):
    """Protenix reports how many templates it actually used only to its logger, so capture it there.

    Without this the number lives in a log file and nowhere the code can assert on, which is exactly
    how 20 arms ran with 3 templates while the driver believed it had handed them 5.
    """

    def emit(self, record):
        try:
            m = record.getMessage()
        except Exception:
            return
        g = re.search(r"Found (\d+) templates", m)
        if g:
            _TEMPLATE_COUNTS.append(int(g.group(1)))


def parse_seeds(s):
    """in: "300020" or "300050-300099" or a comma mix.  out: list of ints, one per seed."""
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_runner(ckpt_dir, model_name, job_path, dump_dir, dtype):
    """in:  checkpoint dir, model name, a job json to shape the config, dump dir, dtype
    out: (runner, configs) reused for every unit in the queue

    Built once: the checkpoint is 464M parameters and takes 50-80s to load.
    """
    from collections.abc import Mapping

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, model_configs, update_gpu_compatible_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, model_configs[model_name])
    configs = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_dir = ckpt_dir
    configs.model_name = model_name
    configs.input_json_path = job_path
    configs.dump_dir = dump_dir
    configs.use_template = True
    if dtype is not None:
        configs.dtype = dtype
    configs = update_gpu_compatible_configs(configs)
    return InferenceRunner(configs), configs


def named_job(src, name, out_dir):
    """in:  source job json, a unique name, output dir
    out: path to a copy of the json named <name>.json

    Protenix derives its output directory from the job NAME, and configs.dump_dir is not re-read
    after the runner is built, so the name is the only per-fold control over where poses land.
    """
    obj = json.load(open(src))
    obj = obj if isinstance(obj, list) else [obj]
    for o in obj:
        o["name"] = name
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{name}.json"
    json.dump(obj, open(dst, "w"))
    return dst


def fold(runner, configs, job, seed, name, n_cycle, n_sample, expect_templates, dump_base, env):
    """in:  runner/configs, a named job json, seed, cycle and sample counts, expect_templates, env
    out: (seconds, template counts seen per chain); raises ArmError if the fold produced nothing

    expect_templates=None for a free unit (nothing to assert). For a tmpl unit it is the number of
    poses given, and EVERY chain must report that many -- a chain that quietly used fewer is the
    silent-wrong-template failure.
    """
    from runner.inference import infer_predict

    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, str(v))

    # base_dir is captured at construction but is a plain attribute, so set it per fold
    saved_base = getattr(runner.dumper, "base_dir", None)
    os.makedirs(dump_base, exist_ok=True)
    # Protenix writes to <dump>/ERR/ without creating it
    os.makedirs(f"{dump_base}/ERR", exist_ok=True)
    runner.dumper.base_dir = dump_base

    configs.input_json_path = job
    configs.seeds = [seed]
    configs.sample_diffusion.N_sample = n_sample
    runner.model.N_cycle = n_cycle          # predict() uses the N_cycle the model captured
    _TEMPLATE_COUNTS.clear()
    t0 = time.perf_counter()
    try:
        infer_predict(runner, configs)
    finally:
        dt = time.perf_counter() - t0
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        if saved_base is not None:
            runner.dumper.base_dir = saved_base

    if not _TEMPLATE_COUNTS:
        raise ArmError("no template count reported -- featurisation did not run")
    if expect_templates is not None:
        bad = [c for c in _TEMPLATE_COUNTS if c != expect_templates]
        if bad:
            raise ArmError(f"templates consumed {sorted(set(_TEMPLATE_COUNTS))}, expected "
                           f"{expect_templates} on every chain")
    n = len(glob.glob(f"{dump_base}/{name}/**/*_sample_*.cif", recursive=True))
    if n != n_sample:
        raise ArmError(f"produced {n}/{n_sample} poses")
    return dt, sorted(set(_TEMPLATE_COUNTS))


def read_queue(path):
    """in:  path to the queue file
    out: list of (kind, pdb, target, tag, seed, poses), one per fold

    A seed range expands to one unit per seed. `all5` is accepted as an old spelling of `tmpl`.
    """
    units = []
    for lineno, raw in enumerate(open(path), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        f = raw.rstrip("\n").split("\t")
        if len(f) < 5:
            raise SystemExit(f"{path}:{lineno}: need >=5 tab-separated fields, got {len(f)}")
        kind, pdb, target, tag, seedspec = f[:5]
        kind = ALIASES.get(kind, kind)
        poses = f[5].split() if len(f) > 5 and f[5].strip() else []
        if kind not in KINDS:
            raise SystemExit(f"{path}:{lineno}: kind must be one of {KINDS}, got {kind!r}")
        if kind == "free" and poses:
            raise SystemExit(f"{path}:{lineno}: free units take no poses")
        if kind != "free" and not poses:
            raise SystemExit(f"{path}:{lineno}: {kind} needs pose paths")
        for p in poses:
            if not os.path.isabs(p):
                raise SystemExit(f"{path}:{lineno}: pose paths must be absolute ({p})")
        for seed in parse_seeds(seedspec):
            units.append((kind, pdb, target, tag, seed, poses))
    if not units:
        raise SystemExit(f"{path}: no units")
    return units






def append_index(out, pdb, seed, run_dir, kind, precision, ncycle, templated, tmpl_md5=""):
    """in:  experiment dir and the facts about one finished fold
    out: appends a line to <out>/index.tsv

    A lookup table so runs are found by grep instead of walking the tree. run.json stays
    authoritative per fold, so this can be rebuilt if lost.
    """
    p = os.path.join(out, "index.tsv")
    new = not os.path.exists(p)
    with open(p, "a") as fh:
        if new:
            fh.write("pdb\tseed\trun_dir\tkind\tprecision\tn_cycle\ttemplated\ttmpl_md5\n")
        fh.write(f"{pdb}\t{seed}\t{run_dir}\t{kind}\t{precision}\t{ncycle}\t{templated}\t"
                 f"{tmpl_md5}\n")


def reconcile_claims(claim_dir, csv_path, pool_ifaces):
    """in:  claim dir, the results csv, {pdb: interfaces} to know what "finished" means
    out: removes claim dirs whose unit was never scored; returns nothing

    Runs at every startup. A claim is taken before a fold and released only on failure, so a worker
    killed mid-fold leaves one behind -- and a claimed-but-unscored unit is then skipped forever,
    silently shrinking the sweep.
    """
    if not claim_dir or not os.path.isdir(claim_dir):
        return 0
    scored = collections.defaultdict(set)
    for r in csv.DictReader(open(csv_path)):
        scored[(r["iface"].split("_")[0], r["seed"])].add(r["iface"])
    released = 0
    for c in os.listdir(claim_dir):
        parts = c.split("_")
        pdb, seed = parts[0], parts[-1]
        want = pool_ifaces.get(pdb)
        if want and not want <= scored.get((pdb, seed), set()):
            try:
                os.rmdir(f"{claim_dir}/{c}")
                released += 1
            except OSError:
                pass
    if released:
        print(f"  released {released} stale claims (killed mid-fold, now retryable)", flush=True)
    return released


def report(csv_path):
    """in: results csv.  out: prints one row per (interface, seed, arm) and returns 0.

    "selected" is the pose that arm's OWN confidence picks -- the number that matters. The oracle
    max is printed beside it only to show the headroom.
    """
    sys.path.insert(0, DC)
    from fmt import table
    import statistics as st
    arms = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        arms[(r["iface"], r["seed"], r["arm"])].append((float(r["iptm"]), float(r["dockq"])))
    rows = []
    for (iface, seed, arm), v in sorted(arms.items()):
        ip = [x for x, _ in v]
        dq = [d for _, d in v]
        # selected = the pose this arm's own confidence picks; max(dq) is the oracle, for headroom
        rows.append([iface, seed, arm, len(v), st.mean(ip), max(v)[1], max(dq)])
    print(table(["interface", "seed", "arm", "poses", "arm ipTM", "selected DockQ", "best DockQ"],
                rows, align="l l l r r r r",
                dec={"arm ipTM": 3, "selected DockQ": 3, "best DockQ": 3}))
    print(f"\n  arms scored: {len(rows)}")
    return 0


def verify(csv_path, out):
    """in:  results csv, experiment out dir
    out: prints both mismatches, returns 0 if they agree and 1 otherwise (usable as a gate)

    scored but absent    rows with no run directory -- reporting works, self-templating cannot
    present but unscored a run directory with no rows -- the driver will refold that seed into a
                         SECOND directory, since run ids are random
    Note --drop-cifs deletes poses after scoring, so "scored but absent" is expected for those runs.
    """
    if not os.path.exists(csv_path):
        print(f"  no csv at {csv_path}")
        return 1
    scored = collections.defaultdict(set)
    for r in csv.DictReader(open(csv_path)):
        scored[r["iface"].split("_")[0]].add(r["seed"])
    on_disk = collections.defaultdict(set)
    for d in glob.glob(f"{out}/*/*-*"):
        if os.path.isdir(f"{d}/predictions"):
            pdb = os.path.basename(os.path.dirname(d))
            # split on the FIRST hyphen: a run id may contain hyphens ("legacy-no-provenance")
            on_disk[pdb].add(os.path.basename(d).split("-", 1)[0])
    missing = sorted((p, s) for p in scored for s in scored[p] - on_disk.get(p, set()))
    extra = sorted((p, s) for p in on_disk for s in on_disk[p] - scored.get(p, set()))
    print(f"  csv {csv_path}")
    print(f"  out {out}")
    print(f"  complexes scored {len(scored)}   on disk {len(on_disk)}")
    print(f"  SCORED BUT NO STRUCTURE : {len(missing)}" + (f"  e.g. {missing[:4]}" if missing else ""))
    print(f"  STRUCTURE BUT NOT SCORED: {len(extra)}" + (f"  e.g. {extra[:4]}" if extra else ""))
    if not missing and not extra:
        print("  csv and filesystem agree")
        return 0
    return 1


class Ctx:
    """Everything a unit needs that is the same for every unit: args, model, and shared state."""

    def __init__(self, A, runner, configs, jobs, ref, scorer, ifaces, rows_done, units_done, mm, rd):
        self.A, self.runner, self.configs = A, runner, configs
        self.jobs, self.ref, self.scorer = jobs, ref, scorer
        self.ifaces = ifaces        # {pdb: [interface, ...]} -- what target=* scores
        self.rows_done = rows_done      # {(iface, seed, arm)}; mutated as rows land
        self.units_done = units_done    # {(pdb, seed, donor)} -> [pinned chain sets]
        self.mm, self.rd = mm, rd   # template namespace, or None for a free-only queue

    def point_templates_at(self, use_ns):
        """in: whether THIS unit is templated.  out: switches the template source.

        Per unit, not per run: a free unit must see Protenix's own PDB templates, or the baseline it
        provides is silently stripped of them.
        """
        m, r = (self.mm, self.rd) if use_ns else (self.stock_mm, self.stock_rd)
        self.configs.data.template.prot_template_mmcif_dir = m
        self.configs.data.template.release_dates_path = r
        for k, v in (("SELFTMPL_MMCIF_DIR", m), ("SELFTMPL_RELEASE_DATES", r)):
            if v is None or not use_ns:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def parse_args():
    """out: parsed arguments. Defaults point at the free pool, the one run with a canonical home."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue")
    ap.add_argument("--out", default=str(paths.FREE_DIR))
    ap.add_argument("--csv", default=f"{paths.FREE_DIR}/results.csv")
    ap.add_argument("--templates", help="namespace built by write_templates.py; required for "
                                        "templated units, never written to here")
    ap.add_argument("--pool", default=str(paths.PXM2025_POOL_CSV),
                    help="csv whose interfaces define what `target=*` scores")
    ap.add_argument("--claim-dir", default=f"{paths.FREE_DIR}/claims",
                    help="atomic mkdir claims so workers share one queue")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    ap.add_argument("--ncycle", type=int, default=10)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--ckpt-dir", default=str(paths.PROTENIX_CKPT))
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--keep-cifs", action="store_true", default=True,
                    help="poses are kept by default: a pose whose score is in the csv but whose "
                         "coordinates are gone cannot be templated on later, and folds do not "
                         "reproduce")
    ap.add_argument("--drop-cifs", dest="keep_cifs", action="store_false")
    ap.add_argument("--skip-by-template", action="store_true",
                    help="skip a unit already folded with the same (complex, seed, template). Needs a "
                         "csv whose tmpl_chains holds the FULL chain set; older runs recorded the "
                         "required pair and would refold")
    ap.add_argument("--force", action="store_true",
                    help="refold a unit already folded with the same complex, seed and template; "
                         "the new fold gets its own directory, nothing is overwritten")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check --csv against --out and exit; nothing is folded")
    return ap.parse_args()


def check_queue(A, units):
    """in: args and the parsed units.  out: (mmcif dir, release-dates path) or (None, None).

    Refuses three things rather than discovering them mid-sweep: templated units with no namespace,
    an incomplete namespace, and a templated queue aimed at the shared free pool -- whose poses would
    then become candidates for templating themselves.
    """
    templated = [u for u in units if u[0] != "free"]
    if templated and not A.templates:
        raise SystemExit(f"{len(templated)} templated units need --templates "
                         "(run write_templates.py first)")
    if templated and A.out == str(paths.FREE_DIR):
        raise SystemExit("refusing to write a templated queue into the free pool; pass --out")
    if not A.templates:
        return None, None
    mm, rd = f"{A.templates}/mmcif", f"{A.templates}/release_dates.json"
    for p in (mm, rd):
        if not os.path.exists(p):
            raise SystemExit(f"namespace incomplete: {p} missing -- run write_templates.py")
    return mm, rd


def interfaces_by_pdb(pool_csv):
    """in: the pool csv.  out: {pdb: {interfaces}} -- what a unit with target=* gets scored on."""
    out = collections.defaultdict(set)
    for r in csv.DictReader(open(pool_csv)):
        out[r["pdb"]].add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")
    return out


def open_csv(csv_path):
    """in: results csv path.  out: (rows_done, units_done); creates the file if absent.

    rows_done  {(iface, seed, arm)}        -- stops a retried unit appending duplicate rows
    units_done {(pdb, seed, donor)} -> [set of pinned chains]
               identity by CONTENT: the same complex, seed and template is the same fold whatever
               tag it was given. Keying on the tag instead is what once let two interfaces of one
               complex swap donors.
    """
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n")
    rows_done, units_done = set(), collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        rows_done.add((r["iface"], r["seed"], r["arm"]))
        pdb = r["iface"].split("_")[0]
        chains = frozenset(c for grp in r.get("tmpl_chains", "").split() for c in grp.split("+") if c)
        units_done[(pdb, r["seed"], r.get("donor_pose", "-1"))].append(chains)
    return rows_done, units_done


def unit_already_folded(units_done, pdb, seed, donor, chains):
    """in: the unit's identity.  out: True if this complex+seed+template was already folded.

    EXACT set equality. A template is the full set of chains written into it, so two folds are the
    same fold only if those sets match. A subset test looks tempting -- older csvs record the required
    PAIR rather than the full set -- but it can declare two different templates identical and skip a
    fold that never ran, which is the one error worth avoiding. Rows in the old pair form simply do
    not match, so they are refolded rather than wrongly skipped.
    """
    return chains in set(units_done.get((pdb, str(seed), donor), ()))


def provenance(A, target, tag, poses):
    """in: the unit.  out: (donor_pose, tmpl_chains) for the csv.

    A fold is identified by (pose, seed, pinned chains). The tag alone once let two interfaces of one
    complex swap donors, and these two fields are what make reuse safe.
    """
    if not poses:
        return "-1", ""
    chains = ""
    chosen = f"{A.templates}/jobs/{target}_{tag}/chosen.json" if A.templates else None
    if chosen and os.path.exists(chosen):
        with open(chosen) as fh:
            chains = " ".join("+".join(c.get("chains", [])) for c in json.load(fh))
    return " ".join(poses), chains


def score_fold(ctx, pdb, target, tag, seed, rdir, name, want, poses):
    """in: a finished fold and the interfaces to score it on.
    out: (rows written for how many interfaces, last stderr) -- appends to the csv as it goes.

    Only interfaces MISSING from the csv are scored: a retried unit would otherwise append a second
    row set and quietly double-weight that seed in every average.
    """
    donor, chains = provenance(ctx.A, target, tag, poses)
    got, err = 0, ""
    for iface in [i for i in want if (i, str(seed), tag) not in ctx.rows_done]:
        out = subprocess.run([PY, f"{DC}/scorer.py", "--dir", f"{rdir}/{name}", "--iface", iface,
                              "--seed", str(seed), "--arm", tag, "--donor", donor,
                              "--chains", chains,
                              "--ref", ctx.ref(pdb), "--scorer", ctx.scorer],
                             cwd=DC, capture_output=True, text=True)
        err = out.stderr or err
        if out.stdout.strip():
            with open(ctx.A.csv, "a") as fh:      # flushed per interface, not at the end
                fh.write(out.stdout)
            ctx.rows_done.add((iface, str(seed), tag))
            got += 1
    return got, err


def run_unit(ctx, unit):
    """in: one (kind, pdb, target, tag, seed, poses).  out: "ran", "skipped" or "failed".

    Claims the unit, folds it, scores EVERY interface of its complex -- the fold produces the whole
    complex whatever it was aimed at -- and releases the claim if anything fails, since a claimed but
    unscored unit is skipped forever.
    """
    kind, pdb, target, tag, seed, poses = unit
    A = ctx.A
    want = sorted(ctx.ifaces.get(pdb, ()))
    if not want:
        print(f"  {pdb}: no interfaces to score, skipping", flush=True)
        return "skipped"
    donor, chains = provenance(A, target, tag, poses)
    chain_set = frozenset(c for grp in chains.split() for c in grp.split("+") if c)
    # identity is (complex, seed, template), not the tag; opt-in because a csv that records the
    # required pair instead of the full chain set will not match and would refold
    if A.skip_by_template and not A.force \
            and unit_already_folded(ctx.units_done, pdb, seed, donor, chain_set):
        return "skipped"
    if all((i, str(seed), tag) in ctx.rows_done for i in want):
        return "skipped"

    claim = f"{A.claim_dir}/{pdb}_{target}_{tag}_{seed}".replace("*", "all") if A.claim_dir else None
    if claim:
        try:
            os.mkdir(claim)
        except FileExistsError:
            return "skipped"            # another worker has it

    def release():
        if claim:
            try:
                os.rmdir(claim)
            except OSError:
                pass

    ctx.point_templates_at(kind != "free")
    if kind == "free":
        job_src, env, expect = f"{ctx.jobs}/{pdb}-final-updated.json", {}, None
    else:
        job_src = f"{A.templates}/jobs/{target}_{tag}/job.json"
        env = dict(TMPL_ENV, PROTENIX_MAX_TEMPLATES=str(len(poses)))
        expect = len(poses)
    if not os.path.exists(job_src):
        print(f"  {target} {tag}: no job at {job_src}", flush=True)
        release()
        return "failed"

    rdir = runmeta.run_dir(A.out, pdb, seed, runmeta.run_id())
    name = f"{target.replace('*', pdb)}_{tag}"
    try:
        dt, tcounts = fold(ctx.runner, ctx.configs, named_job(job_src, name, f"{rdir}/jobs"),
                           int(seed), name, A.ncycle, A.nsample, expect, rdir, env)
    except Exception as e:
        print(f"  {pdb} {target} {tag} seed {seed}: FAILED -- {type(e).__name__}: {str(e)[:120]}",
              flush=True)
        release()
        return "failed"

    tmpl_md5 = "+".join(runmeta_md5(p) for p in poses) if poses else ""
    runmeta.write_run_json(
        rdir, kind=kind, pdb=pdb,
        # motivating_interface = which interface chose this pose; interfaces_scored = what was scored
        motivating_interface=None if target == "*" else target,
        seed=seed, arm=tag, precision=A.dtype, n_cycle=A.ncycle, n_sample=A.nsample,
        fold_seconds=round(dt, 1), interfaces_scored=want,
        templates=[{"source_pose": p, "md5": m}
                   for p, m in zip(poses, tmpl_md5.split("+"))] or None,
        templates_found=tcounts, template_namespace=A.templates,
        templated=bool(poses), run_dir=os.path.basename(rdir))
    append_index(A.out, pdb, seed, os.path.basename(rdir), kind, A.dtype, A.ncycle, bool(poses),
                 tmpl_md5)

    got, err = score_fold(ctx, pdb, target, tag, seed, rdir, name, want, poses)
    result = "ran"
    if got or all((i, str(seed), tag) in ctx.rows_done for i in want):
        # print pdb too: target is "*" for a free fold
        print(f"  {kind} {pdb} {target} {tag} seed {seed}: {got}/{len(want)} interfaces "
              f"in {dt:.0f}s", flush=True)
    else:
        print(f"  {pdb} {target} {tag} seed {seed}: SCORING produced no rows", flush=True)
        for line in (err or "").strip().splitlines()[-3:]:
            print(f"      {line}", flush=True)
        release()
        result = "failed"
    if not A.keep_cifs:
        shutil.rmtree(f"{rdir}/{name}", ignore_errors=True)
    return result


def main():
    """Load the model once, then run every unit in the queue."""
    A = parse_args()
    if A.report:
        return report(A.csv)
    if A.verify:
        return verify(A.csv, A.out)
    for req in ("queue", "out"):
        if not getattr(A, req):
            raise SystemExit(f"--{req} is required")

    units = read_queue(A.queue)
    jobs, ref, scorer = paths.dataset(A.dataset)
    mm, rd = check_queue(A, units)
    ifaces = interfaces_by_pdb(A.pool)

    os.makedirs(A.out, exist_ok=True)
    rows_done, units_done = open_csv(A.csv)
    if A.claim_dir:
        os.makedirs(A.claim_dir, exist_ok=True)
        reconcile_claims(A.claim_dir, A.csv, ifaces)

    # runner.error_dir is derived from configs.dump_dir at build time, not per fold
    os.makedirs(f"{A.out}/ERR", exist_ok=True)
    t_load = time.time()
    runner, configs = build_runner(A.ckpt_dir, A.model_name,
                                   f"{jobs}/{units[0][1]}-final-updated.json", A.out, A.dtype)
    ctx = Ctx(A, runner, configs, jobs, ref, scorer, ifaces, rows_done, units_done, mm, rd)
    ctx.stock_mm = configs.data.template.prot_template_mmcif_dir
    ctx.stock_rd = configs.data.template.release_dates_path
    logging.getLogger().addHandler(_CountHandler())
    print(f"  model loaded in {time.time()-t_load:.0f}s | dtype={A.dtype} | {len(units)} units "
          f"| templates={A.templates}", flush=True)

    tally = collections.Counter(run_unit(ctx, u) for u in units)
    print(f"queue complete: {tally['ran']} folds run, {tally['skipped']} already scored, "
          f"{tally['failed']} failed", flush=True)
    return 0


def runmeta_md5(path, n=12):
    """in: a pose path.  out: first n chars of its md5, so run.json pins the exact file used."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:n]
    except OSError:
        return "missing"


if __name__ == "__main__":
    sys.exit(main())
