#!/usr/bin/env python
"""Fold a queue of units with the checkpoint loaded once, score each, append rows to a csv.

    folder.py --queue q.tsv --out <dir> --csv <dir>/results.csv [--templates <ns>]
    folder.py --report --csv <results.csv>      summarise what has been scored
    folder.py --verify --csv <csv> --out <dir>  cross-check the csv against files on disk

SUMMARY
    The one driver for every kind of fold. The checkpoint is 464M parameters and takes 50-80s to load,
    so one long-lived process works through many units rather than one process per fold. Several
    processes can share a queue: each claims a unit by creating a directory before folding it.

    TWO THINGS THAT WILL BITE YOU
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
    --out:        where folds land. Default the dataset's free pool.
    --csv:        results csv, and the to-do list: an (iface, seed, arm) already present is skipped.
                  Default <out>/results.csv.
    --claim-dir:  atomic mkdir claims so several workers share one queue. Default <out>/claims.
    --pool:       csv whose interfaces define what every fold is scored on. Default the dataset's.
    --dataset:    pxm2025 or foldbench (paths.DATASETS). Default pxm2025. Picks the job jsons, the
                  reference cifs, the scorer, the pool csv and the free pool.
    --ncycle:     recycles per fold. Default 10.
    --nsample:    poses per fold. Default 5.
    --expect-poses: cifs a fold must write, when that differs from --nsample. Modes that decode
                  more than once per fold (PROTENIX_POSE_Z_BRANCH re-decodes every pose) write
                  N_sample + (N_sample+1)*M cifs from one fold; without this the pose-count check
                  would reject every such fold as short.
    --dtype:      bf16, fp32 or fp16. Default bf16.
    --keep-cifs:  keep the poses. On by default; --drop-cifs deletes them after scoring.
    --skip-by-template: skip a unit already folded with the same (complex, seed, template). Off by
                  default, because a csv that records the required pair rather than the full chain set
                  will not match and would refold everything.
    --overlap-scoring: score each unit on one background thread while the next folds, instead of
                  between folds (~13% of wall time on 2026-09-25). At most one unit is being scored
                  at a time and its claim is held until scoring ends. Off by default.
    --report:     print what has been scored and exit.
    --verify:     cross-check the csv against the poses on disk and exit.

OUTPUTS
    <out>/<pdb>/<seed>-<run_id>/predictions/   the poses, as <pdb>_sample_N.cif with their confidence
                                               jsons. One layout for every run, free or templated, and
                                               the same shape as the original pool. run_id is random
                                               because folds do not reproduce.
    <out>/<pdb>/<seed>-<run_id>/run.json       per-fold provenance
    <out>/index.tsv                            one line per fold, greppable, with the template md5
    <csv>                                      one row per scored pose:

        iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm

    `arm` is the tag: a label. `donor_pose` and `tmpl_chains` are the provenance -- a templated fold is
    identified by (pose, seed, pinned chains). The tag alone used to be the only link back to the
    template, and when two interfaces of one complex both used tag K00 they silently swapped donors.

EXAMPLE
    folder.py --queue q.tsv --out experimental_data/free
    folder.py --queue exp/queue.tsv --out exp/poses --csv exp/results.csv \
              --templates exp/templates --claim-dir exp/claims
    folder.py --verify --csv exp/results.csv --out exp/poses

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
    Ctx.point_templates_at(): per unit, the self-template namespace or Protenix's own PDB templates
    run_unit():         claim one unit and fold it; score it now, or hand finish_unit() to main()
    finish_unit():      score a folded unit, move its poses, release the claim if scoring fails
    run_queue():        every unit in order, scoring in the background with --overlap-scoring
    named_job():        copy a unit's job json under a unique name, which is what fixes its output dir
    fold():             run one fold; assert it produced poses and the expected template count
    provenance():       the unit -> (donor_pose, tmpl_chains), the two fields that identify a fold
    score_fold():       run scorer.py per interface and append its rows to the csv
    spawn():            start a scorer, retrying a launch that fails with OSError (nothing ran)
    normalise_poses():  flatten Protenix's nested output to <run dir>/predictions/, never overwriting
    _move_new():        os.replace that refuses an existing destination
    reconcile_claims(): at startup, free claims whose unit was never scored (a killed worker's work)
    claim_is_live():    whether a claim's owning worker is still running (_pid_alive() checks the pid)
    append_index():     one line per fold in index.tsv
    runmeta_md5()       checksum a pose, so run.json pins the exact file templated on
    report():           print what has been scored, per (interface, seed, arm)
    verify():           cross-check the csv against the poses on disk
    main():             load the model once, then run every unit
"""
import argparse
import collections
import concurrent.futures
import csv
import functools
import glob
import hashlib
import json
import math
import logging
import os
import re
import shutil
import subprocess
import socket
import sys
import time

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths     # noqa: E402  every location comes from here, never a literal

sys.path.insert(0, str(paths.PROTENIX_SRC))
# runmeta IS THIS WORKER'S OWN HELPER and now sits beside this file, so it is imported from here
# rather than through paths.CODE_DIR. It used to live in diffusion_commit/, which made that whole
# tree a runtime dependency of every fold -- archiving it would have killed the next worker relaunch
# with an ImportError while the already-running workers carried on, which is the worst shape a
# breakage can take.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runmeta   # noqa: E402

DC = os.path.dirname(os.path.realpath(__file__))   # scorer.py must be the copy beside this file
PY = str(paths.PROTENIX_PYTHON)

# With --overlap-scoring the scorer is started from a background thread while the main thread folds.
# Started with vfork (Python's default), 14 of the variable-seed arm's first 787 folds failed to launch
# it with OSError EFAULT "Bad address"; fork + exec is the documented way to opt out of vfork.
subprocess._USE_VFORK = False
SPAWN_TRIES = 3     # a scorer that fails to START wrote nothing, so launching it again is safe

# mandatory: without it the model emits NaN coordinates silently
os.environ.setdefault("LAYERNORM_TYPE", "openfold")

# template flags that carry cross-chain geometry; shell-overridable for bisecting
TMPL_ENV = {k: os.environ.get(k, v) for k, v in {
    "PROTENIX_MULTICHAIN_MASK": "0", "PROTENIX_TMPL_KEEP_FRAME": "1",
    "PROTENIX_TMPL_NODEDUP": "1", "PROTENIX_MAX_TEMPLATE_CANDIDATES": "20"}.items()}

KINDS = ("free", "tmpl", "posez")   # posez: free job + ONE pose injected into z (not a template)
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
    # PROTENIX_MC_DROPOUT_APPLY_RATE=<p>: probability that a fold's recycle carry-over gets MC dropout
    # (config default 0.4). 0 = never. Unset = the config's value, i.e. unchanged.
    _mc = os.environ.get("PROTENIX_MC_DROPOUT_APPLY_RATE")
    if _mc is not None:
        configs.mc_dropout_apply_rate = float(_mc)
    configs = update_gpu_compatible_configs(configs)
    return InferenceRunner(configs), configs


def named_job(src, name, out_dir, templates=None):
    """in:  source job json, a unique name, output dir, and optionally a template file for every chain
    out: path to a copy of the json named <name>.json

    Protenix derives its output directory from the job NAME, and configs.dump_dir is not re-read
    after the runner is built, so the name is the only per-fold control over where poses land.
    `templates` replaces every protein chain's templatesPath -- an empty file makes a template-free
    fold (see Ctx.no_templates).
    """
    obj = json.load(open(src))
    obj = obj if isinstance(obj, list) else [obj]
    for o in obj:
        o["name"] = name
        if templates is not None:
            for e in o.get("sequences", []):
                if "proteinChain" in e:
                    e["proteinChain"]["templatesPath"] = templates
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{name}.json"
    json.dump(obj, open(dst, "w"))
    return dst


def fold(runner, configs, job, seed, name, n_cycle, n_sample, expect_templates, dump_base, env,
         expect_poses=None):
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
    want = n_sample if expect_poses is None else expect_poses
    if n != want:
        raise ArmError(f"produced {n}/{want} poses")
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
        if kind == "posez" and len(poses) != 1:
            raise SystemExit(f"{path}:{lineno}: posez takes exactly one pose, got {len(poses)}")
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


def _pid_alive(pid):
    """in: a pid.  out: whether a process with that pid exists"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def claim_is_live(path):
    """in: a claim directory.  out: whether a running worker still holds it

    An owner file names the host and pid that took the claim. Same host and the pid alive means the
    fold is in progress and must not be handed to another worker. A claim from a DIFFERENT host cannot
    be checked, so it is left alone rather than stolen. No owner file means the claim predates this
    bookkeeping: treat it as live for an hour, then releasable.
    """
    f = f"{path}/owner"
    if not os.path.exists(f):
        try:
            return (time.time() - os.path.getmtime(path)) < 3600
        except OSError:
            return False
    try:
        host, pid = open(f).read().split()
        if host != socket.gethostname():
            return True
        return _pid_alive(int(pid))
    except (ValueError, OSError):
        return True          # unreadable owner: do not steal on a guess


def reconcile_claims(claim_dir, csv_path, pool_ifaces, nsample=5):
    """in:  claim dir, the results csv, {pdb: interfaces} to know what "finished" means
    out: removes claim dirs whose unit was never scored; returns nothing

    Runs at every startup. A claim is taken before a fold and released only on failure, so a worker
    killed mid-fold leaves one behind -- and a claimed-but-unscored unit is then skipped forever,
    silently shrinking the sweep.

    A LIVE worker's in-flight claim looks exactly like an abandoned one: unscored either way. Releasing
    those meant every new worker stole a fold from a running one and folded it a second time -- two
    processes, same unit, same output directory. So each claim now records its owner, and a claim is
    released only when that owner is gone. Claims written before this existed carry no owner, and are
    given the benefit of the doubt for an hour so a restart cannot steal a fold already in progress.
    """
    if not claim_dir or not os.path.isdir(claim_dir):
        return 0
    # KEYED BY TAG as well as seed. Claims are "<pdb>_<target>_<tag>_<seed>" and the csv's arm column is
    # the tag. Keying on (pdb, seed) alone works for free folds, whose seeds are unique, but every
    # templated fold shares one seed -- so one scored fold made every dead claim of its complex look
    # scored, and a fold killed mid-run was skipped forever (4 stranded in the 2026-09-25 arm).
    # PER SAMPLE, NOT PER INTERFACE. This recorded only WHICH interfaces appear, so a dead claim
    # whose scoring died after one pose looked finished and was retained forever -- the same
    # one-row-means-done fault that iface_scored fixed on the folding side, left in place on the
    # recovery side, which is precisely where a partial unit has to be noticed.
    per_sample = collections.defaultdict(set)
    for r in csv.DictReader(open(csv_path)):
        try:
            k = int(r["sample"])
            if not (math.isfinite(float(r["dockq"])) and math.isfinite(float(r["iptm"]))):
                continue
        except (TypeError, ValueError, KeyError):
            continue
        per_sample[(r["iface"].split("_")[0], r["arm"], r["seed"], r["iface"])].add(k)

    def finished(pdb, tag, seed, want):
        return all(per_sample.get((pdb, tag, seed, i), set()) >= set(range(nsample))
                   for i in want)

    released = 0
    for c in os.listdir(claim_dir):
        parts = c.split("_")
        pdb, tag, seed = parts[0], parts[-2], parts[-1]
        want = pool_ifaces.get(pdb)
        if want and not finished(pdb, tag, seed, want):
            if claim_is_live(f"{claim_dir}/{c}"):
                continue
            try:
                shutil.rmtree(f"{claim_dir}/{c}")
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
        arms[(r["iface"], r["seed"], r["arm"])].append(
            (float(r["iptm"]), int(r["sample"]), float(r["dockq"])))
    rows = []
    for (iface, seed, arm), v in sorted(arms.items()):
        ip = [x for x, _, _ in v]
        dq = [d for _, _, d in v]
        # THE TIE RULE IS POSE IDENTITY, NEVER DOCKQ. `max(v)[1]` over (iptm, dockq) tuples made an
        # ipTM tie select the higher GROUND TRUTH, so the printed "selected" was part oracle. Ties
        # are not hypothetical: the committed FoldBench table has 274 (interface, seed) groups whose
        # top pair_iptm is tied across different DockQ, PXM2025 has 598, and the scorer rounds
        # confidence to four decimals, which is what manufactures them. Breaking the tie on the
        # lowest sample index is deterministic and cannot see the label. This matches
        # three_classes() in extra_training/analysis/train_seed_depth_curve.py, which is what every
        # REPORTED number uses and which already keyed on identity -- so this printer was the only
        # place the oracle rule survived, and no published figure came from it.
        best = min(v, key=lambda t: (-t[0], t[1]))
        rows.append([iface, seed, arm, len(v), st.mean(ip), best[2], max(dq)])
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
        self.free_templates = True  # main() sets the dataset's; see point_templates_at
        self.no_templates = None    # an empty template file, for free units when free_templates is off

    def point_templates_at(self, use_ns):
        """in: whether THIS unit is templated.  out: switches the template source.

        Per unit, not per run: a free unit must see Protenix's own PDB templates where the dataset's free
        pool did (pxm2025). Where it did not (FoldBench; paths.Dataset.free_templates), run_unit also
        hands the free unit an empty template file, so it finds none, exactly like the pool.
        """
        m, r = (self.mm, self.rd) if use_ns else (self.stock_mm, self.stock_rd)
        self.configs.data.template.prot_template_mmcif_dir = m
        self.configs.data.template.release_dates_path = r


def parse_args():
    """in: the command line.  out: parsed arguments; defaults point at the dataset's free pool, the one
    run with a canonical home"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--templates", help="namespace built by build_templates.py; required for "
                                        "templated units, never written to here")
    ap.add_argument("--pool", default=None,
                    help="csv whose interfaces define what every fold is scored on")
    ap.add_argument("--claim-dir", default=None,
                    help="atomic mkdir claims so workers share one queue")
    ap.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    ap.add_argument("--ncycle", type=int, default=10)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--expect-poses", type=int, default=None,
                    help="cifs a fold must write when it differs from --nsample")
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
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
    ap.add_argument("--overlap-scoring", action="store_true",
                    help="score each unit on a background thread while the next one folds")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check --csv against --out and exit; nothing is folded")
    A = ap.parse_args()
    ds = paths.DATASETS[A.dataset]
    A.out = A.out or str(ds.free)
    A.csv = A.csv or f"{A.out}/results.csv"
    A.claim_dir = A.claim_dir or f"{A.out}/claims"
    A.pool = A.pool or str(ds.pool_csv)
    return A


def check_queue(A, units):
    """in: args and the parsed units.  out: (mmcif dir, release-dates path) or (None, None).

    Refuses three things rather than discovering them mid-sweep: templated units with no namespace,
    an incomplete namespace, and a templated queue aimed at the shared free pool -- whose poses would
    then become candidates for templating themselves.
    """
    templated = [u for u in units if u[0] not in ("free", "posez")]
    if templated and not A.templates:
        raise SystemExit(f"{len(templated)} templated units need --templates "
                         "(run build_templates.py first)")
    if templated and A.out == str(paths.DATASETS[A.dataset].free):
        raise SystemExit("refusing to write a templated queue into the free pool; pass --out")
    if not A.templates:
        return None, None
    mm, rd = f"{A.templates}/mmcif", f"{A.templates}/release_dates.json"
    for p in (mm, rd):
        if not os.path.exists(p):
            raise SystemExit(f"namespace incomplete: {p} missing -- run build_templates.py")
    return mm, rd


def interfaces_by_pdb(pool_csv):
    """in: a csv naming the interfaces.  out: {pdb: {interfaces}} -- what target=* gets scored on.

    TWO SCHEMAS, because this is an interface REGISTRY and the file that happens to carry it differs
    per dataset. Our folds point it at their pool.csv, which is one row per (interface, seed, sample)
    and spells the columns chain1/chain2. FoldBench points it at chain_roles.csv, which is one row
    per interface -- 160 against 39,998 -- and spells them chain_1/chain_2. Reading only the chain
    columns, either spelling, means the registry can live in whichever file actually is one rather
    than in whichever file happened to be lying around.

    Scores are not read here and never were: the old caller passed a results table and this function
    ignored everything but the first three columns, which is why FoldBench's 50-seed results could
    not be deleted without breaking folding.
    """
    out = collections.defaultdict(set)
    for r in csv.DictReader(open(pool_csv)):
        c1 = r.get("chain1", r.get("chain_1"))
        c2 = r.get("chain2", r.get("chain_2"))
        if c1 is None or c2 is None:
            raise SystemExit(f"STOP: {pool_csv} has neither chain1/chain2 nor chain_1/chain_2; "
                             f"columns are {list(r)}")
        out[r["pdb"]].add(f"{r['pdb']}_{c1}{c2}")
    return out


def open_csv(csv_path):
    """in: results csv path.  out: (rows_done, units_done); creates the file if absent.

    rows_done  {(iface, seed, arm)} -> {sample index}   -- which POSES of that arm are scored
    units_done {(pdb, seed, donor)} -> [set of pinned chains]
               identity by CONTENT: the same complex, seed and template is the same fold whatever
               tag it was given. Keying on the tag instead is what once let two interfaces of one
               complex swap donors.

    COMPLETION IS PER SAMPLE, NOT PER ARM. This used to be a set of (iface, seed, arm), so ANY one
    row marked that arm finished: a unit whose scoring died after sample 0 was skipped forever and
    restarting a worker could not repair it. The depth curves would then silently average a 1-pose
    draw against 5-pose draws. A row with a non-finite or unparseable score is not a score, so it is
    dropped from the completion state and the sample is scored again.
    """
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n")
    rows_done, units_done = collections.defaultdict(set), collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        try:
            dq, ip, k = float(r["dockq"]), float(r["iptm"]), int(r["sample"])
        except (TypeError, ValueError, KeyError):
            continue                                  # not a score; the sample is still to do
        if not (math.isfinite(dq) and math.isfinite(ip)):
            continue
        rows_done[(r["iface"], r["seed"], r["arm"])].add(k)
        pdb = r["iface"].split("_")[0]
        chains = frozenset(c for grp in r.get("tmpl_chains", "").split() for c in grp.split("+") if c)
        units_done[(pdb, r["seed"], r.get("donor_pose", "-1"))].append(chains)
    return rows_done, units_done


def iface_scored(rows_done, iface, seed, arm, n_sample):
    """True only when EVERY expected pose of this interface/seed/arm carries a finite score.

    The expected set is explicit -- {0 .. n_sample-1} -- so a missing middle sample is as visible as
    a missing last one.
    """
    return rows_done.get((iface, str(seed), arm), set()) >= set(range(n_sample))


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
    for iface in [i for i in want
                  if not iface_scored(ctx.rows_done, i, seed, tag, ctx.A.nsample)]:
        cmd = [PY, f"{DC}/scorer.py", "--dir", f"{rdir}/{name}", "--iface", iface, "--seed", str(seed),
               "--arm", tag, "--donor", donor, "--chains", chains, "--ref", ctx.ref(pdb),
               "--scorer", ctx.scorer]
        out = spawn(cmd)
        err = out.stderr or err
        if out.stdout.strip():
            key = (iface, str(seed), tag)
            have = ctx.rows_done[key]
            # ONLY THE MISSING SAMPLES. The scorer always scores the whole interface, so a retry of a
            # partly-scored unit re-emitted rows for samples the csv already had -- duplicating them
            # and double-weighting that pose in every average. Rows are filtered here, by sample, and
            # a row whose scores are not finite numbers is not a score and is dropped.
            fresh = []
            for line in out.stdout.strip().splitlines():
                f = line.split(",")
                if len(f) < 8 or not f[5].strip().isdigit():
                    continue
                k = int(f[5])
                if k in have:
                    continue
                try:
                    if not (math.isfinite(float(f[6])) and math.isfinite(float(f[7]))):
                        continue
                except ValueError:
                    continue
                fresh.append((k, line))
            if fresh:
                with open(ctx.A.csv, "a") as fh:  # flushed per interface, not at the end
                    fh.write("\n".join(l for _k, l in fresh) + "\n")
                for k, _l in fresh:
                    have.add(k)
                got += 1
            elif have:
                print(f"    {iface} seed {seed}: already scored, nothing appended", flush=True)
    return got, err


def spawn(cmd):
    """in: a scorer command.  out: its CompletedProcess -- the launch retried if it fails to start.

    Only an OSError from starting the process is retried: the scorer never ran, so nothing was written.
    """
    for attempt in range(1, SPAWN_TRIES + 1):
        try:
            return subprocess.run(cmd, cwd=DC, capture_output=True, text=True)
        except OSError as e:
            if attempt == SPAWN_TRIES:
                raise
            print(f"      scorer failed to start ({e}); retry {attempt}/{SPAWN_TRIES - 1}", flush=True)
            time.sleep(2)


def saved_run_dir(out, pdb, seed, tag=None, donor=None):
    """The ONE finished run directory of the REQUESTED unit under --out, or None.

    Finished means it holds run.json AND some poses. A worker that died mid-fold leaves a directory
    without run.json, and a seed can have several, so anything but exactly one match is no answer.

    MATCHED ON THE UNIT, NOT JUST (pdb, seed). Keying on the complex and seed alone recovered a
    saved run belonging to a DIFFERENT arm: every templated fold of a complex shares one seed, so a
    run dumped for arm "br03" satisfied a request for "free". run.json records arm and donor_pose,
    so the requested arm and template are required to match when they are known.
    """
    hits = []
    for d in sorted(glob.glob(f"{out}/{pdb}/{seed}-*")):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        if not glob.glob(f"{d}/**/*_sample_*.cif", recursive=True):
            continue
        try:
            with open(rj) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(meta.get("seed")) != str(seed):
            continue                       # a renamed directory must not be able to lie about it
        if tag is not None and meta.get("arm") not in (None, tag):
            continue
        if donor is not None and str(meta.get("donor_pose", donor)) != str(donor):
            continue
        hits.append(d)
    if len(hits) > 1:
        print(f"  {pdb} seed {seed} {tag}: {len(hits)} saved runs match; refusing to guess "
              f"({[os.path.basename(h) for h in hits]})", flush=True)
        return None
    return hits[0] if hits else None


def rescore_saved(ctx, pdb, target, tag, seed, want, rd=None, poses=()):
    """Score the POSES ALREADY ON DISK for an incomplete unit. Returns interfaces newly scored.

    WHY THIS EXISTS. A unit whose scoring died partway cannot be repaired by folding again: the
    trunk guard refuses ("trunk file already exists"), and lifting that guard would be worse --
    folds are not reproducible on this hardware, so a second fold of the same seed is a DIFFERENT
    pose, and its scores joined to the first run's trunk would be an identity error. The poses from
    the first run are still there, so the fix is to score those. score_fold already appends only the
    samples the csv is missing, so this is safe to run repeatedly.
    """
    if rd is None:
        rd = saved_run_dir(ctx.A.out, pdb, seed, tag=tag)
    if rd is None:
        return None
    # THE UNIT'S OWN PROVENANCE. Passing [] recorded donor_pose -1 and no template chains for every
    # recovered unit, so a templated fold's rows claimed to be free folds.
    got, err = score_fold(ctx, pdb, target, tag, seed, rd, ".", want, list(poses))
    if got:
        print(f"  RESCORED from saved poses {pdb} {tag} seed {seed}: {got} interface(s) "
              f"recovered from {os.path.basename(rd)}", flush=True)
    elif err:
        print(f"  rescore of {pdb} {tag} seed {seed} produced nothing: "
              f"{(err or '').strip().splitlines()[-1][:120] if err else ''}", flush=True)
    return got


def _move_new(src, dst):
    """in: source and destination paths.  out: src moved to dst; raises if dst already exists.

    os.replace overwrites silently, which is how one fold's poses replaced another's unnoticed.
    """
    if os.path.exists(dst):
        raise FileExistsError(f"refusing to overwrite an existing pose file: {dst}")
    os.replace(src, dst)


def normalise_poses(rdir, name, pdb):
    """in: a finished fold's directory, its job name, the complex.
    out: poses moved to <rdir>/predictions/<pdb>_sample_N.cif; returns how many moved

    Protenix nests its output as <rdir>/<jobname>/seed_<seed>/predictions/<jobname>_sample_N.cif, which
    is deeper than the pool's <rdir>/predictions/<pdb>_sample_N.cif and carries the job name in the
    filename. Two layouts meant every reader needed to know which run wrote a pose; flattening to the
    pool's shape here means one layout everywhere, free and templated alike.
    """
    dest = f"{rdir}/predictions"
    os.makedirs(dest, exist_ok=True)
    moved = 0
    for src in sorted(glob.glob(f"{rdir}/{name}/**/*_sample_*.cif", recursive=True)):
        n = os.path.basename(src).split("_sample_")[1]
        _move_new(src, f"{dest}/{pdb}_sample_{n}")
        moved += 1
    for src in sorted(glob.glob(f"{rdir}/{name}/**/*summary_confidence_sample_*.json",
                                recursive=True)):
        n = os.path.basename(src).split("_sample_")[1]
        _move_new(src, f"{dest}/{pdb}_summary_confidence_sample_{n}")
    # the scorer's own json output lives in the job directory too, and is NOT a pose; move it rather
    # than losing it when the job directory goes
    # A unit scores every interface of its complex, and each job writes its own *_json directory, so the
    # second one to be lifted collides with the first. os.replace cannot overwrite a NON-EMPTY directory
    # (Errno 39), which killed the worker outright rather than failing one fold. Merge file-by-file
    # instead, keeping the job name so two interfaces' scorer output stay distinguishable.
    for sub in sorted(glob.glob(f"{rdir}/{name}/*_json")):
        dst = f"{rdir}/{os.path.basename(sub)}"
        if not os.path.exists(dst):
            os.replace(sub, dst)
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(sub)):
            tgt = f"{dst}/{f}"
            if os.path.exists(tgt):
                tgt = f"{dst}/{name}_{f}"
            os.replace(f"{sub}/{f}", tgt)
        try:
            os.rmdir(sub)
        except OSError:
            pass
    return moved


def run_unit(ctx, unit):
    """in: one (kind, pdb, target, tag, seed, poses).  out: "ran", "skipped" or "failed" -- or, with
    --overlap-scoring, the finish_unit() call for main() to run in the background

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
    if A.skip_by_template and unit_already_folded(ctx.units_done, pdb, seed, donor, chain_set):
        return "skipped"
    if all(iface_scored(ctx.rows_done, i, seed, tag, A.nsample) for i in want):
        return "skipped"
    claim = f"{A.claim_dir}/{pdb}_{target}_{tag}_{seed}".replace("*", "all") if A.claim_dir else None
    if claim:
        try:
            os.mkdir(claim)
            # name the owner so another worker's reconcile_claims can tell an in-flight fold from an
            # abandoned one, instead of releasing both and folding this unit twice
            with open(f"{claim}/owner", "w") as fh:
                fh.write(f"{socket.gethostname()} {os.getpid()}")
        except FileExistsError:
            return "skipped"            # another worker has it

    def release():
        if claim:
            try:
                shutil.rmtree(claim)      # the claim holds an owner file, so rmdir would fail
            except OSError:
                pass

    # SAVED POSES, INCOMPLETE SCORES -> RESCORE, DO NOT REFOLD.
    #
    # UNDER THE CLAIM, NOT BEFORE IT. This ran before the claim was taken, so two workers both
    # entered recovery on the same unit and both appended the same missing samples -- their
    # rows_done are per-process and cannot see each other -- and a restarted worker could enter
    # recovery while the original was still scoring. The claim is exclusive (mkdir), so taking it
    # first makes recovery single-writer, and the completion state is re-read from the csv here
    # rather than trusted from process start.
    #
    # ANY INCOMPLETE SAVED RUN, INCLUDING ZERO ROWS. The previous version required some sample to be
    # scored already, which excluded the commonest interruption of all: inference finishes, scoring
    # dies before its first row. Those units then fell through to the folding path and hit the trunk
    # guard, so a complete set of poses sat on disk unscorable.
    if A.claim_dir:
        ctx.rows_done, ctx.units_done = open_csv(A.csv)
    if all(iface_scored(ctx.rows_done, i, seed, tag, A.nsample) for i in want):
        release()
        return "skipped"                  # another worker finished it while we queued
    rd = saved_run_dir(A.out, pdb, seed, tag=tag, donor=donor)
    if rd is not None:
        rescore_saved(ctx, pdb, target, tag, seed, want, rd, poses)
        if all(iface_scored(ctx.rows_done, i, seed, tag, A.nsample) for i in want):
            release()
            return "rescored"
        print(f"  {pdb} {tag} seed {seed}: still incomplete after rescoring {os.path.basename(rd)}; "
              "the saved poses themselves may be short", flush=True)
        release()
        return "failed"

    ctx.point_templates_at(kind not in ("free", "posez"))
    if kind in ("free", "posez"):
        # a template-free dataset's free folds must consume none, or they do not match its pool
        job_src, env = f"{ctx.jobs}/{pdb}-final-updated.json", {}
        expect = None if ctx.free_templates else 0
        # posez: the pose enters z through the CONFIDENCE HEAD's distance projections, not as a
        # template -- undiluted and with the inter-chain block intact, where the template path
        # averages a self-pose with every real hit and zeroes cross-chain features by default.
        # POSEZ_CYCLES / POSEZ_GAIN override the injection point and strength for the whole queue.
        if kind == "posez":
            env = dict(env,
                       PROTENIX_POSE_Z_CIF=poses[0],
                       PROTENIX_POSE_Z_CYCLES=os.environ.get("POSEZ_CYCLES", "0"),
                       PROTENIX_POSE_Z_GAIN=os.environ.get("POSEZ_GAIN", "1.0"))
        # TRUNK_LOAD_DIR also serves FREE units (the compute-matched control): resume this seed's own saved trunk, <dir>/<pdb>/free_<seed>.npz,
        # with no template. Missing file = the unit fails, never a fresh fold.
        _tl = os.environ.get("TRUNK_LOAD_DIR")
        if _tl:
            # TAG@SUFFIX names the trunk: a unit tagged "br03@p2" resumes the trunk dumped by the
            # unit tagged "br03". Needed because the default name is keyed on the SEED alone, so 20
            # repeats of one seed would all collide on free_<seed>.npz -- and repeats of one seed is
            # exactly the design here. No "@" in the tag => the old free_<seed>.npz name.
            _tt = tag.split("@")[0] if "@" in tag else None
            _tf = (f"{_tl}/{pdb}/{_tt}_{seed}.npz" if _tt else f"{_tl}/{pdb}/free_{seed}.npz")
            if not os.path.exists(_tf):
                print(f"  {pdb} {tag}: no saved trunk for seed {seed} ({_tf})", flush=True)
                release()
                return "failed"
            env = dict(env, PROTENIX_LOAD_TRUNK=_tf)
    else:
        job_src = f"{A.templates}/jobs/{target}_{tag}/job.json"
        env = dict(TMPL_ENV, PROTENIX_MAX_TEMPLATES=str(len(poses)))
        expect = len(poses)
        # TRUNK_LOAD_DIR=<dir>: resume from the donor's OWN seed's saved trunk, <dir>/<pdb>/free_<seed>.npz, the seed
        # read from the donor pose path (<seed>-<run>/predictions/). Missing file = the unit fails, never a fresh fold.
        _tl = os.environ.get("TRUNK_LOAD_DIR")
        if _tl:
            _m = re.search(r"/(\d+)-[^/]+/predictions/", poses[0])
            _tf = f"{_tl}/{pdb}/free_{_m.group(1)}.npz" if _m else None
            if not _tf or not os.path.exists(_tf):
                print(f"  {pdb} {tag}: no saved trunk for donor {poses[0]} ({_tf})", flush=True)
                release()
                return "failed"
            env = dict(env, PROTENIX_LOAD_TRUNK=_tf)
    if not os.path.exists(job_src):
        print(f"  {target} {tag}: no job at {job_src}", flush=True)
        release()
        return "failed"
    # TRUNK_DUMP_DIR=<dir>: save this fold's z_trunk/s_trunk (Protenix PROTENIX_DUMP_TRUNK hook) to
    # <dir>/<pdb>/<tag>_<seed>.npz, one file per fold, never overwritten. Unset = nothing saved.
    _td = os.environ.get("TRUNK_DUMP_DIR")
    if _td:
        _tf = f"{_td}/{pdb}/{tag}_{seed}.npz"
        if os.path.exists(_tf):
            print(f"  {pdb} {tag} seed {seed}: FAILED -- trunk file already exists: {_tf}", flush=True)
            release()
            return "failed"
        env = dict(env, PROTENIX_DUMP_TRUNK=_tf)
    # CONF_ZDROP_DIR=<dir> (with PROTENIX_CONF_ZDROP=<p> in the environment): the self-distillation dropout
    # check writes this fold's per-pose clean and z-dropout chain_pair_iptm to <dir>/<pdb>/<tag>_<seed>.json.
    _zd = os.environ.get("CONF_ZDROP_DIR")
    if _zd:
        _zf = f"{_zd}/{pdb}/{tag}_{seed}.json"
        if os.path.exists(_zf):
            print(f"  {pdb} {tag} seed {seed}: FAILED -- zdrop file already exists: {_zf}", flush=True)
            release()
            return "failed"
        env = dict(env, PROTENIX_CONF_ZDROP_OUT=_zf)

    rdir = runmeta.run_dir(A.out, pdb, seed, runmeta.run_id())
    name = f"{target.replace('*', pdb)}_{tag}"
    # one directory per fold, created EXCLUSIVELY. A reused directory is how 173 fixed-arm folds
    # overwrote each other's poses in 27 directories while every check passed; fail before the GPU runs
    try:
        os.makedirs(os.path.dirname(rdir), exist_ok=True)
        os.mkdir(rdir)
    except FileExistsError:
        print(f"  {pdb} {target} {tag} seed {seed}: FAILED -- run directory already exists: {rdir}",
              flush=True)
        release()
        return "failed"
    try:
        empty = (ctx.no_templates
                 if kind in ("free", "posez") and not ctx.free_templates else None)
        dt, tcounts = fold(ctx.runner, ctx.configs, named_job(job_src, name, f"{rdir}/jobs", empty),
                           int(seed), name, A.ncycle, A.nsample, expect, rdir, env,
                           expect_poses=A.expect_poses)
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

    args = (ctx, kind, pdb, target, tag, seed, rdir, name, want, poses, dt, release)
    if ctx.A.overlap_scoring:
        return functools.partial(finish_unit, *args)     # main() scores it while the next unit folds
    return finish_unit(*args)


def finish_unit(ctx, kind, pdb, target, tag, seed, rdir, name, want, poses, dt, release):
    """in: a folded unit and its claim's release().  out: "ran" or "failed"

    Scores every interface, then moves the poses into <run dir>/predictions/. With --overlap-scoring
    this runs on the background thread while the NEXT unit folds on the GPU; the claim is held until it
    returns, so no other worker can take a unit that is folded but not yet scored. Any exception is a
    FAILED line and a released claim, never a silent loss.
    """
    try:
        got, err = score_fold(ctx, pdb, target, tag, seed, rdir, name, want, poses)
        # scoring reads the driver's own layout, so normalise only after it has run
        try:
            normalise_poses(rdir, name, pdb)
        except FileExistsError as e:
            # the scores are already in the csv; keep the claim so the unit is not refolded, but say so
            print(f"  {pdb} {target} {tag} seed {seed}: FAILED -- {e}", flush=True)
            return "failed"
        result = "ran"
        # COMPLETE MEANS EVERY INTERFACE AND EVERY SAMPLE. `got or ...` treated any positive number
        # of newly scored interfaces as success, so a unit that scored 1 of 5 poses on 1 of 3
        # interfaces returned "ran" and KEPT its claim -- never retried, silently partial.
        complete = all(iface_scored(ctx.rows_done, i, seed, tag, ctx.A.nsample) for i in want)
        if complete:
            # print pdb too: target is "*" for a free fold
            print(f"  {kind} {pdb} {target} {tag} seed {seed}: {got}/{len(want)} interfaces "
                  f"in {dt:.0f}s", flush=True)
        else:
            missing = [i for i in want
                       if not iface_scored(ctx.rows_done, i, seed, tag, ctx.A.nsample)]
            print(f"  {pdb} {target} {tag} seed {seed}: SCORING INCOMPLETE -- "
                  f"{len(missing)} of {len(want)} interface(s) short of "
                  f"{ctx.A.nsample} samples: {missing[:3]}", flush=True)
            for line in (err or "").strip().splitlines()[-3:]:
                print(f"      {line}", flush=True)
            release()
            result = "failed"
        shutil.rmtree(f"{rdir}/{name}", ignore_errors=True)      # now empty of poses
        if not ctx.A.keep_cifs:
            shutil.rmtree(f"{rdir}/predictions", ignore_errors=True)
        return result
    except Exception as e:
        print(f"  {pdb} {target} {tag} seed {seed}: FAILED -- scoring: {type(e).__name__}: "
              f"{str(e)[:120]}", flush=True)
        release()
        return "failed"

def run_queue(ctx, units):
    """in: the shared handles and every unit.  out: Counter of "ran" / "skipped" / "failed"

    Sequential by default. With --overlap-scoring, ONE background thread scores and at most ONE unit is
    being scored: before handing over unit k+1, wait for unit k -- scoring overlaps the next fold but
    never piles up, and only that thread writes the csv. The last unit's scoring is awaited.
    """
    tally = collections.Counter()
    if not ctx.A.overlap_scoring:
        tally.update(run_unit(ctx, u) for u in units)
        return tally
    pending = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as scorer_thread:
        for u in units:
            r = run_unit(ctx, u)
            if not callable(r):
                tally[r] += 1
                continue
            if pending is not None:
                tally[pending.result()] += 1
            pending = scorer_thread.submit(r)
        if pending is not None:
            tally[pending.result()] += 1
    return tally


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
    ds = paths.DATASETS[A.dataset]
    jobs, ref, scorer = str(ds.jobs), ds.ref, ds.scorer
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
    # free folds match the dataset's pool. FoldBench's were template-free ("Found 0 templates" on every
    # chain), so its free folds get an EMPTY template file per chain: Protenix parses no hits, logs
    # "Found 0 templates" and pads the template slots to 4 exactly as for the pool. (Pointing Protenix at
    # an empty mmCIF directory does not work: it reads parsed templates from its cache.)
    ctx.free_templates = paths.DATASETS[A.dataset].free_templates
    if not ctx.free_templates:
        ctx.no_templates = f"{A.out}/.no_templates.a3m"
        open(ctx.no_templates, "w").close()
    logging.getLogger().addHandler(_CountHandler())
    print(f"  model loaded in {time.time()-t_load:.0f}s | dtype={A.dtype} | {len(units)} units "
          f"| templates={A.templates}", flush=True)

    tally = run_queue(ctx, units)
    print(f"queue complete: {tally['ran']} folds run, {tally['skipped']} already scored, "
          f"{tally['failed']} failed", flush=True)

    # SECOND PASS over whatever is still unscored. A worker walks the queue ONCE, so a unit that
    # becomes claimable only AFTER it has gone past is orphaned: nobody comes back for it, and the
    # complex never reaches its full count. That is not hypothetical -- releasing a dead worker's
    # claim does exactly this, and on 2026-10-06 one fold of 8che sat unclaimed and unfolded at 49/50
    # while both workers had already moved on to the next complex, holding two decoy interfaces out
    # of the comparison.
    #
    # Each pass re-reads the csv, so anything another worker has since finished is skipped, and claims
    # still keep two workers off the same unit. Passes stop as soon as one runs nothing -- bounded, so
    # a genuinely failing unit cannot spin here.
    for sweep in range(1, 4):
        ctx.rows_done, ctx.units_done = open_csv(A.csv)
        if A.claim_dir:
            reconcile_claims(A.claim_dir, A.csv, ifaces)
        again = run_queue(ctx, units)
        if not again["ran"]:
            break
        print(f"  sweep {sweep}: picked up {again['ran']} orphaned folds "
              f"({again['failed']} failed)", flush=True)
        for k, v in again.items():
            tally[k] += v
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
