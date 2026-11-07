#!/usr/bin/env python
"""THE shared definition of a CPU-recoverable unit. One resolver, used by the drain gate and the
rescorer, so the two can never disagree about what can be recovered without a GPU.

WHY IT IS THIS STRICT. The drain gate once classified a unit as CPU-rescoreable when any pose cif
existed under it. A unit with its trunk, two of five poses and incomplete scores therefore counted
as recoverable, but CPU scoring cannot conjure the three missing poses -- so the gate could hand
the cards to a validation fold while training still held units that need refolding.

WHAT A RECOVERABLE UNIT IS, and why each clause is here:

  ONE ELIGIBLE RUN, SAMPLES 0..n-1. A unit can have several saved runs, <seed>-<hash>, from a
  refold or a restarted worker. Samples spread across two runs are not a pose set: the scorer is
  pointed at ONE directory, and poses from different runs come from different folds.

  run.json MUST NAME THIS pdb, seed AND arm. A directory merely matching poses/<pdb>/<seed>-* can
  belong to another run; the arm matters because the same complex and seed can be folded free and
  templated, and their poses are not interchangeable. Seed is compared as a string because
  run.json stores it as an int.

  EVERY POSE NEEDS A READABLE SUMMARY JSON CARRYING chain_pair_iptm. A results row holds both dockq
  and iptm, and ost_dockq.parse reads iptm out of chain_pair_iptm in that file. A run whose json is
  missing, truncated or missing that key cannot produce the row the csv expects, so scoring it
  would fail halfway and leave the unit short again. Existence is not enough -- the file is parsed.

  TWO COMPLETE ELIGIBLE RUNS ARE AMBIGUOUS. They are different folds holding different
  coordinates, and nothing here may choose between them: a caller that picked the first would be
  deciding which fold the label describes. Ambiguity blocks and is reported.

  A PARTIAL RUN CANNOT MAKE A UNIT LOOK RECOVERABLE. Partial and complete runs are judged
  separately, and a complete run is never rejected because a partial one sorts before it -- that
  is the bug the rescorer had through run_dir_for, which returned the first directory naming the
  unit regardless of whether its pose set was whole.

THE OTHER DIRECTION MATTERS JUST AS MUCH: a unit wrongly called GPU-blocking holds the validation
fold indefinitely. So "complete" is decided only on the files the scorer actually reads, and
nothing else -- not mtimes, not the trunk, not the csv.
"""
import glob, json, os, re

SAMPLE = re.compile(r"_sample_(\d+)\.cif$")
IPTM_KEY = "chain_pair_iptm"

COMPLETE, AMBIGUOUS, PARTIAL, NONE = "complete", "ambiguous", "partial", "none"


def runs_for(poses_dir, pdb, seed):
    """every saved run directory of this unit, as <poses>/<pdb>/<seed>-<hash>"""
    return sorted(d for d in glob.glob(os.path.join(poses_dir, pdb, f"{seed}-*"))
                  if os.path.isdir(d))


def provenance_ok(run_dir, pdb, seed, arm):
    """out: (True, "") when run.json names this pdb, seed and arm; else (False, why)"""
    rj = os.path.join(run_dir, "run.json")
    if not os.path.exists(rj):
        return False, "no run.json"
    try:
        with open(rj) as fh:
            m = json.load(fh)
    except (OSError, ValueError) as e:
        return False, f"unreadable run.json ({type(e).__name__})"
    if str(m.get("pdb")) != str(pdb):
        return False, f"run.json pdb {m.get('pdb')!r} != {pdb!r}"
    if str(m.get("seed")) != str(seed):
        return False, f"run.json seed {m.get('seed')!r} != {seed!r}"
    if arm is not None and str(m.get("arm")) != str(arm):
        return False, f"run.json arm {m.get('arm')!r} != {arm!r}"
    return True, ""


def conf_ok(path):
    """out: (True, "") when the summary json parses and carries the key iptm is read from"""
    if not os.path.exists(path):
        return False, "absent"
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        return False, f"unreadable ({type(e).__name__})"
    if not isinstance(d, dict) or IPTM_KEY not in d:
        return False, f"no {IPTM_KEY}"
    v = d[IPTM_KEY]
    if not isinstance(v, list) or not v:
        return False, f"{IPTM_KEY} is not a non-empty list"
    return True, ""


def run_samples(run_dir):
    """{sample index: (cif, conf_path, conf_ok, why)} for one saved run"""
    out = {}
    for cif in sorted(glob.glob(os.path.join(run_dir, "predictions", "*_sample_*.cif"))):
        m = SAMPLE.search(os.path.basename(cif))
        if not m:
            continue
        conf = cif.replace("_sample_", "_summary_confidence_sample_")[:-4] + ".json"
        ok, why = conf_ok(conf)
        out[int(m.group(1))] = (cif, conf, ok, why)
    return out


def classify(poses_dir, pdb, seed, nsample, arm=None):
    """in: a unit (arm optional; when given, provenance must match it).
    out: (status, run_dir_or_None, detail)

    status: "complete" | "ambiguous" | "partial" | "none".
    Only "complete" is CPU-recoverable; everything else needs a GPU or a human.
    """
    runs = runs_for(poses_dir, pdb, seed)
    if not runs:
        return NONE, None, "no saved run"
    want = set(range(nsample))
    good, notes = [], []
    for d in runs:
        base = os.path.basename(d)
        ok, why = provenance_ok(d, pdb, seed, arm)
        if not ok:
            notes.append(f"{base}: {why}")
            continue
        s = run_samples(d)
        usable = {k for k, (_c, _j, cok, _w) in s.items() if cok}
        if want <= usable:
            good.append(d)
            continue
        miss_cif = sorted(want - set(s))
        bad_conf = sorted((k, s[k][3]) for k in (want & set(s)) if not s[k][2])
        notes.append(f"{base}: missing cif {miss_cif}" +
                     (f", bad summary {bad_conf}" if bad_conf else ""))
    if len(good) == 1:
        return COMPLETE, good[0], os.path.basename(good[0])
    if len(good) > 1:
        return AMBIGUOUS, None, (f"{len(good)} complete eligible runs: "
                                 f"{[os.path.basename(d) for d in good]}")
    return PARTIAL, None, "; ".join(notes) or "no eligible run"
