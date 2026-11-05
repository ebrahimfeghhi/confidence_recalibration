#!/usr/bin/env python
"""Which saved runs of a unit hold a COMPLETE, rescoreable pose set?

WHY THIS IS NOT "glob found a cif". The drain gate used to classify a unit as CPU-rescoreable when
any pose cif existed under it. Raised in review: a unit with its trunk, two of five poses and
incomplete scores was called rescoreable, but CPU scoring cannot recover the three missing poses --
so the gate could hand the cards to a validation fold while training still held units that need
refolding. Completeness has to be per saved run and per sample index.

WHY PER RUN, NOT PER UNIT. A unit can have several saved runs, <seed>-<hash>, from a refold or a
restarted worker. Samples spread across two runs are not a pose set: the scorer is pointed at ONE
run directory, and poses from different runs came from different folds. So a complete set means a
single directory carrying every index 0 .. nsample-1.

PROVENANCE: a row carries both dockq and iptm, and iptm is read from the pose's own
summary_confidence json, so a run missing those jsons cannot produce the rows the csv expects. Both
files are required per sample.

AMBIGUITY IS AN ERROR, NOT A CHOICE. If two runs are complete, something folded this unit twice and
the two are not interchangeable -- they hold different coordinates. Callers are told, rather than
silently given the first match.
"""
import glob, os, re

SAMPLE = re.compile(r"_sample_(\d+)\.cif$")


def runs_for(poses_dir, pdb, seed):
    """every saved run directory of this unit, as <poses>/<pdb>/<seed>-<hash>"""
    return sorted(d for d in glob.glob(os.path.join(poses_dir, pdb, f"{seed}-*"))
                  if os.path.isdir(d))


def run_samples(run_dir, pdb):
    """{sample index: (cif, conf json or None)} for one saved run"""
    out = {}
    for cif in glob.glob(os.path.join(run_dir, "predictions", "*_sample_*.cif")):
        m = SAMPLE.search(os.path.basename(cif))
        if not m:
            continue
        k = int(m.group(1))
        conf = cif.replace("_sample_", "_summary_confidence_sample_")[:-4] + ".json"
        out[k] = (cif, conf if os.path.exists(conf) else None)
    return out


def classify(poses_dir, pdb, seed, nsample):
    """in: a unit.  out: (status, run_dir_or_None, detail)

    status is one of:
      "complete"   exactly one saved run carries every sample 0..nsample-1 with its conf json
      "ambiguous"  more than one does; the caller must not pick for itself
      "partial"    a run exists but none is complete -- needs a REFOLD, not a rescore
      "none"       no saved run at all -- needs a refold
    """
    runs = runs_for(poses_dir, pdb, seed)
    if not runs:
        return "none", None, "no saved run"
    want = set(range(nsample))
    good, detail = [], []
    for d in runs:
        s = run_samples(d, pdb)
        have = {k for k, (_c, j) in s.items() if j is not None}
        if want <= have:
            good.append(d)
        else:
            miss_cif = sorted(want - set(s))
            miss_conf = sorted(k for k in (want & set(s)) if s[k][1] is None)
            detail.append(f"{os.path.basename(d)}: missing cif {miss_cif}, "
                          f"missing conf {miss_conf}")
    if len(good) == 1:
        return "complete", good[0], os.path.basename(good[0])
    if len(good) > 1:
        return "ambiguous", None, f"{len(good)} complete runs: " \
                                  f"{[os.path.basename(d) for d in good]}"
    return "partial", None, "; ".join(detail)
