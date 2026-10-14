#!/usr/bin/env python
"""The pool index: which poses exist, and what each one scored.

WHY ITS OWN FILE. This lived in build_matrices.py because that was the first script to need it, and
when validate.py needed the same thing it grew a SECOND copy with a different return shape. Two
functions named load_pool reading one csv is how a schema change becomes a silent bug: you fix the copy
you remember. One definition, two views.

WHAT IS IN THE CSV. One row per (pose, interface), written by the PXMeter scoring path:

    pdb, chain1, chain2, seed, sample, dockq, pair_iptm

`pair_iptm` is the model's own per-chain-pair confidence -- an INPUT to selection, available at test
time. `dockq` is ground truth against the crystal and is read ONLY by evaluation. Keeping both in one
loader is convenient and slightly dangerous, so it is worth saying plainly: nothing in the selection
path may read the dockq field.

`extra` adds interfaces the reference does not contain -- chain pairs the PREDICTIONS declare by being
in contact. Those rows carry an EMPTY dockq, because no reference interface means no ground truth.
Empty becomes NaN rather than 0.0 on purpose: 0.0 would read as "a very bad interface" in every mean
and every acceptable-rate denominator instead of "not evaluable".
"""
import collections
import csv
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402


def _rows(extra=None):
    for src in [str(paths.PXM2025_POOL_CSV)] + ([extra] if extra else []):
        with open(src) as fh:
            for r in csv.DictReader(fh):
                yield r


def load_pool(extra=None):
    """{interface: [(seed, sample, dockq_gt, pair_iptm), ...]} -- the form build_matrices wants."""
    pool = collections.defaultdict(list)
    for r in _rows(extra):
        dq = float(r["dockq"]) if r["dockq"] not in ("", None) else float("nan")
        pool[f"{r['pdb']}_{r['chain1']}{r['chain2']}"].append(
            (r["seed"], int(r["sample"]), dq, float(r["pair_iptm"])))
    return pool


def dockq_by_pose(extra=None):
    """{interface: {(seed, sample): dockq_gt}} -- the form validate.py wants, same source."""
    out = collections.defaultdict(dict)
    for iface, rows in load_pool(extra).items():
        for s, x, dq, _ in rows:
            out[iface][(s, x)] = dq
    return out
