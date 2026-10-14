#!/usr/bin/env python
"""Build one template namespace per (interface, arm) for the top-5 experiment.

fold.py resolves a templated unit's job as {templates}/jobs/{target}_{tag}/job.json, so each
(interface, arm) needs its own job built from its own 5 poses. The 5 SEEDS of an arm share one job --
same templates, different random seed -- so this builds 2 jobs per interface, not 10.

TEMPLATE IDS ARE ASSIGNED SEQUENTIALLY IN BASE 36, not hashed. Every namespace writes its cifs into
ONE shared mmcif directory keyed by a 4-character id, and hashing ~2900 ids into that space collides
by the birthday bound -- it already happened once in this project, 720 collisions out of ~3300, where
one fold overwrote a template another was still using. A counter cannot collide.

    build_templates.py --queue queue_top5.tsv --ns <namespace dir> [--jobs 8]
"""
import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))   # the builder lives beside this file
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def pose_key(path):
    """(pdb, seed, sample) for a pool pose, from its path. Raises rather than guessing.

    The path is the only place a donor records which pose it is, and the chain mapping is keyed on
    exactly this triple. A silent mis-parse would look up the wrong pose's chains and produce a
    template for the wrong protomer -- which is the failure this whole mechanism exists to prevent --
    so anything that does not match the pool layout is an error, not a skip.
    """
    m = re.search(rf"/free/(\w+)/(\d+)-{re.escape(paths.LEGACY_RUN)}/predictions/"
                  r"\w+_sample_(\d+)\.cif$", path)
    if not m:
        raise ValueError(f"cannot parse a pool pose from {path}")
    return m.group(1), m.group(2), int(m.group(3))


def required_chains(iface, poses, maps, roles):
    """One "ab,ag" string per pose: the MODEL chains that pose's own interface uses.

    Straight from chain_map.py, i.e. the mapping PXMeter used when it scored the pose. Without this
    the builder falls back to whichever protomer has the most contact, which differed from the ranked
    one for 1053 of 2920 donors -- the pose was chosen for the DockQ of one chain pair and templated
    from another.
    """
    ab_ref, ag_ref = roles[iface]
    out = []
    for p in poses:
        pdb, seed, samp = pose_key(p)
        m = maps.get((pdb, seed, samp))
        if not m or ab_ref not in m or ag_ref not in m:
            raise KeyError(f"{iface}: no chain mapping for {pdb} seed {seed} sample {samp}")
        out.append(f"{m[ab_ref]},{m[ag_ref]}")
    return out


def tid(i):
    s = ""
    for _ in range(4):
        s = B36[i % 36] + s
        i //= 36
    return s


def one(task):
    key, pdb, poses, tids, ns, req = task
    out = f"{ns}/jobs/{key}"
    if os.path.exists(f"{out}/job.json"):
        return key, "cached", ""
    # BOTH env vars, always. Setting only the mmcif dir sends the release dates to the module's
    # default shared cache outside this namespace, and fold.py then refuses the namespace as
    # incomplete -- after the whole build has run.
    env = dict(os.environ, SELFTMPL_MMCIF_DIR=f"{ns}/mmcif",
               SELFTMPL_RELEASE_DATES=f"{ns}/release_dates.json")
    r = subprocess.run([sys.executable, f"{CODE}/build_selftmpl_general.py",
                        "--pdb", pdb, "--out", out, "--poses", *poses, "--tids", *tids,
                        "--require-chains", *req],
                       capture_output=True, text=True, env=env)
    ok = os.path.exists(f"{out}/job.json")
    return key, "ok" if ok else "FAIL", (r.stderr or r.stdout)[-300:] if not ok else ""


def write_release_dates(ns):
    """Emit {ns}/release_dates.json covering every template cif in the namespace.

    fold.py refuses a namespace without this file. The builder only writes it when
    SELFTMPL_RELEASE_DATES points inside the namespace; deriving it from the cifs on disk makes the
    namespace self-consistent regardless of how it was built. Schema is nested on purpose --
    _parse_release_dates keeps only {"id": {"release_date": ...}} and silently drops a flat mapping.
    """
    import glob as _g
    import json as _j
    ids = sorted(os.path.basename(p)[:-4] for p in _g.glob(f"{ns}/mmcif/*.cif"))
    _j.dump({i: {"release_date": "2000-01-01"} for i in ids},
            open(f"{ns}/release_dates.json", "w"))
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--ns", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    # The frame that decides WHICH chains of a donor get pinned. Default is the crystal-anchored
    # mapping; pass the anchor-frame pair to build templates with no reference to the native structure.
    # Both must move together -- roles name the reference chains that maps is keyed on.
    ap.add_argument("--maps", default=None)
    ap.add_argument("--roles", default=None)
    A = ap.parse_args()

    # one job per (target, tag); the queue repeats it once per seed
    uniq = collections.OrderedDict()
    for line in open(A.queue):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6:
            continue
        uniq.setdefault(f"{f[2]}_{f[3]}", (f[1], f[2], f[5].split()))

    maps, roles = metrics.load_chain_maps(A.maps), metrics.load_roles(A.roles)

    os.makedirs(f"{A.ns}/mmcif", exist_ok=True)
    tasks, n = [], 0
    for key, (pdb, iface, poses) in uniq.items():
        req = required_chains(iface, poses, maps, roles)
        tasks.append((key, pdb, poses, [tid(n + i) for i in range(len(poses))], A.ns, req))
        n += len(poses)
    print(f"  namespaces to build: {len(tasks)}   template ids: {n} (sequential, no collisions)",
          flush=True)

    import multiprocessing as mp
    done = collections.Counter()
    with mp.Pool(A.jobs) as pool:
        for i, (key, st, err) in enumerate(pool.imap_unordered(one, tasks), 1):
            done[st] += 1
            if st == "FAIL":
                print(f"  FAIL {key}: {err.strip()[:200]}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {dict(done)}", flush=True)
    n_ids = write_release_dates(A.ns)
    print(f"  release_dates.json covers {n_ids} template ids", flush=True)
    return 1 if done["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
