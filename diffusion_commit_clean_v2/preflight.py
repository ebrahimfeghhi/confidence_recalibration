#!/usr/bin/env python
"""Check a template namespace before any GPU time is spent on it.

SUMMARY
    Everything that went wrong in this experiment was knowable without folding: donors templated from
    the wrong protomer (1,053 of 2,920 once), a namespace missing its release-date file, ids colliding
    in a shared 4-character space. Each surfaced hours later, or worse, did not surface at all --
    Protenix logs "Found 5 templates" and folds happily on a template for the wrong chains.

    Five checks, cheapest first:

      donor    every template was built from the pose the QUEUE names. A namespace built for an
               earlier selection can otherwise be reused for a different donor.
      paths    every job's templatesPath is absolute and exists (a relative one failed 603 folds).

      ids      every donor's id is unique across the namespace, has a cif, and has a release-date
               entry. Ids are 4 characters because Protenix reads them as pdb codes, so a collision
               means one fold silently overwrites a template another is using.
      chains   the chains the builder chose CONTAIN the pair this interface ranked the pose on. The
               pose was selected on one pair; templating another measures something else.
      hits     every a3m hit names a chain that exists in its cif with residues, and the hits are
               mutually distinct -- identical hit strings get deduped away and a 5-template arm
               quietly becomes a 1-template arm.

    NOT CHECKED HERE: that featurisation yields a non-zero atom mask. That needs the model's own config
    construction, and a hand-rolled version produced all-zero tensors that were briefly mistaken for a
    real finding. It is verified once on a real fold instead of approximated here.

INPUTS
    --ns:     the namespace to check. Required.
    --queue:  the queue it was built from. Default queue_top5.tsv.
    --maps:   chain-map csv naming the pair each donor should be pinned to. Must be the frame the queue
              was SELECTED in -- checking anchor-frame templates against the crystal mapping reports
              every deliberate difference as a failure.
    --roles:  roles csv, same frame as --maps.
    --show:   how many examples to print per failing check. Default 6.

OUTPUTS
    stdout: a table of checks and failure counts, then examples
    exit code: non-zero if any check failed, so this works as a gate before launching

EXAMPLE
    preflight.py --ns exp/templates --queue exp/queue.tsv --maps exp/frame_maps.csv --roles exp/frame_roles.csv

FUNCTIONS
    main(): walk every namespace the queue asks for and run the three checks
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemmi  # noqa: E402
import pool_index  # noqa: E402
from build_templates import pose_key  # noqa: E402
from fmt import table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--roles", default=None)
    A = ap.parse_args()

    maps, roles = pool_index.load_chain_maps(A.maps), pool_index.load_roles(A.roles)
    rd = json.load(open(f"{A.ns}/release_dates.json")) if os.path.exists(
        f"{A.ns}/release_dates.json") else {}
    cifs = {os.path.basename(p)[:-4] for p in glob.glob(f"{A.ns}/mmcif/*.cif")}

    want = collections.OrderedDict()
    for line in open(A.queue):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            want.setdefault(f"{f[2]}_{f[3]}", (f[2], f[5].split()))

    fail = collections.Counter()
    ex = collections.defaultdict(list)
    seen_tids = {}
    chain_cache = {}

    for key, (iface, poses) in want.items():
        jd = f"{A.ns}/jobs/{key}"
        if not os.path.exists(f"{jd}/job.json"):
            fail["namespace missing"] += 1
            ex["namespace missing"].append(key)
            continue
        chosen = json.load(open(f"{jd}/chosen.json"))
        if len(chosen) != len(poses):
            fail["donor count wrong"] += 1
            ex["donor count wrong"].append(f"{key}: {len(chosen)}/{len(poses)}")
        # every alignment path must exist and be absolute: Protenix opens it from its own cwd, and a
        # relative one failed 603 folds with none succeeding
        for e in json.load(open(f"{jd}/job.json"))[0]["sequences"]:
            tp = e.get("proteinChain", {}).get("templatesPath")
            if tp is not None and not (os.path.isabs(tp) and os.path.exists(tp)):
                fail["templatesPath not absolute or missing"] += 1
                ex["templatesPath not absolute or missing"].append(f"{key}: {tp}")
        ab_ref, ag_ref = roles[iface]
        for d in chosen:
            t = d["tid"]
            # the template must be built from the donor the QUEUE names, not a cached earlier one
            if d["idx"] >= len(poses) or d["cif"] != poses[d["idx"]]:
                fail["WRONG DONOR (template is not the queued pose)"] += 1
                ex["WRONG DONOR (template is not the queued pose)"].append(
                    f"{key}: built from {os.path.basename(os.path.dirname(os.path.dirname(d['cif'])))}")
            if len(t) != 4:
                fail["tid not 4 chars"] += 1
            if t in seen_tids and seen_tids[t] != key:
                fail["tid COLLISION"] += 1
                ex["tid COLLISION"].append(f"{t}: {seen_tids[t]} vs {key}")
            seen_tids[t] = key
            if t not in cifs:
                fail["tid has no cif"] += 1
            if t not in rd:
                fail["tid missing release date"] += 1
            # the pair this interface ranked the pose on
            pdb, seed, samp = pose_key(d["cif"])
            m = maps.get((pdb, seed, samp))
            if not m:
                fail["pose unmapped"] += 1
                continue
            need = {m[ab_ref], m[ag_ref]}
            if not need <= set(d["chains"]):
                fail["WRONG CHAINS"] += 1
                ex["WRONG CHAINS"].append(
                    f"{key} {os.path.basename(d['cif'])}: need {'/'.join(sorted(need))}, "
                    f"got {'/'.join(d['chains'])}")
            f_cif = f"{A.ns}/mmcif/{t}.cif"
            if f_cif not in chain_cache and os.path.exists(f_cif):
                st = gemmi.read_structure(f_cif)
                st.setup_entities()
                chain_cache[f_cif] = {c.name: len(c) for c in st[0]}
            have = chain_cache.get(f_cif, {})
            for ch in d["chains"]:
                if have.get(ch, 0) < 1:
                    fail["cif lacks a chosen chain"] += 1
        # a3m hits distinct, and each names a real chain
        for a3m in glob.glob(f"{jd}/a3m/*.a3m"):
            body = open(a3m).read().split("\n")
            names = [l for l in body if l.startswith(">") and "selftmpl" in l]
            seqs = [body[i + 1] for i, l in enumerate(body)
                    if l.startswith(">") and "selftmpl" in l and i + 1 < len(body)]
            if len(set(seqs)) != len(seqs):
                fail["duplicate a3m hit (would dedup)"] += 1
                ex["duplicate a3m hit (would dedup)"].append(os.path.basename(a3m) + " in " + key)
            for nm in names:
                mm = re.match(r">(\w{4})_(\w+)/", nm)
                if not mm:
                    fail["a3m hit name malformed"] += 1
                    continue
                t, ch = mm.groups()
                f_cif = f"{A.ns}/mmcif/{t}.cif"
                if chain_cache.get(f_cif, {}).get(ch, 0) < 1:
                    fail["a3m names a chain absent from its cif"] += 1
                    ex["a3m names a chain absent from its cif"].append(f"{key}: {nm[:40]}")

    print(table(["check", "failures"],
                [[k, v] for k, v in fail.most_common()] or [["all checks passed", 0]],
                align="l r"))
    for k, v in ex.items():
        print(f"\n  {k}:")
        for e in v[:A.show]:
            print(f"    {e}")
    print(f"\n  namespaces: {len(want)}   distinct tids: {len(seen_tids)}   "
          f"cifs: {len(cifs)}   release-date entries: {len(rd)}")
    return 1 if sum(fail.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
