#!/usr/bin/env python
"""Turn each chosen donor pose into a template Protenix can actually load.

SUMMARY
    Runs after structural_diversity_filter.py, over the queue it wrote. For every donor: pick which
    chains of the pose go into the template, rewrite the pose as a template cif under a fake 4-character
    pdb id, register a release date for that id, and write the job json the fold will consume.

    WHICH CHAINS. The pose was ranked on one specific antibody-antigen pair, so the template has to
    encode that pair or the fold measures something else. The pair is passed in and forced; the
    remaining entities are filled by whichever chain contacts the forced pair most. Without forcing,
    the builder picked the most-contacting protomer overall, which on multi-copy assemblies was the
    wrong one for 1,053 of 2,920 donors.

    FOUR-CHARACTER IDS. Protenix identifies a template by a pdb id, so a pose has to masquerade as one.
    That is only 36^4 values, and the real PDB library holds >10k of them, so ids are assigned
    sequentially within a namespace private to this run rather than hashed into a shared space.

    RELEASE DATES ARE WRITTEN ONCE, BY THIS PROCESS, after every namespace is built. Protenix caches
    release dates once per process, so an id registered mid-run is rejected and its template silently
    dropped while the fold still completes and scores. Writing the file at the end also removes the
    lock the old builder needed: when each donor wrote the shared file itself, two workers could read a
    partial copy and lose each other's ids.

    ONE FILE. This was two: an orchestrator and a per-donor builder invoked as a subprocess, because the
    builder read its output directories from environment variables at import time and so needed a fresh
    process per namespace. Paths are arguments now, so it is a function in a pool.

INPUTS
    --queue:  the queue from structural_diversity_filter.py. Required. One namespace per
              (interface, tag); the queue repeats that pair once per seed.
    --ns:     namespace directory to build. Required.
    --maps:   anchor_relabel.py's chain maps: which pose chains a donor's interface pair is. Required.
    --roles:  pose_interfaces.py's interfaces csv, in the same (anchor) frame. Required.
    --jobs:   worker processes. Default 8. CPU only.

OUTPUTS
    <ns>/mmcif/<tid>.cif:        each donor pose rewritten as a template, under its 4-character id
    <ns>/jobs/<iface>_<tag>/job.json:    the Protenix input for that fold, with templatesPath set
    <ns>/jobs/<iface>_<tag>/chosen.json: which pose, id and chains were used -- read later to record
                                 what a fold templated
    <ns>/jobs/<iface>_<tag>/a3m/q<i>.a3m: one hit per donor per entity, naming the template and chain
    <ns>/release_dates.json:     every id in the namespace, dated 2000-01-01 so nothing filters it out

EXAMPLE
    build_templates.py --queue exp/queue.tsv --ns exp/templates \\
                       --maps exp/chain_frame/chain_maps_anchor.csv --roles exp/interfaces.csv --jobs 16

FUNCTIONS
    seqof():           a chain -> its one-letter sequence
    pick_chains():     a pose, the entity sequences, the forced pair -> one chain per entity
    template_cif():    a pose cif -> a cif the template parser can read, under a given id
    required_chains(): the queue's interface and poses -> the chain pair each donor must be pinned to
    pose_key():        a pool pose path -> (pdb, seed, sample)
    tid():             an index -> a 4-character base-36 id
    build_one():       one namespace: chains, template cifs, a3m hits, job json, chosen.json
    write_release_dates(): every id in the namespace -> release_dates.json
    main():            assign ids, build every namespace in parallel, then write the release dates
"""
import argparse
import collections
import difflib
import itertools
import json
import os
import sys
from io import StringIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemmi  # noqa: E402
import paths  # noqa: E402
import pool_index  # noqa: E402
from contacts import CONTACT_CUTOFF  # noqa: E402
from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: E402

B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
TLC = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def seqof(ch):
    """in: a gemmi chain.  out: its one-letter sequence, skipping anything not a standard residue."""
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def pick_chains(cif, ent_seqs, require=()):
    """in: a pose cif, the query entity sequences, chains that must be included.
    out: (one chain per entity, their mutual contact count), or (None, reason)

    `require` forces the pair the pose was RANKED on. Without it this returns whichever protomer has
    the most mutual contact, which on a multi-copy assembly is often a different one.
    """
    require = set(require)
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    chains = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
    if not chains:
        return None, "no protein chains"
    pos = {c.name: np.array([[a.pos.x, a.pos.y, a.pos.z] for r in c for a in r
                             if a.element.name != "H"])
           for c in st[0] if c.name in chains}
    cands = []
    for q in ent_seqs:
        cs = [nm for nm, s in chains.items()
              if difflib.SequenceMatcher(None, s, q, autojunk=False).ratio() >= 0.9]
        if not cs:
            return None, f"no chain matches an entity of length {len(q)}"
        cands.append(cs)
    n_comb = int(np.prod([len(c) for c in cands]))
    if n_comb > 4096:
        return None, f"too many chain combinations ({n_comb})"
    missing = require - {c for cs in cands for c in cs}
    if missing:
        return None, f"required chains {sorted(missing)} match no entity"

    def contacts(a, b):
        """atoms of `a` with at least one atom of `b` within the cutoff"""
        d = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
        return int((d.min(axis=1) < CONTACT_CUTOFF).sum())

    best, best_n, cache = None, -1, {}
    for combo in itertools.product(*cands):
        if len(set(combo)) != len(combo):        # one chain cannot serve two entities
            continue
        if not require <= set(combo):            # must include the pair the pose was ranked on
            continue
        n = 0
        for a, b in itertools.combinations(sorted(set(combo)), 2):
            if (a, b) not in cache:
                cache[(a, b)] = contacts(a, b)
            n += cache[(a, b)]
        if n > best_n:
            best, best_n = list(combo), n
    if best is None:
        return None, "no valid chain assignment"
    return best, best_n


def template_cif(src_text, tid):
    """in: a Protenix pose cif, the id to write it under.  out: cif text the template parser accepts.

    Adds the revision date and the struct_asym loop the parser needs, and lowercases the peptide
    linking value it is case-sensitive about.
    """
    d = MMCIF2Dict(StringIO(src_text))
    aid = d.get("_atom_site.label_asym_id", [])
    eid = d.get("_atom_site.label_entity_id", [])
    seen, pairs = [], []
    for a, e in zip(aid, eid):
        if a not in seen:
            seen.append(a)
            pairs.append((a, e))
    out = []
    for ln in src_text.replace("PEPTIDE LINKING", "peptide linking").splitlines():
        out.append(ln)
        if ln.startswith("data_"):
            out.append("_pdbx_audit_revision_history.revision_date   2000-01-01")
    out += ["loop_", "_struct_asym.id", "_struct_asym.entity_id"]
    out += [f"{a} {e}" for a, e in pairs]
    out.append("#")
    return "\n".join(out) + "\n"


def required_chains(iface, donors, maps, roles):
    """in: the interface that picked these donors, their pose paths, a chain map and roles.
    out: one "ab,ag" string per donor -- the pair that donor must be pinned to
    """
    ab_ref, ag_ref = roles[iface]
    req = []
    for pose in donors:
        pdb, seed, samp = pose_key(pose)
        m = maps.get((pdb, seed, samp))
        if not m or ab_ref not in m or ag_ref not in m:
            raise SystemExit(f"{iface}: no chain mapping for {os.path.basename(pose)}")
        req.append(f"{m[ab_ref]},{m[ag_ref]}")
    return req


def pose_key(pose):
    """in: a pool pose path.  out: (pdb, seed, sample), raising rather than guessing on a new layout."""
    base = os.path.basename(pose)
    parts = pose.split("/")
    try:
        pdb = parts[-4]
        seed = parts[-3].split("-")[0]
        samp = int(base.split("_sample_")[1][:-4])
    except (IndexError, ValueError) as e:
        raise SystemExit(f"cannot read (pdb, seed, sample) from {pose}: {e}")
    return pdb, seed, samp


def tid(i):
    """in: an index.  out: a 4-character base-36 id, assigned sequentially so ids cannot collide."""
    s = ""
    for _ in range(4):
        s = B36[i % 36] + s
        i //= 36
    return s


def build_one(task):
    """in: (key, pdb, pose paths, ids, namespace, required chains, jobs dir).
    out: (key, "ok" | "cached" | "FAIL", reason)
    """
    key, pdb, donor_poses, tids, ns, req, jobs_dir = task
    out = f"{ns}/jobs/{key}"
    # REUSE ONLY WHAT IS PROVABLY THE SAME: same donors under the same ids. Ids are re-assigned in
    # queue order on every build, so after a re-selection a cached job can point at an id whose cif
    # this build is about to overwrite with a different donor -- a wrong template, silently.
    if os.path.exists(f"{out}/job.json") and os.path.exists(f"{out}/chosen.json"):
        old = json.load(open(f"{out}/chosen.json"))
        if [(c["cif"], c["tid"]) for c in old] == [(donor_poses[c["idx"]], tids[c["idx"]])
                                                   for c in old if c["idx"] < len(donor_poses)]:
            return key, "cached", ""
    try:
        os.makedirs(f"{out}/a3m", exist_ok=True)
        os.makedirs(f"{ns}/mmcif", exist_ok=True)
        base = json.load(open(f"{jobs_dir}/{pdb}-final-updated.json"))
        obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
        obj["name"] = f"{pdb}_st"
        ent_seqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        if not ent_seqs:
            return key, "FAIL", "no protein entities"

        chosen = []
        for d, cif in enumerate(donor_poses):
            chains, info = pick_chains(cif, ent_seqs, require=req[d].split(","))
            if chains is None:
                continue
            t = tids[d]
            tmp = f"{ns}/mmcif/.{t}.tmp"
            with open(tmp, "w") as fh:          # write-then-rename: no half-written cif is ever read
                fh.write(template_cif(open(cif).read(), t))
            os.replace(tmp, f"{ns}/mmcif/{t}.cif")
            chosen.append(dict(idx=d, cif=cif, tid=t, chains=chains, contacts=info))
        if not chosen:
            return key, "FAIL", "no usable donors"

        for pi, e in enumerate([e for e in obj["sequences"] if "proteinChain" in e]):
            q = e["proteinChain"]["sequence"]
            Lq = len(q)
            lines, hits = [">query", q], set()
            for k, dn in enumerate(chosen):
                # each hit must differ from the query and from every other hit, or the parser reads it
                # as an MSA row or dedups it away and the arm quietly loses a template
                hs = list(q)
                for p in ((Lq // 3 + 5 * k) % Lq, (Lq // 2 + 1 + pi + 7 * k) % Lq):
                    hs[p] = "A" if hs[p] != "A" else "G"
                hit = "".join(hs)
                if hit == q or hit in hits:
                    return key, "FAIL", f"donor {k} hit collides"
                hits.add(hit)
                lines += [f">{dn['tid']}_{dn['chains'][pi]}/1-{Lq} [subseq from] mol:protein "
                          f"length:{Lq}  selftmpl", hit]
            a3m = f"{out}/a3m/q{pi}.a3m"
            open(a3m, "w").write("\n".join(lines) + "\n")
            # ABSOLUTE. Protenix opens this path itself, and folder.py runs with its own cwd, so a
            # relative path here resolves against the wrong directory and every fold dies in
            # featurisation with FileNotFoundError -- 603 folds failed this way, none succeeded.
            # It does NOT survive renaming the experiment directory (renaming the 2026-09-25 arm broke
            # all 10,190), so the namespace is named at build time and never moved.
            e["proteinChain"]["templatesPath"] = os.path.abspath(a3m)
        json.dump([obj], open(f"{out}/job.json", "w"))
        json.dump(chosen, open(f"{out}/chosen.json", "w"), indent=1)
        return key, "ok", ""
    except Exception as e:
        return key, "FAIL", f"{type(e).__name__}: {e}"


def write_release_dates(ns):
    """in: the namespace.  out: release_dates.json covering every template cif in it; returns the count.

    The schema is nested on purpose: Protenix's parser keeps only {"id": {"release_date": ...}} and
    silently drops a flat mapping. The date is old so nothing filters the template out for being newer
    than the target.
    """
    import glob
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ns}/mmcif/*.cif"))
    json.dump({i: {"release_date": "2000-01-01"} for i in ids},
              open(f"{ns}/release_dates.json", "w"))
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--ns", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    uniq = collections.OrderedDict()
    for line in open(A.queue):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            uniq.setdefault(f"{f[2]}_{f[3]}", (f[1], f[2], f[5].split()))

    maps, roles = pool_index.load_chain_maps(A.maps), pool_index.load_roles(A.roles)
    jobs_dir, _, _ = paths.dataset(A.dataset)
    os.makedirs(f"{A.ns}/mmcif", exist_ok=True)

    tasks, n = [], 0
    for key, (pdb, iface, donor_poses) in uniq.items():
        req = required_chains(iface, donor_poses, maps, roles)
        tasks.append((key, pdb, donor_poses, [tid(n + i) for i in range(len(donor_poses))],
                      A.ns, req, jobs_dir))
        n += len(donor_poses)
    print(f"  namespaces to build: {len(tasks)}   template ids: {n} (sequential, no collisions)",
          flush=True)

    import multiprocessing as mp
    done = collections.Counter()
    with mp.Pool(A.jobs) as pool:
        for i, (key, st, err) in enumerate(pool.imap_unordered(build_one, tasks), 1):
            done[st] += 1
            if st == "FAIL":
                print(f"  FAIL {key}: {err[:200]}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {dict(done)}", flush=True)
    print(f"  release_dates.json covers {write_release_dates(A.ns)} template ids", flush=True)
    return 1 if done["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
