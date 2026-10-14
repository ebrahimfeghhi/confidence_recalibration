#!/usr/bin/env python
"""Build a multi-pose self-templating job for ANY PXM-2025 complex, not just 9x1h.

Templates a query on several of its own predicted poses at once, with real inter-chain geometry
(cross-chain mask off, per-chain centring off at run time).

WHY A GENERAL VERSION. build_multipose_template.py hardcodes 9x1h's layout -- light (A,B), heavy
(C,D), antigen (E,F). Of the 60 complexes in this run, 20 are multi-copy assemblies with 2-6 protein
entities in various arrangements, so the chain choice has to be derived per complex.

THE CHAIN PROBLEM. Protenix takes ONE template hit per protein ENTITY, and applies it to every copy
of that entity. On a multi-copy assembly the donor has several chains matching each entity, and they
are NOT interchangeable: on 9x1h the two Fabs bind different antigen copies, so pairing chain A with
antigen E instead of F hands the model an antibody and an antigen that are not in contact. With the
cross-chain mask ON that is harmless; with it OFF -- which is the whole point here -- it is fatal.
Candidate chains are enumerated per entity, but the combination is NOT chosen by contact count alone:
--require-chains pins the chains the caller ranked the pose on, and the heuristic only fills in any
remaining entities. Contact count alone picked a different protomer for 1053 of 2920 donors in the
top-5 experiment -- poses selected on the DockQ of chains B/E were templated from A/C/F. The caller
takes the right chains from chain_map.py, which records the mapping the scorer itself used, so the
template encodes the same pair the pose was ranked on. It is required, not optional, because an
unpinned call silently produces a plausible template for the wrong protomer.

DEDUP TRAPS, both of which have silently reduced multi-template arms to one template in this project:
  a3m level     -- hits dedup by SEQUENCE, so every donor gets a different mutated hit string, and
                   the script asserts they are distinct.
  feature level -- dedups on the mmcif sequence; defeated only by PROTENIX_TMPL_NODEDUP=1 at run time.
PROTENIX_MAX_TEMPLATES must also be >= the donor count (trained default is 4).

Template ids must be exactly 4 chars or Protenix logs "Found 0 templates" and runs UNTEMPLATED while
still emitting plausible poses.

Usage:
  build_selftmpl_general.py --pdb 9e7o --out <dir> --tids t001 t002 ... --poses p0.cif ... [--no-cache-write]
"""
import argparse
import difflib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# cifconv converts a Protenix-output cif into a template-parseable one. It now lives beside this file
# in diffusion_commit_clean; sys.path.insert(0, dirname(__file__)) above is what finds it.
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402  (needs the path insert above)
import gemmi  # noqa: E402
from cifconv import convert  # noqa: E402

EXP = str(paths.DATA_DIR)
# The base job json to clone. Defaults to the pxm2025 pool; SELFTMPL_JOBS overrides it so the same
# builder serves the FoldBench pool (both name the file <pdb>-final-updated.json).
P = f"{EXP}/pxm2025"
JOBS = os.environ.get("SELFTMPL_JOBS", f"{P}/jsons")
# PRIVATE NAMESPACE PER RUN. Template ids must be 4 chars (Protenix treats them as pdb ids), which is
# only 36^4 ~ 1.68M values -- and the shared /home/ebrahim/mmcif already holds >10k of them, so by the
# birthday bound a collision in a global namespace is near-certain, not unlikely. No hashing scheme
# fixes that. Instead each run gets its own template dir and its own release-date file via these two
# env vars: the namespace then holds only that run's handful of ids, collisions are impossible by
# construction, and nothing mutates a file another process has already cached.
# NO DEFAULT. This used to fall back to /home/ebrahim/mmcif -- the REAL PDB template library of
# ~10,000 structures. We always override it, but an unset variable would have written invented
# 4-char templates into the genuine library, where they would be indistinguishable from real entries
# and would collide with them. Fail instead.
MMCIF_DIR = os.environ.get("SELFTMPL_MMCIF_DIR")
if not MMCIF_DIR:
    raise SystemExit("SELFTMPL_MMCIF_DIR must be set -- refusing to guess a template directory")
CACHE_F = os.environ.get("SELFTMPL_RELEASE_DATES", "/home/ebrahim/common/release_date_cache.json")
CUT = 5.0
TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seqof(ch):
    return "".join(TLC[r.name] for r in ch if r.name in TLC)


def pick_chains(cif, ent_seqs, require=()):
    """One donor chain per query entity, choosing the most mutually-contacting combination.

    `require` forces named chains into the combination. Without it this picks whichever protomer has
    the most mutual contact, which is NOT necessarily the one the caller cares about: on a multi-copy
    assembly, poses selected by the DockQ of chains B/E were being templated from chains A/C/F, a
    different protomer -- 1053 of 2920 donors in the top-5 experiment. Since the pose was RANKED on
    one chain pair, the template has to encode that same pair or the experiment measures something
    else. The caller gets the right pair from chain_map.py, which records what the scorer itself used.

    Returns (chains, contacts) or (None, reason)."""
    require = set(require)
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    chains = {c.name: seqof(c) for c in st[0] if len(seqof(c)) >= 4}
    if not chains:
        return None, "no protein chains"
    pos = {}
    for c in st[0]:
        if c.name in chains:
            pos[c.name] = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in c for a in r
                                    if a.element.name != "H"])
    # candidates per entity, by sequence identity
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

    def contacts(a, b):
        d = np.linalg.norm(pos[a][:, None, :] - pos[b][None, :, :], axis=-1)
        return int((d.min(axis=1) < CUT).sum())

    best, best_score = None, -1
    cache = {}
    missing = require - {c for cs in cands for c in cs}
    if missing:
        return None, f"required chains {sorted(missing)} match no entity"
    for combo in itertools.product(*cands):
        if len(set(combo)) != len(combo):      # one chain cannot serve two entities
            continue
        if not require <= set(combo):          # must be the protomer the caller ranked
            continue
        tot = 0
        for a, b in itertools.combinations(sorted(set(combo)), 2):
            if (a, b) not in cache:
                cache[(a, b)] = contacts(a, b)
            tot += cache[(a, b)]
        if tot > best_score:
            best, best_score = combo, tot
    if best is None:
        return None, "no valid chain assignment"
    return list(best), best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--tids", nargs="+", required=True)
    ap.add_argument("--no-cache-write", action="store_true")
    ap.add_argument("--require-chains", nargs="+", required=True,
                    help="one comma-separated chain list per --poses entry, forcing that pose's "
                         "template to use those chains. REQUIRED: on multi-copy assemblies the "
                         "most-contacting protomer is often not the one the pose was ranked on.")
    ap.add_argument("--prepend-real", default="",
                    help="real_tmpl_ids.py output: put the FREE arm's real templates BEFORE the "
                         "self-poses, so template_id 0..k-1 are real and k.. are self")
    A = ap.parse_args()
    assert len(A.tids) >= len(A.poses), "need one template id per pose"
    os.makedirs(f"{A.out}/a3m", exist_ok=True)

    base = json.load(open(f"{JOBS}/{A.pdb}-final-updated.json"))
    obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
    obj["name"] = f"{A.pdb}_st"
    ent_seqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    if not ent_seqs:
        sys.exit("no protein entities")

    os.makedirs(MMCIF_DIR, exist_ok=True)
    cache = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
    donors = []
    if len(A.require_chains) != len(A.poses):
        sys.exit("--require-chains needs exactly one entry per --poses entry")
    for d, cif in enumerate(A.poses):
        req = A.require_chains[d].split(",")
        chains, info = pick_chains(cif, ent_seqs, require=req)
        if chains is None:
            print(f"   SKIP donor {d}: {info}")
            continue
        tid = A.tids[d]
        assert len(tid) == 4, f"tid {tid} must be 4 chars"
        if A.no_cache_write and tid not in cache:
            sys.exit(f"tid {tid} not pre-registered")
        cache.setdefault(tid, {"release_date": "2000-01-01"})
        # write-then-rename so a concurrent featuriser never sees a half-written cif
        _txt = convert(open(cif).read(), tid)
        _tmp = f"{MMCIF_DIR}/.{tid}.tmp"
        with open(_tmp, "w") as _fh:
            _fh.write(_txt)
        os.replace(_tmp, f"{MMCIF_DIR}/{tid}.cif")
        donors.append(dict(idx=d, cif=cif, tid=tid, chains=chains, contacts=info))
        print(f"   donor {d}: {tid}  chains {'/'.join(chains)}  contacts {info}")
    if not donors:
        sys.exit("no usable donors")
    if not A.no_cache_write:
        # ATOMIC, LOCKED, AND RE-READ UNDER THE LOCK. Two workers building templates at once were
        # racing here: this is a read-modify-write on a file they share, and `open(...,"w")` truncates
        # it, so the other worker could read a partial file and lose its tids. Re-reading inside the
        # lock also stops one worker's entries being clobbered by the other's stale copy. Symptom was
        # "templates consumed [0]" on arms whose templates had just been written.
        import fcntl, tempfile
        os.makedirs(os.path.dirname(CACHE_F), exist_ok=True)
        lock_f = CACHE_F + ".lock"
        with open(lock_f, "w") as _lk:
            fcntl.flock(_lk, fcntl.LOCK_EX)
            try:
                merged = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
            except Exception:
                merged = {}
            merged.update(cache)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE_F))
            with os.fdopen(fd, "w") as fh:
                json.dump(merged, fh)
            os.replace(tmp, CACHE_F)          # atomic: readers see old or new, never partial
            fcntl.flock(_lk, fcntl.LOCK_UN)

    pi = 0
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = e["proteinChain"]["sequence"]
        Lq = len(q)
        lines, hits = [">query", q], set()

        # --prepend-real: restore the FREE arm's real templates ahead of the self-poses. Written
        # verbatim from the original hmmsearch a3m so that PROTENIX_TMPL_NODEDUP=1 (which run2 needs
        # to keep the 5 near-identical self-poses distinct) cannot change WHICH real templates
        # survive -- only these 4 rows are in the file, so there is nothing left to dedup away.
        # The a3m lists several subsequence ranges per pdb chain and the selected domain name does
        # not record which; the FIRST occurrence is taken, matching the parser's own file order.
        nreal = 0
        if A.prepend_real:
            spec = {r["sequence"]: r for r in json.load(open(A.prepend_real))}
            r = spec.get(q)
            if r is None:
                sys.exit(f"--prepend-real has no entry for entity of length {Lq}")
            body = open(r["templates_path"]).read().split("\n")
            for want in r["names"]:
                for li in range(0, len(body) - 1):
                    if body[li].startswith(f">{want}/"):
                        lines += [body[li], body[li + 1]]
                        hits.add(body[li + 1])
                        nreal += 1
                        break
                else:
                    sys.exit(f"real template {want} not found in {r['templates_path']}")
            print(f"   entity {pi}: prepended {nreal} real templates {r['names']}")
        for k, dn in enumerate(donors):
            hs = list(q)
            for p in ((Lq // 3 + 5 * k) % Lq, (Lq // 2 + 1 + pi + 7 * k) % Lq):
                hs[p] = "A" if hs[p] != "A" else "G"
            hit = "".join(hs)
            assert hit != q, "hit identical to query -> parsed as an MSA row"
            assert hit not in hits, f"duplicate hit for donor {k} -> a3m would dedup it"
            hits.add(hit)
            lines += [f">{dn['tid']}_{dn['chains'][pi]}/1-{Lq} [subseq from] mol:protein "
                      f"length:{Lq}  selftmpl", hit]
        a3m = f"{A.out}/a3m/q{pi}.a3m"
        open(a3m, "w").write("\n".join(lines) + "\n")
        e["proteinChain"]["templatesPath"] = a3m
        pi += 1
    json.dump([obj], open(f"{A.out}/job.json", "w"))
    json.dump(donors, open(f"{A.out}/chosen.json", "w"), indent=1)
    print(f"   wrote {A.out}/job.json: {len(donors)} templates x {pi} entities")


if __name__ == "__main__":
    main()
