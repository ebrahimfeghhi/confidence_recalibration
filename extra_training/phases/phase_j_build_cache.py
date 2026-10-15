#!/usr/bin/env python
"""PHASE J -- pair every cached pose with its LABEL and with its SAMPLE INDEX.

WHAT A CACHED UNIT IS MISSING. The fold wrote, per (structure, seed), the trunk tensors and the
N_sample poses the diffusion module produced (`training/fold/trunks/<pdb>/<arm>_<seed>.npz`). The FT3
loss needs two more things that no fold produces:

  1. THE LABEL -- ground-truth coordinates in the model's atom order, the coordinate mask, and the
     masks and frames the pLDDT / PDE / PAE / resolved losses read. These come out of Protenix's own
     training dataset over our job-level bioassemblies (phase_j_dataset.py), permuted PER POSE.
  2. THE SAMPLE INDEX. The poses in `ci_coordinate` are NOT in `_sample_<k>` order -- measured as a
     permutation on 3,497 of 3,533 validation units. Everything else about a pose is keyed on k:
     its DockQ in results.csv, its summary json, its cif. Without the index, a cached pose cannot be
     joined to the truth of how good it is.

HOW THE SAMPLE INDEX IS ESTABLISHED. By COORDINATES, not by scores. phase_j_rescore_cache.py matches
poses to summaries by chain_pair_iptm and says in its own output why that mapping must not be reused:
two distinct poses can carry identical confidence scores (9fve/400073: samples 3 and 4 agree to five
decimals while differing by 89.6 A), so a score match is not identity. The `_sample_<k>.cif` holds
the same atoms the pose tensor holds, written from the same tensor, so comparing coordinates IS
identity. The match is required to be unique by a wide margin: the chosen cif agrees to within
--coord-tol (cif coordinates carry three decimals) while every other cif of that unit is further away
than --coord-margin.

WHY THE POSE IS NOT A MINI-ROLLOUT. FT3 trains the head on a 20-step mini-rollout of the diffusion
module. Our poses are full inference samples, which is the point of the project: the head is deployed
on inference poses and is being calibrated on the distribution it will actually see. The cache
therefore stands in for the mini-rollout, and phase_j_replay.py holds the pose FIXED on both sides so
that the replay measures the cache rather than the sampler.

WHAT IS WRITTEN. One npz per unit beside the trunks, holding the per-pose permuted label, the sample
indices, and the loss-side feature arrays the trunk npz does not carry. Nothing is overwritten and no
pose is touched.
"""
# WHAT FAILS HERE, WHY, AND WHAT IS LOST: docs/build_cache_exclusions.md. It carries the measured
# cause of every structure this script has refused, the fix where there is one, and the two guards
# below stated as what they assert rather than what they do. Keep it current in the same commit as
# any change to those guards -- a refusal nobody wrote down becomes a silent exclusion.
import argparse
import collections
import copy
import functools
import glob
import json
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
print = functools.partial(print, flush=True)

# The loss-side arrays the trunk npz does NOT carry. protenix/model/loss.py reads every one of these
# out of the input feature dict when train_confidence_only is on:
#   resolution              :1735  gates the confidence loss on [min_resolution, max_resolution]
#   is_rna / is_dna         :1763  nucleotide flag for the bespoke-lddt radius
#   is_ligand               :1765  polymer flag (also used by the head; checked for agreement)
#   plddt_m_rep_atom_mask   :1766  which atoms the pLDDT loss scores
#   distogram_rep_atom_mask :1783  representative atom for the PDE loss
#   pae_rep_atom_mask       :1794  representative atom for the PAE loss
#   frame_atom_index        :1793  the frames the PAE loss aligns on
#   has_frame               :1795  which tokens have one
LOSS_FEATS = ("resolution", "is_rna", "is_dna", "is_ligand", "plddt_m_rep_atom_mask",
              "distogram_rep_atom_mask", "pae_rep_atom_mask", "frame_atom_index", "has_frame")

# Arrays the trunk npz already holds for the HEAD. They must agree atom for atom with the dataset's,
# or the pose and the label are describing different molecules -- which is the one failure that would
# train the head on nonsense while every tensor shape still lined up.
ALIGN_FEATS = ("asym_id", "has_frame", "atom_to_token_idx", "atom_to_tokatom_idx",
               "distogram_rep_atom_mask", "is_ligand")

# filled from the model config in main(), not hardcoded: the gate must be the one the loss applies
RES_MIN = RES_MAX = None


def cif_coords(path):
    """Atom coordinates of a predicted cif, in file order -- which is the model's atom order."""
    import gemmi
    st = gemmi.read_structure(path)
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch for a in r],
                    dtype=np.float64)


def seed_dir(fold, pdb, seed):
    """The ONE pose directory of (pdb, seed).

    A WORKER THAT DIED MID-UNIT LEAVES ITS CLAIM DIRECTORY BEHIND, so a seed can have more than one
    directory and only the finished one holds run.json -- 9dzj has two such leftovers. Taking
    `glob(...)[0]` picked whichever the filesystem listed first, which is not necessarily the
    finished one. Exactly one finished directory is required; 0 or 2 is refused rather than guessed.
    """
    dirs = [d for d in glob.glob(f"{fold}/poses/{pdb}/{seed}-*")
            if os.path.exists(os.path.join(d, "run.json"))]
    if len(dirs) != 1:
        raise ValueError(f"{len(dirs)} finished pose directories for {pdb} seed {seed} "
                         f"(need exactly 1): {[os.path.basename(d) for d in dirs]}")
    return dirs[0]


def sample_cifs(unit_dir, pdb):
    """The `_sample_<k>.cif` files of a unit, in k order, under EITHER on-disk layout.

    TWO LAYOUTS EXIST IN THE SAME FOLD. Almost every unit was flattened to `<unit>/predictions/
    <pdb>_sample_<k>.cif`, but 9dzj seeds 500000 and 500001 kept Protenix's native tree,
    `<unit>/<pdb>_<arm>/seed_<seed>/predictions/<pdb>_<arm>_sample_<k>.cif` -- a different directory
    AND a different basename. Globbing only the flat path found no cifs for them, which
    sample_indices rejects as a pose/cif count mismatch; the units would have been skipped for a
    reason that has nothing to do with their data.
    """
    pats = [f"{unit_dir}/predictions/{pdb}_sample_*.cif",
            f"{unit_dir}/predictions/{pdb}_*_sample_*.cif",
            f"{unit_dir}/*/seed_*/predictions/{pdb}_sample_*.cif",
            f"{unit_dir}/*/seed_*/predictions/{pdb}_*_sample_*.cif"]
    hits = {}
    for pat in pats:
        for f in glob.glob(pat):
            if "summary" in os.path.basename(f):
                continue
            hits.setdefault(int(f.split("sample_")[-1].split(".")[0]), f)
    return [hits[k] for k in sorted(hits)]


def sample_indices(poses, cifs, tol, margin):
    """pose i -> the k of the `<pdb>_sample_<k>.cif` it was written from.

    IDENTITY, NOT SIMILARITY. Each cif is required to match exactly one pose to within `tol`, with
    every other pose further than `margin`; anything else means the join cannot be trusted and the
    unit is refused rather than guessed at.
    """
    if len(cifs) != len(poses):
        raise ValueError(f"{len(poses)} cached pose(s) but {len(cifs)} sample cif(s)")
    d = np.zeros((len(poses), len(cifs)))
    for j, p in enumerate(cifs):
        c = cif_coords(p)
        if c.shape != poses[0].shape:
            raise ValueError(f"{os.path.basename(p)} has {c.shape[0]} atoms, the pose has "
                             f"{poses[0].shape[0]}")
        for i in range(len(poses)):
            d[i, j] = np.abs(poses[i] - c).max()
    out = np.full(len(poses), -1, dtype=np.int64)
    for j, p in enumerate(cifs):
        k = int(p.split("sample_")[-1].split(".")[0])
        col = d[:, j]
        i = int(col.argmin())
        rest = np.delete(col, i)
        if col[i] > tol:
            raise ValueError(f"sample_{k}: nearest cached pose differs by {col[i]:.4f} A "
                             f"(tolerance {tol})")
        if rest.min() < margin:
            raise ValueError(f"sample_{k}: two cached poses within {margin} A of it "
                             f"({col[i]:.4f} and {rest.min():.4f}) -- the join is ambiguous")
        if out[i] != -1:
            raise ValueError(f"cached pose {i} claimed by sample_{out[i]} and sample_{k}")
        out[i] = k
    if (out < 0).any():
        raise ValueError("not every cached pose was claimed by a sample cif")
    return out


def load_trunk(path):
    z = np.load(path)
    feat = {k[len("ci_feat_"):]: z[k] for k in z.files if k.startswith("ci_feat_")}
    return z, feat, z["ci_coordinate"].astype(np.float64)


def merged_lookup(train):
    """{pdb -> [index into the merged dataset]}. train[index] is what __getitem__ takes."""
    out = {}
    for n in range(len(train)):
        di = train.dataset_indices[n]
        wi = train.within_dataset_indices[n]
        pdb = str(train.datasets[di]._get_sample_indice(idx=wi).pdb_id)
        out.setdefault(pdb, []).append(n)
    return out


def check_alignment(npz_feat, ds_feat, pdb):
    """The pose's atom order and the label's must be the SAME order.

    Both sides come from Protenix's featuriser, but by different routes: the pose from INFERENCE over
    a job json of sequences, the label from the TRAINING dataset over the job bioassembly. Nothing
    guarantees they agree -- a residue the crystal did not order, a chain written in another order,
    an entity whose full_sequence is not the fold's would all shift the atom indexing while leaving
    every tensor the right shape. These six arrays pin the mapping from atoms to tokens, so if they
    agree elementwise the two descriptions are the same molecule in the same order.
    """
    import torch
    bad = []
    for k in ALIGN_FEATS:
        if k not in npz_feat or k not in ds_feat:
            bad.append(f"{k}: absent from {'the npz' if k not in npz_feat else 'the dataset item'}")
            continue
        a = np.asarray(npz_feat[k])
        b = ds_feat[k]
        b = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
        if a.shape != b.shape:
            bad.append(f"{k}: npz {a.shape} vs dataset {b.shape}")
        elif not np.array_equal(a.astype(np.int64), b.astype(np.int64)):
            n = int((a.astype(np.int64) != b.astype(np.int64)).sum())
            bad.append(f"{k}: {n} of {a.size} entries differ")
    if bad:
        raise ValueError(f"{pdb}: the cached pose and the label are not in the same atom order: "
                         + "; ".join(bad))


def label_atom_coords(cif_path):
    """The job cif's heavy-atom coordinates as an [N, 3] array -- the label's independent witness.

    Also returns the LARGEST heavy-atom count of any single residue in this file, which is what the
    drop budget below is denominated in. It is measured from the structure rather than written down,
    so there is no per-residue constant to get wrong.
    """
    import gemmi
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.remove_hydrogens()
    pts, biggest = [], 0
    for ch in st[0]:
        for r in ch:
            n = 0
            for at in r:
                pts.append([at.pos.x, at.pos.y, at.pos.z]); n += 1
            biggest = max(biggest, n)
    return np.array(pts, dtype=np.float64), biggest


def check_label(lc, lm, atoms, pdb, unit, tol=0.01, budget_atoms=0):
    """Two invariants a correct per-pose label must satisfy, checked rather than assumed.

    WHY THIS EXISTS. The replay compares a cached FT3 loss to an uncached one, and the reviewer's
    objection to that test is exact: both paths can agree while both receive the same WRONG label.
    These two checks do not go through either path.

    1. THE MULTISET IS POSE-INVARIANT. The per-pose correction is a PERMUTATION -- symmetric chains
       and symmetric atoms are relabelled to match the pose -- so it reorders coordinates and can
       never introduce or move one. If the sorted set of resolved label coordinates differs between
       two poses of the same structure, something other than a permutation happened.
    2. THE LABEL IS THE DEPOSITED STRUCTURE. Every resolved label coordinate must lie within
       `tol` of an atom of the job cif the label was built from. This is what catches a label that is
       internally consistent but describes the wrong molecule -- the failure mode the
       chain-resolution bugs produced; a wrong copy is Angstroms away, not microns.

       NEAREST-NEIGHBOUR, NOT AN EXACT SET MATCH. The first version rounded both sides to three
       decimals and compared sets, which fails whenever the true value sits near a 0.0005 boundary:
       the label is float32 of a 3-decimal cif value, so at |x| ~ 80 A the round-trip error is ~1e-4
       and a coordinate of -80.7395 rounds down on one side and up on the other. It passed on 12qa
       and reported 34 of 8rwf's 2,153 atoms as missing -- every one of them 0.001 A from a real atom,
       spread over 32 unrelated residues. Measured, not reasoned: that was rounding, not a bad label.
    """
    key = lambda v: np.sort(np.ascontiguousarray(v, dtype=np.float32)
                            .view([("x", "f4"), ("y", "f4"), ("z", "f4")]).ravel(),
                            order=("x", "y", "z"))
    ref = key(lc[0][lm[0].astype(bool)])
    for i in range(1, len(lc)):
        v = key(lc[i][lm[i].astype(bool)])
        if v.shape != ref.shape or not np.array_equal(v, ref):
            raise ValueError(f"{pdb}/{unit}: pose {i}'s label is not a permutation of pose 0's "
                             f"({v.shape[0]} vs {ref.shape[0]} resolved atoms)")
    from scipy.spatial import cKDTree
    pts = np.array([[float(v) for v in t] for t in ref], dtype=np.float64)
    d, _i = cKDTree(atoms).query(pts, k=1)
    far = int((d > tol).sum())
    if not far:
        return lm, 0
    # A RESIDUE THE JOB CIF DELIBERATELY DROPPED CANNOT BE SUPERVISED, AND IS NOT AN ERROR.
    # phase_j_job_cifs vetoes residues it cannot place -- `unplaced` and `misplaced` in
    # unplaced_residues.csv -- while Protenix's parse reconstructs them from the entity sequence, so
    # the label ends up holding atoms the job cif does not. Measured on 8av9: the bioassembly carries
    # 4426 atoms against the cif's 4365, and the 61 extra are exactly the 7 vetoed residues (H 2
    # unplaced + 2 misplaced, L 2 + 1). Four of them survive into the resolved label. Refusing the
    # whole structure over that loses 10 units; marking those atoms UNRESOLVED is what
    # coordinate_mask exists to express, and the FT3 loss already skips unresolved atoms.
    #
    # THE BUDGET IS WHAT STOPS THIS BEING A LOOPHOLE. It is derived, not chosen: the number of
    # residues the job cif RECORDED dropping, times the largest heavy-atom count of any residue in
    # that same file. Both come from the data. 9djw's wrong-copy label sat 104 A out on 2011 of 2011
    # atoms and 7stz loses a whole 540-residue chain whose record claims 0 drops -- neither fits in
    # a budget of a few residues, so both still fail loudly.
    if budget_atoms and far <= budget_atoms:
        out, masked_counts = [], []
        tree = cKDTree(atoms)
        for i in range(len(lc)):
            m = lm[i].astype(bool).copy()
            idx = np.flatnonzero(m)                 # ORIGINAL atom indices of the resolved set
            # THE QUERY MUST USE THE UNSORTED COORDINATES. This called key(), which SORTS, so `dd`
            # came back in sorted order while `idx` was in original order -- and `idx[worst]` then
            # mapped sorted positions onto original indices. Review round 7 reproduced the
            # consequence: the invalid atom 100 A out was RETAINED and a legitimate atom was masked
            # instead. The sort is correct for the permutation check above, which compares two sets;
            # it is wrong here, where a position has to be identified.
            pts_i = np.ascontiguousarray(lc[i][m], dtype=np.float64)
            dd, _j = tree.query(pts_i, k=1)
            bad = np.flatnonzero(dd > tol)          # positions WITHIN the resolved set
            m[idx[bad]] = False
            # ASSERT, do not trust. Every atom still marked resolved must now pass the same check
            # the structure was refused for; if one does not, the masking did not do what it claims
            # and a label would be trained on regardless.
            left = np.flatnonzero(m)
            if left.size:
                dl, _ = tree.query(np.ascontiguousarray(lc[i][m], dtype=np.float64), k=1)
                still = int((dl > tol).sum())
                if still:
                    raise ValueError(
                        f"{pdb}/{unit}: after masking, {still} resolved label atom(s) of pose {i} "
                        f"are STILL more than {tol} A from the job cif (worst {dl.max():.3f} A). "
                        f"The masking did not remove what it was asked to.")
            masked_counts.append(int(bad.size))
            out.append(m.astype(np.uint8))
        # every pose is a permutation of the same resolved set, so every pose must mask the same
        # number of atoms; a difference means the per-pose correspondence is not what it claims
        if len(set(masked_counts)) > 1:
            raise ValueError(f"{pdb}/{unit}: masking removed {masked_counts} atoms across poses, "
                             f"but every pose's label is a permutation of the same set, so the "
                             f"counts must agree.")
        return out, masked_counts[0] if masked_counts else 0
    raise ValueError(f"{pdb}/{unit}: {far} of {len(pts)} resolved label atoms are more than "
                     f"{tol} A from any atom of the job cif the label was built from "
                     f"(worst {d.max():.3f} A)"
                     + (f"; the recorded drops allow at most {budget_atoms} atom(s)"
                        if budget_atoms else "; no deliberate drops are recorded for it"))


SCHEMA = 2        # bump whenever the payload's meaning changes, so old labels stop being reused


def sha(path, n=1 << 20):
    """sha256 of a file, or "" if it is absent."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _indices_by_pdb(path):
    """{pdb_id: sha256 of that structure's OWN rows} from an indices csv.

    PER STRUCTURE, NOT PER FILE. The fingerprint hashed the whole job_indices.csv, which covers all
    501 structures at once -- so regenerating the indices for the 142 structures on the refold list
    would have changed that one file and invalidated EVERY label, including the 7,450 units of the
    375 + 37 structures the refold does not touch. Those would have rebuilt for no reason, hours of
    CPU to arrive at identical arrays. A structure's label depends on ITS rows, so that is what is
    hashed.
    """
    import csv as _csv
    rows = collections.defaultdict(list)
    with open(path) as fh:
        rdr = _csv.reader(fh)
        header = next(rdr, None)
        try:
            col = header.index("pdb_id")
        except (AttributeError, ValueError):
            return {}
        for r in rdr:
            if len(r) > col:
                rows[r[col].strip().strip('"')].append(",".join(r))
    return {k: hashlib.sha256("\n".join(v).encode()).hexdigest()[:16] for k, v in rows.items()}


def fingerprint_context(cfg, verbose=True):
    """(code_fp, loss_fp) -- the two parts of a label's provenance that are not per-structure.

    EXTRACTED SO THE TRAINER CAN RECOMPUTE THE SAME FINGERPRINT. It lived inline in main(), so
    anything else wanting to VERIFY a stored fingerprint had to reimplement it, and a reimplementation
    that drifted would either reject every label or accept every label -- the latter being what
    review round 8 found: `fingerprint` was required to be PRESENT and never compared, so a label
    carrying fingerprint="definitely-stale" passed.
    """
    _PX = "/home/ebrahim/Protenix"
    _ext = [os.path.join(_PX, x) for x in (
        "protenix/data/core/parser.py", "protenix/data/core/filter.py",
        "protenix/data/pipeline/dataset.py", "protenix/utils/permutation/permutation.py",
        "protenix/model/loss.py")]
    _local = [os.path.join(HERE, f) for f in
              ("phase_j_build_cache.py", "phase_j_dataset.py", "phase_j_job_cifs.py")]
    # THE REPO ROOT IS ROOT's PARENT, not its grandparent. One level too high resolved outside the
    # checkout, the file never existed, and the patch was silently dropped from the fingerprint --
    # the very thing it was added to cover.
    _patch = os.path.join(os.path.dirname(ROOT), "run_model", "protenix_local.patch")
    code_fp = hashlib.sha256(b"".join(
        open(f, "rb").read() for f in _local + _ext + [_patch]
        if os.path.exists(f))).hexdigest()[:16]
    _absent = [f for f in _ext if not os.path.exists(f)]
    if _absent and verbose:
        print(f"WARNING: {len(_absent)} external file(s) absent from the code fingerprint: "
              f"{[os.path.basename(f) for f in _absent]}")
    # THE LOSS WEIGHTS ARE PART OF THE LABEL'S PROVENANCE: alpha_pae ships as 0.0, so a label built
    # under the stock weights is not the label a run with the PAE loss enabled should reuse.
    try:
        w = cfg.loss.weight
        loss_fp = hashlib.sha256(repr(sorted(
            (k, float(getattr(w, k))) for k in dir(w)
            if not k.startswith("_") and isinstance(getattr(w, k), (int, float))
        )).encode()).hexdigest()[:16]
    except Exception:
        loss_fp = "n/a"
    return code_fp, loss_fp


def unit_fingerprint(a, pdb, trunk_p, code_fp, loss_fp):
    """What this unit's label was built FROM, as one short string.

    WHY A LABEL MUST CARRY ONE. Reuse used to be `os.path.exists(out)`, decided before any check ran,
    and the payload recorded no inputs. So fixing the chain selection or the entity preparation did
    not invalidate a single label already on disk: the next run skipped them all and training would
    have mixed labels built from the old job cifs with labels built from the new ones, with nothing
    able to tell which was which. The job cif in particular is the file the whole label is derived
    from, and it is exactly what the 2026-10-14 geometry fix rewrites.
    """
    return "|".join([
        f"schema={SCHEMA}",
        f"code={code_fp}",
        f"loss={loss_fp}",
        f"cif={sha(os.path.join(a.mmcif_dir, f'{pdb}.cif'))}",
        f"bio={sha(os.path.join(a.bio_dir, f'{pdb}.pkl.gz'))}",
        f"idx={_indices_by_pdb(a.indices).get(pdb, 'absent')}",
        f"trunk={sha(trunk_p)}",
    ])


# Every array a cached unit must carry for the replay and the loss to read it. A payload missing one
# of these is not reusable however good its fingerprint looks.
#
# THE lf_* ARRAYS ARE PART OF THIS. They were left out, so an npz holding only the five structural
# members passed reuse and was then REFUSED by the replay -- the gate that runs later caught what the
# gate that decides reuse did not, which is the wrong way round.
REQUIRED = ("fingerprint", "schema", "label_coordinate", "label_coordinate_mask", "sample_index")


def stale(out, want, n_pose=None):
    """True when `out` cannot be reused: absent, unreadable, incomplete, or built from other inputs.

    THE FINGERPRINT ALONE IS NOT INTEGRITY. This checked only that the `fingerprint` member matched,
    so an npz holding nothing BUT a correct fingerprint was considered a usable label -- and a
    truncated write that happened to keep that member would be reused forever. The required arrays
    must be present, agree with each other on the pose count, and carry no non-finite coordinate.
    """
    if not os.path.exists(out):
        return True
    try:
        with np.load(out, allow_pickle=False) as z:
            if any(k not in z.files for k in REQUIRED):
                return True
            if str(z["fingerprint"]) != want:
                return True
            if int(z["schema"]) != SCHEMA:
                return True
            if any(f"lf_{k}" not in z.files for k in LOSS_FEATS):
                return True
            lc, lm, ks = z["label_coordinate"], z["label_coordinate_mask"], z["sample_index"]
            if lc.ndim != 3 or lc.shape[:2] != lm.shape[:2] or lc.shape[0] != ks.shape[0]:
                return True
            if n_pose is not None and lc.shape[0] != n_pose:
                return True
            # resolution has to be a single finite number inside the loss window, or the confidence
            # terms are silently zero for every pose of this unit
            rr = np.asarray(z["lf_resolution"]).reshape(-1)
            if rr.size < 1 or not np.isfinite(rr[0]):
                return True
            if not np.isfinite(lc[lm.astype(bool)]).all():
                return True
            if len(set(int(v) for v in ks)) != len(ks):
                return True
    except Exception:                                   # truncated or not an npz
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--out-dir", default=None,
                    help="where label npz files are written. DEFAULTS TO <fold>/labels, so it "
                         "follows --fold. It used to be hardcoded to training/fold/labels, which "
                         "meant --fold training/fold_regen wrote its labels into the fold root's "
                         "label directory -- labels built from one root's trunks, filed under "
                         "another. The fingerprint stores trunk=sha(trunk_p) so a mismatch would be "
                         "caught on the next run rather than trained on, but the files would "
                         "already be in the wrong place.")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--coord-tol", type=float, default=0.01,
                    help="max |pose - cif| A for a sample match; cif coordinates have 3 decimals")
    ap.add_argument("--coord-margin", type=float, default=0.5,
                    help="every OTHER pose must be at least this far from a matched cif")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated pdb ids")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail unless every unit of the MANIFEST has a label in one of "
                         "--label-dirs. Successful completion otherwise means only that the units "
                         "which had trunks were labelled, which is not training-set completeness. "
                         "Pass this before training; leave it off for incremental builds.")
    ap.add_argument("--manifest-selection", default=f"{ROOT}/train_val_splits/cache_train_structures.csv",
                    help="THE MANIFEST IS THE SELECTION TIMES THE SEEDS, NOT ANY queue.tsv. Measured "
                         "2026-10-15: training/fold, fold_regen and fold_deferred hold queues of "
                         "733, 723 and 1180 units, summing to 2636 against an intended 5010 -- each "
                         "queue is the REMNANT of one stage, whatever was left to fold at the time. "
                         "Asserting completeness against one of them would pass over 733 units "
                         "while 4277 went unchecked.")
    ap.add_argument("--manifest-seeds", default="500000-500009",
                    help="the seed range the manifest expects per structure, inclusive.")
    ap.add_argument("--label-dirs", default="",
                    help="comma-separated label directories to satisfy the manifest from. The "
                         "trunks live in several roots and so do their labels, so completeness is a "
                         "question about the UNION. Defaults to --out-dir alone.")
    ap.add_argument("--manifest-skip", default="",
                    help="comma-separated pdb ids the manifest must NOT expect, each a deliberate, "
                         "documented drop -- 9mx1 is one: 2941 residues against Protenix's 2560 "
                         "token limit (runner/inference.py:399), so it cannot be folded at all.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", default=None, help="csv of what was built")
    a = ap.parse_args()

    import torch
    import phase_j_dataset as DS
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.utils.permutation.permutation import SymmetricPermutation

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    err = os.path.join(ROOT, "training", "prep", "dataset_errors")
    os.makedirs(err, exist_ok=True)
    train, _ = get_datasets(cfg, err)
    global RES_MIN, RES_MAX
    RES_MIN, RES_MAX = cfg.loss.resolution.min, cfg.loss.resolution.max
    byp = merged_lookup(train)
    perm = SymmetricPermutation(cfg, error_dir=err)
    print(f"dataset: {len(train)} datapoints over {len(byp)} structures")

    units = sorted(glob.glob(f"{a.fold}/trunks/*/*.npz"))
    # AN EMPTY DISCOVERY IS A FAILURE. Globbing whatever trunks exist means a wrong --fold prints
    # "0 cached unit(s) to label", writes an empty report and exits 0 -- a zero-work run that reads
    # as success.
    if not units:
        raise SystemExit(f"STOP: no trunk npz under {a.fold}/trunks. Wrong --fold, or the fold has "
                         "not written any unit yet.")
    # THE EXPECTED SET IS EXPLICIT, NOT WHATEVER IS ON DISK. Discovery cannot see a unit that has no
    # trunk file, so a fold that died partway looks complete. The queue is the contract.
    queue = f"{a.fold}/queue.tsv"
    want_units = set()          # the queue's contract, empty when the queue is not consulted
    if os.path.exists(queue) and not (a.only or a.limit or a.stride != 1):
        # queue.tsv columns are: mode, pdb, target, tag, SEED. Reading the seed from f[0] took the
        # mode instead, so every queued unit looked absent and the note fired on all 501.
        for line in open(queue):
            f = line.split()
            if len(f) >= 5:
                want_units.add((f[1].strip().lower(), f[4].strip()))
        have = {(os.path.basename(os.path.dirname(u)).lower(),
                 os.path.basename(u)[:-4].split("_")[-1]) for u in units}
        gap = want_units - have
        if gap:
            # A NOTE IS NOT A GATE. Incremental builds are the normal case while a fold is in
            # flight, so this cannot be fatal by default -- but a run that finishes having labelled
            # a SUBSET exits 0, and "build_cache succeeded" then reads as "the training set is
            # complete" when it is not. --require-complete makes the queue's contract enforceable.
            msg = (f"{len(gap)} of {len(want_units)} queued unit(s) have no trunk yet, "
                   f"e.g. {sorted(gap)[:3]} -- this run labels {len(have)} of them")
            if a.require_complete:
                raise SystemExit(f"STOP: {msg}\n  --require-complete was given: every queued unit "
                                 f"must have a trunk and a label before training. Re-run without "
                                 f"it to build incrementally while the fold is still going.")
            print(f"NOTE: {msg}")
            print("  (this run is INCOMPLETE by the queue's own contract; pass --require-complete "
                  "to make that fatal, and do so before training)")
        gap_units = gap
    if a.only:
        keep = {x.strip().lower() for x in a.only.split(",") if x.strip()}
        units = [u for u in units if os.path.basename(os.path.dirname(u)).lower() in keep]
        if not units:
            raise SystemExit(f"STOP: --only {a.only} matched no unit under {a.fold}/trunks")
    units = units[a.offset::a.stride]
    if a.limit:
        units = units[:a.limit]
    print(f"{len(units)} cached unit(s) to label")
    # ONE FINGERPRINT OF THE CODE AND THE LOSS CONFIG, shared by every unit of this run.
    # THE EXTERNAL CODE IS PART OF THE LABEL. Hashing only our three local scripts left out what
    # actually derives the label: Protenix's mmCIF parser, its training dataset and the symmetric
    # permutation. A change there -- including the local patch we carry -- would not have
    # invalidated a single cached label.
    _PX = os.environ.get("PROTENIX_ROOT", os.path.expanduser("~/Protenix"))
    code_fp, loss_fp = fingerprint_context(cfg)

    # --out-dir FOLLOWS --fold unless given explicitly. Resolved here, after parsing, so the
    # default tracks whatever --fold ended up being.
    if not a.out_dir:
        a.out_dir = os.path.join(a.fold, "labels")
    print(f"fold root {a.fold}\nlabels -> {a.out_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    rows, skipped, t0, reused = [], [], time.time(), 0
    # THE DATASET ITEM IS BUILT ONCE PER STRUCTURE, not once per unit: every seed of a structure
    # shares the same label, and building it is the expensive half.
    # THE JOB CIF'S OWN RECORD of residues it could not place, per structure. It is the only reason
    # a label atom may legitimately be absent from the cif, so it is the budget for masking one.
    dropped_residues, masked_rows = {}, []
    _ur = os.path.join(os.path.dirname(a.mmcif_dir), "unplaced_residues.csv")
    if os.path.exists(_ur):
        import csv as _csv2
        for _r in _csv2.DictReader(open(_ur)):
            dropped_residues[_r["pdb"]] = dropped_residues.get(_r["pdb"], 0) + \
                int(_r.get("unplaced") or 0) + int(_r.get("misplaced") or 0)
        print(f"drop record: {_ur} -> {sum(1 for v in dropped_residues.values() if v)} structure(s) "
              f"with at least one deliberately dropped residue")
    else:
        print(f"WARNING no drop record at {_ur}; no label atom may be masked, so a structure whose "
              f"job cif vetoed a residue will be refused rather than supervised without it")
    cached_item, cif_atoms = {}, {}
    for n, up in enumerate(units, 1):
        pdb = os.path.basename(os.path.dirname(up))
        arm_seed = os.path.basename(up)[:-4]
        out = os.path.join(a.out_dir, pdb, f"{arm_seed}.npz")
        # REUSE ONLY WHAT WAS BUILT FROM THESE INPUTS. `os.path.exists(out)` let a label written
        # before an upstream fix survive it untouched.
        fprint = unit_fingerprint(a, pdb, up, code_fp, loss_fp)
        # THE TRUNK'S POSE COUNT, so a label that does not describe this unit's poses is not reused.
        try:
            with np.load(up, allow_pickle=False) as _z:
                _np_pose = int(np.asarray(_z["ci_coordinate"]).shape[0]) \
                    if "ci_coordinate" in _z.files else None
        except Exception:
            _np_pose = None
        if not a.overwrite and not stale(out, fprint, n_pose=_np_pose):
            reused += 1
            continue
        try:
            if pdb not in byp:
                raise ValueError(f"no dataset datapoint for {pdb}")
            if pdb not in cached_item:
                # FIRST DATAPOINT OF THE STRUCTURE. With crop_size past the largest job there is no
                # crop, so every datapoint of a structure describes the whole system; which interface
                # was the sampling reason does not change the atoms.
                cached_item[pdb] = train[byp[pdb][0]]
            item = cached_item[pdb]
            feat_ds, label_dict, label_full = (item["input_feature_dict"], item["label_dict"],
                                               item["label_full_dict"])
            z, feat_npz, poses = load_trunk(up)
            check_alignment(feat_npz, feat_ds, pdb)
            ud = seed_dir(a.fold, pdb, arm_seed.split("_")[-1])
            ks = sample_indices(poses, sample_cifs(ud, pdb), a.coord_tol, a.coord_margin)

            # PER POSE. The chain and atom permutation is chosen to match THIS pose, so a label
            # permuted for one pose is the wrong label for another. permute_label_to_match_mini_
            # rollout MUTATES label_dict via dict.update, so each pose gets its own copy and the
            # structure's pristine label is never consumed.
            lc, lm = [], []
            for i in range(len(poses)):
                x = torch.as_tensor(poses[i], dtype=torch.float32)[None]
                ld = copy.deepcopy(label_dict)
                ld, _log = perm.permute_label_to_match_mini_rollout(x, feat_ds, ld,
                                                                    copy.deepcopy(label_full))
                lc.append(ld["coordinate"].detach().cpu().numpy().astype(np.float32))
                lm.append(ld["coordinate_mask"].detach().cpu().numpy().astype(np.uint8))

            if pdb not in cif_atoms:
                cif_atoms[pdb] = label_atom_coords(os.path.join(a.mmcif_dir, f"{pdb}.cif"))
            _cif_pts, _max_res_atoms = cif_atoms[pdb]
            _budget = dropped_residues.get(pdb, 0) * _max_res_atoms
            lm, n_unresolved = check_label(lc, lm, _cif_pts, pdb, arm_seed,
                                           budget_atoms=_budget)
            if n_unresolved:
                masked_rows.append({"pdb": pdb, "unit": arm_seed, "atoms_masked": n_unresolved,
                                    "budget_atoms": _budget,
                                    "dropped_residues": dropped_residues.get(pdb, 0)})
            payload = {"fingerprint": np.array(fprint), "schema": np.array(SCHEMA),
                       "label_coordinate": np.stack(lc), "label_coordinate_mask": np.stack(lm),
                       "sample_index": ks, "pdb": pdb, "unit": arm_seed,
                       "dataset_index": byp[pdb][0], "n_pose": len(poses)}
            # THE RESOLUTION GATE, CHECKED RATHER THAN DISCOVERED. loss.py:1735-1752 compares
            # feat["resolution"] against loss.resolution and ZEROES all four confidence terms when it
            # falls outside -- no error, no warning, a total loss of exactly 0.000000. Every prepared
            # row read -1.0 until the job cif was given its resolution, so this is a measured failure
            # mode, not a hypothetical. A unit the loss would ignore is refused here instead, where
            # it is counted.
            rr = float(np.asarray(feat_ds["resolution"]).reshape(-1)[0])
            if not (RES_MIN <= rr <= RES_MAX):
                raise ValueError(f"resolution {rr} is outside the loss's [{RES_MIN}, {RES_MAX}]; "
                                 "the confidence loss would be zero for every pose of this unit")
            for k in LOSS_FEATS:
                if k not in feat_ds:
                    raise ValueError(f"the dataset item has no {k}, which the confidence loss reads")
                v = feat_ds[k]
                payload[f"lf_{k}"] = (v.detach().cpu().numpy() if torch.is_tensor(v)
                                      else np.asarray(v))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # WRITTEN THROUGH A FILE HANDLE. np.savez_compressed given a PATH appends ".npz" to
            # it, so a temp name ending in ".part" became "....part.npz" and the rename failed.
            tmp = out + ".part"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **payload)
            os.replace(tmp, out)       # atomic: a reader never sees a half-written label
            rows.append({"pdb": pdb, "unit": arm_seed, "n_pose": len(poses),
                         "sample_index": " ".join(str(int(x)) for x in ks),
                         "n_atom": int(poses.shape[1]),
                         "resolved_atoms": int(np.stack(lm).sum(1).max())})
        except Exception as e:                      # noqa: BLE001 -- the reason is the output
            skipped.append({"pdb": pdb, "unit": arm_seed, "why": str(e).replace("\n", " ")[:300]})
            print(f"  SKIP {pdb}/{arm_seed}: {str(e)[:200]}")
        if n % 25 == 0 or n == len(units):
            el = time.time() - t0
            print(f"  {n}/{len(units)}  built {len(rows)}  skipped {len(skipped)}  "
                  f"{el/n:.2f} s/unit")

    print(f"\nbuilt {len(rows)} label set(s), reused {reused} already matching their inputs, "
          f"skipped {len(skipped)}")
    if masked_rows:
        import collections as _c2
        byp = _c2.Counter(r["pdb"] for r in masked_rows)
        tot = sum(r["atoms_masked"] for r in masked_rows)
        print(f"MASKED UNRESOLVED: {tot} label atom(s) over {len(masked_rows)} unit(s) and "
              f"{len(byp)} structure(s) had no counterpart in the job cif and were marked "
              f"unresolved rather than supervised. Each is within that structure's recorded drop "
              f"budget; see unplaced_residues.csv.")
        for _p, _n in byp.most_common(8):
            _ex = next(r for r in masked_rows if r["pdb"] == _p)
            print(f"    {_p}: {_n} unit(s), {_ex['atoms_masked']} atom(s) each, "
                  f"{_ex['dropped_residues']} dropped residue(s), budget {_ex['budget_atoms']}")
    if a.report:
        import csv
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "unit", "n_pose", "sample_index", "n_atom",
                                               "resolved_atoms"], lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {a.report}")
        if skipped:
            sp = a.report.replace(".csv", "_skipped.csv")
            with open(sp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["pdb", "unit", "why"], lineterminator="\n")
                w.writeheader(); w.writerows(skipped)
            print(f"wrote {sp}")
    # THE MANIFEST MUST ACCOUNT FOR EVERY INTENDED UNIT, AND BY LABEL ON DISK, not by what this run
    # happened to do. A run that reused 3,000 labels and built 11 is a success and says nothing about
    # the other 2,000; conversely a label can be absent because its trunk never arrived, because it
    # skipped, or because an earlier run wrote it somewhere else (see --out-dir, which used not to
    # follow --fold). Only looking for the file distinguishes "complete" from "this run worked".
    if a.require_complete:
        import csv as _csv
        lo, hi = (int(x) for x in a.manifest_seeds.split("-"))
        skip = {x.strip().lower() for x in a.manifest_skip.split(",") if x.strip()}
        with open(a.manifest_selection) as fh:
            pdbs = [r["pdb"].strip().lower() for r in _csv.DictReader(fh)]
        intended = {(p, str(sd)) for p in pdbs if p not in skip for sd in range(lo, hi + 1)}
        dirs = [d.strip() for d in a.label_dirs.split(",") if d.strip()] or [a.out_dir]
        found, missing = set(), []
        for u in sorted(intended):
            if any(os.path.exists(os.path.join(d, u[0], f"free_{u[1]}.npz")) for d in dirs):
                found.add(u)
            else:
                missing.append(u)
        print(f"\nMANIFEST: {len(found)} of {len(intended)} intended unit(s) have a label")
        print(f"  manifest = {len(pdbs) - len(skip & set(pdbs))} structure(s) x seeds "
              f"{lo}-{hi}, from {a.manifest_selection}")
        print(f"  skipped by name: {sorted(skip) or 'none'}")
        print(f"  label dirs searched: {dirs}")
        if missing:
            import collections as _c
            byp = _c.Counter(u[0] for u in missing)
            raise SystemExit(
                f"STOP: {len(missing)} intended unit(s) over {len(byp)} structure(s) have NO label "
                f"in any of {dirs}.\n  worst: {byp.most_common(5)}\n"
                f"  The training set is INCOMPLETE. Either build them, or add a deliberate drop to "
                f"--manifest-skip and record it in docs/cached_training.md.")

    # A SKIP IS A FAILURE, not a statistic. Training on a subset chosen by which labels happened to
    # build is the kind of silent selection this project keeps finding; the caller decides.
    if skipped:
        raise SystemExit(f"STOP: {len(skipped)} unit(s) have no label. Each reason is in the skipped "
                         "csv; investigate before training, and record anything genuinely dropped "
                         "in docs/cached_training.md.")


if __name__ == "__main__":
    main()
