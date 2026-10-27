#!/usr/bin/env python
"""Pass 1 of the ABAG-Rank export: fix each complex's inter-chain pair set. CPU only, no GPU.

WHY A FIXED SET PER COMPLEX. ABAG-Rank's 03_prepare_dataset.py computes per-complex normalisation
stats with np.stack(pae_list, axis=0) over the complex's samples, which requires every pose of a
complex to have the SAME number of pairs. A per-pose distance cutoff does not give that -- on 10fj
the 12 A inter-chain counts range 1,784 to 2,018 across its 50 poses, so the stack would raise.
Their own stage 1 avoids it by writing ALL inter-chain pairs (259,600 for 10fj, ~76 GB over our
38,150 poses) and letting the dataloader cut.

So this takes the UNION of each complex's per-pose cutoff sets: measured on 10fj that is 4,096
pairs, 2.2x the per-pose mean and 1.05% of T^2, about 1.3 GB over the whole dump. Lengths are then
constant within a complex, stage 3 stacks unchanged, and the dataloader's own per-sample
valid_pair_mask still applies the real cutoff -- so the model sees exactly what it would have seen.

THE REPRESENTATIVE ATOM IS THE DISTOGRAM ONE, not "the first atom of the token". The trunk npz
carries ci_feat_distogram_rep_atom_mask for exactly this, and using anything else would put the
distances on a different convention from everything that reads them.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rep_coords(z, ca_mask, pose):
    """(T, 3) CA coordinates for one pose, and which tokens have one.

    CA, NOT THE DISTOGRAM REPRESENTATIVE. ci_feat_distogram_rep_atom_mask selects CB for ordinary
    amino acids and CA only for glycine; ABAG-Rank's compute_residue_distance_info() takes CA when
    present and falls back to CB. Measured on 10fj the two masks differ at 1,158 of 4,795 atoms and
    their representatives sit 1.42 A apart on average (max 1.56), coinciding for exactly the 46
    glycine tokens. At a 12 A cutoff a 1.4 A shift moves real pairs across the boundary, and a pair
    the fixed union omits can never be recovered downstream. So the CA mask from the LABEL npz
    (lf_pae_rep_atom_mask) is used instead -- verified identical across a complex's seeds. Caught
    in review.
    """
    a2t = z["ci_feat_atom_to_token_idx"]
    mask = ca_mask
    T = len(z["ci_feat_asym_id"])
    out = np.full((T, 3), np.nan, dtype=np.float32)
    seen = np.zeros(T, dtype=bool)
    xyz = z["ci_coordinate"][pose]
    for ai in np.where(mask)[0]:
        t = int(a2t[ai])
        if not seen[t]:
            out[t] = xyz[ai]
            seen[t] = True
    return out, seen


def source_fingerprint(pdb, fold):
    """Identity of the exact files a pair set was built from.

    A POSE COUNT DOES NOT ESTABLISH THIS. Swap one seed file for a different seed and the directory
    still holds ten files and fifty poses, so a count-based check reuses pair indices derived from
    a pose population that no longer exists. Edited coordinates or token metadata are invisible to
    a count too. Name, size and mtime of every trunk AND its label file are cheap and catch all
    three; a content hash would be stronger but would read the 1.4 TB cache. NANOSECOND mtime, not
    int seconds: a file rewritten twice inside one second at the same size would otherwise keep its
    fingerprint. Both caught in review.
    """
    parts = []
    for f in sorted(glob.glob(os.path.join(fold, "trunks", pdb, "free_*.npz"))):
        lab = f.replace("/trunks/", "/labels/")
        for q in (f, lab):
            try:
                st = os.stat(q)
                parts.append(f"{os.path.basename(q)}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                parts.append(f"{os.path.basename(q)}:MISSING")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def pair_set(pdb, fold, cutoff):
    """Union over every (seed, pose) of this complex of its inter-chain pairs under `cutoff`."""
    files = sorted(glob.glob(os.path.join(fold, "trunks", pdb, "free_*.npz")))
    if not files:
        return None
    union = None
    asym = a2t0 = ca0 = None
    npose = 0
    n_fallback = [0]
    for f in files:
        z = np.load(f)
        lab = f.replace("/trunks/", "/labels/")
        if not os.path.exists(lab):
            raise SystemExit(f"STOP: {pdb}: no label npz beside {os.path.basename(f)}; the CA mask "
                             f"lives there and guessing it would silently change the pair set.")
        ca = np.load(lab)["lf_pae_rep_atom_mask"].astype(bool)
        # TOKEN IDENTITY ACROSS SEEDS IS ASSERTED, NOT ASSUMED. The union only means anything if
        # index i is the SAME residue in every seed. Equal shapes do not establish that: a seed
        # with the same token count but a different ordering would silently merge incompatible
        # pairs, and nothing downstream could detect it. Caught in review.
        if asym is None:
            asym = z["ci_feat_asym_id"]
            a2t0 = z["ci_feat_atom_to_token_idx"]
            ca0 = ca
            inter = asym[:, None] != asym[None, :]
        else:
            for name, got, want in (("asym_id", z["ci_feat_asym_id"], asym),
                                    ("atom_to_token_idx", z["ci_feat_atom_to_token_idx"], a2t0),
                                    ("ca_rep_atom_mask", ca, ca0)):
                if not np.array_equal(got, want):
                    raise SystemExit(
                        f"STOP: {pdb}: {name} differs between seeds ({os.path.basename(files[0])} "
                        f"vs {os.path.basename(f)}). Index i is not the same residue in both, so "
                        f"a union over their pairs would be meaningless.")
        for p in range(z["ci_coordinate"].shape[0]):
            rep, seen = rep_coords(z, ca, p)
            d = np.linalg.norm(rep[:, None, :] - rep[None, :, :], axis=-1)
            ok = seen[:, None] & seen[None, :]
            # <= MATCHES UPSTREAM's `ca_distances <= distance_cutoff`, not <.
            m = inter & ok & (d <= cutoff)
            # KNOWN GAP, TO BE CLOSED IN THE EXPORTER, NOT HERE: upstream drops residue pairs
            # whose residue_one_letter is 'X' BEFORE applying the cutoff and before choosing the
            # fallback, so its fallback is the closest pair among NON-X residues. This union has no
            # residue letters to filter by -- they are not in the trunk cache at all (ci_s_inputs
            # carries no restype one-hot; 400 of its 449 columns are identically zero), so they
            # must come from the sequences or the pose CIFs. And they matter: 103 of 5,549
            # sequences in seq_to_pdb_index.json contain X. Where a pose's closest pair involves an
            # X residue, upstream would fall back to the closest NON-X pair, which this union may
            # not contain. Caught in review; the exporter must pass a residue-validity mask in.
            # THE ZERO-CONTACT FALLBACK, which upstream has and a plain union does not:
            # preprocess_shards keeps the single CLOSEST inter-chain pair when a pose has none
            # under the cutoff. Without this, such a pose gets whatever pairs other poses
            # contributed -- possibly at 25 A -- instead of its own closest. Those poses are
            # exactly the contact-free decoys a reranker must learn to reject. Caught in review.
            if not m.any():
                dd = np.where(inter & ok, d, np.inf)
                if np.isfinite(dd).any():
                    i0, j0 = np.unravel_index(np.argmin(dd), dd.shape)
                    m = np.zeros_like(m)
                    m[i0, j0] = True
                    n_fallback[0] += 1
            else:
                # ALSO add this pose's closest pair even when it has contacts, so the union can
                # serve the fallback for any pose; it is one pair and usually already present.
                dd = np.where(inter & ok, d, np.inf)
                if np.isfinite(dd).any():
                    i0, j0 = np.unravel_index(np.argmin(dd), dd.shape)
                    m[i0, j0] = True
            union = m if union is None else (union | m)
            npose += 1
    if union is None or not union.any():
        return None
    i, j = np.where(union)
    return {"pdb": pdb, "n_tokens": int(len(asym)), "n_poses": npose,
            "n_fallback": n_fallback[0],
            "n_pairs": int(len(i)), "inter_idx": i.astype(np.int32),
            "inter_jdx": j.astype(np.int32), "asym_id": asym.astype(np.int32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="training/full_train/structures_no9dzj.csv")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--cutoff", type=float, default=12.0,
                    help="matches config_ABAG_rank_no_esm.yaml's distance_cutoff")
    ap.add_argument("--out", default="training/abag/pairsets")
    ap.add_argument("--limit", type=int, default=0, help="first N structures, for a smoke test")
    ap.add_argument("--allow-missing", action="store_true",
                    help="exit 0 even when some requested structure produced no pair set. Off by "
                         "default: a partial population that exits 0 is how a dataset loses "
                         "structures silently.")
    a = ap.parse_args()

    pdbs = [r["pdb"] for r in csv.DictReader(open(os.path.join(ROOT, a.selection)))]
    if a.limit:
        pdbs = pdbs[:a.limit]
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    tot_pairs = tot_poses = done = skipped = tot_bytes = n_fb = 0
    excluded = []
    for n, pdb in enumerate(pdbs, 1):
        dst = os.path.join(out, f"{pdb}.npz")
        # A CACHED PAIR SET IS REUSED ONLY IF ITS CONFIGURATION MATCHES. Its contents depend on the
        # cutoff, the fold it was built from and how many poses existed at the time; reusing one
        # across a cutoff change or a new seed would keep stale pairs while writing fresh metadata
        # that claims otherwise. Caught in review.
        if os.path.exists(dst):
            try:
                old = np.load(dst)
                ok = (float(old["cutoff"]) == a.cutoff
                      and str(old["fold"]) == a.fold
                      and str(old["atom_convention"]) == "CA"
                      and str(old["source"]) == source_fingerprint(pdb, os.path.join(ROOT, a.fold)))
            except Exception:                     # noqa: BLE001
                ok = False
            if ok:
                done += 1
                continue
            print(f"  REBUILD {pdb}: cached pair set does not match this configuration", flush=True)
        ps = pair_set(pdb, os.path.join(ROOT, a.fold), a.cutoff)
        if ps is None:
            excluded.append(pdb)
            # With the closest-pair fallback in place this now means only "no trunks" or "no
            # inter-chain token pair at all". A structure whose poses simply have no contact under
            # the cutoff keeps its closest pair and stays in the population, as upstream does.
            skipped += 1
            print(f"  SKIP {pdb}: no trunks, or no inter-chain token pair exists", flush=True)
            continue
        np.savez_compressed(dst, inter_idx=ps["inter_idx"], inter_jdx=ps["inter_jdx"],
                            asym_id=ps["asym_id"], n_tokens=ps["n_tokens"],
                            n_poses=ps["n_poses"], cutoff=a.cutoff, fold=a.fold,
                            atom_convention="CA", n_fallback=ps["n_fallback"],
                            source=source_fingerprint(pdb, os.path.join(ROOT, a.fold)))
        n_fb += ps["n_fallback"]
        tot_pairs += ps["n_pairs"]
        tot_poses += ps["n_poses"]
        # bytes this complex will cost in the dump: one float32 PAE and one float32 CA distance
        # per pair per pose. Accumulated per complex because pairs/complex and poses/complex both
        # vary, so a product of the two means is not the sum of the products.
        tot_bytes += ps["n_pairs"] * ps["n_poses"] * 4 * 2
        done += 1
        if n % 25 == 0 or n == len(pdbs):
            print(f"  {n}/{len(pdbs)}  last {pdb}: T={ps['n_tokens']} poses={ps['n_poses']} "
                  f"pairs={ps['n_pairs']:,}", flush=True)
    print(f"\n  {done} complex(es) written, {skipped} skipped, "
          f"{n_fb} pose(s) had NO pair within the cutoff and kept their closest")
    if tot_pairs:
        print(f"  {tot_pairs:,} pairs over {tot_poses:,} poses; mean {tot_pairs/max(done,1):,.0f} "
              f"pairs/complex")
        print(f"  dump size for these complexes: {tot_bytes / 1e9:.3f} GB; "
              f"projected for {len(pdbs)} complexes: "
              f"{tot_bytes / max(done, 1) * len(pdbs) / 1e9:.2f} GB")
    json.dump({"cutoff": a.cutoff, "selection": a.selection, "n_complexes": done,
               "requested": len(pdbs), "excluded": excluded, "atom_convention": "CA"},
              open(os.path.join(out, "_meta.json"), "w"), indent=1)
    # A PARTIAL EXPORT MUST NOT EXIT 0. Skipping prints a warning, and a warning in a long log is
    # how a dataset silently loses structures; the HDF5 exporter would then build a population
    # nobody chose. --allow-missing makes the exclusion a decision, and _meta.json records exactly
    # which structures it covers so the exporter can assert agreement. Caught in review.
    if excluded and not a.allow_missing:
        print(f"\n  STOP: {len(excluded)} of {len(pdbs)} requested structure(s) produced no pair "
              f"set: {excluded[:8]}{' ...' if len(excluded) > 8 else ''}\n"
              f"  The export population would silently differ from the selection. Pass "
              f"--allow-missing to accept it deliberately; _meta.json lists the exclusions.",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
