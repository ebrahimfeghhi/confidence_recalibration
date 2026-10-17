"""PREFLIGHT: the trunk's features agree atom for atom with the dataset's, for every structure.

WHY THIS IS A SEPARATE SCRIPT. phase_k_train used to build each structure's full dataset item and
call BC.check_alignment on it inline. That item is the run's memory -- ~518 MB each, 164 per rank,
82-95 GB per rank -- and the loss turns out not to need it at all: a key-access probe found the loss
reads nine feature keys and two label keys, all already in the cached label npz, and
phase_k_minimal_item_equivalence.py showed both paths give bit-identical losses. So --minimal-items
drops the item.

check_alignment is the ONE other thing the item was used for, and it is not optional: it is what
proves the pose and the label describe the same molecule. Dropping the item without it would make
the single worst failure mode -- a pose paired with another molecule's label -- silent. So the check
moves here, runs once per structure instead of once per epoch, and writes a marker that
--minimal-items requires.

THE MARKER IS BOUND TO A POPULATION. It records the manifest digest, computed the same way
phase_k_train computes it, so a marker from a different selection, skip list or seed range is
refused rather than accepted.

COST: one item build per structure, measured at 3.6-10.4 s (avg 6.8), so ~56 min for 491 on one
process. CPU only -- no GPU is touched, so it can run beside a training job.

  python -I phases/phase_k_alignment_preflight.py --out training/alignment_preflight.json
"""
import argparse, csv, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "alignment_preflight.json"))
    ap.add_argument("--manifest-selection",
                    default=os.path.join(ROOT, "train_val_splits", "cache_train_structures.csv"))
    ap.add_argument("--manifest-skip",
                    default="9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3")
    ap.add_argument("--seeds", default="500000-500009")
    ap.add_argument("--fold", default=os.path.join(ROOT, "training", "fold"))
    ap.add_argument("--indices", default=os.path.join(ROOT, "training/prep/job_indices.csv"))
    ap.add_argument("--bio-dir", default=os.path.join(ROOT, "training/prep/job_bioassembly"))
    ap.add_argument("--mmcif-dir", default=os.path.join(ROOT, "training/prep/job_cifs"))
    ap.add_argument("--msa-root", default=os.path.join(ROOT, "building_msa/work/mmcif_msa"))
    ap.add_argument("--seq-index",
                    default=os.path.join(ROOT, "building_msa/work/pdb_seqs/seq_to_pdb_index.json"))
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N structures")
    a = ap.parse_args()

    if os.environ.get("LAYERNORM_TYPE") != "openfold":
        raise SystemExit("STOP: set LAYERNORM_TYPE=openfold, to match how the trunks were written.")

    import phase_j_dataset as DS
    import phase_j_build_cache as BC
    from protenix.data.pipeline.dataset import get_datasets

    skip = {x.strip().lower() for x in a.manifest_skip.split(",") if x.strip()}
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rows = [r for r in csv.DictReader(open(a.manifest_selection))
            if r.get("tokens", "").isdigit() and r["pdb"].lower() not in skip]
    pdbs = sorted({r["pdb"] for r in rows})

    # THE SAME DIGEST phase_k_train computes, over the same strings: "<pdb>/<label basename>" for
    # every unit, sorted. Reconstructed here rather than imported because the trainer builds it
    # inline; a mismatch is a REFUSAL at launch, never a false pass, so the failure is safe.
    units = sorted(f"{p}/free_{s}.npz" for p in pdbs for s in range(lo, hi + 1))
    digest = hashlib.sha256("\n".join(units).encode()).hexdigest()[:16]
    print(f"manifest digest {digest} over {len(units)} unit(s), {len(pdbs)} structure(s)")

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    train_ds, _ = get_datasets(cfg, os.path.join(ROOT, "training", "prep", "dataset_errors"))
    byp = BC.merged_lookup(train_ds)

    todo = pdbs[:a.limit] if a.limit else pdbs
    bad, missing, t0 = [], [], time.time()
    for i, pdb in enumerate(todo, 1):
        trunk = os.path.join(a.fold, "trunks", pdb, f"free_{lo}.npz")
        if not os.path.exists(trunk):
            missing.append(pdb); continue
        try:
            it = train_ds[byp[pdb][0]]
            _z, feat_npz, _p = BC.load_trunk(trunk)
            BC.check_alignment(feat_npz, it["input_feature_dict"], pdb)
        except Exception as e:
            bad.append((pdb, f"{type(e).__name__}: {e}"))
        del it
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {(time.time()-t0)/i:.1f} s/structure  "
                  f"{len(bad)} mismatch(es)", flush=True)

    # COMPLETENESS IS PART OF THE PASS. --limit exists for debugging, but the digest is computed
    # over the WHOLE manifest, so a marker written after checking one structure used to carry
    # ok=true with the full population's digest -- and the trainer, which compares the digest and
    # reads ok, accepted it as authorisation for all 491. Review round 5 reproduced that. The
    # marker now records whether every structure in the manifest was actually checked, and a
    # partial run can never be ok.
    complete = len(todo) == len(pdbs)
    ok = not bad and not missing and complete
    rec = {"ok": ok, "complete": complete, "n_manifest_structures": len(pdbs),
           "limit": int(a.limit), "manifest_digest": digest,
           "n_structures": len(todo), "n_units": len(units),
           "manifest_selection": a.manifest_selection, "manifest_skip": a.manifest_skip,
           "seeds": a.seeds, "checked_seed": lo,
           "mismatches": [{"pdb": p, "error": e} for p, e in bad],
           "missing_trunks": missing,
           "schema": int(BC.SCHEMA),
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {a.out}")
    if missing:
        print(f"PREFLIGHT FAILED: {len(missing)} structure(s) have no trunk for seed {lo}: "
              f"{missing[:8]}")
    if bad:
        print(f"PREFLIGHT FAILED: {len(bad)} structure(s) whose trunk disagrees with the dataset:")
        for p, e in bad[:10]:
            print(f"  {p}: {e}")
    if not complete:
        print(f"PREFLIGHT INCOMPLETE: --limit {a.limit} checked {len(todo)} of {len(pdbs)} "
              f"structure(s), so this marker records ok=false and the trainer will refuse it. "
              f"Re-run without --limit to authorise training.")
    if ok:
        print(f"PREFLIGHT PASSED: {len(todo)} structure(s) aligned, in "
              f"{(time.time()-t0)/60:.1f} min. --minimal-items may use this marker.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
