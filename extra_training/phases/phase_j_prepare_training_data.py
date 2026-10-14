#!/usr/bin/env python
"""PHASE J, step 1 -- turn the selected training structures into Protenix's training format.

WHAT THIS PRODUCES
    <bio-dir>/<pdb_id>.pkl.gz   one gzipped bioassembly dict per structure
    <out-csv>                   the indices csv: one row per sampleable chain / interface

WHY IT IS NEEDED, and it is not only about the dataloader. Protenix's trainer reads these, but the
decisive reason is the LOSS: label_dict -- ground-truth coordinates in the model's atom order, the
coordinate and representative-atom masks, and the frames the PAE loss aligns on -- is built from the
bioassembly dict. A cache of confidence-head INPUTS cannot be trained against without it, which is
review round 3's finding 6.

THE CIFS COME FROM RCSB, NOT FROM SAbDab2. SAbDab2's per-instance cifs are trimmed: they carry no
`pdbx_struct_assembly` block, and Protenix's parser reads it first (parser.py:170), so all 535 raise
KeyError and produce no bioassembly dict. Verified on a full RCSB entry, which parses in ~4 s. So one
whole mmCIF per PDB ENTRY is fetched, and prepare_training_data enumerates every chain and interface
in it -- which also means the per-instance selection below decides what we FOLD, while the entry
decides what the loss can be built for.

THE SAbDab2 INSTANCE ID IS STILL THE KEY for folding. SAbDab2 ships one cif per antibody-antigen
instance and the INSTANCE column IS the file stem: pdb_00008jqn_H_L -> pdb_00008jqn_H_L.cif. The
chain letters are CASE SENSITIVE (8jqn has both _H_L and _h_l, different complexes), and a
single-domain antibody has no light chain, which SAbDab2 writes as "+": pdb_00009nk9_A_+.cif. 110 of
535 selected structures are such nanobodies, so rebuilding the name from Hchain and Lchain -- which
gives "<pdb>_A_" for those -- resolved none of them. Using the instance id as given avoids inventing
a naming convention that the data already states.

THE CLUSTER FILE is RCSB's clusters-by-entity-40.txt format: one line per cluster, whitespace
separated <PDBID>_<entity_id>. It drives the weighted sampler, which weights a datapoint by
1/cluster_size. Our selection already caps 2 structures per (antibody cluster, antigen cluster)
pairing, so the redundancy that weighting exists to damp is largely gone; passing no cluster file
gives every datapoint cluster_size 1, i.e. uniform weights over an already-deduplicated set. Pass
--cluster-file to use RCSB's instead. This choice is recorded rather than defaulted silently.

USAGE
    phase_j_prepare_training_data.py                 # resolve and gate only, write the file list
    phase_j_prepare_training_data.py --run --n-cpu 64  # then invoke prepare_training_data.py
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTENIX = "/home/ebrahim/Protenix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/results/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    # JOB CIFS BY DEFAULT, not whole entries. A whole entry builds the deposited assembly -- every
    # copy, every chain -- which is NOT what we fold, so a label built from it cannot be compared to
    # a pose. The default pointed at rcsb_training_cifs, so anyone running this without the explicit
    # flag silently prepared the wrong system.
    ap.add_argument("--cif-dir", default=f"{ROOT}/train_prep/job_cifs",
                    help="per-JOB mmCIFs from phase_j_job_cifs.py: the antibody and its polypeptide "
                         "antigen only. Pass data/rcsb_training_cifs only to prepare whole entries.")
    # FETCHING IS OFF BY DEFAULT NOW. A missing job cif means phase_j_job_cifs.py has not run for
    # that structure; downloading the whole entry instead would silently substitute the assembly for
    # the job, which is the exact mismatch this phase exists to avoid.
    ap.add_argument("--fetch", action="store_true",
                    help="download a missing entry from RCSB. Only meaningful with "
                         "--cif-dir data/rcsb_training_cifs.")
    # OUTPUTS MATCH THE INPUT. --cif-dir now defaults to the JOB cifs, so the outputs must be the
    # job-level ones too. Left at indices.csv / bioassembly, a default run read job cifs and wrote
    # over the whole-entry outputs, while phase_j_dataset.py read job_indices.csv / job_bioassembly
    # -- which that run never wrote. The two halves would have disagreed silently, the dataset
    # loading whatever stale job-level files happened to be on disk.
    ap.add_argument("--out-csv", default=f"{ROOT}/train_prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/train_prep/job_bioassembly")
    ap.add_argument("--file-list", default=f"{ROOT}/train_prep/job_cifs.txt")
    ap.add_argument("--cluster-file", default=None)
    ap.add_argument("--n-cpu", type=int, default=32)
    ap.add_argument("--run", action="store_true", help="actually invoke prepare_training_data.py")
    ap.add_argument("--skipped", default=f"{ROOT}/train_prep/skipped.csv",
                    help="structures Protenix's parser cannot build, with the reason. A failure not "
                         "already listed here is a STOP; one that is listed is skipped.")
    a = ap.parse_args()

    sel = list(csv.DictReader(open(a.selection)))
    if not sel:
        raise SystemExit(f"STOP: {a.selection} is empty")
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}

    import urllib.request
    os.makedirs(a.cif_dir, exist_ok=True)
    pdbs = sorted({r["pdb"] for r in sel})
    paths, missing, fetched = [], [], 0
    for pdb in pdbs:
        dst = os.path.join(a.cif_dir, f"{pdb}.cif")
        if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            if not a.fetch:
                missing.append((pdb, "", "absent; run phase_j_job_cifs.py (or pass --fetch "
                                               "to download a whole entry, which is NOT the job)"))
                continue
            # .part THEN RENAME: a truncated cif parses without complaint and yields a short chain
            # set, which would silently cost interfaces in the label the loss is built from.
            tmp = dst + ".part"
            try:
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.cif", tmp)
                os.replace(tmp, dst)
                fetched += 1
            except Exception as e:                      # noqa: BLE001
                if os.path.exists(tmp):
                    os.remove(tmp)
                missing.append((pdb, "", f"download failed: {type(e).__name__}"))
                continue
        paths.append(dst)
    if fetched:
        print(f"fetched {fetched} mmCIF(s) from RCSB into {a.cif_dir}")

    # A MISSING CIF IS A STOP, not a skip. Every structure in the frozen selection is one the cache
    # is costed for; silently preparing 500 of 535 would make the cache disagree with the manifest
    # that describes it, and nothing downstream re-checks the count.
    if missing:
        for pdb, _inst, why in missing[:10]:
            print(f"  MISSING {pdb}: {why}")
        raise SystemExit(f"STOP: {len(missing)} of {len(pdbs)} selected entries have no mmCIF.")

    dup = len(paths) - len(set(paths))
    if dup:
        raise SystemExit(f"STOP: {dup} selected rows resolve to the SAME cif; the per-instance "
                         "naming should make that impossible, so the selection is inconsistent.")

    os.makedirs(os.path.dirname(a.file_list), exist_ok=True)
    os.makedirs(a.bio_dir, exist_ok=True)
    with open(a.file_list, "w") as fh:
        fh.write("\n".join(paths) + "\n")
    print(f"resolved {len(paths)} mmCIF(s) for {len(sel)} selected rows "
          f"({len(pdbs)} distinct entries) -> {a.file_list}")

    cmd = [f"{PROTENIX}/.venv/bin/python", f"{PROTENIX}/scripts/prepare_training_data.py",
           "-i", a.file_list, "-o", a.out_csv, "-b", a.bio_dir, "-n", str(a.n_cpu)]
    if a.cluster_file:
        cmd += ["-c", a.cluster_file]
    print("  " + " ".join(cmd))
    if not a.run:
        print("dry run: pass --run to execute")
        return
    env = dict(os.environ, PYTHONPATH=f"{PROTENIX}:{os.environ.get('PYTHONPATH', '')}")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"STOP: prepare_training_data.py exited {rc}")
    built = {f[:-7] for f in os.listdir(a.bio_dir) if f.endswith(".pkl.gz")}
    n_rows = sum(1 for _ in open(a.out_csv)) - 1 if os.path.exists(a.out_csv) else 0
    print(f"bioassembly pickles: {len(built)} of {len(paths)} cifs")
    print(f"indices rows: {n_rows}")

    # SKIP WITH A RECORDED REASON, never accommodate. A structure Protenix's parser cannot build is
    # dropped and written down; nothing about the run is relaxed to make it fit. A failure that is
    # NOT already in the skip file stops the phase, so a new one cannot join the old ones silently.
    known = {}
    if os.path.exists(a.skipped):
        known = {r["pdb"]: r.get("reason", "") for r in csv.DictReader(open(a.skipped))}
    failed = sorted({os.path.basename(p)[:-4] for p in paths} - built)
    new = [f for f in failed if f not in known]
    if new:
        raise SystemExit(
            f"STOP: {len(new)} cif(s) produced no bioassembly dict and are not in {a.skipped}: "
            f"{new[:10]}\nRun each one alone to get its traceback, then add it with a reason.")
    if failed:
        for f in failed:
            print(f"  SKIPPED {f}: {known[f]}")
        print(f"{len(failed)} structure(s) skipped with a recorded reason; "
              f"{len(built)} prepared")

    # THE PREPARED SET MUST BE THE SELECTION. train_prep/ outlived a selection once already: it was
    # built at 535 rows, the selection was then rewritten to 501, and the directory kept 58
    # structures no longer selected while 25 selected ones had no prepared input at all -- a cache
    # built from it would have been both short and contaminated, with nothing comparing the two.
    want = {r["pdb"] for r in sel} - set(known)
    extra, short = sorted(built - want), sorted(want - built)
    if short:
        raise SystemExit(f"STOP: {len(short)} selected structure(s) have no bioassembly dict: "
                         f"{short[:10]}")
    if extra:
        print(f"  {len(extra)} prepared structure(s) are no longer selected: {extra[:6]}")
        for f in extra:
            os.remove(os.path.join(a.bio_dir, f + ".pkl.gz"))
        print(f"  removed them; {len(built) - len(extra)} remain, matching the selection")


if __name__ == "__main__":
    main()
