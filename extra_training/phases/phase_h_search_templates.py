#!/usr/bin/env python
"""PHASE H -- find structural templates for every validation chain, with Protenix's own search.

WHAT A TEMPLATE IS. A known, already-solved structure whose sequence resembles our chain. Handing
one to the model gives it a starting guess for the fold, which is a different kind of information
from an alignment: the alignment says how the sequence varies, the template says what a relative
actually looked like.

WHY A DRIVER IS NEEDED
    runner/template_search.py exposes update_template_info(json_data, ...), which for every protein
    chain lacking a usable templatesPath runs hmmsearch in that chain's MSA directory and writes
    hmmsearch.a3m beside pairing.a3m / non_pairing.a3m. Protenix calls it from runner/batch_inference.py
    ONLY. Our folds go through runner/inference.py (via folder.py), which never calls it, so without
    this step every job json would point at an hmmsearch.a3m that does not exist.

WHY UP FRONT AND SINGLE-PROCESS
    MSA directories are keyed by SEQUENCE, so two complexes that share a chain share its directory.
    Two fold workers reaching that chain at the same time would run hmmsearch into the same path
    concurrently. Doing it once, before any folding, removes the race and makes the cost visible
    instead of hiding it inside the first fold of each complex.

    pxm2025 folds use templates (paths.Dataset.free_templates defaults True, and their job jsons carry
    real hmmsearch.a3m files), so the validation set is built the same way.

THE DATABASE is pdb_seqres_2022_09_28.fasta. template_search.py downloads it to
    $PROTENIX_ROOT_DIR/search_database/ on first use, so PROTENIX_ROOT_DIR is set by the caller to keep
    it inside tts_abag rather than $HOME.

INPUTS   --jobs-dir: the directory of validation job jsons. Each is rewritten in place with its
                     templatesPath fields pointing at the hmmsearch.a3m that now exists.
         --msa-dir:  where the per-chain MSA directories are, for the final count.
OUTPUTS  <msa_dir>/<chain>/hmmsearch.a3m per unique chain sequence, and a report on stdout.
         Exit code 1 if any protein chain still lacks a template file.

AN OPEN METHODOLOGICAL POINT, recorded rather than silently decided. pxm2025 folds use templates and
FoldBench folds do not. A confidence head calibrated on template-bearing poses would then be
evaluated on FoldBench without them. The validation set is built WITH templates to match pxm2025,
which is the arm it is compared against, but this difference has to be stated in the write-up.
"""
import argparse
import glob
import json
import os
import sys

# PROTENIX_ROOT_DIR IS SET HERE, not left to the caller. template_search.py downloads
# pdb_seqres_2022_09_28.fasta to $PROTENIX_ROOT_DIR/search_database/ on first use, and with the
# variable unset that lands in $HOME -- outside tts_abag, which everything in this project is
# supposed to stay inside, and on a quota that is not the one we manage.
PROTENIX = os.environ.get("PROTENIX_DIR", "/home/ebrahim/Protenix")
os.environ.setdefault("PROTENIX_ROOT_DIR", "/home/ebrahim/tts_abag/extra_training/protenix_root")
os.makedirs(os.path.join(os.environ["PROTENIX_ROOT_DIR"], "search_database"), exist_ok=True)
sys.path.insert(0, PROTENIX)

from runner.template_search import update_template_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", default="/home/ebrahim/tts_abag/extra_training/val_jobs")
    ap.add_argument("--msa-dir", default="/home/ebrahim/tts_abag/extra_training/msa_work/mmcif_msa")
    a = ap.parse_args()
    print(f"PROTENIX_ROOT_DIR={os.environ['PROTENIX_ROOT_DIR']}")

    files = sorted(f for f in glob.glob(f"{a.jobs_dir}/*.json"))
    if not files:
        sys.exit(f"STOP: no job jsons in {a.jobs_dir}")
    print(f"job jsons: {len(files)}")

    updated, failed, chains, already = 0, [], 0, 0
    for i, f in enumerate(files, 1):
        data = json.load(open(f))
        n_before = sum(1 for t in data for s in t["sequences"] if "proteinChain" in s)
        try:
            changed = update_template_info(data)
        except Exception as e:                                   # noqa: BLE001
            failed.append((os.path.basename(f), repr(e)[:200]))
            continue
        if changed:
            with open(f, "w") as fh:
                json.dump(data, fh, indent=2)
            updated += 1
        else:
            already += 1
        chains += n_before
        # verify, per chain, that the file is actually there now
        for t in data:
            for s in t["sequences"]:
                c = s.get("proteinChain")
                if not c:
                    continue
                tp = c.get("templatesPath")
                if not tp or not os.path.exists(tp):
                    failed.append((os.path.basename(f), f"no template for a {len(c['sequence'])} aa chain"))
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} jsons   templates written for {updated}   "
                  f"already had them {already}   problems {len(failed)}", flush=True)

    print(f"\nprotein chains seen: {chains}")
    print(f"jsons updated: {updated}   already complete: {already}   problems: {len(failed)}")
    for n, why in failed[:20]:
        print(f"  PROBLEM {n}: {why}")
    if failed:
        sys.exit(f"STOP: {len(failed)} chains have no template file")

    # The per-chain count, which the per-json check above does not give: MSA directories are keyed by
    # sequence, so chains shared between complexes are one file checked many times.
    n_tmpl = sum(1 for d in os.listdir(a.msa_dir)
                 if os.path.isfile(os.path.join(a.msa_dir, d, "hmmsearch.a3m")))
    n_msa = sum(1 for d in os.listdir(a.msa_dir)
                if os.path.isfile(os.path.join(a.msa_dir, d, "pairing.a3m")))
    print(f"hmmsearch.a3m on disk: {n_tmpl} of {n_msa} chain directories")
    print("every protein chain in every validation json has an hmmsearch.a3m")


if __name__ == "__main__":
    main()
