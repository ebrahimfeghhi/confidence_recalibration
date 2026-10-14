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

SELF-TEMPLATING IS BLOCKED BY DATE, which is the whole reason this phase does more than call
Protenix's search.

    Protenix filters template hits by release date, but the per-query part of that rule is
    TRAIN-ONLY: template_featurizer.py:443 sets query_release_date only when stage == "train", so at
    inference the cutoff collapses to the global max_template_date, 2021-09-30. The template database
    covers depositions to 2022-09-28, and 48 of the 79 validation structures were deposited on or
    before 2021-09-30 -- so 60% of them could be handed their OWN deposited entry as a template at
    100% identity, and validation is what picks the checkpoint.

    Since neither update_template_info() nor the featurizer accepts a per-query date on this path,
    this phase filters each hmmsearch.a3m itself, applying Protenix's own training rule: drop any hit
    released on or after (the query's release date - DAYS_BEFORE_QUERY_DATE), which is 60 days in
    template_utils.py:72. A hit whose date is unknown is dropped, matching their prefilter, which
    returns False with "date unknown" rather than admitting it.

    A SHARED MSA DIRECTORY TAKES THE STRICTEST CUTOFF. Directories are keyed by sequence, so a chain
    used by two structures has one template file; it gets the earliest of their cutoffs.

    THE CONSEQUENCE, ACCEPTED DELIBERATELY: an old structure ends up with few templates or none, since
    almost the entire database postdates it. That is the honest reading of "templates must predate the
    query", and it is worth a sentence in the write-up: 46 of the 79 validation structures are
    template-light for this reason while everything else is template-bearing.

THIS PHASE MUST ALSO BE RUN OVER THE TRAINING FOLDS, for the same reason and more urgently.
    The confidence head is trained on poses from the 557-cluster pool, of which 1,354 of 2,886
    entries (47%) were deposited on or before 2021-09-30. Trained on self-templated poses, the head
    learns to be confident under conditions it will never meet at evaluation -- which is worse than
    an inflated validation number, because it corrupts the thing being fitted rather than the thing
    being measured.

    Every pool structure has a date, so the filter applies to all of them: SAbDab2 supplies a
    deposition date for 2,886 of 2,886, and Protenix's cache covers 2,002 of those.

    ENFORCEMENT IS STRUCTURAL, NOT PROCEDURAL. This phase writes <templatesPath>.filtered beside each
    template file, recording the cutoff it applied, and phase_i_fold_validation.sh refuses to fold any
    job whose templatesPath lacks that marker. So a fold cannot silently use unfiltered templates,
    whichever job set it is given.

WHY THE GLOBAL CAP IS KEPT at 2021-09-30 rather than dropped to match Protenix's training stage
    (which uses 3000-01-01, i.e. no cap). Three reasons, the last decisive:
      - pxm2025 is entirely post-2021: 90 of its 123 complexes have dates, spanning 2023-08-26 to
        2025-11-12, all after the cap. So for pxm2025 the cap already IS a complete self-exclusion
        rule, and there is no leak to fix.
      - 33 of those 123 have no date in any source, so a per-query rule could not be applied there
        uniformly anyway.
      - the benchmarks will be refolded to apply the retrained head. Widening their templates at the
        same time would mean a change in DockQ or ipTM could be the head or the templates, with no way
        to tell which. The refold has to vary one thing.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

# PROTENIX_ROOT_DIR IS SET HERE, not left to the caller. template_search.py downloads
# pdb_seqres_2022_09_28.fasta to $PROTENIX_ROOT_DIR/search_database/ on first use, and with the
# variable unset that lands in $HOME -- outside tts_abag, which everything in this project is
# supposed to stay inside, and on a quota that is not the one we manage.
PROTENIX = os.environ.get("PROTENIX_DIR", "/home/ebrahim/Protenix")
os.environ.setdefault("PROTENIX_ROOT_DIR", "/home/ebrahim/tts_abag/extra_training/protenix_assets")
os.makedirs(os.path.join(os.environ["PROTENIX_ROOT_DIR"], "search_database"), exist_ok=True)
sys.path.insert(0, PROTENIX)

from runner.template_search import update_template_info  # noqa: E402


def load_release_dates(path):
    """in: Protenix's release_date_cache.json.  out: {pdb_id_lower: date string}"""
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        d = v.get("release_date") if isinstance(v, dict) else v
        if d:
            out[k.lower()] = d
    return out


def query_cutoffs(files, rel, splits_csv, days, global_cap="2021-09-30"):
    """in: the job jsons, {pdb: date}, SAbDab2's splits csv, the day margin.
    out: ({msa_dir: cutoff date string}, {pdb: cutoff}, [pdb without any date])

    `global_cap` is the model's own template cutoff, 2021-09-30 for Protenix v2 at inference. Keeping
    it means a post-2021 validation structure is treated exactly as a pxm2025 fold is.

    The cutoff for a structure is its own release date minus `days`, which is Protenix's training
    rule. Release date comes from their cache where present, and falls back to SAbDab2's deposition
    date -- 9n6e, for one, is absent from the cache.
    """
    import csv as _csv
    from datetime import datetime, timedelta
    depo = {}
    if os.path.exists(splits_csv):
        with open(splits_csv) as fh:
            for r in _csv.DictReader(fh):
                pid = str(r.get("PDB_ID", ""))[-4:].lower()
                if pid and r.get("PDBdepo"):
                    depo.setdefault(pid, r["PDBdepo"][:10])
    per_pdb, per_dir, undated = {}, {}, []
    for f in files:
        # FROM THE JSON'S OWN "name", not the filename. Deriving the pdb id from the file name ties
        # the date lookup to a naming convention -- and that convention had to change once already
        # (phase G now writes <pdb>-final-updated.json because folder.py loads that), which would
        # silently have broken every cutoff. The name inside the file is what the job calls itself.
        try:
            tasks = json.load(open(f))
            pdb = str(tasks[0].get("name", "")).lower()
        except Exception:                                       # noqa: BLE001
            pdb = ""
        if not pdb:
            pdb = os.path.basename(f).split("-")[0].removesuffix(".json").lower()
            print(f"  NOTE {os.path.basename(f)} has no name field; falling back to {pdb!r}")
        raw = rel.get(pdb) or depo.get(pdb)
        if not raw:
            undated.append(pdb)
            continue
        # THE MINIMUM OF BOTH RULES, which is what Protenix computes
        # (template_featurizer.py:358-361): the per-query date keeps a structure from seeing its own
        # era, and the global cap keeps it from seeing anything the model was never trained on. For a
        # post-2021 validation structure the global cap binds, which is exactly what pxm2025 folds
        # get; for a pre-2021 one its own date binds, which is the self-templating fix.
        own = (datetime.strptime(raw[:10], "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
        cut = min(own, global_cap)
        per_pdb[pdb] = cut
        for task in json.load(open(f)):
            for sq in task["sequences"]:
                c = sq.get("proteinChain")
                tp = c and c.get("templatesPath")
                if not tp:
                    continue
                d = os.path.dirname(tp)
                # STRICTEST WINS: one directory, many structures, earliest cutoff.
                if d not in per_dir or cut < per_dir[d]:
                    per_dir[d] = cut
    return per_dir, per_pdb, undated


FILTER_VERSION = 2          # bump when the filtering rule changes, so old markers stop validating


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def filter_a3m(path, cutoff, rel):
    """in: an hmmsearch.a3m, a cutoff date, {pdb: date}.  out: (kept, dropped, dropped_undated)

    Records are ">pdbid_chain/range ...". A hit is kept only if its own release date is strictly
    before the cutoff; an unknown date is dropped, which is what Protenix's prefilter does.

    DERIVED FROM A PRESERVED RAW COPY, NEVER FILTERED IN PLACE. Filtering the file itself destroyed
    hits irreversibly: apply an early cutoff, later relax it, and the removed templates never come
    back -- the only way to recover them was to delete the file and re-run hmmsearch. The first call
    moves the search output to hmmsearch.raw.a3m and every filtered file is regenerated from it, so
    a changed cutoff is just a re-derivation.

    THE MARKER IS BOUND TO THE FILE, not merely to the cutoff. It records the filtered file's own
    sha256, the cutoff and the filter version, so replacing a filtered file with raw templates while
    keeping its marker no longer passes: the hash will not match. phase_i checks the same three
    values before folding.
    """
    raw = path[:-4] + ".raw.a3m" if path.endswith(".a3m") else path + ".raw"
    marker = path + ".filtered"
    if not os.path.exists(raw):
        if not os.path.exists(path):
            return None
        os.rename(path, raw)
    if os.path.exists(marker) and os.path.exists(path):
        try:
            m = json.load(open(marker))
            if (m.get("cutoff") == cutoff and m.get("version") == FILTER_VERSION
                    and m.get("sha256") == _sha(path)):
                return None            # already filtered at this cutoff, and the file is unchanged
            # A STRICTER EXISTING FILTER IS NEVER LOOSENED. Template directories are keyed by
            # SEQUENCE, so one directory serves every structure using that chain, and query_cutoffs()
            # only sees the jobs of the CURRENT invocation. Run phase H on a job set that excludes the
            # oldest user of a directory and the recomputed cutoff is later than the one already
            # applied; re-filtering from the raw file would hand that older structure templates from
            # its own era. Keeping the stricter filter is correct for every user -- the newer query
            # merely sees fewer templates -- and phase_i_inputs.py accepts any marker at least as
            # strict as the job set requires (it fails only on `got > need`).
            old = m.get("cutoff")
            if (isinstance(old, str) and old < cutoff and m.get("version") == FILTER_VERSION
                    and m.get("sha256") == _sha(path)):
                print(f"  NOTE keeping stricter cutoff {old} on {os.path.basename(path)} "
                      f"(this job set would have applied {cutoff})")
                return None
        except Exception:                                       # noqa: BLE001
            pass
    kept_lines, kept, dropped, undated = [], 0, 0, 0
    keep_this = False
    with open(raw) as fh:
        for line in fh:
            if line.startswith(">"):
                pdb = line[1:5].lower()
                d = rel.get(pdb)
                if d is None:
                    keep_this = False; undated += 1
                elif d[:10] < cutoff:
                    keep_this = True; kept += 1
                else:
                    keep_this = False; dropped += 1
            if keep_this:
                kept_lines.append(line)
    with open(path, "w") as fh:
        fh.writelines(kept_lines)
    with open(marker, "w") as fh:
        json.dump({"cutoff": cutoff, "version": FILTER_VERSION, "sha256": _sha(path),
                   "raw": os.path.basename(raw), "kept": kept, "dropped_recent": dropped,
                   "dropped_undated": undated}, fh)
    return kept, dropped, undated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", default="/home/ebrahim/tts_abag/extra_training/val_jobs")
    ap.add_argument("--msa-dir", default="/home/ebrahim/tts_abag/extra_training/building_msa/work/mmcif_msa")
    ap.add_argument("--release-dates",
                    default=os.path.expanduser("~/common/release_date_cache.json"),
                    help="Protenix's release_date_cache.json; the featurizer reads the same file")
    ap.add_argument("--splits", default="/home/ebrahim/tts_abag/extra_training/SAbDab2_data/"
                                        "zenodo_splits/splits_final/abag_split.csv",
                    help="fallback deposition dates for structures absent from the cache")
    ap.add_argument("--days-before", type=int, default=60,
                    help="Protenix's DAYS_BEFORE_QUERY_DATE (template_utils.py:72)")
    ap.add_argument("--global-cap", default="2021-09-30",
                    help="the model's own template cutoff, which Protenix applies at inference. The "
                         "effective cutoff is the earlier of this and (query date - days-before).")
    ap.add_argument("--no-date-filter", action="store_true",
                    help="skip the self-templating filter. Only for measuring its effect.")
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

    # ---------------------------------------------------------------- the self-templating filter
    # AFTER the search, because update_template_info() has no per-query date argument and the
    # featurizer's own per-query rule is train-only. See the header for why this is necessary rather
    # than optional: without it, 48 of 79 validation structures could be handed their own entry.
    if a.no_date_filter:
        print("\nWARNING --no-date-filter: templates are NOT date-filtered, so a structure deposited")
        print("        before 2021-09-30 may be handed its own entry at 100% identity. Measurement only.")
    else:
        rel = load_release_dates(a.release_dates)
        if not rel:
            sys.exit(f"STOP: no release dates at {a.release_dates}. Without them every hit's date is "
                     "unknown, and the filter cannot distinguish a legitimate template from the "
                     "query's own entry. Protenix's featurizer reads this same file.")
        per_dir, per_pdb, undated = query_cutoffs(files, rel, a.splits, a.days_before, a.global_cap)
        print(f"\nself-templating filter: {len(rel):,} release dates, "
              f"{len(per_pdb)} structures dated, {len(undated)} undated")
        if undated:
            sys.exit(f"STOP: no release or deposition date for {undated[:5]} ({len(undated)} total). "
                     "A structure with no date has no cutoff, so its templates cannot be filtered.")
        own = sum(1 for c in per_pdb.values() if c < a.global_cap)
        print(f"  cutoff is the structure's own date for {own}, the global cap for {len(per_pdb)-own}")
        tot_kept = tot_drop = tot_undated = n_filtered = n_empty = 0
        rows = []
        for d, cut in sorted(per_dir.items()):
            f_a3m = os.path.join(d, "hmmsearch.a3m")
            if not os.path.exists(f_a3m):
                continue
            res = filter_a3m(f_a3m, cut, rel)
            if res is None:
                continue
            kept, dropped, und = res
            n_filtered += 1
            tot_kept += kept; tot_drop += dropped; tot_undated += und
            if kept == 0:
                n_empty += 1
            rows.append((os.path.basename(d), cut, kept, dropped, und))
        print(f"  filtered {n_filtered} template files: kept {tot_kept:,} hits, "
              f"dropped {tot_drop:,} as too recent and {tot_undated:,} as undated")
        print(f"  files left with no templates at all: {n_empty} of {n_filtered}")
        rep = os.path.join(os.path.dirname(a.msa_dir.rstrip("/")), "phase_h_template_filter.csv")
        with open(rep, "w") as fh:
            fh.write("msa_dir,cutoff,kept,dropped_recent,dropped_undated\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        print(f"  per-chain detail in {rep}")

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
