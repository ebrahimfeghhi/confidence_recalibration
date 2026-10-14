#!/usr/bin/env python
"""Phase I's pre-GPU input check. Called by phase_i_fold_validation.sh before it takes a card.

EVERY INPUT VERIFIED BEFORE 16 HOURS OF GPU IS COMMITTED, and by content rather than by count. A job
json whose templatesPath or MSA path does not exist produces a fold that runs and scores nothing,
which is the most expensive way to discover a missing file.

THE TEMPLATE CHECK RECOMPUTES THE CUTOFF, it does not merely trust the marker. Phase H writes
<templatesPath>.filtered holding {cutoff, version, sha256}. Checking the hash alone proves the file
has not changed since filtering -- it does not prove the file was filtered for THIS query. One MSA
directory is shared by every structure with that chain sequence, and phase H gives the directory the
earliest cutoff among its users. If phase H last ran on a job set that did not include the oldest
structure, the marker holds a later cutoff, the hash still matches, and a pre-2021 structure receives
templates from its own era -- its own entry at 100% identity among them. So this recomputes the
cutoff the CURRENT job set requires, using phase H's own query_cutoffs(), and requires the marker to
be at least that strict, at the current filter version, with a hash that matches the file.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

# the repo root, from this file, so renaming the checkout needs no edit here
_REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)


def load_phase_h(d):
    """import phase H as a module, so the cutoff rule has exactly one implementation"""
    p = os.path.join(d, "phase_h_search_templates.py")
    spec = importlib.util.spec_from_file_location("phase_h", p)
    m = importlib.util.module_from_spec(spec)
    os.environ.setdefault("PROTENIX_ROOT_DIR", os.path.expanduser("~"))
    spec.loader.exec_module(m)
    return m


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--suffix", default="-final-updated.json")
    ap.add_argument("--release-dates", default=os.path.expanduser("~/common/release_date_cache.json"))
    ap.add_argument("--splits", default=f"{_REPO}/extra_training/SAbDab2_data/"
                                        "zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--days-before", type=int, default=60)
    ap.add_argument("--global-cap", default="2021-09-30")
    a = ap.parse_args()

    H = load_phase_h(a.phases)
    files = sorted(glob.glob(os.path.join(a.jobs, "*" + a.suffix)))
    if not files:
        print(f"FAIL no job files matching *{a.suffix} in {a.jobs}")
        return 1

    rel = H.load_release_dates(a.release_dates)
    per_dir, per_pdb, undated = H.query_cutoffs(files, rel, a.splits, a.days_before, a.global_cap)
    # AN UNDATED QUERY IS A STOP, not a warning. Without a date there is no cutoff to enforce, so
    # nothing rules out the structure seeing its own era.
    if undated:
        print(f"FAIL {len(undated)} job(s) have no release or deposition date, so no cutoff can be "
              f"enforced: {sorted(set(undated))[:6]}")
        return 1

    bad, n_chain = [], 0
    for f in files:
        base = os.path.basename(f)
        try:
            tasks = json.load(open(f))
        except (OSError, ValueError) as e:
            bad.append(f"{base} unreadable: {e}")
            continue
        pdb = str(tasks[0].get("name", "")).lower()
        for task in tasks:
            for s in task["sequences"]:
                c = s.get("proteinChain")
                if not c:
                    continue
                n_chain += 1
                # THE SCHEMA THE FEATURISER REQUIRES, checked before a card is taken. This gate
                # verified that every path resolves but not that the json is loadable, so a
                # `"count": "1"` string sailed through it and died inside the dataloader --
                # `range(1, entity["count"] + 1)` raises TypeError on a str. The fold then
                # "completes" in 0.04s having written nothing, which is the most confusing possible
                # failure: a worker that exits successfully with no pose.
                cnt = c.get("count")
                if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 1:
                    bad.append(f"{base} count={cnt!r} ({type(cnt).__name__}); the featuriser does "
                               f"range(1, count + 1) and needs an int >= 1")
                sq = c.get("sequence")
                if not isinstance(sq, str) or not sq:
                    bad.append(f"{base} sequence is {type(sq).__name__}, expected a non-empty str")
                for key in ("pairedMsaPath", "unpairedMsaPath", "templatesPath"):
                    p = c.get(key)
                    if not p or not os.path.exists(p):
                        bad.append(f"{base} {key}={p}")
                        continue
                    if key != "templatesPath":
                        continue
                    mk = p + ".filtered"
                    if not os.path.exists(mk):
                        bad.append(f"{base} templates never date-filtered: {p}")
                        continue
                    try:
                        m = json.load(open(mk))
                    except (OSError, ValueError):
                        bad.append(f"{base} unreadable filter marker: {mk}")
                        continue
                    need = per_dir.get(os.path.dirname(p)) or per_pdb.get(pdb)
                    if need is None:
                        bad.append(f"{base} no cutoff computed for {p}")
                        continue
                    got = m.get("cutoff")
                    # AT LEAST AS STRICT. A marker cutoff later than required means this query can
                    # see structures from its own era; an earlier one is merely conservative.
                    if not isinstance(got, str) or got > need:
                        bad.append(f"{base} filtered at cutoff {got}, but this job set requires "
                                   f"{need} or earlier (self-templating window): {p}")
                        continue
                    if m.get("version") != H.FILTER_VERSION:
                        bad.append(f"{base} marker filter version {m.get('version')}, current is "
                                   f"{H.FILTER_VERSION} -- the filtering rule changed: {p}")
                        continue
                    if m.get("sha256") != sha(p):
                        bad.append(f"{base} template file does not match its marker hash "
                                   f"(replaced since filtering?): {p}")

    cuts = sorted(set(per_pdb.values()))
    print(f"jobs {len(files)}  protein chains {n_chain}  template dirs {len(per_dir)}")
    print(f"required cutoffs: {len(cuts)} distinct, {cuts[0]} .. {cuts[-1]} "
          f"(global cap {a.global_cap}, margin {a.days_before}d)")
    pre = [p for p, c in per_pdb.items() if c < a.global_cap]
    print(f"  {len(pre)} job(s) bound by their OWN date rather than the cap -- these are the ones a "
          f"stale marker would let self-template")
    if bad:
        print(f"FAIL {len(bad)} problem(s):")
        for b in bad[:8]:
            print("   " + b)
        return 1
    print("inputs OK: every MSA and template path resolves; every template file is filtered at or "
          "below the cutoff this job set requires, at the current filter version, hash matching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
