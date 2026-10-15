#!/usr/bin/env python
"""PHASE J -- write the CORRECTED job jsons for the structures on the refold list.

WHAT IT CHANGES. A role's `proteinChain.sequence` is replaced by the corrected query
`phase_j_job_cifs.py --native-gaps` computed for it, which restores the internal native residues the
original query dropped -- AND its `pairedMsaPath`, `unpairedMsaPath` and `templatesPath`, which must
move to the corrected MSA store. Everything else is copied through untouched.

THE PATHS ARE NOT OPTIONAL. The first version changed only the sequence, so a corrected 221-residue
query kept pointing at `mmcif_msa/<dir>/pairing.a3m` -- an alignment with 214 match columns, built
for the sequence this one replaced. An a3m is one match column per query residue, so that pairing is
a length contradiction: at best the featurizer rejects it, at worst it is absorbed silently. The
templates are the same problem, built by hmmsearch against the old query. A job is therefore REFUSED
if its MSA paths would still resolve into the old store.

WHAT IT REFUSES TO TOUCH. Only runs classified `omission`. A `linker` belongs to neither role and an
`unresolved` or `ambiguous` run has not been judged, so a structure carrying one of those in the role
being corrected is SKIPPED with its reason, not partially corrected. That is why 8dpl is absent from
the output: its 115-136 run begins TVSSGTGG, the canonical end of a VH, so where the domain stops and
the tether starts is a residue-level decision nobody has taken.

MATCHED BY SEQUENCE, NOT BY POSITION. The role -> chain assignment lives in the splits row, not in
the json, and the json's chain order is not guaranteed to follow it. So the chain to correct is found
by an EXACT match against the old query string recorded in the gap report. A json with no exact match,
or more than one, is refused -- that means the report and the json disagree about what was folded,
which is not something to resolve by guessing.
"""
import argparse
import collections
import csv
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ACTIONABLE = "omission"
# The files that ARE multiple-sequence alignments. hmmsearch.a3m is a template hit list.
MSA_FILES = ("pairing.a3m", "non_pairing.a3m")
BLOCKING = ("unresolved", "ambiguous", "linker?")



def msa_mismatch(path, corrected):
    """None if `path` is an a3m FOR THIS corrected query, else why it is not.

    EXISTENCE IS NOT ENOUGH, and successful parsing is not either. The gate checked only that the
    file was there, so an unrelated MSA satisfied `--expect 1`. Worse, Protenix's reader silently
    TRUNCATES rows longer than the query and PADS shorter ones, so a wrong-length alignment does not
    raise -- it quietly becomes a different alignment. Both things the model would be trained on.

    So: the query row must equal the corrected sequence exactly, and every row must carry exactly
    len(corrected) match columns, since an a3m is one match column per query residue by definition.
    """
    try:
        rows, h, buf = [], None, []
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if h is not None:
                        rows.append("".join(buf))
                    h, buf = line, []
                else:
                    buf.append(line)
        if h is not None:
            rows.append("".join(buf))
    except OSError as e:
        return f"could not be read: {e}"
    if not rows:
        return "is empty"
    q = rows[0].replace("-", "").upper()
    if q != corrected.strip().upper():
        return (f"is an a3m for a DIFFERENT query: its query row is {len(q)} residues, this role's "
                f"corrected query is {len(corrected)}")
    want = len(corrected)
    for i, r in enumerate(rows):
        mc = sum(1 for c in r if not c.islower())
        if mc != want:
            return f"row {i} has {mc} match columns, the corrected query has {want}"
    return None


def under(path, root):
    """True when `path` is inside `root`. A BARE startswith IS WRONG HERE: the corrected store is
    `mmcif_msa_corrected` and the old one is `mmcif_msa`, so the corrected path starts with the old
    root and every job was rejected for pointing at a store it had just been moved out of. The
    separator is what makes it a directory boundary rather than a string prefix.
    """
    r = os.path.normpath(str(root))
    p = os.path.normpath(str(path))
    return p == r or p.startswith(r + os.sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True, help="the --native-gaps csv from phase_j_job_cifs.py")
    ap.add_argument("--jobs-dir", default=f"{ROOT}/training/jobs",
                    help="the ORIGINAL job jsons, read only")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/jobs_corrected")
    ap.add_argument("--seq-to-dir",
                    default=f"{ROOT}/building_msa/queries_corrected/seq_to_dir.json",
                    help="the sequence -> directory index from phase_j_corrected_queries.py. The "
                         "directory a corrected MSA lands in is named from the FASTA header, so "
                         "inventing a name here instead of reading it is how a whole search gets "
                         "written under names no job points at")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa_corrected",
                    help="the CORRECTED msa store the rewritten jobs must point at")
    ap.add_argument("--old-msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa",
                    help="the original store; a job still pointing here is refused")
    ap.add_argument("--expect", type=int, default=0,
                    help="how many jobs this run MUST write. With it, the run is a readiness GATE: "
                         "anything less exits non-zero and no stale file is left behind")
    ap.add_argument("--allow-missing-msa", action="store_true",
                    help="write the corrected paths even though the MSAs are not built yet. For "
                         "preparing jobs ahead of the rebuild; a fold must NOT be queued from these")
    ap.add_argument("--report", default=None, help="csv of what was corrected and what was skipped")
    a = ap.parse_args()

    if not os.path.exists(a.seq_to_dir):
        raise SystemExit(f"STOP: no {a.seq_to_dir}. Run phase_j_corrected_queries.py first; without "
                         "it this would invent directory names that phase F will not use.")
    seq2dir = json.load(open(a.seq_to_dir))

    rows = list(csv.DictReader(open(a.gaps)))
    if not rows:
        raise SystemExit(f"STOP: {a.gaps} has no rows")
    if "corrected_query" not in rows[0]:
        raise SystemExit(f"STOP: {a.gaps} has no corrected_query column; it predates "
                         "phase_j_job_cifs.py's corrected-query emission. Regenerate it.")

    by_pdb = collections.defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)

    os.makedirs(a.out_dir, exist_ok=True)
    done, skipped = [], []
    for pdb, rs in sorted(by_pdb.items()):
        blocked = sorted({r["kind"] for r in rs if r["kind"] in BLOCKING})
        if blocked:
            skipped.append(dict(pdb=pdb, why=f"carries {','.join(blocked)} run(s); needs a "
                                             "residue-level decision before it can be corrected"))
            continue
        act = [r for r in rs if r["kind"] == ACTIONABLE and r["corrected_query"]]
        if not act:
            skipped.append(dict(pdb=pdb, why="no actionable omission with a corrected query"))
            continue
        src = os.path.join(a.jobs_dir, f"{pdb}-final-updated.json")
        if not os.path.exists(src):
            skipped.append(dict(pdb=pdb, why=f"no job json at {src}"))
            continue
        doc = json.load(open(src))

        # one corrected sequence per ROLE; a role appears once per run, and every run of a role
        # carries the same corrected_query because it restores all of that role's omissions at once
        per_role = {}
        for r in act:
            prev = per_role.get(r["role"])
            if prev is not None and prev != (r["old_query"], r["corrected_query"]):
                per_role[r["role"]] = None          # disagreement: refuse this role
            else:
                per_role[r["role"]] = (r["old_query"], r["corrected_query"])
        if any(v is None for v in per_role.values()):
            skipped.append(dict(pdb=pdb, why="two runs of one role disagree on the corrected query"))
            continue

        changed, why = 0, None
        repathed = 0
        for role, (old, new) in sorted(per_role.items()):
            hits = [(ti, si) for ti, task in enumerate(doc) for si, sq in enumerate(task["sequences"])
                    if "proteinChain" in sq
                    and (sq["proteinChain"].get("sequence") or "").strip().upper() == old.upper()]
            if len(hits) != 1:
                why = (f"role {role}: {len(hits)} chain(s) in the json match the old query exactly "
                       "(need 1) -- the report and the json disagree about what was folded")
                break
            ti, si = hits[0]
            pc = doc[ti]["sequences"][si]["proteinChain"]
            pc["sequence"] = new
            # REPOINT THE MSA AND TEMPLATE PATHS INTO THE CORRECTED STORE, keeping the per-sequence
            # directory name so the correspondence stays readable.
            for key, base in (("pairedMsaPath", "pairing.a3m"),
                              ("unpairedMsaPath", "non_pairing.a3m"),
                              ("templatesPath", "hmmsearch.a3m")):
                oldp = pc.get(key)
                if not oldp:
                    continue
                if not under(oldp, a.old_msa_root):
                    why = (f"role {role}: {key} is {oldp}, which is not under --old-msa-root "
                           f"{a.old_msa_root}; refusing to guess where it should move")
                    break
                # THE NAME COMES FROM THE INDEX, not from the old directory. phase F names its
                # output from the FASTA header, so reusing the old basename pointed every job at a
                # directory that would never exist.
                sub = seq2dir.get(new.strip().upper())
                if sub is None:
                    why = (f"role {role}: the corrected sequence is not in {a.seq_to_dir}. Re-run "
                           "phase_j_corrected_queries.py so the FASTA and these paths agree.")
                    break
                newp = os.path.join(a.msa_root, sub, base)
                if not a.allow_missing_msa:
                    # ONLY THE MSAs. hmmsearch.a3m also ends in .a3m but is a TEMPLATE file: its
                    # first row is a PDB hit, not the query, so the validator rejected it and took
                    # perfectly valid paired/unpaired MSAs down with it. Same trap as in
                    # phase_j_remap_msa.py, which excludes those files by name for the same reason.
                    bad = msa_mismatch(newp, new) if base in MSA_FILES else None
                    if not os.path.exists(newp):
                        why = (f"role {role}: {newp} does not exist. The corrected MSAs have not "
                               "been built, so this job cannot be folded yet. Pass "
                               "--allow-missing-msa to prepare the json anyway, and do NOT queue a "
                               "fold from it.")
                        break
                    if bad:
                        why = f"role {role}: {newp} {bad}"
                        break
                pc[key] = newp
                repathed += 1
            if why:
                break
            changed += 1
        if why:
            skipped.append(dict(pdb=pdb, why=why))
            continue

        # LAST GATE, BEFORE THE WRITE. It ran after os.replace, so a job it then rejected was
        # already on disk and reported as skipped -- a file nobody had approved, sitting where a
        # queue builder would find it.
        #
        # ONLY THE CORRECTED ROLES ARE CHECKED. The first version scanned every path in the json and
        # rejected all 125 structures, because a complex has three chains and typically one is
        # corrected: the OTHER chains keep their original MSAs, which is right -- their sequences did
        # not change, so those alignments still describe them exactly. Requiring every path to move
        # would have discarded 825 perfectly valid ones.
        stale = [f"{k}={v}" for _r, (_o, nq) in sorted(per_role.items())
                 for t in doc for sq in t["sequences"] if "proteinChain" in sq
                 and (sq["proteinChain"].get("sequence") or "").strip().upper() == nq.upper()
                 for k, v in sq["proteinChain"].items()
                 if k.endswith(("MsaPath", "templatesPath")) and under(v, a.old_msa_root)]
        if stale:
            skipped.append(dict(pdb=pdb, why=f"{len(stale)} CORRECTED-role path(s) still point at "
                                             f"the old MSA store, e.g. {stale[0][:120]}"))
            continue
        out = os.path.join(a.out_dir, f"{pdb}-final-updated.json")
        tmp = out + ".part"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, out)                        # atomic: a reader never sees a half-written job
        done.append(dict(pdb=pdb, roles=";".join(sorted(per_role)), n_roles=changed,
                         residues_restored=sum(int(r["n_missing"]) for r in act),
                         paths_repointed=repathed,
                         old_lens=";".join(str(len(v[0])) for _k, v in sorted(per_role.items())),
                         new_lens=";".join(str(len(v[1])) for _k, v in sorted(per_role.items()))))

    # A READINESS GATE HAS TO FAIL, AND HAS TO LEAVE NOTHING BEHIND. Without this the strict run
    # exited 0 while skipping every job whose MSA was absent -- and the jobs written by an EARLIER
    # --allow-missing-msa run were still sitting in the out-dir, so a queue builder would have found
    # 141 pending-path jsons and folded them. Reported as skips, read as ready.
    stale_left = []
    if not a.allow_missing_msa:
        written = {d["pdb"] for d in done}
        for fn in sorted(os.listdir(a.out_dir)) if os.path.isdir(a.out_dir) else []:
            if not fn.endswith("-final-updated.json"):
                continue
            pdb = fn.split("-")[0]
            if pdb not in written:
                os.replace(os.path.join(a.out_dir, fn), os.path.join(a.out_dir, fn + ".notready"))
                stale_left.append(fn)

    print(f"corrected job jsons written: {len(done)} -> {a.out_dir}")
    print(f"  roles corrected   : {sum(d['n_roles'] for d in done)}")
    print(f"  residues restored : {sum(d['residues_restored'] for d in done)}")
    print(f"\nSKIPPED: {len(skipped)}")
    for sk in skipped:
        print(f"   {sk['pdb']}: {sk['why']}")
    if a.report:
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "roles", "n_roles", "residues_restored",
                                               "paths_repointed", "old_lens", "new_lens"],
                               lineterminator="\n")
            w.writeheader(); w.writerows(done)
        sp = a.report.replace(".csv", "_skipped.csv")
        with open(sp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "why"], lineterminator="\n")
            w.writeheader(); w.writerows(skipped)
        print(f"\nwrote {a.report} and {sp}")
    if stale_left:
        print(f"\n{len(stale_left)} job(s) this run could NOT validate were renamed to .notready so "
              f"nothing can queue a fold from them, e.g. {stale_left[0]}")
    if a.expect and len(done) != a.expect:
        raise SystemExit(f"STOP: --expect {a.expect} job(s) but {len(done)} validated. "
                         f"{len(skipped)} skipped. NOT READY to queue a refold.")
    if not a.allow_missing_msa and skipped:
        raise SystemExit(f"STOP: {len(skipped)} job(s) did not validate. A strict run is a readiness "
                         "gate: it must write every expected job or fail.")


if __name__ == "__main__":
    main()
