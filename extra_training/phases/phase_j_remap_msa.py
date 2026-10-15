#!/usr/bin/env python
"""PHASE J -- carry an existing MSA onto a CORRECTED query by inserting pure-gap columns.

THE PROBLEM. The corrected queries restore internal residues the original inputs dropped, so each is
a few residues longer than the query its stored MSA was built against. An a3m is one match column per
query residue, so the stored alignment no longer describes the sequence. The mmseqs server that would
supply fresh alignments is returning 503, and re-searching is a two-day job even when it is up.

WHAT THIS DOES, AND WHAT IT REFUSES TO DO. It does NOT re-align anything. Every existing column is
carried through byte for byte; the only change is that `n` pure-gap columns are inserted at one known
point, and the restored amino acids are inserted into the query row at the same point. The insertion
point comes from the APPROVED omission list -- `query_insert_at`, emitted by phase_j_job_cifs from the
native-position map -- and is never rediscovered by aligning the old query to the corrected one. That
is the whole reason this is safe to do: a generic re-alignment is where a silent one-column shift
would come from, and there isn't one.

THE TRANSFORMATION IS REVERSIBLE, AND THAT IS CHECKED PER ROW. Deleting the inserted columns must
reproduce the original row exactly, for every row, or the file is refused. Also checked: the query
equals the corrected sequence, every row has exactly the corrected number of match columns, every
homolog's ungapped residues are unchanged, and the headers and row order are identical.

LOWERCASE IS PRESERVED. In a3m, lowercase means an insertion relative to the query at that locus and
carries real information; stripping it would lose it. The new columns are inserted immediately before
the match column at the insertion point, which leaves any preceding lowercase run attached to the
match column it follows -- its a3m meaning.

THE LIMITATION, STATED PLAINLY. The restored residues receive NO aligned homolog information: their
columns are all gaps. A gap means "no aligned evidence here", not "the relatives lack this residue",
and the model cannot tell those apart. Predictions at those positions may therefore be worse than
with a fresh search. This buys a correct-length, unshifted alignment without introducing a guess; it
does not buy the evidence a real search would have found.

RESTRICTED TO VERIFIED INTERNAL INSERTIONS. A role is processed only if its runs are all `omission`
with a usable `query_insert_at`, and only if removing those insertions from the corrected query
reproduces the old query exactly. Substitutions, domain-boundary changes and ambiguous maps are out
of scope and are skipped by name.
"""
import argparse
import collections
import csv
import hashlib
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_a3m(path):
    """[(header, sequence)] in file order, sequences joined across wrapped lines."""
    rows, h, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if h is not None:
                    rows.append((h, "".join(buf)))
                h, buf = line, []
            else:
                buf.append(line)
    if h is not None:
        rows.append((h, "".join(buf)))
    return rows


def match_columns(seq):
    """Indices in `seq` that are MATCH columns -- uppercase or gap. Lowercase is an a3m insertion."""
    return [i for i, c in enumerate(seq) if not c.islower()]


def insert_columns(seq, at, fill):
    """Insert `fill` immediately before match column `at`. Returns the new string and the string
    index it was inserted at, so the reverse check can delete exactly those characters."""
    cols = match_columns(seq)
    pos = cols[at] if at < len(cols) else len(seq)
    return seq[:pos] + fill + seq[pos:], pos


def ungapped(seq):
    return "".join(c for c in seq if c not in "-.").upper()


def remap(rows, at, restored, old_len, new_len):
    """Carry every row onto the corrected query. Returns (new_rows, failures).

    THE ABSOLUTE COLUMN COUNT IS CHECKED, NOT ONLY THE DELTA. The first version required each row to
    GAIN n match columns and nothing more, so a row that arrived malformed left malformed and nothing
    reported it: a 215-column hit against a 214-residue query became 222 against 221, with zero
    failures. The delta was right and the file was still broken. An a3m is one match column per query
    residue BY DEFINITION, so every input row must have exactly `old_len` and every output row
    exactly `new_len`. Malformed rows are not hypothetical -- the rejected mmseqs2 rebuild produced
    them on an unchanged query.
    """
    n = len(restored)
    out, bad = [], []
    for k, (h, s) in enumerate(rows):
        if len(match_columns(s)) != old_len:
            bad.append((k, h, f"input row has {len(match_columns(s))} match columns, "
                              f"query has {old_len}"))
            continue
        fill = restored if k == 0 else "-" * n
        new, pos = insert_columns(s, at, fill)
        # REVERSIBILITY, per row: deleting exactly what was inserted must give the original back.
        if new[:pos] + new[pos + n:] != s:
            bad.append((k, h, "not reversible"))
            continue
        if len(match_columns(new)) != new_len:
            bad.append((k, h, f"output row has {len(match_columns(new))} match columns, "
                              f"corrected query has {new_len}"))
            continue
        if len(match_columns(new)) != len(match_columns(s)) + n:
            bad.append((k, h, "did not gain exactly n columns"))
            continue
        if k and ungapped(new) != ungapped(s):
            bad.append((k, h, "homolog residues changed"))
            continue
        out.append((h, new))
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True, help="the --native-gaps csv from phase_j_job_cifs.py")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa",
                    help="the EXISTING msa store, read only")
    ap.add_argument("--out-root", default=f"{ROOT}/building_msa/work/mmcif_msa_corrected",
                    help="a NEW store; the original is never written to")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    import json
    idx = json.load(open(a.seq_index)) if os.path.exists(a.seq_index) else {}

    rows = list(csv.DictReader(open(a.gaps)))
    if not rows or "query_insert_at" not in rows[0]:
        raise SystemExit(f"STOP: {a.gaps} has no query_insert_at column; regenerate it with the "
                         "current phase_j_job_cifs.py. Without it this would have to rediscover the "
                         "insertion point by alignment, which is exactly what must not happen.")

    per = collections.defaultdict(list)
    for r in rows:
        per[(r["pdb"], r["role"])].append(r)

    done, skipped = [], []
    for (pdb, role), rs in sorted(per.items()):
        kinds = {r["kind"] for r in rs}
        if kinds != {"omission"}:
            skipped.append(dict(pdb=pdb, role=role,
                                why=f"out of scope: carries {','.join(sorted(kinds - {'omission'}))}"))
            continue
        if any(not str(r["query_insert_at"]).strip() for r in rs):
            skipped.append(dict(pdb=pdb, role=role, why="no insertion point for one of its runs"))
            continue
        old, new = rs[0]["old_query"], rs[0]["corrected_query"]
        if not new:
            skipped.append(dict(pdb=pdb, role=role, why="no corrected query"))
            continue
        # ONE INSERTION ONLY, for now. Two runs in one role need the second point shifted by the
        # first insertion, which is easy to get wrong by one; refused rather than risked.
        if len(rs) != 1:
            skipped.append(dict(pdb=pdb, role=role,
                                why=f"{len(rs)} runs in this role; only a single insertion is in "
                                    "scope"))
            continue
        at, restored = int(rs[0]["query_insert_at"]), rs[0]["sequence"]
        # THE APPROVED MAP, CHECKED BOTH WAYS before a single file is touched.
        if old[:at] + restored + old[at:] != new or new[:at] + new[at + len(restored):] != old:
            skipped.append(dict(pdb=pdb, role=role,
                                why="the approved insertion does not reproduce the corrected query, "
                                    "or removing it does not reproduce the old one"))
            continue

        src = idx.get(old)
        if src is None:
            cand = [d for d in os.listdir(a.msa_root)
                    if os.path.isdir(os.path.join(a.msa_root, d)) and d.endswith(f"_{role}")]
            src = None
            for d in cand:
                p = os.path.join(a.msa_root, d, "pairing.a3m")
                if os.path.exists(p):
                    q = read_a3m(p)
                    if q and q[0][1].replace("-", "").upper() == old.upper():
                        src = d
                        break
        if src is None or not os.path.isdir(os.path.join(a.msa_root, str(src))):
            skipped.append(dict(pdb=pdb, role=role,
                                why="no stored MSA directory matches this old query"))
            continue

        sdir = os.path.join(a.msa_root, str(src))
        ddir = os.path.join(a.out_root, str(src))
        os.makedirs(ddir, exist_ok=True)
        # ONLY THE MSAs. hmmsearch.a3m, .filtered and .raw are TEMPLATE search outputs -- their first
        # row is a PDB hit, not the query ("7kqy_A/1-217 [subseq from] ..."), so they are not an a3m
        # in this sense and cannot be remapped. Nor may they be COPIED: they were built against the
        # old query, and carrying them forward would silently pair a corrected input with stale
        # templates. Templates are regenerated against the corrected query by phase H.
        MSA_FILES = ("pairing.a3m", "non_pairing.a3m")
        files, fails = 0, []
        for fn in sorted(os.listdir(sdir)):
            sp = os.path.join(sdir, fn)
            if fn not in MSA_FILES:
                continue
            orig = read_a3m(sp)
            if not orig:
                fails.append((fn, "empty")); continue
            if orig[0][1].replace("-", "").upper() != old.upper():
                fails.append((fn, "query row is not the old query")); continue
            nr, bad = remap(orig, at, restored, len(old), len(new))
            if bad or len(nr) != len(orig):
                fails.append((fn, f"{len(bad)} row(s) failed, e.g. {bad[0][2] if bad else 'count'}"))
                continue
            if nr[0][1].replace("-", "").upper() != new.upper():
                fails.append((fn, "remapped query is not the corrected query")); continue
            if [h for h, _s in nr] != [h for h, _s in orig]:
                fails.append((fn, "headers changed")); continue
            tmp = os.path.join(ddir, fn + ".part")
            with open(tmp, "w") as fh:
                for h, s in nr:
                    fh.write(f"{h}\n{s}\n")
            os.replace(tmp, os.path.join(ddir, fn))
            files += 1
        if files != len(MSA_FILES):
            fails.append(("", f"expected {len(MSA_FILES)} MSA file(s), remapped {files}"))
        if fails:
            skipped.append(dict(pdb=pdb, role=role,
                                why="; ".join(f"{f}: {w}" for f, w in fails[:3])))
            shutil.rmtree(ddir, ignore_errors=True)
            continue
        done.append(dict(pdb=pdb, role=role, msa_dir=str(src), a3m_files=files,
                         inserted=restored, at=at, old_len=len(old), new_len=len(new)))

    print(f"roles remapped: {len(done)} -> {a.out_root}")
    print(f"  a3m files rewritten: {sum(d['a3m_files'] for d in done)}")
    print(f"  residues inserted  : {sum(len(d['inserted']) for d in done)}")
    print(f"\nSKIPPED: {len(skipped)}")
    c = collections.Counter(s["why"].split(";")[0].split(",")[0] for s in skipped)
    for w, k in c.most_common(10):
        print(f"   {k:4d}  {w}")
    if a.report:
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "role", "msa_dir", "a3m_files", "inserted",
                                               "at", "old_len", "new_len"], lineterminator="\n")
            w.writeheader(); w.writerows(done)
        sp = a.report.replace(".csv", "_skipped.csv")
        with open(sp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "role", "why"], lineterminator="\n")
            w.writeheader(); w.writerows(skipped)
        print(f"\nwrote {a.report} and {sp}")
    print("\nLIMITATION: the restored columns are ALL GAPS -- no aligned homolog evidence there. A gap "
          "means 'nothing aligned here', not 'the relatives lack this residue', and the model cannot "
          "distinguish those. Predictions at those positions may be worse than with a fresh search.")


if __name__ == "__main__":
    main()
