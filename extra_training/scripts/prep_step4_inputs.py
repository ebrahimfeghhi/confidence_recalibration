#!/usr/bin/env python
"""Make ColabFold's a3m output consumable by Protenix's step3/step4, and write their two index files.

WHY THIS IS NEEDED
    Protenix's MSA pipeline assumes the a3m files come from THEIR step1 fasta, whose headers are
    "<pdb_id>_<entity_id>". Their example data shows it: scripts/msa/data/mmcif_msa_initial/0.a3m
    begins ">102m_1", and step4's rematch() looks that string up in first_pdbid_to_seq, keyed by
    "_".join(seq_to_pdb_id_entity_id[seq][0]).

    Our installed ColabFold writes a CONSTANT query header instead -- every one of the 5,305 a3m
    files begins ">101" -- while carrying the real query name in the FILENAME. Run unmodified on
    those files, step4 resolves the same key for all of them and writes every MSA into one output
    directory, keeping the last and silently discarding 5,304. Nothing downstream would look wrong.

    Protenix's step1 cannot supply the mapping either: it reads _entity_poly.pdbx_seq_one_letter_code_can
    from the mmCIF, which the SAbDab2 cifs do not carry (4,550 of 4,555 skipped), and its sequences
    are the full deposited constructs -- signal peptides and tags included -- not the SAbDab2 chain
    sequences our MSAs were actually searched with.

WHAT IT DOES
    1. Copies each <name>.a3m into --out-dir with the constant query header replaced by ">" + name.
       BOTH occurrences are replaced: step3 finds the uniref/env boundary by looking for the query
       header a second time, so rewriting only the first line would leave uniref_index at 0 and add
       no taxids at all.
    2. Symlinks uniref_tax.m8 into --out-dir, because step3 reads it as "{input_msa_dir}/uniref_tax.m8".
    3. Writes the two json files step4 takes, both derived from the query fasta:
         seq_to_pdb_id_entity_id.json  {sequence: [[name]]}   -> first_pdbid_to_seq = {name: sequence}
         seq_to_pdb_index.json         {sequence: name}       -> output directory per sequence
       Using the query name as the index makes each MSA directory self-identifying, and it is the same
       file build_val_jobs.py reads to resolve a chain's MSA paths.

VERIFICATION, refusing to proceed on a mismatch
    Every a3m's line 2 must equal the fasta sequence for its filename. That is the only thing tying a
    file to a query once the header is a constant, and it was measured at 5,304 of 5,305 before this
    script existed. The exception is a 35-nt RNA antigen (c228_6cf2_ag1), which mmseqs searched as a
    protein with U transcribed to X; it has no meaningful protein MSA and is dropped, not patched.

OUTPUTS
    <out-dir>/<name>.a3m                        headers rewritten
    <out-dir>/uniref_tax.m8                     symlink
    <index-dir>/seq_to_pdb_id_entity_id.json
    <index-dir>/seq_to_pdb_index.json
    <index-dir>/prep_step4_report.csv           name, n_seqs, header_lines_rewritten, ok
"""
import argparse
import csv
import json
import os
import sys

W = "/home/ebrahim/tts_abag/extra_training/msa_work"


def read_fasta(path):
    """in: a fasta path.  out: {name: sequence}"""
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            out[name] = ""
        elif name:
            out[name] += line
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3m-dir", default=f"{W}/msa_initial_idx")
    ap.add_argument("--fasta", default=f"{W}/query.fasta")
    ap.add_argument("--out-dir", default=f"{W}/msa_initial_named")
    ap.add_argument("--index-dir", default=f"{W}/pdb_seqs")
    a = ap.parse_args()

    fa = read_fasta(a.fasta)
    print(f"queries in fasta: {len(fa)}   unique sequences: {len(set(fa.values()))}")
    if len(set(fa.values())) != len(fa):
        sys.exit("STOP: the query fasta has duplicate sequences, so a sequence-keyed index is ambiguous")

    files = sorted(f for f in os.listdir(a.a3m_dir)
                   if f.endswith(".a3m") and f != "uniref.a3m")
    print(f"a3m files: {len(files)}")
    if not files:
        sys.exit("STOP: no a3m files found")

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.index_dir, exist_ok=True)

    seq_to_id, seq_to_index, rows = {}, {}, []
    dropped, rewritten_total = [], 0
    for fn in files:
        name = fn[:-4]
        src = os.path.join(a.a3m_dir, fn)
        with open(src) as fh:
            lines = fh.readlines()
        if len(lines) < 2:
            dropped.append((name, "fewer than 2 lines"))
            continue
        hdr, seq = lines[0], lines[1].strip()
        want = fa.get(name)
        if want is None:
            dropped.append((name, "no such query in the fasta"))
            continue
        if seq.upper() != want.upper():
            # the one known case is an RNA antigen searched as protein (U -> X); anything else is a
            # file/query mix-up and must not be guessed at
            dropped.append((name, f"line-2 sequence != fasta sequence ({seq[:20]}... vs {want[:20]}...)"))
            continue

        new_hdr = f">{name}\n"
        n_rw = 0
        with open(os.path.join(a.out_dir, fn), "w") as out:
            for ln in lines:
                if ln == hdr:
                    out.write(new_hdr)
                    n_rw += 1
                else:
                    out.write(ln)
        rewritten_total += n_rw
        seq_to_id[want] = [[name]]
        seq_to_index[want] = name
        rows.append(dict(name=name, n_lines=len(lines), header_lines_rewritten=n_rw, ok=True))

    m8 = os.path.join(a.a3m_dir, "uniref_tax.m8")
    link = os.path.join(a.out_dir, "uniref_tax.m8")
    if os.path.exists(m8) and not os.path.exists(link):
        os.symlink(m8, link)

    # MERGE, never overwrite: this runs once per search batch (the main query set and the short
    # peptide antigens are searched separately, because their ColabFold query databases are keyed
    # independently), and step4 takes ONE pair of index files covering every sequence.
    def merge(path, new_items):
        old = {}
        if os.path.exists(path):
            old = json.load(open(path))
        before = len(old)
        old.update(new_items)
        with open(path, "w") as fh:
            json.dump(old, fh)
        print(f"  {os.path.basename(path)}: {before} -> {len(old)} sequences")
        return old

    merge(f"{a.index_dir}/seq_to_pdb_id_entity_id.json", seq_to_id)
    merge(f"{a.index_dir}/seq_to_pdb_index.json", seq_to_index)
    rep = f"{a.index_dir}/prep_step4_report.csv"
    fresh = not os.path.exists(rep)
    with open(rep, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch", "name", "n_lines", "header_lines_rewritten", "ok"])
        if fresh:
            w.writeheader()
        for r in rows:
            w.writerow(dict(batch=os.path.basename(a.a3m_dir.rstrip("/")), **r))

    two = sum(1 for r in rows if r["header_lines_rewritten"] == 2)
    print(f"written: {len(rows)}   dropped: {len(dropped)}")
    print(f"query-header lines rewritten: {rewritten_total}  "
          f"(files with exactly 2, i.e. a uniref AND an env block: {two})")
    for n, why in dropped:
        print(f"  DROPPED {n}: {why}")
    print(f"index files in {a.index_dir}")
    if not rows:
        sys.exit("STOP: nothing written")


if __name__ == "__main__":
    main()
