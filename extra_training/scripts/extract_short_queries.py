#!/usr/bin/env python
"""Emit the antigen chains that the first MSA pass left out because of its 20-residue minimum.

WHY
    The first query set used --min-len 20, so 235 unique antigen chains of 5-19 residues got no MSA.
    FoldBench, which we evaluate on, contains such chains and puts them through the same search: of
    its 8 protein chains under 20 aa, 4 come back query-only but 4 carry real depth (7, 4, 431 and
    3,320 sequences). Synthesising a query-only MSA would therefore misrepresent half of them, so they
    are searched properly instead -- and this script is just the set difference that says which.

    They are searched as a SEPARATE batch rather than appended to the first one: ColabFold keys its
    query database by position, so adding queries would shift every key and the kept uniref.a3m could
    no longer be resumed from.

INPUTS   --all      a fasta built with the lower minimum length (extract_msa_queries --min-len 5)
         --already  the fasta already searched (query.fasta)
OUTPUT   --out      the records in --all whose SEQUENCE is absent from --already
"""
import argparse
import sys


def read_fasta(path):
    """in: a fasta path.  out: [(name, sequence)] in file order"""
    out, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--already", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    allr = read_fasta(a.all)
    have = {q.upper() for _n, q in read_fasta(a.already)}
    new = [(n, q) for n, q in allr if q.upper() not in have]

    if not new:
        sys.exit("STOP: nothing new to search -- the two fastas already cover the same sequences")
    seqs = {q.upper() for _n, q in new}
    if len(seqs) != len(new):
        sys.exit(f"STOP: {len(new)} records but only {len(seqs)} distinct sequences; "
                 "duplicate sequences would collide as <name>.a3m")
    with open(a.out, "w") as fh:
        for n, q in new:
            fh.write(f">{n}\n{q}\n")
    L = sorted(len(q) for _n, q in new)
    print(f"{a.all}: {len(allr)} records   {a.already}: {len(have)} sequences")
    print(f"new to search: {len(new)}   lengths min {L[0]} median {L[len(L)//2]} max {L[-1]}")
    print(f"  of these, >=15 aa (where real MSA depth appears): {sum(1 for x in L if x >= 15)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
