#!/usr/bin/env python
"""Derive a query-only-MSA copy of chosen templated folds: same template, MSA cut to its first row.

SUMMARY
    An ablation for asking how much a templated fold's confidence rests on its MSA. For the chosen folds
    of an experiment (queue tags per complex), this writes a new experiment directory whose template
    namespace is the original's with ONE change: every protein chain's paired and unpaired MSA is replaced
    by a file holding only its first record -- the query sequence itself, no homologs. The template
    structures are the same files (the mmcif directory and release dates are symlinked), each chain keeps
    its templatesPath, and the queue lines are copied unchanged (same donor, seed, pinned interface), so
    the MSA is the only difference.

    Checks, before anything is written: every MSA's first record is the query and matches the chain's
    sequence; afterwards run preflight.py on the new namespace, as for any build.

INPUTS
    --exp:    the source experiment (its queue.tsv and templates/). Required.
    --folds:  comma-separated <pdb>:<tag> to derive, e.g. 7zoz:A05,7zoz:A00. Required.
    --out:    the new experiment directory. Required; must not already hold a queue.

OUTPUTS
    <out>/queue.tsv                        the chosen lines, unchanged
    <out>/templates/mmcif, release_dates.json   symlinks to the source namespace's
    <out>/templates/jobs/<iface>_<tag>/    job.json with query-only MSAs, chosen.json, msa/<n>_{paired,unpaired}.a3m

EXAMPLE
    analysis/query_only_msa.py --exp experimental_data/protenix/foldbench/self_template_fixed_50_seed \\
        --folds 7zoz:A05,7zoz:A00 --out experimental_data/protenix/foldbench/self_template_msa1_7zoz

FUNCTIONS
    first_record(): an a3m file -> its first record (header, sequence)
    main():         select, derive, write
"""
import argparse
import json
import os
import shutil


def first_record(path):
    """in: an a3m path.  out: (header line, sequence) of its first record; raises if it has none"""
    header, seq = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                break
            header = line
        elif header is not None:
            seq.append(line)
    if header is None or not seq:
        raise SystemExit(f"{path}: no first record")
    return header, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    want = {tuple(f.split(":")) for f in A.folds.split(",")}
    if os.path.exists(f"{A.out}/queue.tsv"):
        raise SystemExit(f"{A.out}/queue.tsv exists; choose a new --out")

    lines = [l.rstrip("\n").split("\t") for l in open(f"{A.exp}/queue.tsv") if l.strip()]
    chosen = [f for f in lines if (f[1], f[3]) in want]
    if len(chosen) != len(want):
        raise SystemExit(f"found {len(chosen)} of {len(want)} folds in {A.exp}/queue.tsv")

    src_ns, ns = os.path.abspath(f"{A.exp}/templates"), os.path.abspath(f"{A.out}/templates")
    os.makedirs(f"{ns}/jobs", exist_ok=True)
    for name in ("mmcif", "release_dates.json"):
        if not os.path.lexists(f"{ns}/{name}"):
            os.symlink(f"{src_ns}/{name}", f"{ns}/{name}")
    for f in chosen:
        key = f"{f[2]}_{f[3]}"
        job = json.load(open(f"{src_ns}/jobs/{key}/job.json"))
        obj = job[0] if isinstance(job, list) else job
        os.makedirs(f"{ns}/jobs/{key}/msa", exist_ok=True)
        for n, e in enumerate(x for x in obj["sequences"] if "proteinChain" in x):
            c = e["proteinChain"]
            for field, kind in (("pairedMsaPath", "paired"), ("unpairedMsaPath", "unpaired")):
                if not c.get(field):
                    continue
                header, seq = first_record(c[field])
                if seq.replace("-", "") != c["sequence"]:
                    raise SystemExit(f"{key} chain {n}: first MSA record is not the query sequence")
                dst = f"{ns}/jobs/{key}/msa/{n}_{kind}.a3m"
                open(dst, "w").write(f"{header}\n{seq}\n")
                c[field] = dst
        json.dump(job, open(f"{ns}/jobs/{key}/job.json", "w"))
        shutil.copy(f"{src_ns}/jobs/{key}/chosen.json", f"{ns}/jobs/{key}/chosen.json")
    with open(f"{A.out}/queue.tsv", "w") as fh:
        fh.writelines("\t".join(f) + "\n" for f in chosen)
    print(f"  {len(chosen)} folds -> {A.out} (MSAs cut to the query; templates, seed and donors unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
