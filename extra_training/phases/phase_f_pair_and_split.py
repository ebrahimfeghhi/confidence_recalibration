#!/usr/bin/env python
"""PHASE F -- turn phase E's alignments into the three files Protenix actually reads.

WHAT PROTENIX WANTS, per chain, in its own directory:
    pairing.a3m       the relatives that could be matched to another chain by organism
    non_pairing.a3m   everything else
    hmmsearch.a3m     templates -- phase H writes that one

WHY THE SPLIT EXISTS. To predict an antibody bound to an antigen, the model wants to see the SAME
organism in both chains' alignments: a mouse antibody beside a mouse antigen, and so on. That is what
"paired" means, and it is why phase D went to the trouble of keeping taxonomy ids. Sequences with no
usable organism go in the unpaired file, which still contributes depth but cannot be matched up.

Nothing downstream can read anything until this phase finishes.

THREE STEPS, two of them Protenix's own code
    1. (here)  rename the query header in every a3m, and write step4's two index files.
    2. step3-uniref_add_taxid.py       writes the NCBI taxid into each UniRef header.
    3. step4-split_msa_to_uniref_and_others.py   splits each a3m into the pairing/non_pairing pair.

WHY STEP 1 IS NEEDED AT ALL. Protenix's pipeline assumes the a3m came from THEIR step1 fasta, whose
headers are "<pdb_id>_<entity_id>" -- their example scripts/msa/data/mmcif_msa_initial/0.a3m begins
">102m_1", and step4's rematch() looks that string up in first_pdbid_to_seq. Our ColabFold writes a
CONSTANT query header instead: every one of the 5,305 a3m files begins ">101", with the real name only
in the filename. Run unmodified, step4 resolves the same key for all of them, writes every MSA into
one directory, keeps the last and discards 5,304 -- and nothing downstream looks wrong.

Protenix's step1 cannot supply the mapping either: it reads
_entity_poly.pdbx_seq_one_letter_code_can from the mmCIF, which the SAbDab2 cifs do not carry (4,550
of 4,555 skipped), and its sequences are the full deposited constructs, tags included, rather than the
chain sequences our MSAs were searched with.

BOTH HEADER OCCURRENCES ARE REWRITTEN. step3 finds the uniref/env boundary by looking for the query
header a SECOND time; rewriting only line 1 would leave that index at 0 and add no taxids at all.

THE FASTA IS EACH BATCH'S FROZEN query.searched.fasta, never msa_work/query.fasta. The a3m filenames
come from the fasta ColabFold was given, so verifying a file against any other fasta is comparing it
to something it was not built from. Phase C's current output legitimately differs by a record.

ONE SET OF INDEX FILES, BUILT ONCE. step4 takes a single pair of json files covering every sequence,
and the batches are searched separately because ColabFold keys its query database by position. An
earlier version ran once per batch and MERGED into the jsons on each run, which works until a batch is
re-run against a changed fasta and leaves stale entries behind. This phase reads every batch in one
process and writes the jsons fresh.

USAGE
    phase_f_pair_and_split.py [batch-dir ...]     default: phase D's three output directories

OUTPUTS
    msa_work/msa_named_<batch>/            a3m with their real query header, plus a uniref_tax.m8 link
    msa_work/pdb_seqs/seq_to_pdb_id_entity_id.json
    msa_work/pdb_seqs/seq_to_pdb_index.json
    msa_work/pdb_seqs/phase_f_report.csv   one row per alignment: batch, name, lines, rewrites
    msa_work/mmcif_msa_with_taxid/<name>.a3m
    msa_work/mmcif_msa/<name>/{pairing,non_pairing}.a3m
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

R = "/home/ebrahim/tts_abag"
W = f"{R}/extra_training/msa_work"
P = os.path.realpath(os.environ.get("PROTENIX_DIR", "/home/ebrahim/Protenix"))
PY = f"{P}/.venv/bin/python"
DEFAULT_BATCHES = [f"{W}/msa_initial_idx", f"{W}/msa_short_idx", f"{W}/msa_bench_short_idx"]


def read_fasta(path):
    """in: a fasta path.  out: {name: sequence}"""
    out, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                out[name] = ""
            elif name:
                out[name] += line
    return out


def check_frozen(batch):
    """in: a phase D output directory.  out: {name: sequence} from its frozen searched fasta

    The checksum is verified, not merely the file's presence. A fasta that has changed since the
    search means the a3m filenames and the fasta records no longer correspond, and every check below
    would be comparing an alignment against a sequence it was not built from.
    """
    qfa = os.path.join(batch, "query.searched.fasta")
    if not os.path.exists(qfa):
        sys.exit(f"STOP: {qfa} missing. Phase D freezes the fasta it searched; phase F needs it.")
    sha = os.path.join(batch, "query.searched.fasta.sha256")
    if os.path.exists(sha):
        want = open(sha).read().split()[0]
        got = hashlib.sha256(open(qfa, "rb").read()).hexdigest()
        if want != got:
            sys.exit(f"STOP: {qfa} does not match {sha}. It has changed since the search, so the a3m "
                     "filenames no longer correspond to its records. Do not proceed.")
    else:
        print(f"  WARNING {sha} absent; cannot verify the fasta has not changed")
    return read_fasta(qfa)


NUCLEOTIDES = set("ACGTUN")


def classify_mismatch(want, got):
    """in: the fasta sequence and the a3m's line 2.  out: "nucleotide->X" or a specific reason

    Only "nucleotide->X" is treated as expected downstream. Everything else names what it actually
    is, so an unexplained difference cannot hide inside the tolerated class.
    """
    if len(want) != len(got):
        return f"length {len(want)} vs {len(got)}"
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(want.upper(), got.upper())) if a != b]
    if not diffs:
        return "identical after case folding"
    if all(b == "X" and a in NUCLEOTIDES for _i, a, b in diffs):
        return "nucleotide->X"
    bad = diffs[:3]
    return (f"{len(diffs)} differing position(s) not explained by nucleotide->X, e.g. "
            + ", ".join(f"{i}:{a}->{b}" for i, a, b in bad))


def rename_headers(batch, fa, out_dir):
    """in: a batch dir, its {name: sequence}, where to write.  out: (rows, dropped, seq maps)

    Copies each a3m with BOTH occurrences of the constant query header replaced by its real name, and
    symlinks uniref_tax.m8 in, because step3 reads it as "{input_msa_dir}/uniref_tax.m8".
    """
    files = sorted(f for f in os.listdir(batch)
                   if f.endswith(".a3m") and f not in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"))
    if not files:
        sys.exit(f"STOP: no alignments in {batch}")
    os.makedirs(out_dir, exist_ok=True)
    rows, dropped, seq_to_id, seq_to_index = [], [], {}, {}
    for fn in files:
        name = fn[:-4]
        with open(os.path.join(batch, fn)) as fh:
            lines = fh.readlines()
        if len(lines) < 2:
            dropped.append((name, "fewer than 2 lines"))
            continue
        hdr, seq = lines[0], lines[1].strip()
        want = fa.get(name)
        if want is None:
            dropped.append((name, "no such query in the frozen fasta"))
            continue
        if seq.upper() != want.upper():
            # THE MISMATCH MUST BE THE KNOWN TRANSFORMATION, not merely a mismatch. Classifying every
            # line-2 difference as "the known nucleic-acid case" would accept ordinary sequence
            # corruption and drop it silently. The known case is specific: a nucleic-acid antigen
            # searched as protein, where mmseqs wrote X over each nucleotide. So it qualifies only if
            # the lengths match and EVERY differing position is a nucleotide in the fasta that became
            # exactly X in the a3m.
            why = classify_mismatch(want, seq)
            dropped.append((name, f"line-2 sequence != fasta sequence [{why}] "
                                  f"({seq[:20]}... vs {want[:20]}...)"))
            continue
        n_rw = 0
        with open(os.path.join(out_dir, fn), "w") as out:
            for ln in lines:
                if ln == hdr:
                    out.write(f">{name}\n")
                    n_rw += 1
                else:
                    out.write(ln)
        seq_to_id[want] = [[name]]
        seq_to_index[want] = name
        rows.append(dict(batch=os.path.basename(batch.rstrip("/")), name=name, n_lines=len(lines),
                         header_lines_rewritten=n_rw))
    m8 = os.path.join(batch, "uniref_tax.m8")
    link = os.path.join(out_dir, "uniref_tax.m8")
    # lexists, NOT exists. os.path.exists follows the link and is False for a DANGLING one, so a
    # stale link left by an earlier run made os.symlink raise FileExistsError instead of being
    # replaced. lexists sees the link itself.
    if os.path.lexists(link):
        os.remove(link)
    if os.path.exists(m8):
        os.symlink(m8, link)
    elif not os.path.exists(m8):
        print(f"  NOTE {batch} has no uniref_tax.m8; its alignments cannot gain taxids")
    return rows, dropped, seq_to_id, seq_to_index


def run(cmd, log, label, cores, timeout):
    """in: argv, a log path, a label, a taskset range, seconds.  out: the exit code, never trusted"""
    full = ["taskset", "-c", cores] + cmd
    print(f"  {label}: {' '.join(str(c) for c in full[:6])} ... -> {os.path.basename(log)}")
    with open(log, "ab") as fh:
        try:
            rc = subprocess.call(full, stdout=fh, stderr=fh, cwd=P, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  {label}: TIMED OUT after {timeout}s")
            return 124
    print(f"  {label}: exit {rc} (not trusted; the gates below decide)")
    return rc


def taxid_coverage(d, names=None, every=97):
    """in: a directory of a3m, optionally the names belonging to one batch.
    out: (hit records seen, hit records carrying a taxid, files sampled)

    OVER HIT RECORDS, NOT OVER FILES. The previous version asked whether a file contained AT LEAST
    ONE tagged header and stopped reading at the first one -- so a file with 100 UniRef hits and a
    single tagged hit counted as fully tagged. What matters is the fraction of hits step3 actually
    annotated, because every untagged hit is a sequence that cannot be paired.

    PER BATCH when `names` is given. The batches are searched separately and step3 runs once per
    batch, so a failure confined to one of them averages away in a pooled figure.
    """
    import re
    tagged_re = re.compile(r"^>UniRef100_.+_[0-9]+/")
    files = sorted(f for f in os.listdir(d) if f.endswith(".a3m"))
    if names is not None:
        files = [f for f in files if f[:-4] in names]
    sample = files[::every] if every > 1 else files
    seen = tagged = 0
    for fn in sample:
        with open(os.path.join(d, fn)) as fh:
            for line in fh:
                if line.startswith(">UniRef100"):
                    seen += 1
                    if tagged_re.match(line):
                        tagged += 1
    return seen, tagged, len(sample)


def fresh_outputs(expected, path_of, marker, label, allow_reuse, validate=None):
    """in: the names that must have an output, name -> path, the start marker, a reuse validator.
    out: nothing; exits on any shortfall.

    EVERY EXPECTED OUTPUT, NOT ONE OF THEM. The gate here used to be `if not fresh: exit`, which
    passed the moment a single file was new -- and the counts that followed it were taken over the
    whole directory, so they could not tell this run's output from an earlier attempt's. A step3 or
    step4 that crashed part-way through therefore left some chains carrying stale taxids or a stale
    split while the phase reported OK, and the staleness only surfaces as a chain that cannot pair.

    A file older than the marker is REUSE, and reuse is only acceptable when it is deliberate
    (--allow-stale-reuse), validates, and is named in the log. Anything missing is fatal either way.
    """
    t = os.path.getmtime(marker)
    fresh, reused, stale, missing = [], [], [], []
    for n in sorted(expected):
        p = path_of(n)
        if not os.path.exists(p):
            missing.append(n)
        elif os.path.getmtime(p) >= t:
            fresh.append(n)
        elif validate is not None and not validate(n, p):
            stale.append(n)
        else:
            reused.append(n)
    print(f"  {label}: {len(fresh)} written by this run, {len(reused)} reused, "
          f"{len(stale)} stale and unusable, {len(missing)} missing, of {len(expected)} expected")
    if missing:
        sys.exit(f"STOP: {len(missing)} expected {label} output(s) do not exist: {missing[:6]}. "
                 "The step exited 0 without producing them, so some chains have no usable output "
                 "and a directory-wide count cannot see it.")
    if stale:
        sys.exit(f"STOP: {len(stale)} {label} output(s) predate this run and do not validate: "
                 f"{stale[:6]}. They are left over from an attempt that did not finish.")
    if reused and not allow_reuse:
        sys.exit(f"STOP: {len(reused)} {label} output(s) predate this run: {reused[:6]}. They may be "
                 "correct, but this run did not produce them, so nothing here establishes that. "
                 "Re-run the step, or pass --allow-stale-reuse to accept them explicitly.")
    if reused:
        print(f"     REUSED BY REQUEST: {reused[:10]}{' ...' if len(reused) > 10 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--index-dir", default=f"{W}/pdb_seqs")
    ap.add_argument("--taxid-dir", default=f"{W}/mmcif_msa_with_taxid")
    ap.add_argument("--msa-dir", default=f"{W}/mmcif_msa")
    ap.add_argument("--cores", default="0-63",
                    help="taskset range; 64 is the project cap (MSA_RUNBOOK.md:119)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ignore-step-rc", action="store_true",
                    help="continue when step3 or step4 exits nonzero. Off by default: their output "
                         "from an earlier attempt would otherwise satisfy every count below.")
    ap.add_argument("--allow-missing-batches", action="store_true",
                    help="process only the batch directories that exist. Off by default: an absent "
                         "batch means phase D has not finished, and its chains would get no MSA.")
    ap.add_argument("--max-seq-mismatch", type=int, default=5,
                    help="how many a3m may legitimately disagree with their fasta record. One per "
                         "batch is expected: a nucleic-acid antigen the search transcribed U to X.")
    ap.add_argument("--allow-stale-reuse", action="store_true",
                    help="accept step3/step4 outputs that predate this run. Off by default: a step "
                         "that exits 0 after crashing part-way leaves some chains stale, and a "
                         "directory-wide count cannot tell those from this run's output.")
    ap.add_argument("--min-pairing-frac", type=float, default=0.50,
                    help="fraction of chains of 20+ residues that must have a NON-EMPTY pairing.a3m. "
                         "This phase exists to produce pairing, so its completion criterion has to "
                         "test pairing. All-empty is the failure it must catch.")
    ap.add_argument("--min-tagged-frac", type=float, default=0.90)
    a = ap.parse_args()
    # EVERY REQUESTED BATCH, NOT THE ONES THAT HAPPEN TO EXIST. Filtering silently let the phase
    # succeed on a subset: run it before the short-antigen batch finished and it would index 5,305
    # sequences, pass every gate, and leave 246 chains with no MSA at all. The defaults are three
    # known directories, so an absent one is a pipeline-ordering error, not an option.
    batches = list(a.batches)
    absent = [b for b in batches if not os.path.isdir(b)]
    if absent and not a.allow_missing_batches:
        sys.exit(f"STOP: {len(absent)} requested batch directory does not exist: {absent}\n"
                 "Phase D must have run for every batch before phase F. Pass "
                 "--allow-missing-batches only to process a deliberate subset.")
    batches = [b for b in batches if os.path.isdir(b)]
    if not batches:
        sys.exit(f"STOP: none of these batch directories exist: {a.batches}")
    print(f"  batches: {[os.path.basename(b) for b in batches]}")

    # ---------------------------------------------------------------- step 1
    # INVALIDATE BEFORE MUTATING ANYTHING. The record below is written at the end, but until now
    # nothing removed the PREVIOUS one first -- so a rerun that rewrote outputs and then failed left
    # the old record in place, describing files that no longer match it, and the next resume trusted
    # it. Every path from here on rewrites alignments, so the record stops being true at this line
    # and must stop existing at this line. It is moved aside rather than deleted, so a failed rerun
    # leaves evidence of what the previous run had claimed.
    os.makedirs(a.index_dir, exist_ok=True)
    _rec = os.path.join(a.index_dir, "phase_f_complete.json")
    if os.path.exists(_rec):
        os.replace(_rec, _rec + ".superseded")
        print(f"  invalidated the previous completion record -> {os.path.basename(_rec)}.superseded")
        print("  (outputs are about to be rewritten; no resume may trust the old record from here)")

    print(f"=== phase F step 1: query headers and the step4 index, {len(batches)} batch(es) ===")
    all_rows, all_dropped, seq_to_id, seq_to_index = [], [], {}, {}
    nq_total = 0
    for b in batches:
        fa = check_frozen(b)
        nq_total += len(fa)
        named = f"{W}/msa_named_{os.path.basename(b.rstrip('/'))}"
        rows, dropped, s2i, s2x = rename_headers(b, fa, named)
        # A SEQUENCE IN TWO BATCHES WOULD COLLIDE in a sequence-keyed index, silently keeping one
        # alignment for both. The batches are disjoint by construction -- each later one is a set
        # difference against the earlier fastas -- so this is a check that construction held.
        # WITHIN the batch first. seq_to_index[seq] = name overwrites silently, so two chains in one
        # batch sharing a sequence yield ONE directory and the other chain has no MSA at all. Phase C
        # defends this upstream with its duplicate check, but phase F is where the damage lands, and
        # the cross-batch case below already has two independent defences.
        if len(s2x) != len(rows):
            raise SystemExit(
                f"STOP: {os.path.basename(b)} has {len(rows)} renamed alignments but only "
                f"{len(s2x)} distinct sequences. Two chains share a sequence, so a sequence-keyed "
                "index keeps one alignment for both and the other chain gets no MSA.")
        clash = set(s2x) & set(seq_to_index)
        if clash:
            sys.exit(f"STOP: {len(clash)} sequence(s) appear in more than one batch, so the "
                     f"sequence-keyed index is ambiguous. First: {sorted(clash)[0][:40]}...")
        seq_to_id.update(s2i)
        seq_to_index.update(s2x)
        all_rows += rows
        all_dropped += dropped
        print(f"  {os.path.basename(b)}: {len(fa)} queries -> {len(rows)} renamed, {len(dropped)} dropped")
        for n, why in dropped:
            print(f"     DROPPED {n}: {why}")

    os.makedirs(a.index_dir, exist_ok=True)
    for path, obj in ((f"{a.index_dir}/seq_to_pdb_id_entity_id.json", seq_to_id),
                      (f"{a.index_dir}/seq_to_pdb_index.json", seq_to_index)):
        with open(path, "w") as fh:
            json.dump(obj, fh)
        print(f"  wrote {os.path.basename(path)}: {len(obj)} sequences")
    with open(f"{a.index_dir}/phase_f_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch", "name", "n_lines", "header_lines_rewritten"])
        w.writeheader()
        w.writerows(all_rows)

    two = sum(1 for r in all_rows if r["header_lines_rewritten"] == 2)
    print(f"  query-header lines rewritten twice (a uniref AND an env block): {two} of {len(all_rows)}")
    # DROPS ARE CLASSIFIED, NOT TOLERATED IN BULK. A flat 98% floor permitted 106 silent losses on
    # batch 1 while the only documented legitimate cause is a nucleic-acid chain the search
    # transcribed U to X -- one per batch. So: any drop for a reason OTHER than that sequence
    # mismatch is fatal, and the mismatches themselves are capped. This is the 39-of-39 discipline
    # phase A uses, rather than a percentage that happens to cover the known case.
    known = [d for d in all_dropped if "[nucleotide->X]" in d[1]]
    other = [d for d in all_dropped if d not in known]
    print(f"  drops: {len(known)} sequence mismatches (nucleic acid searched as protein), "
          f"{len(other)} for other reasons")
    if other:
        for n, why in other[:5]:
            print(f"     UNEXPECTED DROP {n}: {why}")
        sys.exit(f"STOP: {len(other)} alignment(s) dropped for reasons other than the known "
                 "nucleotide->X transformation. Each is a chain with no MSA, and the reason is "
                 "printed above rather than folded into the tolerated class.")
    if len(known) > a.max_seq_mismatch:
        sys.exit(f"STOP: {len(known)} sequence mismatches, above --max-seq-mismatch "
                 f"{a.max_seq_mismatch}. One per batch is expected; more means the a3m files and the "
                 "frozen fasta have diverged, which is what the checksum is meant to prevent.")
    need = nq_total - len(known)
    if len(seq_to_index) != need:
        sys.exit(f"STOP: the index covers {len(seq_to_index)} sequences; {nq_total} queries minus "
                 f"{len(known)} known mismatches is {need}. The difference is unexplained.")
    print(f"  index covers {len(seq_to_index)} of {nq_total} queries, all drops accounted for")

    # ---------------------------------------------------------------- step 2: taxids
    print("=== phase F step 2: step3, write NCBI taxids into the UniRef headers ===")
    os.makedirs(a.taxid_dir, exist_ok=True)
    # THE RETURN CODE IS CHECKED, and the outputs must be from THIS run. Ignoring it meant existing
    # files from an earlier attempt satisfied every count after step3 or step4 had failed outright --
    # "nonzero bytes" is not evidence that the current run produced anything. A marker is dropped
    # first and the outputs must postdate it.
    t0 = os.path.join(W, ".phase_f_step3_start")
    open(t0, "w").close()
    for b in batches:
        lbl = os.path.basename(b.rstrip("/"))
        rc = run([PY, "scripts/msa/step3-uniref_add_taxid.py",
                  "--input_msa_dir", f"{W}/msa_named_{lbl}",
                  "--output_msa_dir", a.taxid_dir,
                  "--num_workers", str(a.workers), "--mp_read_workers", str(a.workers)],
                 f"{W}/phase_f_step3_{lbl}.log", f"step3 {lbl}", a.cores, 14400)
        if rc != 0 and not a.ignore_step_rc:
            sys.exit(f"STOP: step3 on {lbl} exited {rc}. See {W}/phase_f_step3_{lbl}.log. Existing "
                     "output from an earlier attempt would otherwise satisfy the counts below.")
    # PER EXPECTED QUERY. Every renamed alignment must have a taxid-tagged counterpart from this
    # run; a reused one must still carry the query it claims to.
    expect = {r["name"] for r in all_rows}
    name_to_seq = {n: sq for sq, n in seq_to_index.items()}
    def ok_tax(n, p):
        try:
            with open(p) as fh:
                fh.readline()
                return fh.readline().strip().replace("-", "").upper() == name_to_seq[n].upper()
        except (OSError, KeyError, IndexError):
            return False
    fresh_outputs(expect, lambda n: os.path.join(a.taxid_dir, f"{n}.a3m"), t0, "step3",
                  a.allow_stale_reuse, ok_tax)
    n_out = len([f for f in os.listdir(a.taxid_dir) if f.endswith(".a3m")])
    seen = tagged = 0
    for b in batches:
        lbl = os.path.basename(b.rstrip("/"))
        bn = {r["name"] for r in all_rows if r["batch"] == lbl}
        bs, bt, nf = taxid_coverage(a.taxid_dir, bn)
        bpct = 100 * bt // bs if bs else -1
        print(f"  {lbl}: {nf} files sampled, {bs} UniRef hit records, {bt} carry a taxid ({bpct}%)")
        if bs and bpct < a.min_tagged_frac * 100:
            sys.exit(f"STOP: in batch {lbl} only {bpct}% of UniRef hit records carry a taxid "
                     f"(want >= {a.min_tagged_frac:.0%}). Every untagged hit is a sequence that "
                     "cannot be paired, and a failure in one batch does not average away.")
        seen += bs; tagged += bt
    print(f"  {n_out} a3m out; {tagged} of {seen} sampled UniRef hit records carry a taxid")
    if n_out < need:
        sys.exit(f"STOP: step3 wrote {n_out} a3m for {nq_total} queries (want >= {need})")
    if seen < 50:
        sys.exit(f"STOP: only {seen} UniRef hit records in the whole sample -- phase E looks wrong")

    # ---------------------------------------------------------------- step 3: the split
    print("=== phase F step 3: step4, split into pairing.a3m and non_pairing.a3m ===")
    os.makedirs(a.msa_dir, exist_ok=True)
    t1 = os.path.join(W, ".phase_f_step4_start")
    open(t1, "w").close()
    rc = run([PY, "scripts/msa/step4-split_msa_to_uniref_and_others.py",
              "--input_msa_dir", a.taxid_dir, "--output_msa_dir", a.msa_dir,
              "--seq_to_pdb_id", f"{a.index_dir}/seq_to_pdb_id_entity_id.json",
              "--seq_to_pdb_index", f"{a.index_dir}/seq_to_pdb_index.json",
              "--num_workers", "32"],
             f"{W}/phase_f_step4.log", "step4", a.cores, 14400)
    if rc != 0 and not a.ignore_step_rc:
        sys.exit(f"STOP: step4 exited {rc}. See {W}/phase_f_step4.log.")

    # ONE DIRECTORY PER SEQUENCE. The header-collision bug this phase exists to prevent produces
    # exactly ONE directory, so a threshold like "at least 100" would pass it. The gate is the
    # sequence count, and then that each directory's name is a query we actually indexed.
    dirs = [d for d in os.listdir(a.msa_dir)
            if os.path.isfile(os.path.join(a.msa_dir, d, "pairing.a3m"))]
    # BOTH HALVES OF THE SPLIT, per expected query. non_pairing.a3m was never checked at all, and it
    # is not optional: Protenix reads it as the unpaired MSA, so a chain with a fresh pairing.a3m and
    # a stale or absent non_pairing.a3m folds against someone else's alignment or none.
    for half in ("pairing.a3m", "non_pairing.a3m"):
        fresh_outputs(expect, lambda n, h=half: os.path.join(a.msa_dir, n, h), t1,
                      f"step4 {half}", a.allow_stale_reuse,
                      lambda n, p: os.path.exists(p))
    if len(dirs) < need:
        sys.exit(f"STOP: {len(dirs)} pairing.a3m for {nq_total} queries (want >= {need}). "
                 "One directory for everything is the header-collision failure this phase prevents.")
    known = set(seq_to_index.values())
    stray = [d for d in dirs if d not in known]
    if stray:
        sys.exit(f"STOP: {len(stray)} output directories are not names we indexed: {stray[:5]}")
    # THE COMPLETION CRITERION HAS TO TEST PAIRING, which is what this phase exists to produce.
    # Counting directories does not: if step4 ran but parsed no taxids, EVERY pairing.a3m is empty,
    # len(dirs) is correct, every name is one we indexed, and the phase reports OK. Step 2 above
    # proves taxids are in the alignments; nothing proved step4 used them.
    #
    # GATED ON THE LONG CHAINS ONLY, the same shape as the depth gates in phases D and E. A 5-19
    # residue peptide that retrieved no UniRef hits has nothing to pair, so an empty pairing.a3m
    # there is the correct answer; a 200-residue antibody chain with an empty one is a fault.
    len_of = {name: len(seq) for seq, name in seq_to_index.items()}
    def nonempty(d):
        return os.path.getsize(os.path.join(a.msa_dir, d, "pairing.a3m")) > 0
    long_d = [d for d in dirs if len_of.get(d, 0) >= 20]
    short_d = [d for d in dirs if 0 < len_of.get(d, 0) < 20]
    long_ok = [d for d in long_d if nonempty(d)]
    short_ok = [d for d in short_d if nonempty(d)]
    pct = 100 * len(long_ok) // len(long_d) if long_d else -1
    print(f"  pairing.a3m non-empty: {len(long_ok)} of {len(long_d)} chains of 20+ residues ({pct}%)")
    if short_d:
        print(f"                         {len(short_ok)} of {len(short_d)} shorter chains "
              "-- empty is expected where nothing was retrieved")
    if long_d and pct < int(a.min_pairing_frac * 100):
        sys.exit(f"STOP: only {pct}% of chains of 20+ residues have a non-empty pairing.a3m "
                 f"(floor {a.min_pairing_frac:.0%}). step4 ran but produced no pairs, so no chain "
                 "can be paired with another and the MSAs are unusable for a complex.")
    if not long_d:
        sys.exit("STOP: no chain of 20+ residues reached the split; the input looks wrong.")
    # ---------------------------------------------------------------- the completion record
    # WRITTEN ONLY HERE, AFTER EVERY GATE ABOVE HAS PASSED. phase_f_report.csv is written in step 1,
    # before step3 and step4 have run at all, so its existence says nothing about taxids or the
    # split -- and the orchestrator's resume check was comparing aggregate counts against it, which
    # leftover files from a failed attempt can satisfy while a specific chain has neither half of
    # its split. This record is the only thing that means "phase F finished": it names the batches
    # it consumed with their frozen-fasta checksums, and lists every chain it produced, so a resume
    # can verify each expected chain individually instead of trusting a total.
    done = {
        "version": 1,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "batches": [{"dir": os.path.abspath(b),
                     "sha256": open(os.path.join(b, "query.searched.fasta.sha256")).read().split()[0]
                     if os.path.exists(os.path.join(b, "query.searched.fasta.sha256")) else None,
                     "queries": len(check_frozen(b))} for b in batches],
        "msa_dir": os.path.abspath(a.msa_dir),
        "taxid_dir": os.path.abspath(a.taxid_dir),
        # SIZES, NOT JUST NAMES. Existence alone cannot distinguish a usable alignment from one
        # truncated to zero bytes by an interrupted rerun, and "non-empty" is not a valid
        # requirement either -- a short peptide that retrieved nothing legitimately has an empty
        # pairing.a3m. Recording the byte counts this run produced lets a resume detect ANY change
        # without having to know which emptiness is legitimate.
        "chains": {d: [os.path.getsize(os.path.join(a.msa_dir, d, "pairing.a3m")),
                       os.path.getsize(os.path.join(a.msa_dir, d, "non_pairing.a3m"))]
                   for d in sorted(dirs)},
    }
    # PUBLISHED ATOMICALLY, so a crash mid-write cannot leave a half-written record that parses.
    rec = os.path.join(a.index_dir, "phase_f_complete.json")
    fd, tmp = tempfile.mkstemp(dir=a.index_dir, prefix=".pfc_", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(done, fh, indent=1)
    os.replace(tmp, rec)
    print(f"  wrote {rec}: {len(dirs)} chains over {len(batches)} batch(es), with byte counts")
    print(f"=== phase F OK: {len(dirs)} chains ready for Protenix in {a.msa_dir} ===")


if __name__ == "__main__":
    main()
