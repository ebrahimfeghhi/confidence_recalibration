"""Queue the folds that can run NOW: the ones whose query is not changing.

THE SPLIT. Every (structure, seed) unit with no trunk on disk falls into one of two groups:

  A. the structure is NOT in the refold list, so its query is unchanged and the MSA already on
     disk is still valid for it. Runnable immediately.
  B. the structure IS in the refold list, so its query grows by the residues being restored. Its
     stored MSA has the wrong column count for the new query and folding it now would produce an
     input that has to be thrown away. Deferred until the corrected MSAs are built.

Keeping these apart is the whole point. `watch_foldbench_then_training.sh` hands all three cards to
the training queue unattended when the FoldBench top-off drains, so any group-B row sitting in that
file would be folded against a stale query with nobody watching.

WHY THE QUEUE WAS ALREADY SHORT BY SEVEN STRUCTURES. The queue was trimmed against the EARLIER
refold list, which was built from a G/S composition guess at the VH/VL splice point. The domain
audit dropped 32nz, 8cz8, 8pww, 8qp6, 8vy9, 9ejz and 9tuy from that list -- six had pure tether
queued for restoration and 8qp6 is a single-chain Fab whose "contamination" is CH1. Their queries
are therefore NOT changing, which moves them from group B to group A, and they have to be put back
into the queue they were trimmed out of.

MSA PRESENCE IS CHECKED THROUGH THE JOB JSON, NOT BY GUESSING A DIRECTORY NAME. An MSA directory is
named after whichever structure first contributed that sequence, so 8vy9's alignments live in
`c171_6n4b_H_c50590` and 9oun's in `c766_9ouo_H_0268aa`. Globbing for the pdb id in the directory
name reported seven group-A structures as having no MSA at all; every one of them resolves fine
through the path its job actually carries.
"""
import argparse, csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def job_msas_present(jobs_dir, pdb):
    """(ok, n_chains, n_missing) for the MSA paths this structure's job really references."""
    hits = glob.glob(os.path.join(jobs_dir, f"{pdb}-*.json")) or \
           glob.glob(os.path.join(jobs_dir, f"{pdb}.json"))
    if not hits:
        return False, 0, 0
    d = json.load(open(hits[0]))
    rec = d[0] if isinstance(d, list) else d
    n = bad = 0
    for ent in rec.get("sequences", []):
        pc = ent.get("proteinChain")
        if not pc:
            continue
        n += 1
        for k in ("pairedMsaPath", "unpairedMsaPath"):
            p = pc.get(k)
            if p and not os.path.exists(p):
                bad += 1
                break
    return bad == 0, n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True, help="training/fold or validation/fold")
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--refold", required=True, help="the refold list for this split")
    ap.add_argument("--seeds", required=True, help="e.g. 500000-500009")
    ap.add_argument("--structures", default="", help="csv with a pdb column; default: the job dir")
    ap.add_argument("--tag", default="free")
    ap.add_argument("--out", required=True, help="queue to WRITE (group A only)")
    ap.add_argument("--deferred", default="", help="where to record group B")
    ap.add_argument("--apply", action="store_true",
                    help="without this nothing is written; the counts are printed and that is all")
    a = ap.parse_args()

    s0, s1 = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(s0, s1 + 1))
    if a.structures:
        pdbs = [r["pdb"] for r in csv.DictReader(open(a.structures))]
    else:
        pdbs = sorted({os.path.basename(p).split("-")[0].split(".")[0]
                       for p in glob.glob(os.path.join(a.jobs_dir, "*.json"))})
    refold = {r["pdb"] for r in csv.DictReader(open(a.refold))}

    A, B, nomsa, done = [], [], {}, 0
    for p in pdbs:
        for s in seeds:
            if os.path.exists(os.path.join(a.fold_dir, "trunks", p, f"{a.tag}_{s}.npz")):
                done += 1
                continue
            (B if p in refold else A).append((p, s))
    for p in sorted({x for x, _ in A}):
        ok, n, bad = job_msas_present(a.jobs_dir, p)
        if not ok:
            nomsa[p] = (n, bad)

    runnable = [(p, s) for p, s in A if p not in nomsa]
    print("%s  %d structures x %d seeds = %d units" % (a.fold_dir, len(pdbs), len(seeds),
                                                       len(pdbs) * len(seeds)))
    print("  trunk already on disk        : %d" % done)
    print("  A) runnable now              : %d units over %d structures"
          % (len(runnable), len({p for p, _ in runnable})))
    print("  B) deferred, query changing  : %d units over %d structures"
          % (len(B), len({p for p, _ in B})))
    if nomsa:
        print("  WITHHELD, a job MSA path is missing: %d structures" % len(nomsa))
        for p, (n, bad) in sorted(nomsa.items()):
            print("     %-5s %d of %d chain path(s) absent" % (p, bad, n))

    if not a.apply:
        print("\n  --apply not given: nothing written")
        return
    with open(a.out, "w") as fh:
        for p, s in sorted(runnable):
            fh.write(f"{a.tag}\t{p}\t*\t{a.tag}\t{s}\n")
    print("\nwrote %s (%d rows)" % (a.out, len(runnable)))
    if a.deferred:
        with open(a.deferred, "w") as fh:
            fh.write("pdb,seed,reason\n")
            for p, s in sorted(B):
                fh.write(f"{p},{s},in the refold list; needs the corrected MSA\n")
        print("wrote %s (%d rows)" % (a.deferred, len(B)))


if __name__ == "__main__":
    main()
