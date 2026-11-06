#!/usr/bin/env python
"""THE FROZEN POPULATION for the extended 853-structure training set, with its per-structure depth.

WHY A `seeds` COLUMN. phase_k_train builds its manifest as the selection csv times ONE seed range.
That is right while every structure has the same depth and wrong now: 437 structures carry seeds
500000-500014 and 416 carry 500000 alone, because the other trunks of those 416 were reclaimed on
2026-11-04 (290.6 GiB). A single range is wrong for whichever group it is not describing, and the
trainer's incompleteness guard then refuses the run for being the shape it is meant to be. The
column declares each structure's own depth; a row without one still falls back to --seeds.

THE DEPTH IS DECLARED FROM THE TRIM DECISION, NOT DISCOVERED FROM THE DISK. Globbing the trunks
would turn a lost unit into a quietly smaller training set rather than an error, which is the defect
phase_k_train's review round 8 removed and this file must not reintroduce. So the 437 come from
docs/extra_seed_structures.txt -- the frozen list the trim and the seed chain both acted on -- and
everything else in the pool is declared at --shallow-seeds.

IT REPORTS THE DISK AS A SEPARATE QUESTION. --require-ready checks every declared (pdb, seed) for a
trunk and a label and refuses if any is absent, which is the pre-launch gate. Without it the counts
are printed and the csv is still written, because the labels are built AFTER this file names what to
build.

TOKENS COME FROM THE INDICES CSV, max over a structure's rows, because --epoch-order tokens sorts
by it and a structure with no number sorts first and silently leads the size-ordered schedule.
"""
import argparse, collections, csv, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/combined/pool.csv",
                    help="the population: every structure named by this pool")
    ap.add_argument("--deep", default="docs/extra_seed_structures.txt",
                    help="the frozen non-degenerate list that kept its seeds and got the new ones")
    ap.add_argument("--deep-seeds", default="500000-500014")
    ap.add_argument("--shallow-seeds", default="500000",
                    help="the depth of everything else: the surviving trunk after the trim")
    ap.add_argument("--indices", default="training/prep/all857_indices.csv")
    ap.add_argument("--prev", default="train_val_splits/cache_train_structures.csv",
                    help="the previous selection, read only to carry forward its extra columns")
    ap.add_argument("--labels", default="training/fold/labels")
    ap.add_argument("--trunks", default="training/fold/trunks")
    ap.add_argument("--skip", default="9mx1",
                    help="documented exclusions; 9mx1 is 2,941 residues against a 2,560-token limit")
    ap.add_argument("--out", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--require-ready", action="store_true",
                    help="refuse unless every declared (pdb, seed) has BOTH a trunk and a label")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    skip = {x.strip().lower() for x in a.skip.split(",") if x.strip()}
    pool = sorted({r["pdb"].strip().lower() for r in csv.DictReader(open(a.pool))})
    deep = {l.strip().lower() for l in open(a.deep) if l.strip() and l[0] != "#"}
    pop = [p for p in pool if p not in skip]

    # EVERY DEEP STRUCTURE MUST BE IN THE POOL, or the frozen list and the pool disagree about the
    # population and one of them is stale. That is a refusal, not a warning: the seed chain folded
    # the frozen list, so a name it carries that the pool does not is 15 units of GPU spent on a
    # structure this csv would then leave out.
    orphan = sorted(deep - set(pool))
    if orphan:
        raise SystemExit(f"STOP: {len(orphan)} structure(s) in {a.deep} are not in {a.pool}: "
                         f"{orphan[:8]}. The frozen list and the pool disagree.")

    tokens = {}
    with open(a.indices) as fh:
        for r in csv.DictReader(fh):
            p = (r.get("pdb_id") or "").strip().lower()
            try:
                n = int(float(r.get("num_tokens") or 0))
            except ValueError:
                continue
            if p and n > tokens.get(p, 0):
                tokens[p] = n
    prev = {}
    if os.path.exists(a.prev):
        prev = {r["pdb"].strip().lower(): r for r in csv.DictReader(open(a.prev))}

    def seeds_of(p):
        return a.deep_seeds if p in deep else a.shallow_seeds

    def expand(txt):
        out = set()
        for part in txt.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                out |= set(range(lo, hi + 1))
            else:
                out.add(int(part))
        return out

    no_tok = [p for p in pop if p not in tokens]
    if no_tok:
        raise SystemExit(f"STOP: {len(no_tok)} structure(s) have no num_tokens in {a.indices}, so "
                         f"--epoch-order tokens would sort them first: {no_tok[:8]}")

    rows, depth_hist = [], collections.Counter()
    for p in pop:
        sd = seeds_of(p)
        depth_hist[len(expand(sd))] += 1
        base = prev.get(p, {})
        rows.append({
            "pdb": p,
            "seeds": sd,
            "tokens": tokens[p],
            "cluster": base.get("cluster", ""),
            "instance": base.get("instance", ""),
            "ab_cluster": base.get("ab_cluster", ""),
            "agclusters": base.get("agclusters", ""),
            "n_chains": base.get("n_chains", ""),
            "resolution": base.get("resolution", ""),
            "method": base.get("method", ""),
            "msa_ok": base.get("msa_ok", ""),
        })

    units = sum(len(expand(r["seeds"])) for r in rows)
    print(f"population        : {len(pool)} pooled structure(s) minus {sorted(skip)} = {len(pop)}")
    print(f"  deep ({a.deep_seeds})   : {sum(1 for p in pop if p in deep)}")
    print(f"  shallow ({a.shallow_seeds}) : {sum(1 for p in pop if p not in deep)}")
    print(f"  depth histogram : {dict(sorted(depth_hist.items()))}")
    print(f"  declared units  : {units}")
    print(f"  carried forward from {a.prev}: {sum(1 for p in pop if p in prev)} row(s)")

    def present(root, p, s):
        return os.path.exists(os.path.join(root, p, f"free_{s}.npz"))

    miss_t = miss_l = 0
    ex_t, ex_l = [], []
    for r in rows:
        for s in sorted(expand(r["seeds"])):
            if not present(a.trunks, r["pdb"], s):
                miss_t += 1
                if len(ex_t) < 5:
                    ex_t.append(f"{r['pdb']}/free_{s}")
            if not present(a.labels, r["pdb"], s):
                miss_l += 1
                if len(ex_l) < 5:
                    ex_l.append(f"{r['pdb']}/free_{s}")
    print(f"  declared units with NO trunk: {miss_t}  {ex_t}")
    print(f"  declared units with NO label: {miss_l}  {ex_l}")

    if a.require_ready and (miss_t or miss_l):
        print("NOT READY: the trainer refuses a manifest with a missing label or trunk.")
        return 1

    if a.write:
        if os.path.abspath(a.out) in (os.path.abspath(a.pool), os.path.abspath(a.prev)):
            raise SystemExit("STOP: --out would overwrite an input")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cols = ["cluster", "pdb", "instance", "ab_cluster", "agclusters", "tokens", "n_chains",
                "resolution", "method", "msa_ok", "seeds"]
        tmp = a.out + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, a.out)
        print(f"  wrote {a.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
