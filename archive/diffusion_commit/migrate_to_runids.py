#!/usr/bin/env python
"""One-off: put every free fold into <experiment>/<pdb>/<seed>-<10char>/ with its own run.json.

WHY. Run directories had drifted into three shapes in one tree:

    free/<pdb>/<seed>/legacy-no-provenance/          the original pool folds (no metadata)
    free/free/<pdb>/<seed>/<timestamp-pid-hash>/     the 100-seed sweep -- nested because --out was
                                                     the tree root and fold.py adds its own kind level
    free/<pdb>/<seed>/<timestamp-pid-hash>/          a few from an earlier layout

AFTER: one shape, one folder per fold, the id unique per run.

    free/<pdb>/<seed>-<10char>/run.json
    free/<pdb>/<seed>-legacy-no-provenance/run.json   <- the pool folds keep their honest marker

THE MARKER IS DELIBERATE. Those 7743 folds predate run provenance: no id, and their configuration is
known only from what the defaults were at the time (bf16, N_cycle 10). Minting a random id for them
would make them indistinguishable from runs whose settings were actually recorded, so the absence
stays visible in the name and run.json says so in words.

The 10-char id replaces a 4-char-hash scheme that collided: template ids were md5 into 36^3 = 46656
slots shared by every fold of a run, and 720 of ~3300 ended up reused, so one fold could overwrite a
template another was using. Ids that name a FOLDER can be long; only Protenix's template ids must be
4 chars, and those are now scoped to one fold.

    migrate_to_runids.py --root <experimental_data/free> [--apply]
"""
import argparse
import json
import os
import random
import shutil
import string
import time

ALPHABET = string.ascii_lowercase + string.digits
LEGACY = "legacy-no-provenance"


def new_id(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(10))


def plan(root):
    """[(src run dir, pdb, seed, old_name)] for every fold found under either tree shape."""
    out = []
    for pdb in sorted(os.listdir(root)):
        pdir = os.path.join(root, pdb)
        if not os.path.isdir(pdir) or pdb in ("logs", "ERR"):
            continue
        # tree B: an extra "free" level from --out colliding with fold.py's kind level
        depth_extra = pdb == "free"
        bases = [os.path.join(pdir, x) for x in sorted(os.listdir(pdir))] if depth_extra else [pdir]
        for base in bases:
            if not os.path.isdir(base):
                continue
            real_pdb = os.path.basename(base) if depth_extra else pdb
            for seed in sorted(os.listdir(base)):
                sdir = os.path.join(base, seed)
                if not os.path.isdir(sdir) or not seed.isdigit():
                    continue
                for run in sorted(os.listdir(sdir)):
                    rdir = os.path.join(sdir, run)
                    if os.path.isdir(rdir):
                        out.append((rdir, real_pdb, seed, run))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="rng seed, so a dry run matches the apply")
    A = ap.parse_args()
    rng = random.Random(A.seed)

    moves, legacy, meta_written = [], 0, 0
    for rdir, pdb, seed, run in plan(A.root):
        if run == LEGACY:
            name = f"{seed}-{LEGACY}"
            legacy += 1
        else:
            name = f"{seed}-{new_id(rng)}"
        moves.append((rdir, os.path.join(A.root, pdb, name), pdb, seed, run))

    clash = [m for m in moves if os.path.exists(m[1])]
    print(f"folds to move   : {len(moves)}  (legacy, marker kept: {legacy})")
    print(f"destination clash: {len(clash)}")
    if clash:
        print("REFUSING -- destination exists:")
        for m in clash[:5]:
            print("  ", m[1])
        return 1
    for m in moves[:3]:
        print(f"  {m[0]}\n    -> {m[1]}")

    if not A.apply:
        print("\ndry run -- pass --apply to perform the moves")
        return 0

    for src, dst, pdb, seed, run in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        rj = os.path.join(dst, "run.json")
        if os.path.exists(rj):
            # keep what the fold recorded, add the fields the new layout guarantees
            try:
                d = json.load(open(rj))
            except Exception:
                d = {}
        else:
            d = {}
        d.setdefault("pdb", pdb)
        d["seed"] = int(seed)                       # authoritative here, not in the folder name
        d.setdefault("kind", "free")
        d["run_dir"] = os.path.basename(dst)
        d["migrated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if run == LEGACY:
            d["provenance"] = ("UNKNOWN -- this fold predates run.json. precision and recycle count "
                               "are INFERRED from the defaults in force at the time (bf16, "
                               "N_cycle=10), not recorded.")
        d["templated"] = os.path.isdir(os.path.join(dst, "templates"))
        with open(rj, "w") as fh:
            json.dump(d, fh, indent=1)
        meta_written += 1

    # prune the emptied seed/kind levels, deepest first
    for _ in range(4):
        for cur, dirs, files in os.walk(A.root, topdown=False):
            if cur != A.root and not dirs and not files:
                os.rmdir(cur)

    # an index so runs are findable by grep rather than by walking every folder
    idx = os.path.join(A.root, "index.tsv")
    with open(idx, "w") as fh:
        fh.write("pdb\tseed\trun_dir\tkind\tprecision\tn_cycle\ttemplated\n")
        for pdb in sorted(os.listdir(A.root)):
            pdir = os.path.join(A.root, pdb)
            if not os.path.isdir(pdir):
                continue
            for run in sorted(os.listdir(pdir)):
                rj = os.path.join(pdir, run, "run.json")
                if not os.path.exists(rj):
                    continue
                try:
                    d = json.load(open(rj))
                except Exception:
                    continue
                fh.write(f"{d.get('pdb', pdb)}\t{d.get('seed', '')}\t{run}\t{d.get('kind', '')}\t"
                         f"{d.get('precision', '')}\t{d.get('n_cycle', '')}\t"
                         f"{d.get('templated', '')}\n")
    print(f"moved {len(moves)} folds, wrote {meta_written} run.json, indexed at {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
