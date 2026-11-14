#!/usr/bin/env python
"""Recover the per-pose OST chain mapping for poses whose ost_json is a FAILURE record. CPU only.

THE DEFECT THIS UNDOES. The folds of 2026-11-05 ran against the pre-2eb1f2fd scorer, whose `-c`
retry built its chain map by assuming the reference and the model agree on chain names. They never
do here: reference chains carry the deposited label_asym_id and model chains are named in job order.
For 4zto -- reference D/E/F, model A/B/C, and gemmi reporting the reference's AUTH names M/I/Q --
the retry sent OST "M:M I:I Q:Q" and OST refused the entire call:

    RuntimeError: Target chain "M" is not available in target processed for chain mapping
    ({'D', 'F', 'E'})

So `ost_json/<k>.json` holds {"status": "FAILURE", "traceback": ...} with no mapping in it, for
every pose of every seed folded before the fix. 4zto's first ten seeds (500000-500009, 50 poses) are
in that state; its last five (500010-500014, folded 2026-11-06 under the fixed scorer) resolved
normally. That is the whole population of the problem: the mapping was never LOST, only never
written down.

WHAT THIS DOES. Re-runs the FIXED scorer -- eval/ost_dockq, the one file every OST score in this
project goes through -- on the same pose cif and the same reference, and writes OST's answer to
`ost_json/<k>.json.rescue`. phase_k_eval.ost_chain_pairs reads that file when and only when the
original carries no mapping, so a recorded decision can never be overwritten by a re-run.

THE FAILURE RECORD IS NOT TOUCHED. It stays on disk as the provenance of why a rescue file exists.
Nothing here deletes or moves any data.

THE SUFFIX ORDER IS DELIBERATE: `<k>.json.rescue`, not `<k>.rescue.json`. Two live callers glob
`ost_json/*.json` -- insample_fullset's pre-flight gate and phase_j_drop_superseded_rows' quarantine
reader -- and a second file per pose matching that pattern would quietly change what they count.
This follows the convention ost_dockq already uses for `<k>.json.invalid`.

WHY THIS IS NOT THE chain_roles FALLBACK. chain_roles.csv is a STATIC table, one assignment per
structure, and phase_k_eval refuses it by default because it is measured wrong for chemically
identical chains -- for 7zra there is no single correct answer, so the table cannot be repaired. A
rescue json is not a table: it is OST's own per-pose decision, from the same binary and the same
flags that wrote every other mapping in this fold. The only thing late about it is the clock.

PROVEN, NOT ASSERTED. `--verify` re-runs the poses that DID resolve and compares. On 4zto's 25 such
poses the re-run reproduces the stored `chain_mapping` on 25 of 25 and a byte-identical `dockq` list
on 25 of 25. 4zto's three chains are also chemically DISTINCT by sequence -- reference D is a
113-residue light chain, E a 124-residue heavy chain, F a 10-residue peptide antigen -- so this is
NOT the identical-copies case the roles refusal warns about; the sequence-matched map agrees with
OST on 25 of 25 as well. (OST's own `chem_groups` puts D with F, which is why its automatic mapper
had to decide by geometry; the sequences say otherwise and the two agree on the answer.)

USAGE
    python -I run_logging/rescue_ost_mapping.py --pdb 4zto --verify
    python -I run_logging/rescue_ost_mapping.py                  # every structure in the fold
    python -I run_logging/rescue_ost_mapping.py --pdb 4zto --dry-run

EXIT STATUS is 0 when every pose it looked at now has a mapping, 1 when any is still unresolved --
and the unresolved ones are printed BY NAME either way, because a pose that silently vanishes from a
pool is the failure this whole file exists to stop.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # extra_training
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, os.path.join(REPO, "eval"))
import phase_j_build_cache as BC                                    # noqa: E402
import ost_dockq                                                    # noqa: E402

# OST spawns threads per structure. A GPU pass may be live on this box, so the default is a NARROW
# pin rather than ost_dockq's 0-39: the work is ~1 s per pose and there is no reason to take half
# the machine for it. OST_TASKSET in the environment still wins.
os.environ.setdefault("OST_TASKSET", "0-15")


def read_mapping(path, k):
    """(chain_mapping, why) for one ost_json, under exactly phase_k_eval.ost_chain_pairs' checks.

    Returns (None, reason) for every way the file can fail to carry a mapping for sample k. The
    reason is the string that gets printed, so it has to name the file and the defect.
    """
    if not os.path.exists(path):
        return None, "absent"
    try:
        with open(path) as fh:
            j = json.load(fh)
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__})"
    if not isinstance(j, dict):
        return None, "not a json object"
    if j.get("status") == "FAILURE":
        exc = str(j.get("exception") or j.get("traceback") or "").strip().splitlines()
        return None, f"FAILURE record: {exc[0][:90] if exc else 'no detail'}"
    mdl = str(j.get("model", ""))
    if not mdl.endswith(f"_sample_{k}.cif"):
        return None, f"describes {os.path.basename(mdl) or '<nothing>'}, not sample {k}"
    cm = j.get("chain_mapping")
    if not isinstance(cm, dict):
        return None, "no chain_mapping"
    return cm, ""


def unit_dirs(fold, pdb):
    """Every finished pose directory of a structure, newest layout or native tree.

    seed_dirs_all is the shared enumerator -- a refold leaves TWO finished directories for one
    seed and both are legitimate pose sources, so both are rescued. Which one a given evaluation
    joins to is resolve_seed_dir's decision, not this script's, and rescuing only one would leave
    the other unmapped for whoever picks it.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(fold, "poses", pdb, "*"))):
        if os.path.isdir(os.path.join(d, "ost_json")) or os.path.isdir(d):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=os.path.join(ROOT, "training/fold"))
    ap.add_argument("--refs", default=os.path.join(ROOT, "training/refs"))
    ap.add_argument("--pdb", action="append", default=[],
                    help="restrict to these structures; default is every structure in the fold")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--verify", action="store_true",
                    help="also re-run the poses that DID resolve and compare chain_mapping and "
                         "dockq against what is on disk -- the correctness proof for the rescue")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be rescued and write nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-run even where a usable rescue file already exists")
    # NEVER-SCORED IS NOT THE SAME DEFECT AND IS NOT RESCUED BY DEFAULT.
    #
    # This script undoes ONE thing: OST ran, crashed on a bad -c map, and left a FAILURE record
    # where a mapping should be. A pose with no ost_json at all was never scored, which means it
    # also has no DockQ row -- so writing it a chain mapping would hand the pipeline a half-built
    # unit that LOOKS complete at join 1 and is empty at join 3. 9dc3 is the case: ten units, no
    # ost_json anywhere, zero rows in results.csv, and chain ids two letters wide ("AB", "ID")
    # which ost_chain_pairs cannot represent at all (it requires a single-letter model chain and
    # reads its asym index off the alphabet). Scoring that structure is a separate decision about
    # the fold, not a mapping repair, so it is NAMED and skipped.
    ap.add_argument("--include-absent", action="store_true",
                    help="also run OST where there is no ost_json at all -- a NEVER-SCORED pose, "
                         "not a failed mapping. Off by default; those poses are named instead.")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    pdbs = a.pdb or sorted(os.path.basename(p) for p in
                           glob.glob(os.path.join(a.fold, "poses", "*"))
                           if os.path.isdir(p))
    if not pdbs:
        raise SystemExit(f"STOP: no structures under {a.fold}/poses")

    t0 = time.time()
    n_ok = n_resc = n_wrote = n_fail = n_norefs = n_nocif = n_absent = 0
    absent = []
    ver_n = ver_map = ver_dq = 0
    still = []
    touched = []
    for pdb in pdbs:
        ref = os.path.join(a.refs, f"{pdb}.cif")
        per = {"ok": 0, "rescued": 0, "unresolved": 0}
        for ud in unit_dirs(a.fold, pdb):
            cifs = {}
            try:
                for c in BC.sample_cifs(ud, pdb):
                    cifs[int(c.split("sample_")[-1].split(".")[0])] = c
            except Exception as e:                                       # noqa: BLE001
                print(f"  {pdb}/{os.path.basename(ud)}: cannot list sample cifs "
                      f"({type(e).__name__}: {e})", flush=True)
            for k in range(a.nsample):
                orig = os.path.join(ud, "ost_json", f"{k}.json")
                resc = os.path.join(ud, "ost_json", f"{k}.json.rescue")
                cm, why = read_mapping(orig, k)
                if cm is not None:
                    n_ok += 1
                    per["ok"] += 1
                    if a.verify and k in cifs:
                        # THE PROOF. Same pose cif, same reference, the fixed scorer -- and the
                        # answer has to be the one already on disk. A mismatch here would mean the
                        # rescue route is not reproducing OST's decision, and the whole mechanism
                        # would be unsafe; it is reported loudly rather than counted.
                        tmp = resc + ".verify"
                        if ost_dockq.run(cifs[k], ref, tmp, timeout=a.timeout):
                            g = json.load(open(tmp))
                            ver_n += 1
                            ver_map += (g.get("chain_mapping") == cm)
                            ver_dq += (g.get("dockq") == json.load(open(orig)).get("dockq"))
                            if g.get("chain_mapping") != cm:
                                print(f"  VERIFY MISMATCH {pdb}/{os.path.basename(ud)} sample {k}: "
                                      f"disk {cm} vs re-run {g.get('chain_mapping')}", flush=True)
                        else:
                            print(f"  VERIFY could not re-run {pdb}/{os.path.basename(ud)} "
                                  f"sample {k}", flush=True)
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                    continue
                # The original carries no mapping. Is a rescue already there?
                rcm, rwhy = read_mapping(resc, k)
                if rcm is not None and not a.force:
                    n_resc += 1
                    per["rescued"] += 1
                    continue
                name = f"{pdb}/{os.path.basename(ud)} sample {k}"
                if why == "absent" and not a.include_absent:
                    n_absent += 1
                    absent.append(name)
                    continue
                if not os.path.exists(ref):
                    n_norefs += 1
                    still.append(f"{name}: {why}; no reference cif {ref}")
                    per["unresolved"] += 1
                    continue
                if k not in cifs:
                    n_nocif += 1
                    still.append(f"{name}: {why}; no _sample_{k}.cif on disk")
                    per["unresolved"] += 1
                    continue
                if a.dry_run:
                    print(f"  WOULD RESCUE {name} ({why})", flush=True)
                    per["rescued"] += 1
                    n_resc += 1
                    continue
                # WRITTEN VIA A TEMP AND os.replace. A half-written rescue json read by a concurrent
                # evaluation is a mapping that parses and is wrong, which is worse than none.
                tmp = resc + ".tmp"
                ok = ost_dockq.run(cifs[k], ref, tmp, timeout=a.timeout)
                gcm, gwhy = read_mapping(tmp, k) if ok else (None, "OST produced no usable json")
                if gcm is None:
                    n_fail += 1
                    per["unresolved"] += 1
                    still.append(f"{name}: {why}; re-run did not resolve it either ({gwhy})")
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    continue
                os.replace(tmp, resc)
                n_wrote += 1
                n_resc += 1
                per["rescued"] += 1
                touched.append(name)
        if per["rescued"] or per["unresolved"]:
            print(f"  {pdb}: {per['ok']} pose(s) already mapped, {per['rescued']} rescued, "
                  f"{per['unresolved']} still unresolved", flush=True)

    print(f"\n{len(pdbs)} structure(s) in {time.time() - t0:.0f}s: {n_ok} pose(s) already carried a "
          f"mapping, {n_resc} have one from a rescue file ({n_wrote} written now), "
          f"{n_fail + n_norefs + n_nocif} still have none "
          f"(re-run failed {n_fail}, no reference {n_norefs}, no sample cif {n_nocif})")
    if a.verify:
        print(f"VERIFY on the poses that already resolved: {ver_n} re-run, chain_mapping "
              f"reproduced on {ver_map}, dockq reproduced on {ver_dq}")
        if ver_n and (ver_map != ver_n or ver_dq != ver_n):
            raise SystemExit("STOP: re-running OST did not reproduce a mapping that is already on "
                             "disk, so the rescue route cannot be trusted for the poses that have "
                             "none. Nothing written here should be used.")
    if n_absent:
        _ap = sorted({x.split("/")[0] for x in absent})
        print(f"\nNEVER SCORED -- {n_absent} pose(s) across {len(_ap)} structure(s) have no "
              f"ost_json at all, so there is no failed mapping to repair and no DockQ row either: "
              f"{', '.join(_ap)}. Pass --include-absent to run OST on them anyway; whether they "
              f"belong in the fold is a separate question from this one.")
        for x in absent[:20]:
            print(f"    {x}")
        if len(absent) > 20:
            print(f"    ... and {len(absent) - 20} more")
    if still:
        # BY NAME, ALWAYS. The point of this script is that a pose nobody can map is a named pose,
        # not a quietly smaller pool.
        print(f"\nSTILL UNRESOLVED -- {len(still)} pose(s), by name:")
        for s in still[:200]:
            print(f"    {s}")
        if len(still) > 200:
            print(f"    ... and {len(still) - 200} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
