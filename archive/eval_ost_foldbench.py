"""
Evaluate antibody-antigen DockQ using FoldBench's exact methodology.
Saves DockQ for ALL diffusion samples per seed (not just the best-ranked one),
so any selection strategy can be tested in post-processing without re-running OST.

Key design:
  - OST is run ONCE per (structure, seed, sample) even for multi-interface structures,
    then DockQ is extracted for each interface from the single OST output.
  - chain_mapping (ref→model chain dict) is saved for reference.

Output: one row per (assembly_id, chain1, chain2, seed, sample)
Columns: assembly_id, chain1, chain2, seed, sample,
         dockq,                 # interface DockQ (primary metric)
         pair_iptm,             # this interface's chain-pair ipTM
         global_iptm,           # whole-complex ipTM
         ptm,                   # whole-complex pTM
         pair_ranking_score,    # 0.8*pair_iptm   + 0.2*ptm - 100*clash
         global_ranking_score,  # 0.8*global_iptm + 0.2*ptm - 100*clash
         chain_mapping          # OST ref->model chain map (debug)

Usage:
  python eval_ost_foldbench.py --pred_dir foldbench_abag/predictions_template
"""
import argparse
import csv
import json
import os
import subprocess
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
import sys
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
import paths

PROTENIX_PYTHON = str(paths.PROTENIX_PYTHON)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ost_dockq as _ost      # the single OST entry point for the whole project

OST_BIN = str(paths.OST_BIN)
OST_ENV = {**os.environ, "LD_LIBRARY_PATH": str(paths.OST_LIB)}

GROUND_TRUTH = paths.GROUND_TRUTHS
TARGETS_CSV  = paths.TARGETS_CSV

# Process pool, not threads: evaluate_group does in-process biotite CIF work (postprocess +
# the isolated-pair rescue's chain subsetting), which holds the GIL. With threads, dense
# failures (e.g. 8f60's antigen rescue on every group) serialize the whole pool and it crawls.
# Separate processes sidestep the GIL entirely. OST itself is a subprocess, so this is I/O-bound.
# Shared box: never hog it. Override with OST_WORKERS=N (default keeps the old behaviour).
MAX_WORKERS = int(os.environ.get("OST_WORKERS", min(160, (os.cpu_count() or 8))))
FIELDNAMES  = ["assembly_id", "chain1", "chain2", "seed", "sample",
               "dockq", "pair_iptm", "global_iptm", "ptm",
               "pair_ranking_score", "global_ranking_score", "chain_mapping",
               "dockq_status"]

POSTPROCESS_SCRIPT = """
import sys
import biotite.structure.io.pdbx as pdbx
src, dst = sys.argv[1], sys.argv[2]
cif_file = pdbx.CIFFile.read(src)
block = cif_file.block
atom_site = block.get('atom_site')
n = len(atom_site['group_PDB'])
atom_site['occupancy']      = pdbx.CIFColumn(pdbx.CIFData(['1'] * n))
atom_site['B_iso_or_equiv'] = pdbx.CIFColumn(pdbx.CIFData(['0'] * n))
dict_entity_id = {}
for eid, grp in zip(atom_site['label_entity_id'].as_array().tolist(),
                    atom_site['group_PDB'].as_array().tolist()):
    if eid not in dict_entity_id:
        dict_entity_id[eid] = ('polymer' if grp == 'ATOM' else 'non-polymer', grp)
    elif dict_entity_id[eid][1] != grp:
        dict_entity_id[eid] = ('polymer', grp)
block['entity'] = pdbx.CIFCategory({
    'id':   list(dict_entity_id.keys()),
    'type': [dict_entity_id[k][0] for k in dict_entity_id.keys()],
})
cif_file.write(dst)
"""


try:
    import biotite.structure.io.pdbx as _pdbx
    _HAVE_BIOTITE = True
except Exception:
    _HAVE_BIOTITE = False


def _postprocess_cif_inproc(src: Path, dst: Path) -> bool:
    """Same patch as POSTPROCESS_SCRIPT, but in-process (no subprocess)."""
    cif_file = _pdbx.CIFFile.read(str(src))
    block = cif_file.block
    atom_site = block.get("atom_site")
    n = len(atom_site["group_PDB"])
    atom_site["occupancy"]      = _pdbx.CIFColumn(_pdbx.CIFData(["1"] * n))
    atom_site["B_iso_or_equiv"] = _pdbx.CIFColumn(_pdbx.CIFData(["0"] * n))
    dict_entity_id = {}
    for eid, grp in zip(atom_site["label_entity_id"].as_array().tolist(),
                        atom_site["group_PDB"].as_array().tolist()):
        if eid not in dict_entity_id:
            dict_entity_id[eid] = ("polymer" if grp == "ATOM" else "non-polymer", grp)
        elif dict_entity_id[eid][1] != grp:
            dict_entity_id[eid] = ("polymer", grp)
    block["entity"] = _pdbx.CIFCategory({
        "id":   list(dict_entity_id.keys()),
        "type": [dict_entity_id[k][0] for k in dict_entity_id.keys()],
    })
    cif_file.write(str(dst))
    return True


def postprocess_cif(src: Path, dst: Path) -> bool:
    """Patch occupancy/B-factor/entity so OST can read a predicted CIF.

    Done in-process when biotite is importable (faster, and the eval no longer
    needs the Protenix venv just for this); otherwise falls back to running the
    same code via a PROTENIX_PYTHON subprocess.
    """
    if _HAVE_BIOTITE:
        try:
            return _postprocess_cif_inproc(src, dst)
        except Exception:
            pass  # any in-process failure -> try the subprocess path below
    r = subprocess.run(
        [PROTENIX_PYTHON, "-c", POSTPROCESS_SCRIPT, str(src), str(dst)],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0


# THE OST CALL LIVES IN eval/ost_dockq.py -- these are thin aliases, kept because this file's
# call sites (and its docstrings) refer to them by name. The command line, the --rigid-scores strip,
# the "exit 0 but empty json" check and the identity-map retry are all defined there now; they used
# to be duplicated here, and the duplicate is what drifted.
_run_ost_once = _ost.invoke     # one call, no skip, no retry -- caller controls the chain map
run_ost = _ost.run              # skip-if-done, auto map, then identity-map retry on failure


def chain_to_asym_id(chain_id: str) -> int:
    """Convert a chain label (e.g. 'A', 'AA') to a 0-based index into chain_pair_iptm.
    Use only the leading alphabetic run, so a digit-suffixed model chain like 'A0' maps to 0
    instead of ord('0')-based garbage (FoldBench's own predictions name chains 'A0'/'B0'/...)."""
    base = chain_id.split("-")[0]
    alpha = ""
    for ch in base:
        if ch.isalpha():
            alpha += ch
        else:
            break
    idx = 0
    for ch in alpha:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _mapping_lookup(chain_mapping: dict, chain: str):
    """ref chain label -> model chain. Exact match first (chain_mapping keys are exact reference
    names including any copy suffix). If the requested label simply isn't a key, fall back to a
    UNIQUE copy-suffix-stripped key — this recovers a duplicate-copy native (8b7h: CSV asks 'A',
    OST mapped 'A-2'). The exact-first order preserves homo-oligomer correctness: when both 'A'
    and 'A-2' are keys, 'A' matches exactly and is never collapsed; an ambiguous strip returns None."""
    if chain in chain_mapping:
        return chain_mapping[chain]
    base = chain.split("-")[0]
    hits = [v for k, v in chain_mapping.items() if k.split("-")[0] == base]
    return hits[0] if len(hits) == 1 else None


def compute_confidence_scores(conf: dict, chain_mapping: dict, chain1: str, chain2: str) -> dict:
    """Return pair_iptm, pair_ranking_score, global_iptm, ptm, global_ranking_score.

    Uses OST chain_mapping (ref->model) so that pair_iptm correctly indexes
    chain_pair_iptm even when model chains are not in the same order as reference chains.
    """
    nan = float("nan")
    ptm       = conf.get("ptm", 0)
    has_clash = conf.get("has_clash", 0)

    matrix = conf.get("chain_pair_iptm", [])
    if matrix and chain_mapping:
        m1 = _mapping_lookup(chain_mapping, chain1)
        m2 = _mapping_lookup(chain_mapping, chain2)
        if m1 is not None and m2 is not None:
            try:
                i, j = chain_to_asym_id(m1), chain_to_asym_id(m2)
                pair_iptm = (matrix[i][j] + matrix[j][i]) / 2
                pair_ranking_score = 0.8 * pair_iptm + 0.2 * ptm - 100 * has_clash
            except (IndexError, TypeError):
                pair_iptm = pair_ranking_score = nan
        else:
            pair_iptm = pair_ranking_score = nan
    else:
        pair_iptm = pair_ranking_score = nan

    global_iptm = conf.get("iptm", nan)
    global_ranking_score = (
        0.8 * global_iptm + 0.2 * ptm - 100 * has_clash
        if global_iptm == global_iptm else nan
    )

    return {
        "pair_iptm": pair_iptm,
        "pair_ranking_score": pair_ranking_score,
        "global_iptm": global_iptm,
        "ptm": ptm,
        "global_ranking_score": global_ranking_score,
    }


def _load_ost_json(path: Path):
    """Parse an OST output JSON; return dict or None (never raises)."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _iface_match_index(ost_data: dict, chain1: str, chain2: str):
    """Index of the dockq_interfaces entry for reference interface (chain1, chain2).

    Positions 0,1 of each dockq_interfaces tuple are the REFERENCE chains (they equal
    dockq_reference_interfaces[i]); 2,3 are the model chains. Match exactly first, then by
    copy-suffix-stripped labels (strip a trailing '-N') so a native where OST maps the model
    onto a duplicate copy is still found (e.g. 8b7h: request (A,C) vs reported (A-2,C-2)). The
    suffix match is accepted ONLY when unique — a genuine same-letter homo-oligomer
    (A, A-2, A-3 ...) yields several and must not be guessed.
    Returns (index | None, ambiguous: bool).
    """
    ifaces = ost_data.get("dockq_interfaces") or []
    for i, iface in enumerate(ifaces):
        c0, c1 = iface[0], iface[1]
        if (c0 == chain1 and c1 == chain2) or (c0 == chain2 and c1 == chain1):
            return i, False
    s1, s2 = chain1.split("-")[0], chain2.split("-")[0]
    hits = [i for i, iface in enumerate(ifaces)
            if {iface[0].split("-")[0], iface[1].split("-")[0]} == {s1, s2}]
    if len(hits) == 1:
        return hits[0], False
    return None, len(hits) > 1


def _usable(ost_data: dict, iface_list) -> bool:
    """True if ANY requested reference interface is present in the OST output (exact or unique
    suffix match). Decides whether to retry OST with a forced identity chain map."""
    if not ost_data:
        return False
    return any(_iface_match_index(ost_data, c1, c2)[0] is not None
               for _, c1, c2 in iface_list)


def extract_dockq(ost_data: dict, chain1: str, chain2: str, chain_mapping: dict):
    """Return (dockq, status) for a reference interface.

    status:
      ok               interface found (exact or unique copy-suffix match) -> real DockQ
      ost_fail         OST produced no usable chain mapping -> NaN
      ambiguous_copy   only ambiguous (multi-copy homo-oligomer) suffix matches -> NaN
      interface_absent both requested chains ARE mapped but this pair isn't scored
                       -> genuine dock failure, DockQ 0.0
      copy_mismatch    a requested chain is absent but a copy-suffixed variant IS mapped
                       (asked 'C', mapping has 'C-2'; 8b7h cross-copy) -> real dock under a
                       different label, so DockQ stays NaN (do NOT coerce to 0)
      chain_unmapped   a requested chain was never mapped and no copy-variant either (8f60's
                       antigen C) -> model produced no such interface, DockQ 0.0
    """
    nan = float("nan")
    idx, ambiguous = _iface_match_index(ost_data, chain1, chain2)
    if idx is not None:
        return ost_data["dockq"][idx], "ok"

    mapping = chain_mapping or {}
    if not mapping:
        return nan, "ost_fail"          # empty mapping == OST outright failure (8c5h/8dtn/8ts0)
    if ambiguous:
        return nan, "ambiguous_copy"

    def _variant(ch):
        base = ch.split("-")[0]
        return any(k != ch and k.split("-")[0] == base for k in mapping)

    k1, k2 = chain1 in mapping, chain2 in mapping
    if k1 and k2:
        return 0.0, "interface_absent"
    if (not k1 and _variant(chain1)) or (not k2 and _variant(chain2)):
        return nan, "copy_mismatch"
    return 0.0, "chain_unmapped"


def _subset_cif(src: Path, dst: Path, keep_chains) -> bool:
    """Write dst keeping only atom_site rows whose label_asym_id is in keep_chains, with a minimal
    all-polymer entity block. Returns False if biotite is unavailable or it fails."""
    if not _HAVE_BIOTITE:
        return False
    try:
        f = _pdbx.CIFFile.read(str(src))
        blk = f.block
        at = blk.get("atom_site")
        lab = at["label_asym_id"].as_array().tolist()
        idx = [i for i, c in enumerate(lab) if c in keep_chains]
        if not idx:
            return False
        blk["atom_site"] = _pdbx.CIFCategory(
            {c: _pdbx.CIFData([at[c].as_array().tolist()[i] for i in idx]) for c in at.keys()})
        ent, eids = at["label_entity_id"].as_array().tolist(), []
        for i in idx:
            if ent[i] not in eids:
                eids.append(ent[i])
        blk["entity"] = _pdbx.CIFCategory({"id": eids, "type": ["polymer"] * len(eids)})
        f.write(str(dst))
        return True
    except Exception:
        return False


def _isolated_pair_dockq(model_cif: Path, native_cif: Path, chain1: str, chain2: str):
    """DockQ of interface (chain1, chain2) scored in ISOLATION: subset both structures to just
    those two chains so OST's chem-group mapper can't mis-group the antigen with an antibody chain
    (e.g. 8f60, whose BTLA antigen is Ig-like and gets grouped with the antibody, dropping it).
    Validated score-preserving vs the full complex on working targets (8urf). Returns float|None.
    Requires biotite (returns None otherwise)."""
    if not _HAVE_BIOTITE:
        return None
    pair = {chain1, chain2}
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            mp, mps, ns, outj = td / "m.cif", td / "ms.cif", td / "ns.cif", td / "o.json"
            if not postprocess_cif(model_cif, mp):
                return None
            if not (_subset_cif(mp, mps, pair) and _subset_cif(native_cif, ns, pair)):
                return None
            if not _run_ost_once(mps, ns, outj):
                return None
            d = _load_ost_json(outj)
            if not d:
                return None
            idx, _ = _iface_match_index(d, chain1, chain2)
            if idx is not None:
                return d["dockq"][idx]
            dq = d.get("dockq") or []            # single-interface subset -> the only entry
            return dq[0] if len(dq) == 1 else None
    except Exception:
        return None


def evaluate_group(task):
    """Run OST once for a (pdb_base, seed, sample) and return one row per interface.

    iface_list: [(assembly_id, chain1, chain2), ...]
    All interfaces share the same model CIF and native CIF, so OST runs once for all.
    """
    pdb_base, seed, sample, model_cif, native_cif, conf, iface_list = task
    ost_data = None
    proc_cif = ost_json = None
    want = sorted({c for _, c1, c2 in iface_list for c in (c1, c2)})
    try:
        with tempfile.NamedTemporaryFile(suffix=".cif",  delete=False) as tf: proc_cif = Path(tf.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf: ost_json = Path(tf.name)
        if postprocess_cif(model_cif, proc_cif):
            if run_ost(proc_cif, native_cif, ost_json):   # auto map (+ internal -c retry on crash)
                ost_data = _load_ost_json(ost_json)
            if not _usable(ost_data, iface_list):         # JSON OK but no requested interface ->
                cmap = [f"{c}:{c}" for c in want]         # force an identity map and re-parse
                if _run_ost_once(proc_cif, native_cif, ost_json, chain_map=cmap):
                    forced = _load_ost_json(ost_json)
                    if forced is not None and (_usable(forced, iface_list) or ost_data is None):
                        ost_data = forced
    except Exception as e:
        tqdm.write(f"ERROR {pdb_base} seed={seed} sample={sample}: {e}")
    finally:
        if proc_cif and proc_cif.exists(): proc_cif.unlink()
        if ost_json and ost_json.exists(): ost_json.unlink()

    chain_mapping     = (ost_data.get("chain_mapping") or {}) if ost_data else {}
    chain_mapping_str = json.dumps(chain_mapping) if ost_data else ""

    rows = []
    for assembly_id, chain1, chain2 in iface_list:
        if ost_data:
            dockq_score, dockq_status = extract_dockq(ost_data, chain1, chain2, chain_mapping)
        else:
            dockq_score, dockq_status = float("nan"), "ost_fail"
        if dockq_status != "ok":
            # OST's joint chain-mapper couldn't score this interface (antigen dropped/mis-grouped,
            # wrong copy, or whole-complex fail). Retry scoring just this pair in isolation —
            # score-preserving when the normal path works, recovers a real DockQ when it doesn't.
            iso = _isolated_pair_dockq(model_cif, native_cif, chain1, chain2)
            if iso is not None:
                dockq_score, dockq_status = iso, "ok_isolated"
        scores = compute_confidence_scores(conf, chain_mapping, chain1, chain2)
        rows.append({
            "assembly_id": assembly_id, "chain1": chain1, "chain2": chain2,
            "seed": seed, "sample": sample, **scores,
            "dockq": dockq_score, "dockq_status": dockq_status,
            "chain_mapping": chain_mapping_str,
        })
    return rows


def load_interfaces():
    """Returns pdb_base -> [(assembly_id, chain1, chain2, native_cif)]."""
    pdb_to_ifaces = defaultdict(list)
    with open(TARGETS_CSV) as f:
        for row in csv.DictReader(f):
            pdb_id     = row["pdb_id"]
            chain1     = row["interface_chain_id_1"]
            chain2     = row["interface_chain_id_2"]
            pdb_base   = pdb_id.split("-")[0]
            native_cif = GROUND_TRUTH / f"{pdb_id}.cif"
            if not native_cif.exists():
                print(f"  WARNING: no ground truth for {pdb_id}, skipping")
                continue
            pdb_to_ifaces[pdb_base].append((pdb_id, chain1, chain2, native_cif))
    return pdb_to_ifaces


def load_done(out_csv):
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["assembly_id"], row["chain1"], row["chain2"],
                              row["seed"], int(row["sample"])))
                except (ValueError, KeyError):
                    pass
    return done


def assert_schema_ok(out_csv):
    """Refuse to append onto a CSV whose header != FIELDNAMES. On append, csv.DictWriter writes
    values in FIELDNAMES order with NO new header, so a schema change (e.g. the added
    dockq_status column) silently produces a ragged file and mixes pre/post-classification rows.
    Re-score into a fresh --out_csv instead."""
    if out_csv.exists() and out_csv.stat().st_size > 0:
        with open(out_csv, newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDNAMES:
            raise SystemExit(
                f"ERROR: {out_csv}\n"
                f"  existing header: {header}\n"
                f"  current schema:  {FIELDNAMES}\n"
                f"  Appending would corrupt the file. Pass --out_csv <new path> to re-score "
                f"cleanly with the dockq_status column, or move/delete the old file."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=str(paths.PROTENIX_PREDICTIONS))
    parser.add_argument("--out_csv",  default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_csv  = Path(args.out_csv) if args.out_csv else \
               paths.CSV_DIR / f"dockq_ost_{pred_dir.name}_all_samples.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    assert_schema_ok(out_csv)

    pdb_to_ifaces = load_interfaces()
    n_ifaces = sum(len(v) for v in pdb_to_ifaces.values())
    print(f"Structures: {len(pdb_to_ifaces)}  Interfaces: {n_ifaces}")

    done = load_done(out_csv)
    print(f"Already done: {len(done)}")

    tasks = []
    for pdb_base, iface_entries in pdb_to_ifaces.items():
        pred_pdb_dir = pred_dir / pdb_base
        if not pred_pdb_dir.exists():
            continue

        # All interfaces for this structure share the same native CIF
        native_cif = iface_entries[0][3]
        iface_list = [(aid, c1, c2) for aid, c1, c2, _ in iface_entries]

        seed_dirs = []
        for p in pred_pdb_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name == "baseline_seed":
                seed_dirs.append(("baseline_seed", p))
            elif p.name.startswith("seed_"):
                seed_dirs.append((p.name.split("_")[1], p))
        seed_dirs.sort(key=lambda x: -1 if x[0] == "baseline_seed" else int(x[0]))
        for seed, seed_dir in seed_dirs:
            pred_subdir = seed_dir / "predictions"
            conf_files  = sorted(pred_subdir.glob(f"{pdb_base}_summary_confidence_sample_*.json"))

            for conf_file in conf_files:
                sample = int(conf_file.stem.split("_sample_")[1])

                # Skip only if ALL interfaces for this group are already done
                if all((aid, c1, c2, str(seed), sample) in done for aid, c1, c2 in iface_list):
                    continue

                cif = conf_file.with_name(
                    conf_file.name.replace("_summary_confidence", "").replace(".json", ".cif")
                )
                if not cif.exists():
                    continue

                try:
                    conf = json.loads(conf_file.read_text())
                except Exception:
                    conf = {}

                tasks.append((pdb_base, seed, sample, cif, native_cif, conf, iface_list))

    rows_per_group = len(next(iter(pdb_to_ifaces.values()))) if pdb_to_ifaces else 0
    print(f"OST tasks to run: {len(tasks)}  (each produces {rows_per_group}+ rows)")
    if not tasks:
        print("Nothing to do.")
        return

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    write_lock   = threading.Lock()

    with open(out_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate_group, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="OST run", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                        # A group re-runs if ANY of its interfaces is missing, but
                        # only the missing interfaces should be written — drop rows
                        # already in the CSV so resume never duplicates.
                        rows = [
                            r for r in rows
                            if (r["assembly_id"], r["chain1"], r["chain2"],
                                str(r["seed"]), int(r["sample"])) not in done
                        ]
                        if rows:
                            with write_lock:
                                writer.writerows(rows)
                                csvfile.flush()
                    except Exception as e:
                        tqdm.write(f"ERROR: {e}")
                    finally:
                        pbar.update(1)

    print(f"\nDone. Written to {out_csv}")


if __name__ == "__main__":
    main()
