#!/usr/bin/env python
"""Build chain_roles.csv and pool.csv for the Wave 1 training structures.

WHY NOT DERIVED FROM THE JOB JSONS. The obvious shortcut is that Protenix labels predicted chains
A, B, C... in job-json order, so an interface could be named positionally. phase_g does record
those MODEL letters -- but it puts them in chain_roles.csv and writes pool.csv from the REFERENCE
labels instead:

    ab = [ref_of[i] for i, (r, _) in enumerate(chains) if r in ("H", "L")]

where ref_of comes from auth chain -> label_asym_id read off the deposited structure. That is why
the existing training pool names 10or's interfaces B-S and F-S rather than its positional A-D. The
scorer joins on those reference labels, so a positionally-named pool would not match.

SO THIS REUSES phase_g's OWN FUNCTIONS rather than reimplementing the mapping: _chains for the
role/sequence order, ref_auth_to_label and relabel_unique for the reference side. Reimplementing
would mean two chain-mapping rules in the repo that could drift, and the mapping is exactly the
part that is easy to get subtly wrong -- 9n6e's roles name chains E/F/G while its local cif holds
A,B,C,D,H,I,L.

THE ORDER IS ASSERTED, NOT ASSUMED. The job jsons were written from phase_j0's chain_seqs(), which
does NOT apply _chains' min_len=5 filter. If any structure carries a chain shorter than that, the
json would hold a sequence _chains drops and every later letter would shift by one -- naming the
wrong chain in every interface of that structure. Each structure's json sequences are compared
against _chains element by element and a mismatch is a hard stop.

Interfaces are every (antibody, antigen) pair, alphabetically ordered, excluding the intra-antibody
H-L pair -- the convention the pxm2025 pool uses and which phase_g follows.

Usage:
  python -I run_logging/build_wave1_pool.py --queue training/full_train/wave1_queue.json --dry-run
  python -I run_logging/build_wave1_pool.py --queue training/full_train/wave1_queue.json \
      --out-pool training/full_train/pool_wave1.csv \
      --out-roles training/fold/chain_roles_wave1.csv
"""
import argparse, collections, csv, json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "phases"))

# phase_g parses argv at import, so load only its module-level defs
_gsrc = open(os.path.join(ROOT, "phases", "phase_g_build_val_jobs.py")).read()
_G = {"__file__": os.path.join(ROOT, "phases", "phase_g_build_val_jobs.py"), "__name__": "nm"}
exec(compile(_gsrc.split("def main(")[0], "phase_g_head", "exec"), _G)
_chains, ref_auth_to_label, relabel_unique = _G["_chains"], _G["ref_auth_to_label"], _G["relabel_unique"]
A = _G["A"]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="training/full_train/wave1_queue.json")
ap.add_argument("--splits", default="SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
ap.add_argument("--src-cifs", default="SAbDab2_data/training_cifs")
ap.add_argument("--jobs-dir", default="training/jobs")
ap.add_argument("--refs-dir", default="training/fold/refs")
ap.add_argument("--out-pool", default="training/full_train/pool_wave1.csv")
ap.add_argument("--out-roles", default="training/fold/chain_roles_wave1.csv")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

want = {x["pdb"]: x["instance"] for x in json.load(open(os.path.join(ROOT, a.queue)))["queue"]}

sp = pd.read_csv(os.path.join(ROOT, a.splits), low_memory=False)
byinst = {str(r.INSTANCE): r for r in sp.itertuples(index=False)}

# role -> auth chain name, exactly as phase_g builds it
auth_names = {}
for r in sp.itertuples(index=False):
    d = {}
    if isinstance(r.Hchain, str) and r.Hchain.strip():
        d["H"] = r.Hchain.strip()
    if isinstance(r.Lchain, str) and r.Lchain.strip():
        d["L"] = r.Lchain.strip()
    if isinstance(r.agchains, str):
        agt = str(r.agtypes).split("/") if isinstance(getattr(r, "agtypes", None), str) else None
        parts = r.agchains.split("/")
        if agt is not None and len(parts) != len(agt):
            sys.exit(f"STOP: {r.PDB_ID} lists {len(agt)} antigen types but {len(parts)} chain "
                     f"names; ag<i> is positional in both.")
        for i, c in enumerate(parts):
            if c.strip():
                d[f"ag{i}"] = c.strip()
    auth_names[str(r.INSTANCE)] = d

pool, roles, problems, relabelled = [], [], collections.defaultdict(list), []
for pdb, inst in sorted(want.items()):
    row = byinst.get(inst)
    if row is None:
        problems["no_split_row"].append((pdb, inst)); continue
    chains = _chains(row)
    letters = [chr(ord("A") + i) for i in range(len(chains))]

    # ASSERT THE JOB JSON AGREES, element by element
    jp = os.path.join(ROOT, a.jobs_dir, f"{pdb}-final-updated.json")
    if not os.path.exists(jp):
        problems["no_job_json"].append((pdb, jp)); continue
    jseqs = [list(s.values())[0]["sequence"] for s in json.load(open(jp))[0]["sequences"]]
    cseqs = [q for _r, q in chains]
    if jseqs != cseqs:
        problems["json_order_mismatch"].append(
            (pdb, f"json {len(jseqs)} chain(s) {[len(x) for x in jseqs]} vs "
                  f"_chains {len(cseqs)} {[len(x) for x in cseqs]}"))
        continue

    src = os.path.join(ROOT, a.src_cifs, f"{inst}.cif")
    if not os.path.exists(src):
        problems["no_source_cif"].append((pdb, src)); continue
    a2l, shared = ref_auth_to_label(src)
    if shared:
        os.makedirs(os.path.join(ROOT, a.refs_dir), exist_ok=True)
        src = relabel_unique(src, os.path.join(ROOT, a.refs_dir, f"{pdb}.relabelled.cif"))
        a2l, shared = ref_auth_to_label(src)
        if shared:
            problems["shared_labels_after_relabel"].append((pdb, str(shared))); continue
        relabelled.append(pdb)

    auth = auth_names.get(inst, {})
    ref_of = {}
    for i, (r, _q) in enumerate(chains):
        an = auth.get(r)
        ref_of[i] = a2l.get(an) if an else None
        roles.append(dict(pdb=pdb, model_chain=letters[i], ref_chain=ref_of[i] or "",
                          auth_chain=an or "", role=r))
    miss = [(letters[i], r) for i, (r, _q) in enumerate(chains) if not ref_of[i]]
    if miss:
        problems["unresolved_ref_chain"].append((pdb, str(miss))); continue

    ab = [ref_of[i] for i, (r, _q) in enumerate(chains) if r in ("H", "L")]
    ag = [ref_of[i] for i, (r, _q) in enumerate(chains) if r.startswith("ag")]
    for abc in ab:
        for agc in ag:
            c1, c2 = sorted((abc, agc))
            pool.append(dict(pdb=pdb, chain1=c1, chain2=c2))

print(f"structures requested: {len(want)}")
print(f"  pool rows: {len(pool)} over {len({r['pdb'] for r in pool})} structure(s)")
print(f"  role rows: {len(roles)}")
print(f"  relabelled for duplicate reference labels: {len(relabelled)}")
bad = sum(len(v) for v in problems.values())
for k, v in sorted(problems.items()):
    print(f"  {k}: {len(v)}")
    for x in v[:4]:
        print(f"      {x}")
    if len(v) > 4:
        print(f"      ... and {len(v)-4} more")
if bad:
    print(f"\n{bad} structure(s) could not be resolved -- NOTHING WRITTEN")
    sys.exit(1)
if a.dry_run:
    print("\nDRY RUN -- nothing written")
    sys.exit(0)
for path, rows, hdr in ((a.out_pool, pool, ["pdb", "chain1", "chain2"]),
                        (a.out_roles, roles,
                         ["pdb", "model_chain", "ref_chain", "auth_chain", "role"])):
    p = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, hdr); w.writeheader(); w.writerows(rows)
    os.replace(p + ".tmp", p)
    print(f"wrote {path}: {len(rows)} row(s)")
