#!/usr/bin/env python
"""Step 1: does the DEPOSITED ENTRY work as the DockQ reference in place of the instance cif?

WHY THIS HAS TO BE MEASURED. 17 of the 30 new validation structures have no SAbDab2 instance cif,
and that one file supplies both the pool's auth -> label_asym_id mapping and the reference itself
(phase_g_build_val_jobs stages it as validation/refs/<pdb>.cif, which paths.py points the scorer
at). The proposal is to use the deposited entry for both. The entry carries extra copies and
chains the instance cif does not, so OST's automatic chain mapper could land somewhere else even
where DockQ happens to agree.

The old 53-structure benchmark settles it for free: it has folded poses on disk AND instance-cif
refs, so the same pose can be scored both ways and compared. CPU only, no refolding, and it
writes nothing into the benchmark.

THREE THINGS REVIEW REQUIRED, AND WHY EACH:

  1. FRESH OUTPUT JSON PER REFERENCE. ost_dockq.run() returns early when `out` already holds a
     usable json, so reusing a path would compare a result against itself and report perfect
     agreement. This calls invoke() -- documented as "No skip-if-exists" -- and gives every
     (structure, seed, pose, reference) its own path under a scratch directory.

  2. MORE THAN DockQ. For each pose we compare DockQ, the reference-to-model chain mapping that
     OST chose (`model_chains`), and the model chain pair ipTM is read from. A mapping change is
     a real difference even at identical DockQ, because every later read of that interface
     inherits it.

  3. ALL 53 STRUCTURES, a few poses each, and the SAME BIOLOGICAL INTERFACE on both sides even
     though the reference labels differ. The interface is identified by ROLE (H/L vs ag<i>), the
     labels are resolved per reference, and multi-copy structures are reported separately.

Usage:
  python -I run_logging/compare_entry_as_reference.py --poses 3 --workers 12
"""
import argparse, collections, csv, glob, json, os, sys
import concurrent.futures as cf
import pandas as pd, gemmi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "eval"))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import ost_dockq                                                           # noqa: E402

_gsrc = open(os.path.join(ROOT, "phases", "phase_g_build_val_jobs.py")).read()
_G = {"__file__": os.path.join(ROOT, "phases", "phase_g_build_val_jobs.py"), "__name__": "nm"}
exec(compile(_gsrc.split("def main(")[0], "g", "exec"), _G)
_chains = _G["_chains"]

ap = argparse.ArgumentParser()
ap.add_argument("--selection", default="validation/full_val/val53_structures.csv")
ap.add_argument("--splits", default="SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
ap.add_argument("--entries", default="SAbDab2_data/rcsb_validation_cifs")
ap.add_argument("--staged-refs", default="validation/refs")
ap.add_argument("--poses", type=int, default=3, help="poses per structure")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--scratch", default=os.environ.get("CMP_SCRATCH", "/tmp/entry_ref_cmp"))
ap.add_argument("--tol", type=float, default=0.01, help="|dDockQ| treated as negligible")
ap.add_argument("--out", default="validation/full_val/entry_as_reference_report.csv")
a = ap.parse_args()

sp = pd.read_csv(os.path.join(ROOT, a.splits), low_memory=False)
byinst = {str(r.INSTANCE): r for r in sp.itertuples(index=False)}


def auth_of(row):
    d = {}
    if isinstance(row.Hchain, str) and row.Hchain.strip():
        d["H"] = row.Hchain.strip()
    if isinstance(row.Lchain, str) and row.Lchain.strip():
        d["L"] = row.Lchain.strip()
    if isinstance(row.agchains, str):
        for i, c in enumerate(row.agchains.split("/")):
            if c.strip():
                d[f"ag{i}"] = c.strip()
    return d


def label_of_role(cif, row):
    """role -> label_asym_id in `cif`, disambiguating a multi-subchain auth chain by role sequence.

    This is the resolver the plan proposes, applied to whichever reference is being measured. A role
    that does not land on exactly one label subchain returns None, and the structure is reported
    rather than guessed at -- 8z7j fails here because SAbDab names chains S1/S2 the entry lacks.
    """
    st = gemmi.read_structure(cif)
    st.setup_entities()
    subs = {}
    for ch in st[0]:
        for sub in ch.subchains():
            rs = [x for x in sub]
            if len(rs) >= 5:
                subs.setdefault(ch.name, []).append(
                    (sub.subchain_id(), gemmi.one_letter_code([x.name for x in rs])))
    auth, out = auth_of(row), {}
    for role, q in _chains(row):
        cands = subs.get(auth.get(role), [])
        if len(cands) == 1:
            out[role] = cands[0][0]
        elif len(cands) > 1:
            hit = [sid for sid, s in cands if s and (q[:50] in s or s[:50] in q)]
            out[role] = hit[0] if len(hit) == 1 else None
        else:
            out[role] = None
    return out


def n_copies(cif, row):
    """how many chains carry each role's sequence -- a structure with >1 holds repeated copies"""
    st = gemmi.read_structure(cif)
    st.setup_entities()
    seqs = [gemmi.one_letter_code([r.name for r in ch]) for ch in st[0]]
    m = 1
    for _role, q in _chains(row):
        m = max(m, sum(1 for s in seqs if s and (q[:60] in s or s[:60] in q)))
    return m


def jobs(sel):
    """one work item per (structure, seed, pose), carrying both references and both label maps"""
    out = []
    for r in sel:
        pdb, inst = r["pdb"], r["instance"]
        row = byinst.get(inst)
        if row is None:
            continue
        staged = os.path.join(ROOT, a.staged_refs, f"{pdb}.cif")
        entry = os.path.join(ROOT, a.entries, f"{pdb}.cif")
        if not (os.path.exists(staged) and os.path.exists(entry)):
            continue
        try:
            lab_s, lab_e = label_of_role(staged, row), label_of_role(entry, row)
            cp = n_copies(entry, row)
        except Exception as e:
            print(f"  {pdb}: could not resolve labels: {type(e).__name__}: {e}", flush=True)
            continue
        ifaces = [(rl_ab, rl_ag) for rl_ab, _ in _chains(row) if rl_ab in ("H", "L")
                  for rl_ag, _ in _chains(row) if rl_ag.startswith("ag")]
        if not ifaces:
            continue
        poses = sorted(glob.glob(
            f"{ROOT}/validation/fold/poses/{pdb}/*/predictions/{pdb}_sample_*.cif"))[:a.poses]
        for p in poses:
            d = os.path.dirname(p)
            seed = os.path.basename(os.path.dirname(d)).split("-")[0]
            conf = os.path.join(d, os.path.basename(p).replace("_sample_",
                                                               "_summary_confidence_sample_")
                                .replace(".cif", ".json"))
            out.append(dict(pdb=pdb, seed=seed, pose=p, conf=conf if os.path.exists(conf) else None,
                            staged=staged, entry=entry, lab_s=lab_s, lab_e=lab_e,
                            ifaces=ifaces, copies=cp))
    return out


def one(j):
    """score this pose against BOTH references, in two fresh json files, and compare per interface"""
    tag = f"{j['pdb']}_{j['seed']}_{os.path.basename(j['pose'])[:-4]}"
    d = os.path.join(a.scratch, j["pdb"])
    os.makedirs(d, exist_ok=True)
    res = {}
    for which, ref in (("staged", j["staged"]), ("entry", j["entry"])):
        out = os.path.join(d, f"{tag}.{which}.json")        # FRESH PATH PER REFERENCE
        if os.path.exists(out):
            os.remove(out)
        ok = ost_dockq.invoke(j["pose"], ref, out)          # invoke: no skip-if-exists
        if not ok:
            ids = ost_dockq._identity_chain_map(ref)
            ok = bool(ids) and ost_dockq.invoke(j["pose"], ref, out, chain_map=ids)
        res[which] = ost_dockq.parse(out, conf_json=j["conf"]) if ok else None

    rows = []
    for rl_ab, rl_ag in j["ifaces"]:
        c_s = (j["lab_s"].get(rl_ab), j["lab_s"].get(rl_ag))
        c_e = (j["lab_e"].get(rl_ab), j["lab_e"].get(rl_ag))
        rec = dict(pdb=j["pdb"], seed=j["seed"], pose=os.path.basename(j["pose"]),
                   role_pair=f"{rl_ab}-{rl_ag}", copies=j["copies"],
                   staged_chains=",".join(str(x) for x in c_s),
                   entry_chains=",".join(str(x) for x in c_e))
        for which, cc in (("staged", c_s), ("entry", c_e)):
            v = None
            if res.get(which) and all(cc):
                try:
                    v = ost_dockq.interface(res[which], *cc)
                except ValueError as e:
                    rec[f"{which}_note"] = str(e)[:60]
            rec[f"{which}_dockq"] = None if not v else v.get("dockq")
            rec[f"{which}_map"] = "" if not v else ",".join(v.get("model_chains") or ())
            rec[f"{which}_iptm"] = None if not v else v.get("pair_iptm")
        rows.append(rec)
    return rows


sel = list(csv.DictReader(open(os.path.join(ROOT, a.selection))))
work = jobs(sel)
print(f"{len(sel)} structure(s); {len(work)} pose(s) to score against 2 references each "
      f"= {2*len(work)} OST call(s)")
print(f"scratch: {a.scratch}  (nothing is written into the benchmark)")
mc = sorted({j['pdb'] for j in work if j['copies'] > 1})
print(f"structures holding repeated copies of a role's sequence: {len(mc)} -> {mc}")
print()

out = []
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    futs = {ex.submit(one, j): j for j in work}
    done = 0
    for f in cf.as_completed(futs):
        try:
            out += f.result()
        except Exception as e:
            j = futs[f]
            print(f"  {j['pdb']} {j['seed']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(work)} pose(s)", flush=True)

os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
if out:
    keys = sorted({k for r in out for k in r})
    with open(os.path.join(ROOT, a.out), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)

# A ROW WHERE THE ENTRY SIDE HAS NO LABEL IS NOT A DISAGREEMENT. The resolver returns None when
# SAbDab names a chain the deposited entry does not carry -- 9nr6's instance is pdb_00009nr6_F1_F2
# and the entry has no F1/F2, so H and L resolve to nothing. That is the fail-closed behaviour the
# plan relies on: such a structure is excluded, not guessed at. Counted separately so it cannot be
# mistaken for the entry disagreeing about DockQ, and so its size is visible.
unresolved = [r for r in out if "None" in r["entry_chains"]]
out = [r for r in out if "None" not in r["entry_chains"]]
both = [r for r in out if r["staged_dockq"] is not None and r["entry_dockq"] is not None]
only_s = [r for r in out if r["staged_dockq"] is not None and r["entry_dockq"] is None]
only_e = [r for r in out if r["staged_dockq"] is None and r["entry_dockq"] is not None]
neither = [r for r in out if r["staged_dockq"] is None and r["entry_dockq"] is None]

print()
print("=== RESULT ===")
print(f"  entry-side label unresolved (excluded by the fail-closed rule): {len(unresolved)} row(s) "
      f"over {len({r['pdb'] for r in unresolved})} structure(s) -> "
      f"{sorted({r['pdb'] for r in unresolved})}")
print(f"  interface-pose comparisons: {len(out)}  over {len({r['pdb'] for r in out})} structure(s)")
print(f"    scored both ways:        {len(both)}")
print(f"    staged only:             {len(only_s)}")
print(f"    entry only:              {len(only_e)}")
print(f"    neither:                 {len(neither)}")
if both:
    d = sorted(abs(r["staged_dockq"] - r["entry_dockq"]) for r in both)
    over = [r for r in both if abs(r["staged_dockq"] - r["entry_dockq"]) > a.tol]
    print(f"  |dDockQ|: max {d[-1]:.4f}  median {d[len(d)//2]:.4f}  "
          f"over tol {a.tol}: {len(over)} of {len(both)}")
    mm = [r for r in both if r["staged_map"] != r["entry_map"]]
    print(f"  ref->model chain mapping differs: {len(mm)} of {len(both)}")
    ip = [r for r in both if r["staged_iptm"] is not None and r["entry_iptm"] is not None
          and abs(r["staged_iptm"] - r["entry_iptm"]) > 1e-9]
    print(f"  ipTM read from a different model pair: {len(ip)} of {len(both)}")
    for r in (over + mm + ip)[:15]:
        print(f"     {r['pdb']} {r['seed']} {r['pose']} {r['role_pair']} copies={r['copies']}: "
              f"staged {r['staged_dockq']} [{r['staged_chains']} -> {r['staged_map']}] "
              f"ipTM {r['staged_iptm']} | entry {r['entry_dockq']} "
              f"[{r['entry_chains']} -> {r['entry_map']}] ipTM {r['entry_iptm']}")
    print()
    print("  VERDICT: " + ("PASS -- DockQ within tolerance and mappings identical"
                           if not over and not mm and not ip else
                           "FAIL -- differences above; report before changing anything"))
print(f"  per-comparison rows -> {a.out}")
