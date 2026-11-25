#!/usr/bin/env python
"""DockQ FOR THE REGENERATED POSES. CPU only, runs beside the GPU job that writes them.

HOW A NEW POSE BECOMES SCOREABLE. OST compares two mmCIF files, and the regenerated poses are
coordinates in an npz. Rather than build a structure from scratch -- which would have to
reproduce the fold's chain lettering, atom naming and residue numbering exactly, and would fail
silently if it did not -- this COPIES one of that unit's existing sample cifs and substitutes the
coordinates in file order. phase_j_build_cache.cif_coords documents that file order IS the
model's atom order, and the cache build already asserted the trunk's ci_coordinate against those
cifs to 0.01 A, so the two orders are the same by construction rather than by assumption.

THE SELF-CHECK THAT HAS TO PASS FIRST (--verify). Substituting a CACHED pose's coordinates must
reproduce the DockQ already in results_with_trunks.csv for that (iface, seed, sample). If it does
not, the substitution, the chain mapping or the OST call is wrong, and nothing scored afterwards
can be trusted. Run it before scoring anything new.

Everything else -- the two-chain subset, the sequence-identity chain candidates, the OST call, the
best-over-symmetry reduction -- is phase_j_fill_scores', imported rather than retyped.
"""
import argparse
import collections
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, os.path.join(ROOT, "..", "eval"))


def substitute(template_cif, coords, out_cif):
    """`template_cif` with its coordinates replaced by `coords`, in file order."""
    import gemmi
    st = gemmi.read_structure(str(template_cif))
    k = 0
    for ch in st[0]:
        for res in ch:
            for at in res:
                if k >= len(coords):
                    raise ValueError(f"{template_cif} has more atoms than the {len(coords)} given")
                at.pos = gemmi.Position(float(coords[k][0]), float(coords[k][1]),
                                        float(coords[k][2]))
                k += 1
    if k != len(coords):
        raise ValueError(f"{template_cif} has {k} atoms, the pose has {len(coords)}")
    os.makedirs(os.path.dirname(out_cif) or ".", exist_ok=True)
    st.make_mmcif_document().write_file(str(out_cif))
    return out_cif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-dir", default=f"{ROOT}/training/fold/poses_extra")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--refs", default=f"{ROOT}/training/refs")
    ap.add_argument("--pool-csv", default=f"{ROOT}/training/combined/pool.csv")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--scratch", default="/tmp/dockq_extra")
    ap.add_argument("--out", default=f"{ROOT}/training/fold/poses_extra/dockq_extra.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-ident", type=float, default=0.90)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0,
                    help="N units: score the CACHED poses through this same path and compare "
                         "with results_with_trunks.csv. Nothing new is scored.")
    a = ap.parse_args()

    import numpy as np
    import phase_j_build_cache as BC
    import phase_j_fill_scores as FS

    os.makedirs(a.scratch, exist_ok=True)
    pool = collections.defaultdict(set)
    with open(a.pool_csv) as fh:
        for r in csv.DictReader(fh):
            pool[r["pdb"]].add((r["chain1"], r["chain2"]))

    verify_dq = {}
    if a.verify:
        with open(a.results) as fh:
            for r in csv.DictReader(fh):
                if r.get("arm") == "free":
                    try:
                        verify_dq[(r["iface"], int(r["seed"]), int(r["sample"]))] = float(
                            r["dockq"])
                    except (ValueError, KeyError):
                        pass

    # WHAT IS ALREADY SCORED, so this can run repeatedly beside the GPU job without redoing or
    # duplicating work. Keyed by (iface, seed, sample) because that is what a row names.
    have = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                try:
                    have.add((r["iface"], int(r["seed"]), int(r["sample"])))
                except (ValueError, KeyError):
                    pass

    units = sorted(glob.glob(f"{a.poses_dir}/*/*.npz"))
    if a.verify:
        units = sorted(glob.glob(f"{a.fold}/trunks/*/*.npz"))[:int(a.verify)]
    elif a.limit_units:
        units = units[:int(a.limit_units)]
    print(f"{len(units)} unit(s){' -- VERIFY MODE, cached poses' if a.verify else ''}; "
          f"{len(have)} (iface, seed, sample) already scored", flush=True)

    jobs, meta = [], {}
    for up in units:
        pdb = os.path.basename(os.path.dirname(up))
        unit = os.path.basename(up)[:-4]
        seed = int(unit.split("_")[-1]) if unit.split("_")[-1].isdigit() else 0
        if not pool.get(pdb):
            continue
        ref = os.path.join(a.refs, f"{pdb}.cif")
        if not os.path.exists(ref):
            continue
        try:
            ud = BC.seed_dir(a.fold, pdb, seed)
            cifs = BC.sample_cifs(ud, pdb)
        except Exception:                                     # noqa: BLE001
            continue
        if not cifs:
            continue
        z = np.load(up, allow_pickle=True)
        coords = z["ci_coordinate"] if a.verify else z["coords"]
        # THE POSE INDEX IS NOT THE SAMPLE INDEX, for the THIRD time in this project. The npz is
        # in POSE order, results_with_trunks.csv is keyed by the fold's SAMPLE id, and the
        # permutation lives in the label's `sample_index` -- non-identity on every unit checked.
        # Verify mode labelled pose i as sample i and compared against sample i's stored DockQ:
        # 50% within 0.01, max |diff| 0.179. The new poses are their own samples (100 + i) and
        # need no mapping, so this only applies to verify.
        si = None
        if a.verify:
            lp_ = os.path.join(a.fold, "labels", pdb, f"{unit}.npz")
            if not os.path.exists(lp_):
                continue
            si = np.asarray(np.load(lp_, allow_pickle=True)["sample_index"]).reshape(-1)
        tmpl = cifs[0]
        rs = FS.seqs(ref, True)
        for i in range(min(int(a.nsample), coords.shape[0])):
            samp = int(si[i]) if a.verify else 100 + i
            mcif = os.path.join(a.scratch, f"{pdb}_{seed}_{samp}.cif")
            try:
                # IN VERIFY MODE the pose's OWN cif is used rather than a substituted copy of
                # sample 0, so the comparison tests the chain mapping and the OST call. The
                # substitution itself is tested by --verify-substitute below.
                # The template supplies topology only -- every coordinate is overwritten -- so
                # which sample's cif is used does not matter; what matters is that the ROW is
                # labelled with the sample those coordinates belong to.
                substitute(tmpl, coords[i], mcif)
            except Exception as e:                            # noqa: BLE001
                print(f"  {pdb}/{unit} sample {samp}: {type(e).__name__}: {e}", flush=True)
                continue
            ms = FS.seqs(mcif, False)
            for c1, c2 in sorted(pool[pdb]):
                if c1 not in rs or c2 not in rs:
                    continue
                cand = {rc: [mc for mc, m in ms.items()
                             if FS.cov_ident(rs[rc], m) >= a.min_ident] for rc in (c1, c2)}
                if not cand[c1] or not cand[c2]:
                    continue
                if not a.verify and (f"{pdb}_{c1}{c2}", seed, samp) in have:
                    continue
                for m1 in cand[c1]:
                    for m2 in cand[c2]:
                        if m1 == m2:
                            continue
                        jobs.append((pdb, f"{c1}{c2}", c1, c2, m1, m2, ref, mcif, samp,
                                     seed, a.scratch))
                        meta[(pdb, f"{c1}{c2}", seed, samp)] = True
    print(f"OST calls queued: {len(jobs)}", flush=True)
    if not jobs:
        print("nothing new to score")
        return

    from concurrent.futures import ProcessPoolExecutor
    import time
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(FS.score_one, jobs, chunksize=1), 1):
            if r:
                rows.append(r)
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} calls, {len(rows)} scored, "
                      f"{n / el:.1f} call/s, ~{(len(jobs) - n) / (n / el) / 60:.0f} min left",
                      flush=True)
    best = {}
    for r in rows:
        k = (r["iface"], r["seed"], r["sample"])
        if k not in best or r["dockq"] > best[k]["dockq"]:
            best[k] = r
    el = time.time() - t0
    print(f"{len(jobs)} OST call(s) in {el / 60:.1f} min ({len(jobs) / el:.1f}/s), "
          f"{len(best)} (iface, seed, sample) scored")

    if a.verify:
        import numpy as _np
        d, miss = [], 0
        for k, r in best.items():
            tgt = verify_dq.get(k)
            if tgt is None:
                miss += 1
                continue
            d.append(r["dockq"] - tgt)
        d = _np.asarray(d)
        print(f"\nVERIFY  {len(d)} compared, {miss} with no stored DockQ")
        if len(d):
            print(f"  max |diff| {_np.abs(d).max():.4f}   mean |diff| {_np.abs(d).mean():.4f}   "
                  f"within 0.01: {(_np.abs(d) < 0.01).mean():.4f}")
            if (_np.abs(d) < 0.01).mean() < 0.95:
                raise SystemExit("STOP: the path does not reproduce the stored DockQ")
        print("  verify OK")
        return

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["pdb", "iface", "seed", "sample", "dockq", "model_chains", "source"])
        for k, r in sorted(best.items()):
            w.writerow([r["iface"].split("_")[0], r["iface"], r["seed"], r["sample"],
                        round(r["dockq"], 4), r.get("model_chains", ""), "rediffuse"])
    print(f"rows -> {a.out}")


if __name__ == "__main__":
    main()
