#!/usr/bin/env python
"""Decode poses from saved steering checkpoints, score them against the crystal, tabulate and plot.

    LAYERNORM_TYPE=openfold decode_and_score.py --dump <stem> --template-cif <cif> --ref <crystal cif> \
        --pdb 8w5q --ifaces AC,BC,BD --seed-tag 300019 --n 20 --decode-seed 1234 --out <dir> \
        --cond baseline=<ckpt0000.npz> --cond x1=<ckpt0200.npz> ...

Each condition: load z/s (float32) from the checkpoint, sample N poses with the normal 200-step sampler
(SAME noise seed for every condition, so conditions are paired), compute ipTM, write CIFs, score every
interface with scorer.py (pxm) against the crystal. Outputs in --out: <cond>/ (CIFs, coordinate.npz,
confidence.json), scores.csv (one row per pose x interface), summary.csv, dockq_scatter.png.
The baseline condition must equal the dump's own z/s exactly; that is asserted.
"""
import argparse, json, os, subprocess, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_reps as S

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--template-cif", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--pdb", required=True)
ap.add_argument("--ifaces", required=True); ap.add_argument("--seed-tag", required=True)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--decode-seed", type=int, default=1234)
ap.add_argument("--cond", action="append", required=True, help="name=path/to/ckpt.npz; first is the baseline")
ap.add_argument("--out", required=True)
ap.add_argument("--target-cif", default="", help="if given, also report whole-complex RMSD of each pose to this target, and pairwise RMSD among the poses")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); dev = "cuda"
PY = "/home/ebrahim/Protenix/.venv/bin/python"; SCORER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorer.py")

model, configs = S.build_model("/home/ebrahim/checkpoint", "protenix-v2", dev)
feat, z0, s0, si, coords = S.load_dump(A.dump, dev)
n_cycle = int(np.load(A.dump + "_reps.npz")["n_cycle"])
conds = [c.split("=", 1) for c in A.cond]
rows = []
rmsd_rows = []
for name, path in conds:
    C = np.load(path); z = torch.from_numpy(C["z_trunk"]).float().to(dev); s = torch.from_numpy(C["s_trunk"]).float().to(dev)
    if name == conds[0][0]:
        assert torch.equal(z, z0) and torch.equal(s, s0), "baseline checkpoint differs from the dump's own z/s"
        print(f"[{name}] baseline == dump z/s exactly")
    x = S.decode(model, configs, feat, si, z, s, A.n, A.decode_seed, dev)
    summ, _ = S.confidence(model, configs, feat, si, s, z, x, n_cycle)
    d = f"{A.out}/{name}"; os.makedirs(d, exist_ok=True)
    xa = x.cpu().numpy(); np.savez(f"{d}/coordinate.npz", coordinate=xa)
    for k in range(A.n):     # scorer.py reads <pdb>_sample_<k>.cif
        pass
    S.write_cifs(A.template_cif, xa, d, "tmp")
    for k in range(A.n): os.replace(f"{d}/tmp_{k:02d}.cif", f"{d}/{A.pdb}_sample_{k}.cif")
    if A.target_cif:
        tg = S.cif_coords(A.target_cif)
        rt = [S.kabsch_rmsd(xa[k], tg) for k in range(A.n)]
        pw = [S.kabsch_rmsd(xa[i], xa[j]) for i in range(A.n) for j in range(i + 1, A.n)]
        rmsd_rows.append(dict(cond=name, rmsd_to_target_mean=float(np.mean(rt)), rmsd_to_target_max=float(np.max(rt)),
                              pairwise_rmsd_mean=float(np.mean(pw)), pairwise_rmsd_max=float(np.max(pw))))
        json.dump(rt, open(f"{d}/rmsd_to_target.json", "w"))
    json.dump([dict(sample=k, iptm=float(summ[k]["iptm"]), ranking_score=float(summ[k]["ranking_score"])) for k in range(A.n)],
              open(f"{d}/confidence.json", "w"), indent=1)
    print(f"[{name}] decoded {A.n} poses from {path}; mean ipTM {np.mean([float(m['iptm']) for m in summ]):.3f}", flush=True)
    for i in A.ifaces.split(","):
        r = subprocess.run([PY, SCORER, "--dir", d, "--iface", f"{A.pdb}_{i}", "--seed", A.seed_tag, "--arm", name,
                            "--donor", "-1", "--ref", A.ref, "--scorer", "pxm"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        got = [l for l in r.stdout.splitlines() if l.strip()]
        assert len(got) == A.n, f"{name} {i}: scorer returned {len(got)} rows, expected {A.n}"
        rows += got
open(f"{A.out}/scores.csv", "w").write("iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n" + "\n".join(rows) + "\n")
if rmsd_rows:
    import pandas as pd
    pd.DataFrame(rmsd_rows).to_csv(f"{A.out}/rmsd_summary.csv", index=False); print(pd.DataFrame(rmsd_rows).round(2).to_string(index=False))
print("wrote", f"{A.out}/scores.csv", len(rows), "rows", flush=True)
