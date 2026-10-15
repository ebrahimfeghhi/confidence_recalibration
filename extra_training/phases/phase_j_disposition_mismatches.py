"""PHASE J -- disposition the label-audit COORD MISMATCH structures with OST DockQ.

WHY. phase_j_label_audit.py reports five structures whose sequences match SAbDab2 exactly but whose
coordinates do not superpose under ONE rigid transform (8dpl 28.5 A, 8f6x 34.9 A, 9djw 15.8 A,
8d53 0.34 A, 8v8f 0.29 A). Two measurements already exist and NEITHER settles the question:

  - every chain superposing at ~0 A on its own says the ATOMS are right, but says nothing about how
    the chains are arranged relative to each other. Review round 6 made exactly this point: near-zero
    per-chain RMSD with a 28-35 A pooled RMSD means the relative arrangement differs, which is what a
    whole-complex check is for.
  - phase_j_check_contacts.py shows our antigen touches our antibody (2.28-2.94 A, all 8 chains), so
    our label is a real complex -- but "a contact exists" is not "the same contact".

WHAT THIS ADDS. The interface itself, compared between the two files: our label as the model, SAbDab2's
instance cif as the reference. DockQ near 1 means the same interface in a different coordinate frame,
which is a frame difference and harmless. A low DockQ means the two files describe DIFFERENT
arrangements of the same chains, and the label must be inspected or excluded rather than explained.

DockQ COMES FROM OST, NEVER HAND-ROLLED, via this repo's eval/ost_dockq.py -- the same wrapper the
scoring path uses, so a number here means what it means everywhere else. `--rigid-scores` is filtered
out by that wrapper because a crash in it discards otherwise-valid DockQ.

This reads cifs and writes one csv. It folds nothing and changes nothing.
"""
import argparse
import csv
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import ost_dockq                                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", default=[],
                    help="label_audit_*.csv to read; repeat for training and validation")
    ap.add_argument("--labels", action="append", default=[],
                    help="job_cifs dir matching each --audit, in the same order")
    ap.add_argument("--sabdab", action="append", default=[],
                    help="SAbDab2 instance-cif dir matching each --audit, in the same order")
    ap.add_argument("--verdict", default="COORD MISMATCH",
                    help="which audit verdict to disposition")
    ap.add_argument("--out", default=f"{ROOT}/train_val_splits/label_mismatch_disposition.csv")
    a = ap.parse_args()
    if not (len(a.audit) == len(a.labels) == len(a.sabdab)):
        raise SystemExit("--audit, --labels and --sabdab must be repeated the same number of times")

    rows = []
    for audit, labdir, sabdir in zip(a.audit, a.labels, a.sabdab):
        for r in csv.DictReader(open(audit)):
            if r["verdict"] != a.verdict:
                continue
            pdb, inst = r["pdb"], r["instance"]
            mdl = os.path.join(labdir, f"{pdb}.cif")
            ref = os.path.join(sabdir, f"{inst}.cif")
            rec = dict(pdb=pdb, instance=inst, audit_rmsd=r["rmsd"], audit_maxdev=r["maxdev"],
                       atoms_over_1A=r.get("atoms_over_1A", ""),
                       per_chain_rmsd=r.get("per_chain_rmsd", ""))
            if not (os.path.exists(mdl) and os.path.exists(ref)):
                rec["status"] = "missing cif"; rows.append(rec); continue
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "dq.json")
                # OST maps chains itself. We deliberately pass NO chain_map: the whole question is
                # whether the two files' chains correspond, so imposing our own mapping would assume
                # the answer. A mapping OST refuses is itself the finding.
                if not ost_dockq.invoke(mdl, ref, out):
                    rec["status"] = "ost failed"; rows.append(rec); continue
                res = ost_dockq.parse(out)
                try:
                    d = json.load(open(out))
                except Exception:
                    d = {}
            if not res:
                rec["status"] = "no interface scored"; rows.append(rec); continue
            best = max(res.items(), key=lambda kv: kv[1].get("dockq") or -1)
            rec["status"] = "scored"
            rec["n_interfaces"] = len(res)
            rec["best_interface"] = best[0]
            rec["best_dockq"] = f'{best[1].get("dockq"):.4f}' if best[1].get("dockq") is not None else ""
            rec["all_dockq"] = ";".join(
                f'{k}:{(v.get("dockq") if v.get("dockq") is not None else float("nan")):.4f}'
                for k, v in sorted(res.items()))
            rec["chain_mapping"] = json.dumps(d.get("chain_mapping") or {}, separators=(",", ":"))[:200]
            rows.append(rec)

    cols = ["pdb", "instance", "status", "n_interfaces", "best_interface", "best_dockq", "all_dockq",
            "audit_rmsd", "audit_maxdev", "atoms_over_1A", "per_chain_rmsd", "chain_mapping"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"dispositioned {len(rows)} structure(s) -> {a.out}")
    for r in rows:
        print(f'  {r["pdb"]:6s} {r.get("status",""):20s} best_dockq={r.get("best_dockq",""):>7s} '
              f'ifaces={r.get("n_interfaces","")} pooled_rmsd={r["audit_rmsd"]:>8s} '
              f'atoms>1A={r.get("atoms_over_1A","")}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
