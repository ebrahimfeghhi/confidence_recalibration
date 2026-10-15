"""Falsifiability self-test for phase_j_label_audit.py.

WHY THIS FILE EXISTS. Three separate checks in this project passed because they COULD NOT FAIL: an
identity check that walked difflib's matching blocks (where substitutions are invisible by
construction), a coverage count that tallied RMSD rows instead of `ok` verdicts, and a structure
that passed with both antibody chains unpaired. A clean audit result is evidence only if the audit
is known to report the defects it claims to look for.

So this script DAMAGES copies of real labels in known ways and asserts the audit notices. It builds
its own fixtures from the repo's own files -- nothing is pre-staged in a scratch directory -- so the
reviewer can run it from a clean checkout:

    python phases/phase_j_label_audit_selftest.py

Every case is run against BOTH a one-to-one structure and a FUSED one, because the fused path is
newer: SAbDab2 keeps an scFv as one chain while our label splits it in two, and the audit handles
that by concatenating our two chains and comparing the result. A test that only ever exercised
one-to-one chains would leave the concatenation unfalsified -- which is exactly the mistake above.

| case         | what is damaged                                  | the audit must report |
|--------------|--------------------------------------------------|-----------------------|
| clean        | nothing (positive control)                       | ok                    |
| substitution | one residue renamed, coordinates untouched       | SUBSTITUTION          |
| swap         | the coordinates of two chains exchanged          | anything but ok       |
| truncate     | 30 residues deleted from inside the variable domain | anything but ok     |
| shift        | one chain translated by 5 A                      | COORD MISMATCH        |

A case whose damage the audit does NOT notice is a FAILURE of the audit, and this script exits
non-zero so it cannot be overlooked in a log.
"""
import csv, os, shutil, subprocess, sys, tempfile

import gemmi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
AUDIT = f"{ROOT}/phases/phase_j_label_audit.py"

# ONE-TO-ONE and FUSED, named explicitly rather than discovered, so the test is deterministic and a
# reviewer can see which path each row exercises. 32nz: our label splits the scFv (A=122, B=115)
# where SAbDab2 fuses it (A=238), so it is the concatenation path.
TARGETS = [("10gh", "one-to-one"), ("32nz", "fused (scFv split in our label)")]


def instance_of(pdb):
    with open(f"{ROOT}/train_val_splits/cache_train_structures.csv") as fh:
        for r in csv.DictReader(fh):
            if r["pdb"] == pdb:
                return r["instance"]
    raise SystemExit(f"{pdb} is not in cache_train_structures.csv")


def damage(src, dst, case):
    """Write a copy of `src` to `dst`, damaged according to `case`."""
    st = gemmi.read_structure(src)
    st.setup_entities()
    chains = [ch for ch in st[0] if sum(1 for r in ch if r.name != "HOH") > 30]
    if case == "clean":
        pass
    elif case == "substitution":
        # rename one residue. Coordinates are untouched, so ONLY an identity comparison can see it.
        for ch in chains:
            res = [r for r in ch][len(list(ch)) // 3]
            res.name = "TRP" if res.name != "TRP" else "ARG"
            break
    elif case == "swap":
        # exchange the coordinates of two chains: every atom is still present and every sequence
        # still matches, so only a POSITION check catches it. This is the copy-mixing failure mode.
        if len(chains) < 2:
            return False
        a, b = chains[0], chains[1]
        pa = [[at.pos for at in r] for r in a]
        pb = [[at.pos for at in r] for r in b]
        for r, ps in zip(a, pb[: len(list(a))]):
            for at, p in zip(r, ps):
                at.pos = p
        for r, ps in zip(b, pa[: len(list(b))]):
            for at, p in zip(r, ps):
                at.pos = p
    elif case == "truncate":
        # Delete 30 residues from INSIDE the variable domain: a coverage defect, not an identity one.
        #
        # IT MATTERS THAT THE DELETION LANDS WHERE THE REFERENCE CAN SEE IT. Deleting the END of the
        # chain instead was tried and the audit reported `ok` -- correctly. SAbDab2 crops to the
        # variable region, so our label's chain is the longer one: 10gh label A is 238 residues
        # (VH + CH1) against SAbDab2's 119. The last quarter of A is CH1, which the reference does
        # not contain, and an audit cannot verify residues its reference lacks. That is the coverage
        # limit this file's docstring states, not a miss -- but it is also the reason a truncation
        # test must be aimed deliberately, or it proves nothing. Residue 20 onward is inside the
        # variable domain for every antibody chain here.
        ch = chains[0]
        start = min(20, max(0, len(ch) - 31))
        for i in range(min(start + 30, len(ch)) - 1, start - 1, -1):
            del ch[i]
    elif case == "shift":
        # translate one chain by 5 A. Sequences and coverage are untouched; the single pooled
        # superposition is what must catch this.
        for r in chains[0]:
            for at in r:
                at.pos = gemmi.Position(at.pos.x + 5.0, at.pos.y, at.pos.z)
    else:
        raise ValueError(case)
    st.setup_entities()
    doc = st.make_mmcif_document()
    doc.write_file(dst)
    return True


def run_case(tmp, pdb, case):
    lab_dir = os.path.join(tmp, f"{pdb}_{case}")
    os.makedirs(lab_dir, exist_ok=True)
    src = f"{ROOT}/training/prep/job_cifs/{pdb}.cif"
    if not os.path.exists(src):
        raise SystemExit(f"missing label {src} -- run phases/phase_j_labels_and_audit.sh training")
    if not damage(src, os.path.join(lab_dir, f"{pdb}.cif"), case):
        return None
    sel = os.path.join(tmp, f"sel_{pdb}.csv")
    with open(sel, "w") as fh:
        fh.write("pdb,instance\n" + f"{pdb},{instance_of(pdb)}\n")
    out = os.path.join(tmp, f"{pdb}_{case}.csv")
    subprocess.run([PY, "-I", AUDIT, "--which", "training", "--selection", sel,
                    "--labels", lab_dir, "--out", out],
                   check=True, capture_output=True)
    with open(out) as fh:
        return next(csv.DictReader(fh))


# what each case must produce: a predicate over the audit's own row, so the assertion is about the
# REPORT a reviewer would read, not about internal state
EXPECT = {
    "clean":        ("ok",                    lambda r: r["verdict"] == "ok"),
    "substitution": ("SUBSTITUTION",          lambda r: r["verdict"] == "SUBSTITUTION"
                                                        and int(r["n_subs"] or 0) >= 1),
    "swap":         ("anything but ok",       lambda r: r["verdict"] != "ok"),
    "truncate":     ("anything but ok",       lambda r: r["verdict"] != "ok"),
    "shift":        ("COORD MISMATCH",        lambda r: r["verdict"] == "COORD MISMATCH"),
}


def main():
    tmp = tempfile.mkdtemp(prefix="label_audit_selftest_")
    bad = 0
    try:
        for pdb, kind in TARGETS:
            print(f"\n=== {pdb} -- {kind} ===")
            for case, (want, pred) in EXPECT.items():
                row = run_case(tmp, pdb, case)
                if row is None:
                    print(f"  {case:13s} SKIPPED (needs two chains)")
                    continue
                got = row["verdict"]
                ok = pred(row)
                bad += 0 if ok else 1
                print(f"  {case:13s} want {want:18s} got {got:34s} "
                      f"rmsd={row['rmsd'] or '-':>7s} nsub={row['n_subs'] or 0:>2s} "
                      f"{'PASS' if ok else 'FAIL -- THE AUDIT MISSED THIS'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL CASES DETECTED' if not bad else f'{bad} CASE(S) NOT DETECTED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
