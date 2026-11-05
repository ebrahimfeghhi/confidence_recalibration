#!/usr/bin/env python
"""Tests for dedupe_results.py -- that it resolves on evidence and fails closed without it.

The property that matters: an unresolved duplicate must produce a nonzero exit AND no snapshot.
A snapshot that quietly picked a contested label would be treated downstream as settled.

Run: python -I run_logging/test_dedupe_results.py
"""
import fcntl, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FAILED = []

HDR = "iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm"


def check(label, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  {extra}" if not cond and extra else ""))
    if not cond:
        FAILED.append(label)


def build(root, rows, runs, pool=(("8abc", "A", "B"),), roles=(("8abc", "A", "A"),
                                                                ("8abc", "B", "B")), ost=None):
    """rows: csv lines. runs: {(pdb,seed,tag): {sample: [iptm row in the json]}}

    pool and roles are written too, because identity is now checked against the SPECIFIC
    interface's chain pair: pool.csv gives the reference pair for an iface name, chain_roles maps
    those reference chains to model chains, and the pair ipTM is read at those indices.
    """
    csvp = os.path.join(root, "results.csv")
    with open(csvp, "w") as fh:
        fh.write(HDR + "\n")
        for r in rows:
            fh.write(r + "\n")
    with open(os.path.join(root, "pool.csv"), "w") as fh:
        fh.write("pdb,chain1,chain2\n")
        for pdb, c1, c2 in pool:
            fh.write(f"{pdb},{c1},{c2}\n")
    with open(os.path.join(root, "chain_roles.csv"), "w") as fh:
        fh.write("pdb,model_chain,ref_chain,auth_chain,role\n")
        for pdb, ref, model in roles:
            fh.write(f"{pdb},{model},{ref},{ref},H\n")
    for (pdb, seed, tag), samples in runs.items():
        d = os.path.join(root, "poses", pdb, f"{seed}-{tag}")
        os.makedirs(os.path.join(d, "predictions"), exist_ok=True)
        json.dump({"pdb": pdb, "seed": int(seed), "arm": "free", "n_sample": 5},
                  open(os.path.join(d, "run.json"), "w"))
        for k, ips in samples.items():
            open(os.path.join(d, "predictions", f"{pdb}_sample_{k}.cif"), "w").write("data_x\n")
            json.dump({"chain_pair_iptm": ips if ips and isinstance(ips[0], list) else [ips]},
                      open(os.path.join(d, "predictions",
                                        f"{pdb}_summary_confidence_sample_{k}.json"), "w"))
        # OST's own output, when the case supplies one: this is the mapping the scorer used when
        # it wrote the row, and it is what identity is checked against.
        for k, cm in (ost or {}).get((pdb, seed, tag), {}).items():
            os.makedirs(os.path.join(d, "ost_json"), exist_ok=True)
            json.dump({"chain_mapping": cm},
                      open(os.path.join(d, "ost_json", f"{k}.json"), "w"))
    return csvp


def run(root, csvp, out=None):
    cmd = [PY, "-I", os.path.join(HERE, "dedupe_results.py"), "--csv", csvp,
           "--poses", os.path.join(root, "poses"), "--nsample", "1",
           "--pool", os.path.join(root, "pool.csv"),
           "--roles", os.path.join(root, "chain_roles.csv")]
    if out:
        cmd += ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


print("resolvable: the row whose ipTM matches the retained pose wins")
root = tempfile.mkdtemp()
try:
    # the json holds full precision; the csv holds 4 decimals -- the real case
    csvp = build(root,
                 ["8abc_AB,500000,free,-1,,0,0.4660,0.5226",
                  "8abc_AB,500000,free,-1,,0,0.4660,0.5231"],
                 {("8abc", "500000", "aaa"): {0: [[0.9, 0.523106], [0.523106, 0.9]]}})
    rc, out = run(root, csvp)
    check("exit 0 when every duplicate resolves", rc == 0, out[-300:])
    check("reports 1 resolved", "resolved against the retained pose:   1" in out, out[-300:])
    o = os.path.join(root, "snap.csv")
    rc, out = run(root, csvp, out=o)
    check("writes a snapshot", rc == 0 and os.path.exists(o))
    if os.path.exists(o):
        body = open(o).read()
        check("keeps the row matching the retained pose (0.5231)", "0.5231" in body)
        check("drops the row from the vanished fold (0.5226)", "0.5226" not in body)
finally:
    shutil.rmtree(root, ignore_errors=True)

print("unresolvable: fail closed, no snapshot")
for name, rows, runs in [
    ("neither row matches the retained pose",
     ["8abc_AB,500000,free,-1,,0,0.4660,0.1111",
      "8abc_AB,500000,free,-1,,0,0.4660,0.2222"],
     {("8abc", "500000", "aaa"): {0: [[0.9, 0.9], [0.9, 0.9]]}}),
    ("both rows match the retained pose",
     ["8abc_AB,500000,free,-1,,0,0.4660,0.5000",
      "8abc_AB,500000,free,-1,,0,0.4660,0.5000"],
     {("8abc", "500000", "aaa"): {0: [[0.9, 0.5], [0.5, 0.9]]}}),
    ("two complete retained runs -- ambiguous",
     ["8abc_AB,500000,free,-1,,0,0.4660,0.5000",
      "8abc_AB,500000,free,-1,,0,0.4660,0.6000"],
     {("8abc", "500000", "aaa"): {0: [[0.9, 0.5], [0.5, 0.9]]},
      ("8abc", "500000", "bbb"): {0: [[0.9, 0.6], [0.6, 0.9]]}}),
    ("the poses are gone -- no evidence at all",
     ["8abc_AB,500000,free,-1,,0,0.4660,0.5000",
      "8abc_AB,500000,free,-1,,0,0.4660,0.6000"],
     {}),
]:
    root = tempfile.mkdtemp()
    try:
        csvp = build(root, rows, runs)
        o = os.path.join(root, "snap.csv")
        rc, out = run(root, csvp, out=o)
        check(f"{name}: nonzero exit", rc != 0, out[-200:])
        check(f"{name}: NO snapshot written", not os.path.exists(o))
    finally:
        shutil.rmtree(root, ignore_errors=True)

print("a value matching a DIFFERENT chain pair must not count as identity")
root = tempfile.mkdtemp()
try:
    # 8abc_AB is model A,B -> M[0][1] = 0.4209. A row carrying 0.6320 matches M[0][2], another
    # pair entirely -- the flattened search accepted exactly this and resolved it wrongly.
    csvp = build(root,
                 ["8abc_AB,500000,free,-1,,0,0.4660,0.6273",
                  "8abc_AB,500000,free,-1,,0,0.4660,0.6320"],
                 {("8abc", "500000", "aaa"): {0: [[0.9, 0.4209, 0.6320],
                                                  [0.4209, 0.9, 0.1],
                                                  [0.6320, 0.1, 0.9]]}},
                 roles=(("8abc", "A", "A"), ("8abc", "B", "B"), ("8abc", "C", "C")))
    o = os.path.join(root, "snap.csv")
    rc, out = run(root, csvp, out=o)
    check("nonzero exit when only another pair's value matches", rc != 0, out[-200:])
    check("NO snapshot written", not os.path.exists(o))
    check("says the value belongs to a different chain pair",
          "DIFFERENT chain pair" in out, out[-300:])
finally:
    shutil.rmtree(root, ignore_errors=True)

print("refuses to overwrite its own input")
root = tempfile.mkdtemp()
try:
    csvp = build(root, ["8abc_AB,500000,free,-1,,0,0.4,0.5"], {})
    rc, out = run(root, csvp, out=csvp)
    check("exit nonzero when --out is the input", rc != 0 and "must not be the input" in out)
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
print("OST's own mapping decides, not the role table")
root = tempfile.mkdtemp()
try:
    # roles say the interface is model (A,B) -> ipTM[0][1] = 0.5000.
    # OST recorded that it mapped it to (B,C) -> ipTM[1][2] = 0.7777, and that is the value the
    # row was written from. Under the role table the OTHER row would win, so this asserts the
    # retained ost json actually changes which row is kept -- the real 7uij/8w2e case.
    M = [[1.0, 0.5000, 0.1000],
         [0.5000, 1.0, 0.7777],
         [0.1000, 0.7777, 1.0]]
    csvp = build(root,
                 ["8abc_AB,500000,free,-1,,0,0.4660,0.5000",
                  "8abc_AB,500000,free,-1,,0,0.4660,0.7777"],
                 {("8abc", "500000", "aaa"): {0: M}},
                 ost={("8abc", "500000", "aaa"): {0: {"A": "B", "B": "C"}}})
    out = os.path.join(root, "snap.csv")
    rc, log = run(root, csvp, out=out)
    check("resolves cleanly", rc == 0, log)
    kept = [ln for ln in open(out).read().splitlines() if ln.startswith("8abc_AB")]
    check("keeps the row OST's mapping identifies", len(kept) == 1 and kept[0].endswith("0.7777"),
          f"kept={kept}")
    check("says which mapping identified it", "ost chain_mapping" in log, log)

    # with the json gone it must fall back to the role table and keep the other row
    os.remove(os.path.join(root, "poses", "8abc", "500000-aaa", "ost_json", "0.json"))
    out2 = os.path.join(root, "snap2.csv")
    rc2, log2 = run(root, csvp, out=out2)
    kept2 = [ln for ln in open(out2).read().splitlines() if ln.startswith("8abc_AB")] if rc2 == 0 else []
    check("falls back to chain_roles when no ost json is retained",
          rc2 == 0 and len(kept2) == 1 and kept2[0].endswith("0.5000"), f"rc={rc2} kept={kept2}")

    # an ost json that leaves one of the pair's chains unmapped is not usable either
    os.makedirs(os.path.join(root, "poses", "8abc", "500000-aaa", "ost_json"), exist_ok=True)
    json.dump({"chain_mapping": {"A": "B"}},
              open(os.path.join(root, "poses", "8abc", "500000-aaa", "ost_json", "0.json"), "w"))
    out3 = os.path.join(root, "snap3.csv")
    rc3, log3 = run(root, csvp, out=out3)
    kept3 = [ln for ln in open(out3).read().splitlines() if ln.startswith("8abc_AB")] if rc3 == 0 else []
    check("an unmapped chain in the ost json falls back too",
          rc3 == 0 and len(kept3) == 1 and kept3[0].endswith("0.5000"), f"rc={rc3} kept={kept3}")
finally:
    shutil.rmtree(root, ignore_errors=True)

print("--replace takes the rescore writer lock before it reads")
root = tempfile.mkdtemp()
try:
    # one clean duplicate, so the snapshot itself is never the thing being tested here
    M = [[1.0, 0.5226], [0.5226, 1.0]]
    csvp = build(root,
                 ["8abc_AB,500000,free,-1,,0,0.4660,0.5226",
                  "8abc_AB,500000,free,-1,,0,0.4660,0.5231"],
                 {("8abc", "500000", "aaa"): {0: M}})
    out = os.path.join(root, "snap.csv")

    def replace_run():
        r = subprocess.run([PY, "-I", os.path.join(HERE, "dedupe_results.py"), "--csv", csvp,
                            "--poses", os.path.join(root, "poses"), "--nsample", "1",
                            "--pool", os.path.join(root, "pool.csv"),
                            "--roles", os.path.join(root, "chain_roles.csv"),
                            "--out", out, "--replace"], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    before = open(csvp).read()
    lk = open(csvp + ".rescore.lock", "a+")
    fcntl.flock(lk, fcntl.LOCK_EX)
    rc, log = replace_run()
    check("refuses while a rescore holds the lock", rc != 0 and "rescore holds" in log, log[-300:])
    check("and the canonical csv is untouched", open(csvp).read() == before)
    fcntl.flock(lk, fcntl.LOCK_UN)
    lk.close()

    # with the lock free it gets past that guard -- here it stops on the live fold sessions
    # instead, which is the pre-existing check and proves the lock was not what blocked it
    rc2, log2 = replace_run()
    check("with the lock free it is no longer the lock that blocks",
          "rescore holds" not in log2, log2[-300:])
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
print(f"FAILURES: {len(FAILED)}" + (f" -> {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
