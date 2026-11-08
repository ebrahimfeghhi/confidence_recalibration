"""The eval's coverage guard must read the TRIMMED seed set, not assume a contiguous window.

WHAT BROKE A 60-EPOCH RUN. Validation keeps 30 seeds out of a 100-wide sweep, chosen per structure
by anchored-coverage-v2, so the kept ids are SPARSE: 29tj keeps 400000, 400002, 400004, 400005,
400011, ... up to 400092. phase_k_eval enumerated range(seed_base, seed_base + budget) and reported
every unkept id inside that window as a MISSING TRUNK -- 321 of 1590 on the valfull pool -- so the
epoch-5 held-out eval stopped, and the trainer, correctly refusing an uninterpretable curve,
stopped with it. The data was never missing. Only 5 of 83 structures kept a contiguous set, so the
sparse case is the rule and the contiguous one the exception.

These checks run against the REAL manifest and the REAL trunks on disk. A synthetic fixture would
have been built with the same assumption the bug was made of, and would have passed before the fix.

THE VACUITY TRAP, stated up front: a manifest that merely restated the contiguous window would make
every "the manifest is authoritative" check pass while proving nothing. Section 2 therefore asserts
that the manifest's answer DIFFERS from the window for a majority of structures, and section 3
reproduces the 321-unit failure before asserting it is gone.
"""
import csv
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

POOL = os.path.join(ROOT, "training/valfull/pool.csv")
MAN = os.path.join(ROOT, "training/valfull/seed_manifest.json")
FOLD = os.path.join(ROOT, "validation/fold")
BUDGET, SEED_BASE = 30, 400000
WINDOW = tuple(range(SEED_BASE, SEED_BASE + BUDGET))

FAIL = []
# mkdtemp, not a fixed path: two runs of this file must not share mutated manifests, and it honours
# TMPDIR so the sandbox decides where they land.
TMP = tempfile.mkdtemp(prefix="test_seed_manifest_")


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def raises(fn, needle, name):
    """The guard must REFUSE, and refuse for the stated reason -- not merely fail somehow."""
    try:
        fn()
    except SystemExit as e:
        ok(needle in str(e), name, f"got: {str(e)[:70]}")
        return
    except Exception as e:                                  # noqa: BLE001
        ok(False, name, f"raised {type(e).__name__}, not SystemExit")
        return
    ok(False, name, "did not refuse at all")


class A:
    """phase_k_eval passes its argparse Namespace around as a config bag; this is that shape."""
    def __init__(self, **kw):
        self.seed_manifest, self.seed_base, self.budget = "", SEED_BASE, BUDGET
        self.__dict__.update(kw)


def pool_pdbs():
    with open(POOL) as fh:
        return sorted({r["pdb"] for r in csv.DictReader(fh)})


def written():
    with open(MAN) as fh:
        return json.load(fh)["structures"]


def tmp_manifest(tag, mutate):
    man = json.load(open(MAN))
    mutate(man)
    p = os.path.join(TMP, f"test_seed_manifest_{tag}.json")
    with open(p, "w") as fh:
        json.dump(man, fh)
    return p


def trunk(pdb, seed):
    return os.path.join(FOLD, "trunks", pdb, f"free_{seed}.npz")


def main():
    import phase_k_eval as EV

    for p, what in ((MAN, "manifest"), (POOL, "pool")):
        if not os.path.exists(p):
            print(f"STOP: the {what} {p} is absent, so nothing here would be measured.")
            return 1
    pdbs = pool_pdbs()
    man = written()

    def fresh(**kw):
        EV._SEED_MANIFEST.clear()
        return A(**kw)

    print("\n1  THE DEFAULT IS UNCHANGED -- an untrimmed fold must behave exactly as before")
    ok(EV.intended_seeds(fresh(), "22gi") == WINDOW,
       "no --seed-manifest gives the contiguous window", f"{WINDOW[0]}..{WINDOW[-1]}")

    print("\n2  THE MANIFEST CHANGES THE ANSWER (not a restatement of the window)")
    a = fresh(seed_manifest=MAN)
    differ = [p for p in pdbs if EV.intended_seeds(a, p) != WINDOW]
    ok(len(differ) > len(pdbs) // 2,
       "a majority of structures keep a set the window does NOT name",
       f"{len(differ)} of {len(pdbs)} differ")
    ok(all(EV.intended_seeds(a, p) == tuple(int(s) for s in man[p]["seeds"]) for p in pdbs),
       "and the returned set is exactly what the manifest wrote")
    ok(all(len(set(EV.intended_seeds(a, p))) == BUDGET for p in pdbs),
       "sparse does not mean shallow: every structure still carries the full budget")
    ok(set(man) == set(pdbs),
       "the manifest covers the pool exactly, no more and no less",
       f"{len(man)} structure(s)")

    print("\n3  AGAINST THE REAL TRUNKS: reproduce the stop, then show it is gone")
    win_miss = [f"{p}/free_{s}" for p in pdbs for s in WINDOW if not os.path.exists(trunk(p, s))]
    ok(len(win_miss) > 0,
       "the contiguous window still reproduces the failure that stopped the run",
       f"{len(win_miss)} of {len(pdbs) * BUDGET} 'missing', e.g. {win_miss[0]}")
    a = fresh(seed_manifest=MAN)
    man_miss = [f"{p}/free_{s}" for p in pdbs for s in EV.intended_seeds(a, p)
                if not os.path.exists(trunk(p, s))]
    ok(man_miss == [],
       "every seed the trim KEPT has its trunk -- the data was never missing",
       f"0 of {len(pdbs) * BUDGET} missing" if not man_miss else f"{man_miss[:3]}")

    print("\n4  THE GUARD STILL REFUSES -- the assertion is preserved, not weakened")
    a = fresh(seed_manifest=MAN)
    raises(lambda: EV.intended_seeds(a, "zzzz"), "does not name structure",
           "a structure the manifest omits is fatal (a partial manifest cannot shrink the pool)")
    p0 = pdbs[0]
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("short", lambda d: d["structures"][p0].__setitem__(
            "seeds", d["structures"][p0]["seeds"][:-1]))), p0),
        "but --budget is 30",
        "a manifest keeping 29 against --budget 30 is fatal (the depth claim)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("oor", lambda d: d["structures"][p0]["seeds"].__setitem__(
            0, 999999))), p0),
        "outside its own",
        "a seed outside the manifest's own seed_range is fatal")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("dup", lambda d: d["structures"][p0]["seeds"].__setitem__(
            1, d["structures"][p0]["seeds"][0]))), p0),
        "duplicate seed",
        "a duplicated seed is fatal (it would be weighted twice in the argmax)")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("empty", lambda d: d.__setitem__("structures", {}))), p0),
        "names no structures",
        "an empty manifest is fatal")
    _bad = os.path.join(TMP, "test_seed_manifest_unreadable.json")
    with open(_bad, "w") as fh:
        fh.write("{not json")
    raises(lambda: EV.intended_seeds(fresh(seed_manifest=_bad), p0), "unreadable",
           "an unparseable manifest is fatal")

    print("\n5  BOTH CALL SITES RESOLVE TO THE SAME STRUCTURE")
    pairs, unresolved = EV.interface_chain_pairs(POOL, os.path.join(FOLD, "chain_roles.csv"))
    ok(not unresolved, "every pool interface maps to model chains", f"{len(pairs)} structure(s)")
    ifaces = [i for v in pairs.values() for i, _x, _y in v]
    ok(len(ifaces) > 0, "the pool yields interfaces at all", f"{len(ifaces)}")
    ok([i for i in ifaces if i.rsplit("_", 1)[0] not in man] == [],
       "the scored-key guard's iface -> pdb rsplit lands on a manifest structure every time",
       f"e.g. {ifaces[0]} -> {ifaces[0].rsplit('_', 1)[0]}")

    print("\n6  WIRING")
    ok("--seed-manifest" in open(EV.__file__).read(), "the flag exists in the evaluator's CLI")
    _tr = open(os.path.join(ROOT, "phases/phase_k_train.py")).read()
    ok('"--seed-manifest", _sman' in _tr, "the trainer forwards it")
    ok('os.path.dirname(os.path.abspath(train_pool))' in _tr,
       "derived from the POOL's directory, not the fold being trained on")
    ok(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(POOL)),
                                   "seed_manifest.json")),
       "and that derivation finds the real file for this pool")

    print("\n7  THE FLAG SURVIVES THE SHARDED DRIVER, which is the path the trainer takes")
    # phase_k_eval_sharded.py uses a STRICT parse_args plus one space-separated --extra string, so
    # a bare --seed-manifest would be rejected outright and a path with a space would be split into
    # two arguments and silently change what the metric covers.
    _sh = open(os.path.join(ROOT, "phases/phase_k_eval_sharded.py")).read()
    ok("--extra" in _sh and "a.extra.split()" in _sh,
       "the driver takes --extra and splits it back into argv")
    ok(_sh.count("*extra") >= 2,
       "and forwards it to BOTH the per-shard pass and the merge pass",
       f"{_sh.count('*extra')} call site(s)")
    _tr = open(os.path.join(ROOT, "phases/phase_k_train.py")).read()
    ok('cmd += ["--extra", " ".join(extra)]' in _tr,
       "the trainer packs its extra argv into that one string")
    _sman = os.path.join(os.path.dirname(os.path.abspath(POOL)), "seed_manifest.json")
    _round = " ".join(["--seed-manifest", _sman]).split()
    ok(_round == ["--seed-manifest", _sman],
       "and this pool's manifest path round-trips through join/split intact",
       "no whitespace in the path")

    print("\n8  MALFORMED MANIFESTS REFUSE WITH A SENTENCE, not a traceback")
    # Review found every shape error below raising AttributeError/IndexError/ValueError from an
    # unguarded .get()/[]/int(). On the sharded path that surfaces as a traceback buried in
    # shard<i>.log instead of a line saying what is wrong with the file.
    def _raw(tag, text):
        q = os.path.join(TMP, f"raw_{tag}.json")
        with open(q, "w") as fh:
            fh.write(text)
        return q

    raises(lambda: EV.intended_seeds(fresh(seed_manifest=_raw("toplist", "[1,2,3]")), p0),
           "at the top level", "a top-level JSON array refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=_raw("stlist", '{"structures": [1,2]}')), p0),
        "'structures' as a", "'structures' as an array refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=_raw("bare", '{"structures": {"%s": [400000]}}' % p0)), p0),
        "no 'seeds' list", "a bare seed list (the plausible hand-written shape) refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=_raw("nonint",
                                 '{"structures": {"%s": {"seeds": ["x"]}}}' % p0)), p0),
        "non-integer seed", "a non-integer seed refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=_raw("noseeds",
                                 '{"structures": {"%s": {"seeds": []}}}' % p0)), p0),
        "lists no seeds", "an empty seed list refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("rng1", lambda d: d.__setitem__("seed_range", [400000]))),
        p0), "not a [lo, hi] pair", "a one-element seed_range refuses")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest(
            "rngstr", lambda d: d.__setitem__("seed_range", "400000-400099"))), p0),
        "is empty", "a STRING seed_range refuses instead of silently meaning [4, 0]")
    raises(lambda: EV.intended_seeds(
        fresh(seed_manifest=tmp_manifest("kbad", lambda d: d.__setitem__("k", 99))), p0),
        "declares k=99", "a manifest whose own k disagrees with --budget refuses")

    print("\n9  AND A VALID MANIFEST WITHOUT seed_range IS ACCEPTED")
    # The old fallback invented [seed_base, seed_base + budget - 1] and then rejected a correctly
    # trimmed manifest for exactly the seeds the trim KEPT (400092 is outside 400000-400029),
    # blaming the manifest for a range it never declared.
    _nr = tmp_manifest("norange", lambda d: d.pop("seed_range", None))
    a2 = fresh(seed_manifest=_nr)
    try:
        _got = EV.intended_seeds(a2, p0)
        ok(_got == tuple(int(x) for x in man[p0]["seeds"]),
           "no declared seed_range means no range check, not an invented one",
           f"{len(_got)} seed(s), max {max(_got)}")
    except SystemExit as e:
        ok(False, "no declared seed_range means no range check, not an invented one",
           f"refused: {str(e)[:70]}")
    ok(any(s > SEED_BASE + BUDGET - 1 for s in man[p0]["seeds"]),
       "and that structure really does keep a seed the old invented range excluded",
       f"max kept {max(int(x) for x in man[p0]['seeds'])} > {SEED_BASE + BUDGET - 1}")

    print("\n10  THE METRIC RECORDS WHICH SEED SET PRODUCED IT")
    a3 = fresh(seed_manifest=MAN)
    EV.intended_seeds(a3, p0)
    _meta = EV._SEED_MANIFEST_META.get(MAN, {})
    ok(bool(_meta.get("digest")), "the manifest's digest is captured on load",
       str(_meta.get("digest")))
    _src = open(EV.__file__).read()
    ok('out["seed_manifest_digest"]' in _src and 'out["seed_manifest"]' in _src,
       "and written into the metric json beside the number",
       "budget + seed_base no longer imply the intended set")

    print("\n11  THE 83-STRUCTURE SET (both waves) IS COMPLETE AND SELF-DESCRIBING")
    P83 = os.path.join(ROOT, "training/valfull83")
    if not os.path.exists(os.path.join(P83, "pool.csv")):
        ok(False, "training/valfull83 exists", "absent")
    else:
        _pool83 = os.path.join(P83, "pool.csv")
        _rol83 = os.path.join(P83, "chain_roles.csv")
        _man83 = os.path.join(P83, "seed_manifest.json")
        ok(all(os.path.exists(x) for x in (_pool83, _rol83, _man83)),
           "the pool directory carries its own pool, roles and manifest")
        _pairs, _unres = EV.interface_chain_pairs(_pool83, _rol83)
        ok(len(_unres) == 0 and len(_pairs) == 83,
           "all 83 structures resolve, none unmappable",
           f"{len(_pairs)} structures, {len(_unres)} unresolved")
        _n = sum(len(v) for v in _pairs.values())
        ok(_n == 141, "and all 141 interfaces are present", f"{_n}")
        # the fold's OWN roles must still fail here, or the merge changed nothing
        _p2, _u2 = EV.interface_chain_pairs(_pool83, os.path.join(FOLD, "chain_roles.csv"))
        ok(len(_u2) == 49,
           "the fold's 53-structure roles still leaves 49 unmappable (so the merge matters)",
           f"{len(_u2)} unresolved")
        a83 = fresh(seed_manifest=_man83)
        _miss = [f"{p}/{s}" for p in _pairs for s in EV.intended_seeds(a83, p)
                 if not os.path.exists(trunk(p, s))]
        ok(_miss == [], "every one of the 83 x 30 trunks is on disk",
           f"0 of {83 * BUDGET} missing" if not _miss else str(_miss[:3]))
        _m53 = set(json.load(open(MAN))["structures"])
        _m83 = set(json.load(open(_man83))["structures"])
        ok(_m53 < _m83 and len(_m83) == 83,
           "the 83 manifest is a strict superset of the 53", f"{len(_m83 - _m53)} added")
        _tr = open(os.path.join(ROOT, "phases/phase_k_train.py")).read()
        ok('_srol if os.path.exists(_srol)' in _tr,
           "and the trainer prefers a roles table beside the pool, falling back to the fold")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
