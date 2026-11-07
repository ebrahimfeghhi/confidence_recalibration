#!/usr/bin/env python
"""--epoch-order manifest: the ordering rule, and a pin that the trainer still applies it.

WHAT THIS CAN AND CANNOT CHECK. Both epoch builders are closures inside phase_k_train's main
function, so they cannot be imported and the REAL sequence a GPU run walks is only observable
from that run. bench_ckpt_by_size.py --probe does that part: it reads the token count off the
trainer's own MEM lines and refuses to recommend a threshold unless the sequence is
non-decreasing. This file covers the two things that do not need a card:

  1  THE RULE. A faithful reimplementation of build_epoch_cross_seed's chunking, checked for the
     two properties the probe depends on -- manifest order is preserved exactly, and a short
     final chunk is dropped, which on an ascending manifest means the LARGEST structures.
  2  THE PIN. That phase_k_train.py still guards every shuffle with a.epoch_order. The original
     bug was that `--shuffle none` looked sufficient while an unconditional shuffle ran after it;
     a reimplementation alone would keep passing if someone deleted a guard from the trainer.

Run:  python -I run_logging/test_epoch_order.py
"""
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def chunk_steps(order, by, S, manifest_order, seed=0, mode=None, tok=None):
    """build_epoch_cross_seed's structure ordering and chunking, reimplemented.

    mode="tokens" is the probe's order: ascending by token count, pdb as tie-break. The old
    manifest_order flag is kept so the existing cases still read the same.
    """
    rng = random.Random(seed)
    if mode == "tokens":
        o = sorted(order, key=lambda q: (tok.get(q, 0), q))
    elif manifest_order:
        o = list(order)
    else:
        o = sorted(order)
        rng.shuffle(o)
    steps, dropped = [], []
    for i in range(0, len(o), S):
        chunk = o[i:i + S]
        if len(chunk) < S:
            dropped = list(chunk)
            break
        units = []
        for p in chunk:
            u = list(by[p])
            if not manifest_order:
                rng.shuffle(u)
            units += [(p, x) for x in u]
        if not manifest_order:
            rng.shuffle(units)
        steps.append(units)
    return steps, dropped


# ---- 1. the rule -------------------------------------------------------------------------
# An ascending "manifest": 7 structures, tokens 100..700, two seeds each.
NAMES = [f"s{t}" for t in range(100, 800, 100)]
TOK = {n: int(n[1:]) for n in NAMES}
BY = {n: ["seedA", "seedB"] for n in NAMES}

steps, dropped = chunk_steps(NAMES, BY, S=3, manifest_order=True)
walked = [p for st in steps for p, _ in st]
check("manifest order is preserved exactly",
      walked == [n for n in NAMES for _ in range(2)][:len(walked)],
      f"{walked[:4]} ...")
toks = [TOK[p] for p in walked]
check("the walked token sequence is non-decreasing",
      all(b >= a for a, b in zip(toks, toks[1:])), f"{toks}")
check("a short final chunk IS dropped, and it is the LARGEST structures",
      dropped == ["s700"], f"dropped {dropped}")
check("trimming the SMALLEST to a multiple of S keeps the largest",
      chunk_steps(NAMES[1:], BY, S=3, manifest_order=True)[1] == [],
      "6 structures at S=3 drops nothing")

# the default must still shuffle, or this flag changed training
sh_walked = [p for st in chunk_steps(NAMES, BY, S=3, manifest_order=False, seed=5)[0]
             for p, _ in st]
check("the DEFAULT order still shuffles (so the flag changed nothing by default)",
      sh_walked != walked, f"{sh_walked[:4]} ...")

# ---- 2. the pin --------------------------------------------------------------------------
src = open(TRAIN).read()
check("the flag exists with both choices",
      '"--epoch-order"' in src and '"shuffle", "manifest"' in src)
# PIN THE WHOLE BRANCH, not fragments of it. The three-way if/elif/else IS the guard: there
# is no longer a shuffle merely preceded by a condition, so a regex hunting for an
# "unguarded" shuffle cannot tell one from the other any more. Pinning the exact block can.
BRANCH = '''
        _manifest_order = (a.epoch_order in ("manifest", "tokens"))
        if a.epoch_order == "tokens":
            order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))
        elif _manifest_order:
            order = list(by)
        else:
            order = sorted(by)
            rng_e.shuffle(order)'''.strip(chr(10))
check("both builders carry the exact 3-way order branch", src.count(BRANCH) == 2,
      f"{src.count(BRANCH)} site(s)")
check("the within-structure seed shuffle is still guarded",
      'if not _manifest_order:\n                    rng_e.shuffle(u)' in src)
check("the structure shuffle appears ONLY in that else branch",
      src.count("rng_e.shuffle(order)") == 2,
      f"{src.count('rng_e.shuffle(order)')} occurrence(s)")
# BOTH builders, not one. build_epoch and build_epoch_cross_seed each had the shuffle.
_sites = src.count('_manifest_order = (a.epoch_order')
check("both builders read a.epoch_order", _sites == 2, f"{_sites} site(s)")
# (the old unguarded-shuffle regex is gone: it matched the else branch's own two lines,
#  which are correctly guarded, so it failed on the very shape it required.)
check("the dropped-chunk report exists",
      "DROPPED as a short" in src and "--rank-structs-per-step" in src)

# ---- 3. the tokens mode, which is what the probe actually needs ------------------------
# THE REAL BUG THIS COVERS: `by` is built from `flat`, which is sorted by pdb, so "manifest"
# order is a pdb sort and NOT the csv's row order. A probe that trusted it walked
# 535, 631, 578, 404 and OOMed on the 5th structure alphabetically (1272 tokens) having
# measured nothing above 631. Names here are deliberately ANTI-correlated with size.
ANTI = {"a_big": 1473, "b_mid": 700, "c_small": 128, "d_huge": 1543, "e_tiny": 100, "f_mid": 600}
aby = {k: ["seedA", "seedB"] for k in ANTI}
tw = [p for st in chunk_steps(list(ANTI), aby, S=3, manifest_order=True,
                              mode="tokens", tok=ANTI)[0] for p, _ in st]
seq = [ANTI[p] for p in tw]
check("tokens mode walks ASCENDING by token count",
      all(b >= a for a, b in zip(seq, seq[1:])), f"{seq}")
check("tokens mode is NOT the pdb sort (the bug it fixes)",
      tw != [p for p in sorted(ANTI) for _ in range(2)][:len(tw)],
      f"pdb order would be {sorted(ANTI)}")
# a structure with no numeric tokens sorts first rather than crashing
NOTOK = dict(ANTI); NOTOK.pop("e_tiny")
tw2 = [p for st in chunk_steps(list(NOTOK) + ["z_unknown"],
                               {**aby, "z_unknown": ["seedA", "seedB"]},
                               S=3, manifest_order=True, mode="tokens", tok=NOTOK)[0]
       for p, _ in st]
check("a structure with no numeric `tokens` sorts first, not crashes",
      tw2[0] == "z_unknown", f"{tw2[:3]}")

# ---- 4. the trainer still contains it --------------------------------------------------
check("the flag offers tokens", '"shuffle", "manifest", "tokens"' in src)
check("both builders sort by tokens",
      src.count('order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))') == 2,
      f"{src.count('order = sorted(by, key=lambda _q: (_sel_tokens.get(_q, 0), _q))')} site(s)")
check("the manifest's token column is captured, not discarded",
      '_sel_tokens = {r["pdb"].strip().lower(): int(r["tokens"])' in src)
check("the help no longer claims manifest order is the csv's row order",
      "SORTED BY PDB, not the selection csv" in src)

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print(f"  all checks passed")
