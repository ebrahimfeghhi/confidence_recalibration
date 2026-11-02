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


def chunk_steps(order, by, S, manifest_order, seed=0):
    """build_epoch_cross_seed's structure ordering and chunking, reimplemented."""
    rng = random.Random(seed)
    o = list(order) if manifest_order else sorted(order)
    if not manifest_order:
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
need = [
    '_manifest_order = (a.epoch_order == "manifest")',
    'order = list(by) if _manifest_order else sorted(by)',
    'if not _manifest_order:\n            rng_e.shuffle(order)',
    'if not _manifest_order:\n                    rng_e.shuffle(u)',
]
for frag in need:
    check(f"trainer still contains: {frag.splitlines()[0][:52]}", frag in src)
# BOTH builders, not one. build_epoch and build_epoch_cross_seed each had the shuffle.
_sites = src.count('_manifest_order = (a.epoch_order')
check("both builders read a.epoch_order", _sites == 2, f"{_sites} site(s)")
check("no unguarded shuffle of the structure order is left",
      not re.search(r"^\s*order = sorted\(by\)\n\s*rng_e\.shuffle\(order\)", src, re.M))
check("the dropped-chunk report exists",
      "DROPPED as a short" in src and "--rank-structs-per-step" in src)

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print(f"  all checks passed")
