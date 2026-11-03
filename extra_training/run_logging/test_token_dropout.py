#!/usr/bin/env python
"""--token-keep-union / --token-keep-other: the draw's properties, and the two it must not break.

THE IDEA. Instead of the union mask as a FIXED crop, draw a token subset every epoch: keep a
token inside the union with one probability and outside it with a lower one. A dropped token has
its ROW AND COLUMN removed from z, not zeroed, so 0.5 of tokens is 0.25 of the pair work -- the
compute saving and the augmentation come from the same mechanism.

THE TWO INVARIANTS, both of which are easy to get silently wrong:

  1  CROSS-RANK AGREEMENT. Under --rank-cross-seed the ranks hold different SEEDS OF THE SAME
     STRUCTURE inside one optimiser step, so every rank must draw the IDENTICAL mask for that
     structure or those poses are scored on different token sets and the relative loss grades the
     disagreement rather than the docking. The first version of this used hash(), which Python
     SALTS PER PROCESS -- so every rank would have drawn differently. It uses hashlib now.
  2  TWO CHAINS MUST SURVIVE. ipTM is undefined for a chain pair with one chain missing, so the
     draw asserts it rather than producing an empty pool.

Run:  python -I run_logging/test_token_dropout.py
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def draw(seed, ep, pdb, union, pu, po):
    """The trainer's draw, reimplemented from the same bytes."""
    b = f"{int(seed)}|{int(ep)}|{pdb}".encode()
    rs = np.random.default_rng(int.from_bytes(hashlib.sha256(b).digest()[:8], "big"))
    return rs.random(union.shape[0]) < np.where(union, pu, po)


N = 800
UNION = np.zeros(N, dtype=bool)
UNION[:int(0.598 * N)] = True          # the measured union fraction
PU, PO = 0.70, 0.20

# ---- 1. determinism and variation -----------------------------------------------------
a = draw(0, 3, "7abc", UNION, PU, PO)
check("the same (seed, epoch, structure) gives the IDENTICAL mask",
      np.array_equal(a, draw(0, 3, "7abc", UNION, PU, PO)))
check("a different EPOCH gives a different mask (it is an augmentation)",
      not np.array_equal(a, draw(0, 4, "7abc", UNION, PU, PO)))
check("a different STRUCTURE gives a different mask",
      not np.array_equal(a, draw(0, 3, "8xyz", UNION, PU, PO)))
check("a different --seed gives a different mask",
      not np.array_equal(a, draw(1, 3, "7abc", UNION, PU, PO)))

# ---- 2. THE CROSS-PROCESS INVARIANT, which hash() would have broken -------------------
code = ("import hashlib,numpy as np;"
        "b=b'0|3|7abc';"
        "rs=np.random.default_rng(int.from_bytes(hashlib.sha256(b).digest()[:8],'big'));"
        "print(int(''.join('1' if x else '0' for x in (rs.random(8) < 0.5)), 2))")
outs = set()
for salt in ("1", "0", "random"):
    env = {**os.environ, "PYTHONHASHSEED": salt}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    outs.add(r.stdout.strip())
check("the seed survives PYTHONHASHSEED changing between processes (ranks agree)",
      len(outs) == 1, f"{len(outs)} distinct result(s) across salts: {outs}")
# and demonstrate that hash() would NOT have
hcode = "print(hash('7abc'))"
houts = set()
for salt in ("1", "0", "2"):
    r = subprocess.run([sys.executable, "-c", hcode], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": salt})
    houts.add(r.stdout.strip())
check("hash() genuinely DOES vary per process, so the bug was real",
      len(houts) > 1, f"{len(houts)} distinct values: {houts}")

# ---- 3. the probabilities land where configured ---------------------------------------
many = np.stack([draw(0, e, "7abc", UNION, PU, PO) for e in range(300)])
check("union tokens are kept at ~--token-keep-union",
      abs(many[:, UNION].mean() - PU) < 0.01, f"{many[:, UNION].mean():.4f} vs {PU}")
check("outside tokens are kept at ~--token-keep-other",
      abs(many[:, ~UNION].mean() - PO) < 0.01, f"{many[:, ~UNION].mean():.4f} vs {PO}")
_tot = many.mean()
check("the overall kept fraction matches the arithmetic",
      abs(_tot - (0.598 * PU + 0.402 * PO)) < 0.01,
      f"{_tot:.4f} vs {0.598 * PU + 0.402 * PO:.4f}; pair work "
      f"{_tot ** 2 / 0.598 ** 2:.2f}x the fixed crop")
check("EVERY token is reachable across epochs (no token is permanently invisible)",
      bool(many.any(axis=0).all()), "each of 800 tokens appears at least once in 300 draws")
check("and no token is always present either (the periphery really varies)",
      not bool(many.all(axis=0).any()))

# ---- 4. the source invariants ---------------------------------------------------------
src = open(TRAIN).read()
check("the seed is hashlib, NOT hash()",
      "hashlib.sha256(_sd_bytes).digest()" in src
      and "abs(hash((int(a.seed)" not in src)
check("the draw is keyed on (seed, EPOCH, structure), not the step",
      'f"{int(a.seed)}|{int(_crop_epoch[0])}|{pdb_}"' in src)
check("the epoch cell is set at the boundary, before the prefetch workers start",
      "_crop_epoch[0] = ep" in src
      and src.index("_crop_epoch[0] = ep") < src.index("_pf_start(ep_units)"))
# PER-PAIR SCOREABILITY, raised in review of c518bc3d. "Two chains survived somewhere" does not
# establish that pair (3,1) of a four-chain complex is still scoreable, and 81% of this pool has
# 2+ scored pairs. The check is per OST-resolved pair and does NOT require any ground-truth
# contact to survive -- contacts vanishing is the augmentation being tested.
check("the guard is PER SCORED PAIR, not two-chains-somewhere",
      "for _x, _y in _prs:" in src and "for scored pair" in src
      and "np.unique(_asym[_keep]).size < 2" not in src)
check("it demands a minimum on EACH side of the pair",
      "_nx < _MIN_CH or _ny < _MIN_CH" in src)
check("it does NOT require a ground-truth contact to survive",
      "the augmentation being tested is precisely" in src)
check("a mask without stored pairs is REFUSED, not checked more weakly",
      "does not carry (no {pdb_}__pairs entry)" in src)
check("the builder stores the scored pairs as asym indices",
      'out[pdb + "__pairs"]' in open(os.path.join(ROOT, "run_logging",
                                                  "build_union_crop_masks.py")).read())
check("the loader does not count __pairs as a structure",
      'k.endswith("__pairs")' in src)
check("both probabilities are resume-guard fields",
      src.count('"token_keep_union": a.token_keep_union,') == 2
      and src.count('"token_keep_other": a.token_keep_other,') == 2,
      "changing the augmentation distribution must be detected on resume")
check("pre-flag bundles get migration defaults for them",
      '"token_keep_union": -1.0, "token_keep_other": -1.0,' in src)
check("both probabilities are required together", "must BOTH be given" in src)
check("zero is refused: that is a crop, not a dropout", "Zero would delete a whole" in src)
check("the periphery may not outrank the interface", "That is not the experiment." in src)
check("--crop-tokens is required, since it supplies the union",
      "needs --crop-tokens, which supplies the " in src)
check("the realised kept fraction is reported per epoch",
      "token draw: kept" in src and "x the fixed" in src)
check("the default leaves every existing arm unchanged",
      'ap.add_argument("--token-keep-union", type=float, default=-1.0,' in src)

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
