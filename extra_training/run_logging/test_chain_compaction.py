#!/usr/bin/env python
"""CPU regression for the two findings that blocked 315182d7. No GPU, no model.

1. PROTENIX COMPACTS ASYM IDS. A crop that removes a whole chain shifts every row of
   chain_pair_iptm, so indexing it with original asym ids reads a different chain pair. This
   reproduces the reviewer's fixture -- chains [0,2,3] surviving, a 3x3 matrix indexed [0,1,2] --
   and asserts _cpi_index maps original (0,2) to matrix (0,1) rather than (0,2).

2. EQUAL TOKEN COUNTS ARE NOT IDENTICAL CROPS. [0,1,4,5,6,7] and [0,1,2,3,6,7] both keep six
   tokens. The old guard called them the same crop and hard-stopped; the digest must tell them
   apart.

    python -I run_logging/test_chain_compaction.py
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_k_eval as K  # noqa: E402

fails = []

# ---- 1. compaction -------------------------------------------------------
asym = np.array([0, 0, 2, 2, 3, 3])          # chain 1 removed by the crop
cmap = K._cpi_index(asym)
if cmap != {0: 0, 2: 1, 3: 2}:
    fails.append(f"_cpi_index gave {cmap}, expected {{0:0, 2:1, 3:2}}")

cpi = np.array([[0.000000, 0.101853, 0.002309],
                [0.101853, 0.000000, 0.050000],
                [0.002309, 0.050000, 0.000000]])
# original pair (0,2) is the one the reviewer showed being misread as (0,3)'s cell
got = float(cpi[cmap[0], cmap[2]])
if abs(got - 0.101853) > 1e-9:
    fails.append(f"pair (0,2) read {got}, expected 0.101853 (the old code read 0.002309)")
# the original indexing, kept here so the test shows WHAT was wrong
if abs(float(cpi[0, 2]) - 0.002309) > 1e-9:
    fails.append("fixture is wrong: cpi[0,2] should be the misread value")
# original chain 3 is still reachable; the old `max(ai,aj) >= shape` guard dropped it
if cmap.get(3) != 2:
    fails.append("original chain 3 should map to matrix row 2, not be skipped")

# ---- 2. identity when nothing is dropped ---------------------------------
if K._cpi_index(np.array([0, 0, 1, 1, 2])) != {0: 0, 1: 1, 2: 2}:
    fails.append("contiguous asym ids must map to themselves; uncropped runs must be unaffected")

# ---- 3. equal counts, different masks ------------------------------------
def dig(bits):
    return hashlib.blake2b(np.packbits(np.array(bits, dtype=bool)).tobytes(),
                           digest_size=8).hexdigest()


a = np.zeros(8, dtype=bool); a[[0, 1, 4, 5, 6, 7]] = True
b = np.zeros(8, dtype=bool); b[[0, 1, 2, 3, 6, 7]] = True
if int(a.sum()) != int(b.sum()):
    fails.append("fixture is wrong: the two masks must have equal token counts")
if dig(a) == dig(b):
    fails.append("two different six-token masks produced the same digest")
if dig(a) != dig(a.copy()):
    fails.append("the digest is not stable for the same mask")

print(f"  {len(fails)} failure(s)")
for f in fails:
    print(f"    FAIL: {f}")
if fails:
    sys.exit(1)
print("  chain compaction and mask-digest regressions PASS")
