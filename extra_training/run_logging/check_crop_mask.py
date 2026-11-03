#!/usr/bin/env python
"""Refuse a crop mask whose OWN METADATA does not establish the configuration an arm requires.

WHY THIS IS SEPARATE FROM THE CONTENT DIGEST. phase_k_train's digest proves the mask has not
CHANGED between a run and its resume. It says nothing about whether the file was correct to begin
with: a mask built without --force-gt, or with --force-gt but without --ost-pairs, hashes just as
stably as the right one. The launcher needs the second guarantee before it spends days, and
"the file exists" was all it checked. Raised in review.

WHAT IT ASSERTS, from __meta and the per-structure __n entries:
  --require-force-gt    the ground-truth contacts were forced in, so interface coverage is 1.0 by
                        construction rather than 1.0 at the median
  --require-ost-pairs   the forced set covers the pairs the TRAINER supervises, which it resolves
                        per pose through OST and which disagree with the static pool mapping on 52
                        of 368 structures
  --structures N        the mask names exactly N structures, so a truncated build cannot pass
  --poses N             every structure's union was built from exactly N examined poses
  --margin / --contact  the geometry matches what the arm's note claims

Exit 0 and print a one-line summary, or exit 1 naming what failed.

  python -I run_logging/check_crop_mask.py training/full_train/union_crop_m5_gt_ost.npz \
      --require-force-gt --require-ost-pairs --structures 368 --poses 50 --margin 5.0
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mask")
ap.add_argument("--require-force-gt", action="store_true")
ap.add_argument("--require-ost-pairs", action="store_true")
ap.add_argument("--structures", type=int, default=0)
ap.add_argument("--poses", type=int, default=0)
ap.add_argument("--margin", type=float, default=0.0)
ap.add_argument("--contact", type=float, default=0.0)
a = ap.parse_args()

try:
    z = np.load(a.mask, allow_pickle=True)
except Exception as e:                                                       # noqa: BLE001
    sys.exit(f"STOP: cannot read {a.mask}: {type(e).__name__}: {e}")

bad = []
if "__meta" not in z.files:
    sys.exit(f"STOP: {a.mask} has no __meta, so nothing about how it was built can be verified. "
             f"It predates the provenance block; rebuild it.")
meta = json.loads(str(np.asarray(z["__meta"]).reshape(-1)[0]))

keys = [k for k in z.files if not (k.endswith("__n") or k == "__meta")]
nn = {k[:-3]: np.asarray(z[k]) for k in z.files if k.endswith("__n")}

if a.require_force_gt and not meta.get("force_gt"):
    bad.append("--force-gt was NOT used, so interface coverage is 1.0 only at the median: the "
               "bottom decile loses contacts the loss scores, against a label that cannot be "
               "cropped to match")
if a.require_ost_pairs and not meta.get("ost_pairs"):
    bad.append("--ost-pairs was NOT used, so the forced set covers the STATIC pool mapping and "
               "not necessarily the pair the trainer supervises (they disagree on 52 of 368 "
               "structures; 7stf differs by a whole chain)")
if a.structures and len(keys) != a.structures:
    bad.append(f"names {len(keys)} structure(s), not the required {a.structures} -- a truncated "
               f"or differently-pooled build")
if a.poses:
    shortfall = {k: int(v[0]) for k, v in nn.items() if int(v[0]) != a.poses}
    if shortfall:
        bad.append(f"{len(shortfall)} structure(s) were built from a pose count other than "
                   f"{a.poses}, e.g. {dict(list(shortfall.items())[:4])}")
    want = meta.get("ost_poses_per_structure")
    if a.require_ost_pairs and want and want != a.poses:
        bad.append(f"__meta says OST examined {want} pose(s) per structure, not {a.poses}")
if a.margin and abs(float(meta.get("margin", -1)) - a.margin) > 1e-9:
    bad.append(f"margin is {meta.get('margin')} A, not the required {a.margin} A")
if a.contact and abs(float(meta.get("contact", -1)) - a.contact) > 1e-9:
    bad.append(f"contact is {meta.get('contact')} A, not the required {a.contact} A")

if bad:
    print(f"STOP: {a.mask} does not establish the required configuration:")
    for b in bad:
        print(f"  - {b}")
    print(f"  __meta: {json.dumps({k: meta[k] for k in sorted(meta) if k != 'ost_extra_structures'})}")
    sys.exit(1)

_extra = meta.get("ost_extra_structures") or []
print(f"  mask OK: {len(keys)} structure(s), margin {meta.get('margin')} A, contact "
      f"{meta.get('contact')} A, force_gt={meta.get('force_gt')}, "
      f"ost_pairs={meta.get('ost_pairs')}, built {meta.get('built_utc')} by "
      f"{meta.get('builder_git')}"
      + (f"; {len(_extra)} structure(s) needed an OST-resolved pair the static mapping does not "
         f"name" if meta.get("ost_pairs") else ""))
